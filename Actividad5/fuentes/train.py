# -*- coding: utf-8 -*-
"""
train.py — Entrenamiento, ajuste de hiperparámetros y registro en MLflow
========================================================================
Proyecto: VOREKO IA · Actividad 5 (Gestión de Proyectos de IA, Tecmilenio)

Problema: regresión del COSTO REAL DE CONSTRUCCIÓN POR m2 de edificios
residenciales usando sólo información disponible al INICIO del proyecto
(dataset público UCI "Residential Building"). Es el insumo del motor de
alertas de desviación de costo de VOREKO.

Modelos comparados (≥ 2, según la actividad):
  * Ridge (regresión lineal regularizada L2)
  * Random Forest
  * XGBoost
  + dos referencias sin ajuste: mediana (DummyRegressor) y la "regla
    presupuestal" (costo preliminar × razón mediana real/preliminar).

Protocolo:
  1. Hold-out 80/20 estratificado por quintiles del objetivo (semilla 42).
  2. GridSearchCV 5-fold sobre entrenamiento, criterio RMSE (refit) y
     registro de MAE y R².
  3. Comparación estadística con validación cruzada ANIDADA
     (externa RepeatedKFold 5×3, interna GridSearchCV 5-fold): prueba t
     pareada corregida (Nadeau y Bengio, 2003), Wilcoxon y Friedman.
  4. Evaluación final en el 20 % de prueba (no usado para elegir nada).
  5. Registro en MLflow: parámetros, métricas, artefactos, dataset, modelos
     con firma y registro del mejor en el Model Registry.

Uso:
    python fuentes/train.py                       # experimento completo
    python fuentes/train.py --rapido              # grids reducidos (pruebas)
    MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python fuentes/train.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.base import BaseEstimator, RegressorMixin, clone  # noqa: E402
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,  # noqa: E402
                             mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos_prep as dp  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
RAIZ = dp.RAIZ
DIR_RES = RAIZ / "resultados"
DIR_FIG = DIR_RES / "figuras"
SEMILLA = 42
EXPERIMENTO = "VOREKO-A5-costo-real-m2"
NOMBRE_REGISTRO = "voreko_costo_real_m2"
# Componentes propios declarados como confiables para la serialización segura (skops) de MLflow 3
TIPOS_CONFIABLES = ["datos_prep.SelectorRezagos", "datos_prep.columnas_categoricas", "datos_prep.columnas_numericas",
                    "sklearn.tree._tree.Tree",  # árboles del Random Forest entrenado aquí
                    "xgboost.sklearn.XGBRegressor", "xgboost.core.Booster"]  # XGBoost entrenado aquí
COLORES = {"Ridge": "#2a78d6", "RandomForest": "#eb6834", "XGBoost": "#1baf7a",
           "ReglaPresupuestal": "#7d7c75", "Mediana": "#b9b8b0"}
TINTA, SEC, REJ = "#1f1f1e", "#5f5e58", "#e4e3dc"


# ---------------------------------------------------------------------------
# Componentes del pipeline
# ---------------------------------------------------------------------------
def construir_pipeline(modelo) -> TransformedTargetRegressor:
    """
    Selección de rezagos → (one-hot de localidad y trimestre + estandarización
    de numéricas) → modelo. El objetivo se modela en log (asimetría 1.26 → −0.41)
    y se regresa a MIRR/m2 para calcular métricas. Todo el preprocesamiento se
    ajusta dentro de cada fold, sin fuga de información.
    """
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), dp.columnas_categoricas),
        ("num", StandardScaler(), dp.columnas_numericas),
    ])
    pipe = Pipeline([("rezagos", dp.SelectorRezagos()), ("prep", prep), ("modelo", modelo)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)


class ReglaPresupuestal(BaseEstimator, RegressorMixin):
    """Referencia de negocio: costo real ≈ costo preliminar × razón mediana observada."""

    def fit(self, X, y):
        self.razon_ = float(np.median(np.asarray(y) / X["costo_prelim_m2"].to_numpy()))
        return self

    def predict(self, X):
        return X["costo_prelim_m2"].to_numpy() * self.razon_


def definir_modelos(rapido: bool = False) -> dict:
    """Espacios de búsqueda con su justificación (ver README y reporte)."""
    rez = ["todos", "r1"]
    if rapido:
        return {
            "Ridge": (Ridge(), {"regressor__rezagos__rezagos": rez, "regressor__modelo__alpha": [1.0, 10.0]}),
            "RandomForest": (RandomForestRegressor(n_estimators=50, random_state=SEMILLA, n_jobs=-1),
                             {"regressor__rezagos__rezagos": ["r1"], "regressor__modelo__min_samples_leaf": [1, 3]}),
            "XGBoost": (XGBRegressor(n_estimators=100, random_state=SEMILLA, n_jobs=2, verbosity=0),
                        {"regressor__rezagos__rezagos": ["r1"], "regressor__modelo__max_depth": [2, 4]}),
        }
    return {
        # alpha en escala logarítmica: de casi sin regularización a regularización fuerte
        "Ridge": (Ridge(), {
            "regressor__rezagos__rezagos": rez,
            "regressor__modelo__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        }),
        # profundidad, tamaño mínimo de hoja y fracción de variables controlan el sobreajuste con 297 filas
        "RandomForest": (RandomForestRegressor(n_estimators=300, random_state=SEMILLA, n_jobs=-1), {
            "regressor__rezagos__rezagos": rez,
            "regressor__modelo__max_depth": [None, 10],
            "regressor__modelo__min_samples_leaf": [1, 3, 5],
            "regressor__modelo__max_features": [0.33, 1.0],
        }),
        # boosting poco profundo con tasa de aprendizaje baja y submuestreo de columnas
        "XGBoost": (XGBRegressor(n_estimators=400, subsample=0.8, random_state=SEMILLA, n_jobs=2, verbosity=0), {
            "regressor__rezagos__rezagos": rez,
            "regressor__modelo__learning_rate": [0.03, 0.1],
            "regressor__modelo__max_depth": [2, 4],
            "regressor__modelo__colsample_bytree": [0.5, 0.8],
            "regressor__modelo__min_child_weight": [1, 5],
        }),
    }


# ---------------------------------------------------------------------------
# Métricas y pruebas estadísticas
# ---------------------------------------------------------------------------
ESCORES = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}


def metricas(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    return {"mae": float(mean_absolute_error(y, p)), "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p)), "mape": float(mean_absolute_percentage_error(y, p))}


def t_pareada_corregida(a, b, frac_prueba: float) -> tuple[float, float]:
    """Prueba t pareada con corrección de Nadeau y Bengio (2003) para CV repetida."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + frac_prueba / (1 - frac_prueba)) * var)
    return float(t), float(2 * stats.t.sf(abs(t), df=k - 1))


def holm(pvalores: dict) -> dict:
    orden = sorted(pvalores.items(), key=lambda kv: kv[1])
    m, ajust, previo = len(orden), {}, 0.0
    for i, (k, p) in enumerate(orden):
        previo = max(previo, min(1.0, (m - i) * p))
        ajust[k] = previo
    return ajust


# ---------------------------------------------------------------------------
# Trazabilidad
# ---------------------------------------------------------------------------
def commit_git() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "sin-git"


def etiquetas_base(manifiesto: dict) -> dict:
    return {
        "proyecto": "VOREKO IA", "actividad": "5", "autor": "José Ignacio Castillo Camacho",
        "git_commit": commit_git(), "version_datos": manifiesto["version_datos"],
        "sha256_contenido_original": manifiesto["huellas_sha256"]["contenido_original_hoja_Data"],
        "sha256_csv_limpio": manifiesto["huellas_sha256"]["csv_limpio"],
        "python": platform.python_version(), "sklearn": sklearn.__version__, "mlflow": mlflow.__version__,
    }


def configurar_mlflow(uri: str | None) -> str:
    uri = uri or os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(EXPERIMENTO) is None:
        artefactos = None if uri.startswith("http") else (RAIZ / "mlruns").as_uri()
        mlflow.create_experiment(EXPERIMENTO, artifact_location=artefactos,
                                 tags={"proyecto": "VOREKO IA", "problema": "regresión costo real por m2"})
    mlflow.set_experiment(EXPERIMENTO)
    return uri


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def _estilo():
    plt.rcParams.update({"axes.edgecolor": REJ, "axes.labelcolor": SEC, "xtick.color": SEC, "ytick.color": SEC,
                         "axes.titleweight": "bold", "axes.titlesize": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9.5})


def fig_pred_vs_real(y, p, nombre, ruta):
    _estilo()
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4))
    lim = [0, max(np.max(y), np.max(p)) * 1.05]
    ejes[0].scatter(y, p, s=22, color=COLORES.get(nombre, "#2a78d6"), alpha=0.8, edgecolor="white", linewidth=0.6)
    ejes[0].plot(lim, lim, "--", color=SEC, linewidth=1)
    ejes[0].set(xlim=lim, ylim=lim, xlabel="Costo real (MIRR/m²)", ylabel="Costo predicho (MIRR/m²)",
                title=f"{nombre}: predicho vs. real (prueba)")
    res = np.asarray(y) - np.asarray(p)
    ejes[1].scatter(p, res, s=22, color=COLORES.get(nombre, "#2a78d6"), alpha=0.8, edgecolor="white", linewidth=0.6)
    ejes[1].axhline(0, color=SEC, linestyle="--", linewidth=1)
    ejes[1].set(xlabel="Costo predicho (MIRR/m²)", ylabel="Residuo (real − predicho)", title="Residuos")
    for ax in ejes:
        ax.grid(color=REJ, linewidth=0.7)
        ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()


def fig_importancias(modelo, nombre, ruta, top=15):
    pipe = modelo.regressor_
    nombres = pipe.named_steps["prep"].get_feature_names_out()
    est = pipe.named_steps["modelo"]
    valores = np.abs(est.coef_) if hasattr(est, "coef_") else est.feature_importances_
    s = pd.Series(valores, index=[n.split("__", 1)[1] for n in nombres]).sort_values().tail(top)
    _estilo()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(s.index, s.values, color=COLORES.get(nombre), height=0.6, edgecolor="white")
    ax.set_title(f"{nombre}: {'|coeficiente| estandarizado' if hasattr(est, 'coef_') else 'importancia'} (top {top})")
    ax.grid(axis="x", color=REJ, linewidth=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    return s.sort_values(ascending=False)


def fig_cv_anidada(tabla, ruta):
    _estilo()
    orden = tabla.groupby("modelo")["rmse"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    datos = [tabla.loc[tabla.modelo == m, "rmse"].to_numpy() for m in orden]
    bp = ax.boxplot(datos, orientation="horizontal", patch_artist=True, widths=0.55, medianprops={"color": TINTA})
    for caja, m in zip(bp["boxes"], orden):
        caja.set_facecolor(COLORES.get(m, "#ccc"))
        caja.set_alpha(0.85)
        caja.set_edgecolor("white")
    for i, d in enumerate(datos, start=1):
        ax.scatter(d, np.random.default_rng(0).normal(i, 0.05, len(d)), s=10, color=TINTA, alpha=0.5, zorder=3)
    ax.set_yticks(range(1, len(orden) + 1), orden)
    ax.set_xlabel("RMSE por fold externo (MIRR/m²) — menor es mejor")
    ax.set_title("Validación cruzada anidada (RepeatedKFold 5×3): distribución del RMSE")
    ax.grid(axis="x", color=REJ, linewidth=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------
def ejecutar_experimento(rapido: bool = False, tracking_uri: str | None = None, registrar: bool = True,
                         repeticiones_externas: int = 3) -> dict:
    t0 = time.time()
    DIR_FIG.mkdir(parents=True, exist_ok=True)
    manifiesto = dp.ejecutar()
    df = pd.read_csv(dp.CSV_LIMPIO)
    X, y = dp.seleccionar_variables(df)
    uri = configurar_mlflow(tracking_uri)
    tags = etiquetas_base(manifiesto)

    # 1. Hold-out estratificado por quintiles del objetivo
    estratos = pd.qcut(y, 5, labels=False)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEMILLA, stratify=estratos)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEMILLA)

    # ---- Run de datos ----
    with mlflow.start_run(run_name=f"01_datos_v{manifiesto['version_datos']}", tags={**tags, "etapa": "datos"}):
        mlflow.log_input(mlflow.data.from_pandas(df, source=str(dp.CSV_LIMPIO), name="residential_building_limpio",
                                                 targets=dp.OBJETIVO), context="limpio")
        mlflow.log_params({"filas": len(df), "predictoras": X.shape[1], "objetivo": dp.OBJETIVO,
                           "filas_entrenamiento": len(X_tr), "filas_prueba": len(X_te), "semilla": SEMILLA,
                           "particion": "hold-out 80/20 estratificado por quintiles"})
        r = manifiesto["reporte"]
        mlflow.log_metrics({"duplicados_exactos": r["4_duplicados"]["duplicados_exactos"],
                            "duplicados_clave_proyecto": r["4_duplicados"]["duplicados_por_clave_proyecto"],
                            "nulos_totales": r["3_nulos"]["nulos_totales"],
                            "violaciones_reglas": r["5_reglas_negocio"]["filas_eliminadas"],
                            "anomalias_razon_costo": r["7_anomalias"]["anomalias_razon_costo"],
                            "columnas_a_MIRR": r["6_estandarizacion"]["columnas_monetarias_a_MIRR"]})
        for archivo in [dp.REPORTE_CALIDAD, dp.DICCIONARIO, dp.MANIFIESTO]:
            mlflow.log_artifact(str(archivo), artifact_path="datos")

    modelos = definir_modelos(rapido)
    resumen, mejores, filas_cv = {}, {}, []

    # ---- 2. GridSearchCV por modelo (run padre + runs hijos por candidato) ----
    for nombre, (est, grid) in modelos.items():
        with mlflow.start_run(run_name=f"02_gridsearch_{nombre}", tags={**tags, "etapa": "ajuste", "modelo": nombre}) as padre:
            gs = GridSearchCV(construir_pipeline(est), grid, scoring=ESCORES, refit="rmse", cv=cv, n_jobs=1)
            inicio = time.time()
            gs.fit(X_tr, y_tr)
            seg = time.time() - inicio
            res = pd.DataFrame(gs.cv_results_)
            for i, fila in res.iterrows():
                with mlflow.start_run(run_name=f"{nombre}_cand_{i:02d}", nested=True,
                                      tags={"modelo": nombre, "etapa": "candidato"}):
                    mlflow.log_params({k.split("__")[-1]: v for k, v in fila["params"].items()})
                    mlflow.log_metrics({"cv_rmse": -fila["mean_test_rmse"], "cv_rmse_std": fila["std_test_rmse"],
                                        "cv_mae": -fila["mean_test_mae"], "cv_r2": fila["mean_test_r2"],
                                        "rank_rmse": int(fila["rank_test_rmse"])})
            mejor = gs.best_estimator_
            p_te = mejor.predict(X_te)
            m_te = metricas(y_te, p_te)
            i_best = gs.best_index_
            m_cv = {"cv_rmse": -res.loc[i_best, "mean_test_rmse"], "cv_rmse_std": res.loc[i_best, "std_test_rmse"],
                    "cv_mae": -res.loc[i_best, "mean_test_mae"], "cv_r2": res.loc[i_best, "mean_test_r2"]}
            mlflow.log_params({**{"estimador": type(est).__name__, "busqueda": "GridSearchCV",
                                  "cv": "KFold(5, shuffle, 42)", "criterio_refit": "RMSE",
                                  "transformacion_objetivo": "log", "candidatos": len(res)},
                               **{"mejor_" + k.split("__")[-1]: v for k, v in gs.best_params_.items()}})
            mlflow.log_metrics({**m_cv, **{f"test_{k}": v for k, v in m_te.items()}, "segundos_busqueda": seg})
            res_ruta = DIR_RES / f"cv_resultados_{nombre}.csv"
            res.drop(columns=["params"]).to_csv(res_ruta, index=False)
            mlflow.log_artifact(str(res_ruta), artifact_path="ajuste")
            fig1 = DIR_FIG / f"pred_vs_real_{nombre}.png"
            fig_pred_vs_real(y_te, p_te, nombre, fig1)
            fig2 = DIR_FIG / f"importancia_{nombre}.png"
            imp = fig_importancias(mejor, nombre, fig2)
            for f in (fig1, fig2):
                mlflow.log_artifact(str(f), artifact_path="figuras")
            firma = infer_signature(X_tr, mejor.predict(X_tr))
            info = mlflow.sklearn.log_model(mejor, name="modelo", signature=firma, input_example=X_tr.head(3),
                                            code_paths=[str(Path(dp.__file__))],
                                            skops_trusted_types=TIPOS_CONFIABLES)
            resumen[nombre] = {"mejores_params": {k.split("__")[-1]: v for k, v in gs.best_params_.items()},
                               **m_cv, **{f"test_{k}": v for k, v in m_te.items()}, "run_id": padre.info.run_id,
                               "model_uri": info.model_uri, "candidatos": len(res), "segundos": round(seg, 1),
                               "top_variables": imp.head(5).round(4).to_dict()}
            mejores[nombre] = (est, grid)

    # ---- 3. Referencias sin ajuste ----
    for nombre, est in {"ReglaPresupuestal": ReglaPresupuestal(), "Mediana": DummyRegressor(strategy="median")}.items():
        with mlflow.start_run(run_name=f"03_referencia_{nombre}", tags={**tags, "etapa": "referencia", "modelo": nombre}):
            sc = {k: [] for k in ["rmse", "mae", "r2"]}
            for tr, va in cv.split(X_tr):
                e = clone(est).fit(X_tr.iloc[tr], y_tr.iloc[tr])
                mm = metricas(y_tr.iloc[va], e.predict(X_tr.iloc[va]))
                for k in sc:
                    sc[k].append(mm[k])
            e = clone(est).fit(X_tr, y_tr)
            m_te = metricas(y_te, e.predict(X_te))
            m_cv = {"cv_rmse": float(np.mean(sc["rmse"])), "cv_rmse_std": float(np.std(sc["rmse"])),
                    "cv_mae": float(np.mean(sc["mae"])), "cv_r2": float(np.mean(sc["r2"]))}
            mlflow.log_params({"estimador": nombre, "ajuste": "ninguno"})
            mlflow.log_metrics({**m_cv, **{f"test_{k}": v for k, v in m_te.items()}})
            resumen[nombre] = {**m_cv, **{f"test_{k}": v for k, v in m_te.items()}}

    # ---- 4. Validación cruzada anidada y pruebas estadísticas ----
    externa = RepeatedKFold(n_splits=5, n_repeats=repeticiones_externas, random_state=SEMILLA)
    with mlflow.start_run(run_name="04_cv_anidada_comparacion", tags={**tags, "etapa": "comparacion"}):
        mlflow.log_params({"cv_externa": f"RepeatedKFold(5x{repeticiones_externas})", "cv_interna": "GridSearchCV KFold(5)",
                           "prueba_parametrica": "t pareada corregida (Nadeau y Bengio, 2003)",
                           "prueba_no_parametrica": "Wilcoxon de rangos con signo; Friedman", "ajuste_p": "Holm"})
        for f, (tr, va) in enumerate(externa.split(X_tr)):
            Xa, Xv, ya, yv = X_tr.iloc[tr], X_tr.iloc[va], y_tr.iloc[tr], y_tr.iloc[va]
            cv_int = KFold(n_splits=5, shuffle=True, random_state=SEMILLA + f)
            for nombre, (est, grid) in mejores.items():
                gs = GridSearchCV(construir_pipeline(est), grid, scoring="neg_root_mean_squared_error", cv=cv_int, n_jobs=1)
                gs.fit(Xa, ya)
                mm = metricas(yv, gs.predict(Xv))
                filas_cv.append({"fold": f, "modelo": nombre, **mm})
                mlflow.log_metric(f"rmse_fold_{nombre}", mm["rmse"], step=f)
            e = ReglaPresupuestal().fit(Xa, ya)
            filas_cv.append({"fold": f, "modelo": "ReglaPresupuestal", **metricas(yv, e.predict(Xv))})
        tabla = pd.DataFrame(filas_cv)
        tabla.to_csv(DIR_RES / "cv_anidada_por_fold.csv", index=False)
        agregado = tabla.groupby("modelo")[["rmse", "mae", "r2", "mape"]].agg(["mean", "std"])
        agregado.columns = [f"{a}_{b}" for a, b in agregado.columns]
        agregado = agregado.sort_values("rmse_mean")
        agregado.to_csv(DIR_RES / "cv_anidada_resumen.csv")
        for m, fila in agregado.iterrows():
            mlflow.log_metrics({f"anidada_rmse_media_{m}": fila["rmse_mean"], f"anidada_r2_media_{m}": fila["r2_mean"]})

        piv = tabla.pivot(index="fold", columns="modelo", values="rmse")
        ajustados = [m for m in mejores]
        pruebas, p_t, p_w = [], {}, {}
        for a, b in combinations(ajustados + ["ReglaPresupuestal"], 2):
            t, pt = t_pareada_corregida(piv[a], piv[b], frac_prueba=0.2)
            try:
                pw = float(stats.wilcoxon(piv[a], piv[b]).pvalue)
            except ValueError:
                pw = 1.0
            clave = f"{a}_vs_{b}"
            p_t[clave], p_w[clave] = pt, pw
            pruebas.append({"comparacion": clave, "dif_media_rmse": float((piv[a] - piv[b]).mean()),
                            "t_corregida": t, "p_t_corregida": pt, "p_wilcoxon": pw,
                            "gana_a_en_folds": int((piv[a] < piv[b]).sum()), "folds": len(piv)})
        ht, hw = holm(p_t), holm(p_w)
        for pr in pruebas:
            pr["p_t_holm"], pr["p_wilcoxon_holm"] = ht[pr["comparacion"]], hw[pr["comparacion"]]
            pr["significativa_0_05"] = bool(pr["p_t_holm"] < 0.05 and pr["p_wilcoxon_holm"] < 0.05)
        pruebas = pd.DataFrame(pruebas)
        pruebas.to_csv(DIR_RES / "pruebas_estadisticas.csv", index=False)
        fr = stats.friedmanchisquare(*[piv[m] for m in ajustados])
        mlflow.log_metrics({"friedman_estadistico": float(fr.statistic), "friedman_p": float(fr.pvalue)})
        for _, pr in pruebas.iterrows():
            mlflow.log_metrics({f"p_t_holm_{pr['comparacion']}": pr["p_t_holm"],
                                f"p_wilcoxon_holm_{pr['comparacion']}": pr["p_wilcoxon_holm"]})
        fig_cv_anidada(tabla, DIR_FIG / "cv_anidada_rmse.png")
        for ruta in ["cv_anidada_por_fold.csv", "cv_anidada_resumen.csv", "pruebas_estadisticas.csv"]:
            mlflow.log_artifact(str(DIR_RES / ruta), artifact_path="comparacion")
        mlflow.log_artifact(str(DIR_FIG / "cv_anidada_rmse.png"), artifact_path="figuras")

    # ---- 5. Selección y registro del modelo ----
    candidatos_ajustados = agregado.loc[[m for m in agregado.index if m in mejores]]
    ganador = candidatos_ajustados["rmse_mean"].idxmin()
    seleccion = {"modelo": ganador, "criterio": "menor RMSE medio en validación cruzada anidada",
                 "friedman_p": float(fr.pvalue)}
    if registrar:
        mv = mlflow.register_model(resumen[ganador]["model_uri"], NOMBRE_REGISTRO)
        cliente = mlflow.MlflowClient()
        cliente.set_registered_model_alias(NOMBRE_REGISTRO, "candidato", mv.version)
        cliente.update_model_version(NOMBRE_REGISTRO, mv.version,
                                     description=f"{ganador} · {seleccion['criterio']} · datos v{manifiesto['version_datos']}")
        cliente.set_model_version_tag(NOMBRE_REGISTRO, mv.version, "validacion", "pendiente datos reales VOREKO")
        seleccion["registro"] = {"nombre": NOMBRE_REGISTRO, "version": mv.version, "alias": "candidato"}

    # ---- 6. Exportación de resultados ----
    exp = mlflow.get_experiment_by_name(EXPERIMENTO)
    corridas = mlflow.search_runs([exp.experiment_id])
    corridas.to_csv(DIR_RES / "mlflow_runs.csv", index=False)
    salida = {"tracking_uri": uri, "experimento": EXPERIMENTO, "experiment_id": exp.experiment_id,
              "runs_registrados": len(corridas), "datos": {k: manifiesto[k] for k in ["version_datos", "filas_limpio", "predictoras"]},
              "particion": {"entrenamiento": len(X_tr), "prueba": len(X_te)}, "modelos": resumen,
              "seleccion": seleccion, "friedman": {"estadistico": float(fr.statistic), "p": float(fr.pvalue)},
              "segundos_totales": round(time.time() - t0, 1), "git_commit": tags["git_commit"], "rapido": rapido}
    (DIR_RES / "resumen_experimento.json").write_text(json.dumps(salida, ensure_ascii=False, indent=2, default=str),
                                                      encoding="utf-8")
    return salida


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Entrenamiento y registro en MLflow — VOREKO A5")
    ap.add_argument("--rapido", action="store_true", help="grids reducidos (sólo para pruebas)")
    ap.add_argument("--tracking-uri", default=None)
    ap.add_argument("--sin-registro", action="store_true", help="no registrar en el Model Registry")
    ap.add_argument("--repeticiones", type=int, default=3, help="repeticiones de la CV externa")
    a = ap.parse_args()
    s = ejecutar_experimento(rapido=a.rapido, tracking_uri=a.tracking_uri, registrar=not a.sin_registro,
                             repeticiones_externas=a.repeticiones)
    print(json.dumps({k: s[k] for k in ["experimento", "runs_registrados", "seleccion", "segundos_totales"]},
                     ensure_ascii=False, indent=2, default=str))
