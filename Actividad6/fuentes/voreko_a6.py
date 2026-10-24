from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

RAIZ_A6 = Path(__file__).resolve().parents[1]
RAIZ_A5 = RAIZ_A6.parent / "Actividad5"
if not (RAIZ_A5 / "fuentes" / "datos_prep.py").exists():
    raise FileNotFoundError(f"No se encontró la Actividad 5 junto a la Actividad 6: {RAIZ_A5}")
sys.path.insert(0, str(RAIZ_A5 / "fuentes"))
import datos_prep as dp
import train as a5

SEMILLA = 42
UMBRAL_SOBRECOSTO = 0.50
RECALL_MINIMO = 0.80
SHA_A5_ESPERADO = "47e7de7fe5d195f1d1852bce258d43333044206db340d88128b069ed3fb5424b"
DIR_CLASIF = RAIZ_A6 / "datos" / "datos_clasif"
CSV_CLASIF = DIR_CLASIF / "residential_building_clasificacion.csv"
MANIFIESTO_CLASIF = DIR_CLASIF / "manifiesto_clasificacion.json"
VERSION_DATOS_CLASIF = "1.0.0-clasif"
EXPERIMENTO = "VOREKO-A6-clasificacion-sobrecostos"
NOMBRE_REGISTRO = "voreko_riesgo_sobrecosto"

EXCLUIDAS_A6 = {
    "duracion_trimestres": ("Duración real, no planeada: en las 372 obras año/trimestre de término = inicio + duración, "
                            "y la fecha de término es posterior al inicio (fuga de información).")
}
PARAMS_XGB_A5 = {"colsample_bytree": 0.8, "learning_rate": 0.1, "max_depth": 2, "min_child_weight": 5, "rezagos": "r1"}

MODELOS = ["Dummy (baseline)", "Regla A5 (XGBoost regresión)", "Regresión logística", "Random Forest", "XGBoost clasificador"]
CLASIFICADORES = ["Regresión logística", "Random Forest", "XGBoost clasificador"]


def sha256(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def cargar_datos_a5() -> tuple[pd.DataFrame, dict]:
    manifiesto = json.loads(dp.MANIFIESTO.read_text(encoding="utf-8"))
    huella = sha256(dp.CSV_LIMPIO)
    if huella != manifiesto["huellas_sha256"]["csv_limpio"] or huella != SHA_A5_ESPERADO:
        raise ValueError("La huella SHA-256 del dataset de la Actividad 5 no coincide: el archivo cambió.")
    return pd.read_csv(dp.CSV_LIMPIO), manifiesto


def desviacion_real(df: pd.DataFrame) -> pd.Series:
    return ((df["costo_real_m2"] - df["costo_prelim_m2"]) / df["costo_prelim_m2"]).round(6)


def construir_etiqueta(df: pd.DataFrame, umbral: float = UMBRAL_SOBRECOSTO) -> pd.Series:
    return (desviacion_real(df) > umbral).astype(int).rename("riesgo_sobrecosto")


def variables_inicio(df: pd.DataFrame) -> pd.DataFrame:
    X, _ = dp.seleccionar_variables(df)
    return X.drop(columns=list(EXCLUIDAS_A6))


def verificar_duracion_real(df: pd.DataFrame) -> dict:
    fin = df["anio_fin"] * 4 + df["trimestre_fin"]
    ini = df["anio_inicio"] * 4 + df["trimestre_inicio"]
    coincide = int(((fin - ini) == df["duracion_trimestres"]).sum())
    return {"obras": len(df), "fin_igual_inicio_mas_duracion": coincide,
            "corr_duracion_vs_desviacion": float(np.corrcoef(df["duracion_trimestres"], desviacion_real(df))[0, 1])}


def guardar_version_clasificacion(df: pd.DataFrame, manifiesto_a5: dict) -> dict:
    DIR_CLASIF.mkdir(parents=True, exist_ok=True)
    salida = df.copy()
    salida["desviacion_real"] = desviacion_real(df)
    for u in (0.20, 0.30, 0.40, 0.50):
        salida[f"riesgo_sobrecosto_{int(u * 100)}"] = construir_etiqueta(df, u)
    salida["riesgo_sobrecosto"] = construir_etiqueta(df)
    salida.to_csv(CSV_CLASIF, index=False, encoding="utf-8")
    y = salida["riesgo_sobrecosto"]
    man = {
        "version_datos": VERSION_DATOS_CLASIF, "deriva_de": {"version": manifiesto_a5["version_datos"],
                                                             "sha256_csv_limpio": manifiesto_a5["huellas_sha256"]["csv_limpio"]},
        "sha256_csv_clasificacion": sha256(CSV_CLASIF), "filas": len(salida),
        "etiqueta": "riesgo_sobrecosto = 1 si (costo_real_m2 − costo_prelim_m2)/costo_prelim_m2 > 0.50",
        "positivos": int(y.sum()), "prevalencia": round(float(y.mean()), 4),
        "obras_exactamente_en_50pct": int((salida["desviacion_real"] == 0.5).sum()),
        "excluidas_a6": EXCLUIDAS_A6, "columnas_agregadas": ["desviacion_real", "riesgo_sobrecosto", "riesgo_sobrecosto_20",
                                                              "riesgo_sobrecosto_30", "riesgo_sobrecosto_40", "riesgo_sobrecosto_50"],
    }
    MANIFIESTO_CLASIF.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    return man


def particion_a5(X: pd.DataFrame, costo_real: pd.Series):
    estratos = pd.qcut(costo_real, 5, labels=False)
    return train_test_split(X.index, test_size=0.2, random_state=SEMILLA, stratify=estratos)


def _pipeline(modelo) -> Pipeline:
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), dp.columnas_categoricas),
        ("num", StandardScaler(), dp.columnas_numericas),
    ])
    return Pipeline([("rezagos", dp.SelectorRezagos()), ("prep", prep), ("modelo", modelo)])


def definir_clasificadores() -> dict:
    return {
        "Dummy (baseline)": (_pipeline(DummyClassifier(strategy="prior")), {}),
        "Regresión logística": (_pipeline(LogisticRegression(class_weight="balanced", max_iter=5000)), {
            "rezagos__rezagos": ["todos", "r1"], "modelo__C": [0.01, 0.1, 1.0, 10.0]}),
        "Random Forest": (_pipeline(RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                           random_state=SEMILLA, n_jobs=-1)), {
            "rezagos__rezagos": ["r1"], "modelo__max_depth": [None, 6], "modelo__min_samples_leaf": [1, 3, 5],
            "modelo__max_features": ["sqrt", 0.5]}),
        "XGBoost clasificador": (_pipeline(XGBClassifier(n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                                                         random_state=SEMILLA, n_jobs=2, verbosity=0,
                                                         eval_metric="logloss")), {
            "rezagos__rezagos": ["r1"], "modelo__max_depth": [2, 3], "modelo__learning_rate": [0.03, 0.1],
            "modelo__min_child_weight": [1, 5]}),
    }


class ReglaRegresionA5:

    def __init__(self, params: dict | None = None):
        self.params = dict(params or PARAMS_XGB_A5)

    def fit(self, X, costo_real):
        p = {k: v for k, v in self.params.items() if k != "rezagos"}
        est = XGBRegressor(n_estimators=400, subsample=0.8, random_state=SEMILLA, n_jobs=2, verbosity=0, **p)
        self.modelo_ = a5.construir_pipeline(est).set_params(regressor__rezagos__rezagos=self.params["rezagos"])
        self.modelo_.fit(X, costo_real)
        return self

    def puntuar(self, X) -> np.ndarray:
        return self.modelo_.predict(X) / X["costo_prelim_m2"].to_numpy()


def ajustar(nombre: str, X, y, costo_real, semilla_cv: int = SEMILLA):
    if nombre == "Regla A5 (XGBoost regresión)":
        return ReglaRegresionA5().fit(X, costo_real)
    est, grid = definir_clasificadores()[nombre]
    if not grid:
        return clone(est).fit(X, y)
    gs = GridSearchCV(clone(est), grid, scoring="average_precision", refit=True,
                      cv=StratifiedKFold(5, shuffle=True, random_state=semilla_cv), n_jobs=1)
    return gs.fit(X, y)


def fijo(nombre: str, ajustado):
    if nombre == "Regla A5 (XGBoost regresión)":
        return ReglaRegresionA5(ajustado.params)
    return clone(ajustado.best_estimator_) if hasattr(ajustado, "best_estimator_") else clone(ajustado)


def puntuar(modelo, X) -> np.ndarray:
    if isinstance(modelo, ReglaRegresionA5):
        return modelo.puntuar(X)
    return modelo.predict_proba(X)[:, 1]


def ajustar_fijo(nombre, modelo_fijo, X, y, costo_real):
    return modelo_fijo.fit(X, costo_real) if nombre == "Regla A5 (XGBoost regresión)" else modelo_fijo.fit(X, y)


def oof(nombre: str, modelo_fijo, X, y, costo_real, cv) -> np.ndarray:
    s = np.full(len(X), np.nan)
    for tr, va in cv.split(X, y):
        m = ajustar_fijo(nombre, clone(modelo_fijo) if not isinstance(modelo_fijo, ReglaRegresionA5)
                         else ReglaRegresionA5(modelo_fijo.params), X.iloc[tr], y.iloc[tr], costo_real.iloc[tr])
        s[va] = puntuar(m, X.iloc[va])
    assert not np.isnan(s).any()
    return s


def rejilla(nombre: str) -> np.ndarray:
    if nombre == "Regla A5 (XGBoost regresión)":
        return np.round(np.arange(1.00, 2.501, 0.01), 2)
    return np.round(np.arange(0.05, 0.951, 0.01), 2)


def conteos(y, pred) -> tuple[int, int, int, int]:
    m = confusion_matrix(y, pred, labels=[0, 1])
    assert m.shape == (2, 2) and m.sum() == len(y)
    vn, fp, fn, vp = m.ravel()
    return int(vn), int(fp), int(fn), int(vp)


def metricas_formula(vn, fp, fn, vp) -> dict:
    div = lambda a, b: a / b if b else 0.0
    p, r, e = div(vp, vp + fp), div(vp, vp + fn), div(vn, vn + fp)
    return {"accuracy": div(vp + vn, vp + vn + fp + fn), "precision": p, "recall": r, "specificity": e,
            "f1": div(2 * p * r, p + r), "balanced_accuracy": (r + e) / 2}


def auc_rangos(y, s) -> float:
    y = np.asarray(y)
    r = stats.rankdata(s)
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluar(y, s, umbral) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(s) >= umbral).astype(int)
    vn, fp, fn, vp = conteos(y, pred)
    m = metricas_formula(vn, fp, fn, vp)
    skl = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0), "specificity": recall_score(y, pred, pos_label=0, zero_division=0),
           "f1": f1_score(y, pred, zero_division=0), "balanced_accuracy": balanced_accuracy_score(y, pred)}
    for k in m:
        assert np.isclose(m[k], skl[k]), f"{k}: fórmula {m[k]} ≠ sklearn {skl[k]}"
    auc = roc_auc_score(y, s)
    assert np.isclose(auc, auc_rangos(y, s))
    return {"umbral": float(umbral), "VP": vp, "FP": fp, "FN": fn, "VN": vn, **m, "roc_auc": float(auc),
            "average_precision": float(average_precision_score(y, s)), "alertas": vp + fp,
            "pct_alertas": (vp + fp) / len(y) * 100}


def tabla_umbrales(y, s, umbrales) -> pd.DataFrame:
    y, s = np.asarray(y), np.asarray(s)
    filas = []
    for u in umbrales:
        pred = (s >= u).astype(int)
        vn, fp, fn, vp = conteos(y, pred)
        m = metricas_formula(vn, fp, fn, vp)
        filas.append({"umbral": float(u), "recall": m["recall"], "precision": m["precision"], "f1": m["f1"],
                      "FP": fp, "FN": fn, "VP": vp, "VN": vn, "alertas": vp + fp, "pct_alertas": (vp + fp) / len(y) * 100})
    return pd.DataFrame(filas)


def elegir_umbral(tabla: pd.DataFrame, recall_min: float = RECALL_MINIMO) -> dict:
    ok = tabla[tabla.recall >= recall_min - 1e-12]
    if len(ok):
        fila = ok.sort_values(["precision", "recall", "alertas"], ascending=[False, False, True]).iloc[0]
        return {"umbral": float(fila.umbral), "cumple_recall_minimo": True}
    fila = tabla.sort_values(["recall", "precision"], ascending=[False, False]).iloc[0]
    return {"umbral": float(fila.umbral), "cumple_recall_minimo": False}


t_corregida = a5.t_pareada_corregida
holm = a5.holm


def ic_t_corregida(d, frac_prueba: float, nivel: float = 0.95) -> tuple[float, float]:
    d = np.asarray(d)
    k = len(d)
    ee = np.sqrt((1 / k + frac_prueba / (1 - frac_prueba)) * np.var(d, ddof=1))
    c = stats.t.ppf(0.5 + nivel / 2, df=k - 1)
    return float(d.mean() - c * ee), float(d.mean() + c * ee)


def mcnemar_exacta(y, pred_a, pred_b) -> dict:
    y, a, b = map(np.asarray, (y, pred_a, pred_b))
    ok_a, ok_b = a == y, b == y
    n01, n10 = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"solo_A_acierta": n01, "solo_B_acierta": n10, "p_exacta": float(p)}


def bootstrap_diferencia(y, s_a, s_b, metrica, n=2000, semilla=SEMILLA) -> dict:
    y, s_a, s_b = map(np.asarray, (y, s_a, s_b))
    rng = np.random.default_rng(semilla)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    difs = []
    for _ in range(n):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        difs.append(metrica(y[i], s_b[i]) - metrica(y[i], s_a[i]))
    difs = np.asarray(difs)
    return {"diferencia": float(metrica(y, s_b) - metrica(y, s_a)), "ic95": [float(np.percentile(difs, 2.5)),
            float(np.percentile(difs, 97.5))], "p_bootstrap_bilateral": float(min(1, 2 * min((difs <= 0).mean(), (difs >= 0).mean())))}


def n_por_grupo_proporciones(p1: float, p2: float, alfa=0.05, potencia=0.80, unilateral=True) -> int:
    za = stats.norm.ppf(1 - alfa) if unilateral else stats.norm.ppf(1 - alfa / 2)
    zb = stats.norm.ppf(potencia)
    pbar = (p1 + p2) / 2
    return int(np.ceil((za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p2 - p1) ** 2))


def factores_locales(modelo, x: pd.DataFrame, X_ref: pd.DataFrame, top: int = 3) -> list[dict]:
    base = float(puntuar(modelo, x)[0])
    filas = []
    for col in x.columns:
        alt = x.copy()
        alt[col] = X_ref[col].mode().iloc[0] if col in dp.CATEGORICAS else X_ref[col].median()
        filas.append({"variable": col, "valor": float(x[col].iloc[0]), "efecto": base - float(puntuar(modelo, alt)[0])})
    return sorted(filas, key=lambda f: -f["efecto"])[:top]
