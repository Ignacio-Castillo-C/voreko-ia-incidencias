# VOREKO IA · Fase II · Entrenamiento y empaquetado del modelo de riesgo de sobrecosto
# José Ignacio Castillo Camacho · AL07161745

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ_REPO / "Actividad6" / "fuentes"))
import voreko_a6 as v

MODELO = "Random Forest"
NOMBRE = "voreko_riesgo_sobrecosto"
VERSION = "1.0.0"
REFERENCIA_A6 = {"umbral": 0.39, "recall": 0.538, "precision": 0.560, "roc_auc": 0.803}
SALIDA_DEFECTO = Path(__file__).resolve().parent / "artefactos"


# --- Utilidades ---

def sha256(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def diccionario() -> pd.DataFrame:
    return pd.read_csv(RAIZ_REPO / "Actividad5" / "datos" / "datos_limp" / "diccionario_datos.csv").set_index("columna")


def unidad_legible(u) -> str:
    if not isinstance(u, str) or u in ("N/A", "categórica"):
        return ""
    return {"MIRR": "millones de riales", "MIRR/m2": "millones de riales por m²", "m2": "m²", "%": "%"}.get(u, u)


# --- Entrenamiento con la configuración de la Actividad 6 ---

def entrenar():
    df, _ = v.cargar_datos_a5()
    X = v.variables_inicio(df)
    y = v.construir_etiqueta(df)
    costo = df["costo_real_m2"]
    idx_tr, idx_te = v.particion_a5(X, costo)
    X_tr, X_te, y_tr, y_te = X.loc[idx_tr], X.loc[idx_te], y.loc[idx_tr], y.loc[idx_te]

    busqueda = v.ajustar(MODELO, X_tr, y_tr, costo.loc[idx_tr])
    original = busqueda.best_estimator_
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=v.SEMILLA)
    s_oof = v.oof(MODELO, v.fijo(MODELO, busqueda), X_tr, y_tr, costo.loc[idx_tr], cv5)
    tabla = v.tabla_umbrales(y_tr, s_oof, v.rejilla(MODELO))
    umbral_rojo = v.elegir_umbral(tabla)["umbral"]
    umbral_amarillo = float(tabla[tabla.recall >= 0.95].umbral.max())
    return df, X, y, idx_tr, idx_te, X_tr, X_te, y_tr, y_te, busqueda, original, umbral_rojo, umbral_amarillo


# --- Empaquetado portátil (sin dependencias de las actividades anteriores) ---

def empaquetar(original: Pipeline, X_tr: pd.DataFrame, y_tr: pd.Series) -> tuple[Pipeline, list, list, list]:
    columnas = list(original.named_steps["rezagos"].columnas_)
    categoricas = [c for c in columnas if c in v.dp.CATEGORICAS]
    numericas = [c for c in columnas if c not in v.dp.CATEGORICAS]
    rf = original.named_steps["modelo"]
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas),
        ("num", StandardScaler(), numericas),
    ])
    portatil = Pipeline([("prep", prep), ("modelo", RandomForestClassifier(**rf.get_params()))])
    portatil.fit(X_tr[columnas], y_tr)
    return portatil, columnas, categoricas, numericas


def verificar(condicion: bool, texto: str, registro: list):
    if not condicion:
        raise AssertionError(texto)
    registro.append(texto)
    print(f"OK  {texto}")


# --- Programa principal ---

def main(salida: Path):
    salida.mkdir(parents=True, exist_ok=True)
    checks = []
    (df, X, y, idx_tr, idx_te, X_tr, X_te, y_tr, y_te, busqueda, original, u_rojo, u_amarillo) = entrenar()
    portatil, columnas, categoricas, numericas = empaquetar(original, X_tr, y_tr)

    p_original = original.predict_proba(X_te)[:, 1]
    p_portatil = portatil.predict_proba(X_te[columnas])[:, 1]
    verificar(float(np.max(np.abs(p_original - p_portatil))) < 1e-12,
              "El modelo empaquetado da las mismas probabilidades que el pipeline de la Actividad 6 en las 75 obras de prueba (diferencia < 1e-12)", checks)
    verificar(abs(u_rojo - REFERENCIA_A6["umbral"]) < 1e-9, f"Umbral de decisión reproducido: {u_rojo:.2f}", checks)

    ev = v.evaluar(y_te, p_portatil, u_rojo)
    for k in ("recall", "precision", "roc_auc"):
        verificar(abs(ev[k] - REFERENCIA_A6[k]) < 0.0005, f"{k} en prueba reproducido: {ev[k]:.3f}", checks)

    nivel = np.where(p_portatil >= u_rojo, "rojo", np.where(p_portatil >= u_amarillo, "amarillo", "verde"))
    semaforo = {k: {"obras": int((nivel == k).sum()), "con_sobrecosto_real": int(y_te.values[nivel == k].sum())}
                for k in ("verde", "amarillo", "rojo")}

    ruta_modelo = salida / "modelo_riesgo.joblib"
    joblib.dump(portatil, ruta_modelo, compress=3)

    dic = diccionario()
    variables = []
    for c in columnas:
        serie = X_tr[c]
        variables.append({
            "nombre": c, "tipo": "categórica" if c in categoricas else "numérica",
            "descripcion": str(dic.loc[c, "descripcion"]) if c in dic.index else c,
            "unidad": unidad_legible(dic.loc[c, "unidad_limpia"]) if c in dic.index else "",
            "minimo": float(serie.min()), "maximo": float(serie.max()), "mediana": float(serie.median()),
            "tipico": int(serie.mode().iloc[0]) if c in categoricas else float(serie.median()),
            "valores_permitidos": sorted(int(x) for x in X[c].unique()) if c in categoricas else None,
        })

    importancias = pd.Series(portatil.named_steps["modelo"].feature_importances_,
                             index=portatil.named_steps["prep"].get_feature_names_out())
    metadatos = {
        "nombre": NOMBRE, "version": VERSION, "algoritmo": "RandomForestClassifier (scikit-learn)",
        "origen": "Actividad 6 · modelo seleccionado (candidato, apoyo a la decisión)",
        "objetivo": "Probabilidad de que el costo real por m² supere en más de 50 % al costo preliminar",
        "dataset": "Residential Building Data Set (UCI id 437, CC BY 4.0) · 372 obras · montos en riales iraníes",
        "fecha_entrenamiento": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "semilla": v.SEMILLA, "hiperparametros": {k.split("__")[-1]: str(x) for k, x in busqueda.best_params_.items()},
        "umbrales": {"rojo": u_rojo, "amarillo": u_amarillo},
        "regla_umbral": "Máxima precisión con recall ≥ 0.80 en predicciones fuera de pliegue (5 pliegues)",
        "metricas_prueba": {k: round(float(ev[k]), 4) for k in
                            ("recall", "precision", "f1", "specificity", "accuracy", "balanced_accuracy", "roc_auc", "average_precision")},
        "matriz_confusion_prueba": {k: int(ev[k]) for k in ("VP", "FP", "FN", "VN")},
        "semaforo_prueba": semaforo,
        "obras": {"entrenamiento": int(len(idx_tr)), "prueba": int(len(idx_te)), "prevalencia_prueba": round(float(y_te.mean()), 4)},
        "variables": variables, "variables_categoricas": categoricas, "variables_numericas": numericas,
        "importancia_global_top10": importancias.sort_values(ascending=False).head(10).round(4).to_dict(),
        "versiones": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                      "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
        "huellas_sha256": {"datos_clasificacion_a6": sha256(v.CSV_CLASIF), "modelo_joblib": sha256(ruta_modelo)},
        "verificaciones": checks,
    }
    (salida / "metadatos_modelo.json").write_text(json.dumps(metadatos, ensure_ascii=False, indent=1), encoding="utf-8")

    desv = v.desviacion_real(df)
    ejemplos = []
    for i, p in zip(idx_te, p_portatil):
        ejemplos.append({"id_proyecto": str(df.loc[i, "id_proyecto"]),
                         "variables": {c: (int(X.loc[i, c]) if c in categoricas else float(X.loc[i, c])) for c in columnas},
                         "sobrecosto_real_pct": round(float(desv.loc[i]) * 100, 1), "riesgo_real": int(y.loc[i]),
                         "probabilidad_referencia": round(float(p), 6)})
    ejemplos.sort(key=lambda e: e["id_proyecto"])
    (salida / "obras_ejemplo.json").write_text(json.dumps(ejemplos, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps({"umbrales": metadatos["umbrales"], "metricas_prueba": metadatos["metricas_prueba"],
                      "semaforo_prueba": semaforo, "variables": len(columnas)}, ensure_ascii=False, indent=1))
    print(f"Artefactos guardados en {salida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--salida", type=Path, default=SALIDA_DEFECTO)
    main(parser.parse_args().salida)
