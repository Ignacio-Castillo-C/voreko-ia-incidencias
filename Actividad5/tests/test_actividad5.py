# -*- coding: utf-8 -*-
"""Pruebas de limpieza, versionado y componentes de entrenamiento (sin red y sin MLflow)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fuentes"))
import datos_prep as dp  # noqa: E402
import train  # noqa: E402


@pytest.fixture(scope="module")
def crudo():
    return dp.leer_original()


@pytest.fixture(scope="module")
def limpio(crudo):
    return dp.limpiar(crudo)


def test_original_intacto(crudo):
    # 2 filas de encabezado + 372 proyectos, 109 columnas
    assert crudo.shape == (374, 109)
    assert dp.sha256_dataframe(crudo) == "7e736374a647e1f64f930dab167dcf93bf8018c9d17b8a5891e5ce257b4671cd"


def test_encabezados_unicos(limpio):
    df, rep = limpio
    assert df.columns.is_unique
    assert rep["1_encabezados"]["codigos_repetidos_en_original"] == 76
    assert {"permisos_emitidos_r1", "permisos_emitidos_r5", "costo_real_m2"} <= set(df.columns)


def test_tipos_y_nulos(limpio):
    df, rep = limpio
    assert rep["2_tipos"]["valores_no_convertibles"] == 0
    num = df.drop(columns=["id_proyecto"]).select_dtypes(include=["number", "bool"])
    assert num.shape[1] == df.shape[1] - 1
    assert df.drop(columns=["id_proyecto"]).isna().sum().sum() == 0


def test_estandarizacion_unidades(crudo, limpio):
    df, _ = limpio
    # V-10 original en 10,000 IRR → MIRR (factor 0.01)
    original = pd.to_numeric(crudo.iloc[2:, -1]).reset_index(drop=True)
    assert np.allclose(df["costo_real_m2"], original * 0.01)
    assert df["anio_inicio"].between(1372, 1388).all()
    assert (df["anio_inicio_greg_aprox"] - df["anio_inicio"] == 621).all()


def test_reglas_negocio(limpio):
    df, rep = limpio
    assert all(v == 0 for v in rep["5_reglas_negocio"]["violaciones_por_regla"].values())
    assert (df["area_construida_m2"] >= df["area_terreno_m2"]).all()


def test_deduplicacion_detecta_duplicados(crudo):
    """La función debe eliminar duplicados exactos y por clave de proyecto inyectados."""
    df, _ = dp.aplanar_encabezados(crudo)
    df, _ = dp.convertir_tipos(df)
    sucio = pd.concat([df, df.iloc[[0, 1]], df.iloc[[2]].assign(precio_venta_real_m2=1)], ignore_index=True)
    dedup, rep = dp.deduplicar(sucio)
    assert rep["duplicados_exactos"] == 2
    assert rep["duplicados_por_clave_proyecto"] == 1
    assert len(dedup) == len(df)


def test_reglas_negocio_detectan_violaciones(crudo):
    df, _ = dp.aplanar_encabezados(crudo)
    df, _ = dp.convertir_tipos(df)
    malo = df.copy()
    malo.loc[0, "trimestre_inicio"] = 7
    malo.loc[1, "duracion_trimestres"] = 99
    _, rep = dp.validar_reglas_negocio(malo)
    assert rep["filas_eliminadas"] == 2


def test_anomalias_marcadas_no_eliminadas(limpio):
    df, rep = limpio
    assert len(df) == 372
    assert rep["7_anomalias"]["anomalias_razon_costo"] == int(df["anomalia_razon_costo"].sum()) > 0


def test_sin_fuga_de_informacion(limpio):
    df, _ = limpio
    X, y = dp.seleccionar_variables(df)
    prohibidas = {"precio_venta_real_m2", "anio_fin", "trimestre_fin", "indice_tiempo_fin", "costo_real_m2",
                  "razon_real_vs_preliminar", "costo_prelim_total"}
    assert not prohibidas & set(X.columns)
    assert X.shape == (372, 105) and y.name == "costo_real_m2"


def test_selector_rezagos(limpio):
    X, _ = dp.seleccionar_variables(limpio[0])
    assert dp.SelectorRezagos("todos").fit(X).transform(X).shape[1] == 105
    assert dp.SelectorRezagos("r1").fit(X).transform(X).shape[1] == 29


def test_pipeline_y_regla_presupuestal(limpio):
    X, y = dp.seleccionar_variables(limpio[0])
    from sklearn.linear_model import Ridge
    modelo = train.construir_pipeline(Ridge(alpha=10)).fit(X, y)
    pred = modelo.predict(X)
    assert pred.shape == y.shape and (pred > 0).all()
    regla = train.ReglaPresupuestal().fit(X, y)
    assert regla.razon_ == pytest.approx(float(np.median(y / X["costo_prelim_m2"])))


def test_metricas_y_pruebas():
    m = train.metricas([1, 2, 3], [1, 2, 4])
    assert m["mae"] == pytest.approx(1 / 3) and m["rmse"] == pytest.approx(np.sqrt(1 / 3))
    t, p = train.t_pareada_corregida([1.0, 1.1, 0.9, 1.0], [1.0, 1.1, 0.9, 1.0], 0.2)
    assert (t, p) == (0.0, 1.0)
    _, p2 = train.t_pareada_corregida(np.full(15, 2.0) + np.linspace(0, .1, 15), np.full(15, 1.0), 0.2)
    assert p2 < 0.05
    h = train.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h["a"] == pytest.approx(0.03) and h["c"] == pytest.approx(0.06) and h["b"] == pytest.approx(0.06)
