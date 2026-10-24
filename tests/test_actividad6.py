import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fuentes"))
import voreko_a6 as v


@pytest.fixture(scope="module")
def datos():
    df, man = v.cargar_datos_a5()
    return df, man


def test_dataset_a5_intacto(datos):
    df, man = datos
    assert v.sha256(v.dp.CSV_LIMPIO) == v.SHA_A5_ESPERADO == man["huellas_sha256"]["csv_limpio"]
    assert df.shape == (372, 117)


def test_etiqueta_y_empates(datos):
    df, _ = datos
    y = v.construir_etiqueta(df)
    d = v.desviacion_real(df)
    assert int((d == 0.5).sum()) == 30
    assert int(y.sum()) == 101 and set(y.unique()) == {0, 1}
    assert (y[d == 0.5] == 0).all()
    assert v.construir_etiqueta(df, 0.2).sum() > v.construir_etiqueta(df, 0.5).sum()


def test_duracion_es_real_y_se_excluye(datos):
    df, _ = datos
    assert v.verificar_duracion_real(df)["fin_igual_inicio_mas_duracion"] == 372
    X = v.variables_inicio(df)
    for c in ["duracion_trimestres", "costo_real_m2", "precio_venta_real_m2", "anio_fin", "trimestre_fin"]:
        assert c not in X.columns


def test_particion_a5(datos):
    df, _ = datos
    X = v.variables_inicio(df)
    tr, te = v.particion_a5(X, df["costo_real_m2"])
    assert len(tr) == 297 and len(te) == 75 and not set(tr) & set(te)
    tr2, te2 = v.particion_a5(X, df["costo_real_m2"])
    assert list(te) == list(te2)


def test_metricas_formula_vs_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
    e = v.evaluar(y, s, 0.5)
    pred = (s >= 0.5).astype(int)
    assert np.isclose(e["accuracy"], accuracy_score(y, pred))
    assert np.isclose(e["precision"], precision_score(y, pred))
    assert np.isclose(e["recall"], recall_score(y, pred))
    assert np.isclose(e["specificity"], recall_score(y, pred, pos_label=0))
    assert np.isclose(e["f1"], f1_score(y, pred))
    assert np.isclose(e["roc_auc"], roc_auc_score(y, s)) and np.isclose(v.auc_rangos(y, s), roc_auc_score(y, s))
    assert np.isclose(e["average_precision"], average_precision_score(y, s))
    assert e["VP"] + e["FP"] + e["FN"] + e["VN"] == 200


def test_ejemplo_tema13():
    y = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0]
    p = [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0]
    assert v.conteos(np.array(y), np.array(p)) == (5, 1, 2, 3)


def test_regla_de_umbral():
    t = pd.DataFrame({"umbral": [0.2, 0.3, 0.4, 0.5], "recall": [0.95, 0.85, 0.80, 0.60],
                      "precision": [0.40, 0.50, 0.55, 0.70], "alertas": [50, 40, 35, 20]})
    assert v.elegir_umbral(t) == {"umbral": 0.4, "cumple_recall_minimo": True}
    t2 = t.assign(recall=[0.7, 0.6, 0.5, 0.4])
    assert v.elegir_umbral(t2)["cumple_recall_minimo"] is False


def test_reproduce_alerta_a5(datos):
    df, _ = datos
    Xa5, _ = v.dp.seleccionar_variables(df)
    tr, te = v.particion_a5(Xa5, df["costo_real_m2"])
    regla = v.ReglaRegresionA5().fit(Xa5.loc[tr], df.loc[tr, "costo_real_m2"])
    e = v.evaluar(v.construir_etiqueta(df).loc[te], regla.puntuar(Xa5.loc[te]), 1.5)
    assert round(e["recall"], 3) == 0.692 and round(e["precision"], 3) == 0.783 and round(e["roc_auc"], 3) == 0.859


def test_estadistica():
    a = np.array([0.60, 0.62, 0.58, 0.65, 0.61, 0.63])
    b = a + np.array([0.05, 0.04, 0.06, 0.05, 0.045, 0.055])
    t, p = v.t_corregida(b, a, 0.2)
    assert p < 0.05
    li, ls = v.ic_t_corregida(b - a, 0.2)
    assert li < 0.05 < ls
    m = v.mcnemar_exacta([1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0])
    assert m["solo_A_acierta"] == 0 and m["solo_B_acierta"] == 2
    assert v.n_por_grupo_proporciones(0.5, 0.6) > v.n_por_grupo_proporciones(0.5, 0.8)
