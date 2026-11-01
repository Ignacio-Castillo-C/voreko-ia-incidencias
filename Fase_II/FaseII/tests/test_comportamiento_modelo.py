# VOREKO IA · Fase II · Comportamiento y validación del modelo desplegado

import random
import time

from sklearn.metrics import average_precision_score, roc_auc_score


def lote(cliente, obras):
    cuerpo = {"obras": [{"id_proyecto": o["id_proyecto"], "variables": o["variables"], "incluir_factores": False} for o in obras]}
    return [x["resultado"] for x in cliente.post("/api/predecir/lote", json=cuerpo).json()["resultados"]]


# --- Validación contra la Actividad 6 ---

def test_metricas_de_la_actividad_6(cliente, obras, metadatos):
    res = lote(cliente, obras)
    y = [o["riesgo_real"] for o in obras]
    p = [r["probabilidad"] for r in res]
    alerta = [r["alerta"] for r in res]
    vp = sum(a and t for a, t in zip(alerta, y))
    fp = sum(a and not t for a, t in zip(alerta, y))
    fn = sum((not a) and t for a, t in zip(alerta, y))
    assert (vp, fp, fn) == (14, 11, 12)
    assert round(vp / (vp + fn), 3) == 0.538 and round(vp / (vp + fp), 3) == 0.560
    assert round(roc_auc_score(y, p), 3) == 0.803
    assert abs(average_precision_score(y, p) - metadatos["metricas_prueba"]["average_precision"]) < 0.002


def test_verde_sin_sobrecostos_reales(cliente, obras):
    res = lote(cliente, obras)
    assert sum(o["riesgo_real"] for o, r in zip(obras, res) if r["nivel"] == "verde") == 0


def test_nivel_coherente_con_umbrales(cliente, obras):
    for r in lote(cliente, obras):
        p, u = r["probabilidad"], r["umbrales"]
        esperado = "rojo" if p >= u["rojo"] else "amarillo" if p >= u["amarillo"] else "verde"
        assert r["nivel"] == esperado and r["alerta"] == (p >= u["rojo"])


# --- Estabilidad ---

def test_reproducible(cliente, obra):
    a = cliente.post("/api/predecir", json=obra).json()
    b = cliente.post("/api/predecir", json=obra).json()
    assert a["probabilidad"] == b["probabilidad"] and a["factores"] == b["factores"]


def test_orden_de_variables_no_importa(cliente, obra):
    a = cliente.post("/api/predecir", json=obra).json()["probabilidad"]
    claves = list(obra["variables"])
    random.Random(7).shuffle(claves)
    obra["variables"] = {k: obra["variables"][k] for k in claves}
    assert cliente.post("/api/predecir", json=obra).json()["probabilidad"] == a


def test_cambios_minimos_cambios_pequenos(cliente, obras):
    for o in obras[:15]:
        base = cliente.post("/api/predecir", json={"variables": o["variables"], "incluir_factores": False}).json()["probabilidad"]
        v = dict(o["variables"])
        v["area_construida_m2"] = v["area_construida_m2"] * 1.001
        nueva = cliente.post("/api/predecir", json={"variables": v, "incluir_factores": False}).json()["probabilidad"]
        assert abs(nueva - base) <= 0.05


# --- Sensibilidad ---

def test_sensibilidad_costo_preliminar(cliente, obras):
    cambios = []
    for o in obras:
        probs = []
        for f in (0.5, 1.0, 2.0):
            v = {**o["variables"], "costo_prelim_m2": o["variables"]["costo_prelim_m2"] * f}
            probs.append(cliente.post("/api/predecir", json={"variables": v, "incluir_factores": False}).json()["probabilidad"])
        cambios.append(probs)
    assert all(0 <= p <= 1 for fila in cambios for p in fila)
    sube_al_bajar = sum(1 for a, b, c in cambios if a >= b)
    assert sube_al_bajar / len(cambios) >= 0.5


# --- Rendimiento ---

def test_tiempo_respuesta(cliente, obra, obras):
    t0 = time.perf_counter()
    for _ in range(20):
        cliente.post("/api/predecir", json=obra)
    promedio = (time.perf_counter() - t0) / 20 * 1000
    t0 = time.perf_counter()
    lote(cliente, obras)
    total_lote = (time.perf_counter() - t0) * 1000
    assert promedio < 500 and total_lote < 3000
