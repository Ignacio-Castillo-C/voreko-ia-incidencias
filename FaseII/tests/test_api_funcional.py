# VOREKO IA · Fase II · Pruebas funcionales de la API y del frontend


# --- Servicio y modelo ---

def test_salud(cliente):
    r = cliente.get("/api/salud")
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["estado"] == "ok" and cuerpo["modelo_cargado"] is True
    assert cuerpo["version_modelo"] == "voreko_riesgo_sobrecosto v1.0.0"


def test_info_modelo(metadatos):
    assert metadatos["umbrales"] == {"rojo": 0.39, "amarillo": 0.14}
    assert len(metadatos["variables"]) == 28
    assert metadatos["obras"] == {"entrenamiento": 297, "prueba": 75, "prevalencia_prueba": 0.3467}
    assert round(metadatos["metricas_prueba"]["roc_auc"], 3) == 0.803


# --- Obras de ejemplo ---

def test_lista_obras(cliente):
    r = cliente.get("/api/obras-ejemplo")
    assert r.status_code == 200
    lista = r.json()
    assert len(lista) == 75
    assert sum(o["riesgo_real"] for o in lista) == 26


def test_obra_por_id(cliente, obras):
    r = cliente.get(f"/api/obras-ejemplo/{obras[5]['id_proyecto']}")
    assert r.status_code == 200 and r.json()["variables"] == obras[5]["variables"]


def test_obra_inexistente(cliente):
    assert cliente.get("/api/obras-ejemplo/NO-EXISTE").status_code == 404


# --- Predicción individual ---

def test_predecir_estructura(cliente, obra):
    r = cliente.post("/api/predecir", json=obra)
    assert r.status_code == 200
    c = r.json()
    assert set(c) >= {"id_proyecto", "probabilidad", "probabilidad_pct", "nivel", "alerta", "umbrales", "mensaje", "factores",
                      "advertencias", "version_modelo", "tiempo_ms"}
    assert c["id_proyecto"] == obra["id_proyecto"]
    assert 0 <= c["probabilidad"] <= 1
    assert c["nivel"] in ("verde", "amarillo", "rojo")
    assert len(c["factores"]) == 3 and all(f["sentido"] for f in c["factores"])


def test_predecir_coincide_con_referencia(cliente, obras):
    for o in obras[:10]:
        p = cliente.post("/api/predecir", json={"variables": o["variables"], "incluir_factores": False}).json()["probabilidad"]
        assert abs(p - round(o["probabilidad_referencia"], 4)) < 1e-9


def test_predecir_sin_factores(cliente, obra):
    obra["incluir_factores"] = False
    assert cliente.post("/api/predecir", json=obra).json()["factores"] == []


# --- Predicción por lote ---

def test_lote_completo(cliente, obras, metadatos):
    cuerpo = {"obras": [{"id_proyecto": o["id_proyecto"], "variables": o["variables"], "incluir_factores": False} for o in obras]}
    r = cliente.post("/api/predecir/lote", json=cuerpo)
    assert r.status_code == 200
    c = r.json()
    assert c["total"] == 75 and c["evaluadas"] == 75 and c["con_error"] == 0
    esperado = {k: v["obras"] for k, v in metadatos["semaforo_prueba"].items()}
    assert c["por_nivel"] == esperado


def test_lote_igual_a_individual(cliente, obras):
    cuerpo = {"obras": [{"id_proyecto": o["id_proyecto"], "variables": o["variables"], "incluir_factores": False} for o in obras[:5]]}
    lote = cliente.post("/api/predecir/lote", json=cuerpo).json()["resultados"]
    for item, o in zip(lote, obras[:5]):
        individual = cliente.post("/api/predecir", json={"variables": o["variables"], "incluir_factores": False}).json()
        assert item["resultado"]["probabilidad"] == individual["probabilidad"]


# --- Frontend, documentación y cabeceras ---

def test_frontend_servido(cliente):
    r = cliente.get("/")
    assert r.status_code == 200 and "VOREKO" in r.text and "app.js" in r.text
    assert cliente.get("/static/app.js").status_code == 200
    assert cliente.get("/static/estilos.css").status_code == 200


def test_documentacion_openapi(cliente):
    esquema = cliente.get("/openapi.json").json()
    for ruta in ("/api/salud", "/api/modelo", "/api/obras-ejemplo", "/api/obras-ejemplo/{id_proyecto}", "/api/predecir",
                 "/api/predecir/lote"):
        assert ruta in esquema["paths"]
    assert cliente.get("/docs").status_code == 200
    assert cliente.get("/redoc").status_code == 200


def test_cabeceras(cliente):
    r = cliente.get("/api/salud", headers={"Origin": "https://app.flutterflow.io"})
    assert r.headers.get("access-control-allow-origin") == "*"
    assert float(r.headers["x-tiempo-respuesta-ms"]) >= 0
