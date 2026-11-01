# VOREKO IA · Fase II · Casos extremos y entradas inválidas

import pytest


def errores(r):
    return {e["variable"]: e["problema"] for e in r.json().get("errores", [])}


# --- Datos faltantes o con tipo incorrecto ---

def test_falta_una_variable(cliente, obra):
    obra["variables"].pop("localidad")
    r = cliente.post("/api/predecir", json=obra)
    assert r.status_code == 422 and errores(r)["localidad"] == "falta la variable"


def test_variables_vacias(cliente):
    r = cliente.post("/api/predecir", json={"variables": {}})
    assert r.status_code == 422 and len(errores(r)) == 28


@pytest.mark.parametrize("valor", ["abc", None, True, [1, 2], {"x": 1}])
def test_tipo_incorrecto(cliente, obra, valor):
    obra["variables"]["costo_prelim_m2"] = valor
    r = cliente.post("/api/predecir", json=obra)
    assert r.status_code == 422 and "costo_prelim_m2" in errores(r)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_no_finitos(cliente, obra, literal):
    import json
    texto = json.dumps(obra).replace('"inpc_r1": ' + json.dumps(obra["variables"]["inpc_r1"]), f'"inpc_r1": {literal}')
    r = cliente.post("/api/predecir", content=texto, headers={"Content-Type": "application/json"})
    assert r.status_code == 422 and "inpc_r1" in errores(r)


# --- Reglas de negocio ---

@pytest.mark.parametrize("variable,valor", [("area_construida_m2", 0), ("area_construida_m2", -150), ("area_terreno_m2", -1),
                                            ("costo_prelim_m2", 0), ("precio_venta_inicial_m2", -0.5)])
def test_no_positivos(cliente, obra, variable, valor):
    obra["variables"][variable] = valor
    r = cliente.post("/api/predecir", json=obra)
    assert r.status_code == 422 and errores(r)[variable] == "debe ser mayor que cero"


@pytest.mark.parametrize("variable,valor", [("trimestre_inicio", 0), ("trimestre_inicio", 5), ("trimestre_inicio", 2.5),
                                            ("localidad", 0), ("localidad", 21), ("localidad", 3.7)])
def test_categorias_invalidas(cliente, obra, variable, valor):
    obra["variables"][variable] = valor
    r = cliente.post("/api/predecir", json=obra)
    assert r.status_code == 422 and "valor no permitido" in errores(r)[variable]


def test_anio_no_entero(cliente, obra):
    obra["variables"]["anio_inicio"] = 1385.5
    assert cliente.post("/api/predecir", json=obra).status_code == 422


def test_varios_errores_juntos(cliente, obra):
    obra["variables"].update({"area_construida_m2": -1, "trimestre_inicio": 9, "inpc_r1": "x"})
    obra["variables"].pop("localidad")
    r = cliente.post("/api/predecir", json=obra)
    assert r.status_code == 422 and len(errores(r)) == 4


# --- Valores fuera del rango de entrenamiento ---

def test_fuera_de_rango_advierte(cliente, obra, metadatos):
    maximo = next(v["maximo"] for v in metadatos["variables"] if v["nombre"] == "area_construida_m2")
    obra["variables"]["area_construida_m2"] = maximo * 10
    r = cliente.post("/api/predecir", json=obra)
    assert r.status_code == 200
    assert any("fuera del rango" in a for a in r.json()["advertencias"])


def test_valor_gigante(cliente, obra):
    obra["variables"]["costo_prelim_m2"] = 1e12
    r = cliente.post("/api/predecir", json=obra)
    assert r.status_code == 200 and 0 <= r.json()["probabilidad"] <= 1 and r.json()["advertencias"]


def test_valor_minimo_positivo(cliente, obra):
    obra["variables"]["costo_prelim_m2"] = 1e-9
    r = cliente.post("/api/predecir", json=obra)
    assert r.status_code == 200 and r.json()["advertencias"]


def test_variable_extra_se_ignora(cliente, obra):
    base = cliente.post("/api/predecir", json=obra).json()["probabilidad"]
    obra["variables"]["color_fachada"] = 3
    r = cliente.post("/api/predecir", json=obra)
    assert r.status_code == 200 and r.json()["probabilidad"] == base
    assert any("color_fachada" in a for a in r.json()["advertencias"])


def test_numeros_enteros_y_decimales_equivalentes(cliente, obra):
    a = cliente.post("/api/predecir", json=obra).json()["probabilidad"]
    obra["variables"]["localidad"] = float(obra["variables"]["localidad"])
    obra["variables"]["area_construida_m2"] = int(obra["variables"]["area_construida_m2"])
    assert cliente.post("/api/predecir", json=obra).json()["probabilidad"] == a


# --- Cuerpo de la solicitud mal formado ---

@pytest.mark.parametrize("texto", ["{malo", "", "[]", '"hola"', '{"variables": [1, 2]}', '{"otra_cosa": 1}'])
def test_cuerpo_mal_formado(cliente, texto):
    r = cliente.post("/api/predecir", content=texto, headers={"Content-Type": "application/json"})
    assert r.status_code == 422 and r.json()["detalle"]


def test_id_demasiado_largo(cliente, obra):
    obra["id_proyecto"] = "X" * 61
    assert cliente.post("/api/predecir", json=obra).status_code == 422


def test_metodo_no_permitido(cliente):
    assert cliente.get("/api/predecir").status_code == 405


def test_ruta_inexistente(cliente):
    assert cliente.get("/api/no-existe").status_code == 404


# --- Lotes ---

def test_lote_vacio(cliente):
    r = cliente.post("/api/predecir/lote", json={"obras": []})
    assert r.status_code == 422 and errores(r)["obras"] == "la lista no puede estar vacía"


def test_lote_demasiado_grande(cliente, obra):
    r = cliente.post("/api/predecir/lote", json={"obras": [obra] * 501})
    assert r.status_code == 413


def test_lote_en_el_limite(cliente, obra):
    obra["incluir_factores"] = False
    r = cliente.post("/api/predecir/lote", json={"obras": [obra] * 500})
    assert r.status_code == 200 and r.json()["evaluadas"] == 500


def test_lote_mixto(cliente, obras):
    buenas = [{"id_proyecto": o["id_proyecto"], "variables": o["variables"], "incluir_factores": False} for o in obras[:3]]
    mala = {"id_proyecto": "MALA", "variables": {**obras[3]["variables"], "localidad": 99}}
    r = cliente.post("/api/predecir/lote", json={"obras": buenas + [mala]})
    c = r.json()
    assert r.status_code == 200 and c["evaluadas"] == 3 and c["con_error"] == 1
    assert c["resultados"][3]["errores"][0]["variable"] == "localidad"


# --- Servicio sin modelo ---

def test_sin_modelo_responde_503(cliente, obra, monkeypatch):
    import main
    monkeypatch.setattr(main.servicio, "modelo", None)
    assert cliente.get("/api/salud").status_code == 503
    assert cliente.post("/api/predecir", json=obra).status_code == 503
    assert cliente.get("/api/modelo").status_code == 503
