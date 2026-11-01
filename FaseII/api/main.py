# VOREKO IA · Fase II · API REST del sistema de alertas de sobrecosto

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from esquemas import RespuestaError, ResultadoLote, ResultadoPrediccion, Salud, SolicitudLote, SolicitudPrediccion
from servicio_modelo import ErrorValidacion, ServicioModelo

VERSION_API = "1.0.0"
BASE = Path(__file__).resolve().parents[1]
DIR_MODELO = Path(os.getenv("VOREKO_DIR_MODELO", BASE / "modelo" / "artefactos"))
DIR_FRONTEND = Path(os.getenv("VOREKO_DIR_FRONTEND", BASE / "frontend"))
LIMITE_LOTE = int(os.getenv("LIMITE_LOTE", "500"))
ORIGENES = [o.strip() for o in os.getenv("VOREKO_CORS", "*").split(",") if o.strip()]

logging.basicConfig(level=os.getenv("NIVEL_LOG", "INFO"), format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("voreko")
servicio = ServicioModelo(DIR_MODELO)
ERRORES = {422: {"model": RespuestaError, "description": "Datos inválidos"}, 503: {"description": "Modelo no cargado"}}


def ejemplo_swagger() -> dict:
    try:
        o = json.loads((DIR_MODELO / "obras_ejemplo.json").read_text(encoding="utf-8"))[0]
        return {"id_proyecto": o["id_proyecto"], "variables": o["variables"], "incluir_factores": True}
    except Exception:
        return {"id_proyecto": "OBRA-DEMO", "variables": {}, "incluir_factores": True}


EJEMPLO = ejemplo_swagger()


# --- Ciclo de vida ---

@asynccontextmanager
async def ciclo_vida(app: FastAPI):
    try:
        servicio.cargar()
        log.info("Modelo cargado: %s | umbral rojo %.2f | umbral amarillo %.2f | %d variables",
                 servicio.version, servicio.u_rojo, servicio.u_amarillo, len(servicio.columnas))
    except Exception as e:
        log.error("No se pudo cargar el modelo desde %s: %s", DIR_MODELO, e)
    yield


app = FastAPI(
    title="VOREKO IA · API de riesgo de sobrecosto",
    version=VERSION_API,
    description="Servicio que estima la probabilidad de que una obra termine con un sobrecosto mayor al 50 % "
                "y la traduce a un semáforo (verde, amarillo, rojo). Modelo candidato de la Actividad 6.",
    lifespan=ciclo_vida,
    docs_url=None,
    redoc_url=None,
)
app.add_middleware(CORSMiddleware, allow_origins=ORIGENES, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def bitacora(request: Request, call_next):
    t0 = time.perf_counter()
    respuesta = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    respuesta.headers["X-Tiempo-Respuesta-ms"] = f"{ms:.1f}"
    if request.url.path.startswith("/api"):
        log.info("%s %s -> %d (%.1f ms)", request.method, request.url.path, respuesta.status_code, ms)
    return respuesta


TRADUCCION = {"missing": "falta el campo", "too_short": "la lista no puede estar vacía", "json_invalid": "el JSON está mal formado",
              "dict_type": "debe ser un objeto con pares nombre: valor", "list_type": "debe ser una lista",
              "string_too_long": "el texto es demasiado largo", "bool_parsing": "debe ser verdadero o falso",
              "model_attributes_type": "el cuerpo debe ser un objeto JSON", "string_type": "debe ser texto"}


@app.exception_handler(RequestValidationError)
async def error_formato(request: Request, exc: RequestValidationError):
    errores = [{"variable": ".".join(str(x) for x in e["loc"][1:]) or "cuerpo", "problema": TRADUCCION.get(e["type"], e["msg"])}
               for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detalle": "La solicitud no tiene el formato esperado", "errores": errores})


@app.exception_handler(ErrorValidacion)
async def error_negocio(request: Request, exc: ErrorValidacion):
    return JSONResponse(status_code=422, content={"detalle": "Los datos de la obra no son válidos", "errores": exc.errores})


def requiere_modelo():
    if servicio.modelo is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado")


# --- Endpoints ---

@app.get("/api/salud", response_model=Salud, tags=["Servicio"], summary="Estado del servicio")
def salud():
    cargado = servicio.modelo is not None
    contenido = {"estado": "ok" if cargado else "sin_modelo", "modelo_cargado": cargado,
                 "version_modelo": servicio.version if cargado else None, "version_api": VERSION_API,
                 "hora_servidor": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return JSONResponse(status_code=200 if cargado else 503, content=contenido)


@app.get("/api/modelo", tags=["Modelo"], summary="Información del modelo, variables y métricas")
def info_modelo():
    requiere_modelo()
    m = servicio.metadatos
    return {k: m[k] for k in ("nombre", "version", "algoritmo", "origen", "objetivo", "dataset", "fecha_entrenamiento",
                              "hiperparametros", "umbrales", "regla_umbral", "metricas_prueba", "matriz_confusion_prueba",
                              "semaforo_prueba", "obras", "variables", "importancia_global_top10", "versiones", "huellas_sha256")} | {"limite_lote": LIMITE_LOTE}


@app.get("/api/obras-ejemplo", tags=["Obras de ejemplo"], summary="Lista de las 75 obras de prueba")
def obras_ejemplo(completo: bool = False):
    requiere_modelo()
    if completo:
        return servicio.obras
    return [{"id_proyecto": o["id_proyecto"], "localidad": o["variables"]["localidad"],
             "anio_inicio": o["variables"]["anio_inicio"], "area_construida_m2": o["variables"]["area_construida_m2"],
             "costo_prelim_m2": o["variables"]["costo_prelim_m2"], "sobrecosto_real_pct": o["sobrecosto_real_pct"],
             "riesgo_real": o["riesgo_real"]} for o in servicio.obras]


@app.get("/api/obras-ejemplo/{id_proyecto}", tags=["Obras de ejemplo"], summary="Variables de una obra de prueba")
def obra_ejemplo(id_proyecto: str):
    requiere_modelo()
    obra = servicio.obra(id_proyecto)
    if obra is None:
        raise HTTPException(status_code=404, detail=f"No existe la obra {id_proyecto}")
    return obra


@app.post("/api/predecir", response_model=ResultadoPrediccion, tags=["Predicción"], summary="Evaluar una obra", responses=ERRORES)
def predecir(solicitud: SolicitudPrediccion = Body(..., openapi_examples={"obra_de_prueba": {
        "summary": "Obra de prueba de la Actividad 6", "value": EJEMPLO}})):
    requiere_modelo()
    t0 = time.perf_counter()
    resultado = servicio.predecir(solicitud.variables, solicitud.incluir_factores)
    return {"id_proyecto": solicitud.id_proyecto, **resultado, "tiempo_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.post("/api/predecir/lote", response_model=ResultadoLote, tags=["Predicción"], summary="Evaluar varias obras",
          responses={**ERRORES, 413: {"description": "Lote demasiado grande"}})
def predecir_lote(solicitud: SolicitudLote = Body(..., openapi_examples={"dos_obras": {
        "summary": "Lote con dos obras", "value": {"obras": [EJEMPLO, {**EJEMPLO, "id_proyecto": "OBRA-COPIA"}]}}})):
    requiere_modelo()
    if len(solicitud.obras) > LIMITE_LOTE:
        raise HTTPException(status_code=413, detail=f"El lote tiene {len(solicitud.obras)} obras; el máximo es {LIMITE_LOTE}")
    t0 = time.perf_counter()
    crudos = servicio.predecir_lote([(o.variables, o.incluir_factores) for o in solicitud.obras])
    ms = round((time.perf_counter() - t0) * 1000, 1)
    salida, por_nivel = [], {"verde": 0, "amarillo": 0, "rojo": 0}
    for o, r in zip(solicitud.obras, crudos):
        if "resultado" in r:
            por_nivel[r["resultado"]["nivel"]] += 1
            salida.append({"id_proyecto": o.id_proyecto, "resultado": {"id_proyecto": o.id_proyecto, **r["resultado"],
                                                                       "tiempo_ms": round(ms / len(crudos), 2)}})
        else:
            salida.append({"id_proyecto": o.id_proyecto, "errores": r["errores"]})
    con_error = sum(1 for s in salida if s.get("errores"))
    return {"total": len(salida), "evaluadas": len(salida) - con_error, "con_error": con_error, "por_nivel": por_nivel,
            "resultados": salida, "tiempo_ms": ms}


# --- Frontend ---

if DIR_FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=DIR_FRONTEND), name="frontend")

    @app.get("/", include_in_schema=False)
    def inicio():
        return FileResponse(DIR_FRONTEND / "index.html")

    @app.get("/docs", include_in_schema=False)
    def swagger():
        return get_swagger_ui_html(openapi_url=app.openapi_url, title=f"{app.title} · Swagger",
                                   swagger_js_url="/static/vendor/swagger-ui-bundle.js",
                                   swagger_css_url="/static/vendor/swagger-ui.css",
                                   swagger_favicon_url="/static/vendor/favicon-32x32.png")

    @app.get("/redoc", include_in_schema=False)
    def redoc():
        return get_redoc_html(openapi_url=app.openapi_url, title=f"{app.title} · ReDoc",
                              redoc_js_url="/static/vendor/redoc.standalone.js", with_google_fonts=False)
