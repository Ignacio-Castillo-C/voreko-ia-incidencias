# VOREKO IA · Actividad 9 · Microservicio de inferencia optimizado (misma interfaz de la Fase II)
# José Ignacio Castillo Camacho · AL07161745

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

RAIZ_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ_REPO / "FaseII" / "api"))
from esquemas import RespuestaError, ResultadoLote, ResultadoPrediccion, Salud, SolicitudLote, SolicitudPrediccion
from servicio_optimizado import ErrorValidacion, ServicioOptimizado

VERSION_API = "1.1.0"
DIR_MODELO = Path(os.getenv("VOREKO_DIR_MODELO_OPT", RAIZ_REPO / "Actividad9" / "modelo" / "optimizado"))
DIR_FRONTEND = Path(os.getenv("VOREKO_DIR_FRONTEND", "/no-existe"))
LIMITE_LOTE = int(os.getenv("LIMITE_LOTE", "500"))
ORIGENES = [o.strip() for o in os.getenv("VOREKO_CORS", "https://app.voreko.mx").split(",") if o.strip()]
CLAVE_API = os.getenv("VOREKO_API_KEY", "")

logging.basicConfig(level=os.getenv("NIVEL_LOG", "INFO"), format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("voreko")
servicio = ServicioOptimizado(DIR_MODELO)
ERRORES = {401: {"description": "Clave de acceso inválida"}, 422: {"model": RespuestaError, "description": "Datos inválidos"},
           503: {"description": "Modelo no cargado"}}


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


app = FastAPI(title="VOREKO IA · Microservicio de inferencia", version=VERSION_API, lifespan=ciclo_vida, docs_url="/docs", redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=ORIGENES, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def control_acceso(request: Request, call_next):
    if CLAVE_API and request.method == "POST" and request.url.path.startswith("/api/predecir"):
        if not hmac.compare_digest(request.headers.get("X-API-Key", ""), CLAVE_API):
            return JSONResponse(status_code=401, content={"detalle": "Falta la clave de acceso o no es válida"})
    t0 = time.perf_counter()
    respuesta = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    respuesta.headers["X-Tiempo-Respuesta-ms"] = f"{ms:.1f}"
    if request.url.path.startswith("/api"):
        log.info("%s %s -> %d (%.1f ms)", request.method, request.url.path, respuesta.status_code, ms)
    return respuesta


@app.exception_handler(RequestValidationError)
async def error_formato(request: Request, exc: RequestValidationError):
    errores = [{"variable": ".".join(str(x) for x in e["loc"][1:]) or "cuerpo", "problema": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detalle": "La solicitud no tiene el formato esperado", "errores": errores})


@app.exception_handler(ErrorValidacion)
async def error_negocio(request: Request, exc: ErrorValidacion):
    return JSONResponse(status_code=422, content={"detalle": "Los datos de la obra no son válidos", "errores": exc.errores})


def requiere_modelo():
    if servicio.modelo is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado")


def registrar_decision(id_proyecto, variables: dict, resultado: dict):
    huella = hashlib.sha256(json.dumps(variables, sort_keys=True).encode()).hexdigest()[:16]
    log.info(json.dumps({"evento": "prediccion", "id_proyecto": id_proyecto, "huella_entrada": huella, "version_modelo": resultado["version_modelo"],
                         "probabilidad": resultado["probabilidad"], "nivel": resultado["nivel"], "advertencias": len(resultado["advertencias"])},
                        ensure_ascii=False))


# --- Endpoints ---

@app.get("/api/salud", response_model=Salud, tags=["Servicio"])
def salud():
    cargado = servicio.modelo is not None
    contenido = {"estado": "ok" if cargado else "sin_modelo", "modelo_cargado": cargado, "version_modelo": servicio.version if cargado else None,
                 "version_api": VERSION_API, "hora_servidor": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return JSONResponse(status_code=200 if cargado else 503, content=contenido)


@app.get("/api/modelo", tags=["Modelo"])
def info_modelo():
    requiere_modelo()
    m = servicio.metadatos
    return {k: m[k] for k in ("nombre", "version", "algoritmo", "origen", "objetivo", "dataset", "umbrales", "metricas_prueba",
                              "obras", "variables", "versiones", "huellas_sha256")} | {"limite_lote": LIMITE_LOTE}


@app.get("/api/obras-ejemplo", tags=["Obras de ejemplo"])
def obras_ejemplo(completo: bool = False):
    requiere_modelo()
    if completo:
        return servicio.obras
    return [{"id_proyecto": o["id_proyecto"], "localidad": o["variables"]["localidad"], "anio_inicio": o["variables"]["anio_inicio"],
             "area_construida_m2": o["variables"]["area_construida_m2"], "costo_prelim_m2": o["variables"]["costo_prelim_m2"],
             "sobrecosto_real_pct": o["sobrecosto_real_pct"], "riesgo_real": o["riesgo_real"]} for o in servicio.obras]


@app.get("/api/obras-ejemplo/{id_proyecto}", tags=["Obras de ejemplo"])
def obra_ejemplo(id_proyecto: str):
    requiere_modelo()
    obra = servicio.obra(id_proyecto)
    if obra is None:
        raise HTTPException(status_code=404, detail=f"No existe la obra {id_proyecto}")
    return obra


@app.post("/api/predecir", response_model=ResultadoPrediccion, tags=["Predicción"], responses=ERRORES)
def predecir(solicitud: SolicitudPrediccion = Body(...)):
    requiere_modelo()
    t0 = time.perf_counter()
    resultado = servicio.predecir(solicitud.variables, solicitud.incluir_factores)
    registrar_decision(solicitud.id_proyecto, solicitud.variables, resultado)
    return {"id_proyecto": solicitud.id_proyecto, **resultado, "tiempo_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.post("/api/predecir/lote", response_model=ResultadoLote, tags=["Predicción"], responses={**ERRORES, 413: {"description": "Lote demasiado grande"}})
def predecir_lote(solicitud: SolicitudLote = Body(...)):
    requiere_modelo()
    if len(solicitud.obras) > LIMITE_LOTE:
        raise HTTPException(status_code=413, detail=f"El lote tiene {len(solicitud.obras)} obras; el máximo es {LIMITE_LOTE}")
    t0 = time.perf_counter()
    crudos = servicio.predecir_lote([(o.variables, o.incluir_factores) for o in solicitud.obras])
    ms = round((time.perf_counter() - t0) * 1000, 2)
    salida, por_nivel = [], {"verde": 0, "amarillo": 0, "rojo": 0}
    for o, r in zip(solicitud.obras, crudos):
        if "resultado" in r:
            por_nivel[r["resultado"]["nivel"]] += 1
            salida.append({"id_proyecto": o.id_proyecto, "resultado": {"id_proyecto": o.id_proyecto, **r["resultado"], "tiempo_ms": round(ms / len(crudos), 3)}})
        else:
            salida.append({"id_proyecto": o.id_proyecto, "errores": r["errores"]})
    con_error = sum(1 for s in salida if s.get("errores"))
    return {"total": len(salida), "evaluadas": len(salida) - con_error, "con_error": con_error, "por_nivel": por_nivel, "resultados": salida, "tiempo_ms": ms}


@app.get("/api/cache", tags=["Servicio"])
def estado_cache():
    requiere_modelo()
    return servicio.cache.estado()


# --- Frontend opcional ---

if DIR_FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=DIR_FRONTEND), name="frontend")

    @app.get("/", include_in_schema=False)
    def inicio():
        return FileResponse(DIR_FRONTEND / "index.html")
