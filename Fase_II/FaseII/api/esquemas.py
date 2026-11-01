# VOREKO IA · Fase II · Esquemas de entrada y salida de la API

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SolicitudPrediccion(BaseModel):
    id_proyecto: str | None = Field(None, max_length=60, description="Identificador libre de la obra")
    variables: dict[str, Any] = Field(..., description="Variables de la obra disponibles al inicio (ver GET /api/modelo)")
    incluir_factores: bool = Field(True, description="Calcular los tres factores que más influyen en el resultado")


class SolicitudLote(BaseModel):
    obras: list[SolicitudPrediccion] = Field(..., min_length=1, description="Obras a evaluar (máximo definido por LIMITE_LOTE)")


class Factor(BaseModel):
    variable: str
    descripcion: str
    unidad: str
    valor: float
    valor_tipico: float
    efecto: float
    sentido: str


class ResultadoPrediccion(BaseModel):
    id_proyecto: str | None = None
    probabilidad: float
    probabilidad_pct: float
    nivel: Literal["verde", "amarillo", "rojo"]
    alerta: bool
    umbrales: dict[str, float]
    mensaje: str
    factores: list[Factor]
    advertencias: list[str]
    version_modelo: str
    tiempo_ms: float


class ResultadoLoteItem(BaseModel):
    id_proyecto: str | None = None
    resultado: ResultadoPrediccion | None = None
    errores: list[dict] | None = None


class ResultadoLote(BaseModel):
    total: int
    evaluadas: int
    con_error: int
    por_nivel: dict[str, int]
    resultados: list[ResultadoLoteItem]
    tiempo_ms: float


class Salud(BaseModel):
    estado: Literal["ok", "sin_modelo"]
    modelo_cargado: bool
    version_modelo: str | None
    version_api: str
    hora_servidor: str


class ErrorDetalle(BaseModel):
    variable: str
    problema: str


class RespuestaError(BaseModel):
    detalle: str
    errores: list[ErrorDetalle] = []
