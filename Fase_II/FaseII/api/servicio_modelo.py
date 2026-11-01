# VOREKO IA · Fase II · Servicio del modelo de riesgo de sobrecosto

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MENSAJES = {
    "rojo": "Riesgo alto de sobrecosto. Antes de iniciar conviene revisar el presupuesto, actualizar precios unitarios y definir una contingencia.",
    "amarillo": "Riesgo intermedio. Se recomienda dar seguimiento mensual al gasto contra lo presupuestado.",
    "verde": "Riesgo bajo según el modelo. Mantener el seguimiento normal de la obra.",
}
POSITIVAS = ["area_construida_m2", "area_terreno_m2", "costo_prelim_m2", "costo_prelim_m2_base", "precio_venta_inicial_m2"]


class ErrorValidacion(ValueError):

    def __init__(self, errores: list[dict]):
        super().__init__("; ".join(f"{e['variable']}: {e['problema']}" for e in errores))
        self.errores = errores


@dataclass
class ServicioModelo:
    directorio: Path
    modelo: object = None
    metadatos: dict = field(default_factory=dict)
    obras: list = field(default_factory=list)

    # --- Carga ---

    def cargar(self) -> "ServicioModelo":
        self.modelo = joblib.load(self.directorio / "modelo_riesgo.joblib")
        self.modelo.named_steps["modelo"].set_params(n_jobs=1)
        self.metadatos = json.loads((self.directorio / "metadatos_modelo.json").read_text(encoding="utf-8"))
        self.obras = json.loads((self.directorio / "obras_ejemplo.json").read_text(encoding="utf-8"))
        self._vars = {v["nombre"]: v for v in self.metadatos["variables"]}
        self.columnas = [v["nombre"] for v in self.metadatos["variables"]]
        self.categoricas = set(self.metadatos["variables_categoricas"])
        self.u_rojo = float(self.metadatos["umbrales"]["rojo"])
        self.u_amarillo = float(self.metadatos["umbrales"]["amarillo"])
        self._referencia = {c: self._vars[c]["tipico"] for c in self.columnas}
        return self

    @property
    def version(self) -> str:
        return f"{self.metadatos['nombre']} v{self.metadatos['version']}"

    # --- Validación ---

    def validar(self, variables: dict) -> tuple[dict, list[str]]:
        errores, advertencias = [], []
        faltantes = [c for c in self.columnas if c not in variables]
        for c in faltantes:
            errores.append({"variable": c, "problema": "falta la variable"})
        extra = sorted(set(variables) - set(self.columnas))
        if extra:
            advertencias.append(f"Se ignoraron variables que el modelo no usa: {', '.join(extra)}")
        limpio = {}
        for c in self.columnas:
            if c not in variables:
                continue
            x = variables[c]
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
                errores.append({"variable": c, "problema": "debe ser un número finito"})
                continue
            x = float(x)
            meta = self._vars[c]
            if c in self.categoricas:
                if not x.is_integer() or int(x) not in meta["valores_permitidos"]:
                    errores.append({"variable": c, "problema": f"valor no permitido; use uno de {meta['valores_permitidos']}"})
                    continue
                limpio[c] = int(x)
                continue
            if c in POSITIVAS and x <= 0:
                errores.append({"variable": c, "problema": "debe ser mayor que cero"})
                continue
            if c == "anio_inicio" and not x.is_integer():
                errores.append({"variable": c, "problema": "debe ser un año entero"})
                continue
            if x < meta["minimo"] or x > meta["maximo"]:
                advertencias.append(f"{meta['descripcion']} ({c}) = {x:g} está fuera del rango visto en entrenamiento "
                                    f"[{meta['minimo']:g} – {meta['maximo']:g}]; el resultado es menos confiable")
            limpio[c] = x
        if errores:
            raise ErrorValidacion(errores)
        return limpio, advertencias

    # --- Predicción ---

    def nivel(self, p: float) -> str:
        return "rojo" if p >= self.u_rojo else "amarillo" if p >= self.u_amarillo else "verde"

    def _marco(self, filas: list[dict]) -> pd.DataFrame:
        X = pd.DataFrame(filas, columns=self.columnas)
        for c in self.columnas:
            X[c] = X[c].astype("int64" if c in self.categoricas else "float64")
        return X

    def _factores(self, fila: dict, base: float, p_alt: np.ndarray, top: int = 3) -> list[dict]:
        efectos = sorted(((c, base - float(p)) for c, p in zip(self.columnas, p_alt)), key=lambda t: -abs(t[1]))
        salida = []
        for c, e in efectos[:top]:
            meta = self._vars[c]
            salida.append({"variable": c, "descripcion": meta["descripcion"], "unidad": meta["unidad"], "valor": fila[c],
                           "valor_tipico": self._referencia[c], "efecto": round(e, 4),
                           "sentido": "aumenta el riesgo" if e > 0 else "reduce el riesgo" if e < 0 else "sin efecto"})
        return salida

    def _resultado(self, p: float, factores: list, advertencias: list) -> dict:
        nivel = self.nivel(p)
        return {"probabilidad": round(p, 4), "probabilidad_pct": round(p * 100, 1), "nivel": nivel,
                "alerta": p >= self.u_rojo, "umbrales": {"rojo": self.u_rojo, "amarillo": self.u_amarillo},
                "mensaje": MENSAJES[nivel], "factores": factores, "advertencias": advertencias, "version_modelo": self.version}

    def predecir(self, variables: dict, incluir_factores: bool = True) -> dict:
        fila, advertencias = self.validar(variables)
        filas = [fila]
        if incluir_factores:
            filas += [{**fila, c: self._referencia[c]} for c in self.columnas]
        p = self.modelo.predict_proba(self._marco(filas))[:, 1]
        base = float(p[0])
        return self._resultado(base, self._factores(fila, base, p[1:]) if incluir_factores else [], advertencias)

    def predecir_lote(self, solicitudes: list[tuple[dict, bool]]) -> list[dict]:
        salida, validas = [], []
        for k, (variables, incluir) in enumerate(solicitudes):
            try:
                fila, adv = self.validar(variables)
                validas.append((k, fila, adv, incluir))
                salida.append(None)
            except ErrorValidacion as e:
                salida.append({"errores": e.errores})
        if validas:
            p = self.modelo.predict_proba(self._marco([f for _, f, _, _ in validas]))[:, 1]
            for (k, fila, adv, incluir), pk in zip(validas, p):
                if incluir:
                    salida[k] = {"resultado": self.predecir(fila, True) | {"advertencias": adv}}
                else:
                    salida[k] = {"resultado": self._resultado(float(pk), [], adv)}
        return salida

    def obra(self, id_proyecto: str) -> dict | None:
        return next((o for o in self.obras if o["id_proyecto"] == id_proyecto), None)
