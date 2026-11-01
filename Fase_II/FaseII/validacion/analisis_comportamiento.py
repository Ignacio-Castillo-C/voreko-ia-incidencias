# VOREKO IA · Fase II · Análisis del comportamiento del modelo a través de la API

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "api"))
import main

SALIDA = BASE / "validacion" / "resultados"
AZUL, NARANJA, GRIS, TINTA = "#2a78d6", "#eb6834", "#8a8a85", "#1f2a30"
NIVEL_COLOR = {"verde": "#2e9e5b", "amarillo": "#e9b10c", "rojo": "#d2412f"}
FACTORES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b9b6ae",
                     "axes.titleweight": "bold", "axes.titlesize": 11, "grid.color": "#e6e3dc"})


# --- Utilidades ---

def lote(c, filas):
    cuerpo = {"obras": [{"id_proyecto": str(i), "variables": v, "incluir_factores": False} for i, v in enumerate(filas)]}
    return [x["resultado"]["probabilidad"] for x in c.post("/api/predecir/lote", json=cuerpo).json()["resultados"]]


def guardar(fig, nombre):
    fig.savefig(SALIDA / f"{nombre}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# --- Análisis ---

def main_analisis():
    SALIDA.mkdir(parents=True, exist_ok=True)
    resumen = {}
    with TestClient(main.app) as c:
        meta = c.get("/api/modelo").json()
        obras = c.get("/api/obras-ejemplo?completo=true").json()
        ur, ua = meta["umbrales"]["rojo"], meta["umbrales"]["amarillo"]
        y = np.array([o["riesgo_real"] for o in obras])
        p = np.array(lote(c, [o["variables"] for o in obras]))

        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        bins = np.arange(0, 1.0001, 0.05)
        ax.hist(p[y == 0], bins=bins, color=AZUL, alpha=.85, label=f"Sin sobrecosto real (n={int((y == 0).sum())})", edgecolor="white")
        ax.hist(p[y == 1], bins=bins, color=NARANJA, alpha=.85, label=f"Con sobrecosto real (n={int(y.sum())})", edgecolor="white")
        for u, t in ((ua, f"amarillo {ua:.2f}"), (ur, f"rojo {ur:.2f}")):
            ax.axvline(u, color=TINTA, lw=1.2, ls="--")
            ax.text(u + 0.01, ax.get_ylim()[1] * 0.97, t, fontsize=9, va="top", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
        ax.set_xlabel("Probabilidad de sobrecosto devuelta por la API"); ax.set_ylabel("Obras")
        ax.set_title("Distribución de probabilidades en las 75 obras de prueba"); ax.legend(frameon=False, fontsize=9)
        guardar(fig, "01_distribucion_probabilidades")

        nivel = np.where(p >= ur, "rojo", np.where(p >= ua, "amarillo", "verde"))
        cruce = pd.crosstab(pd.Categorical(nivel, ["verde", "amarillo", "rojo"]), y).reindex(columns=[0, 1], fill_value=0)
        cruce.index.name, cruce.columns = "nivel", ["sin_sobrecosto_real", "con_sobrecosto_real"]
        cruce.to_csv(SALIDA / "semaforo_vs_real.csv")
        resumen["semaforo_vs_real"] = cruce.to_dict(orient="index")

        filas_s = []
        for var, etiqueta in (("costo_prelim_m2", "Costo preliminar por m²"), ("precio_venta_inicial_m2", "Precio de venta inicial por m²"),
                              ("area_construida_m2", "Área construida")):
            for f in FACTORES:
                variantes = []
                for o in obras:
                    v = dict(o["variables"])
                    v[var] = v[var] * f
                    if var == "costo_prelim_m2":
                        v["costo_prelim_m2_base"] = v["costo_prelim_m2_base"] * f
                    variantes.append(v)
                pf = np.array(lote(c, variantes))
                filas_s.append({"variable": var, "etiqueta": etiqueta, "factor": f, "prob_media": pf.mean(),
                                "pct_rojo": (pf >= ur).mean() * 100, "pct_verde": (pf < ua).mean() * 100})
        sens = pd.DataFrame(filas_s)
        sens.round(4).to_csv(SALIDA / "sensibilidad.csv", index=False)
        resumen["sensibilidad"] = sens.round(4).to_dict(orient="records")

        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
        for ax, (var, g) in zip(axes, sens.groupby("variable", sort=False)):
            ax.plot(g.factor, g.pct_rojo, color=NIVEL_COLOR["rojo"], lw=2, marker="o", ms=4, label="% en rojo")
            ax.plot(g.factor, g.pct_verde, color=NIVEL_COLOR["verde"], lw=2, marker="o", ms=4, label="% en verde")
            ax.axvline(1.0, color=GRIS, lw=1, ls=":")
            ax.set_title(g.etiqueta.iloc[0]); ax.set_xlabel("Factor aplicado (1 = valor original)"); ax.grid(True, axis="y")
        axes[0].set_ylabel("% de las 75 obras"); axes[0].legend(frameon=False, fontsize=9)
        fig.suptitle("Sensibilidad: ¿cómo cambia el semáforo si se modifica un dato?", fontweight="bold", y=1.03)
        guardar(fig, "02_sensibilidad")

        extremos = []
        base = dict(obras[0]["variables"])
        for var in ("area_construida_m2", "costo_prelim_m2", "precio_venta_inicial_m2", "tasa_interes_r1", "inpc_r1"):
            for f in (1e-6, 0.01, 100, 1e6):
                v = {**base, var: base[var] * f}
                r = c.post("/api/predecir", json={"variables": v, "incluir_factores": False}).json()
                extremos.append({"variable": var, "factor": f, "probabilidad": r["probabilidad"], "nivel": r["nivel"],
                                 "advertencias": len(r["advertencias"])})
        ext = pd.DataFrame(extremos)
        ext.to_csv(SALIDA / "valores_extremos.csv", index=False)
        resumen["valores_extremos"] = {"casos": len(ext), "prob_min": float(ext.probabilidad.min()), "prob_max": float(ext.probabilidad.max()),
                                       "todos_con_advertencia": bool((ext.advertencias > 0).all()),
                                       "obra_base": obras[0]["id_proyecto"],
                                       "prob_base": float(p[0])}

        imp = pd.Series(meta["importancia_global_top10"], dtype=float)
        nombres = {v["nombre"]: v["descripcion"] for v in meta["variables"]}
        etiquetas = [nombres.get(k.split("__", 1)[-1], k.split("__", 1)[-1]) for k in imp.index]
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.barh(etiquetas[::-1], imp.values[::-1], color=AZUL, height=0.6)
        ax.set_xlabel("Importancia en el bosque aleatorio"); ax.set_title("Las 10 variables que más usa el modelo")
        ax.grid(True, axis="x")
        guardar(fig, "03_importancia_variables")
        resumen["importancia_top10"] = dict(zip(etiquetas, imp.round(4).tolist()))

    (SALIDA / "analisis_comportamiento.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: resumen[k] for k in ("semaforo_vs_real", "valores_extremos")}, ensure_ascii=False, indent=1, default=float))
    print(sens[sens.factor.isin([0.5, 1.0, 2.0])].round(3).to_string(index=False))


if __name__ == "__main__":
    main_analisis()
