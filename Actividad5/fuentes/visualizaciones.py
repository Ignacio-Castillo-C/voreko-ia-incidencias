# -*- coding: utf-8 -*-
"""
visualizaciones.py — Análisis exploratorio para detectar patrones y anomalías
=============================================================================
Cada función genera una figura PNG en resultados/figuras/ y devuelve los
números que la sustentan, para que la interpretación del notebook y del
reporte se base en datos y no en la apariencia de la gráfica.

    python fuentes/visualizaciones.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos_prep as dp  # noqa: E402

DIR_FIG = dp.RAIZ / "resultados" / "figuras"
AZUL, NARANJA, GRIS, TINTA, SEC, REJ = "#2a78d6", "#eb6834", "#9a998f", "#1f1f1e", "#5f5e58", "#e4e3dc"


def _estilo():
    plt.rcParams.update({"axes.edgecolor": REJ, "axes.labelcolor": SEC, "xtick.color": SEC, "ytick.color": SEC,
                         "axes.titleweight": "bold", "axes.titlesize": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9.5})


def _guardar(nombre):
    DIR_FIG.mkdir(parents=True, exist_ok=True)
    ruta = DIR_FIG / nombre
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    return ruta


def distribucion_objetivo(df):
    _estilo()
    y = df[dp.OBJETIVO]
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.6))
    ejes[0].hist(y, bins=30, color=AZUL, edgecolor="white")
    ejes[0].set(title="Costo real por m² (escala original)", xlabel="MIRR/m²", ylabel="Proyectos")
    ejes[1].hist(np.log(y), bins=30, color=AZUL, edgecolor="white")
    ejes[1].set(title="Costo real por m² (escala logarítmica)", xlabel="log(MIRR/m²)")
    for ax in ejes:
        ax.grid(axis="y", color=REJ, linewidth=0.7)
        ax.set_axisbelow(True)
    return _guardar("eda_01_distribucion_objetivo.png"), {
        "asimetria_original": round(float(y.skew()), 2), "asimetria_log": round(float(np.log(y).skew()), 2),
        "mediana": round(float(y.median()), 2), "max": round(float(y.max()), 2)}


def tendencia_temporal(df):
    """Costo real y preliminar por año de inicio (dos paneles, misma escala)."""
    _estilo()
    g = df.groupby("anio_inicio").agg(real=(dp.OBJETIVO, "median"), prelim=("costo_prelim_m2", "median"),
                                      razon=("razon_real_vs_preliminar", "median"), n=("id_proyecto", "count"))
    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.8))
    ejes[0].plot(g.index, g["prelim"], marker="o", color=GRIS, linewidth=2, label="Preliminar (al inicio)")
    ejes[0].plot(g.index, g["real"], marker="o", color=AZUL, linewidth=2, label="Real (al terminar)")
    ejes[0].set(title="Mediana del costo por m² según año de inicio", xlabel="Año de inicio (calendario persa)",
                ylabel="MIRR/m²")
    ejes[0].legend(frameon=False)
    ejes[1].bar(g.index, g["razon"], color=NARANJA, edgecolor="white", width=0.7)
    ejes[1].axhline(1, color=SEC, linestyle="--", linewidth=1)
    ejes[1].set(title="Mediana de la razón costo real / preliminar", xlabel="Año de inicio", ylabel="Veces el presupuesto")
    ejes[1].set_xticks(g.index[::2], [str(int(v)) for v in g.index[::2]])
    for ax in ejes:
        ax.grid(color=REJ, linewidth=0.7)
        ax.set_axisbelow(True)
    return _guardar("eda_02_tendencia_temporal.png"), {
        "costo_real_mediano_primer_anio": round(float(g["real"].iloc[0]), 2),
        "costo_real_mediano_ultimo_anio": round(float(g["real"].iloc[-1]), 2),
        "razon_max_anio": int(g["razon"].idxmax()), "razon_max": round(float(g["razon"].max()), 2),
        "razon_min_anio": int(g["razon"].idxmin()), "razon_min": round(float(g["razon"].min()), 2),
        "proyectos_por_anio_min": int(g["n"].min()), "proyectos_por_anio_max": int(g["n"].max())}


def preliminar_vs_real(df):
    """Detecta visualmente las desviaciones de costo atípicas (las 12 marcadas)."""
    _estilo()
    an = df["anomalia_razon_costo"]
    fig, ax = plt.subplots(figsize=(6.8, 5))
    ax.scatter(df.loc[~an, "costo_prelim_m2"], df.loc[~an, dp.OBJETIVO], s=18, color=AZUL, alpha=0.7,
               edgecolor="white", linewidth=0.5, label="Proyecto")
    ax.scatter(df.loc[an, "costo_prelim_m2"], df.loc[an, dp.OBJETIVO], s=48, color=NARANJA, edgecolor=TINTA,
               linewidth=0.6, label=f"Desviación atípica ({int(an.sum())})", zorder=3)
    lim = [df["costo_prelim_m2"].min() * 0.8, df[dp.OBJETIVO].max() * 1.2]
    ax.plot(lim, lim, "--", color=SEC, linewidth=1, label="Real = preliminar")
    med = df["razon_real_vs_preliminar"].median()
    ax.plot(lim, [v * med for v in lim], ":", color=TINTA, linewidth=1, label=f"Real = {med:.2f} × preliminar")
    ax.set(xscale="log", yscale="log", xlabel="Costo preliminar por m² (MIRR, escala log)",
           ylabel="Costo real por m² (MIRR, escala log)", title="Costo preliminar frente a costo real")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(color=REJ, linewidth=0.7, which="both")
    ax.set_axisbelow(True)
    corr = np.corrcoef(np.log(df["costo_prelim_m2"]), np.log(df[dp.OBJETIVO]))[0, 1]
    return _guardar("eda_03_preliminar_vs_real.png"), {
        "correlacion_log": round(float(corr), 3), "razon_mediana": round(float(med), 3),
        "pct_sobre_presupuesto": round(float((df["razon_real_vs_preliminar"] > 1).mean() * 100), 1),
        "anomalias": int(an.sum()),
        "anios_anomalias": sorted(df.loc[an, "anio_inicio"].unique().tolist()),
        "prelim_max_en_anomalias": round(float(df.loc[an, "costo_prelim_m2"].max()), 2),
        "valores_distintos_prelim_bajo_1": int(df.loc[df.costo_prelim_m2 < 1, "costo_prelim_m2"].nunique())}


def costo_por_localidad(df):
    _estilo()
    orden = df.groupby("localidad")[dp.OBJETIVO].median().sort_values().index
    fig, ax = plt.subplots(figsize=(10, 3.8))
    datos = [df.loc[df.localidad == l, dp.OBJETIVO] for l in orden]
    bp = ax.boxplot(datos, patch_artist=True, widths=0.6, medianprops={"color": TINTA}, flierprops={"markersize": 3})
    for c in bp["boxes"]:
        c.set_facecolor(AZUL)
        c.set_alpha(0.75)
        c.set_edgecolor("white")
    ax.set_xticks(range(1, len(orden) + 1), [str(l) for l in orden])
    ax.set(xlabel="Localidad (código agrupado, ordenado por mediana)", ylabel="Costo real (MIRR/m²)",
           title="Costo real por m² según localidad")
    ax.grid(axis="y", color=REJ, linewidth=0.7)
    ax.set_axisbelow(True)
    n = df.localidad.value_counts()
    return _guardar("eda_04_costo_por_localidad.png"), {
        "localidades": int(df.localidad.nunique()), "min_proyectos_localidad": int(n.min()),
        "max_proyectos_localidad": int(n.max())}


def redundancia_rezagos(df):
    """Matriz de correlación de los 19 indicadores económicos en el rezago 1 + objetivo."""
    _estilo()
    cols = [c for c in df.columns if c.endswith("_r1")]
    sub = df[cols + [dp.OBJETIVO]].copy()
    sub.columns = [c.replace("_r1", "") for c in cols] + ["COSTO REAL m²"]
    c = sub.corr()
    fig, ax = plt.subplots(figsize=(9, 8.6))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(c)), c.columns, fontsize=7.5)
    ax.set_title("Correlación entre indicadores económicos (rezago 1) y el costo real")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    X, _ = dp.seleccionar_variables(df)
    m = X.corr().abs().to_numpy()
    sup = m[np.triu_indices_from(m, 1)]
    mismo_ind = [df[f"{b}_r1"].corr(df[f"{b}_r5"]) for b in {re_c[:-3] for re_c in cols}]
    return _guardar("eda_05_correlacion_indicadores.png"), {
        "pares_predictoras": int(len(sup)), "pares_r_mayor_0_99": int((sup >= 0.99).sum()),
        "pares_r_mayor_0_9": int((sup >= 0.9).sum()),
        "corr_mediana_r1_vs_r5_mismo_indicador": round(float(np.median(mismo_ind)), 3)}


def generar_todas(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = pd.read_csv(dp.CSV_LIMPIO)
    salida = {}
    for f in [distribucion_objetivo, tendencia_temporal, preliminar_vs_real, costo_por_localidad, redundancia_rezagos]:
        ruta, datos = f(df)
        salida[ruta.name] = datos
    return salida


if __name__ == "__main__":
    import json
    print(json.dumps(generar_todas(), ensure_ascii=False, indent=2, default=str))
