# -*- coding: utf-8 -*-
"""
datos_prep.py — Funciones de obtención, limpieza, estandarización y versionado
==============================================================================
Proyecto: VOREKO IA · Actividad 5 (Gestión de Proyectos de IA, Tecmilenio)
Dataset público: "Residential Building Data Set" (UCI Machine Learning
Repository, id 437, licencia CC BY 4.0, DOI 10.24432/C5S896).

Relación con VOREKO: el módulo de alertas de VOREKO busca detectar
desviaciones de costo. Este dataset contiene, para 372 edificios
residenciales, el costo preliminar estimado al inicio del proyecto y el costo
real de construcción al terminarlo, además de 19 indicadores económicos en 5
rezagos. Sirve como dataset público de referencia mientras VOREKO acumula
datos propios.

Uso desde terminal:
    python fuentes/datos_prep.py              # descarga (si falta), limpia y versiona
    python fuentes/datos_prep.py --forzar     # vuelve a descargar el original
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# Rutas y fuentes
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]
DIR_INI = RAIZ / "datos" / "datos_ini"
DIR_LIMP = RAIZ / "datos" / "datos_limp"
ARCHIVO_ORIGINAL = DIR_INI / "Residential-Building-Data-Set.xlsx"
CSV_ORIGINAL = DIR_INI / "residential_building_original.csv"
CSV_LIMPIO = DIR_LIMP / "residential_building_limpio.csv"
DICCIONARIO = DIR_LIMP / "diccionario_datos.csv"
REPORTE_CALIDAD = DIR_LIMP / "reporte_calidad.md"
MANIFIESTO = DIR_LIMP / "manifiesto_versiones.json"

VERSION_DATOS = "1.0.0"

# Fuente oficial y espejos públicos (verificados: contenido celda por celda idéntico)
URL_UCI_ZIP = "https://archive.ics.uci.edu/static/public/437/residential+building+data+set.zip"
URL_ESPEJOS = [
    "https://raw.githubusercontent.com/rita316/TehranRealEstateProfitability/master/Residential-Building-Data-Set.xlsx",
    "https://raw.githubusercontent.com/trisa-biswas/Residential-Building-Price-Prediction/master/data/Residential-Building-Data-Set.xlsx",
]

# ---------------------------------------------------------------------------
# Diccionario de variables (descripciones del archivo original, traducidas)
# unidad: "10k" = 10,000 riales; "10M" = 10,000,000 riales
# ---------------------------------------------------------------------------
VARIABLES = {
    "V-1": ("localidad", "Localidad del proyecto (código postal agrupado 1–20)", "N/A", "categórica"),
    "V-2": ("area_construida_m2", "Área total construida del edificio", "m2", "m2"),
    "V-3": ("area_terreno_m2", "Área del terreno", "m2", "m2"),
    "V-4": ("costo_prelim_total", "Costo preliminar total estimado con precios al inicio", "10M IRR", "MIRR"),
    "V-5": ("costo_prelim_m2", "Costo preliminar estimado por m2 con precios al inicio", "10k IRR", "MIRR/m2"),
    "V-6": ("costo_prelim_m2_base", "Costo preliminar equivalente por m2 en el año base (1383)", "10k IRR", "MIRR/m2"),
    "V-7": ("duracion_trimestres", "Duración de la construcción", "trimestres", "trimestres"),
    "V-8": ("precio_venta_inicial_m2", "Precio de la unidad por m2 al inicio del proyecto", "10k IRR", "MIRR/m2"),
    "V-9": ("precio_venta_real_m2", "Precio real de venta (salida; se excluye por fuga de información)", "10k IRR", "MIRR/m2"),
    "V-10": ("costo_real_m2", "Costo real de construcción por m2 (VARIABLE OBJETIVO)", "10k IRR", "MIRR/m2"),
    "V-11": ("permisos_emitidos", "Número de permisos de construcción emitidos", "N/A", "N/A"),
    "V-12": ("indice_servicios_construccion", "Índice de servicios de construcción (BSI), año base", "N/A", "N/A"),
    "V-13": ("indice_precios_materiales", "Índice de precios al mayoreo de materiales (WPI), año base", "N/A", "N/A"),
    "V-14": ("area_permisos_m2", "Área total de los permisos emitidos por el municipio", "m2", "m2"),
    "V-15": ("liquidez_acumulada", "Liquidez acumulada", "10M IRR", "MIRR"),
    "V-16": ("inversion_privada_edificacion", "Inversión del sector privado en edificios nuevos", "10M IRR", "MIRR"),
    "V-17": ("indice_precio_suelo", "Índice de precio del suelo, año base", "10M IRR", "MIRR"),
    "V-18": ("num_creditos", "Número de créditos otorgados por bancos en el trimestre", "N/A", "N/A"),
    "V-19": ("monto_creditos", "Monto de créditos otorgados por bancos en el trimestre", "10M IRR", "MIRR"),
    "V-20": ("tasa_interes", "Tasa de interés de los créditos", "%", "%"),
    "V-21": ("costo_promedio_fin_m2", "Costo promedio de construcción privada al terminar", "10k IRR/m2", "MIRR/m2"),
    "V-22": ("costo_promedio_inicio_m2", "Costo promedio de construcción privada al iniciar", "10k IRR/m2", "MIRR/m2"),
    "V-23": ("tipo_cambio_oficial", "Tipo de cambio oficial respecto al dólar", "IRR", "MIRR"),
    "V-24": ("tipo_cambio_mercado", "Tipo de cambio no oficial (mercado) respecto al dólar", "IRR", "MIRR"),
    "V-25": ("inpc", "Índice de precios al consumidor (CPI), año base", "N/A", "N/A"),
    "V-26": ("inpc_vivienda", "CPI de vivienda, agua, combustible y energía, año base", "N/A", "N/A"),
    "V-27": ("indice_bolsa", "Índice del mercado de valores", "N/A", "N/A"),
    "V-28": ("poblacion_ciudad", "Población de la ciudad", "N/A", "N/A"),
    "V-29": ("precio_oro_onza", "Precio del oro por onza", "IRR", "MIRR"),
}
# Factor para llevar cada unidad original a millones de riales (MIRR)
FACTOR_MIRR = {"10k IRR": 0.01, "10k IRR/m2": 0.01, "10M IRR": 10.0, "IRR": 1e-6}

OBJETIVO = "costo_real_m2"
# Variables excluidas del modelo y razón (reglas de negocio / fuga de información)
EXCLUIDAS = {
    "precio_venta_real_m2": "Salida del proyecto: sólo se conoce al vender (fuga de información)",
    "anio_fin": "Fecha de término: posterior al momento de estimar (fuga de información)",
    "trimestre_fin": "Fecha de término: posterior al momento de estimar (fuga de información)",
    "costo_prelim_total": "Redundante: costo_prelim_m2 × area_construida_m2 (identidad verificada)",
    "indice_tiempo_fin": "Derivada de la fecha de término (fuga de información)",
}


# ---------------------------------------------------------------------------
# Utilidades de integridad
# ---------------------------------------------------------------------------
def sha256_archivo(ruta: str | Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def sha256_dataframe(df: pd.DataFrame) -> str:
    """Huella del contenido (independiente del formato del archivo)."""
    return hashlib.sha256(df.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 1. Obtención del dato original
# ---------------------------------------------------------------------------
def obtener_datos(destino: Path = ARCHIVO_ORIGINAL, forzar: bool = False) -> Path:
    """
    Usa el archivo local si existe; si no, lo descarga de UCI y, si UCI no está
    disponible, de los espejos públicos de GitHub. Nunca modifica el original.
    """
    destino = Path(destino)
    if destino.exists() and not forzar:
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    errores = []
    try:
        with urllib.request.urlopen(URL_UCI_ZIP, timeout=30) as r:
            with zipfile.ZipFile(io.BytesIO(r.read())) as z:
                nombre = next(n for n in z.namelist() if n.endswith(".xlsx"))
                destino.write_bytes(z.read(nombre))
                return destino
    except Exception as e:  # red bloqueada, sitio caído, etc.
        errores.append(f"UCI: {type(e).__name__}")
    for url in URL_ESPEJOS:
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                destino.write_bytes(r.read())
                return destino
        except Exception as e:
            errores.append(f"{url}: {type(e).__name__}")
    raise RuntimeError("No se pudo obtener el dataset: " + "; ".join(errores))


def leer_original(ruta: Path = ARCHIVO_ORIGINAL) -> pd.DataFrame:
    """Lee la hoja 'Data' tal cual (sin encabezado), como cuadrícula de texto."""
    return pd.read_excel(ruta, sheet_name="Data", header=None, dtype=object)


def leer_descripciones(ruta: Path = ARCHIVO_ORIGINAL) -> pd.DataFrame:
    return pd.read_excel(ruta, sheet_name="Descriptions", dtype=object)


# ---------------------------------------------------------------------------
# 2. Limpieza y estandarización
# ---------------------------------------------------------------------------
def aplanar_encabezados(crudo: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    El original tiene dos filas de encabezado: la fila 0 agrupa (fechas,
    variables del proyecto, rezagos 1–5, salidas) y la fila 1 trae el código
    de variable, repetido en cada rezago (V-11 … V-29 aparecen 5 veces).
    Se construyen nombres únicos y legibles: p. ej. 'permisos_emitidos_r3'.
    """
    grupos = crudo.iloc[0].ffill()
    codigos = crudo.iloc[1].astype(str).str.strip()
    nombres = []
    for grupo, codigo in zip(grupos, codigos):
        rezago = re.search(r"TIME LAG (\d)", str(grupo))
        if codigo in VARIABLES:
            base = VARIABLES[codigo][0]
            nombres.append(f"{base}_r{rezago.group(1)}" if rezago else base)
        else:
            nombres.append({"START YEAR": "anio_inicio", "START QUARTER": "trimestre_inicio",
                            "COMPLETION YEAR": "anio_fin", "COMPLETION QUARTER": "trimestre_fin"}[codigo])
    df = crudo.iloc[2:].reset_index(drop=True).copy()
    df.columns = nombres
    info = {
        "filas_encabezado": 2,
        "columnas": len(nombres),
        "codigos_repetidos_en_original": int(pd.Series(codigos).duplicated().sum()),
        "nombres_unicos_tras_aplanar": bool(pd.Series(nombres).is_unique),
    }
    return df, info


def convertir_tipos(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Todo llega como texto (object) por las filas de encabezado: se convierte a numérico."""
    antes = df.dtypes.astype(str).value_counts().to_dict()
    convertido = df.apply(pd.to_numeric, errors="coerce")
    no_convertibles = int(convertido.isna().sum().sum() - df.isna().sum().sum())
    enteros = ["anio_inicio", "trimestre_inicio", "anio_fin", "trimestre_fin", "localidad", "duracion_trimestres"]
    for c in enteros:
        if convertido[c].notna().all() and (convertido[c] % 1 == 0).all():
            convertido[c] = convertido[c].astype("int64")
    return convertido, {"tipos_antes": antes, "tipos_despues": convertido.dtypes.astype(str).value_counts().to_dict(),
                        "valores_no_convertibles": no_convertibles}


def tratar_nulos(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Regla: se elimina el registro si falta el objetivo; en variables
    explicativas se imputa la mediana (se registra cuántas celdas se tocaron).
    """
    nulos_por_col = df.isna().sum()
    sin_objetivo = int(df[OBJETIVO].isna().sum())
    df = df[df[OBJETIVO].notna()].copy()
    imputadas = 0
    for c in df.columns:
        n = int(df[c].isna().sum())
        if n:
            df[c] = df[c].fillna(df[c].median())
            imputadas += n
    return df, {"nulos_totales": int(nulos_por_col.sum()), "filas_sin_objetivo_eliminadas": sin_objetivo,
                "celdas_imputadas_mediana": imputadas}


CLAVE_PROYECTO = ["localidad", "area_construida_m2", "area_terreno_m2", "costo_prelim_m2",
                  "duracion_trimestres", "precio_venta_inicial_m2", "anio_inicio", "trimestre_inicio"]


def deduplicar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Criterios, en este orden:
      1. Filas idénticas en todas las columnas → se conserva la primera.
      2. Mismo proyecto (clave física + financiera + fecha de inicio) → se
         conserva el registro con costo real no nulo y, si empata, el primero.
      3. Casi duplicados: misma clave salvo diferencias < 0.5 % en áreas y
         costos (posibles capturas dobles con redondeo) → se reportan.
      4. Columnas duplicadas (mismos valores en todas las filas) y columnas
         constantes → se eliminan del dataset limpio (no aportan información).
    """
    n0 = len(df)
    exactas = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    por_clave = int(df.duplicated(subset=CLAVE_PROYECTO).sum())
    df = df.sort_values(OBJETIVO, na_position="last").drop_duplicates(subset=CLAVE_PROYECTO, keep="first")
    df = df.sort_index()

    redondeado = df[CLAVE_PROYECTO].copy()
    for c in ["area_construida_m2", "area_terreno_m2", "costo_prelim_m2", "precio_venta_inicial_m2"]:
        redondeado[c] = np.round(np.log(redondeado[c].clip(lower=1e-9)) / 0.005)
    casi = int(redondeado.duplicated(keep=False).sum())

    transpuesta = df.T
    col_dup = transpuesta.index[transpuesta.duplicated()].tolist()
    df = df.drop(columns=col_dup)
    constantes = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=constantes)
    return df.reset_index(drop=True), {
        "filas_iniciales": n0, "duplicados_exactos": exactas, "duplicados_por_clave_proyecto": por_clave,
        "registros_en_grupos_casi_duplicados": casi, "columnas_duplicadas_eliminadas": col_dup,
        "columnas_constantes_eliminadas": constantes, "filas_finales": len(df),
    }


def validar_reglas_negocio(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reglas de dominio de obra: cada violación se cuenta y el registro se elimina."""
    reglas = {
        "trimestres_en_1_4": df["trimestre_inicio"].between(1, 4) & df["trimestre_fin"].between(1, 4),
        "localidad_en_1_20": df["localidad"].between(1, 20),
        "areas_y_costos_positivos": (df[["area_construida_m2", "area_terreno_m2", "costo_prelim_m2", OBJETIVO]] > 0).all(axis=1),
        "area_construida_mayor_o_igual_terreno": df["area_construida_m2"] >= df["area_terreno_m2"],
        "costo_total_igual_m2_por_area": np.isclose(df["costo_prelim_total"],
                                                     df["costo_prelim_m2"] * df["area_construida_m2"] / 1000, rtol=0.01),
        "fin_igual_inicio_mas_duracion": ((df["anio_fin"] * 4 + df["trimestre_fin"])
                                          - (df["anio_inicio"] * 4 + df["trimestre_inicio"])) == df["duracion_trimestres"],
        "duracion_positiva": df["duracion_trimestres"] > 0,
    }
    tabla = {k: int((~v).sum()) for k, v in reglas.items()}
    valido = np.logical_and.reduce(list(reglas.values()))
    return df[valido].reset_index(drop=True), {"violaciones_por_regla": tabla, "filas_eliminadas": int((~valido).sum())}


def estandarizar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    - Años del calendario persa en 2 dígitos (81) → 4 dígitos (1381) y año
      gregoriano aproximado (+621); índice temporal continuo en trimestres.
    - Montos en unidades mezcladas (10 mil, 10 millones o riales) → millones
      de riales (MIRR) en todas las columnas monetarias.
    """
    df = df.copy()
    for pref in ["inicio", "fin"]:
        df[f"anio_{pref}"] = df[f"anio_{pref}"] + 1300
        df[f"anio_{pref}_greg_aprox"] = df[f"anio_{pref}"] + 621
        df[f"indice_tiempo_{pref}"] = (df[f"anio_{pref}"] - 1372) * 4 + df[f"trimestre_{pref}"]
    convertidas = []
    for codigo, (base, _, unidad, _) in VARIABLES.items():
        if unidad in FACTOR_MIRR:
            cols = [c for c in df.columns if c == base or re.fullmatch(rf"{base}_r\d", c)]
            for c in cols:
                df[c] = df[c] * FACTOR_MIRR[unidad]
            convertidas += cols
    return df, {"columnas_monetarias_a_MIRR": len(convertidas),
                "rango_anios_persas": [int(df["anio_inicio"].min()), int(df["anio_fin"].max())],
                "rango_anios_gregorianos_aprox": [int(df["anio_inicio_greg_aprox"].min()), int(df["anio_fin_greg_aprox"].max())]}


def marcar_anomalias(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Anomalías (no se eliminan; se marcan y se documentan):
      - Costo real por m2 fuera de [Q1 − 3·IQR, Q3 + 3·IQR] en escala logarítmica.
      - Razón costo real / costo preliminar fuera del mismo criterio.
    Criterio: son proyectos reales del dataset oficial, no errores de captura;
    eliminarlos sesgaría el modelo hacia proyectos "normales" y ocultaría los
    sobrecostos que VOREKO necesita detectar.
    """
    df = df.copy()
    df["razon_real_vs_preliminar"] = df[OBJETIVO] / df["costo_prelim_m2"]

    def fuera(serie):
        s = np.log(serie)
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        return (s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)

    df["anomalia_costo_m2"] = fuera(df[OBJETIVO])
    df["anomalia_razon_costo"] = fuera(df["razon_real_vs_preliminar"])
    return df, {"anomalias_costo_m2": int(df["anomalia_costo_m2"].sum()),
                "anomalias_razon_costo": int(df["anomalia_razon_costo"].sum()),
                "razon_mediana": round(float(df["razon_real_vs_preliminar"].median()), 3)}


def limpiar(crudo: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pipeline completo de limpieza. Devuelve el dataset limpio y el reporte paso a paso."""
    reporte = {}
    df, reporte["1_encabezados"] = aplanar_encabezados(crudo)
    df, reporte["2_tipos"] = convertir_tipos(df)
    df, reporte["3_nulos"] = tratar_nulos(df)
    df, reporte["4_duplicados"] = deduplicar(df)
    df, reporte["5_reglas_negocio"] = validar_reglas_negocio(df)
    df, reporte["6_estandarizacion"] = estandarizar(df)
    df, reporte["7_anomalias"] = marcar_anomalias(df)
    df.insert(0, "id_proyecto", [f"RB-{i:03d}" for i in range(1, len(df) + 1)])
    return df, reporte


# ---------------------------------------------------------------------------
# 3. Selección de variables para el modelo
# ---------------------------------------------------------------------------
NO_PREDICTORAS = {"id_proyecto", OBJETIVO, "razon_real_vs_preliminar", "anomalia_costo_m2",
                  "anomalia_razon_costo", "anio_inicio_greg_aprox", "anio_fin_greg_aprox"} | set(EXCLUIDAS)
CATEGORICAS = ["localidad", "trimestre_inicio"]


def seleccionar_variables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X con la información disponible AL INICIO del proyecto; y = costo real por m2."""
    X = df[[c for c in df.columns if c not in NO_PREDICTORAS]].copy()
    return X, df[OBJETIVO].copy()


class SelectorRezagos(BaseEstimator, TransformerMixin):
    """
    Hiperparámetro de ingeniería de variables. Los 19 indicadores económicos
    vienen en 5 rezagos muy correlacionados entre sí (183 pares con |r| ≥ 0.99).
    'todos' conserva los 5 rezagos; 'r1' conserva sólo el trimestre más cercano
    al inicio, reduciendo de 105 a 29 predictoras.
    """

    def __init__(self, rezagos: str = "todos"):
        self.rezagos = rezagos

    def fit(self, X, y=None):
        self.columnas_ = [c for c in X.columns if self.rezagos == "todos" or not re.search(r"_r[2-5]$", c)]
        return self

    def transform(self, X):
        return X[self.columnas_]


def columnas_numericas(X):
    return [c for c in X.columns if c not in CATEGORICAS]


def columnas_categoricas(X):
    return [c for c in X.columns if c in CATEGORICAS]


# ---------------------------------------------------------------------------
# 4. Diccionario, reporte de calidad y versionado
# ---------------------------------------------------------------------------
def construir_diccionario(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    inverso = {v[0]: (k, v) for k, v in VARIABLES.items()}
    for c in df.columns:
        base = re.sub(r"_r\d$", "", c)
        rezago = re.search(r"_r(\d)$", c)
        if base in inverso:
            codigo, (_, desc, u_orig, u_limpia) = inverso[base]
        else:
            codigo, desc, u_orig, u_limpia = "derivada", {
                "id_proyecto": "Identificador asignado en la limpieza",
                "anio_inicio": "Año de inicio (calendario persa, 4 dígitos)",
                "trimestre_inicio": "Trimestre de inicio (persa; 1 = primavera)",
                "anio_fin": "Año de término (calendario persa, 4 dígitos)",
                "trimestre_fin": "Trimestre de término (persa)",
                "anio_inicio_greg_aprox": "Año de inicio gregoriano aproximado (+621)",
                "anio_fin_greg_aprox": "Año de término gregoriano aproximado (+621)",
                "indice_tiempo_inicio": "Trimestres transcurridos desde 1372 T0 al inicio",
                "indice_tiempo_fin": "Trimestres transcurridos desde 1372 T0 al término",
                "razon_real_vs_preliminar": "Costo real / costo preliminar por m2 (desviación de costo)",
                "anomalia_costo_m2": "Bandera: costo real por m2 atípico (3·IQR en log)",
                "anomalia_razon_costo": "Bandera: desviación de costo atípica (3·IQR en log)",
            }.get(c, ""), "", ""
        if c == OBJETIVO:
            rol = "objetivo"
        elif c in EXCLUIDAS:
            rol = "excluida: " + EXCLUIDAS[c]
        elif c in NO_PREDICTORAS:
            rol = "auxiliar (no predictora)"
        else:
            rol = "predictora"
        filas.append({"columna": c, "codigo_original": codigo, "rezago": int(rezago.group(1)) if rezago else "",
                      "descripcion": desc, "unidad_original": u_orig, "unidad_limpia": u_limpia,
                      "tipo": str(df[c].dtype), "rol": rol})
    return pd.DataFrame(filas)


def reporte_markdown(reporte: dict, huellas: dict, n_pred: int) -> str:
    r = reporte
    l = ["# Reporte de calidad de datos — Residential Building (UCI 437)", "",
         f"Versión de datos {VERSION_DATOS} · generado por `fuentes/datos_prep.py` (salida determinista)", "",
         "| Paso | Hallazgo | Acción |", "|---|---|---|",
         f"| 1. Encabezados | 2 filas de encabezado; {r['1_encabezados']['codigos_repetidos_en_original']} códigos repetidos por rezagos | Aplanado a {r['1_encabezados']['columnas']} nombres únicos (`*_r1…r5`) |",
         f"| 2. Tipos | Todas las columnas como texto: {r['2_tipos']['tipos_antes']} | Conversión numérica: {r['2_tipos']['tipos_despues']}; no convertibles: {r['2_tipos']['valores_no_convertibles']} |",
         f"| 3. Nulos | {r['3_nulos']['nulos_totales']} valores nulos | Filas sin objetivo eliminadas: {r['3_nulos']['filas_sin_objetivo_eliminadas']}; celdas imputadas: {r['3_nulos']['celdas_imputadas_mediana']} |",
         f"| 4. Duplicados | Exactos: {r['4_duplicados']['duplicados_exactos']}; por clave de proyecto: {r['4_duplicados']['duplicados_por_clave_proyecto']}; en grupos casi duplicados: {r['4_duplicados']['registros_en_grupos_casi_duplicados']} | Columnas duplicadas eliminadas: {len(r['4_duplicados']['columnas_duplicadas_eliminadas'])}; constantes: {len(r['4_duplicados']['columnas_constantes_eliminadas'])} |",
         f"| 5. Reglas de negocio | Violaciones: {r['5_reglas_negocio']['violaciones_por_regla']} | Filas eliminadas: {r['5_reglas_negocio']['filas_eliminadas']} |",
         f"| 6. Estandarización | Años persas de 2 dígitos; montos en 10 mil, 10 millones o riales | Años a 4 dígitos ({r['6_estandarizacion']['rango_anios_persas']}) y gregoriano aprox. ({r['6_estandarizacion']['rango_anios_gregorianos_aprox']}); {r['6_estandarizacion']['columnas_monetarias_a_MIRR']} columnas a millones de riales |",
         f"| 7. Anomalías | Costo m2 atípico: {r['7_anomalias']['anomalias_costo_m2']}; desviación atípica: {r['7_anomalias']['anomalias_razon_costo']} (mediana de razón {r['7_anomalias']['razon_mediana']}) | Se marcan, no se eliminan |",
         "", f"Columnas duplicadas eliminadas: {', '.join(r['4_duplicados']['columnas_duplicadas_eliminadas']) or 'ninguna'}",
         f"Columnas constantes eliminadas: {', '.join(r['4_duplicados']['columnas_constantes_eliminadas']) or 'ninguna'}",
         "", "## Variables excluidas del modelo", ""]
    l += [f"- `{k}`: {v}" for k, v in EXCLUIDAS.items()]
    l += ["", f"Predictoras disponibles al inicio del proyecto: **{n_pred}**.", "", "## Huellas SHA-256", ""]
    l += [f"- {k}: `{v}`" for k, v in huellas.items()]
    return "\n".join(l) + "\n"


def ejecutar(forzar: bool = False) -> dict:
    ruta = obtener_datos(forzar=forzar)
    crudo = leer_original(ruta)
    DIR_LIMP.mkdir(parents=True, exist_ok=True)
    crudo.to_csv(CSV_ORIGINAL, index=False, header=False, encoding="utf-8")  # copia fiel en texto plano
    limpio, reporte = limpiar(crudo)
    limpio.to_csv(CSV_LIMPIO, index=False, encoding="utf-8")
    dic = construir_diccionario(limpio)
    dic.to_csv(DICCIONARIO, index=False, encoding="utf-8")
    X, _ = seleccionar_variables(limpio)
    huellas = {
        "archivo_original_xlsx": sha256_archivo(ruta),
        "contenido_original_hoja_Data": sha256_dataframe(crudo),
        "csv_limpio": sha256_archivo(CSV_LIMPIO),
    }
    REPORTE_CALIDAD.write_text(reporte_markdown(reporte, huellas, X.shape[1]), encoding="utf-8")
    manifiesto = {
        "version_datos": VERSION_DATOS,
        "fuente": "UCI Machine Learning Repository, Residential Building Data Set (id 437), CC BY 4.0, doi:10.24432/C5S896",
        "obtenido_de": [URL_UCI_ZIP] + URL_ESPEJOS,
        "filas_original": int(len(crudo) - 2), "columnas_original": int(crudo.shape[1]),
        "filas_limpio": int(len(limpio)), "columnas_limpio": int(limpio.shape[1]),
        "predictoras": int(X.shape[1]), "objetivo": OBJETIVO,
        "huellas_sha256": huellas, "reporte": json.loads(json.dumps(reporte, default=str)),
    }
    MANIFIESTO.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifiesto


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Limpieza y versionado del dataset Residential Building")
    ap.add_argument("--forzar", action="store_true", help="volver a descargar el original")
    m = ejecutar(forzar=ap.parse_args().forzar)
    print(json.dumps({k: m[k] for k in ["version_datos", "filas_original", "filas_limpio", "columnas_limpio",
                                         "predictoras", "huellas_sha256"]}, ensure_ascii=False, indent=2))
