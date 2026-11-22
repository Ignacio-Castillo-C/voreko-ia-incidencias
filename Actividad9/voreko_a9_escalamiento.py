# Actividad 9 · VOREKO IA · Escalamiento eficiente, optimización de costos y gobernanza responsable
# José Ignacio Castillo Camacho · AL07161745


# 1. Instalación de dependencias
import os
import subprocess
import sys

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
PAQUETES = ["scikit-learn==1.9.1", "xgboost==3.4.1", "mlflow-skinny==3.17.2", "pandas==2.2.3", "numpy==2.1.3", "scipy==1.16.3",
            "joblib==1.4.2", "fastapi==0.115.12", "uvicorn==0.34.2", "pydantic==2.11.4", "httpx==0.28.1", "skl2onnx==1.20.0",
            "onnx==1.23.2", "onnxruntime==1.29.0", "psutil", "matplotlib"]
import importlib.metadata as metadatos


def instalado(paquete):
    nombre, _, version = paquete.partition("==")
    try:
        return not version or metadatos.version(nombre) == version
    except metadatos.PackageNotFoundError:
        return False


FALTANTES = [p for p in PAQUETES if not instalado(p)]
if FALTANTES and os.getenv("VOREKO_SIN_INSTALAR") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PAQUETES], check=True)
    print("Dependencias instaladas. El entorno se reinicia; vuelve a ejecutar todas las celdas.")
    if "google.colab" in sys.modules or "ipykernel" in sys.modules:
        os.kill(os.getpid(), 9)
    sys.exit(0)
print("Dependencias listas:", ", ".join(f"{p.partition('==')[0]} {metadatos.version(p.partition('==')[0])}" for p in PAQUETES[:6]))


# 2. Repositorio, modelo de la Fase II y datos
import asyncio
import copy
import hashlib
import json
import random
import shutil
import socket
import time
import tracemalloc
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
BASE = Path.cwd()
REPO = Path(os.getenv("VOREKO_REPO", BASE / "voreko-ia-incidencias"))
if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/Ignacio-Castillo-C/voreko-ia-incidencias.git", str(REPO)], check=True)

SALIDA = BASE / "actividad9_salidas"
if SALIDA.exists():
    shutil.rmtree(SALIDA)
for carpeta in ["modelo/actual", "modelo/optimizado", "tablas", "figuras", "logs", "entornos"]:
    (SALIDA / carpeta).mkdir(parents=True, exist_ok=True)
DIR_V1, DIR_OPT, TAB, FIG = SALIDA / "modelo" / "actual", SALIDA / "modelo" / "optimizado", SALIDA / "tablas", SALIDA / "figuras"

entrenamiento = subprocess.run([sys.executable, str(REPO / "FaseII" / "modelo" / "entrenar_modelo.py"), "--salida", str(DIR_V1)],
                               capture_output=True, text=True, cwd=REPO)
print("\n".join(l for l in entrenamiento.stdout.splitlines() if l.startswith("OK")))
assert entrenamiento.returncode == 0, entrenamiento.stderr[-2000:]

import numpy as np
import pandas as pd
import joblib
import psutil
import httpx
import onnx
import onnxruntime as ort
import sklearn
from skl2onnx import to_onnx
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, roc_auc_score

sys.path.insert(0, str(REPO / "Actividad6" / "fuentes"))
import voreko_a6 as v

SEMILLA = 42
rng = np.random.default_rng(SEMILLA)
random.seed(SEMILLA)
META_V1 = json.loads((DIR_V1 / "metadatos_modelo.json").read_text(encoding="utf-8"))
COLUMNAS = [c["nombre"] for c in META_V1["variables"]]
CATEGORICAS = META_V1["variables_categoricas"]
U_ROJO, U_AMARILLO = META_V1["umbrales"]["rojo"], META_V1["umbrales"]["amarillo"]
TIPICO = {c["nombre"]: c["tipico"] for c in META_V1["variables"]}

df, _ = v.cargar_datos_a5()
X_todo = v.variables_inicio(df)[COLUMNAS].reset_index(drop=True)
y_todo = v.construir_etiqueta(df).reset_index(drop=True)
idx_tr, idx_te = v.particion_a5(v.variables_inicio(df), df["costo_real_m2"])
pos = {k: i for i, k in enumerate(df.index)}
i_tr, i_te = np.array([pos[k] for k in idx_tr]), np.array([pos[k] for k in idx_te])
X_tr, X_te, y_tr, y_te = X_todo.iloc[i_tr].reset_index(drop=True), X_todo.iloc[i_te].reset_index(drop=True), y_todo.iloc[i_tr].reset_index(drop=True), y_todo.iloc[i_te].reset_index(drop=True)

modelo_v1 = joblib.load(DIR_V1 / "modelo_riesgo.joblib")
modelo_v1.named_steps["modelo"].set_params(n_jobs=1)
p_v1_te = modelo_v1.predict_proba(X_te)[:, 1]
p_v1_todo = modelo_v1.predict_proba(X_todo)[:, 1]
print("Obras:", len(X_todo), "| entrenamiento:", len(X_tr), "| prueba:", len(X_te), "| variables:", len(COLUMNAS))
print("Fase II reproducida:", {"recall": round(recall_score(y_te, p_v1_te >= U_ROJO), 3), "precision": round(precision_score(y_te, p_v1_te >= U_ROJO), 3),
                               "roc_auc": round(roc_auc_score(y_te, p_v1_te), 3)})


# 3. Inventario de la arquitectura actual
render = (REPO / "render.yaml").read_text(encoding="utf-8")
dockerfile = (REPO / "FaseII" / "Dockerfile").read_text(encoding="utf-8")
rf_v1 = modelo_v1.named_steps["modelo"]
nodos_v1 = int(sum(e.tree_.node_count for e in rf_v1.estimators_))


def tam_paquetes(nombres):
    total = 0
    for n in nombres:
        try:
            import importlib
            mod = importlib.import_module(n)
            ruta = Path(mod.__file__).parent
            total += sum(f.stat().st_size for f in ruta.rglob("*") if f.is_file())
        except Exception:
            pass
    return total / 1e6


INVENTARIO = pd.DataFrame([
    ("Plataforma", "Render (PaaS), plan " + render.split("plan:")[1].split()[0], "render.yaml"),
    ("Región", render.split("region:")[1].split()[0], "render.yaml"),
    ("Instancias", "1 contenedor, sin réplicas ni autoescalado", "render.yaml"),
    ("Servidor", "uvicorn con 1 worker (proceso único)", "FaseII/Dockerfile (CMD)"),
    ("Patrón", "Monolito: API + frontend + modelo en el mismo contenedor", "FaseII/Dockerfile"),
    ("Modelo", f"Random Forest scikit-learn, {rf_v1.n_estimators} árboles, {nodos_v1:,} nodos, float64", "metadatos_modelo.json"),
    ("Filas por consulta con factores", f"{1 + len(COLUMNAS)} (1 obra + {len(COLUMNAS)} variaciones)", "FaseII/api/servicio_modelo.py"),
    ("Tamaño del modelo", f"{(DIR_V1 / 'modelo_riesgo.joblib').stat().st_size / 1e6:.2f} MB (joblib comprimido)", "modelo_riesgo.joblib"),
    ("Dependencias de inferencia", f"{tam_paquetes(['sklearn', 'scipy', 'pandas', 'numpy', 'joblib', 'fastapi', 'starlette', 'pydantic', 'pydantic_core', 'uvicorn']):.0f} MB instalados (scikit-learn, SciPy, pandas, NumPy, FastAPI)", "requirements-api.txt"),
    ("Entrenamiento", "Se reentrena dentro de la imagen en cada construcción", "FaseII/Dockerfile (etapa 1)"),
    ("CORS", "VOREKO_CORS = " + render.split("VOREKO_CORS")[1].split("value:")[1].split()[0], "render.yaml"),
    ("Autenticación", "Ninguna en /api/predecir", "FaseII/api/main.py"),
    ("Caché de resultados", "No existe", "FaseII/api/servicio_modelo.py"),
    ("Lotes", f"/api/predecir/lote con máximo {render.split('LIMITE_LOTE')[1].split('value:')[1].split()[0]} obras", "render.yaml"),
    ("Monitoreo", "Bitácora a stdout; tablero y alertas de la Actividad 8 (simulados)", "Actividad8/"),
], columns=["componente", "situacion_actual", "evidencia"])
INVENTARIO.to_csv(TAB / "01_inventario_arquitectura_actual.csv", index=False)
print(INVENTARIO.to_string(index=False))


# 4. Poda del ensamble con compuerta de calidad
def podar(modelo, k):
    m = copy.deepcopy(modelo)
    rf = m.named_steps["modelo"]
    rf.estimators_ = rf.estimators_[:k]
    rf.n_estimators = k
    return m


def metricas(y, p, u=U_ROJO):
    return {"recall": recall_score(y, p >= u), "precision": precision_score(y, p >= u, zero_division=0), "roc_auc": roc_auc_score(y, p)}


def semaforo(p):
    return np.where(p >= U_ROJO, "rojo", np.where(p >= U_AMARILLO, "amarillo", "verde"))


cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
pliegues = list(cv5.split(X_todo, y_todo))
oof_por_k = {}
modelos_pliegue = []
for tr, te in pliegues:
    m = clone(modelo_v1)
    m.named_steps["modelo"].set_params(n_jobs=-1)
    m.fit(X_todo.iloc[tr], y_todo.iloc[tr])
    m.named_steps["modelo"].set_params(n_jobs=1)
    modelos_pliegue.append((te, m))

ARBOLES = [300, 200, 150, 100, 75, 50, 25]
filas = []
REF = metricas(y_te, p_v1_te)
for k in ARBOLES:
    mk = podar(modelo_v1, k)
    p_te = mk.predict_proba(X_te)[:, 1]
    p_all = mk.predict_proba(X_todo)[:, 1]
    oof = np.zeros(len(X_todo))
    for te, m in modelos_pliegue:
        oof[te] = podar(m, k).predict_proba(X_todo.iloc[te])[:, 1]
    oof_por_k[k] = oof
    mt, mo = metricas(y_te, p_te), metricas(y_todo, oof)
    filas.append({"arboles": k, "nodos": int(sum(e.tree_.node_count for e in mk.named_steps["modelo"].estimators_)),
                  "recall_prueba": mt["recall"], "precision_prueba": mt["precision"], "auc_prueba": mt["roc_auc"],
                  "recall_oof": mo["recall"], "precision_oof": mo["precision"], "auc_oof": mo["roc_auc"],
                  "concordancia_semaforo_372": float((semaforo(p_all) == semaforo(p_v1_todo)).mean()),
                  "concordancia_rojo_prueba": float(((p_te >= U_ROJO) == (p_v1_te >= U_ROJO)).mean()),
                  "dif_max_prob": float(np.abs(p_all - p_v1_todo).max())})
PODA = pd.DataFrame(filas)
PODA["pasa_compuerta"] = ((PODA.recall_prueba >= 0.45) & (PODA.precision_prueba >= 0.40) & (PODA.auc_prueba >= REF["roc_auc"] - 0.02)
                          & (PODA.auc_oof >= PODA.auc_oof.iloc[0] - 0.02) & (PODA.concordancia_semaforo_372 >= 0.95)
                          & (PODA.concordancia_rojo_prueba >= 0.95))
K_OPT = int(PODA[PODA.pasa_compuerta].arboles.min())
PODA.to_csv(TAB / "02_poda_ensamble.csv", index=False)
print(PODA.round(3).to_string(index=False))
print("Árboles elegidos:", K_OPT)
modelo_podado = podar(modelo_v1, K_OPT)


# 5. Conversión a ONNX y verificación de equivalencia
def a_onnx(modelo, ruta, precision):
    muestra = X_tr.iloc[:1].copy()
    for c in COLUMNAS:
        muestra[c] = muestra[c].astype("int64" if c in CATEGORICAS else ("float32" if precision == 32 else "float64"))
    onx = to_onnx(modelo, muestra, options={id(modelo.named_steps["modelo"]): {"zipmap": False}}, target_opset={"": 17, "ai.onnx.ml": 3})
    ruta.write_bytes(onx.SerializeToString())
    s = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
    tipos = {i.name: (np.int64 if i.type == "tensor(int64)" else np.float64 if i.type == "tensor(double)" else np.float32) for i in s.get_inputs()}
    return s, tipos


def puntuar_onnx(sesion, tipos, X):
    return sesion.run(["probabilities"], {c: X[c].to_numpy().astype(tipos[c]).reshape(-1, 1) for c in COLUMNAS})[0][:, 1].astype(float)


variantes = {}
filas = []
for nombre, modelo, prec in [("onnx32_300", modelo_v1, 32), ("onnx64_300", modelo_v1, 64), (f"onnx32_{K_OPT}", modelo_podado, 32), (f"onnx64_{K_OPT}", modelo_podado, 64)]:
    ruta = SALIDA / "modelo" / f"{nombre}.onnx"
    s, t = a_onnx(modelo, ruta, prec)
    p_ref = modelo.predict_proba(X_todo)[:, 1]
    p = puntuar_onnx(s, t, X_todo)
    variantes[nombre] = (s, t, ruta)
    filas.append({"variante": nombre, "arboles": modelo.named_steps["modelo"].n_estimators, "precision_bits": prec,
                  "tamano_mb": ruta.stat().st_size / 1e6, "dif_max_vs_sklearn": float(np.abs(p - p_ref).max()),
                  "dif_media_vs_sklearn": float(np.abs(p - p_ref).mean()), "cambios_rojo_372": int(((p >= U_ROJO) != (p_ref >= U_ROJO)).sum()),
                  "cambios_semaforo_372": int((semaforo(p) != semaforo(p_ref)).sum()),
                  "auc_prueba": roc_auc_score(y_te, puntuar_onnx(s, t, X_te))})
EQUIV = pd.DataFrame(filas)
EQUIV.to_csv(TAB / "03_equivalencia_onnx.csv", index=False)
print(EQUIV.round(6).to_string(index=False))

fila32 = EQUIV[EQUIV.variante == f"onnx32_{K_OPT}"].iloc[0]
ELEGIDA = f"onnx32_{K_OPT}" if fila32.cambios_semaforo_372 == 0 else f"onnx64_{K_OPT}"
print("Variante desplegada:", ELEGIDA)
shutil.copy(variantes[ELEGIDA][2], DIR_OPT / "modelo_riesgo.onnx")
shutil.copy(DIR_V1 / "obras_ejemplo.json", DIR_OPT / "obras_ejemplo.json")
META_OPT = copy.deepcopy(META_V1)
META_OPT.update({"version": "1.1.0", "algoritmo": f"RandomForestClassifier podado a {K_OPT} árboles, exportado a ONNX ({'float32' if '32' in ELEGIDA else 'float64'})",
                 "archivo_modelo": "modelo_riesgo.onnx", "salida_probabilidades": "probabilities",
                 "origen": "Actividad 9 · versión optimizada del modelo de la Fase II"})
p_opt_te = puntuar_onnx(variantes[ELEGIDA][0], variantes[ELEGIDA][1], X_te)
META_OPT["metricas_prueba"] = {k: round(float(x), 4) for k, x in metricas(y_te, p_opt_te).items()}
META_OPT["huellas_sha256"] = {**META_V1["huellas_sha256"], "modelo_onnx": hashlib.sha256((DIR_OPT / "modelo_riesgo.onnx").read_bytes()).hexdigest()}
META_OPT["versiones"] = {**META_V1["versiones"], "onnx": onnx.__version__, "onnxruntime": ort.__version__}
(DIR_OPT / "metadatos_modelo.json").write_text(json.dumps(META_OPT, ensure_ascii=False, indent=1), encoding="utf-8")


# 6. Microbenchmark de inferencia
OBRAS_TE = [{c: (int(X_te.iloc[i][c]) if c in CATEGORICAS else float(X_te.iloc[i][c])) for c in COLUMNAS} for i in range(len(X_te))]


def filas_consulta(obra):
    return [obra] + [{**obra, c: TIPICO[c]} for c in COLUMNAS]


def marco(filas_):
    X = pd.DataFrame(filas_, columns=COLUMNAS)
    for c in COLUMNAS:
        X[c] = X[c].astype("int64" if c in CATEGORICAS else "float64")
    return X


def puntuar_filas(sesion, tipos, filas_):
    entrada = {c: np.asarray([f[c] for f in filas_], dtype=tipos[c]).reshape(-1, 1) for c in COLUMNAS}
    return sesion.run(["probabilities"], entrada)[0][:, 1]


def medir(funcion, repeticiones):
    funcion()
    t = []
    for _ in range(repeticiones):
        a = time.perf_counter()
        funcion()
        t.append((time.perf_counter() - a) * 1000)
    return float(np.median(t)), float(np.percentile(t, 95))


LOTE_372 = X_todo.copy()
IMPLEMENTACIONES = {
    "scikit-learn 300 árboles (actual)": ("sk", modelo_v1),
    f"scikit-learn {K_OPT} árboles": ("sk", modelo_podado),
    "ONNX float32 300 árboles": ("ox", variantes["onnx32_300"]),
    f"ONNX {'float32' if '32' in ELEGIDA else 'float64'} {K_OPT} árboles (propuesta)": ("ox", variantes[ELEGIDA]),
}
filas = []
for nombre, (tipo, obj) in IMPLEMENTACIONES.items():
    if tipo == "sk":
        f1 = lambda: obj.predict_proba(marco([OBRAS_TE[0]]))
        f29 = lambda: obj.predict_proba(marco(filas_consulta(OBRAS_TE[1])))
        f372 = lambda: obj.predict_proba(LOTE_372)
    else:
        s, t, _ = obj
        f1 = lambda: puntuar_filas(s, t, [OBRAS_TE[0]])
        f29 = lambda: puntuar_filas(s, t, filas_consulta(OBRAS_TE[1]))
        f372 = lambda: puntuar_onnx(s, t, LOTE_372)
    m1, p1 = medir(f1, 300)
    m29, p29 = medir(f29, 300)
    m372, p372 = medir(f372, 60)
    filas.append({"implementacion": nombre, "ms_1_obra": m1, "p95_ms_1_obra": p1, "ms_consulta_29_filas": m29, "p95_ms_consulta_29_filas": p29,
                  "ms_lote_372": m372, "obras_por_segundo_lote": 372 / (m372 / 1000)})
MICRO = pd.DataFrame(filas)
MICRO["aceleracion_consulta"] = MICRO.ms_consulta_29_filas.iloc[0] / MICRO.ms_consulta_29_filas
MICRO.to_csv(TAB / "04_microbenchmark_inferencia.csv", index=False)
print(MICRO.round(3).to_string(index=False))

codigo_mem = """
import sys, json, psutil, gc
proc = psutil.Process()
gc.collect(); antes = proc.memory_info().rss
tipo, ruta = sys.argv[1], sys.argv[2]
if tipo == 'sk':
    import joblib, sklearn, pandas
    m = joblib.load(ruta)
else:
    import onnxruntime as ort, numpy
    m = ort.InferenceSession(ruta, providers=['CPUExecutionProvider'])
gc.collect(); print(json.dumps({'rss_base_mb': antes / 1e6, 'rss_modelo_mb': proc.memory_info().rss / 1e6}))
"""
joblib.dump(modelo_podado, SALIDA / "modelo" / f"rf_{K_OPT}.joblib", compress=3)
filas = []
for nombre, tipo, ruta in [("scikit-learn 300 árboles (actual)", "sk", DIR_V1 / "modelo_riesgo.joblib"), (f"scikit-learn {K_OPT} árboles", "sk", SALIDA / "modelo" / f"rf_{K_OPT}.joblib"),
                           ("ONNX float32 300 árboles", "ox", variantes["onnx32_300"][2]), ("ONNX propuesta", "ox", DIR_OPT / "modelo_riesgo.onnx")]:
    r = json.loads(subprocess.run([sys.executable, "-c", codigo_mem, tipo, str(ruta)], capture_output=True, text=True).stdout)
    filas.append({"implementacion": nombre, "archivo_mb": Path(ruta).stat().st_size / 1e6, "rss_proceso_con_modelo_mb": r["rss_modelo_mb"]})
MEMORIA = pd.DataFrame(filas)
MEMORIA["dependencias_mb"] = [tam_paquetes(['sklearn', 'scipy', 'pandas', 'numpy', 'joblib', 'fastapi', 'starlette', 'pydantic', 'pydantic_core', 'uvicorn'])] * 2 + [tam_paquetes(['onnxruntime', 'numpy', 'fastapi', 'starlette', 'pydantic', 'pydantic_core', 'uvicorn', 'google.protobuf', 'flatbuffers', 'packaging'])] * 2
MEMORIA.to_csv(TAB / "05_memoria_y_tamano.csv", index=False)
print(MEMORIA.round(1).to_string(index=False))


# 7. Entorno de ejecución ligero del microservicio
import venv


def crear_entorno(nombre, requisitos):
    ruta = SALIDA / "entornos" / nombre
    venv.create(ruta, with_pip=True)
    py = ruta / "bin" / "python"
    t = time.perf_counter()
    subprocess.run([str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check", "-r", str(requisitos)], check=True)
    instalacion = time.perf_counter() - t
    sitio = next(ruta.glob("lib/python*/site-packages"))
    excluir = ("pip", "setuptools", "_distutils_hack")
    tam = sum(f.stat().st_size for f in sitio.rglob("*") if f.is_file() and not any(part.startswith(excluir) for part in f.relative_to(sitio).parts[:1]))
    return py, {"entorno": nombre, "paquetes_mb": tam / 1e6, "instalacion_s": instalacion}


PY_ACTUAL, ent_a = crear_entorno("actual", REPO / "FaseII" / "requirements-api.txt")
PY_OPT, ent_o = crear_entorno("optimizado", REPO / "Actividad9" / "api" / "requirements-api-optimizada.txt")
verificacion = subprocess.run([str(PY_OPT), "-c", "import importlib.util as u; print([m for m in ('sklearn', 'pandas', 'scipy', 'joblib') if u.find_spec(m)])"],
                              capture_output=True, text=True).stdout.strip()
ENTORNOS = pd.DataFrame([ent_a, ent_o])
ENTORNOS["imagen_estimada_mb"] = ENTORNOS.paquetes_mb + 150
ENTORNOS["librerias_pesadas_presentes"] = ["sklearn, pandas, scipy, joblib", verificacion]
ENTORNOS.to_csv(TAB / "05b_entornos_ejecucion.csv", index=False)
print(ENTORNOS.round(1).to_string(index=False))


# 8. Pruebas de carga contra la API actual y la optimizada
CPU_SERVIDOR = "0" if (psutil.cpu_count() or 1) >= 2 else None
CPU_CLIENTE = "1" if (psutil.cpu_count() or 1) >= 2 else None
CLAVE_PRUEBA = hashlib.sha256(b"voreko-a9").hexdigest()[:32]


def correr_async(corutina):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, corutina).result()


def puerto_libre():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def iniciar_servidor(tipo, cache=0, clave=""):
    puerto = puerto_libre()
    entorno = {**os.environ, "NIVEL_LOG": "INFO", "VOREKO_CORS": "*", "VOREKO_CACHE": str(cache), "VOREKO_API_KEY": clave,
               "VOREKO_DIR_MODELO": str(DIR_V1), "VOREKO_DIR_MODELO_OPT": str(DIR_OPT), "VOREKO_DIR_FRONTEND": str(REPO / "FaseII" / "frontend")}
    if tipo == "actual":
        app_dir, app = REPO / "FaseII" / "api", "main:app"
    else:
        app_dir, app = REPO / "Actividad9" / "api", "main_optimizada:app"
        entorno.pop("VOREKO_DIR_FRONTEND")
    interprete = str(PY_ACTUAL) if tipo == "actual" else str(PY_OPT)
    cmd = [interprete, "-m", "uvicorn", app, "--app-dir", str(app_dir), "--host", "127.0.0.1", "--port", str(puerto), "--workers", "1"]
    if CPU_SERVIDOR is not None and shutil.which("taskset"):
        cmd = ["taskset", "-c", CPU_SERVIDOR] + cmd
    bitacora = open(SALIDA / "logs" / f"servidor_{tipo}_{cache}_{puerto}.log", "w")
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, env=entorno, stdout=bitacora, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{puerto}"
    while True:
        try:
            if httpx.get(url + "/api/salud", timeout=1).status_code == 200:
                break
        except Exception:
            pass
        if time.perf_counter() - t0 > 120:
            raise RuntimeError("El servidor no arrancó")
        time.sleep(0.05)
    return proc, url, time.perf_counter() - t0


def detener(proc):
    proc.terminate()
    try:
        proc.wait(10)
    except Exception:
        proc.kill()


def perturbar(obra, generador, escala=0.03):
    salida = dict(obra)
    for c in COLUMNAS:
        if c not in CATEGORICAS and c not in ("anio_inicio", "indice_tiempo_inicio"):
            salida[c] = float(obra[c]) * float(1 + generador.uniform(-escala, escala))
    return salida


def generador_trafico(prob_repeticion, semilla):
    g = np.random.default_rng(semilla)
    vistas = []

    def siguiente():
        if vistas and g.random() < prob_repeticion:
            return vistas[int(g.integers(len(vistas)))]
        nueva = perturbar(OBRAS_TE[int(g.integers(len(OBRAS_TE)))], g)
        vistas.append(nueva)
        return nueva
    return siguiente


async def carga(url, concurrencia, duracion, siguiente, encabezados=None):
    lat, errores = [], 0
    fin = time.perf_counter() + duracion
    limites = httpx.Limits(max_connections=concurrencia, max_keepalive_connections=concurrencia)
    async with httpx.AsyncClient(base_url=url, limits=limites, timeout=30, headers=encabezados or {}) as cliente:
        async def trabajador():
            nonlocal errores
            while time.perf_counter() < fin:
                cuerpo = {"id_proyecto": "A9", "variables": siguiente(), "incluir_factores": True}
                a = time.perf_counter()
                try:
                    r = await cliente.post("/api/predecir", json=cuerpo)
                    if r.status_code != 200:
                        errores += 1
                except Exception:
                    errores += 1
                lat.append((time.perf_counter() - a) * 1000)
        t0 = time.perf_counter()
        await asyncio.gather(*[trabajador() for _ in range(concurrencia)])
        real = time.perf_counter() - t0
    return np.array(lat), errores, real


def ejecutar_carga(url, pid, concurrencia, duracion, siguiente, encabezados=None):
    proc = psutil.Process(pid)
    hijos = [proc] + proc.children(recursive=True)
    for p in hijos:
        p.cpu_percent(None)
    if CPU_CLIENTE is not None:
        os.sched_setaffinity(0, {int(CPU_CLIENTE)})
    lat, err, real = correr_async(carga(url, concurrencia, duracion, siguiente, encabezados))
    cpu = sum(p.cpu_percent(None) for p in hijos)
    rss = sum(p.memory_info().rss for p in hijos) / 1e6
    if CPU_CLIENTE is not None:
        os.sched_setaffinity(0, set(range(psutil.cpu_count())))
    return {"concurrencia": concurrencia, "solicitudes": len(lat), "errores": err, "throughput_rps": len(lat) / real,
            "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)), "p99_ms": float(np.percentile(lat, 99)),
            "media_ms": float(lat.mean()), "cpu_servidor_pct": cpu, "rss_servidor_mb": rss}, lat


DURACION = float(os.getenv("VOREKO_DURACION_CARGA", "15"))
NIVELES = [1, 2, 4, 8, 16, 32]
CONFIG_CARGA = [("Actual (scikit-learn 300, sin caché)", "actual", 0, 0.0), ("Optimizada sin caché", "optimizada", 0, 0.0),
                ("Optimizada con caché (30 % repetidas)", "optimizada", 5000, 0.30)]
filas, arranques, muestras_servicio = [], [], {}
for nombre, tipo, cache, rep in CONFIG_CARGA:
    proc, url, t_arranque = iniciar_servidor(tipo, cache)
    rss_reposo = sum(p.memory_info().rss for p in [psutil.Process(proc.pid)] + psutil.Process(proc.pid).children(recursive=True)) / 1e6
    prueba = httpx.post(url + "/api/predecir", json={"id_proyecto": "A9", "variables": OBRAS_TE[0], "incluir_factores": True})
    arranques.append({"configuracion": nombre, "arranque_s": t_arranque, "rss_reposo_mb": rss_reposo, "bytes_respuesta": len(prueba.content)})
    correr_async(carga(url, 2, 2, generador_trafico(rep, 7)))
    for n in NIVELES:
        r, lat = ejecutar_carga(url, proc.pid, n, DURACION, generador_trafico(rep, SEMILLA + n))
        filas.append({"configuracion": nombre, **r})
        if n == 1:
            muestras_servicio[nombre] = lat
        print(nombre, n, {k: round(x, 1) for k, x in r.items() if isinstance(x, float)})
    if cache:
        estado = httpx.get(url + "/api/cache").json()
        arranques[-1]["tasa_aciertos_cache"] = estado["tasa_aciertos"]
    detener(proc)
CARGA = pd.DataFrame(filas)
CARGA["cumple_slo_p95"] = CARGA.p95_ms < 300
CARGA.to_csv(TAB / "06_pruebas_carga.csv", index=False)
ARRANQUE = pd.DataFrame(arranques)
ARRANQUE.to_csv(TAB / "07_arranque_y_reposo.csv", index=False)
print(ARRANQUE.round(2).to_string(index=False))

CAPACIDAD = (CARGA[CARGA.cumple_slo_p95].groupby("configuracion").throughput_rps.max().reindex([c[0] for c in CONFIG_CARGA]).rename("capacidad_rps_slo").reset_index())
CAPACIDAD["servicio_medio_ms_c1"] = [float(muestras_servicio[c[0]].mean()) for c in CONFIG_CARGA]
CAPACIDAD.to_csv(TAB / "08_capacidad_por_vcpu.csv", index=False)
print(CAPACIDAD.round(1).to_string(index=False))


# 9. Inferencia por lotes para la reevaluación nocturna
N_LOTE = 100_000
base_lote = X_todo.iloc[rng.integers(0, len(X_todo), N_LOTE)].reset_index(drop=True)
for c in COLUMNAS:
    if c not in CATEGORICAS and c not in ("anio_inicio", "indice_tiempo_inicio"):
        base_lote[c] = base_lote[c] * rng.uniform(0.97, 1.03, N_LOTE)
filas = []
t = time.perf_counter()
for i in range(0, N_LOTE, 500):
    modelo_v1.predict_proba(base_lote.iloc[i:i + 500])
t_sk_500 = time.perf_counter() - t
t = time.perf_counter()
modelo_v1.predict_proba(base_lote)
t_sk_1 = time.perf_counter() - t
s_opt, t_opt, _ = variantes[ELEGIDA]
t = time.perf_counter()
for i in range(0, N_LOTE, 10_000):
    puntuar_onnx(s_opt, t_opt, base_lote.iloc[i:i + 10_000])
t_ox = time.perf_counter() - t
base_lote.insert(0, "id_proyecto", [f"OBRA-{i:06d}" for i in range(N_LOTE)])
base_lote.to_csv(SALIDA / "logs" / "obras_activas.csv", index=False)
t = time.perf_counter()
job = subprocess.run([str(PY_OPT), str(REPO / "Actividad9" / "despliegue" / "job_reevaluacion_lote.py"), "--entrada", str(SALIDA / "logs" / "obras_activas.csv"),
                      "--salida", str(SALIDA / "logs" / "semaforos.csv"), "--modelo", str(DIR_OPT)], capture_output=True, text=True)
t_job = time.perf_counter() - t
assert job.returncode == 0, job.stderr[-1500:]
res_job = json.loads(job.stdout.strip().splitlines()[-1])
print(res_job)
LOTES = pd.DataFrame([
    {"implementacion": "Actual: /api/predecir/lote en bloques de 500 (scikit-learn 300)", "obras": N_LOTE, "segundos": t_sk_500, "obras_por_segundo": N_LOTE / t_sk_500},
    {"implementacion": "scikit-learn 300 en un solo bloque", "obras": N_LOTE, "segundos": t_sk_1, "obras_por_segundo": N_LOTE / t_sk_1},
    {"implementacion": f"Propuesta: trabajo nocturno ONNX {K_OPT} árboles, bloques de 10,000", "obras": N_LOTE, "segundos": res_job["segundos"], "obras_por_segundo": res_job["obras_por_segundo"]},
    {"implementacion": "Propuesta: trabajo nocturno completo (arranque, lectura CSV, escritura)", "obras": N_LOTE, "segundos": t_job, "obras_por_segundo": N_LOTE / t_job},
])
LOTES.to_csv(TAB / "09_inferencia_lotes.csv", index=False)
print(LOTES.round(2).to_string(index=False))


# 10. Escenarios de demanda (simulados)
PERFIL_HORA = np.array([0.2, 0.1, 0.1, 0.1, 0.2, 0.5, 2, 6, 10, 11, 10, 9, 8, 6, 7, 8, 8, 7, 5, 3, 2, 1.5, 1, 0.5])
PERFIL_HORA = PERFIL_HORA / PERFIL_HORA.sum()
ESCENARIOS = pd.DataFrame([
    {"escenario": "Piloto Guadalajara", "contratistas": 150, "obras_por_contratista": 2.0, "consultas_obra_dia": 5},
    {"escenario": "Regional Occidente", "contratistas": 4_000, "obras_por_contratista": 2.5, "consultas_obra_dia": 5},
    {"escenario": "Nacional", "contratistas": 40_000, "obras_por_contratista": 2.5, "consultas_obra_dia": 5},
    {"escenario": "Nacional con módulos integrados", "contratistas": 40_000, "obras_por_contratista": 2.5, "consultas_obra_dia": 20},
])
ESCENARIOS["obras_activas"] = (ESCENARIOS.contratistas * ESCENARIOS.obras_por_contratista).astype(int)
ESCENARIOS["consultas_dia_habil"] = ESCENARIOS.obras_activas * ESCENARIOS.consultas_obra_dia
TIPOS_DIA = {"lunes_pico": (1.30, 4), "habil": (1.00, 18), "fin_de_semana": (0.35, 8)}
ESCENARIOS["consultas_mes"] = (ESCENARIOS.consultas_dia_habil * sum(f * n for f, n in TIPOS_DIA.values())).astype(int)
ESCENARIOS["pico_hora_rps"] = ESCENARIOS.consultas_dia_habil * TIPOS_DIA["lunes_pico"][0] * PERFIL_HORA.max() / 3600
ESCENARIOS.to_csv(TAB / "10_escenarios_demanda.csv", index=False)
print(ESCENARIOS.round(2).to_string(index=False))


# 11. Simulación del SLA por configuración (eventos discretos con tiempos de servicio medidos)
import heapq

SERV = {"sk": muestras_servicio[CONFIG_CARGA[0][0]] / 1000, "ox": muestras_servicio[CONFIG_CARGA[1][0]] / 1000}
TASA_REPETIDAS = float(ARRANQUE.tasa_aciertos_cache.dropna().iloc[0])
ARR_SK = float(ARRANQUE.arranque_s.iloc[0])
ARR_OX = float(ARRANQUE.arranque_s.iloc[1])
ARR_PLATAFORMA = 5.0
ARR_RENDER_FREE = 60.0


def llegadas_dia(consultas, factor, g):
    tasa_min = np.repeat(consultas * factor * PERFIL_HORA / 60, 60)
    rafaga = np.repeat(g.lognormal(0, 0.35, 288), 5)
    rafaga = rafaga / rafaga.mean()
    n = g.poisson(tasa_min * rafaga)
    return np.sort(np.concatenate([m * 60 + g.uniform(0, 60, k) for m, k in enumerate(n) if k]))


def simular(llegadas, servicio, min_hora, n_max, arranque_s, arranque_frio_s, auto, util=0.6, bajada_min=5, timeout_s=30):
    min_hora = np.asarray(min_hora, dtype=int)
    libres = [0.0] * int(min_hora[0])
    heapq.heapify(libres)
    espera = np.zeros(len(llegadas))
    servidores_min = np.zeros(1440)
    minuto, carga, bajo, frios = 0, 0.0, 0, 0
    for i, (t, s) in enumerate(zip(llegadas, servicio)):
        m = int(t // 60)
        while minuto < m:
            ahora = (minuto + 1) * 60
            nmin = int(min_hora[min(23, minuto // 60)])
            servidores_min[minuto] = len(libres)
            if auto:
                deseadas = int(np.clip(np.ceil(carga / 60 / util), nmin, n_max))
                if deseadas > len(libres):
                    for _ in range(deseadas - len(libres)):
                        heapq.heappush(libres, ahora + arranque_s)
                    bajo = 0
                elif deseadas < len(libres):
                    bajo += 1
                    if bajo >= bajada_min and libres[0] <= ahora:
                        heapq.heappop(libres)
                        bajo = 0
                else:
                    bajo = 0
            else:
                while len(libres) < max(1, nmin):
                    heapq.heappush(libres, ahora + arranque_s)
            carga = 0.0
            minuto += 1
        if not libres:
            libres = [t + arranque_frio_s]
            frios += 1
        inicio = max(t, heapq.heappop(libres))
        espera[i] = inicio - t + s
        heapq.heappush(libres, inicio + s)
        carga += s
    servidores_min[minuto:] = len(libres)
    return espera, espera > timeout_s, servidores_min, frios


RENDER = pd.DataFrame([("0.5c-512mb", 0.5, 0.5, 7), ("1c-2g", 1, 2, 25), ("2c-4g", 2, 4, 85), ("4c-8g", 4, 8, 175), ("8c-16g", 8, 16, 300), ("12c-24g", 12, 24, 450)],
                      columns=["instancia", "cpu", "ram_gb", "usd_mes"])
RENDER_PRO = 25
CR = {"cpu_activo": 0.000024, "mem_activo": 0.0000025, "cpu_reposo": 0.0000025, "mem_reposo": 0.0000025, "solicitudes_millon": 0.40,
      "job_cpu": 0.000018, "job_mem": 0.000002, "libre_cpu": 180_000, "libre_mem": 360_000, "libre_solicitudes": 2_000_000,
      "registro_gb_mes": 0.10, "egreso_gb": 0.12}
MEM_CR_GIB = 0.5
IMAGEN_GB = float(ENTORNOS.imagen_estimada_mb.iloc[1]) / 1024
BYTES_RESPUESTA = float(ARRANQUE.bytes_respuesta.iloc[1])
POLITICAS_MIN = {"mín 0 todo el día": [0] * 24, "mín 1 de 6 a 22 h": [0] * 6 + [1] * 16 + [0] * 2, "mín 1 las 24 h": [1] * 24}

CONFIGS = [
    {"id": "A0", "nombre": "Actual: Render free, scikit-learn, 1 instancia que se apaga tras 15 min", "tipo": "sk", "cpu": 0.5, "min_hora": [0] * 24, "n_max": 1,
     "auto": True, "bajada": 15, "frio": ARR_RENDER_FREE, "plataforma": "render_free"},
    {"id": "A1", "nombre": "Actual escalada vertical: Render, scikit-learn, 1 instancia dimensionada al pico", "tipo": "sk", "vertical": True, "plataforma": "render_fija"},
    {"id": "B", "nombre": "Actual escalada horizontal: Render Pro, scikit-learn, réplicas 1c-2g con autoescalado", "tipo": "sk", "cpu": 1, "min_hora": [1] * 24, "n_max": 10,
     "auto": True, "bajada": 5, "frio": ARR_SK + ARR_PLATAFORMA, "plataforma": "render_auto"},
    {"id": "C1", "nombre": "Optimizada vertical mínima: Render 0.5c-512mb, ONNX podado, 1 instancia", "tipo": "ox", "cpu": 0.5, "min_hora": [1] * 24, "n_max": 1,
     "auto": False, "bajada": 5, "frio": ARR_OX + ARR_PLATAFORMA, "plataforma": "render_fija", "precio": 7},
    {"id": "C2", "nombre": "Propuesta: Cloud Run, ONNX podado, caché en datos, lote nocturno, autoescalado 1 vCPU", "tipo": "ox", "cpu": 1, "n_max": 10,
     "auto": True, "bajada": 15, "frio": ARR_OX + ARR_PLATAFORMA, "plataforma": "cloud_run", "cache_datos": True},
]


def correr_dia(cfg, consultas, factor, g, min_hora=None):
    lleg = llegadas_dia(consultas, factor, g)
    if cfg.get("cache_datos"):
        lleg = lleg[g.random(len(lleg)) >= TASA_REPETIDAS]
    if cfg.get("vertical"):
        inst = cfg["_instancia"]
        cpu, mh, nmax, auto, frio = inst.cpu, [max(1, int(inst.cpu))] * 24, max(1, int(inst.cpu)), False, ARR_SK + ARR_PLATAFORMA
    else:
        cpu, mh, nmax, auto, frio = cfg["cpu"], min_hora if min_hora is not None else cfg["min_hora"], cfg["n_max"], cfg["auto"], cfg["frio"]
    serv = g.choice(SERV[cfg["tipo"]], len(lleg)) * (1 / cpu if cpu < 1 else 1.0)
    arranque = ARR_SK + ARR_PLATAFORMA if cfg["tipo"] == "sk" else ARR_OX + ARR_PLATAFORMA
    e, er, sv, frios = simular(lleg, serv, mh, nmax, arranque, frio, auto, bajada_min=cfg.get("bajada", 5))
    return {"solicitudes": len(lleg), "p50_ms": np.percentile(e, 50) * 1000, "p95_ms": np.percentile(e, 95) * 1000, "p99_ms": np.percentile(e, 99) * 1000,
            "pct_bajo_300": float((e < 0.3).mean()), "disponibilidad": float(1 - er.mean()), "cpu_seg": float(serv.sum()), "arranques_frio": frios,
            "instancia_prom": float(sv.mean()), "instancia_max": int(sv.max()), "instancia_horas": float(sv.sum() / 60), "_serie": sv}


def cumple(r):
    return r["pct_bajo_300"] >= 0.95 and r["disponibilidad"] >= 0.995


def evaluar(cfg, esc, semilla, min_hora=None):
    filas_ = []
    for tipo_dia, (factor, dias) in TIPOS_DIA.items():
        r = correr_dia(cfg, esc.consultas_dia_habil, factor, np.random.default_rng(semilla + int(factor * 100)), min_hora)
        filas_.append({"tipo_dia": tipo_dia, "dias": dias, **r})
    return pd.DataFrame(filas_)


def elegir_vertical(cfg, esc, semilla):
    for _, inst in RENDER.iterrows():
        cfg["_instancia"] = inst
        r = correr_dia(cfg, esc.consultas_dia_habil, TIPOS_DIA["lunes_pico"][0], np.random.default_rng(semilla))
        if cumple(r) and r["p95_ms"] < 300 * 0.8:
            return inst
    return RENDER.iloc[-1]


def costo(cfg, esc, res, min_hora=None):
    dias = res.dias.to_numpy()
    mes = lambda col: float((res[col] * dias).sum())
    solicitudes = mes("solicitudes")
    plat = cfg["plataforma"]
    if plat == "render_free":
        partes = {"cuota_plataforma": 0.0, "computo": 0.0}
    elif plat == "render_fija":
        partes = {"cuota_plataforma": 0.0, "computo": float(cfg["_instancia"].usd_mes) if cfg.get("vertical") else float(cfg["precio"])}
    elif plat == "render_auto":
        partes = {"cuota_plataforma": RENDER_PRO, "computo": mes("instancia_horas") * 25 / 730}
    else:
        mh = np.asarray(min_hora if min_hora is not None else [1] * 24)
        activo = mes("cpu_seg") + mes("arranques_frio") * (ARR_OX + ARR_PLATAFORMA)
        reposo = float(sum(max(0.0, mh.sum() * 3600 - r.cpu_seg) * r.dias for _, r in res.iterrows()))
        seg_lote = max(60.0, esc.obras_activas / float(LOTES.obras_por_segundo.iloc[3]) + ARR_PLATAFORMA)
        egreso_gb = solicitudes * BYTES_RESPUESTA / 2**30
        partes = {"cuota_plataforma": 0.0,
                  "computo": activo * (CR["cpu_activo"] + MEM_CR_GIB * CR["mem_activo"]),
                  "instancia_minima_en_reposo": reposo * (CR["cpu_reposo"] + MEM_CR_GIB * CR["mem_reposo"]),
                  "solicitudes": solicitudes / 1e6 * CR["solicitudes_millon"],
                  "trabajo_nocturno": 30 * seg_lote * (CR["job_cpu"] + MEM_CR_GIB * CR["job_mem"]),
                  "registro_imagen": max(0.0, IMAGEN_GB - 0.5) * CR["registro_gb_mes"],
                  "egreso_red": max(0.0, egreso_gb - 1) * CR["egreso_gb"]}
        partes["descuento_capa_gratuita"] = -(min(activo, CR["libre_cpu"]) * CR["cpu_activo"] + min(activo * MEM_CR_GIB, CR["libre_mem"]) * CR["mem_activo"]
                                              + min(solicitudes, CR["libre_solicitudes"]) / 1e6 * CR["solicitudes_millon"])
    total = sum(v_ for k_, v_ in partes.items() if k_ != "descuento_capa_gratuita")
    return {**partes, "total_usd_mes": total, "total_con_capa_gratuita": total + partes.get("descuento_capa_gratuita", 0.0),
            "solicitudes_api_mes": solicitudes}


def resumen_sla(res):
    pico = res[res.tipo_dia == "lunes_pico"].iloc[0]
    w = res.solicitudes * res.dias
    return {"p50_ms_pico": pico.p50_ms, "p95_ms_pico": pico.p95_ms, "p99_ms_pico": pico.p99_ms, "pct_bajo_300_pico": pico.pct_bajo_300,
            "disponibilidad_pico": pico.disponibilidad, "pct_bajo_300_mes": float((res.pct_bajo_300 * w).sum() / w.sum()),
            "disponibilidad_mes": float((res.disponibilidad * w).sum() / w.sum()), "arranques_frio_mes": float((res.arranques_frio * res.dias).sum()),
            "instancias_max_pico": int(pico.instancia_max), "instancias_prom_pico": float(pico.instancia_prom)}


filas_sla, filas_costo, filas_pol, series = [], [], [], {}
for _, esc in ESCENARIOS.iterrows():
    for cfg0 in CONFIGS:
        cfg = dict(cfg0)
        semilla = SEMILLA + sum(map(ord, cfg["id"]))
        if cfg.get("vertical"):
            elegir_vertical(cfg, esc, semilla)
        if cfg["plataforma"] == "cloud_run":
            mejor = None
            for nombre_pol, mh in POLITICAS_MIN.items():
                res = evaluar(cfg, esc, semilla, mh)
                sla = resumen_sla(res)
                c = costo(cfg, esc, res, mh)
                ok = sla["pct_bajo_300_pico"] >= 0.95 and sla["disponibilidad_pico"] >= 0.995 and sla["pct_bajo_300_mes"] >= 0.95 and sla["disponibilidad_mes"] >= 0.995
                filas_pol.append({"escenario": esc.escenario, "politica": nombre_pol, **sla, "total_usd_mes": c["total_usd_mes"], "cumple_sla": ok})
                if ok and (mejor is None or c["total_usd_mes"] < mejor[3]["total_usd_mes"]):
                    mejor = (nombre_pol, res, sla, c)
            nombre_pol, res, sla, c = mejor
            nombre_cfg = cfg["nombre"] + f" ({nombre_pol})"
        else:
            res = evaluar(cfg, esc, semilla)
            sla, c = resumen_sla(res), costo(cfg, esc, res)
            nombre_cfg = cfg["nombre"] + (f" ({cfg['_instancia'].instancia})" if cfg.get("vertical") else "")
        filas_sla.append({"escenario": esc.escenario, "id": cfg["id"], "configuracion": nombre_cfg, **sla})
        filas_costo.append({"escenario": esc.escenario, "id": cfg["id"], "configuracion": nombre_cfg, **c})
        series[(esc.escenario, cfg["id"])] = res[res.tipo_dia == "lunes_pico"].iloc[0]["_serie"]
        print(esc.escenario, cfg["id"], nombre_cfg[-30:], round(sla["p95_ms_pico"], 1), round(sla["pct_bajo_300_mes"], 4), round(sla["disponibilidad_mes"], 5), round(c["total_usd_mes"], 2))
SLA = pd.DataFrame(filas_sla)
SLA["cumple_sla"] = (SLA.pct_bajo_300_pico >= 0.95) & (SLA.disponibilidad_pico >= 0.995) & (SLA.pct_bajo_300_mes >= 0.95) & (SLA.disponibilidad_mes >= 0.995)
COSTOS = pd.DataFrame(filas_costo).fillna(0.0)
COSTOS = COSTOS.merge(SLA[["escenario", "id", "cumple_sla"]], on=["escenario", "id"])
COSTOS["usd_por_1000_consultas"] = COSTOS.total_usd_mes / COSTOS.escenario.map(ESCENARIOS.set_index("escenario").consultas_mes) * 1000
for ref in ["A1", "B"]:
    base_costo = COSTOS[COSTOS.id == ref].set_index("escenario").total_usd_mes
    COSTOS[f"ahorro_vs_{ref}_pct"] = 1 - COSTOS.total_usd_mes / COSTOS.escenario.map(base_costo)
POLITICAS = pd.DataFrame(filas_pol)
SLA.to_csv(TAB / "11_simulacion_sla.csv", index=False)
COSTOS.to_csv(TAB / "12_costos_mensuales.csv", index=False)
POLITICAS.to_csv(TAB / "13_politicas_instancia_minima.csv", index=False)
print(SLA[["escenario", "id", "p95_ms_pico", "pct_bajo_300_mes", "disponibilidad_mes", "instancias_max_pico", "cumple_sla"]].round(4).to_string(index=False))
print(COSTOS[["escenario", "id", "total_usd_mes", "usd_por_1000_consultas", "cumple_sla", "ahorro_vs_A1_pct", "ahorro_vs_B_pct"]].round(4).to_string(index=False))
print(POLITICAS[["escenario", "politica", "p95_ms_pico", "pct_bajo_300_mes", "arranques_frio_mes", "total_usd_mes", "cumple_sla"]].round(4).to_string(index=False))

SENS = []
pol_nac = SLA[(SLA.escenario == "Nacional") & (SLA.id == "C2")].configuracion.iloc[0].split("(")[-1].rstrip(")")
for tasa in [0.0, 0.15, 0.30, 0.45]:
    TASA_REPETIDAS_ORIG, TASA_REPETIDAS = TASA_REPETIDAS, tasa
    for _, esc in ESCENARIOS.iloc[2:].iterrows():
        res = evaluar(dict(CONFIGS[-1]), esc, SEMILLA, POLITICAS_MIN[pol_nac])
        c = costo(CONFIGS[-1], esc, res, POLITICAS_MIN[pol_nac])
        SENS.append({"escenario": esc.escenario, "politica": pol_nac, "consultas_repetidas_servidas_desde_datos": tasa, "total_usd_mes": c["total_usd_mes"],
                     "p95_ms_pico": resumen_sla(res)["p95_ms_pico"]})
    TASA_REPETIDAS = TASA_REPETIDAS_ORIG
SENS = pd.DataFrame(SENS)
SENS["ahorro_vs_A1_pct"] = 1 - SENS.total_usd_mes / SENS.escenario.map(COSTOS[COSTOS.id == "A1"].set_index("escenario").total_usd_mes)
SENS.to_csv(TAB / "13b_sensibilidad_cache_datos.csv", index=False)
print(SENS.round(3).to_string(index=False))


# 12. Auditoría de fairness con predicciones fuera de pliegue
OOF = oof_por_k[300]
OOF_OPT = oof_por_k[K_OPT]
zona_mediana = df.groupby("localidad").precio_venta_inicial_m2.median()
cortes_zona = zona_mediana.quantile([1 / 3, 2 / 3]).to_numpy()
GRUPOS = pd.DataFrame({
    "segmento_economico": pd.qcut(df.costo_prelim_m2.values, 3, labels=["económica", "media", "alta"]).astype(str),
    "tamano_obra": pd.qcut(df.area_construida_m2.values, 3, labels=["pequeña", "mediana", "grande"]).astype(str),
    "zona_plusvalia": pd.cut(df.localidad.map(zona_mediana).values, [-np.inf, *cortes_zona, np.inf], labels=["baja", "media", "alta"]).astype(str),
})


def metricas_grupo(y, p, grupos, u=U_ROJO):
    filas_ = []
    for atributo in grupos.columns:
        for valor in sorted(grupos[atributo].unique()):
            msk = (grupos[atributo] == valor).to_numpy()
            yy, pp = y[msk], p[msk] >= u
            vp, fp = int((yy & pp).sum()), int((~yy & pp).sum())
            filas_.append({"atributo": atributo, "grupo": valor, "obras": int(msk.sum()), "prevalencia": yy.mean(), "tasa_alerta_roja": pp.mean(),
                           "tpr_recall": vp / max(1, yy.sum()), "fpr": fp / max(1, (~yy).sum()), "precision": vp / max(1, pp.sum()),
                           "auc": roc_auc_score(yy, p[msk]) if 0 < yy.sum() < len(yy) else np.nan})
    return pd.DataFrame(filas_)


def disparidades(tabla):
    salida = []
    for atributo, t in tabla.groupby("atributo"):
        salida.append({"atributo": atributo, "paridad_demografica_ratio": t.tasa_alerta_roja.min() / t.tasa_alerta_roja.max(),
                       "paridad_demografica_dif": t.tasa_alerta_roja.max() - t.tasa_alerta_roja.min(),
                       "igualdad_oportunidades_dif_tpr": t.tpr_recall.max() - t.tpr_recall.min(), "dif_fpr": t.fpr.max() - t.fpr.min(),
                       "equalized_odds_dif": max(t.tpr_recall.max() - t.tpr_recall.min(), t.fpr.max() - t.fpr.min()),
                       "predictive_parity_dif": t.precision.max() - t.precision.min(), "dif_prevalencia": t.prevalencia.max() - t.prevalencia.min()})
    return pd.DataFrame(salida)


Y = y_todo.to_numpy().astype(bool)
FAIR = metricas_grupo(Y, OOF, GRUPOS)
DISP = disparidades(FAIR)
boot = []
for b in range(1000):
    idx = rng.integers(0, len(Y), len(Y))
    t = metricas_grupo(Y[idx], OOF[idx], GRUPOS.iloc[idx].reset_index(drop=True))
    d = disparidades(t)
    d["b"] = b
    boot.append(d)
BOOT = pd.concat(boot)
IC = BOOT.groupby("atributo")[["paridad_demografica_ratio", "igualdad_oportunidades_dif_tpr", "dif_fpr", "predictive_parity_dif"]].quantile([0.025, 0.975]).unstack()
IC.columns = [f"{a}_ic{int(q * 1000) / 10}" for a, q in IC.columns]
DISP = DISP.merge(IC.reset_index(), on="atributo")
TOL = {"paridad_demografica_ratio": 0.80, "igualdad_oportunidades_dif_tpr": 0.10, "dif_fpr": 0.10, "predictive_parity_dif": 0.15}
DISP["pasa_igualdad_oportunidades"] = DISP.igualdad_oportunidades_dif_tpr <= TOL["igualdad_oportunidades_dif_tpr"]
DISP["pasa_fpr"] = DISP.dif_fpr <= TOL["dif_fpr"]
DISP["pasa_paridad_demografica"] = DISP.paridad_demografica_ratio >= TOL["paridad_demografica_ratio"]
FAIR.to_csv(TAB / "14_fairness_por_grupo.csv", index=False)
DISP.to_csv(TAB / "15_fairness_disparidades.csv", index=False)
print(FAIR.round(3).to_string(index=False))
print(DISP.round(3).to_string(index=False))

FAIR_OPT = metricas_grupo(Y, OOF_OPT, GRUPOS)
DISP_OPT = pd.concat([disparidades(FAIR).assign(modelo="300 árboles (actual)"), disparidades(FAIR_OPT).assign(modelo=f"{K_OPT} árboles (optimizado)")])
DISP_OPT.to_csv(TAB / "15b_fairness_actual_vs_optimizado.csv", index=False)
print(DISP_OPT.round(3).to_string(index=False))


# 13. Mitigación de sesgos y reevaluación
PEOR = DISP.sort_values("equalized_odds_dif", ascending=False).atributo.iloc[0]
print("Atributo con mayor brecha:", PEOR)
g_peor = GRUPOS[PEOR].to_numpy()


def pesos_reponderacion(y, g):
    w = np.zeros(len(y))
    for gv in np.unique(g):
        for yv in (0, 1):
            m = (g == gv) & (y == yv)
            if m.sum():
                w[m] = (np.mean(g == gv) * np.mean(y == yv)) / m.mean()
    return w


oof_rw = np.zeros(len(Y))
for tr, te in pliegues:
    m = clone(modelo_v1)
    m.named_steps["modelo"].set_params(n_jobs=-1)
    m.fit(X_todo.iloc[tr], y_todo.iloc[tr], modelo__sample_weight=pesos_reponderacion(Y[tr].astype(int), g_peor[tr]))
    oof_rw[te] = m.predict_proba(X_todo.iloc[te])[:, 1]

decision_um = np.zeros(len(Y), dtype=bool)
umbrales_grupo = []
for k, (tr, te) in enumerate(pliegues):
    objetivo = recall_score(Y[tr], OOF[tr] >= U_ROJO)
    for gv in np.unique(g_peor):
        mtr = (g_peor[tr] == gv)
        yy, pp = Y[tr][mtr], OOF[tr][mtr]
        cand = np.round(np.arange(0.10, 0.80, 0.01), 2)
        tprs = np.array([((pp >= u) & yy).sum() / max(1, yy.sum()) for u in cand])
        u = float(cand[np.argmin(np.abs(tprs - objetivo) + 1e-6 * np.abs(cand - U_ROJO))])
        umbrales_grupo.append({"pliegue": k, "grupo": gv, "umbral": u})
        mte = g_peor[te] == gv
        decision_um[te[mte]] = OOF[te][mte] >= u
UMBRALES_GRUPO = pd.DataFrame(umbrales_grupo)

decision_arriba = np.zeros(len(Y), dtype=bool)
umbrales_arriba = []
for k, (tr, te) in enumerate(pliegues):
    tpr_grupo = {gv: recall_score(Y[tr][g_peor[tr] == gv], OOF[tr][g_peor[tr] == gv] >= U_ROJO) for gv in np.unique(g_peor)}
    objetivo = max(tpr_grupo.values())
    for gv in np.unique(g_peor):
        mtr = g_peor[tr] == gv
        yy, pp = Y[tr][mtr], OOF[tr][mtr]
        u = U_ROJO
        for cand in np.round(np.arange(U_ROJO, 0.09, -0.01), 2):
            if ((pp >= cand) & yy).sum() / max(1, yy.sum()) >= objetivo - 1e-9:
                u = float(cand)
                break
        umbrales_arriba.append({"pliegue": k, "grupo": gv, "umbral": u})
        mte = g_peor[te] == gv
        decision_arriba[te[mte]] = OOF[te][mte] >= u
UMBRALES_GRUPO = pd.concat([UMBRALES_GRUPO.assign(estrategia="igualar al promedio"), pd.DataFrame(umbrales_arriba).assign(estrategia="nivelar hacia arriba")])


def resumen_estrategia(nombre, decision, score):
    p_equiv = np.where(decision, 1.0, 0.0) if score is None else score
    u = 0.5 if score is None else U_ROJO
    t = metricas_grupo(Y, p_equiv, GRUPOS, u)
    d = disparidades(t).set_index("atributo").loc[PEOR]
    rec = recall_score(Y, decision)
    prec = precision_score(Y, decision)
    return {"estrategia": nombre, "atributo": PEOR, "recall_global": rec, "precision_global": prec,
            "auc_global": roc_auc_score(Y, score) if score is not None else np.nan, "dif_tpr": d.igualdad_oportunidades_dif_tpr,
            "dif_fpr": d.dif_fpr, "equalized_odds_dif": d.equalized_odds_dif, "paridad_demografica_ratio": d.paridad_demografica_ratio,
            "predictive_parity_dif": d.predictive_parity_dif, "pasa_calidad": rec >= 0.45 and prec >= 0.40}, t.assign(estrategia=nombre)


r0, t0 = resumen_estrategia("Sin mitigación (umbral único 0.39)", OOF >= U_ROJO, OOF)
r1, t1 = resumen_estrategia("Preprocesamiento: reponderación de muestras", oof_rw >= U_ROJO, oof_rw)
r2, t2 = resumen_estrategia("Posprocesamiento: umbral por grupo igualando al promedio", decision_um, None)
r3, t3 = resumen_estrategia("Posprocesamiento: umbral por grupo nivelando hacia arriba", decision_arriba, None)
MITIG = pd.DataFrame([r0, r1, r2, r3])
MITIG["pasa_fairness"] = (MITIG.dif_tpr <= TOL["igualdad_oportunidades_dif_tpr"]) & (MITIG.dif_fpr <= TOL["dif_fpr"])
MITIG.to_csv(TAB / "16_mitigacion_sesgos.csv", index=False)
pd.concat([t0, t1, t2, t3]).query("atributo == @PEOR").to_csv(TAB / "17_mitigacion_por_grupo.csv", index=False)
UMBRALES_GRUPO.to_csv(TAB / "18_umbrales_por_grupo.csv", index=False)
print(MITIG.round(3).to_string(index=False))
print(UMBRALES_GRUPO.groupby(["estrategia", "grupo"]).umbral.agg(["mean", "min", "max"]).round(3))


# 14. Auditoría técnica: robustez, reproducibilidad, seguridad y trazabilidad
filas = []
g = np.random.default_rng(SEMILLA)
for nombre, f in [("Actual (scikit-learn 300)", lambda X: modelo_v1.predict_proba(X)[:, 1]),
                  ("Propuesta (ONNX podado)", lambda X: puntuar_onnx(s_opt, t_opt, X))]:
    base_p = f(X_te)
    for escala in [0.01, 0.03, 0.05]:
        cambios, dp = [], []
        for r in range(20):
            Xp = X_te.copy()
            for c in COLUMNAS:
                if c not in CATEGORICAS and c not in ("anio_inicio", "indice_tiempo_inicio"):
                    Xp[c] = Xp[c] * g.uniform(1 - escala, 1 + escala, len(Xp))
            p = f(Xp)
            cambios.append((semaforo(p) != semaforo(base_p)).mean())
            dp.append(np.abs(p - base_p).mean())
        filas.append({"modelo": nombre, "perturbacion_pct": escala * 100, "cambio_semaforo_pct": float(np.mean(cambios) * 100),
                      "cambio_prob_medio": float(np.mean(dp))})
ROBUSTEZ = pd.DataFrame(filas)
ROBUSTEZ.to_csv(TAB / "19_robustez.csv", index=False)
print(ROBUSTEZ.round(4).to_string(index=False))

pruebas = []
proc_a, url_a, _ = iniciar_servidor("actual")
proc_o, url_o, _ = iniciar_servidor("optimizada", 5000, CLAVE_PRUEBA)
cuerpo = {"id_proyecto": "AUD-1", "variables": OBRAS_TE[0], "incluir_factores": True}
ra = httpx.post(url_a + "/api/predecir", json=cuerpo, headers={"Origin": "https://sitio-desconocido.example"})
ro_sin = httpx.post(url_o + "/api/predecir", json=cuerpo)
ro_con = httpx.post(url_o + "/api/predecir", json=cuerpo, headers={"X-API-Key": CLAVE_PRUEBA})
fuera = dict(OBRAS_TE[0], area_construida_m2=OBRAS_TE[0]["area_construida_m2"] * 50)
rf_ = httpx.post(url_o + "/api/predecir", json={"variables": fuera, "incluir_factores": True}, headers={"X-API-Key": CLAVE_PRUEBA})
neg = httpx.post(url_o + "/api/predecir", json={"variables": dict(OBRAS_TE[0], area_construida_m2=-5)}, headers={"X-API-Key": CLAVE_PRUEBA})
salud_o = httpx.get(url_o + "/api/salud").json()
pruebas += [
    {"dimension": "Seguridad", "prueba": "Predicción sin credenciales en la API actual", "resultado": f"HTTP {ra.status_code}", "esperado": "401", "cumple": ra.status_code == 401},
    {"dimension": "Seguridad", "prueba": "CORS ante un origen desconocido (API actual)", "resultado": ra.headers.get("access-control-allow-origin", "sin encabezado"), "esperado": "dominio de VOREKO", "cumple": ra.headers.get("access-control-allow-origin") not in ("*",)},
    {"dimension": "Seguridad", "prueba": "Predicción sin clave en la API optimizada", "resultado": f"HTTP {ro_sin.status_code}", "esperado": "401", "cumple": ro_sin.status_code == 401},
    {"dimension": "Seguridad", "prueba": "Predicción con clave en la API optimizada", "resultado": f"HTTP {ro_con.status_code}", "esperado": "200", "cumple": ro_con.status_code == 200},
    {"dimension": "Explicabilidad", "prueba": "La respuesta incluye los 3 factores principales", "resultado": str(len(ro_con.json().get("factores", []))), "esperado": "3", "cumple": len(ro_con.json().get("factores", [])) == 3},
    {"dimension": "Robustez de datos", "prueba": "Área 50 veces mayor que la observada", "resultado": "advertencia emitida" if rf_.json().get("advertencias") else "sin advertencia", "esperado": "advertencia", "cumple": bool(rf_.json().get("advertencias"))},
    {"dimension": "Robustez de datos", "prueba": "Área negativa", "resultado": f"HTTP {neg.status_code}", "esperado": "422", "cumple": neg.status_code == 422},
    {"dimension": "Trazabilidad", "prueba": "Versión del modelo en cada respuesta", "resultado": ro_con.json().get("version_modelo", ""), "esperado": "nombre y versión", "cumple": "1.1.0" in ro_con.json().get("version_modelo", "")},
    {"dimension": "Trazabilidad", "prueba": "Huella SHA-256 del modelo ONNX registrada", "resultado": META_OPT["huellas_sha256"]["modelo_onnx"][:16] + "…", "esperado": "huella presente", "cumple": hashlib.sha256((DIR_OPT / "modelo_riesgo.onnx").read_bytes()).hexdigest() == META_OPT["huellas_sha256"]["modelo_onnx"]},
    {"dimension": "Reproducibilidad", "prueba": "Métricas de la Fase II reproducidas al reentrenar", "resultado": f"recall {META_V1['metricas_prueba']['recall']:.3f}, AUC {META_V1['metricas_prueba']['roc_auc']:.3f}", "esperado": "0.538 / 0.803", "cumple": abs(META_V1["metricas_prueba"]["recall"] - 0.538) < 0.001 and abs(META_V1["metricas_prueba"]["roc_auc"] - 0.803) < 0.001},
    {"dimension": "Equivalencia", "prueba": "Semáforo ONNX igual al de scikit-learn en 372 obras", "resultado": f"{int(EQUIV.set_index('variante').loc[ELEGIDA].cambios_semaforo_372)} cambios", "esperado": "0 cambios", "cumple": int(EQUIV.set_index("variante").loc[ELEGIDA].cambios_semaforo_372) == 0},
]
detener(proc_a)
detener(proc_o)
TECNICA = pd.DataFrame(pruebas)
TECNICA.to_csv(TAB / "20_pruebas_auditoria_tecnica.csv", index=False)
print(TECNICA.to_string(index=False))


# 15. Figuras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TINTA, TINTA2, REJILLA = "#1f2430", "#5b6170", "#e6e8ec"
C_ACT, C_OPT, C_CACHE, C_EXTRA = "#b4532a", "#2a6f97", "#3d9970", "#8a8f99"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "xtick.color": TINTA2,
                     "ytick.color": TINTA2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.8})
COLOR_CFG = dict(zip([c[0] for c in CONFIG_CARGA], [C_ACT, C_OPT, C_CACHE]))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for nombre, t in CARGA.groupby("configuracion", sort=False):
    ax[0].plot(t.throughput_rps, t.p95_ms, "o-", color=COLOR_CFG[nombre], label=nombre, lw=1.8, ms=4)
    ax[1].plot(t.concurrencia, t.cpu_servidor_pct, "o-", color=COLOR_CFG[nombre], lw=1.8, ms=4)
ax[0].axhline(300, color=TINTA2, ls="--", lw=1)
ax[0].text(ax[0].get_xlim()[1] * 0.98, 310, "SLO p95 300 ms", ha="right", color=TINTA2, fontsize=8)
ax[0].set_xscale("log"); ax[0].set_yscale("log")
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
ax[0].xaxis.set_major_locator(FixedLocator([40, 60, 100, 200, 300, 500])); ax[0].xaxis.set_minor_formatter(NullFormatter())
ax[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}")); ax[0].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
ax[0].set_xlabel("Throughput (solicitudes por segundo)"); ax[0].set_ylabel("Latencia p95 (ms)")
ax[0].set_title("Latencia p95 contra throughput (1 vCPU)", loc="left", color=TINTA)
ax[1].set_xscale("log", base=2); ax[1].xaxis.set_major_locator(FixedLocator(NIVELES)); ax[1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}")); ax[1].xaxis.set_minor_formatter(NullFormatter()); ax[1].set_xlabel("Usuarios concurrentes"); ax[1].set_ylabel("CPU del servidor (%)")
ax[1].set_title("Uso de CPU del servidor", loc="left", color=TINTA)
ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "01_carga_latencia_throughput.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.2))
orden = MICRO.sort_values("ms_consulta_29_filas")
ax.barh(orden.implementacion, orden.ms_consulta_29_filas, color=[C_OPT if "ONNX" in s else C_ACT for s in orden.implementacion])
for y_, v_ in enumerate(orden.ms_consulta_29_filas):
    ax.text(v_, y_, f"  {v_:.2f} ms", va="center", fontsize=8, color=TINTA)
ax.set_xscale("log"); ax.set_xlabel("Milisegundos por consulta con factores (29 filas, mediana, escala logarítmica)")
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
ax.set_title("Tiempo de inferencia por implementación", loc="left", color=TINTA); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(FIG / "02_microbenchmark.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(1, 4, figsize=(13, 3.6), sharey=False)
for i, (esc, t) in enumerate(COSTOS.groupby("escenario", sort=False)):
    colores = [C_ACT if x.startswith("A") else C_EXTRA if x == "B" else C_OPT for x in t.id]
    barras = ax[i].bar(t.id, t.total_usd_mes, color=colores)
    for b_, (_, r_) in zip(barras, t.iterrows()):
        ax[i].text(b_.get_x() + b_.get_width() / 2, b_.get_height(), (f"${r_.total_usd_mes:,.0f}" if r_.total_usd_mes >= 1 else f"${r_.total_usd_mes:.2f}") + ("" if r_.cumple_sla else "\nsin SLA"),
                   ha="center", va="bottom", fontsize=7.5, color=TINTA if r_.cumple_sla else C_ACT)
    ax[i].set_title(esc, loc="left", color=TINTA, fontsize=9); ax[i].grid(axis="x", visible=False)
    ax[i].set_ylim(0, t.total_usd_mes.max() * 1.35 + 1)
ax[0].set_ylabel("USD por mes")
fig.suptitle("Costo mensual por configuración y escenario", x=0.01, ha="left", color=TINTA, fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "03_costos_por_escenario.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.2))
esc_nac = ESCENARIOS.escenario.iloc[-1]
lleg_demo = llegadas_dia(ESCENARIOS.consultas_dia_habil.iloc[-1], TIPOS_DIA["lunes_pico"][0], np.random.default_rng(SEMILLA))
por_min = np.bincount((lleg_demo // 60).astype(int), minlength=1440) / 60
ax.fill_between(np.arange(1440) / 60, por_min, color=REJILLA, label="Solicitudes por segundo (promedio por minuto)")
ax2 = ax.twinx()
ax2.step(np.arange(1440) / 60, series[(esc_nac, "C2")], color=C_OPT, lw=1.6, label="Instancias Cloud Run (propuesta optimizada)")
ax2.step(np.arange(1440) / 60, series[(esc_nac, "B")], color=C_EXTRA, lw=1.2, ls="--", label="Instancias Render con el modelo actual (B)")
ax.set_xlabel("Hora del día (lunes pico, escenario nacional con módulos integrados)"); ax.set_ylabel("Solicitudes por segundo"); ax2.set_ylabel("Instancias activas")
ax2.spines["right"].set_visible(True); ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5, loc="upper left")
ax.set_title("Autoescalado horizontal frente a la demanda simulada", loc="left", color=TINTA)
fig.tight_layout(); fig.savefig(FIG / "04_autoescalado_dia_pico.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
ORDEN_GRUPOS = {"segmento_economico": ["económica", "media", "alta"], "tamano_obra": ["pequeña", "mediana", "grande"], "zona_plusvalia": ["baja", "media", "alta"]}
NOMBRE_ATRIBUTO = {"segmento_economico": "Segmento económico de la obra", "tamano_obra": "Tamaño de la obra", "zona_plusvalia": "Zona por plusvalía"}
for i, (atributo, t) in enumerate(FAIR.groupby("atributo", sort=False)):
    t = t.set_index("grupo").loc[ORDEN_GRUPOS[atributo]].reset_index()
    x = np.arange(len(t)); w = 0.26
    ax[i].bar(x - w, t.tasa_alerta_roja, w, color=C_EXTRA, label="Tasa de alerta roja")
    ax[i].bar(x, t.tpr_recall, w, color=C_OPT, label="TPR (recall)")
    ax[i].bar(x + w, t.fpr, w, color=C_ACT, label="FPR")
    ax[i].set_xticks(x, t.grupo); ax[i].set_ylim(0, 1); ax[i].grid(axis="x", visible=False)
    ax[i].set_title(NOMBRE_ATRIBUTO[atributo], loc="left", color=TINTA)
ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.suptitle("Métricas por grupo con predicciones fuera de pliegue (372 obras, umbral 0.39)", x=0.01, ha="left", color=TINTA, fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "05_fairness_por_grupo.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 3.2))
tm = pd.concat([t0, t1, t2, t3]).query("atributo == @PEOR")
for j, (est, t) in enumerate(tm.groupby("estrategia", sort=False)):
    t = t.set_index("grupo").loc[ORDEN_GRUPOS[PEOR]].reset_index()
    ax.plot(t.grupo, t.tpr_recall, "o-" if j != 1 else "s--", lw=1.8 if j != 1 else 1.2, ms=6 if j != 1 else 9, mfc="none" if j == 1 else None,
            color=[C_ACT, C_EXTRA, "#c9a227", C_OPT][j], label=est, zorder=3 - (j == 1))
ax.set_ylim(0, 1); ax.set_ylabel("TPR (recall) por grupo"); ax.set_xlabel(NOMBRE_ATRIBUTO[PEOR])
ax.set_title("Efecto de la mitigación en la igualdad de oportunidades", loc="left", color=TINTA)
ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(FIG / "06_mitigacion.png", dpi=200); plt.close(fig)


# 16. Resumen de resultados
RESUMEN = {
    "arboles_elegidos": K_OPT, "variante_onnx": ELEGIDA,
    "aceleracion_consulta_inferencia": float(MICRO.aceleracion_consulta.iloc[-1]),
    "capacidad_rps_actual": float(CAPACIDAD.capacidad_rps_slo.iloc[0]), "capacidad_rps_optimizada": float(CAPACIDAD.capacidad_rps_slo.iloc[1]),
    "capacidad_rps_optimizada_cache": float(CAPACIDAD.capacidad_rps_slo.iloc[2]), "tasa_aciertos_cache": TASA_REPETIDAS,
    "costos_por_escenario": {e: {i: round(float(x), 2) for i, x in t.set_index("id").total_usd_mes.items()} for e, t in COSTOS.groupby("escenario", sort=False)},
    "ahorro_C2_vs_A1": {e: round(float(t.set_index("id").loc["C2"].ahorro_vs_A1_pct), 4) for e, t in COSTOS.groupby("escenario", sort=False)},
    "atributo_mayor_brecha": PEOR, "metricas_prueba_optimizado": META_OPT["metricas_prueba"],
}
(SALIDA / "resumen_actividad9.json").write_text(json.dumps(RESUMEN, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(json.dumps(RESUMEN, ensure_ascii=False, indent=1, default=float))
