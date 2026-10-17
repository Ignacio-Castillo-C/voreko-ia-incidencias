# VOREKO IA — Actividad 5: datos, entrenamiento y registro en MLflow

**Estimación temprana del costo real de construcción por m² para el motor de alertas de desviación de costo de VOREKO.**

Maestría en Inteligencia Artificial · Universidad Tecmilenio — *Gestión de Proyectos de IA*, Módulo 2, Semana 6.
Autor: José Ignacio Castillo Camacho.

## 1. Contexto y objetivo

VOREKO es una plataforma para dar seguimiento a obras residenciales pequeñas y medianas (FlutterFlow + Supabase). En la Fase I se planteó detectar tempranamente las desviaciones de costo, tiempo y riesgo. En la Actividad 4 se clasificaron observaciones de obra. Esta actividad construye otro componente: **predecir, con la información disponible al inicio de una obra, su costo real de construcción por m²**, para compararlo con el presupuesto y anticipar sobrecostos.

Mientras VOREKO acumula datos propios, se usa un dataset público equivalente.

| | |
|---|---|
| **Dataset** | *Residential Building Data Set*, UCI Machine Learning Repository, id 437 — CC BY 4.0 — doi:10.24432/C5S896 |
| **Contenido** | 372 edificios residenciales (Teherán, ≈1993–2011): 8 variables del proyecto, 19 indicadores económicos en 5 rezagos, costo y precio reales |
| **Problema** | Regresión del costo real por m² (MIRR/m²) |
| **Métricas** | MAE, RMSE, R² (y MAPE como lectura de negocio) |
| **Modelos** | Ridge, Random Forest y XGBoost con GridSearchCV; referencias: mediana y regla presupuestal |

## 2. Estructura

```
Actividad5/
├── datos/
│   ├── datos_ini/                 # datos originales (versión 0, nunca se modifican)
│   │   ├── Residential-Building-Data-Set.xlsx
│   │   ├── residential_building_original.csv
│   │   └── FUENTE.md              # origen, licencia, huellas SHA-256
│   └── datos_limp/                # datos limpios (versión 1.0.0)
│       ├── residential_building_limpio.csv
│       ├── diccionario_datos.csv
│       ├── reporte_calidad.md
│       └── manifiesto_versiones.json
├── fuentes/
│   ├── entrena.ipynb              # notebook de Colab: entrenamiento y evaluación (ejecutado)
│   ├── datos_prep.py              # funciones de obtención, limpieza, estandarización y versionado
│   ├── train.py                   # entrenamiento, GridSearchCV, comparación estadística y MLflow
│   └── visualizaciones.py         # análisis exploratorio (patrones y anomalías)
├── resultados/                    # métricas, pruebas estadísticas, export de runs y figuras
├── evidencias_mlflow/             # capturas del panel de MLflow
├── tests/test_actividad5.py       # 12 pruebas automatizadas
├── requirements.txt
├── README.md                      # esta guía
└── CHANGELOG.md                   # versiones de datos y código + reporte final
```

## 3. Reproducción

### Opción A — Google Colab
`fuentes/entrena.ipynb` se ejecuta completo en Colab con CPU: clona este repositorio, instala dependencias, limpia los datos, entrena, registra en MLflow, levanta el panel de MLflow y exporta los resultados (`resultados_colab.zip`). El entrenamiento completo tarda entre 30 y 60 minutos porque la validación anidada repite el grid search 15 veces.

### Opción B — Terminal

```bash
git clone https://github.com/Ignacio-Castillo-C/voreko-ia-incidencias.git && cd voreko-ia-incidencias/Actividad5
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python fuentes/datos_prep.py        # obtiene el original (UCI o espejos) y genera datos_limp/
python fuentes/train.py             # experimento completo y registro en MLflow (≈ 28 min en CPU)
python -m pytest -q tests           # 12 pruebas
mlflow ui --backend-store-uri sqlite:///mlflow.db      # panel en http://127.0.0.1:5000
```

`python fuentes/train.py --rapido --repeticiones 1` ejecuta una versión reducida (prueba de humo) en menos de un minuto.

**Reproducibilidad garantizada por:** semilla fija (42) en particiones y modelos; hold-out estratificado; preprocesamiento ajustado dentro de cada fold; huellas SHA-256 de los datos verificadas por el código y las pruebas; y versiones de librerías, *commit* de Git y versión de datos registrados como etiquetas en cada run de MLflow.

## 4. Decisiones de preparación de datos

| Paso | Criterio | Resultado |
|---|---|---|
| Encabezados | Las 2 filas de encabezado se aplanan con nombres únicos por rezago | 109 columnas únicas (76 códigos estaban repetidos) |
| Tipos | Conversión numérica con conteo de valores no convertibles | 94 decimales y 15 enteras; 0 fallas |
| Nulos | Sin objetivo → se elimina; en predictoras → mediana | 0 nulos (sin cambios) |
| Duplicados | (1) filas exactas; (2) misma clave de proyecto (8 campos físicos, financieros y de fecha), conservando el registro con costo; (3) casi duplicados con diferencias < 0.5 %; (4) columnas duplicadas o constantes | 0 en los cuatro criterios; la función se prueba con duplicados inyectados |
| Reglas de negocio | Trimestres 1–4, localidad 1–20, valores > 0, área construida ≥ terreno, costo total = m² × área, término = inicio + duración | 0 violaciones |
| Estandarización | Años persas → 4 dígitos y gregoriano aproximado; montos → millones de riales | 51 columnas monetarias convertidas |
| Anomalías | 3·IQR sobre log(costo m²) y log(real/preliminar): se marcan, no se eliminan | 12 desviaciones atípicas (proyectos de 1373–1376 con costo preliminar ≤ 0.6 MIRR/m²) |
| Fuga de información | Se excluye lo que sólo se conoce al terminar la obra, y la identidad redundante | 5 columnas excluidas; quedan 105 predictoras |
| Escalado y codificación | `StandardScaler` en numéricas y *one-hot* en localidad y trimestre, **dentro** del pipeline | Sin fuga entre folds |
| Objetivo | log(costo m²) por asimetría 1.26 → −0.41; las métricas se calculan en MIRR/m² | — |

## 5. Entrenamiento y ajuste

| Modelo | Grid (GridSearchCV, KFold 5, refit por RMSE) | Mejor configuración |
|---|---|---|
| Ridge | `alpha` ∈ {0.01, 0.1, 1, 10, 100, 1000} × rezagos ∈ {todos, r1} — 12 | `alpha=100`, todos los rezagos |
| Random Forest (300 árboles) | `max_depth` ∈ {None, 10} × `min_samples_leaf` ∈ {1, 3, 5} × `max_features` ∈ {0.33, 1.0} × rezagos — 24 | `max_depth=10`, `min_samples_leaf=3`, `max_features=1.0`, r1 |
| XGBoost (400 árboles, subsample 0.8) | `learning_rate` ∈ {0.03, 0.1} × `max_depth` ∈ {2, 4} × `colsample_bytree` ∈ {0.5, 0.8} × `min_child_weight` ∈ {1, 5} × rezagos — 32 | `learning_rate=0.1`, `max_depth=2`, `colsample_bytree=0.8`, `min_child_weight=5`, r1 |

La comparación entre modelos se hace con **validación cruzada anidada** (5×3 externa, 5-fold interna), que no reutiliza los mismos datos para elegir hiperparámetros y para medir. Las diferencias se prueban con la *t* pareada corregida de Nadeau y Bengio (2003), Wilcoxon y Friedman, con ajuste de Holm.

## 6. Resultados y conclusiones

| Modelo | RMSE (anidada) | MAE | R² | MAPE | Prueba: MAE / RMSE / R² |
|---|---|---|---|---|---|
| **XGBoost** | **0.333** | **0.189** | **0.953** | **9.6 %** | 0.188 / 0.429 / 0.941 |
| Random Forest | 0.355 | 0.213 | 0.947 | 11.1 % | 0.240 / 0.477 / 0.927 |
| Regla presupuestal | 0.416 | 0.262 | 0.928 | 13.4 % | 0.294 / 0.440 / 0.938 |
| Ridge | 0.573 | 0.282 | 0.785 | 16.0 % | 0.393 / 1.442 / 0.331 |

- **XGBoost** es el mejor en todas las métricas y supera a la regla presupuestal en los 15 folds (Wilcoxon p = 0.0004), aunque con la *t* corregida, más conservadora, p = 0.09. Su ventaja sobre Random Forest **no es estadísticamente significativa**; se elige por menor MAE y MAPE, menor variabilidad entre folds y menor tiempo de ajuste.
- **Ridge** extrapola mal las obras fuera del rango de su época (falla en 3 de 15 folds y en prueba): no es apto como modelo de alertas sin acotar sus predicciones.
- El modelo quedó registrado como `voreko_costo_real_m2` (alias `candidato`). Antes de usarlo en VOREKO se requiere validarlo con obras reales en México.

El detalle de versiones está en `CHANGELOG.md` y el análisis completo en el reporte técnico.

## 7. MLflow

- **Experimento:** `VOREKO-A5-costo-real-m2`, con los runs `01_datos_v1.0.0` (dataset y calidad), `02_gridsearch_<modelo>` (con un run hijo por cada combinación), `03_referencia_<modelo>` y `04_cv_anidada_comparacion` (RMSE por fold como serie y valores p).
- **Se registra en cada run:** parámetros, métricas de CV y de prueba, artefactos (resultados del grid, figuras, reporte de calidad, diccionario), el dataset (`mlflow.log_input`), el modelo con firma y ejemplo de entrada, y las etiquetas `git_commit`, `version_datos` y huellas SHA-256.
- **Model Registry:** `voreko_costo_real_m2`, alias `candidato`.
- La base `mlflow.db` y la carpeta `mlruns/` no se publican (`.gitignore`) porque se regeneran con `train.py`. La evidencia está en `evidencias_mlflow/` y en `resultados/mlflow_runs.csv`.

## 8. Limitaciones

El dataset es de otro país, moneda y periodo inflacionario; sólo tiene 372 obras y algunas localidades con 4 proyectos; además, los costos preliminares bajos tienen poca resolución. Los resultados demuestran el método, no el desempeño esperado en obras de VOREKO.
