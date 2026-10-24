# VOREKO IA

Master en Inteligencia Artificial · Universidad Tecmilenio — *Gestión de Proyectos de IA*.
Autor: José Ignacio Castillo Camacho.

## Actividad 5 — Datos, entrenamiento y registro en MLflow

Estimación temprana del costo real de construcción por m² (dataset UCI *Residential Building*) con Ridge, Random Forest y XGBoost, GridSearchCV, validación cruzada anidada, comparación estadística y seguimiento en MLflow.

| Entregable | Ubicación |
|---|---|
| Datos originales y limpios | [`Actividad5/datos/`](Actividad5/datos/) |
| Funciones de limpieza y script de entrenamiento | [`Actividad5/fuentes/datos_prep.py`](Actividad5/fuentes/datos_prep.py), [`Actividad5/fuentes/train.py`](Actividad5/fuentes/train.py) |
| Cuaderno (ejecutado) | [`Actividad5/fuentes/entrena.ipynb`](Actividad5/fuentes/entrena.ipynb) |
| README / CHANGELOG | [`Actividad5/README.md`](Actividad5/README.md), [`Actividad5/CHANGELOG.md`](Actividad5/CHANGELOG.md) |
| Evidencia de MLflow | [`Actividad5/evidencias_mlflow/`](Actividad5/evidencias_mlflow/) |

## Actividad 6 — Evaluación y validación del sistema de alertas de sobrecosto

Clasificación binaria del riesgo de sobrecosto con regresión logística, Random Forest, XGBoost y la regla derivada de la Actividad 5; matriz de confusión, ROC/AUC, ajuste de umbral, validación cruzada anidada, comparación estadística y diseño A/B.

| Entregable | Ubicación |
|---|---|
| Datos para clasificación | [`Actividad6/datos/datos_clasif/`](Actividad6/datos/datos_clasif/) |
| Funciones | [`Actividad6/fuentes/voreko_a6.py`](Actividad6/fuentes/voreko_a6.py) |
| Cuaderno | [`Actividad6/fuentes/02_VOREKO_Clasificacion_Sobrecostos.ipynb`](Actividad6/fuentes/02_VOREKO_Clasificacion_Sobrecostos.ipynb) |
| Pruebas | [`Actividad6/tests/test_actividad6.py`](Actividad6/tests/test_actividad6.py) |
| README | [`Actividad6/README.md`](Actividad6/README.md) |


## 8. Limitaciones

El dataset es de otro país, moneda y periodo inflacionario; sólo tiene 372 obras y algunas localidades con 4 proyectos; además, los costos preliminares bajos tienen poca resolución. Los resultados demuestran el método, no el desempeño esperado en obras de VOREKO.

