# VOREKO IA — Actividad 6

**Evaluación y validación del sistema inteligente de alertas de sobrecosto**

Master de Inteligencia Artificial · Universidad Tecmilenio · Gestión de Proyectos de Inteligencia Artificial
José Ignacio Castillo Camacho · AL07161745

Clasificación binaria del riesgo de sobrecosto (`riesgo_sobrecosto`: desviación real del costo por m² mayor a 50 %) con el *Residential Building Data Set* (UCI, id 437), a partir de los datos y componentes de la Actividad 5.

## Estructura

```
Actividad6/
├── fuentes/
│   ├── 02_VOREKO_Clasificacion_Sobrecostos.ipynb
│   └── voreko_a6.py
├── datos/datos_clasif/
│   ├── residential_building_clasificacion.csv
│   └── manifiesto_clasificacion.json
├── tests/test_actividad6.py
├── requirements.txt
└── README.md
```

## Componentes reutilizados de la Actividad 5

- `Actividad5/datos/datos_limp/residential_building_limpio.csv` (versión 1.0.0)
- `Actividad5/fuentes/datos_prep.py`
- `Actividad5/fuentes/train.py`

## Ejecución

```bash
pip install -r Actividad6/requirements.txt
python -m pytest -q Actividad6/tests
cd Actividad6/fuentes
jupyter nbconvert --to notebook --execute --inplace 02_VOREKO_Clasificacion_Sobrecostos.ipynb
cd .. && mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Experimento de MLflow: `VOREKO-A6-clasificacion-sobrecostos`. Modelo registrado: `voreko_riesgo_sobrecosto`.
