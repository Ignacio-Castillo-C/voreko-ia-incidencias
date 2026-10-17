# CHANGELOG — VOREKO IA · Actividad 5

Registro de cambios de **datos** y **código**, con el reporte final de resultados.
Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y [versionado semántico](https://semver.org/lang/es/).

- **Datos:** `MAYOR` = cambia el contenido o el significado de las filas o columnas · `MENOR` = se agregan columnas derivadas · `PARCHE` = cambios de documentación o formato que no alteran valores.
- **Código:** `MAYOR` = cambia el protocolo de evaluación · `MENOR` = nuevos modelos o etapas · `PARCHE` = correcciones.
- Cada versión de datos se identifica por su carpeta (`datos_ini/` = versión 0, `datos_limp/` = versión 1.0.0) y su huella SHA-256 en `datos/datos_limp/manifiesto_versiones.json`. Todos los runs de MLflow registran `git_commit`, `version_datos` y las huellas.

---

## [1.0.0] — 2026-10-14 · Reporte final

### Resultados

Experimento completo con semilla fija (42), ejecutado desde `fuentes/entrena.ipynb` en Google Colab (CPU, 2 núcleos; Python 3.13, scikit-learn 1.6.1, XGBoost 3.4.1, MLflow 3.17.1) el 2026-10-17, sobre el commit `7186446` del repositorio. Duración: 52 min; 75 runs registrados.

**Validación cruzada anidada** (externa RepeatedKFold 5×3 = 15 folds; interna GridSearchCV 5-fold), costo real en MIRR/m²:

| Modelo | RMSE | MAE | R² | MAPE |
|---|---|---|---|---|
| **XGBoost** (seleccionado) | **0.333 ± 0.143** | **0.189** | **0.953** | **9.6 %** |
| Random Forest | 0.356 ± 0.129 | 0.212 | 0.947 | 11.1 % |
| Regla presupuestal (referencia) | 0.416 ± 0.140 | 0.262 | 0.928 | 13.4 % |
| Ridge | 0.573 ± 0.391 | 0.282 | 0.785 | 16.0 % |

**Prueba (hold-out del 20 %, 75 proyectos):** XGBoost MAE 0.188, RMSE 0.429, R² 0.941, MAPE 7.8 %; la regla presupuestal obtuvo MAE 0.294, RMSE 0.440, R² 0.938, MAPE 14.1 %.

**Comparación estadística** (p ajustadas por Holm): XGBoost superó a la regla presupuestal en los 15 folds (Wilcoxon p = 0.0004; t corregida de Nadeau y Bengio p = 0.092). Entre XGBoost y Random Forest no hay diferencia significativa (p = 1.00 y 0.28). Friedman entre los tres modelos ajustados: χ² = 5.20, p = 0.074.

**Decisión:** XGBoost (`max_depth=2`, `learning_rate=0.1`, `colsample_bytree=0.8`, `min_child_weight=5`, sólo rezago 1) se registra como `voreko_costo_real_m2` con el alias `candidato`, **no** `produccion`: falta validarlo con obras reales de VOREKO.

### Agregado
- `fuentes/train.py`: GridSearchCV de Ridge, Random Forest y XGBoost con objetivo en log; referencias de mediana y regla presupuestal; validación anidada; prueba t corregida, Wilcoxon, Friedman y Holm; registro en MLflow de parámetros, métricas, dataset, artefactos, modelos con firma y Model Registry.
- `fuentes/visualizaciones.py`: 5 figuras exploratorias que devuelven las cifras que las sustentan.
- `fuentes/entrena.ipynb`: notebook de Colab con todo el flujo, ejecutado con salidas.
- `evidencias_mlflow/`: 9 capturas del panel de MLflow de la ejecución en Colab (experimento, runs, detalle con parámetros, métricas y etiquetas, artefactos, comparación, RMSE por fold y Model Registry).
- Alerta derivada para VOREKO (costo estimado > 1.5 × preliminar), evaluada en prueba: recall 0.692, precisión 0.783, ROC-AUC 0.859.
- `tests/test_actividad5.py`: 12 pruebas de limpieza, reglas de negocio, fuga de información y estadística.

### Hallazgos que guiaron decisiones
- Ridge tiene un buen RMSE en la búsqueda (0.433), pero falla en 3 de 15 folds anidados y en prueba (RMSE 1.44): al ser log-lineal, extrapola de forma multiplicativa para obras fuera del rango de su época. Un solo proyecto (RB-059: 5.8 MIRR/m² de costo preliminar en 1380) recibió una predicción de 21.2 cuando el real fue 9.0. Los modelos de árboles no extrapolan y no tienen este problema.
- La búsqueda eligió `rezagos = r1` en Random Forest y XGBoost, y `todos` en Ridge: la regularización L2 aprovecha la información redundante, mientras que los árboles funcionan mejor con menos variables casi colineales.

## Datos [1.0.0] — 2026-10-14

### Cambiado
- Encabezado de 2 niveles aplanado a 109 nombres únicos (`*_r1 … *_r5` para los 5 rezagos de 19 indicadores).
- 109 columnas de texto convertidas a numéricas (94 decimales, 15 enteras; 0 valores no convertibles).
- Años persas de 2 dígitos → 4 dígitos (1372–1390) y año gregoriano aproximado (1993–2011); índice temporal en trimestres.
- 51 columnas monetarias en unidades mezcladas (10 mil, 10 millones o riales) → millones de riales (MIRR).

### Verificado (sin cambios en filas)
- 0 nulos, 0 duplicados exactos, 0 duplicados por clave de proyecto, 0 casi duplicados, 0 columnas duplicadas o constantes.
- 7 reglas de negocio con 0 violaciones (trimestres 1–4, localidad 1–20, valores positivos, área construida ≥ terreno, costo total = costo m² × área, término = inicio + duración, duración > 0).

### Agregado
- `razon_real_vs_preliminar` y banderas `anomalia_costo_m2` y `anomalia_razon_costo` (12 desviaciones atípicas marcadas con 3·IQR en log; no se eliminan).
- Diccionario de datos, reporte de calidad y manifiesto con huellas SHA-256.

### Excluido del modelo (fuga de información o redundancia)
- `precio_venta_real_m2`, `anio_fin`, `trimestre_fin`, `indice_tiempo_fin` (posteriores al inicio de la obra) y `costo_prelim_total` (identidad con costo m² × área).

## Datos [1.0.0 — corrección de documentación] — 2026-10-14
### Corregido
- El reporte de calidad incluía la hora de generación y cambiaba en cada ejecución sin que cambiaran los datos. Ahora la salida es determinista (el CSV limpio no cambió: misma huella SHA-256).

## Datos [0] — 2026-10-14
### Agregado
- Archivo original `Residential-Building-Data-Set.xlsx` (UCI 437, CC BY 4.0) y copia fiel en CSV. Huella del contenido de la hoja `Data`: `7e736374…4671cd`. Verificado idéntico en dos espejos públicos.

---

## Pendiente (siguiente versión)
- Validar con obras reales de VOREKO, anonimizadas y en pesos mexicanos; recalibrar la regla de alerta.
- Automatizar `datos_prep.py → train.py → pruebas` en CI (GitHub Actions) y registrar el servidor de MLflow compartido.
- Monitorear la deriva del costo por trimestre y el error por localidad antes de promover el modelo a `produccion`.
