# Datos originales (versión 0 — sin modificar)

- **Dataset:** Residential Building Data Set
- **Fuente oficial:** UCI Machine Learning Repository, id 437 — https://archive.ics.uci.edu/dataset/437/residential+building+data+set
- **DOI:** 10.24432/C5S896 · **Licencia:** CC BY 4.0 · **Creador:** M. Rafiei (2015)
- **Artículo de referencia:** Rafiei, M. H., y Adeli, H. (2016). A novel machine learning model for estimation of sale prices of real estate units. *Journal of Construction Engineering and Management, 142*(2).
- **Contenido:** 372 edificios residenciales (Teherán, Irán), 8 variables físicas y financieras del proyecto, 19 indicadores económicos en 5 rezagos y 2 salidas (precio de venta y costo de construcción).

## Obtención

La red del entorno de trabajo no tenía acceso a archive.ics.uci.edu, así que el archivo se descargó de dos espejos públicos en GitHub y se comprobó que la hoja `Data` es idéntica, celda por celda, en ambos:

1. https://raw.githubusercontent.com/rita316/TehranRealEstateProfitability/master/Residential-Building-Data-Set.xlsx (archivo usado)
2. https://raw.githubusercontent.com/trisa-biswas/Residential-Building-Price-Prediction/master/data/Residential-Building-Data-Set.xlsx

`fuentes/datos_prep.py` intenta primero la fuente oficial de UCI y, si no está disponible, recurre a los espejos.

| Archivo | Descripción | SHA-256 |
|---|---|---|
| `Residential-Building-Data-Set.xlsx` | Archivo original (hojas `Data` y `Descriptions`) | `56ccbc261b0f1ceb50e10e3f0cd02794f8cdd16a3ffb9788eeb505e706642655` |
| `residential_building_original.csv` | Copia fiel de la hoja `Data` en texto plano, con sus dos filas de encabezado | contenido: `7e736374a647e1f64f930dab167dcf93bf8018c9d17b8a5891e5ce257b4671cd` |

Estos archivos **no se modifican nunca**. Cualquier cambio se aplica en `datos/datos_limp/` mediante `fuentes/datos_prep.py`.
