# Reporte de calidad de datos — Residential Building (UCI 437)

Versión de datos 1.0.0 · generado por `fuentes/datos_prep.py` (salida determinista)

| Paso | Hallazgo | Acción |
|---|---|---|
| 1. Encabezados | 2 filas de encabezado; 76 códigos repetidos por rezagos | Aplanado a 109 nombres únicos (`*_r1…r5`) |
| 2. Tipos | Todas las columnas como texto: {'object': 109} | Conversión numérica: {'float64': 94, 'int64': 15}; no convertibles: 0 |
| 3. Nulos | 0 valores nulos | Filas sin objetivo eliminadas: 0; celdas imputadas: 0 |
| 4. Duplicados | Exactos: 0; por clave de proyecto: 0; en grupos casi duplicados: 0 | Columnas duplicadas eliminadas: 0; constantes: 0 |
| 5. Reglas de negocio | Violaciones: {'trimestres_en_1_4': 0, 'localidad_en_1_20': 0, 'areas_y_costos_positivos': 0, 'area_construida_mayor_o_igual_terreno': 0, 'costo_total_igual_m2_por_area': 0, 'fin_igual_inicio_mas_duracion': 0, 'duracion_positiva': 0} | Filas eliminadas: 0 |
| 6. Estandarización | Años persas de 2 dígitos; montos en 10 mil, 10 millones o riales | Años a 4 dígitos ([1372, 1390]) y gregoriano aprox. ([1993, 2011]); 51 columnas a millones de riales |
| 7. Anomalías | Costo m2 atípico: 0; desviación atípica: 12 (mediana de razón 1.41) | Se marcan, no se eliminan |

Columnas duplicadas eliminadas: ninguna
Columnas constantes eliminadas: ninguna

## Variables excluidas del modelo

- `precio_venta_real_m2`: Salida del proyecto: sólo se conoce al vender (fuga de información)
- `anio_fin`: Fecha de término: posterior al momento de estimar (fuga de información)
- `trimestre_fin`: Fecha de término: posterior al momento de estimar (fuga de información)
- `costo_prelim_total`: Redundante: costo_prelim_m2 × area_construida_m2 (identidad verificada)
- `indice_tiempo_fin`: Derivada de la fecha de término (fuga de información)

Predictoras disponibles al inicio del proyecto: **105**.

## Huellas SHA-256

- archivo_original_xlsx: `56ccbc261b0f1ceb50e10e3f0cd02794f8cdd16a3ffb9788eeb505e706642655`
- contenido_original_hoja_Data: `7e736374a647e1f64f930dab167dcf93bf8018c9d17b8a5891e5ce257b4671cd`
- csv_limpio: `47e7de7fe5d195f1d1852bce258d43333044206db340d88128b069ed3fb5424b`
