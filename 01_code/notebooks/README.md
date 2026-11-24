# Cuadernos del análisis (documentación)

Estos cinco cuadernos cuentan, con texto, ecuaciones y salidas guardadas, el análisis que llevó al envío final del equipo
(LightGBM con Catastro y texto, ×1,08: 150,79 de error absoluto medio en la tabla privada de Kaggle, segundo de seis
equipos). Son documentación: se escribieron para la carpeta de trabajo original (`scripts/`, `stores/`, `Kaggle/`) y no
se corren desde este repositorio. El pipeline reproducible son los scripts numerados de `01_code/`, que corre el script
maestro `01_code/00_rundirectory.R`; cada sección de los cuadernos tiene su script equivalente.

| Cuaderno | Qué contiene | Script del pipeline |
|---|---|---|
| `00_entorno_y_conexiones.ipynb` | Sistema, librerías y versiones, carpetas, datos de Kaggle y conexiones externas (OpenStreetMap, Earth Engine, Nominatim) | `requirements.txt`; descargas en `01` y `02` |
| `01_analisis_espacial.ipynb` | Mapas en la coordenada exacta, correlaciones, I de Moran, correlograma, LISA, precisión de las coordenadas (secciones 1 a 8); texto como datos: limpieza, ortografía con BETO, diccionarios con negación, cantidades, DTM, TF-IDF, LSA, LDA y embeddings de BETO (secciones 9 y 10) | `05_analisis_espacial_descriptivo.py`, `06_texto_como_datos.py` |
| `02_modelo_pronostico.ipynb` | Las 162 variables por bloque, puntaje de texto fuera de pliegue, especificación del LightGBM, validación, ajuste ×1,08, gemelos de texto y generación de los envíos (comprueba su SHA-256) | `07` a `09` |
| `03_comparacion_familias_validacion.ipynb` | Seis esquemas de validación, siete especificaciones de las seis familias, bootstrap pareado, relación con Kaggle y pruebas descartadas | `07`, `08` y `10` |
| `04_errores_negocio.ipynb` | Sesgo por cuantil de la predicción, compresión de precios altos, avisos fuera de rango, error mayor o menor que el precio, SHAP, simulación de ofertas y precisión necesaria | `11` |

Diferencias con el pipeline: los cuadernos 02 a 04 comparan sus cifras con archivos de la carpeta de las presentaciones
(`document/presentaciones/datos/`), que en el repositorio regeneran los scripts 12 a 19 en `00_data/processed/laminas/`;
el cuaderno 03 muestra además la validación del paso M4 (BETO ajustado y la ablación con Elastic Net de texto), que no
entró al modelo final ni al pipeline.

En las salidas del cuaderno 00 se reemplazaron las rutas locales del equipo de cómputo por `<carpeta del proyecto>` y
`<python de miniconda>`; no se cambió nada más.
