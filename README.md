# Problem Set 3: Making Money with ML?

Big Data y Machine Learning para Economía Aplicada (MECA 4107, 2026-20) · Universidad de los Andes ·
Equipo 02: Jhon Sebastián Vela Salcedo, Harold Santiago Aguilera Mulfo y César Augusto Vallejo Quiroga.

## Objetivo

Una empresa que compra y vende vivienda quiere comprar la mayor cantidad posible de inmuebles en
Chapinero (Bogotá) gastando lo menos posible. Este repositorio construye modelos que predicen el precio de
venta de cada inmueble con datos de Properati: se entrena con anuncios del resto de Bogotá y se predice en
Chapinero, una zona que no aparece en el entrenamiento. Como sobrestimar un precio destruye capital y
subestimarlo solo hace perder un negocio, además del error medio (MAE, la métrica de Kaggle) se analiza
hacia qué lado se equivoca el modelo.

> Pregunta. ¿Cuál es el mejor modelo para predecir precios de vivienda en Chapinero y se puede confiar
> en él para comprar?
>
> Respuesta. El mejor modelo es un boosting de árboles: un LightGBM con pérdida cuadrática que usa 161
> variables de la vivienda, del texto del aviso, del entorno (OpenStreetMap y Google Earth Engine) y de
> Catastro Bogotá, más un puntaje de texto calculado fuera de pliegue, con un ajuste final que corrige la
> compresión de las predicciones hacia la media: 654 + 1,08 (ŷ − 654). Obtuvo 150,79 millones de pesos de error
> absoluto medio en la tabla privada de Kaggle (envío 02), el segundo lugar entre seis equipos, y gana a las
> otras cinco familias con la validación aleatoria, con la espacial por clústeres y con la que imita el reto
> (entrenar sin Chapinero). Sirve para ordenar y filtrar inmuebles, pero no para ofertar sin una tasación: en la
> validación que imita el reto, 26,7 % de las predicciones queda más de 10 % por encima del precio real y una
> política de ofertas automáticas con margen de 10 % (ofrecer el 90 % de la predicción a dueños que aceptan hasta
> un 10 % menos de su precio) pierde dinero, porque el dueño acepta justo cuando el modelo sobrestima. Para que esa política gane, el error absoluto mediano tendría que bajar de 12,9 % a
> 9,1 %.

## Resultados

Error absoluto medio (MAE, millones de pesos) de las predicciones fuera de pliegue. Todas las familias reciben las
mismas 161 variables y el mismo puntaje de texto (`02_outputs/tables/familias_mae_por_esquema.csv`, scripts 07 y 10).
Validación estándar: CV aleatoria de 5 pliegues. Validaciones espaciales: clústeres k-means de las coordenadas (10
pliegues, como `spatialsample::spatial_clustering_cv`) y Chapinero fuera (se entrena sin Chapinero y se predicen sus
307 avisos de entrenamiento, como en el reto). La última columna es el mejor envío de cada familia del equipo en
Kaggle (tabla de envíos, más abajo); salvo en el boosting, esos envíos usaron otras variables, así que no son la
misma especificación de las columnas de validación.

| Familia | Especificación validada | CV aleatoria | Clústeres k-means | Chapinero fuera | Mejor envío en Kaggle (público / privado) |
|---|---|---|---|---|---|
| Boosting | LightGBM, pérdida L2, parada temprana espacial (el modelo final) | 78,8 | 125,0 | 143,8 | 152,48 / 150,79 (`envio_02_lightgbm_con_catastro_x108`) |
| Boosting | CatBoost, pérdida RMSE, profundidad 8 | — | 124,1 | 150,1 | (el mismo de arriba) |
| Red neuronal | Red con atención a los 20 vecinos de entrenamiento | 86,8 | 128,8 | 176,7 | 180,36 / 184,43 (`nn_clase_2x10_D_coordenadas`) |
| Random Forest | 300 árboles, 30 % de las variables por corte, hojas de 5 avisos | 91,6 | 131,8 | 158,6 | 204,3 / 206,3 (`RF_fe_depth_12_leaf_5`) |
| CART | Profundidad elegida con una partición interna espacial | 117,9 | 150,9 | 174,1 | 240,7 / 239,9 (`CART_fe_depth_6_leaf_40`) |
| Elastic Net | Mezcla L1 y L2 elegida con particiones internas espaciales | 124,5 | 149,8 | 185,2 | sin envío |
| Regresión lineal | Mínimos cuadrados con las mismas variables | 124,7 | 150,0 | 185,7 | 244,9 / 253,6 (`LR_fe_nr_1`) |

Lectura. El boosting gana con las tres validaciones (LightGBM con la aleatoria y con Chapinero fuera; CatBoost, por
0,9 millones, con los clústeres) y la ventaja de LightGBM sobre cada una de las demás familias en Chapinero fuera es
significativa (bootstrap pareado de 10.000 réplicas, `familias_bootstrap_chapinero.csv`). La CV aleatoria promete
casi la mitad del error de Kaggle: valida avisos cuya copia, o cuyo vecino en la misma coordenada, está en el
entrenamiento. Chapinero fuera es la validación que mejor anticipa Kaggle: queda entre 7 y 10 millones por debajo de la
tabla privada en los tres envíos del modelo final y los ordena igual que ella (`calibracion_kaggle_envios.csv`). Todas las
familias subestiman Chapinero; el ajuste ×1,08 reduce el sesgo del LightGBM en Chapinero fuera de −52 a −41 millones y
su MAE de 143,8 a 140,8 (`modelo_factor_estiramiento.csv`).

## Cómo reproducir

1. Clone el repositorio. Los datos de Kaggle están en `00_data/` (`train.csv`, `test.csv` y
   `submission_template.csv`) y los intermedios pequeños que hacen falta, en `00_data/processed/`.
2. Instale las librerías: Python 3.13 con `pip install -r requirements.txt` (PyTorch en CPU basta; con GPU:
   `pip install torch --index-url https://download.pytorch.org/whl/cu128`) y R 4.6.1 con los paquetes de
   `requirements_R.txt`.
3. Abra R en la raíz del repositorio y corra una sola instrucción:

```r
source("01_code/00_rundirectory.R")
```

El script maestro carga el entorno común (`00_setup.R`: semilla, rutas y paquetes), corre en orden numérico
todos los scripts `NN_*.R` y `NN_*.py` de `01_code/` y guarda las versiones de R, Python y sus paquetes en
`02_outputs/session_info.txt`. Para correr solo un tramo: `Sys.setenv(PS3_DESDE = "5", PS3_HASTA = "8")`
antes del `source`. Cada script de Python también corre solo desde la raíz, por ejemplo
`python 01_code/09_modelo_final_envios.py`.

Qué corre sin internet ni GPU. Por defecto, todo: el pipeline completo corre sin internet, sin credenciales y sin
GPU, en unos 45 minutos en un portátil. Se recalculan la base de texto, las predicciones del modelo de una
semilla para los valores SHAP, todas las tablas y figuras, los tres envíos a Kaggle (con la comprobación de que son
idénticos a los subidos) y las dos presentaciones. Los pasos que necesitan descargas (OpenStreetMap, Earth Engine,
Catastro), la GPU (BETO) o mucho cómputo (la validación cruzada de las especificaciones y los modelos finales) leen
sus resultados guardados en `00_data/processed/`. Para recalcularlos se usa la variable de entorno `PS3_RECALCULAR`
con los números de los scripts: en R, `Sys.setenv(PS3_RECALCULAR = "07,09")` antes del `source`; en una terminal,
`PS3_RECALCULAR=07 python 01_code/07_validacion_cruzada_familias.py`.

| `PS3_RECALCULAR` | Qué recalcula | Requiere | Tiempo aproximado |
|---|---|---|---|
| `01` | Extracto de OpenStreetMap (BBBike), objetos de OSM, localidades (Nominatim) y capas de Earth Engine | Internet, una cuenta de Earth Engine (`earthengine authenticate`) y un proyecto de Google Cloud registrado en Earth Engine en `PS3_GEE_PROYECTO` | 15 minutos |
| `02` | Capas por manzana de Catastro Bogotá (servicio ArcGIS público) | Internet | 5 minutos |
| `03`, `04` | Variables de entorno de OSM y de Catastro | Las descargas de `01` y `02` | 2 minutos cada uno |
| `06` | Elección de K del LDA, corrección ortográfica y embeddings con BETO | Internet (modelo de Hugging Face); GPU recomendada | 35 minutos con GPU |
| `07` | Validación cruzada de las nueve especificaciones | CPU | 2 horas con 6 procesos |
| `08` | kNNDM y kriging de residuos (15 LightGBM) | CPU | 10 minutos |
| `09` | Modelos finales: LightGBM con 5 semillas y CatBoost con 3 | CPU | 25 minutos |
| `02,16,17` | Datos de las láminas que usan Catastro | Internet | 10 minutos |
| `20` | Imagen didáctica del boosting con una variable | CPU | 1 minuto |
| `1` | Todo lo anterior | Todo lo anterior | unas 4 horas |

Exportar las presentaciones a PDF es opcional y usa PowerPoint por COM (solo Windows): `PS3_EXPORTAR_PDF=1` activa el
script 23; sin esa variable el script avisa y termina sin error. Los PDF finales ya están en `03_slides/`.

Una descarga nueva puede diferir levemente de la usada en los envíos: BBBike actualiza el extracto de OpenStreetMap
cada semana y Nominatim puede cambiar los polígonos. Las variables de entorno y de Catastro se reconstruyeron con los
scripts 03 y 04 a partir de las descargas originales y coinciden exactamente con las guardadas. La base de texto que se
regenera en cada corrida, los datos de las láminas y sus imágenes coinciden con los de la versión entregada, y los
envíos que genera el script 09 coinciden byte a byte con los subidos a Kaggle, también al reentrenar los modelos
finales con `PS3_RECALCULAR=09`.

## Estructura

```
BDML-PS03/
├── README.md
├── requirements_R.txt        versiones de los paquetes de R
├── requirements.txt          versiones de los paquetes de Python
├── 00_data/                  datos de Kaggle: train.csv, test.csv y submission_template.csv
│   └── processed/            intermedios (los que se regeneran no se versionan: ver .gitignore)
│       ├── osm/              objetos de OpenStreetMap
│       ├── texto/            listas de stopwords (DNP y NLTK), auditoría de ortografía y componentes de BETO
│       ├── modelos/          predicciones fuera de pliegue y finales de cada especificación
│       └── laminas/          cifras de las presentaciones
├── 01_code/
│   ├── 00_rundirectory.R     script maestro: reproduce todo
│   ├── 00_setup.R            entorno común de los scripts de R
│   ├── NN_*.R, NN_*.py       scripts numerados, uno por tarea (tabla de abajo)
│   ├── functions/            funciones reutilizables (tabla de abajo)
│   └── notebooks/            cuadernos del análisis con sus salidas (documentación; el maestro no los corre)
├── 02_outputs/
│   ├── figures/              figuras de análisis (.png) y laminas/, las imágenes de las presentaciones
│   ├── tables/               tablas (.csv)
│   └── predictions/          envíos a Kaggle (.csv) y su registro
└── 03_slides/                best_equipo_02 y compare_equipo_02 (.pptx y .pdf finales, .tex), preamble.tex, numbers.tex, tables/
```

| Módulo de `01_code/functions/` | Qué contiene |
|---|---|
| `setup_python.py` | Raíz del repositorio como directorio de trabajo, semilla 2026 y carpetas `DIR_*` |
| `entorno_ps3.py` | Control del recálculo (`PS3_RECALCULAR`), subcarpetas de `00_data/processed/` y guardado de figuras y tablas |
| `flujo_final.py` | Base del modelo final: 161 variables por aviso, pesos adversariales, puntaje ridge de texto fuera de pliegue, pliegues de validación, predicciones fuera de pliegue, ajuste ×1,08 y gemelos de texto |
| `validacion_clase.py` | Pliegues de la lectura de validación cruzada espacial: aleatoria, bloques de 2 km, bloques con buffer de 300 m, localidades y clústeres k-means, con su partición interna |
| `modelo_utils.py` | Ajuste y predicción de las familias: regresión lineal, Elastic Net, CART, Random Forest, LightGBM, CatBoost y la red con atención |
| `knndm_kriging.py` | Pliegues kNNDM, variograma exponencial y kriging simple de residuos |
| `laminas_entorno.py`, `laminas_comunes.py`, `ppt_lib.py` | Carpetas de las láminas, láminas compartidas por las dos presentaciones y utilidades de python-pptx |

## Scripts y salidas

Cada fila dice qué hace el script, qué archivos produce, qué contiene cada archivo y en qué lámina se usa: `best` es
`03_slides/best_equipo_02` (Best Model Deep Dive, 31 láminas) y `compare` es `03_slides/compare_equipo_02`
(Algorithm Comparison, 22 láminas); el número es el de la lámina. Salvo que se diga otra cosa, las figuras están en
`02_outputs/figures/`, las tablas en `02_outputs/tables/` y los datos de las láminas en `00_data/processed/laminas/`.
Las cifras de cada lámina salen de esos archivos; los únicos datos escritos a mano son la tabla de posiciones de
Kaggle (`laminas/clasificacion.csv`) y los puntajes de los envíos (`laminas/envios_equipo.csv`).

| Script | Qué hace | Salida | Qué contiene la salida | Lámina |
|---|---|---|---|---|
| `00_rundirectory.R` | Corre todo el pipeline en orden | `02_outputs/session_info.txt` | Versiones de R, Python y paquetes | — |
| `00_setup.R` | Semilla, rutas, paquetes y funciones | — | — | — |
| `01_descargar_fuentes_externas.py` | Descarga OpenStreetMap (extracto de BBBike, 12 categorías con pyosmium), los polígonos de las 19 localidades (Nominatim) y 6 capas de Google Earth Engine en cada coordenada de los avisos. Por defecto verifica las descargas guardadas | `00_data/processed/osm/amenidades_osm.parquet` | 23.623 objetos de OSM: categoría, tipo de geometría, nombre, red, operador y geometría | Insumo de los mapas: best 1, 4, 14, 16 a 18; compare 1 |
| | | `00_data/processed/localidades_bogota.geojson` | Polígono y nombre de cada localidad | |
| | | `00_data/processed/gee_features.parquet` | Elevación, pendiente, NDVI a 200 m, luces nocturnas, altura construida y población en 22.015 coordenadas | |
| `02_descargar_catastro.py` | Descarga del servicio ArcGIS de Catastro Bogotá el valor de referencia del suelo, el avalúo en propiedad horizontal, los pisos y el estrato por manzana (2019 a 2021). Solo con `PS3_RECALCULAR=02` | `00_data/processed/catastro/*.parquet` (no se versiona) | Geometría de 43.958 manzanas y sus atributos por año | Insumo de best 5 y compare 3 (mapa del valor del suelo) |
| `03_variables_entorno_osm.py` | 26 variables de entorno: oficinas, hoteles, bancos, gimnasios y embajadas en radios de 300 m a 1 km, TransMilenio a 1 km, parques a 500 m, distancia a los cerros, elevación relativa, entorno de avisos sin precio y coordenada redondeada | `00_data/processed/variables_externas_extra.parquet` | property_id y las 26 variables | Variables del modelo: best 9, compare 5 |
| `04_variables_catastro.py` | 11 variables de Catastro por aviso: valor del suelo, avalúo, pisos y estrato de su manzana y de su año, y su entorno a 300 m | `00_data/processed/variables_catastro.parquet` | property_id y las 11 variables | Variables del modelo: best 5 y 9, compare 3 y 5 |
| `05_analisis_espacial_descriptivo.py` | Etapa 1: mapas en la coordenada exacta de las variables de Kaggle, Earth Engine y OSM; correlaciones con el precio (aviso por aviso, por zona y en celdas de 500 m); I de Moran, correlograma y LISA; precisión de las coordenadas | `figures/mapas_variables_kaggle.png`, `mapa_entrenamiento_prueba.png`, `mapas_capas_gee.png`, `mapas_objetos_osm.png` | Mapas de las variables brutas y de la ubicación de entrenamiento y prueba | `moran_global.csv`: best 28, compare 21 |
| | | `figures/correlacion_variables_precio.png`, `matriz_correlacion_spearman.png`, `dispersion_celdas_500m.png` | Correlaciones de Pearson y Spearman con el precio y entre variables | |
| | | `figures/correlograma_precio.png`, `lisa_precio_m2.png`, `precision_coordenadas.png` | Correlación de precios por distancia, clústeres LISA y malla de coordenadas redondeadas | |
| | | `tables/correlaciones_precio.csv`, `correlaciones_por_zona.csv`, `correlaciones_celdas_500m.csv` | Coeficientes por variable (y n) | |
| | | `tables/moran_global.csv`, `correlograma_precio.csv`, `precision_coordenadas.csv` | I de Moran con p-valor, correlación por banda de distancia, avisos con coordenada redondeada | |
| `06_texto_como_datos.py` | Texto como datos: normalización, corrección ortográfica con BETO, firmas de agentes, duplicados, 19 indicadores de diccionario con negación, 7 cantidades extraídas, matriz documento-término, TF-IDF, 50 componentes LSA, 10 tópicos LDA y 32 componentes de BETO | `00_data/processed/texto/texto_variables.parquet` (se regenera) | 91 variables de texto por aviso, su identificador y el grupo de texto duplicado | Base de best 6 a 9 y 24, compare 4, 5 y 20 |
| | | `00_data/processed/texto/tfidf.npz`, `dtm_conteos.npz`, `vocabulario.csv` (se regeneran) | Matrices de 48.930 avisos × 44.888 términos y vocabulario | |
| | | `tables/texto_*.csv` | Correcciones, etapas del vocabulario, duplicados, diccionarios, cantidades, perplejidad y tópicos del LDA, informatividad del texto y de BETO | |
| | | `figures/texto_vocabulario_zipf.png`, `texto_lsa_lda.png`, `texto_informatividad.png`, `texto_mapas.png`, `texto_beto.png` | Ley de Zipf, LSA y LDA, correlación de las variables de texto con el precio, mapas y BETO | |
| `07_validacion_cruzada_familias.py` | Predicciones fuera de pliegue de las seis familias (siete especificaciones con Catastro; LightGBM y CatBoost también sin Catastro) con CV aleatoria, bloques de 2 km, bloques con buffer de 300 m, clústeres k-means y Chapinero fuera; pesos de la mezcla LightGBM + CatBoost | `00_data/processed/modelos/cv_clase_oof_*.parquet`, `cv_clase_res_*.csv` | Predicción fuera de pliegue de cada aviso y MAE, RMSE y sesgo por esquema | best 2, 15, 23; compare 2, 7, 8, 10 a 18 |
| | | `tables/validacion_familias_esquemas.csv` | Todas las especificaciones y esquemas (formato largo) | |
| | | `tables/validacion_mezcla_clusters.csv` | Mezcla elegida en clústeres: pesos 0,4 y 0,6 con Catastro | |
| `08_knndm_kriging.py` | Pliegues kNNDM (distancia al vecino parecida a la de la prueba; q = 1.600) y regression kriging de los residuos del LightGBM, sin fuga | `tables/knndm_rejilla.csv` | Distancia de Wasserstein y distancias al vecino para cada q | best 15, 25 a 28; compare 21 |
| | | `tables/kriging_residuos_knndm.csv`, `kriging_variogramas.csv` | MAE y sesgo sin y con kriging; pepita, meseta y alcance por pliegue | |
| | | `00_data/processed/modelos/rk_knndm_oof_lightgbm_con.parquet` | Predicción kNNDM fuera de pliegue de cada aviso | |
| `09_modelo_final_envios.py` | Modelo final: diccionario de las 162 columnas, hiperparámetros, partición interna, validación en seis esquemas, rejilla del factor ×1,08, comprobación de los gemelos de texto; regenera los tres envíos y verifica su SHA-256 | `predictions/lightgbm_l2_eta005_hojas63_semillas5_catastro.csv` | Envío 01: LightGBM con Catastro (Kaggle 153,50 / 152,43) | best 2, 16, 22, 30; compare 2 |
| | | `predictions/lightgbm_l2_eta005_hojas63_semillas5_catastro_x108.csv` | Envío 02: el mismo ×1,08 (152,48 / 150,79; el final) | |
| | | `predictions/mezcla_lightgbm040_catboost060_prof8_catastro.csv` | Envío 03: 0,4 LightGBM + 0,6 CatBoost (155,21 / 153,94) | |
| | | `tables/modelo_*.csv`, `envios_verificacion_sha256.csv` | Bloques y diccionario de variables, hiperparámetros, validación, factor, gemelos y huellas SHA-256 | |
| | | `figures/modelo_mapas_variables.png`, `modelo_particion_interna.png`, `modelo_factor_estiramiento.png`, `modelo_pronostico_final.png` | Variables en el mapa, partición de la parada temprana, error según el factor y pronóstico final | |
| `10_comparacion_familias.py` | Compara las familias y las validaciones: un pliegue de cada esquema en el mapa, distancia al vecino de entrenamiento, MAE y sesgo por familia, orden de las familias, bootstrap pareado, calibración con Kaggle y pruebas descartadas | `figures/validacion_pliegues_mapa.png`, `validacion_distancia_vecino.png`, `familias_validaciones.png` | Pliegues, distancias y error de las siete especificaciones con tres validaciones | best 14, 15, 23, 27, 30; compare 8 a 11, 15 |
| | | `tables/familias_mae_por_esquema.csv`, `familias_sesgo_por_esquema.csv`, `familias_orden_por_esquema.csv` | Tabla de resultados del README y orden por esquema | |
| | | `tables/familias_bootstrap_chapinero.csv` | MAE con intervalo de 95 % y diferencia con LightGBM | |
| | | `tables/validacion_distancia_vecino.csv`, `calibracion_kaggle_envios.csv`, `pruebas_descartadas.csv` | Distancias, orden de los envíos por validación y Kaggle, sin Catastro, mezcla y kriging | |
| `11_errores_negocio.py` | Errores y negocio: sesgo por cuantil de la predicción, precios altos, avisos fuera de rango, error mayor o menor que el precio, SHAP, simulación de ofertas, maldición del ganador y precisión necesaria | `figures/errores_sesgo_por_cuantil.png`, `errores_chapinero_mapa.png`, `shap_importancia.png`, `negocio_maldicion_ganador.png`, `negocio_precision_necesaria.png` | Calibración, mapa de errores, SHAP y negocio | best 13, 17 a 21, 26 |
| | | `tables/errores_*.csv`, `shap_*.csv`, `negocio_*.csv` | Sesgo por cuantil, precios altos, fuera de rango, lados, importancia SHAP, ofertas y precisión necesaria | |
| `12_datos_laminas_shap_mapas.py` | Valores SHAP del LightGBM de una semilla, distancias al vecino por esquema, predicciones fuera de pliegue y bases de los mapas | `00_data/processed/laminas/shap_*.parquet`, `valores_*.parquet`, `distancias_vecino.parquet`, `prueba_mapa.parquet`, `entrenamiento_mapa.parquet`, `oof_lightgbm_con.parquet` (se regeneran) | Insumos de las imágenes y de los scripts 15 a 19 | best 1, 4, 13, 16, 18, 27, 29; compare 1, 12 |
| `13_datos_laminas_algoritmo.py` | Primer árbol del modelo, apartamento de ejemplo, curva de aprendizaje, pliegues en el mapa, descripción de los datos y errores con signo | `laminas/ppt_arbol.json`, `ppt_ejemplo_boosting.json`, `ppt_curva_aprendizaje.csv`, `ppt_pliegues_mapa.parquet`, `ppt_descripcion.json`, `ppt_faltantes.csv`, `ppt_errores_*.parquet` | Cifras y datos de las láminas del algoritmo y la validación | best 4, 6, 11, 12, 14, 17, 20 |
| `14_datos_laminas_arbol.py` | Elige un árbol real del modelo cuyos tres primeros niveles usan variables distintas | `laminas/ppt_arbol2.json` | Árbol con cortes en unidades legibles | best 10 |
| `15_datos_laminas_ejemplos_bootstrap.py` | Aviso de ejemplo, cascada SHAP, calibración por decil, maldición del ganador y bootstrap | `laminas/ppt_texto_candidatos.parquet`, `ppt_varios.json`, `ppt_bootstrap.json` y otros | Ejemplo de texto y cifras de apoyo | best 6 |
| `16_datos_laminas_negocio_catastro.py` | Calibración con y sin ×1,08, lados del error, compras simuladas, bootstrap pareado, tres apartamentos y Chapinero frente al resto con Catastro. Por defecto lee sus salidas (necesita Catastro) | `laminas/ppt_calibracion2.csv`, `ppt_lados*.csv`, `ppt_compras.csv`, `ppt_bootstrap2.json`, `ppt_tres_avisos.json`, `ppt_histograma_precios.csv`, `ppt_regla_rango2.csv`, `ppt_varios2.json` y otros (se versionan) | Cifras de negocio, validación y descripción | best 2, 4, 5, 12, 13, 16 a 21, 23, 25, 26, 30; compare 3, 9, 15, 18 |
| `17_datos_laminas_estrato_precision.py` | Manzanas de estrato 6 por zona y ganancia por oferta según la precisión. Por defecto lee sus salidas (necesita Catastro) | `laminas/ppt_ganancia_precision.csv` y `ppt_varios2.json` (se versionan) | Ganancia por oferta con margen de 10 % y estrato 6 | best 5, 20, 21; compare 3 |
| `18_datos_laminas_texto.py` | Frecuencias del texto: ley de Zipf, términos frecuentes, términos que distinguen Chapinero, tópicos y limpieza | `laminas/ppt_texto_zipf.csv`, `ppt_texto_frecuentes.csv`, `ppt_texto_chapinero.csv`, `ppt_texto_topicos.csv`, `ppt_texto_limpieza.json` | Cifras de las láminas de texto | best 7, 8; compare 4 |
| `19_datos_laminas_resumenes.py` | Error de las siete especificaciones por validación e importancia SHAP por variable y bloque | `laminas/cv_familias.csv` | MAE, sesgo, RMSE y sesgo en avisos de más de 1.000 millones | best 13, 23; compare 2, 7, 8, 10 a 18 |
| | | `laminas/shap_importancia.csv`, `shap_bloques.csv` | Aporte absoluto medio por variable y bloque | |
| `20_figuras_laminas.py` | Imágenes de las láminas al tamaño de su lugar (300 ppp): mapas, boosting con una variable, pliegues, dependencia SHAP, distancias, Zipf y fórmulas | `figures/laminas/portada_mapa.png` | Mapa de entrenamiento y prueba | best 1 (portada), 3 (f_mae), 4 (mapa_datos), 5 (valor suelo), 7 (zipf), 9 (f_rosen), 11 (boosting), 13 (f_shap), 14 (pliegues, f_sesgo), 16 (pronóstico), 17 (errores), 18 (confianza), 20 (f_oferta), 24 (f_tfidf, f_ridge), 27 (distancias), 29 (dependencia); compare 1, 2 (f_estiramiento), 3, 4, 5, 12 (dependencia_area), 19 (f_atencion), 20 |
| | | `figures/laminas/mapa_datos.png` | Datos de entrenamiento y prueba | |
| | | `figures/laminas/mapa_valor_suelo.png` | Valor del suelo 2021 por manzana | |
| | | `figures/laminas/texto_zipf.png` | Ley de Zipf | |
| | | `figures/laminas/boosting_1d.png` | Boosting con una variable | |
| | | `figures/laminas/pliegues.png` | Un pliegue de cada esquema | |
| | | `figures/laminas/pronostico_mapa.png`, `errores_mapa.png`, `confianza_mapa.png` | Pronóstico, error y variables fuera de rango en Chapinero | |
| | | `figures/laminas/distancias.png`, `dependencia.png`, `dependencia_area.png` | Distancia al vecino y aporte SHAP del área y del valor del suelo | |
| | | `figures/laminas/f_*.png` | Fórmulas (MAE, Rosen, boosting, sesgo, SHAP, oferta, ajuste, TF-IDF, ridge, atención) | |
| `21_construir_presentacion_best.py` | Construye la presentación Best Model Deep Dive con python-pptx (gráficos nativos y notas del orador) | `03_slides/best_equipo_02.pptx` | 31 láminas | best 1 a 31 |
| `22_construir_presentacion_compare.py` | Construye la presentación Algorithm Comparison | `03_slides/compare_equipo_02.pptx` | 22 láminas | compare 1 a 22 |
| `23_exportar_presentaciones_pdf.py` | Opcional (`PS3_EXPORTAR_PDF=1`, Windows con PowerPoint): exporta las dos presentaciones a PDF | `03_slides/best_equipo_02.pdf`, `compare_equipo_02.pdf` | Versión PDF de las presentaciones finales | — |

## Envíos a Kaggle

El equipo hizo 27 envíos (`00_data/processed/laminas/envios_equipo.csv`), al menos uno de cada familia salvo Elastic
Net. El que cuenta para la clasificación final es el 26: el envío 02 de Sebastián Vela (LightGBM con Catastro y texto,
×1,08), con 150,79 en la tabla privada, segundo de seis equipos (Grupo 6, 148,55; Grupo 02, 150,79; Grupo 01, 151,46;
Pishcke Blinders, 153,11; Grupo 05, 153,76; Grupo 03, 164,39). La tabla pública usa el 20 % de la prueba y la privada
el 80 % restante. En `02_outputs/predictions/` están los archivos de los envíos de Sebastián Vela, con nombres que dicen
el modelo y sus hiperparámetros, y su registro (`registro_envios_sebastian_vela.csv`: modelo, hiperparámetros,
SHA-256, validación y puntajes); el script 09 los regenera y comprueba que son idénticos a los subidos. Los archivos de
los envíos de los demás integrantes y de las dos redes 2×10 de Sebastián Vela no están en el repositorio.

| # | Autor | Familia | Archivo | MAE público | MAE privado |
|---|---|---|---|---|---|
| 1 | César Vallejo | Boosting | `LGBM_fe_leaves_31_minchild_10` | 179,9 | 180,2 |
| 2 | César Vallejo | Boosting | `XGB_fe_depth_4_eta_0_06` | 183,7 | 182,5 |
| 3 | César Vallejo | Boosting | `CB_fe_depth_8_l2_3` | 182,3 | 183,2 |
| 4 | César Vallejo | Boosting | `LGBM_fe_leaves_31_minchild_20` | 180,5 | 179,7 |
| 5 | César Vallejo | Boosting | `CB_fe_depth_8_l2_10` | 180,6 | 181,2 |
| 6 | César Vallejo | Boosting | `LGBM_fe_leaves_31_minchild_40` | 181,8 | 181,4 |
| 7 | César Vallejo | Boosting | `LGBM_fe_leaves_15_minchild_10` | 181,7 | 181,5 |
| 8 | César Vallejo | Boosting | `XGB_fe_depth_2_eta_0_06` | 187,5 | 188,2 |
| 9 | César Vallejo | Boosting | `XGB_fe_depth_3_eta_0_03` | 181,2 | 183,5 |
| 10 | César Vallejo | Boosting | `CB_fe_depth_4_l2_3` | 181,2 | 182,6 |
| 11 | César Vallejo | Boosting | `CB_fe_depth_6_l2_10` | 180,4 | 182,2 |
| 12 | César Vallejo | Boosting | `LGBM_fe_leaves_15_minchild_40` | 181,4 | 181,9 |
| 13 | César Vallejo | Random Forest | `RF_fe_depth_12_leaf_5` | 204,3 | 206,3 |
| 14 | César Vallejo | CART | `CART_fe_depth_6_leaf_40` | 240,7 | 239,9 |
| 15 | César Vallejo | Boosting | `LGBM_fe_leaves_63_eta_0_03` | 176,3 | 178,1 |
| 16 | César Vallejo | Red neuronal | `NN_n_10_lr_0_01` | 191,8 | 196,4 |
| 17 | César Vallejo | Regresión lineal | `LR_fe_nr_1` | 244,9 | 253,6 |
| 18 | Harold Aguilera | Ensamble | `superlearner_vec_boost_nn_rf` | 197,0 | 194,1 |
| 19 | César Vallejo | Boosting | `LGBM_fe_leaves_15_depth_6_eta_0_01` | 176,1 | 177,5 |
| 20 | Harold Aguilera | Boosting | `boostingpond_lightgbm_osm_xy` | 183,9 | 184,6 |
| 21 | Sebastián Vela | Red neuronal | `nn_clase_2x10_C_texto` | 186,42 | 189,24 |
| 22 | Sebastián Vela | Red neuronal | `nn_clase_2x10_D_coordenadas` | 180,36 | 184,43 |
| 23 | César Vallejo | Boosting | `LGBM_fe_leaves_31_depth_6_eta_0_01` | 175,58 | 177,18 |
| 24 | César Vallejo | Boosting | `LGBM_fe_leaves_7_depth_3_eta_0_05` | 180,33 | 181,88 |
| 25 | Sebastián Vela | Boosting | `envio_01_lightgbm_con_catastro` | 153,5 | 152,43 |
| 26 (envío final) | Sebastián Vela | Boosting | `envio_02_lightgbm_con_catastro_x108` | 152,48 | 150,79 |
| 27 | Sebastián Vela | Boosting | `envio_03_mezcla_lightgbm_catboost` | 155,21 | 153,94 |

## Fuentes de datos

| Fuente | Qué se usa | Cómo se obtiene | Archivo |
|---|---|---|---|
| Kaggle (avisos de Properati, competencia del curso) | 38.644 avisos de entrenamiento de Bogotá y 10.286 de prueba (Chapinero): precio, áreas, habitaciones, baños, tipo, fecha, coordenadas, título y descripción | Página de la competencia | `00_data/train.csv`, `test.csv`, `submission_template.csv` |
| OpenStreetMap, extracto de Bogotá de BBBike (23 de noviembre de 2026) | Parques, restaurantes, bares y cafés, universidades, colegios, centros comerciales, supermercados, hospitales, policía, estaciones, paraderos, vías principales, oficinas, hoteles, bancos, gimnasios, embajadas y áreas protegidas | Scripts 01 y 03 (pyosmium) | `00_data/processed/osm/amenidades_osm.parquet`, `variables_externas_extra.parquet` |
| OpenStreetMap, Nominatim | Polígonos de las 19 localidades urbanas | Script 01 (osmnx) | `00_data/processed/localidades_bogota.geojson` |
| Google Earth Engine | Elevación y pendiente (SRTM), NDVI a 200 m (Sentinel-2), luces nocturnas (VIIRS) y altura construida y población (GHSL) | Script 01 | `00_data/processed/gee_features.parquet` |
| Catastro Bogotá (UAECD), servicio ArcGIS público | Valor de referencia del suelo, avalúo en propiedad horizontal, pisos y estrato por manzana, de 2019 a 2021 | Scripts 02 y 04 | `00_data/processed/variables_catastro.parquet` |
| ConTexto (DNP, licencia MIT) y NLTK | Stopwords en español y listas de nombres y apellidos | Copias en el repositorio | `00_data/processed/texto/contexto_dnp/`, `stopwords_nltk_es.txt` |
| BETO (`dccuchile/bert-base-spanish-wwm-uncased`, Hugging Face) | Corrección ortográfica y embeddings del texto | Script 06 al recalcular | `00_data/processed/texto/auditoria_ortografia_beto.csv`, `beto_componentes.parquet` |

Lo que no está en el repositorio, y por qué: los embeddings completos de BETO (`texto/beto_embeddings.npy`, 144 MB;
se regeneran con `PS3_RECALCULAR=06`), el extracto de OpenStreetMap (`osm/Bogota.osm.pbf`, 21 MB) y las capas de
Catastro (`catastro/`, 58 MB), que se descargan con `PS3_RECALCULAR=01` y `02`; la base de texto
(`texto/texto_variables.parquet`, `tfidf.npz`, `dtm_conteos.npz` y otros, unos 80 MB) y la mayoría de los datos de las
láminas, que el pipeline regenera en cada corrida; y las credenciales de Earth Engine, que son de cada usuario.

## Convenciones

* Numeración. Dos dígitos consecutivos (`01_`, `02_`, ...), sin números repetidos ni saltos, compartida
  por R y Python. El script maestro se detiene si la numeración no cumple.
* Nombres. `snake_case` y que digan qué hace el script o qué contiene la salida. Los objetos y los
  argumentos de las funciones también llevan nombres descriptivos (no letras sueltas).
* Cada script corre solo. Su primera línea de código carga el entorno común:
  * R: `if (!exists("DIR_PROC")) source(file.path(here::here(), "01_code", "00_setup.R"), encoding = "UTF-8")`
  * Python: `from functions.setup_python import *`
* Sin rutas absolutas. Todo se escribe relativo a la raíz (carpeta con el archivo `.here`).
* Cifras de las diapositivas. Salen de archivos que genera un script (`03_slides/numbers.tex` para las versiones en
  LaTeX y `00_data/processed/laminas/` para las presentaciones en PowerPoint); no se escriben a mano, salvo la tabla de
  posiciones y los puntajes de Kaggle.

## Software

* R 4.6.1 y los paquetes de `requirements_R.txt`.
* Python 3.13.5 (miniconda) y los paquetes de `requirements.txt`. Los principales: numpy 2.3, pandas 2.3,
  scikit-learn 1.7, scipy 1.16, geopandas 1.1, LightGBM 4.7, CatBoost 1.2, PyTorch 2.9, libpysal 4.15, esda 2.10,
  NLTK 3.10 y python-pptx 1.0; para recalcular las descargas y BETO: pyosmium 4.3, osmnx 2.2, earthengine-api 1.7 y
  transformers 5.24.
* Las versiones exactas de la última corrida quedan en `02_outputs/session_info.txt`.
* La corrida original se hizo en Windows 11 con un procesador de 20 hilos, 24 GB de RAM y una GPU NVIDIA RTX 5060
  (esta solo para BETO).
