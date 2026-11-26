# Descarga de datos externos

Scripts en R para descargar las fuentes externas que usa un análisis espacial de viviendas en Bogotá: límites administrativos,
manzanas con estrato, la base catastral 2025 y categorías de OpenStreetMap. Todo se guarda en `external_raw/`, junto a estos
archivos.

## Contenido

| Archivo | Función |
|---|---|
| `main_data_extractor.R` | Orquestador. Ejecuta las etapas `abiertos`, `osm`, `gee` y `procedencia`. Sin etapa, corre todas menos `gee`. |
| `ext_download/datos_abiertos_raw.R` | Descarga las fuentes de Datos Abiertos Bogotá (etapa `abiertos`). |
| `ext_download/osm_raw.R` | Descarga las categorías de OpenStreetMap (etapa `osm`). |
| `ext_download/gee_raw.R` | Descarga las seis capas de Google Earth Engine para cada coordenada única (etapa `gee`). |
| `external_raw/` | Carpeta de salida. Se crea sola si no existe. |

## Requisitos

- R con los paquetes `sf`, `readr`, `dplyr`, `tibble`, `jsonlite` y `osmdata`.
- Para la etapa `gee`, además, el paquete `rgee` (con su Python y `earthengine-api` instalados por `rgee`), credenciales
  de Earth Engine (`rgee::ee_Authenticate()`) y estas variables de entorno: `PS3_GEE_PROYECTO`, el ID de su proyecto de
  Google Cloud registrado en Earth Engine (no tiene valor por defecto), y `GEE_COORDENADAS`, una o más rutas de CSV con
  columnas `lat` y `lon`, separadas por comas.
- Internet. Las descargas no se hacen si el archivo ya existe, salvo con `--reemplazar`.
- Herramientas del sistema: `sha256sum` (para verificar el zip catastral) y GDAL, que viene con `sf` (para convertir la base
  catastral a GeoPackage).
- Espacio en disco: el zip catastral puede pesar varios GB. El script acepta hasta 6 GB.

## Uso

Desde esta carpeta:

```bash
Rscript main_data_extractor.R                                  # abiertos, osm y procedencia, en ese orden
Rscript main_data_extractor.R abiertos                        # solo Datos Abiertos
Rscript main_data_extractor.R abiertos upz localidades         # solo esas fuentes
Rscript main_data_extractor.R abiertos --reemplazar           # vuelve a descargar y reemplaza
Rscript main_data_extractor.R osm                             # todas las categorías de OSM
Rscript main_data_extractor.R osm parques transporte          # solo esas categorías
PS3_GEE_PROYECTO=mi-proyecto GEE_COORDENADAS=ruta/train.csv,ruta/test.csv Rscript main_data_extractor.R gee
Rscript main_data_extractor.R procedencia                     # calcula los MD5 de los archivos
```

Fuentes de `abiertos`: `localidades`, `upl`, `upz`, `manzanas`, `catastro_2025`.

Categorías de `osm`: `parques`, `restaurantes`, `transporte`, `centros_comerciales`, `universidades`.

Los scripts de `ext_download/` también se pueden ejecutar solos, desde esta carpeta, con las mismas opciones.

## Comportamiento

- **No vuelve a descargar lo que ya existe.** Si el archivo de una fuente está en `external_raw/`, se omite. `--reemplazar` lo
  descarga de nuevo y reemplaza el anterior.
- **Escritura segura.** Cada archivo se escribe en un temporal y se renombra al final. Si algo falla, no queda un archivo a medias.
- **Caché de OpenStreetMap.** Cada consulta se divide en 16 teselas. Las teselas descargadas se guardan en
  `external_raw/.cache_osm/`. Si una falla tras varios reintentos, no se escribe nada y puedes volver a ejecutar: las teselas
  ya descargadas se reutilizan. La caché de una categoría se borra al escribir su CSV.
- **Validaciones.** Las fuentes se reproyectan a WGS84 cuando su sistema de referencia es reconocible. Si no lo es, el script se
  detiene. Localidades y UPZ deben tener los códigos únicos; si no, se detienen.
- **Procedencia.** Cada descarga de Datos Abiertos guarda su origen y su huella (la del catastro incluye URL, fecha, tamaño y
  SHA-256 del zip). La etapa `procedencia` calcula los MD5 de los archivos de localidades, UPZ, manzanas, Chapinero y OSM que
  existan, y los escribe en `procedencia_osm.txt`.

## Salidas en `external_raw/`

| Archivo | Contenido |
|---|---|
| `localidades_bogota.geojson` | Localidades de Bogotá (20). |
| `chapinero_poligono.csv` | Polígono de la localidad de Chapinero. |
| `upl_raw.json`, `upl_bogota.geojson` | Unidades de planeamiento local, crudo y reproyectado. |
| `IndUPZ.json` (y otros archivos del zip de UPZ) | Archivos crudos de las UPZ. |
| `upz_bogota.geojson` | Unidades de planeamiento zonal. |
| `manzanas_estratificacion.json` | Manzanas con estrato, en formato ArcGIS, sin cambios. |
| `manzanas_bogota.geojson` | Manzanas con estrato y la marca `flag_zona_industrial` (1 cuando el estrato es 0 o está vacío). |
| `bd_uaecd_2025.gpkg` | Base catastral 2025 convertida a GeoPackage. |
| `bd_uaecd_2025_procedencia.json` | Origen de la base catastral: URL, fecha, tamaño y SHA-256 del zip. |
| `osm_parques.csv`, `osm_restaurantes.csv`, `osm_transporte_masivo.csv`, `osm_centros_comerciales.csv`, `osm_universidades.csv` | Puntos y áreas de cada categoría de OpenStreetMap, con su centroide. |
| `gee_features.csv` | Una fila por coordenada única (`lat`, `lon`) y las seis capas: `elevacion_m`, `pendiente_grados`, `ndvi_200m`, `luces_nocturnas`, `altura_construida_m`, `poblacion_100m`. |
| `procedencia_osm.txt` | Fecha, versiones de paquetes, parámetros de la consulta y MD5 de los archivos. |
