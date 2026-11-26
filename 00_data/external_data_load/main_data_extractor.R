# main_data_extractor.R
# Descarga las fuentes externas a external_raw/, junto a este archivo. Requiere internet y se ejecuta en terminal.
#
#   Rscript main_data_extractor.R                          # abiertos, OSM y procedencia
#   Rscript main_data_extractor.R abiertos [FUENTES] [--reemplazar]
#   Rscript main_data_extractor.R osm [CATEGORIAS]
#   Rscript main_data_extractor.R gee [--reemplazar]   # requiere PS3_GEE_PROYECTO y GEE_COORDENADAS
#   Rscript main_data_extractor.R procedencia
#
# abiertos -> ext_download/datos_abiertos_raw.R   (localidades, upl, upz, manzanas, catastro_2025). Por defecto no descarga lo que
#             ya existe en external_raw; --reemplazar vuelve a bajarlo.
# osm      -> ext_download/osm_raw.R        (parques, restaurantes, transporte, centros_comerciales, universidades)
# gee      -> ext_download/gee_raw.R        (seis capas de Google Earth Engine por coordenada; fuera de la ejecución por defecto)
# procedencia -> md5 de los archivos anteriores en procedencia_osm.txt (R base no tiene sha256).
#
# Se detiene en el primer error. Las opciones de cada etapa se pasan tal cual.

# ===== Configuración =========================================================

.carpeta_script <- {
  archivo <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(archivo) == 1) dirname(normalizePath(sub("^--file=", "", archivo))) else getwd()
}
# Salida de las descargas: external_raw/, junto a este archivo.
DIR_EXTERNAL <- file.path(.carpeta_script, "external_raw")
source(file.path(.carpeta_script, "ext_download", "datos_abiertos_raw.R"))
source(file.path(.carpeta_script, "ext_download", "osm_raw.R"))
source(file.path(.carpeta_script, "ext_download", "gee_raw.R"))

ETAPAS <- c("abiertos", "osm", "gee", "procedencia")
# Sin etapa, el extractor corre estas. gee queda fuera porque necesita credenciales y variables de entorno.
ETAPAS_DEFECTO <- c("abiertos", "osm", "procedencia")

# ===== Procedencia ===========================================================

paso_procedencia <- function() {
  nombres <- c(
    "localidades_bogota.geojson", "upl_bogota.geojson", "upz_bogota.geojson",
    "manzanas_bogota.geojson", "chapinero_poligono.csv", ARCHIVO_GEE,
    vapply(CONSULTAS_OSM, function(x) x$archivo, character(1))
  )
  existentes <- nombres[file.exists(file.path(DIR_EXTERNAL, nombres))]
  rutas <- file.path(DIR_EXTERNAL, existentes)
  lineas <- c(
    paste("Fecha UTC:", format(Sys.time(), "%Y-%m-%d %H:%M:%S", tz = "UTC")),
    paste("osmdata:", as.character(packageVersion("osmdata"))),
    paste("sf:", as.character(packageVersion("sf"))),
    paste("Teselas:", TESELAS_X, "x", TESELAS_Y, "; timeout:", TIMEOUT_OSM, "s"),
    paste("Buffer (m):", BUFFER_M),
    paste("URL localidades:", URL_LOCALIDADES),
    "md5:",
    paste0("  ", existentes, " ", unname(tools::md5sum(rutas)))
  )
  writeLines(lineas, ruta_ext("procedencia_osm.txt"))
  message("  escrito ", ruta_ext("procedencia_osm.txt"))
}

# ===== Entrada ===============================================================

# Etapa opcional como primer argumento. Sin etapa, corre ETAPAS_DEFECTO.
principal <- function(args) {
  if (length(args) > 0 && !(args[1] %in% ETAPAS)) {
    stop("Etapa desconocida: ", args[1], ". Válidas: ", paste(ETAPAS, collapse = ", "))
  }
  etapa <- if (length(args) > 0) args[1] else NULL
  resto <- if (length(args) > 0) args[-1] else character(0)
  etapas <- if (is.null(etapa)) ETAPAS_DEFECTO else etapa

  for (e in etapas) {
    message("##### ", e)
    if (e == "abiertos") {
      principal_abiertos(resto)
    } else if (e == "osm") {
      principal_osm(resto)
    } else if (e == "gee") {
      principal_gee(resto)
    } else {
      if (length(resto) > 0) stop("procedencia no admite opciones")
      paso_procedencia()
    }
  }
  message("Todo listo.")
  invisible(TRUE)
}

# Solo se ejecuta si este archivo es el script que lanza Rscript (no si se carga con source()).
ejecutado <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
if (length(ejecutado) == 1 && basename(ejecutado) == "main_data_extractor.R") {
  principal(commandArgs(trailingOnly = TRUE))
}
