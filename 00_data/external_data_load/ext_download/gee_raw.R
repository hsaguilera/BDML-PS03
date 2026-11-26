# gee_raw.R
# Descarga las seis capas de Google Earth Engine para cada coordenada única de los avisos y escribe
# external_raw/gee_features.csv. Es el equivalente en R del bloque 3 de 01_descargar_fuentes_externas.py
# (proyecto PS3 sebastian).
#
# Capas (mismo nombre y mismo cálculo que en el script de Python):
#   elevacion_m          USGS/SRTMGL1_003: altura sobre el mar
#   pendiente_grados     pendiente calculada sobre SRTM
#   ndvi_200m            NDVI de Sentinel-2 SR armonizado, 2020-2021: mediana con máscara de nubes (SCL) y
#                        promedio en un círculo de 200 m
#   luces_nocturnas      VIIRS mensual 2020: media de avg_rad
#   altura_construida_m  GHSL built height 2018
#   poblacion_100m       GHSL población 2020, por celda de 100 m
#
# Requiere:
#   PS3_GEE_PROYECTO  ID del proyecto de Google Cloud registrado en Earth Engine. No tiene valor por defecto.
#   GEE_COORDENADAS   una o más rutas de CSV con columnas lat y lon, separadas por comas
#                     (por ejemplo, train.csv y test.csv del proyecto).
#   Credenciales locales de Earth Engine: rgee::ee_Authenticate() desde R, o earthengine authenticate.
#   Internet.
#
# Uso (desde external_data_load/):
#   PS3_GEE_PROYECTO=mi-proyecto GEE_COORDENADAS=ruta/train.csv,ruta/test.csv Rscript ext_download/gee_raw.R
#   ... --reemplazar   vuelve a descargar aunque gee_features.csv ya exista
#
# Comportamiento:
#   - si gee_features.csv ya existe, no descarga (salvo --reemplazar);
#   - las consultas se hacen en lotes de TAM_LOTE puntos;
#   - la escritura es atómica: primero un temporal, luego se renombra;
#   - se detiene si alguna coordenada no recibe fila de Earth Engine.
#
# Paquetes: rgee, sf, readr, dplyr, tibble.

# ===== Configuración =========================================================

.carpeta_script <- {
  archivo <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(archivo) == 1) dirname(normalizePath(sub("^--file=", "", archivo))) else getwd()
}
# Salida: external_raw/, junto a main_data_extractor.R. Desde el extractor, DIR_EXTERNAL ya viene definida.
# Si este archivo se ejecuta solo, .carpeta_script es ext_download/ y la salida queda un nivel arriba.
if (!exists("DIR_EXTERNAL")) {
  DIR_EXTERNAL <- if (basename(.carpeta_script) == "ext_download") {
    file.path(dirname(.carpeta_script), "external_raw")
  } else {
    file.path(.carpeta_script, "external_raw")
  }
}

suppressPackageStartupMessages({
  library(rgee)
  library(sf)
  library(readr)
  library(dplyr)
  library(tibble)
})

ARCHIVO_GEE <- "gee_features.csv"
CAPAS_GEE <- c("elevacion_m", "pendiente_grados", "ndvi_200m", "luces_nocturnas",
               "altura_construida_m", "poblacion_100m")
# Caja de Bogotá usada en el script de Python: xmin, ymin, xmax, ymax en EPSG:4326.
CAJA_GEE <- c(-74.25, 4.45, -73.95, 4.85)
ESCALA_M <- 10
TAM_LOTE <- 4000          # puntos por consulta, para no superar el límite de elementos

# ===== Utilidades ============================================================

ruta_ext <- function(nombre) file.path(DIR_EXTERNAL, nombre)

# Lee las coordenadas únicas de una o más rutas separadas por comas.
leer_coordenadas <- function(rutas_texto) {
  if (!nzchar(rutas_texto)) {
    stop("Defina GEE_COORDENADAS con una o más rutas de CSV con columnas lat y lon.")
  }
  rutas <- trimws(strsplit(rutas_texto, ",")[[1]])
  faltan <- rutas[!file.exists(rutas)]
  if (length(faltan) > 0) {
    stop("GEE_COORDENADAS: no existe(n): ", paste(faltan, collapse = ", "))
  }
  bind_rows(lapply(rutas, function(r) read_csv(r, col_select = c(lat, lon), show_col_types = FALSE))) |>
    mutate(lat = as.numeric(lat), lon = as.numeric(lon)) |>
    filter(is.finite(lat), is.finite(lon)) |>
    distinct(lat, lon)
}

# ===== Earth Engine ==========================================================

# Apila las seis capas en una sola imagen, recortada a la caja.
construir_pila <- function(bbox) {
  srtm <- ee$Image("USGS/SRTMGL1_003")

  # Quita sombra de nube (3) y nubes y cirros (8 a 10), según la banda SCL.
  mascara_nubes <- ee_utils_pyfunc(function(img) {
    scl <- img$select("SCL")
    img$updateMask(scl$neq(3)$And(scl$lt(8)$Or(scl$gt(10))))
  })

  s2 <- ee$ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  s2 <- s2$filterBounds(bbox)
  s2 <- s2$filterDate("2020-01-01", "2021-12-31")
  s2 <- s2$filter(ee$Filter$lt("CLOUDY_PIXEL_PERCENTAGE", 40))
  s2 <- s2$map(mascara_nubes)
  s2 <- s2$median()

  pendiente <- ee$Terrain$slope(srtm)$rename("pendiente_grados")

  ndvi <- s2$normalizedDifference(c("B8", "B4"))
  ndvi_200m <- ndvi$reduceNeighborhood(
    reducer = ee$Reducer$mean(),
    kernel = ee$Kernel$circle(radius = 200, units = "meters")
  )$rename("ndvi_200m")

  luces <- ee$ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
  luces <- luces$filterDate("2020-01-01", "2020-12-31")
  luces <- luces$select("avg_rad")$mean()$rename("luces_nocturnas")

  construida <- ee$Image("JRC/GHSL/P2023A/GHS_BUILT_H/2018")$
    select("built_height")$rename("altura_construida_m")
  poblacion <- ee$Image("JRC/GHSL/P2023A/GHS_POP/2020")$
    select("population_count")$rename("poblacion_100m")

  ee$Image$cat(list(
    srtm$rename("elevacion_m"), pendiente, ndvi_200m, luces, construida, poblacion
  ))$clip(bbox)
}

# Consulta un lote de puntos. Devuelve el identificador i y las seis capas, una fila por punto.
extraer_lote <- function(pila, puntos_lote) {
  valores <- ee_extract(
    x = pila,
    y = puntos_lote["i"],
    scale = ESCALA_M,
    fun = ee$Reducer$first(),
    sf = FALSE
  )
  as_tibble(valores) |> select(i, all_of(CAPAS_GEE))
}

# ===== Paso ==================================================================

paso_gee <- function(reemplazar) {
  destino <- ruta_ext(ARCHIVO_GEE)
  if (file.exists(destino) && !reemplazar) {
    message("  ", ARCHIVO_GEE, " ya existe; se omite (use --reemplazar para descargarlo de nuevo)")
    return(invisible(NULL))
  }

  proyecto <- Sys.getenv("PS3_GEE_PROYECTO", "")
  if (!nzchar(proyecto)) {
    stop("Defina PS3_GEE_PROYECTO con el ID de su proyecto de Google Cloud registrado en Earth Engine.")
  }

  coord <- leer_coordenadas(Sys.getenv("GEE_COORDENADAS", ""))
  message("  coordenadas únicas: ", nrow(coord))

  ee_Initialize(project = proyecto)
  pila <- construir_pila(ee$Geometry$Rectangle(CAJA_GEE))

  puntos <- coord |>
    mutate(i = row_number()) |>
    st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE)

  lotes <- split(seq_len(nrow(puntos)), ceiling(seq_len(nrow(puntos)) / TAM_LOTE))
  trozos <- lapply(seq_along(lotes), function(k) {
    message("  lote ", k, " de ", length(lotes))
    extraer_lote(pila, puntos[lotes[[k]], ])
  })
  valores <- bind_rows(trozos)

  if (anyDuplicated(valores$i) || nrow(valores) != nrow(puntos)) {
    stop("Earth Engine devolvió ", nrow(valores), " filas para ", nrow(puntos), " coordenadas. ",
         "No se escribe nada.")
  }

  tabla <- st_drop_geometry(puntos) |>
    as_tibble() |>
    left_join(valores, by = "i") |>
    select(lat, lon, all_of(CAPAS_GEE))

  vacios <- colSums(is.na(tabla[CAPAS_GEE]))
  message("  valores vacíos por capa: ", paste(names(vacios), vacios, sep = "=", collapse = ", "))

  tmp <- paste0(destino, ".tmp")
  write_csv(tabla, tmp)
  file.rename(tmp, destino)
  message("  escrito ", destino, " (", nrow(tabla), " coordenadas)")
  invisible(tabla)
}

# Args: solo se admite --reemplazar.
principal_gee <- function(args) {
  desconocidas <- setdiff(args, "--reemplazar")
  if (length(desconocidas) > 0) {
    stop("Opciones desconocidas para gee: ", paste(desconocidas, collapse = ", "),
         ". Solo se admite --reemplazar.")
  }
  dir.create(DIR_EXTERNAL, recursive = TRUE, showWarnings = FALSE)
  paso_gee(reemplazar = "--reemplazar" %in% args)
  message("Listo.")
  invisible(TRUE)
}

# Solo se ejecuta si este archivo es el script que lanza Rscript (no si se carga con source()).
ejecutado <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
if (length(ejecutado) == 1 && basename(ejecutado) == "gee_raw.R") {
  principal_gee(commandArgs(trailingOnly = TRUE))
}
