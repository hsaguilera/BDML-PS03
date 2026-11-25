# osm_raw.R
# Descarga las categorías de OpenStreetMap (Overpass) y genera sus CSV en external_raw/.
# Comportamiento:
#   - caja = localidades + buffer de 1000 m (en EPSG:9377);
#   - 4 x 4 teselas, cada una con reintentos y caché en .cache_osm/<categoria>/ (.rds);
#   - si una tesela falla tras los reintentos, no se escribe nada;
#   - si el CSV ya existe, el paso se omite; la caché de la categoría se borra al escribir el CSV.
#
# Categorías: parques, restaurantes, transporte, centros_comerciales, universidades.
# Requiere localidades_bogota.geojson (lo genera ext_download/datos_abiertos_raw.R localidades).
#
# Uso (desde external_data_load/):
#   Rscript ext_download/osm_raw.R                     # todas, en orden
#   Rscript ext_download/osm_raw.R parques transporte  # solo esas
#
# Paquetes: sf, osmdata, readr, dplyr, tibble.

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
DIR_CACHE_OSM <- file.path(DIR_EXTERNAL, ".cache_osm")

suppressPackageStartupMessages({
  library(sf)
  library(osmdata)
  library(readr)
  library(dplyr)
  library(tibble)
})

CRS_GEO <- 4326
CRS_METRICO <- 9377
BUFFER_M <- 1000
TESELAS_X <- 4
TESELAS_Y <- 4
TIMEOUT_OSM <- 120        # segundos
PAUSA_S <- 10             # pausa entre teselas descargadas
REINTENTOS <- 5
PAUSA_REINTENTO_S <- 30   # espera base tras un fallo; crece con cada intento

CONSULTAS_OSM <- list(
  parques = list(archivo = "osm_parques.csv", clave = "leisure", valor = "park",
                 tipo_osm = NULL, forma = "area"),
  restaurantes = list(archivo = "osm_restaurantes.csv", clave = "amenity", valor = "restaurant",
                      tipo_osm = "node", forma = "lugar"),
  transporte = list(archivo = "osm_transporte_masivo.csv", clave = "public_transport",
                    valor = "station", tipo_osm = NULL, forma = "lugar"),
  centros_comerciales = list(archivo = "osm_centros_comerciales.csv", clave = "shop",
                             valor = "mall", tipo_osm = NULL, forma = "lugar"),
  universidades = list(archivo = "osm_universidades.csv", clave = "amenity",
                       valor = "university", tipo_osm = NULL, forma = "lugar")
)

# ===== Utilidades ============================================================

ruta_ext <- function(nombre) file.path(DIR_EXTERNAL, nombre)

# Escribe un CSV en un temporal y lo renombra al final: nunca deja un archivo a medias.
escribir_csv_atomico <- function(tabla, nombre) {
  destino <- ruta_ext(nombre)
  tmp <- paste0(destino, ".tmp")
  write_csv(tabla, tmp)
  file.rename(tmp, destino)
  message("  escrito ", destino, " (", nrow(tabla), " filas)")
}

leer_localidades <- function() {
  ruta <- ruta_ext("localidades_bogota.geojson")
  if (!file.exists(ruta)) {
    stop("Falta localidades_bogota.geojson. Ejecuta antes ext_download/datos_abiertos_raw.R localidades.")
  }
  st_read(ruta, quiet = TRUE)
}

# Caja de consulta: límite de Bogotá con buffer, en grados. Orden de osmdata: c(xmin, ymin, xmax, ymax).
caja_consulta <- function(localidades) {
  bogota <- st_union(st_geometry(st_transform(localidades, CRS_METRICO)))
  as.numeric(st_bbox(st_transform(st_buffer(bogota, BUFFER_M), CRS_GEO)))
}

# Parte la caja en teselas c(xmin, ymin, xmax, ymax). Primero recorre x, luego y.
teselas <- function(caja, nx, ny) {
  xs <- seq(caja[1], caja[3], length.out = nx + 1)
  ys <- seq(caja[2], caja[4], length.out = ny + 1)
  res <- list()
  for (i in seq_len(nx)) {
    for (j in seq_len(ny)) {
      res[[length(res) + 1]] <- c(xs[i], ys[j], xs[i + 1], ys[j + 1])
    }
  }
  res
}

# Consulta una clave/valor en todas las teselas. Cada tesela descargada se guarda en caché:
# al reejecutar, las que ya están se leen del disco y no se vuelven a pedir.
consultar_teselas <- function(caja, clave, valor, tipo_osm, paso) {
  tipos <- if (is.null(tipo_osm)) c("node", "way", "relation") else tipo_osm
  res <- list()
  tes <- teselas(caja, TESELAS_X, TESELAS_Y)
  dir_paso <- file.path(DIR_CACHE_OSM, paso)
  dir.create(dir_paso, recursive = TRUE, showWarnings = FALSE)
  for (n in seq_along(tes)) {
    bb <- tes[[n]]
    archivo_tesela <- file.path(dir_paso, sprintf("tesela_%02d_%s.rds", n,
                                                  paste(round(bb, 5), collapse = "_")))
    message("  tesela ", n, " de ", length(tes))
    if (file.exists(archivo_tesela)) {
      message("    ya descargada (caché)")
      res[[length(res) + 1]] <- readRDS(archivo_tesela)
      next
    }
    datos <- NULL
    for (intento in seq_len(REINTENTOS)) {
      datos <- tryCatch(
        opq(bbox = bb, timeout = TIMEOUT_OSM, osm_types = tipos) |>
          add_osm_feature(key = clave, value = valor) |>
          osmdata_sf(),
        error = function(e) {
          message("    intento ", intento, " fallido: ", conditionMessage(e))
          NULL
        }
      )
      if (!is.null(datos)) break
      Sys.sleep(PAUSA_REINTENTO_S * intento)
    }
    if (is.null(datos)) {
      stop("Falló la tesela ", n, " de ", length(tes), " (", clave, "=", valor,
           "). No se escribió nada; las teselas ya descargadas quedan en caché. ",
           "Vuelve a ejecutar: ext_download/osm_raw.R ", paso)
    }
    tmp <- paste0(archivo_tesela, ".tmp")
    saveRDS(datos, tmp)
    file.rename(tmp, archivo_tesela)
    res[[length(res) + 1]] <- datos
    Sys.sleep(PAUSA_S)
  }
  res
}

# Une un elemento de todas las teselas, sin repetir osm_id. NULL si no hay nada.
unir_elemento <- function(lista, elemento) {
  partes <- lapply(lista, function(d) d[[elemento]])
  partes <- Filter(Negate(is.null), partes)
  if (length(partes) == 0) return(NULL)
  unido <- bind_rows(partes)
  unido[!duplicated(unido$osm_id), ]
}

unir_areas <- function(lista) {
  a <- unir_elemento(lista, "osm_polygons")
  b <- unir_elemento(lista, "osm_multipolygons")
  if (is.null(a)) return(b)
  if (is.null(b)) return(a)
  unido <- bind_rows(a, b)
  unido[!duplicated(unido$osm_id), ]
}

# Asegura la columna name (osmdata la omite si ninguna entidad la tiene).
con_nombre <- function(x) {
  if (!("name" %in% names(x))) x$name <- NA_character_
  x
}

# ===== Paso ==================================================================

paso_osm <- function(paso) {
  cfg <- CONSULTAS_OSM[[paso]]
  if (file.exists(ruta_ext(cfg$archivo))) {
    message("  ", cfg$archivo, " ya existe; se omite")
    return(invisible(NULL))
  }
  localidades <- leer_localidades()
  caja <- caja_consulta(localidades)
  message("  caja: ", paste(round(caja, 4), collapse = ", "))

  lista <- consultar_teselas(caja, cfg$clave, cfg$valor, cfg$tipo_osm, paso)

  if (cfg$forma == "area") {
    areas <- unir_areas(lista)
    if (is.null(areas) || nrow(areas) == 0) {
      stop("No hay ", paso, " en la caja; no se escribe ", cfg$archivo)
    }
    areas <- con_nombre(areas)
    am <- st_make_valid(st_transform(areas, CRS_METRICO))
    am <- am[!st_is_empty(am), ]
    area <- as.numeric(st_area(am))
    keep <- area > 0
    am <- am[keep, ]
    area <- area[keep]
    xy <- st_coordinates(st_transform(st_centroid(st_geometry(am)), CRS_GEO))
    tabla <- tibble(osm_id = as.character(am$osm_id), name = as.character(am$name),
                    lon = xy[, 1], lat = xy[, 2], area_m2 = area)
  } else {
    tablas <- list()
    puntos <- unir_elemento(lista, "osm_points")
    if (!is.null(puntos) && nrow(puntos) > 0) {
      puntos <- con_nombre(puntos)
      xy <- st_coordinates(puntos)
      tablas[[length(tablas) + 1]] <- tibble(
        osm_id = as.character(puntos$osm_id), tipo = "punto",
        name = as.character(puntos$name), lon = xy[, 1], lat = xy[, 2])
    }
    areas <- unir_areas(lista)
    if (!is.null(areas) && nrow(areas) > 0) {
      areas <- con_nombre(areas)
      cen <- st_transform(st_centroid(st_geometry(st_transform(st_make_valid(areas), CRS_METRICO))),
                          CRS_GEO)
      xy <- st_coordinates(cen)
      tablas[[length(tablas) + 1]] <- tibble(
        osm_id = as.character(areas$osm_id), tipo = "area",
        name = as.character(areas$name), lon = xy[, 1], lat = xy[, 2])
    }
    if (length(tablas) == 0) {
      stop("No hay resultados para ", cfg$clave, "=", cfg$valor, "; no se escribe ", cfg$archivo)
    }
    tabla <- bind_rows(tablas)
  }

  escribir_csv_atomico(tabla, cfg$archivo)
  unlink(file.path(DIR_CACHE_OSM, paso), recursive = TRUE)
  invisible(tabla)
}

# Args: nombres de categorías. Sin nombres, corre todas, en el orden de CONSULTAS_OSM.
principal_osm <- function(args) {
  desconocidas <- setdiff(args, names(CONSULTAS_OSM))
  if (length(desconocidas) > 0) {
    stop("Categorías desconocidas: ", paste(desconocidas, collapse = ", "),
         ". Válidas: ", paste(names(CONSULTAS_OSM), collapse = ", "))
  }
  seleccion <- if (length(args) > 0) names(CONSULTAS_OSM)[names(CONSULTAS_OSM) %in% args] else names(CONSULTAS_OSM)
  dir.create(DIR_EXTERNAL, recursive = TRUE, showWarnings = FALSE)
  for (paso in seleccion) {
    message("== ", paso)
    paso_osm(paso)
  }
  message("Listo.")
  invisible(TRUE)
}

# Solo se ejecuta si este archivo es el script que lanza Rscript (no si se carga con source()).
ejecutado <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
if (length(ejecutado) == 1 && basename(ejecutado) == "osm_raw.R") {
  principal_osm(commandArgs(trailingOnly = TRUE))
}
