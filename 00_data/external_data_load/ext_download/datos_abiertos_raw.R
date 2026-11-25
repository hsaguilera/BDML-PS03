# datos_abiertos_raw.R
# Descarga las fuentes de Datos Abiertos Bogotá y las guarda en external_raw/.
# Fuentes de Datos Abiertos Bogotá:
#   localidades -> localidades_bogota.geojson (20 localidades, Chapinero) y chapinero_poligono.csv
#   upl         -> upl_raw.json y upl_bogota.geojson (UplCodigo, UplNombre)
#   upz         -> archivos del zip (IndUPZ.json entre ellos) y upz_bogota.geojson (UpzCodigo, UpzNombre)
#   manzanas    -> manzanas_estratificacion.json (crudo, formato ArcGIS) y manzanas_bogota.geojson,
#                  con flag_zona_industrial = 1 cuando ESTRATO es 0 o vacío. Queda en la proyección de origen.
#   catastro_2025 -> bd_uaecd_2025.gpkg: zip de la BD catastral 2025 (FileGDB) convertido con GDAL (sf).
#                  Deja también bd_uaecd_2025_procedencia.json (URL, fecha, tamaño y SHA-256 del zip).
#
# Uso (desde external_data_load/):
#   Rscript ext_download/datos_abiertos_raw.R                       # todas, en orden
#   Rscript ext_download/datos_abiertos_raw.R upz localidades       # solo esas
#   Rscript ext_download/datos_abiertos_raw.R catastro_2025         # solo la BD catastral 2025
#   Rscript ext_download/datos_abiertos_raw.R --reemplazar          # vuelve a descargar y reemplaza
#
# Por defecto NO descarga: si el archivo de una fuente ya existe en external_raw con su nombre, se omite antes de
# descargar. Con --reemplazar vuelve a bajarlo y reemplaza, informando si el nuevo es idéntico o distinto del anterior.
# Cada fuente se descarga a un temporal y solo se escribe si pasa las validaciones.
# Localidades, UPL, UPZ y manzanas se reproyectan a WGS84 (EPSG:4326) si su sistema de referencia es
# reconocible; si no lo es, se detienen. Las manzanas se reproyectan al guardar porque el GeoJSON no
# conserva el sistema de referencia de origen.
# Paquetes: sf, readr, jsonlite. El zip se descomprime con unzip() de base R.

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
  library(sf)
  library(readr)
  library(dplyr)
  library(tibble)
})

TIMEOUT_S <- 120
TAMANO_MAX_BYTES <- 300e6
TAMANO_MAX_CATASTRO_BYTES <- 6e9
CRS_GEO <- 4326
CRS_METRICO <- 9377

URL_LOCALIDADES <- paste0(
  "https://datosabiertos.bogota.gov.co/dataset/856cb657-8ca3-4ee8-857f-37211173b1f8/",
  "resource/497b8756-0927-4aee-8da9-ca4e32ca3a8a/download/loca.json"
)
URL_UPL <- paste0(
  "https://datosabiertos.bogota.gov.co/dataset/808582fc-ffc8-4649-8428-7e1fd8d3820c/",
  "resource/a5c8c591-0708-420f-8eb7-9f3147e21c40/download/unidadplaneamientolocal.json"
)
URL_UPZ <- paste0(
  "https://datosabiertos.bogota.gov.co/dataset/ab602d2c-4383-4bbc-97e2-28fc26c6b735/",
  "resource/a158fb82-0460-43be-a2ce-eafc95b699bb/download/json.zip"
)
URL_MANZANAS <- paste0(
  "https://datosabiertos.bogota.gov.co/dataset/55467552-0af4-4524-a390-a2956035744e/",
  "resource/29f2d770-bd5d-4450-9e95-8737167ba12f/download/manzanaestratificacion.json"
)

URL_CATASTRO_2025 <- paste0(
  "https://datosabiertos.bogota.gov.co/dataset/e4f2dba4-d0a1-4ddc-a47b-a1adc6ef8b48/",
  "resource/27fa44a1-ae32-4158-a6d8-411dbadbbc5e/download/gdb_bd_uaecd_2025.zip"
)
NOMBRE_GPKG_CATASTRO <- "bd_uaecd_2025.gpkg"
ARCHIVO_PROCEDENCIA_CATASTRO <- "bd_uaecd_2025_procedencia.json"

NUM_LOCALIDADES <- 20
NOMBRE_CHAPINERO <- "CHAPINERO"
ARCHIVO_UPZ_CRUDO <- "IndUPZ.json"

# ===== Descarga, zip y escritura =============================================

descargar <- function(url, destino, tamano_max = TAMANO_MAX_BYTES) {
  vieja <- options(timeout = TIMEOUT_S)
  on.exit(options(vieja), add = TRUE)
  download.file(url, destino, mode = "wb", quiet = TRUE)
  tamano <- file.size(destino)
  if (is.na(tamano) || tamano > tamano_max) {
    stop("La descarga de ", url, " no existe o supera el tamaño máximo permitido")
  }
  tamano
}

# TRUE si los destinos ya existen y no se pidió reemplazar. Se comprueba antes de descargar.
omitir <- function(destinos, conservar) {
  if (conservar && all(file.exists(destinos))) {
    message("  ", paste(basename(destinos), collapse = ", "),
            " ya existe en external_raw; no se descarga (usa --reemplazar para volver a bajarlo)")
    return(TRUE)
  }
  FALSE
}

extraer_zip <- function(zip_path, carpeta) {
  listado <- unzip(zip_path, list = TRUE)$Name
  nombres <- listado[!endsWith(listado, "/")]
  if (length(nombres) == 0) stop("El zip no contiene archivos")
  if (any(startsWith(nombres, "/") | grepl("..", nombres, fixed = TRUE))) {
    stop("Ruta sospechosa en el zip")
  }
  base <- basename(nombres)
  if (anyDuplicated(base)) stop("Nombre repetido dentro del zip: ", base[anyDuplicated(base)])
  unzip(zip_path, exdir = carpeta, junkpaths = TRUE)
  file.path(carpeta, base)
}

buscar_crudo_upz <- function(extraidos) {
  idx <- which(tolower(basename(extraidos)) == tolower(ARCHIVO_UPZ_CRUDO))
  if (length(idx) > 0) return(extraidos[idx[1]])
  jsons <- extraidos[tolower(tools::file_ext(extraidos)) %in% c("json", "geojson")]
  if (length(jsons) == 1) return(jsons)
  stop("No encuentro ", ARCHIVO_UPZ_CRUDO, " en el zip. JSON encontrados: ",
       paste(basename(jsons), collapse = ", "))
}

mismo_contenido <- function(a, b) {
  file.size(a) == file.size(b) &&
    identical(readBin(a, "raw", file.size(a)), readBin(b, "raw", file.size(b)))
}

# Copia un archivo crudo a external_raw/. Devuelve FALSE si lo conserva.
guardar_archivo <- function(origen, destino, conservar) {
  if (file.exists(destino)) {
    if (conservar) {
      message("  ", basename(destino), " ya existe; se conserva (usa --reemplazar para cambiarlo)")
      return(invisible(FALSE))
    }
    estado <- if (mismo_contenido(origen, destino)) "idéntico al anterior" else "distinto del anterior"
    message("  ", basename(destino), " ya existe (", estado, "); se reemplaza")
  }
  tmp <- paste0(destino, ".tmp")
  file.copy(origen, tmp, overwrite = TRUE)
  file.rename(tmp, destino)
  message("  guardado ", destino)
  invisible(TRUE)
}

# Escribe un GeoJSON sf con escritura atómica. Devuelve FALSE si lo conserva.
guardar_geojson <- function(x, destino, conservar) {
  if (file.exists(destino)) {
    if (conservar) {
      message("  ", basename(destino), " ya existe; se conserva (usa --reemplazar para cambiarlo)")
      return(invisible(FALSE))
    }
    message("  ", basename(destino), " ya existe; se reemplaza")
  }
  tmp <- paste0(destino, ".tmp")
  st_write(x, tmp, driver = "GeoJSON", delete_dsn = TRUE, quiet = TRUE)
  file.rename(tmp, destino)
  message("  guardado ", destino, " (", nrow(x), " elementos)")
  invisible(TRUE)
}

# ===== Validaciones y transformaciones =======================================

# Devuelve x en WGS84. Si el archivo declara un sistema de referencia que sf reconoce, lo reproyecta
# con PROJ (como hacía paso_localidades). Si no lo reconoce, se detiene.
a_wgs84 <- function(x, nombre) {
  epsg <- st_crs(x)$epsg
  if (is.na(epsg)) stop(nombre, ": no tiene un sistema de referencia reconocible")
  if (epsg != CRS_GEO) {
    message("  ", nombre, ": reproyectando de EPSG:", epsg, " a EPSG:", CRS_GEO)
    x <- st_transform(st_make_valid(x), CRS_GEO)
  }
  x
}

normalizar_nombre <- function(x) {
  toupper(chartr("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunAEIOUUN", as.character(x)))
}

# Deja solo dos campos con nombres fijos. Las geometrías no cambian.
a_campos_fijos <- function(x, campo_codigo, campo_nombre, nuevo_codigo, nuevo_nombre, nombre) {
  faltan <- setdiff(c(campo_codigo, campo_nombre), names(x))
  if (length(faltan) > 0) {
    stop(nombre, ": faltan los campos ", paste(faltan, collapse = ", "),
         ". Campos: ", paste(setdiff(names(x), attr(x, "sf_column")), collapse = ", "))
  }
  if (anyDuplicated(x[[campo_codigo]]) > 0) stop(nombre, ": códigos repetidos en ", campo_codigo)
  x <- a_wgs84(x, nombre)
  salida <- x[, c(campo_codigo, campo_nombre)]
  names(salida)[1:2] <- c(nuevo_codigo, nuevo_nombre)
  salida
}

validar_localidades <- function(loc) {
  if (nrow(loc) != NUM_LOCALIDADES) {
    stop("Se recibieron ", nrow(loc), " localidades; se esperaban ", NUM_LOCALIDADES)
  }
  if (anyNA(loc$LocCodigo) || anyDuplicated(loc$LocCodigo) > 0) {
    stop("LocCodigo falta en alguna localidad o no es único")
  }
  n_chapinero <- sum(normalizar_nombre(loc$LocNombre) == NOMBRE_CHAPINERO)
  if (n_chapinero != 1) stop("Se esperaba una localidad ", NOMBRE_CHAPINERO, "; hay ", n_chapinero)
  invisible(TRUE)
}

# Anillo exterior del polígono más grande de la geometría, como tibble(lon, lat).
anillo_exterior <- function(geom) {
  partes <- st_cast(st_make_valid(geom), "POLYGON")
  k <- which.max(as.numeric(st_area(st_transform(partes, CRS_METRICO))))
  coords <- st_coordinates(partes[k])
  anillo <- coords[coords[, "L1"] == 1, c("X", "Y"), drop = FALSE]
  if (nrow(anillo) > 1 && all(anillo[1, ] == anillo[nrow(anillo), ])) {
    anillo <- anillo[-nrow(anillo), , drop = FALSE]
  }
  tibble(lon = anillo[, 1], lat = anillo[, 2])
}

escribir_chapinero <- function(loc, destino, conservar) {
  if (file.exists(destino) && conservar) {
    message("  ", basename(destino), " ya existe; se conserva (usa --reemplazar para cambiarlo)")
    return(invisible(FALSE))
  }
  sel <- loc[normalizar_nombre(loc$LocNombre) == NOMBRE_CHAPINERO, ]
  poligono <- anillo_exterior(st_geometry(sel))
  if (nrow(poligono) < 3) stop("El polígono de Chapinero tiene menos de 3 vértices")
  tmp <- paste0(destino, ".tmp")
  write_csv(poligono, tmp)
  file.rename(tmp, destino)
  message("  guardado ", destino)
  invisible(TRUE)
}

# ===== Fuentes ===============================================================

procesar_localidades <- function(conservar) {
  if (omitir(file.path(DIR_EXTERNAL, c("localidades_bogota.geojson", "chapinero_poligono.csv")), conservar)) return(invisible(FALSE))
  temporal <- tempfile("localidades_")
  dir.create(temporal)
  on.exit(unlink(temporal, recursive = TRUE), add = TRUE)

  descargado <- file.path(temporal, "loca.json")
  message("localidades: descargando ", URL_LOCALIDADES)
  message("  ", format(descargar(URL_LOCALIDADES, descargado), big.mark = ","), " bytes")
  loc <- st_read(descargado, quiet = TRUE)
  message("  contenido: ", nrow(loc), " elementos; campos: ",
          paste(setdiff(names(loc), attr(loc, "sf_column")), collapse = ", "))
  loc <- a_wgs84(loc, "localidades")
  validar_localidades(loc)
  message("  ", NUM_LOCALIDADES, " localidades, códigos únicos, ", NOMBRE_CHAPINERO, " presente")

  guardar_geojson(loc, file.path(DIR_EXTERNAL, "localidades_bogota.geojson"), conservar)
  escribir_chapinero(loc, file.path(DIR_EXTERNAL, "chapinero_poligono.csv"), conservar)
}

procesar_upl <- function(conservar) {
  if (omitir(file.path(DIR_EXTERNAL, "upl_bogota.geojson"), conservar)) return(invisible(FALSE))
  temporal <- tempfile("upl_")
  dir.create(temporal)
  on.exit(unlink(temporal, recursive = TRUE), add = TRUE)

  descargado <- file.path(temporal, "upl_raw.json")
  message("upl: descargando ", URL_UPL)
  message("  ", format(descargar(URL_UPL, descargado), big.mark = ","), " bytes")
  upl <- st_read(descargado, quiet = TRUE)
  message("  contenido: ", nrow(upl), " elementos")

  guardar_archivo(descargado, file.path(DIR_EXTERNAL, "upl_raw.json"), conservar)
  salida <- a_campos_fijos(upl, "CODIGO_UPL", "NOMBRE", "UplCodigo", "UplNombre", "upl")
  guardar_geojson(salida, file.path(DIR_EXTERNAL, "upl_bogota.geojson"), conservar)
}

procesar_upz <- function(conservar) {
  if (omitir(file.path(DIR_EXTERNAL, "upz_bogota.geojson"), conservar)) return(invisible(FALSE))
  temporal <- tempfile("upz_")
  dir.create(temporal)
  on.exit(unlink(temporal, recursive = TRUE), add = TRUE)

  zip_path <- file.path(temporal, "json.zip")
  message("upz: descargando ", URL_UPZ)
  message("  ", format(descargar(URL_UPZ, zip_path), big.mark = ","), " bytes")
  carpeta <- file.path(temporal, "extraido")
  dir.create(carpeta)
  extraidos <- extraer_zip(zip_path, carpeta)
  message("  archivos en el zip: ", length(extraidos))
  for (ruta in extraidos) guardar_archivo(ruta, file.path(DIR_EXTERNAL, basename(ruta)), conservar)

  crudo <- buscar_crudo_upz(extraidos)
  upz <- st_read(crudo, quiet = TRUE)
  salida <- a_campos_fijos(upz, "CODIGO_UPZ", "NOMBRE", "UpzCodigo", "UpzNombre", "upz")
  guardar_geojson(salida, file.path(DIR_EXTERNAL, "upz_bogota.geojson"), conservar)
}

# Área firmada de un anillo (fórmula del cordón). Negativa si el anillo va en sentido horario.
area_firmada <- function(p) {
  x <- p[, 1]
  y <- p[, 2]
  n <- nrow(p)
  sum(x * y[c(2:n, 1)] - x[c(2:n, 1)] * y) / 2
}

# Anillos de ArcGIS -> POLYGON o MULTIPOLYGON. Los contornos exteriores van en sentido horario y los
# huecos en antihorario; cada hueco se asigna al último contorno exterior anterior.
anillos_a_geometria <- function(anillos) {
  poligonos <- list()
  for (a in anillos) {
    p <- matrix(unlist(a), ncol = length(a[[1]]), byrow = TRUE)[, 1:2, drop = FALSE]
    if (!all(p[1, ] == p[nrow(p), ])) p <- rbind(p, p[1, ])
    if (area_firmada(p) < 0) {
      poligonos[[length(poligonos) + 1]] <- list(p)
    } else if (length(poligonos) > 0) {
      poligonos[[length(poligonos)]] <- c(poligonos[[length(poligonos)]], list(p))
    } else {
      stop("Hueco sin contorno exterior antes")
    }
  }
  if (length(poligonos) == 0) return(st_polygon())
  if (length(poligonos) == 1) return(st_polygon(poligonos[[1]]))
  st_multipolygon(poligonos)
}

# Lee el JSON de manzanas (formato ArcGIS) con jsonlite. Devuelve un sf en el sistema de referencia
# de origen, con los atributos tal como vienen (las fechas quedan en milisegundos desde 1970).
leer_manzanas_esri <- function(ruta) {
  crudo <- jsonlite::fromJSON(ruta, simplifyVector = FALSE)
  if (!identical(crudo$geometryType, "esriGeometryPolygon")) {
    stop("Tipo de geometría no esperado: ", crudo$geometryType)
  }
  wkt <- crudo$spatialReference$wkt
  crs <- if (is.null(wkt)) NA else st_crs(wkt)
  if (is.na(crs)) stop("Sistema de referencia de las manzanas no reconocido")

  elementos <- crudo$features
  datos <- list()
  for (campo in crudo$fields) {
    nombre <- campo$name
    valores <- lapply(elementos, function(f) {
      v <- f$attributes[[nombre]]
      if (is.null(v)) NA else v
    })
    datos[[nombre]] <- if (grepl("String", campo$type, fixed = TRUE)) {
      vapply(valores, as.character, character(1))
    } else {
      vapply(valores, as.numeric, numeric(1))
    }
  }
  geometrias <- lapply(elementos, function(f) anillos_a_geometria(f$geometry$rings))
  st_sf(data.frame(datos, check.names = FALSE, stringsAsFactors = FALSE),
        geometry = st_sfc(geometrias, crs = crs))
}

procesar_manzanas <- function(conservar) {
  if (omitir(file.path(DIR_EXTERNAL, "manzanas_bogota.geojson"), conservar)) return(invisible(FALSE))
  temporal <- tempfile("manzanas_")
  dir.create(temporal)
  on.exit(unlink(temporal, recursive = TRUE), add = TRUE)

  descargado <- file.path(temporal, "manzanaestratificacion.json")
  message("manzanas: descargando ", URL_MANZANAS)
  message("  ", format(descargar(URL_MANZANAS, descargado), big.mark = ","), " bytes")
  guardar_archivo(descargado, file.path(DIR_EXTERNAL, "manzanas_estratificacion.json"), conservar)

  # Formato ArcGIS (ESRIJSON). Se lee con su proyección de origen (CarMAGBOG).
  manzanas <- leer_manzanas_esri(descargado)
  wkt <- st_crs(manzanas)$wkt
  if (is.na(wkt) || !grepl("CarMAGBOG", wkt, fixed = TRUE)) {
    stop("Sistema de referencia de las manzanas no reconocido")
  }
  if (!("CODIGO_MANZANA" %in% names(manzanas))) stop("Falta CODIGO_MANZANA en las manzanas")
  if (!("ESTRATO" %in% names(manzanas))) stop("Falta ESTRATO en las manzanas")

  repetidos <- sum(duplicated(manzanas$CODIGO_MANZANA))
  if (repetidos > 0) message("  aviso: ", repetidos, " códigos de manzana repetidos")

  # Sin estrato (0 o vacío): zonas no residenciales (industriales, comerciales, dotacionales,
  # parques), porque la estratificación solo aplica a inmuebles residenciales.
  manzanas$flag_zona_industrial <- as.integer(is.na(manzanas$ESTRATO) | manzanas$ESTRATO == 0)
  message("  ", sum(manzanas$flag_zona_industrial), " manzanas sin estrato (flag_zona_industrial = 1)")

  # El GeoJSON no conserva un sistema de referencia propio: al releerlo, sf asume WGS84 y las coordenadas
  # de CarMAGBOG quedan como grados inválidos. Por eso se reproyectan a WGS84 antes de guardar.
  manzanas <- st_transform(st_make_valid(manzanas), CRS_GEO)
  guardar_geojson(manzanas, file.path(DIR_EXTERNAL, "manzanas_bogota.geojson"), conservar)
}

# ===== Catastro 2025 (FileGDB del zip -> GeoPackage) =========================

sha256_archivo <- function(ruta) {
  salida <- system2("sha256sum", shQuote(ruta), stdout = TRUE)
  strsplit(salida, " +")[[1]][1]
}

# Descarga el zip, lo descomprime en un temporal dentro de external_raw (mismo sistema de archivos, para que el paso
# final sea un cambio de nombre), convierte el FileGDB a GeoPackage con GDAL (sf::gdal_utils) y escribe la procedencia.
procesar_catastro_2025 <- function(conservar) {
  destino_gpkg <- file.path(DIR_EXTERNAL, NOMBRE_GPKG_CATASTRO)
  if (omitir(destino_gpkg, conservar)) return(invisible(FALSE))

  temporal <- tempfile("catastro_2025_", tmpdir = DIR_EXTERNAL)
  dir.create(temporal)
  on.exit(unlink(temporal, recursive = TRUE), add = TRUE)

  zip_path <- file.path(temporal, "gdb_bd_uaecd_2025.zip")
  message("catastro_2025: descargando ", URL_CATASTRO_2025)
  bytes <- descargar(URL_CATASTRO_2025, zip_path, TAMANO_MAX_CATASTRO_BYTES)
  message("  ", format(bytes, big.mark = ","), " bytes")
  sha_zip <- sha256_archivo(zip_path)
  message("  sha256 del zip: ", sha_zip)

  carpeta <- file.path(temporal, "extraido")
  dir.create(carpeta)
  unzip(zip_path, exdir = carpeta)
  gdbs <- list.dirs(carpeta, recursive = TRUE, full.names = TRUE)
  gdbs <- gdbs[grepl("\\.gdb$", gdbs)]
  if (length(gdbs) != 1) stop("Se esperaba un FileGDB en el zip y hay ", length(gdbs))
  capas <- sf::st_layers(gdbs[1])$name
  message("  FileGDB: ", basename(gdbs[1]), "; capas: ", length(capas))

  tmp_gpkg <- file.path(temporal, NOMBRE_GPKG_CATASTRO)
  sf::gdal_utils(util = "vectortranslate", source = gdbs[1], destination = tmp_gpkg, options = c("-f", "GPKG"))
  if (!file.exists(tmp_gpkg)) stop("La conversión a GeoPackage no produjo archivo")

  if (file.exists(destino_gpkg)) message("  ", NOMBRE_GPKG_CATASTRO, " ya existe; se reemplaza")
  file.rename(tmp_gpkg, destino_gpkg)
  message("  guardado ", destino_gpkg)

  procedencia <- list(
    fuente = "catastro_2025",
    url = URL_CATASTRO_2025,
    fecha_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
    bytes_zip = bytes,
    sha256_zip = sha_zip,
    gpkg = NOMBRE_GPKG_CATASTRO,
    capas = capas,
    agente = "datos_abiertos_raw.R"
  )
  jsonlite::write_json(procedencia, file.path(DIR_EXTERNAL, ARCHIVO_PROCEDENCIA_CATASTRO),
                       auto_unbox = TRUE, pretty = TRUE)
  message("  procedencia en ", file.path(DIR_EXTERNAL, ARCHIVO_PROCEDENCIA_CATASTRO))
  invisible(TRUE)
}

PROCESOS <- list(
  localidades = procesar_localidades,
  upl = procesar_upl,
  upz = procesar_upz,
  manzanas = procesar_manzanas,
  catastro_2025 = procesar_catastro_2025
)
FUENTES <- names(PROCESOS)

# ===== Entrada ===============================================================

# Args: nombres de fuentes y, opcionalmente, --reemplazar. Sin nombres, corre todas. Por defecto no descarga lo que existe.
principal_abiertos <- function(args) {
  conservar <- !("--reemplazar" %in% args)
  pedidas <- setdiff(args, c("--reemplazar", "--conservar"))
  desconocidas <- setdiff(pedidas, FUENTES)
  if (length(desconocidas) > 0) {
    stop("Fuentes desconocidas: ", paste(desconocidas, collapse = ", "),
         ". Válidas: ", paste(FUENTES, collapse = ", "))
  }
  seleccion <- if (length(pedidas) > 0) FUENTES[FUENTES %in% pedidas] else FUENTES
  dir.create(DIR_EXTERNAL, recursive = TRUE, showWarnings = FALSE)
  for (fuente in seleccion) {
    message("== ", fuente)
    PROCESOS[[fuente]](conservar)
  }
  message("Listo.")
  invisible(TRUE)
}

# Solo se ejecuta si este archivo es el script que lanza Rscript (no si se carga con source()).
ejecutado <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
if (length(ejecutado) == 1 && basename(ejecutado) == "datos_abiertos_raw.R") {
  principal_abiertos(commandArgs(trailingOnly = TRUE))
}
