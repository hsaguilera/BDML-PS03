# ==============================================================================
# 00_setup.R  --  Entorno común del PS3 (rutas, semilla, paquetes y funciones)
# Lo carga 00_rundirectory.R. Para que un script corra solo, su primera línea de código es:
#   if (!exists("DIR_PROC")) source(file.path(here::here(), "01_code", "00_setup.R"), encoding = "UTF-8")
# Sin rutas absolutas: la raíz del proyecto es la carpeta que contiene el archivo `.here`.
# Versiones: `requirements_R.txt` fija las versiones; si una difiere, se avisa.
# Sys.setenv(PS3_INSTALAR_VERSIONES = "1") instala exactamente esas versiones.
# ==============================================================================

# --- Raíz del proyecto ---------------------------------------------------------
if (!requireNamespace("here", quietly = TRUE)) install.packages("here")
ROOT <- here::here()
setwd(ROOT)

# --- Versiones fijadas (requirements_R.txt: paquete==versión) -------------------
versiones_fijadas <- character()
if (file.exists("requirements_R.txt")) {
  lineas <- trimws(readLines("requirements_R.txt", warn = FALSE))
  lineas <- lineas[nzchar(lineas) & !startsWith(lineas, "#") & grepl("==", lineas, fixed = TRUE)]
  versiones_fijadas <- setNames(sub(".*==", "", lineas), sub("==.*", "", lineas))
}
version_distinta <- function(paquete) {
  !requireNamespace(paquete, quietly = TRUE) ||
    as.character(utils::packageVersion(paquete)) != versiones_fijadas[[paquete]]
}
if (Sys.getenv("PS3_INSTALAR_VERSIONES") == "1" && length(versiones_fijadas)) {
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
  for (paquete in names(versiones_fijadas)) {
    if (version_distinta(paquete)) {
      remotes::install_version(paquete, version = versiones_fijadas[[paquete]], upgrade = "never")
    }
  }
}

# --- Paquetes (pacman instala los que falten) -----------------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(tidyverse, data.table, glmnet, caret, rpart, ranger, xtable, jsonlite, scales,
               here, sf, osmdata, spatialsample)
paquetes_distintos <- names(versiones_fijadas)[vapply(names(versiones_fijadas), version_distinta, logical(1))]
if (length(paquetes_distintos)) {
  message("Aviso: versión distinta de la fijada en requirements_R.txt para: ",
          paste(paquetes_distintos, collapse = ", "), " (los resultados pueden variar levemente).")
}

# --- Parámetros globales ---------------------------------------------------------
SEED     <- 2026
N_FOLDS  <- 5
DIR_RAW  <- "00_data"             # train.csv, test.csv y submission_template.csv de Kaggle
DIR_PROC <- "00_data/processed"
DIR_FIG  <- "02_outputs/figures"
DIR_TAB  <- "02_outputs/tables"
DIR_PRED <- "02_outputs/predictions"
DIR_SLD  <- "03_slides"
for (carpeta in c(DIR_PROC, DIR_FIG, DIR_TAB, DIR_PRED)) dir.create(carpeta, recursive = TRUE, showWarnings = FALSE)
set.seed(SEED)

# --- Funciones reutilizables -------------------------------------------------------
for (archivo in list.files("01_code/functions", pattern = "\\.R$", full.names = TRUE)) {
  source(archivo, encoding = "UTF-8")
}
