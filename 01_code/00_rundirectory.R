# ==============================================================================
# 00_rundirectory.R  --  Script maestro (PS3: Making Money with ML?)
# Uso:  source("01_code/00_rundirectory.R")   (desde la raíz del repositorio)
# Corre, en orden numérico, todos los scripts NN_*.R y NN_*.py de 01_code y reproduce las salidas.
# Este script solo llama a otros scripts.
#
# Tramo corto (opcional), con variables de entorno:
#   Sys.setenv(PS3_DESDE = "5")    # corre desde el script 05 (si ya existen sus insumos)
#   Sys.setenv(PS3_HASTA = "8")    # corre hasta el script 08
# Python: usa `python` en Windows y `python3` en otros sistemas; se cambia con
#   Sys.setenv(PS3_PYTHON = "ruta/al/python")
# ==============================================================================

# --- Entorno común: raíz, paquetes (con versiones), parámetros y funciones -------
if (!requireNamespace("here", quietly = TRUE)) install.packages("here")
source(file.path(here::here(), "01_code", "00_setup.R"), encoding = "UTF-8")

# --- Scripts numerados (R y Python comparten una sola numeración) ----------------
scripts <- sort(list.files("01_code", pattern = "^[0-9]{2}_.*\\.(R|py)$", full.names = TRUE))
scripts <- scripts[!grepl("^00_", basename(scripts))]
numeros <- as.integer(substr(basename(scripts), 1, 2))

# La numeración debe ser 01, 02, 03, ... sin números repetidos ni saltos
if (anyDuplicated(numeros)) {
  stop("Numeración repetida en 01_code: ", paste(basename(scripts)[numeros %in% numeros[duplicated(numeros)]], collapse = ", "))
}
if (!identical(numeros, seq_along(numeros))) {
  stop("Hay saltos en la numeración de 01_code; faltan: ", paste(setdiff(seq_len(max(numeros)), numeros), collapse = ", "))
}

desde   <- as.integer(Sys.getenv("PS3_DESDE", "0"))
hasta   <- as.integer(Sys.getenv("PS3_HASTA", "99"))
python  <- Sys.getenv("PS3_PYTHON", if (.Platform$OS.type == "windows") "python" else "python3")

for (script in scripts[numeros >= desde & numeros <= hasta]) {
  message("\n>>> Ejecutando ", script)
  if (grepl("\\.py$", script)) {
    estado <- system2(python, shQuote(script))
    if (estado != 0) stop("Falló ", script, " (código de salida ", estado, ")")
  } else {
    source(script, local = new.env(parent = globalenv()), encoding = "UTF-8")
  }
}

# --- Registro del entorno: versiones de R, de Python y de sus paquetes -----------
versiones_python <- tryCatch(
  c(system2(python, "--version", stdout = TRUE, stderr = TRUE),
    system2(python, c("-m", "pip", "freeze"), stdout = TRUE, stderr = FALSE)),
  error = function(e) "Python no disponible"
)
writeLines(c(capture.output(sessionInfo()), "", versiones_python), file.path("02_outputs", "session_info.txt"))
message("\nPipeline completo.")
