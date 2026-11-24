"""Utilidades comunes de los scripts de Python 01 a 11 (pipeline del LightGBM con Catastro y texto).

Complementa a setup_python.py (que fija la raíz, la semilla y las carpetas DIR_*). Aquí quedan:

- el control de recálculo con la variable de entorno PS3_RECALCULAR;
- las subcarpetas de 00_data/processed/ que usan estos scripts;
- funciones para guardar figuras y tablas en 02_outputs/ y para escribir números con coma decimal.

Recálculo. Por defecto, los pasos que necesitan internet, credenciales de Earth Engine, GPU o mucho tiempo de cómputo
leen los intermedios guardados en 00_data/processed/ si existen. Para recalcularlos:

    PS3_RECALCULAR=1          recalcula todos los pasos
    PS3_RECALCULAR=07,10      recalcula solo los scripts 07 y 10

En R, antes de correr el script maestro: Sys.setenv(PS3_RECALCULAR = "1").
"""
import os
import sys
import time
from pathlib import Path

# La consola de Windows (cp1252) no tiene algunos símbolos de los mensajes (≈, −, ×): se reemplazan en vez de fallar.
for _flujo in (sys.stdout, sys.stderr):
    try:
        _flujo.reconfigure(errors="replace")
    except AttributeError:                 # salida redirigida a un objeto sin reconfigure
        pass

import matplotlib

matplotlib.use("Agg")                      # sin ventanas: las figuras solo se guardan
import matplotlib.pyplot as plt  # noqa: E402

from functions.setup_python import DIR_FIG, DIR_PRED, DIR_PROC, DIR_RAW, DIR_TAB, SEED  # noqa: E402

# --- Subcarpetas de 00_data/processed/ ----------------------------------------------------------------------------
DIR_OSM = DIR_PROC / "osm"                         # extracto de OpenStreetMap y objetos extraídos
DIR_CATASTRO = DIR_PROC / "catastro"               # capas por manzana de Catastro Bogotá (solo al recalcular)
DIR_TEXTO = DIR_PROC / "texto"                     # base de texto, auditorías y listas del DNP
DIR_MODELOS = DIR_PROC / "modelos"                 # predicciones fuera de pliegue y finales
DIR_CACHE_TEXTO = DIR_PROC / "cache_puntajes_texto"  # puntajes ridge de texto por pliegue (se regeneran)
for _carpeta in (DIR_OSM, DIR_CATASTRO, DIR_TEXTO, DIR_MODELOS, DIR_CACHE_TEXTO):
    _carpeta.mkdir(parents=True, exist_ok=True)

ARCHIVO_LOCALIDADES = DIR_PROC / "localidades_bogota.geojson"
ARCHIVO_GEE = DIR_PROC / "gee_features.parquet"
ARCHIVO_AMENIDADES = DIR_OSM / "amenidades_osm.parquet"
ARCHIVO_PBF = DIR_OSM / "Bogota.osm.pbf"
ARCHIVO_EXTERNAS_EXTRA = DIR_PROC / "variables_externas_extra.parquet"
ARCHIVO_CATASTRO = DIR_PROC / "variables_catastro.parquet"

CRS_GEO, CRS_M = "EPSG:4326", "EPSG:9377"          # 9377: MAGNA-SIRGAS Origen Nacional (metros), como en clase
CENTRO, FACTOR = 654, 1.08                         # ajuste final del envío 02: 654 + 1,08 (ŷ − 654)
COLOR_TRAIN, COLOR_TEST = "#4C72B0", "#DD8452"

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
_T0 = time.time()


def recalcular(numero):
    """True si PS3_RECALCULAR pide recalcular el script `numero` ("07") o todos ("1")."""
    valor = os.environ.get("PS3_RECALCULAR", "0").strip()
    if valor in ("", "0"):
        return False
    if valor == "1":
        return True
    return f"{int(numero):02d}" in {f"{int(v):02d}" for v in valor.split(",") if v.strip().isdigit()}


def log(mensaje):
    print(f"[{(time.time() - _T0) / 60:5.1f} min] {mensaje}", flush=True)


def existen(*rutas):
    return all(Path(r).exists() for r in rutas)


def exigir(*rutas, como=""):
    """Detiene el script con un mensaje claro si falta un insumo."""
    faltan = [str(r) for r in rutas if not Path(r).exists()]
    if faltan:
        raise FileNotFoundError("Faltan insumos: " + ", ".join(faltan) + (f". {como}" if como else ""))


def guardar_figura(fig, nombre):
    """Guarda la figura en 02_outputs/figures/<nombre>.png y la cierra."""
    ruta = DIR_FIG / f"{nombre}.png"
    fig.savefig(ruta)
    plt.close(fig)
    log(f"Figura: {ruta.as_posix()}")
    return ruta


def guardar_tabla(tabla, nombre, index=True, decimales=None):
    """Guarda la tabla en 02_outputs/tables/<nombre>.csv (UTF-8, punto decimal, para que la lean otros programas)."""
    ruta = DIR_TAB / f"{nombre}.csv"
    t = tabla.round(decimales) if decimales is not None else tabla
    t.to_csv(ruta, index=index, encoding="utf-8")
    log(f"Tabla: {ruta.as_posix()}")
    return ruta


def coma(x, d=1, miles=True):
    """Número con coma decimal y punto de miles: coma(1234.56) -> '1.234,6'."""
    if x is None or (isinstance(x, float) and x != x):
        return "NA"
    s = f"{x:,.{d}f}" if miles else f"{x:.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


__all__ = ["DIR_RAW", "DIR_PROC", "DIR_FIG", "DIR_TAB", "DIR_PRED", "SEED", "DIR_OSM", "DIR_CATASTRO", "DIR_TEXTO",
           "DIR_MODELOS", "DIR_CACHE_TEXTO", "ARCHIVO_LOCALIDADES", "ARCHIVO_GEE", "ARCHIVO_AMENIDADES", "ARCHIVO_PBF",
           "ARCHIVO_EXTERNAS_EXTRA", "ARCHIVO_CATASTRO", "CRS_GEO", "CRS_M", "CENTRO", "FACTOR", "COLOR_TRAIN",
           "COLOR_TEST", "plt", "recalcular", "log", "existen", "exigir", "guardar_figura", "guardar_tabla", "coma"]
