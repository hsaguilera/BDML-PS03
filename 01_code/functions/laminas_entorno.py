"""Carpetas de las láminas (scripts 12 a 23): datos de las láminas, imágenes y presentaciones.

    DATOS = 00_data/processed/laminas/   cifras y tablas que leen los constructores de las presentaciones (csv, json, parquet)
    IMG   = 02_outputs/figures/laminas/  imágenes de las láminas (png a 300 ppp, al tamaño de su lugar en la lámina)
    SALIDA_PPT = 03_slides/              best_equipo_02.pptx y compare_equipo_02.pptx

En 00_data/processed/laminas/ se versionan solo los archivos que no se pueden regenerar sin internet: los que dependen de
las capas de Catastro (scripts 16 y 17) y dos tablas de Kaggle escritas a mano (clasificacion.csv, con la tabla privada de
los seis equipos, y envios_equipo.csv, con los envíos de los tres integrantes y sus puntajes). Lo demás lo regeneran los
scripts 12 a 19 en cada corrida.
"""
import os
from pathlib import Path

from functions.setup_python import DIR_FIG, DIR_PROC, DIR_SLD

DATOS = DIR_PROC / "laminas"
IMG = DIR_FIG / "laminas"
SALIDA_PPT = Path(os.environ.get("SALIDA_PPT", DIR_SLD))     # SALIDA_PPT permite guardar versiones de prueba en otra carpeta
for _c in (DATOS, IMG, SALIDA_PPT):
    _c.mkdir(parents=True, exist_ok=True)
