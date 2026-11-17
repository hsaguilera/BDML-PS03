"""Entorno común de los scripts de Python del PS3 (equivalente a 00_setup.R).

Para que un script NN_*.py corra solo, su primera línea de código es:
    from functions.setup_python import *
Deja el directorio de trabajo en la raíz del repositorio (sin rutas absolutas),
fija la semilla y define las mismas carpetas que 00_setup.R.
"""
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # .../01_code/functions/setup_python.py -> raíz
os.chdir(ROOT)

SEED = 2026
N_FOLDS = 5
DIR_RAW = Path("00_data")              # train.csv, test.csv y submission_template.csv de Kaggle
DIR_PROC = Path("00_data/processed")
DIR_FIG = Path("02_outputs/figures")
DIR_TAB = Path("02_outputs/tables")
DIR_PRED = Path("02_outputs/predictions")
DIR_SLD = Path("03_slides")

for carpeta in (DIR_PROC, DIR_FIG, DIR_TAB, DIR_PRED):
    carpeta.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
try:
    import numpy as np
    np.random.seed(SEED)
except ImportError:  # numpy se instala con requirements.txt
    pass

__all__ = ["ROOT", "SEED", "N_FOLDS", "DIR_RAW", "DIR_PROC", "DIR_FIG", "DIR_TAB", "DIR_PRED", "DIR_SLD"]
