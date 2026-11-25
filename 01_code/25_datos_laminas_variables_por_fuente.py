"""25 · Datos de las láminas: datos con valor (entrenamiento y prueba) y promedio de cada una de las 162 variables.

Qué hace (antes document/presentaciones/25_datos_laminas_variables_por_fuente.py)
    Con la base del modelo final (functions/flujo_final.py, sin validar ni entrenar) y el puntaje de precio del texto
    guardado (cache_puntajes_texto/ridge_final.npy), calcula el porcentaje de avisos con dato y el promedio de cada variable
    del modelo, y cuenta los objetos de OpenStreetMap por categoría.

Salida (00_data/processed/laminas/; se versiona porque la leen los scripts 21 y 22)
    ppt_variables.json
Insumos: scripts 01 a 06 y 09 (puntaje de texto). Tiempo: menos de un minuto.
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

from functions.entorno_ps3 import (ARCHIVO_AMENIDADES, ARCHIVO_LOCALIDADES, CENTRO, DIR_CACHE_TEXTO,  # noqa: E402,F401
                                   DIR_MODELOS, DIR_OSM, DIR_PRED, DIR_RAW, DIR_TEXTO, FACTOR, existen, log,
                                   recalcular)
from functions.laminas_entorno import DATOS, IMG  # noqa: E402,F401  (00_data/processed/laminas/ y 02_outputs/figures/laminas/)

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STORES = DIR_PROC  # carpeta de intermedios del repositorio

from functions import flujo_final as F  # noqa: E402  (base del modelo final: F.g, F.y, F.idx_tr, F.COLUMNAS, ...)
g = F.g
es_tr = np.zeros(len(g), dtype=bool)
es_tr[F.idx_tr] = True
cols = list(F.COLUMNAS["con"]) if isinstance(F.COLUMNAS, dict) else list(F.COLUMNAS)
R = {"columnas": cols, "variables": {}}
for c in cols:
    if c not in g.columns:
        R["variables"][c] = None
        continue
    v = pd.to_numeric(g[c], errors="coerce").astype(float)
    R["variables"][c] = {"entrenamiento": float(100 * v[es_tr].notna().mean()),
                         "prueba": float(100 * v[~es_tr].notna().mean()),
                         "promedio": float(np.nanmean(v)),
                         "promedio_original": None}
    # Para las variables en logaritmo (ln_x = log(1 + x)), también el promedio en la escala original
    base = c[3:]
    if c.startswith("ln_") and base in g.columns:
        R["variables"][c]["promedio_original"] = float(np.nanmean(pd.to_numeric(g[base], errors="coerce")))
# Puntaje ridge de texto del modelo final (fuera de pliegue en entrenamiento; ajustado con todo el entrenamiento en prueba)
f_ridge = STORES / "cache_puntajes_texto" / "ridge_final.npy"
if f_ridge.exists():
    pr = np.load(f_ridge).astype(float)
    R["variables"]["puntaje_ridge"] = {"entrenamiento": float(100 * np.isfinite(pr[es_tr]).mean()),
                                       "prueba": float(100 * np.isfinite(pr[~es_tr]).mean()),
                                       "promedio": float(np.nanmean(pr)), "promedio_original": float(np.nanmean(np.exp(pr))),
                                       "largo": int(len(pr))}
# Bases brutas
osm = pd.read_parquet(STORES / "osm" / "amenidades_osm.parquet", columns=["categoria"])
R["osm_objetos"] = {k: int(v) for k, v in osm["categoria"].value_counts().items()}
extra = STORES / "osm" / "amenidades_osm_extra.parquet"
if extra.exists():
    ox = pd.read_parquet(extra)
    col = "categoria" if "categoria" in ox.columns else ox.columns[0]
    R["osm_extra_objetos"] = {str(k): int(v) for k, v in ox[col].value_counts().items()}
elif (DATOS / "ppt_variables.json").exists():
    # El extracto con las 6 categorías adicionales de OpenStreetMap solo existe al recalcular (PS3_RECALCULAR=01 y 03):
    # sin él se conservan los conteos guardados.
    R["osm_extra_objetos"] = json.load(open(DATOS / "ppt_variables.json", encoding="utf-8")).get("osm_extra_objetos", {})
R["n_filas"] = {"entrenamiento": int(es_tr.sum()), "prueba": int((~es_tr).sum())}
salida = DATOS / "ppt_variables.json"
salida.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print("guardado", salida, len(cols), "columnas;", sum(v is None for v in R["variables"].values()), "sin dato en g")
print(cols)
