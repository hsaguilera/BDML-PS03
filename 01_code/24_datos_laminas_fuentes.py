"""24 · Datos de las láminas: cifras descriptivas de las cinco fuentes de datos (avisos, texto, mapa, satélite y Catastro).

Qué hace (antes document/presentaciones/24_datos_laminas_fuentes.py)
    Con la base del modelo final (functions/flujo_final.py, sin validar ni entrenar) calcula medianas y porcentajes por
    grupo (prueba, entrenamiento fuera de Chapinero y entrenamiento en Chapinero), el precio mediano por decil del valor del
    suelo, y los datos con valor y promedios de las columnas brutas de Kaggle y de las 21 variables de los avisos.

Salida (00_data/processed/laminas/; se versiona porque la leen los scripts 21 y 22)
    ppt_fuentes.json
Insumos: scripts 01 a 06 (base del modelo final) y 00_data/train.csv y test.csv. Tiempo: menos de un minuto.
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


from functions import flujo_final as F  # noqa: E402  (base del modelo final: F.g, F.y, F.idx_tr, F.COLUMNAS, ...)
g = F.g.copy()
es_tr = np.zeros(len(g), dtype=bool)
es_tr[F.idx_tr] = True
chap = g["localidad"].astype(str).str.upper().str.contains("CHAPINERO").values
grupos = {"prueba": ~es_tr, "entrenamiento_resto": es_tr & ~chap, "entrenamiento_chapinero": es_tr & chap}
print({k: int(v.sum()) for k, v in grupos.items()})


def med(col, transf=None):
    v = g[col].astype(float)
    if transf is not None:
        v = transf(v)
    return {k: float(np.nanmedian(v[m])) for k, m in grupos.items()}


def pct(col):
    v = g[col].astype(float)
    return {k: float(100 * np.nanmean(v[m] > 0)) for k, m in grupos.items()}


R = {"n": {k: int(m.sum()) for k, m in grupos.items()}}
R["avisos"] = {"area_m2": med("area_imp"), "alcobas": med("bedrooms"), "banos": med("banos_imp"),
               "parqueaderos": med("parqueaderos_imp"), "pct_apartamento": pct("es_apartamento"),
               "avisos_500m": med("n_avisos_500m"), "pct_apartamentos_500m": med("prop_apartamentos_500m", lambda v: 100 * v)}
R["texto"] = {c: pct(c) for c in ["txt_ascensor", "txt_terraza_balcon", "txt_vigilancia", "txt_chimenea", "txt_gimnasio",
                                  "txt_conjunto", "txt_piscina", "txt_vista", "txt_deposito", "txt_estudio"]}
R["osm_conteos"] = {c: med(c) for c in ["n_oficina_1000m", "n_hotel_1000m", "n_embajada_1000m", "n_bar_cafe_1km",
                                        "n_restaurante_1km", "n_supermercado_1km"]}
R["osm_distancias"] = {c: med(c) for c in ["dist_transmilenio_m", "dist_area_protegida_m", "dist_centro_comercial_m",
                                           "dist_p93_m", "dist_parque_grande_m"]}
R["satelite"] = {c: med(c) for c in ["luces_nocturnas", "elevacion_m", "ndvi_200m", "altura_construida_m", "poblacion_100m",
                                     "pendiente_grados", "elevacion_relativa_1km"]}
mill = lambda v: np.exp(v) / 1e6  # noqa: E731
R["catastro"] = {"suelo_300m": med("ln_vref_300m", mill), "avaluo_ph_300m": med("ln_avaluo_ph_300m", mill),
                 "pisos_300m": med("pisos_300m"), "estrato_medio_300m": med("estrato_medio_300m"),
                 "pct_estrato6_300m": med("prop_estrato6_300m", lambda v: 100 * v)}
tr = g[es_tr].copy()
tr["precio"] = np.asarray(F.y)[es_tr]
tr["decil"] = pd.qcut(tr["ln_vref_300m"], 10, labels=False, duplicates="drop")
dec = tr.groupby("decil").agg(suelo=("ln_vref_300m", lambda v: float(np.median(np.exp(v)) / 1e6)),
                              precio=("precio", "median"), n=("precio", "size")).reset_index()
R["catastro_deciles"] = dec.to_dict(orient="records")
p99 = float(np.exp(np.nanpercentile(tr["ln_vref_300m"], 99)) / 1e6)
R["catastro_p99_suelo"] = p99
R["catastro_pct_prueba_sobre_p99"] = float(100 * np.nanmean(mill(g.loc[~es_tr, "ln_vref_300m"]) > p99))
VARS_AVISOS = ["es_apartamento", "bedrooms", "rooms", "area_imp", "log_area", "area_desde_texto", "area_falta",
               "area_inconsistente", "banos_imp", "banos_desde_texto", "parqueaderos_imp", "piso_texto", "estrato_texto",
               "antiguedad_texto", "tiempo", "x_km", "y_km", "coord_redondeada", "n_avisos_500m", "prop_apartamentos_500m",
               "area_mediana_500m"]
R["avisos_variables"] = {c: {"no_nulos": float(100 * g[c].notna().mean()), "promedio": float(np.nanmean(g[c].astype(float)))}
                         for c in VARS_AVISOS}
kag = pd.concat([pd.read_csv(DIR_RAW / "train.csv"), pd.read_csv(DIR_RAW / "test.csv")], ignore_index=True)
R["avisos_brutas_no_nulos"] = {c: float(100 * kag[c].notna().mean()) for c in
                               ["property_type", "bedrooms", "rooms", "surface_covered", "surface_total", "bathrooms",
                                "month", "year", "lat", "lon", "title", "description"]}
R["n_avisos_total"] = int(len(g))
tr_k, te_k = pd.read_csv(DIR_RAW / "train.csv"), pd.read_csv(DIR_RAW / "test.csv")
R["n_filas"] = {"entrenamiento": int(len(tr_k)), "prueba": int(len(te_k)), "columnas": int(tr_k.shape[1])}
R["precio_promedio_entrenamiento"] = float(tr_k["price"].mean() / 1e6)
R["avisos_variables_grupo"] = {c: {"entrenamiento": float(100 * g.loc[es_tr, c].notna().mean()),
                                   "prueba": float(100 * g.loc[~es_tr, c].notna().mean())} for c in VARS_AVISOS}
R["avisos_brutas_no_nulos_grupo"] = {c: {"entrenamiento": float(100 * tr_k[c].notna().mean()),
                                         "prueba": float(100 * te_k[c].notna().mean())}
                                     for c in ["price", "property_type", "bedrooms", "rooms", "surface_covered",
                                               "surface_total", "bathrooms", "month", "year", "lat", "lon", "title",
                                               "description"]}
salida = DATOS / "ppt_fuentes.json"
salida.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print("guardado", salida)
print(json.dumps({k: R[k] for k in ("n", "catastro_p99_suelo", "catastro_pct_prueba_sobre_p99")}, ensure_ascii=False))
