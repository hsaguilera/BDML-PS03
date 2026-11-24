"""17 · Datos de las láminas: manzanas de estrato 6 en Chapinero y ganancia por oferta según la precisión del modelo.

Qué hace (antes document/presentaciones/datos_ppt4.py; no reentrena modelos)
    1. Manzanas de estrato 6 en Chapinero y en el resto (capa de estratificación de Catastro y localidades).
    2. Ganancia por oferta con margen fijo de 10 % según la precisión del modelo (error logarítmico por lambda).

Salidas (00_data/processed/laminas/; se versionan porque usan Catastro)
    ppt_ganancia_precision.csv y la actualización de ppt_varios2.json (script 16)
Recálculo: con PS3_RECALCULAR=02,16,17 (requiere las capas de Catastro).
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import json
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from functions.entorno_ps3 import ARCHIVO_LOCALIDADES, DIR_CATASTRO, existen, log, recalcular
from functions.laminas_entorno import DATOS

if not (recalcular("17") or recalcular("16")) and existen(DATOS / "ppt_ganancia_precision.csv"):
    log("Se usan los datos guardados de este paso (necesita Catastro; para recalcular: PS3_RECALCULAR=02,16,17)")
    sys.exit(0)
R = json.load(open(DATOS / "ppt_varios2.json", encoding="utf-8"))

loc = gpd.read_file(ARCHIVO_LOCALIDADES).to_crs(9377)
est = gpd.read_parquet(DIR_CATASTRO / "estratificacion_manzanas.parquet").to_crs(9377)
est = est[est["ESTRATO"].between(1, 6)].copy()
est["geometry"] = est.geometry.representative_point()
est = gpd.sjoin(est, loc[["localidad", "geometry"]], how="inner", predicate="within")
for k, s in (("Chapinero", est["localidad"] == "Chapinero"), ("Resto de Bogotá", est["localidad"] != "Chapinero")):
    R[f"catastro_{k}"].update({"manzanas_con_estrato": int(s.sum()), "pct_estrato6": float((est.loc[s, "ESTRATO"] == 6).mean() * 100)})
print({k: v for k, v in R.items() if k.startswith("catastro")})

oof = pd.read_parquet(DATOS / "oof_lightgbm_con.parquet")
oof["pf"] = 654 + 1.08 * (oof["pred"] - 654)
filas = []
for nombre, o in (("kNNDM", oof[oof["esquema"] == "kNNDM"]), ("Chapinero fuera", oof[oof["esquema"].str.startswith("Chapinero")])):
    V, le = o["real"].to_numpy(), np.log(o["pf"].to_numpy() / o["real"].to_numpy())
    for lam in np.round(np.arange(1.0, 0.19, -0.05), 2):
        P = V * np.exp(lam * le)
        em = float(np.median(np.abs(P / V - 1)) * 100)
        for dd in (0.05, 0.10):
            Of = 0.9 * P
            acc = Of >= (1 - dd) * V
            filas.append({"validacion": nombre, "lambda": lam, "error_mediano_pct": em, "descuento": dd,
                          "aceptadas_pct": 100 * acc.mean(), "ganancia_por_oferta": (V - Of)[acc].sum() / len(V),
                          "pagan_de_mas_pct_compras": 100 * (acc & (Of > V)).sum() / max(acc.sum(), 1)})
f = pd.DataFrame(filas)
f.to_csv(DATOS / "ppt_ganancia_precision.csv", index=False)
for (n_, dd), t in f.groupby(["validacion", "descuento"]):
    t = t.sort_values("error_mediano_pct")
    pos = t[t["ganancia_por_oferta"] > 0]
    R[f"equilibrio_m10_{n_}_{dd}"] = float(pos["error_mediano_pct"].max()) if len(pos) else None
    print(n_, dd, t[["error_mediano_pct", "ganancia_por_oferta", "aceptadas_pct"]].round(1).to_string(index=False))
json.dump(R, open(DATOS / "ppt_varios2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print({k: v for k, v in R.items() if k.startswith("equilibrio")})
