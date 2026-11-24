"""19 · Datos de las láminas: error de las siete especificaciones por validación e importancia SHAP por variable y bloque.

Qué hace (no ajusta modelos)
    1. cv_familias.csv: MAE, sesgo, RMSE y sesgo en los avisos de más de 1.000 millones de cada especificación con la CV
       aleatoria, los clústeres k-means y Chapinero fuera (predicciones fuera de pliegue del script 07). Es la tabla de
       las láminas "Resultado", "Las mismas reglas" y del apéndice de error por familia.
    2. shap_importancia.csv y shap_bloques.csv: aporte absoluto medio de cada variable y de cada bloque (vivienda, texto,
       Catastro, entorno) en la prueba, con los valores SHAP del script 12.

Salidas (00_data/processed/laminas/; se regeneran en cada corrida)
    cv_familias.csv, shap_importancia.csv y shap_bloques.csv
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import numpy as np
import pandas as pd

from functions import flujo_final as F
from functions.entorno_ps3 import DIR_MODELOS, exigir, log
from functions.laminas_entorno import DATOS

exigir(DATOS / "shap_prueba.parquet", como="Corra antes el script 12.")

# --- 1. Error por especificación y validación ---------------------------------------------------------------------
FAMILIAS = [("lightgbm", "LightGBM (boosting)"), ("catboost", "CatBoost (boosting)"), ("random_forest", "Random Forest"),
            ("cart", "CART"), ("nn_atencion", "Red con atención"), ("elastic_net", "Elastic Net"),
            ("ols", "Regresión lineal")]
ESQUEMAS = {"aleatoria": "CV aleatoria", "clusters": "Clústeres k-means", "Chapinero (imitando el taller)": "Chapinero fuera"}
filas = []
for fam, etq in FAMILIAS:
    o = pd.read_parquet(DIR_MODELOS / f"cv_clase_oof_{fam}_con.parquet")
    for esq, nombre in ESQUEMAS.items():
        d = o[o["esquema"] == esq]
        if d.empty:
            continue
        e = d["pred"] - d["real"]
        filas.append({"familia": fam, "etq": etq, "esquema": nombre, "mae": e.abs().mean(), "sesgo": e.mean(),
                      "rmse": np.sqrt((e ** 2).mean()), "sesgo_alto": e[d["real"] > 1000].mean()})
pd.DataFrame(filas).to_csv(DATOS / "cv_familias.csv", index=False)
log(f"cv_familias.csv: {len(filas)} filas")

# --- 2. Importancia SHAP por variable y por bloque ------------------------------------------------------------------
shap = pd.read_parquet(DATOS / "shap_prueba.parquet")
nombres = F.COLUMNAS["con"] + ["puntaje_texto_ridge"]
BLOQUE = {**{v: "Vivienda" for v in F.ESTRUCTURALES}, **{v: "Texto" for v in F.TEXTO + ["puntaje_texto_ridge"]},
          **{v: "Catastro" for v in F.CATASTRO}, **{v: "Entorno (OSM, GEE, coordenadas)" for v in F.EXTERNAS + F.EXTRA + F.COORD}}
media_abs = shap[nombres].abs().mean()
media_abs.sort_values(ascending=False).to_csv(DATOS / "shap_importancia.csv")
bloques = media_abs.groupby(BLOQUE).sum().sort_values()
bloques.to_csv(DATOS / "shap_bloques.csv")
log("shap_bloques.csv: " + ", ".join(f"{k} {100 * v / bloques.sum():.0f} %" for k, v in bloques.items()))
