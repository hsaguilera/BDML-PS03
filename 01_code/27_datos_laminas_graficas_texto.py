"""27 · Datos de las láminas: regresión con tópicos frente a regresión con palabras y peso de un tópico por quintil de precio.

Qué hace (antes document/presentaciones/27_datos_laminas_graficas_texto.py)
    Error cuadrático medio fuera de pliegue de ln(precio) en el entrenamiento, para una rejilla de penalizaciones, con
    ridge sobre los 10 pesos de tópico y ridge sobre la matriz TF-IDF (GroupKFold de 5 por grupo de texto, el mismo esquema
    del puntaje del modelo final, que usa alfa = 1); y peso de un tópico por quintil de precio. Solo diagnóstico.

Salida (00_data/processed/laminas/; se versiona porque la leen los scripts 21 y 22)
    ppt_graficas_clase.json
Insumos: script 06. Tiempo: unos minutos.
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

from functions.entorno_ps3 import (ARCHIVO_AMENIDADES, ARCHIVO_LOCALIDADES, CENTRO, DIR_CACHE_TEXTO,  # noqa: E402,F401
                                   DIR_MODELOS, DIR_OSM, DIR_PRED, DIR_RAW, DIR_TEXTO, FACTOR, existen, log,
                                   recalcular)
from functions.laminas_entorno import DATOS, IMG  # noqa: E402,F401  (00_data/processed/laminas/ y 02_outputs/figures/laminas/)

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

TXT = DIR_TEXTO
t0 = time.time()

X = sp.load_npz(TXT / "tfidf.npz").tocsr()
filas = pd.read_parquet(TXT / "dtm_filas.parquet")["property_id"].values
tv = pd.read_parquet(TXT / "texto_variables.parquet").set_index("property_id")
tr = pd.read_csv(DIR_RAW / "train.csv", usecols=["property_id", "price"])
pos = pd.Series(np.arange(len(filas)), index=filas)
i_tr = pos.loc[tr["property_id"].values].values
Xw = X[i_tr]
T = tv.loc[tr["property_id"].values, [f"topico_{k:02d}" for k in range(10)]].to_numpy()
grupos = tv.loc[tr["property_id"].values, "grupo_texto"].values
lp = np.log(tr["price"].to_numpy() / 1e6)
pliegues = list(GroupKFold(5).split(Xw, groups=grupos))
R = {"n": int(len(lp)), "var_lp": float(lp.var())}


def curva(M, alfas):
    res = []
    for a in alfas:
        err = np.zeros(len(lp))
        for f_a, f_b in pliegues:
            m = Ridge(alpha=a).fit(M[f_a], lp[f_a])
            err[f_b] = lp[f_b] - m.predict(M[f_b])
        mse = [float(np.mean(err[f_b] ** 2)) for _, f_b in pliegues]
        res.append({"alfa": float(a), "mse": float(np.mean(err ** 2)), "se": float(np.std(mse) / np.sqrt(len(mse)))})
        print(f"[{time.time() - t0:6.0f} s] alfa {a:g}: MSE {res[-1]['mse']:.4f}", flush=True)
    return res


R["palabras"] = curva(Xw, [0.03, 0.1, 0.3, 1, 3, 10, 30, 100])
R["topicos"] = curva(T, [0.0001, 0.001, 0.01, 0.1, 1, 10, 100])

# Peso del tópico 5 (cercanía a comercio) y del tópico 8 por quintil de precio, en el entrenamiento
q = pd.qcut(tr["price"] / 1e6, 5, labels=False)
lim = pd.qcut(tr["price"] / 1e6, 5).cat.categories
R["quintiles"] = [{"q": int(k), "desde": float(lim[k].left), "hasta": float(lim[k].right)} for k in range(5)]
for k in (5, 8, 7):
    v = tv.loc[tr["property_id"].values, f"topico_{k:02d}"].to_numpy()
    R[f"caja_topico_{k}"] = [{"q": int(j), "p10": float(np.percentile(v[q == j], 10)),
                              "p25": float(np.percentile(v[q == j], 25)), "p50": float(np.percentile(v[q == j], 50)),
                              "p75": float(np.percentile(v[q == j], 75)), "p90": float(np.percentile(v[q == j], 90)),
                              "media": float(v[q == j].mean())} for j in range(5)]
json.dump(R, open(DATOS / "ppt_graficas_clase.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("guardado", round(time.time() - t0), "s")
