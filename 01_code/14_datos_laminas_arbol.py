"""14 · Datos de las láminas: un árbol real del modelo ganador cuyos tres primeros niveles combinan variables distintas.

Qué hace (antes document/presentaciones/datos_arbol2.py)
    Ajusta el LightGBM ganador (252 árboles, semilla 2026) y elige, entre los primeros 60 árboles, el que usa más
    variables distintas en sus tres primeros niveles (lámina "Cada árbol hace preguntas de sí o no").

Salida: 00_data/processed/laminas/ppt_arbol2.json (se regenera en cada corrida). Tiempo: menos de 1 minuto.
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import json  # noqa: E402,F401
import os  # noqa: E402,F401
import re  # noqa: E402,F401
import sys  # noqa: E402,F401
import time  # noqa: E402,F401
from pathlib import Path  # noqa: E402,F401

# Base del modelo final (el mismo espacio de nombres que dejaba cv_clase.py en la carpeta original): g, y, idx_tr, idx_te,
# COLUMNAS, PLIEGUES, puntajes, mu (modelo_utils), vc (validacion_clase), np, pd, ...
from functions.flujo_final import *  # noqa: F401,F403,E402
from functions import knndm_kriging as es  # noqa: E402
from functions.entorno_ps3 import (ARCHIVO_LOCALIDADES, DIR_CACHE_TEXTO, DIR_CATASTRO, DIR_MODELOS, DIR_PRED, DIR_RAW,  # noqa: E402
                                   DIR_TAB, DIR_TEXTO)
from functions.laminas_entorno import DATOS  # noqa: E402  (00_data/processed/laminas/)

PLIEGUES_CLASE = pliegues_clase(("aleatoria",))
a, b = PLIEGUES["Chapinero"]          # pliegue "Chapinero fuera", como lo dejaba cv_clase.py
import lightgbm as lgb
nombres = COLUMNAS["con"] + ["puntaje_texto_ridge"]
P = dict(objective="l2", learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.7, reg_lambda=1.0, random_state=SEMILLA, verbose=-1, n_jobs=4)
X = np.c_[g[COLUMNAS["con"]].to_numpy(float), puntajes("final", idx_tr, idx_te)]
m = lgb.LGBMRegressor(**P, n_estimators=252).fit(X[idx_tr], y[idx_tr])
info = m.booster_.dump_model()["tree_info"]
def tipo(v):
    return {"exp": None}
def inv(v, t):
    if v == "puntaje_texto_ridge" or v.startswith(("ln_vref", "ln_avaluo")):
        return float(np.exp(t))
    if v.startswith("ln_"):
        return float(np.expm1(t))
    return float(t)
def nodo(n, p):
    if "leaf_value" in n or p == 3:
        return {"hoja": True, "valor": float(n.get("leaf_value", n.get("internal_value"))), "n": int(n.get("leaf_count", n.get("internal_count", 0))), "truncado": "leaf_value" not in n}
    v = nombres[n["split_feature"]]
    return {"hoja": False, "variable": v, "umbral": float(n["threshold"]), "umbral_legible": inv(v, float(n["threshold"])), "n": int(n["internal_count"]),
            "izq": nodo(n["left_child"], p + 1), "der": nodo(n["right_child"], p + 1)}
def vars_(n):
    return set() if n["hoja"] else {n["variable"]} | vars_(n["izq"]) | vars_(n["der"])
mejor = None
for k in range(1, 60):
    t = nodo(info[k]["tree_structure"], 0)
    vv = vars_(t)
    if mejor is None or len(vv) > mejor[1]:
        mejor = (k, len(vv), t)
k, nv, t = mejor
json.dump({"indice": k + 1, "arbol": t}, open(DATOS / "ppt_arbol2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
def ver(n, p=0):
    if n["hoja"]:
        print("  " * p + f"hoja {n['valor']:+.2f} (n={n['n']})"); return
    print("  " * p + f"{n['variable']} <= {n['umbral_legible']:.3f} (n={n['n']})"); ver(n["izq"], p + 1); ver(n["der"], p + 1)
print("árbol", k + 1, "con", nv, "variables distintas"); ver(t)
