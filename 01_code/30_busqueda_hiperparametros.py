"""30 · Búsqueda al azar de hiperparámetros de LightGBM como diagnóstico (no cambia el modelo final ni los envíos).

Qué hace (antes document/presentaciones/30_busqueda_hiperparametros.py)
    Evalúa la configuración del modelo final y combinaciones sorteadas (semilla 2026) de tasa de aprendizaje, hojas, avisos
    por hoja, sorteo de avisos y de variables, y penalización, con las 162 variables, pérdida L2 y parada temprana con la
    partición interna de cada esquema, en Chapinero fuera y, mientras alcance el tiempo, en clústeres k-means.

Salida (00_data/processed/laminas/; se versiona)
    busqueda_hiperparametros.json

Recálculo
    Por defecto usa la salida guardada. Con PS3_RECALCULAR=30 la recalcula (tope de 55 minutos de cómputo; necesita los
    puntajes de texto de cache_puntajes_texto/, que functions/flujo_final.py recalcula cuando faltan).
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

from functions.entorno_ps3 import (ARCHIVO_AMENIDADES, ARCHIVO_LOCALIDADES, CENTRO, DIR_CACHE_TEXTO,  # noqa: E402,F401
                                   DIR_MODELOS, DIR_OSM, DIR_PRED, DIR_RAW, DIR_TEXTO, FACTOR, existen, log,
                                   recalcular)
from functions.laminas_entorno import DATOS, IMG  # noqa: E402,F401  (00_data/processed/laminas/ y 02_outputs/figures/laminas/)

import json
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONPATH"] = str(Path("01_code").resolve()) + os.pathsep + os.environ.get("PYTHONPATH", "")
if not recalcular("30") and existen(DATOS / "busqueda_hiperparametros.json"):
    log("Búsqueda de hiperparámetros: se usa la guardada (para recalcularla, PS3_RECALCULAR=30; unos 55 minutos)")
    sys.exit(0)
from functions.flujo_final import *  # noqa: F401,F403,E402  (base del modelo final: g, y, idx_tr, idx_te, COLUMNAS, ...)
PLIEGUES_CLASE = pliegues_clase(("clusters",))  # noqa: F405

sys.stdout.reconfigure(encoding="utf-8")
T0 = time.time()
TOPE = 55 * 60
SALIDA = DATOS / "busqueda_hiperparametros.json"
N_COMB = 35
rng = np.random.default_rng(2026)
FINAL = dict(learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0)
combs = [dict(FINAL, origen="modelo final")]
for _ in range(N_COMB):
    combs.append(dict(learning_rate=float(np.exp(rng.uniform(np.log(0.02), np.log(0.1)))),
                      num_leaves=int(round(np.exp(rng.uniform(np.log(15), np.log(127))))),
                      min_child_samples=int(rng.integers(10, 101)), subsample=float(rng.uniform(0.6, 1.0)),
                      colsample_bytree=float(rng.uniform(0.4, 1.0)), reg_lambda=float(rng.uniform(0, 10)), origen="al azar"))

Xbase = g[COLUMNAS["con"]].to_numpy(float)


def texto_cache(nombre):
    f = STORES / "cache_puntajes_texto" / f"ridge_{nombre}.npy"
    if not f.exists():
        raise FileNotFoundError(f)
    return np.load(f)


PL = {n_: v for n_, v in PLIEGUES_CLASE.items() if v[0] in ("clusters", "Chapinero (imitando el taller)")}
XF = {n_: np.c_[Xbase, texto_cache(n_)] for n_ in PL}


def ajuste(c, n_, Xf, a, b, part, yy, xx, yc, l2, l1, gr, semilla):
    from functions import modelo_utils as mu_
    p = {k: c[k] for k in ("learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree", "reg_lambda")}
    p["particion_interna"] = part
    pr = mu_.ajustar_predecir("lightgbm", Xf, yy, xx, yc, l2, l1, gr, None, a, b, False, False, p, semilla)
    return n_, pr


def tarea(k, n_):
    e, a, b, part = PL[n_]
    return delayed(ajuste)(combs[k], n_, XF[n_], a, b, part, y, xs, ys, lpm2, lp, grupo, SEMILLA)


def guardar(res):
    json.dump({"rangos": {"learning_rate": [0.02, 0.1], "num_leaves": [15, 127], "min_child_samples": [10, 100],
                          "subsample": [0.6, 1.0], "colsample_bytree": [0.4, 1.0], "reg_lambda": [0, 10]},
               "semilla_sorteo": 2026, "semilla_modelo": SEMILLA, "n_combinaciones_al_azar": N_COMB,
               "minutos": (time.time() - T0) / 60, "resultados": res}, open(SALIDA, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


res = [dict(c) for c in combs]
CH = "Chapinero"
# Tanda 1: Chapinero fuera, 6 combinaciones en paralelo
for i0 in range(0, len(combs), 6):
    if time.time() - T0 > TOPE:
        break
    lote = list(range(i0, min(i0 + 6, len(combs))))
    out = Parallel(n_jobs=6)(tarea(k, CH) for k in lote)
    a, b = PL[CH][1], PL[CH][2]
    for k, (_, pr) in zip(lote, out):
        err = pr - y[b]
        res[k]["mae_chapinero"] = float(np.abs(err).mean())
        res[k]["sesgo_chapinero"] = float(err.mean())
    guardar(res)
    log(f"Chapinero fuera: {lote[-1] + 1} de {len(combs)} combinaciones")
# Tanda 2: clústeres k-means, combinación por combinación (10 pliegues en paralelo)
nombres_cl = [n_ for n_, v in PL.items() if v[0] == "clusters"]
for k in range(len(combs)):
    if time.time() - T0 > TOPE:
        log("tope de tiempo: clústeres no evaluado para el resto")
        break
    out = Parallel(n_jobs=5)(tarea(k, n_) for n_ in nombres_cl)
    err = np.concatenate([pr - y[PL[n_][2]] for n_, pr in out])
    res[k]["mae_clusters"] = float(np.abs(err).mean())
    guardar(res)
    log(f"clústeres: combinación {k + 1} de {len(combs)} | MAE {res[k]['mae_clusters']:.1f}")
guardar(res)
log("fin")
