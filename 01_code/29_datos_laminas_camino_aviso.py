"""29 · Datos de las láminas: camino de un aviso por los 241 árboles del boosting.

Qué hace (antes document/presentaciones/29_datos_laminas_camino_aviso.py)
    Reconstruye en memoria el modelo de ilustración (la especificación final con una semilla, entrenada sin los 307 avisos
    de Chapinero, 241 árboles) y, para el apartamento de 125 m² de la cascada SHAP, guarda el camino por el árbol 1, lo que
    suma cada árbol y la profundidad de cada camino. Comprueba que coincide con el camino de ppt_tres_avisos.json.

Salida (00_data/processed/laminas/; se versiona porque la leen los scripts 21 y 22)
    etapas2a4_camino.json
Insumos: scripts 06 a 09 y 16. Tiempo: menos de un minuto.
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

from functions.flujo_final import *  # noqa: F401,F403,E402  (base del modelo final: g, y, idx_tr, idx_te, COLUMNAS, ...)
import lightgbm as lgb  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
tres = json.load(open(DATOS / "ppt_tres_avisos.json", encoding="utf-8"))
av = tres["avisos"][1]
PARADA = int(tres["parada"])
nombres = COLUMNAS["con"] + ["puntaje_texto_ridge"]
P = dict(objective="l2", learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.7, reg_lambda=1.0, random_state=SEMILLA, verbose=-1, n_jobs=4)
a, b = PLIEGUES["Chapinero"]
Xc = np.c_[g[COLUMNAS["con"]].to_numpy(float), puntajes("Chapinero", a, b)]
m_c = lgb.LGBMRegressor(**P, n_estimators=PARADA).fit(Xc[a], y[a])
i = int(np.where(g["property_id"].values[b] == av["property_id"])[0][0])
fila = Xc[b][[i]]
camino = [float(tres["base"])] + [float(m_c.predict(fila, num_iteration=k)[0]) for k in range(1, PARADA + 1)]
dif = max(abs(u - v) for u, v in zip(camino, av["camino"]))
print("diferencia máxima con el camino guardado:", dif)
assert dif < 1e-6
hojas = m_c.predict(fila, pred_leaf=True)[0]
info = m_c.booster_.dump_model()["tree_info"]


def buscar(nodo, hoja, ruta):
    if "leaf_index" in nodo:
        return ruta + [("hoja", nodo)] if nodo["leaf_index"] == hoja else None
    for lado in ("left_child", "right_child"):
        r = buscar(nodo[lado], hoja, ruta + [(lado, nodo)])
        if r:
            return r
    return None


def ruta_arbol(k):
    r = buscar(info[k]["tree_structure"], int(hojas[k]), [])
    pasos = []
    for (lado, nodo), sig in zip(r[:-1], r[1:]):
        hijo = sig[1]
        otro = nodo["right_child"] if lado == "left_child" else nodo["left_child"]
        cuenta = lambda x: int(x.get("leaf_count", x.get("internal_count", 0)))  # noqa: E731
        v_ = nombres[nodo["split_feature"]]
        pasos.append({"variable": v_, "umbral": float(nodo["threshold"]), "respuesta": "sí" if lado == "left_child" else "no",
                      "valor_aviso": None if not np.isfinite(fila[0, nodo["split_feature"]]) else float(fila[0, nodo["split_feature"]]),
                      "n_nodo": cuenta(nodo), "n_rama": cuenta(hijo), "n_otra": cuenta(otro)})
    return pasos, int(r[-1][1]["leaf_count"])


ruta1, n_hoja1 = ruta_arbol(0)
prof = [len(ruta_arbol(k)[0]) for k in range(PARADA)]
suma = [camino[k] - camino[k - 1] for k in range(1, PARADA + 1)]
R = {"property_id": av["property_id"], "real": av["real"], "pred": camino[-1], "base": camino[0], "parada": PARADA,
     "n_entrenamiento": int(len(a)), "camino": camino, "suma_por_arbol": suma, "hoja_por_arbol": [int(h) for h in hojas],
     "hojas_por_arbol": [int(t["num_leaves"]) for t in info], "profundidad": prof, "ruta_arbol1": ruta1,
     "n_hoja_arbol1": n_hoja1, "n_arbol1": int(info[0]["tree_structure"]["internal_count"]),
     "valores": {"area_imp": float(fila[0, nombres.index("area_imp")]), "banos_imp": float(fila[0, nombres.index("banos_imp")]),
                 "puntaje_mill": float(np.exp(fila[0, nombres.index("puntaje_texto_ridge")]))}}
json.dump(R, open(DATOS / "etapas2a4_camino.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("árbol 1:", [(p_["variable"], round(p_["umbral"], 3), p_["respuesta"], p_["n_rama"]) for p_ in ruta1], n_hoja1)
print("suma árboles 1 a 5:", [round(x, 2) for x in suma[:5]], "| total:", round(sum(suma), 1), "| pred", round(camino[-1], 1))
print("profundidad: min", min(prof), "max", max(prof), "| hojas por árbol:", min(R["hojas_por_arbol"]), max(R["hojas_por_arbol"]))
