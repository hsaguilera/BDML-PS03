"""13 · Datos de las láminas: el algoritmo con las estimaciones reales, los pliegues en el mapa y la descripción.

Qué hace (antes document/presentaciones/datos_ppt.py)
    Primer árbol del modelo ganador; un apartamento de ejemplo y la curva de aprendizaje del modelo entrenado sin
    Chapinero; un pliegue de cada esquema en el mapa; descripción de los datos (faltantes, precios); errores con signo
    en kNNDM y en Chapinero fuera.

Salidas (00_data/processed/laminas/; se regeneran en cada corrida)
    ppt_arbol.json, ppt_ejemplo_boosting.json, ppt_curva_aprendizaje.csv, ppt_camino_prediccion.csv,
    ppt_pliegues_mapa.parquet, ppt_descripcion.json, ppt_faltantes.csv, ppt_precios_train.parquet,
    ppt_errores_knndm.parquet y ppt_errores_chapinero.parquet
Insumos: scripts 06 a 09. Tiempo: unos 2 minutos.
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
import lightgbm as lgb  # noqa: E402

nombres = COLUMNAS["con"] + ["puntaje_texto_ridge"]
P = dict(objective="l2", learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.7, reg_lambda=1.0, random_state=SEMILLA, verbose=-1, n_jobs=4)

# ---------------------------------------------------------------------------------------------------------------------
# 1. El primer árbol del modelo ganador (entrenado con todo el entrenamiento), tres niveles
# ---------------------------------------------------------------------------------------------------------------------
X = np.c_[g[COLUMNAS["con"]].to_numpy(float), puntajes("final", idx_tr, idx_te)]
modelo = lgb.LGBMRegressor(**P, n_estimators=252).fit(X[idx_tr], y[idx_tr])
arbol = modelo.booster_.dump_model()["tree_info"][0]["tree_structure"]
INV = {}  # cómo volver a unidades legibles
for j, v in enumerate(nombres):
    if v == "puntaje_texto_ridge" or v in ("ln_vref_300m", "ln_vref_manzana", "ln_avaluo_ph_300m", "ln_avaluo_ph_manzana"):
        INV[v] = "exp"
    elif v.startswith("ln_"):
        INV[v] = "expm1"
    else:
        INV[v] = "id"


def nodo(n, prof):
    if "leaf_value" in n or prof == 3:
        valor = n.get("leaf_value", n.get("internal_value"))
        return {"hoja": True, "valor": float(valor), "n": int(n.get("leaf_count", n.get("internal_count", 0))),
                "truncado": "leaf_value" not in n}
    v = nombres[n["split_feature"]]
    thr = float(n["threshold"])
    legible = {"exp": np.exp(thr), "expm1": np.expm1(thr), "id": thr}[INV[v]]
    return {"hoja": False, "variable": v, "umbral": thr, "umbral_legible": float(legible), "n": int(n["internal_count"]),
            "izq": nodo(n["left_child"], prof + 1), "der": nodo(n["right_child"], prof + 1)}


json.dump({"arbol": nodo(arbol, 0), "base": float(modelo.booster_.dump_model()["tree_info"][0].get("shrinkage", 0.05)),
           "media": float(y[idx_tr].mean())}, open(DATOS / "ppt_arbol.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("árbol listo")

# ---------------------------------------------------------------------------------------------------------------------
# 2. Boosting en acción con Chapinero fuera: curva de aprendizaje y la predicción de un apartamento árbol a árbol
# ---------------------------------------------------------------------------------------------------------------------
a, b = PLIEGUES["Chapinero"]
Xc = np.c_[g[COLUMNAS["con"]].to_numpy(float), puntajes("Chapinero", a, b)]
i_in, i_iv = mu.bloques_internos(xs[a], ys[a], semilla=SEMILLA)
m_es = lgb.LGBMRegressor(**P, n_estimators=5000).fit(Xc[a][i_in], y[a][i_in], eval_set=[(Xc[a][i_iv], y[a][i_iv])], eval_metric="l1",
                                                     callbacks=[lgb.early_stopping(100, verbose=False)])
parada = int(m_es.best_iteration_)
m_c = lgb.LGBMRegressor(**P, n_estimators=800).fit(Xc[a], y[a], eval_set=[(Xc[a], y[a]), (Xc[b], y[b])], eval_metric="l1")
ev = m_c.evals_result_
k_ev = list(ev.keys())
curva = pd.DataFrame({"arboles": np.arange(1, 801), "mae_entrenamiento": ev[k_ev[0]]["l1"], "mae_chapinero": ev[k_ev[1]]["l1"]})
curva.to_csv(DATOS / "ppt_curva_aprendizaje.csv", index=False)
# un apartamento de Chapinero con precio típico
cand = pd.DataFrame({"idx": b, "real": y[b], "apto": g["es_apartamento"].values[b], "area": g["area_imp"].values[b]})
cand["pred"] = m_c.predict(Xc[b], num_iteration=parada)
cand = cand[(cand["apto"] == 1) & (cand["real"] >= 1000)]
cand["err_rel"] = (cand["pred"] - cand["real"]).abs() / cand["real"]
cand = cand[cand["err_rel"] < 0.08].sort_values("real")
i_ej = int(cand.iloc[len(cand) // 2]["idx"])
ks = np.unique(np.r_[np.arange(0, 21), np.arange(25, 801, 25), parada]).astype(int)
camino = [float(y[a].mean())] + [float(m_c.predict(Xc[[i_ej]], num_iteration=int(k))[0]) for k in ks[1:]]
pd.DataFrame({"arboles": ks, "prediccion": camino}).to_csv(DATOS / "ppt_camino_prediccion.csv", index=False)
ejemplo = {"real": float(y[i_ej]), "area": float(g["area_imp"].values[i_ej]), "alcobas": float(g["bedrooms"].values[i_ej]),
           "banos": float(g["banos_imp"].values[i_ej]) if np.isfinite(g["banos_imp"].values[i_ej]) else None,
           "pred_parada": float(m_c.predict(Xc[[i_ej]], num_iteration=parada)[0]), "parada": parada,
           "mae_parada": float(curva.loc[parada - 1, "mae_chapinero"]), "mae_800": float(curva["mae_chapinero"].iloc[-1]),
           "mae_min": float(curva["mae_chapinero"].min()), "arbol_min": int(curva["mae_chapinero"].idxmin() + 1),
           "property_id": str(g["property_id"].values[i_ej])}
json.dump(ejemplo, open(DATOS / "ppt_ejemplo_boosting.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("boosting listo", ejemplo)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Pliegues de cada esquema en el mapa (un pliegue de validación resaltado)
# ---------------------------------------------------------------------------------------------------------------------
xt, yt = xs[idx_tr], ys[idx_tr]
loc_tr = g["localidad"].values[idx_tr]
plg = {"CV aleatoria": vc.aleatoria_cv(len(idx_tr)), "Bloques de 2 km": vc.bloques_cv(xt, yt),
       "Clústeres k-means": vc.clusters_cv(xt, yt)}
fold_k, _, _, _, _ = es.knndm(np.c_[xt, yt], np.c_[xs[idx_te], ys[idx_te]], k=5)
filas = []
for nombre, pl in plg.items():
    val = np.zeros(len(idx_tr), bool)
    val[pl[0][1]] = True
    filas.append(pd.DataFrame({"esquema": nombre, "lon": g["lon"].values[idx_tr], "lat": g["lat"].values[idx_tr], "validacion": val}))
filas.append(pd.DataFrame({"esquema": "kNNDM", "lon": g["lon"].values[idx_tr], "lat": g["lat"].values[idx_tr], "validacion": fold_k == 0}))
filas.append(pd.DataFrame({"esquema": "Chapinero fuera", "lon": g["lon"].values[idx_tr], "lat": g["lat"].values[idx_tr],
                           "validacion": loc_tr == "Chapinero"}))
pd.concat(filas).to_parquet(DATOS / "ppt_pliegues_mapa.parquet", index=False)
print("pliegues listos")

# ---------------------------------------------------------------------------------------------------------------------
# 4. Descripción de los datos
# ---------------------------------------------------------------------------------------------------------------------
kag_tr = pd.read_csv(DIR_RAW / "train.csv")
kag_te = pd.read_csv(DIR_RAW / "test.csv")
falt = []
for c, et in [("surface_total", "Área total"), ("surface_covered", "Área cubierta"), ("rooms", "Ambientes"), ("bedrooms", "Alcobas"),
              ("bathrooms", "Baños"), ("description", "Descripción"), ("title", "Título")]:
    falt.append({"variable": et, "entrenamiento": 100 * kag_tr[c].isna().mean(), "prueba": 100 * kag_te[c].isna().mean()})
tv = pd.read_parquet(DIR_TEXTO / "texto_variables.parquet")
d_tr = kag_tr[["property_id"]].merge(tv[["property_id", "area_imp", "banos_imp"]], on="property_id")
d_te = kag_te[["property_id"]].merge(tv[["property_id", "area_imp", "banos_imp"]], on="property_id")
falt.append({"variable": "Área, tras completarla con el texto", "entrenamiento": 100 * d_tr["area_imp"].isna().mean(),
             "prueba": 100 * d_te["area_imp"].isna().mean()})
falt.append({"variable": "Baños, tras completarlos con el texto", "entrenamiento": 100 * d_tr["banos_imp"].isna().mean(),
             "prueba": 100 * d_te["banos_imp"].isna().mean()})
pd.DataFrame(falt).to_csv(DATOS / "ppt_faltantes.csv", index=False)
desc = {"n_train": len(kag_tr), "n_test": len(kag_te), "precio_min": float(kag_tr["price"].min() / 1e6),
        "precio_max": float(kag_tr["price"].max() / 1e6), "precio_mediana": float(kag_tr["price"].median() / 1e6),
        "precio_media": float(kag_tr["price"].mean() / 1e6),
        "pct_apto_train": float(100 * (kag_tr["property_type"] == "Apartamento").mean()),
        "pct_apto_test": float(100 * (kag_te["property_type"] == "Apartamento").mean()),
        "anios_train": kag_tr["year"].value_counts().sort_index().to_dict(), "anios_test": kag_te["year"].value_counts().sort_index().to_dict(),
        "pct_coord_redondeada_train": None}
json.dump(desc, open(DATOS / "ppt_descripcion.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
pd.DataFrame({"precio": kag_tr["price"].values / 1e6, "chapinero": loc_tr == "Chapinero"}).to_parquet(DATOS / "ppt_precios_train.parquet", index=False)
print("descripción lista", desc)

# ---------------------------------------------------------------------------------------------------------------------
# 5. Dónde se equivoca: validación kNNDM (distancia al vecino como en Kaggle), predicción final estirada
# ---------------------------------------------------------------------------------------------------------------------
rk = pd.read_parquet(DIR_MODELOS / "rk_knndm_oof_lightgbm_con.parquet")
o = pd.DataFrame({"idx": rk["idx"].values, "real": rk["real"].values, "pred": 654 + 1.08 * (rk["base"].values - 654)})
o["dist_vecino"] = es.distancias_cv(np.c_[xt, yt], fold_k)
o["area"] = g["area_imp"].values[o["idx"]]
o["tipo"] = np.where(g["es_apartamento"].values[o["idx"]] == 1, "Apartamento", "Casa")
o["estrato"] = g["estrato_manzana"].values[o["idx"]]
o["coord_redondeada"] = g["coord_redondeada"].values[o["idx"]]
o["largo_texto"] = g["texto_normalizado"].str.split().str.len().values[o["idx"]] if "texto_normalizado" in g.columns else np.nan
o["error"] = o["pred"] - o["real"]
o["error_pct"] = 100 * o["error"] / o["real"]
o.to_parquet(DATOS / "ppt_errores_knndm.parquet", index=False)
ch = pd.read_parquet(DIR_MODELOS / "cv_clase_oof_lightgbm_con.parquet")
ch = ch[ch["esquema"].str.startswith("Chapinero")].copy()
ch["pred"] = 654 + 1.08 * (ch["pred"] - 654)
ch["lon"], ch["lat"] = g["lon"].values[ch["idx"]], g["lat"].values[ch["idx"]]
ch["area"] = g["area_imp"].values[ch["idx"]]
ch["error"] = ch["pred"] - ch["real"]
ch.to_parquet(DATOS / "ppt_errores_chapinero.parquet", index=False)
print("errores listos")
