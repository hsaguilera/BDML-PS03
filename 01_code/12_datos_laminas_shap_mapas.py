"""12 · Datos de las láminas: valores SHAP del modelo ganador, distancias al vecino y mapas de entrenamiento y prueba.

Qué hace (antes document/presentaciones/datos_figuras.py)
    Reajusta el LightGBM ganador con una semilla (2026) y todo el entrenamiento para calcular sus valores SHAP en la
    prueba y en una muestra del entrenamiento; reúne las distancias al vecino de entrenamiento de cada esquema de
    validación, las predicciones fuera de pliegue (kNNDM y Chapinero fuera) y las bases de los mapas.

Salidas (00_data/processed/laminas/; se regeneran en cada corrida y no se versionan por tamaño)
    shap_prueba.parquet, shap_entrenamiento.parquet, valores_prueba.parquet, valores_entrenamiento.parquet,
    importancia_ganancia.csv, hiperparametros_lightgbm.csv, distancias_vecino.parquet, oof_lightgbm_con.parquet,
    prueba_mapa.parquet (con el pronóstico del envío 02), pct_fuera_rango.csv y entrenamiento_mapa.parquet
Insumos: scripts 06 a 09. Tiempo: unos 3 minutos.
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

PLIEGUES_CLASE = pliegues_clase(("aleatoria", "bloques", "clusters"))
a, b = PLIEGUES["Chapinero"]          # pliegue "Chapinero fuera", como lo dejaba cv_clase.py
DATOS.mkdir(exist_ok=True)
import geopandas as gpd  # noqa: E402
import lightgbm as lgb  # noqa: E402
t0 = time.time()
paso = lambda m: print(f"[{time.time() - t0:6.1f} s] {m}", flush=True)

# ---------------------------------------------------------------------------------------------------------------------
# 1. LightGBM ganador (variables con Catastro + puntaje ridge de texto): una semilla, valores SHAP
# ---------------------------------------------------------------------------------------------------------------------
nombres = COLUMNAS["con"] + ["puntaje_texto_ridge"]
X = np.c_[g[COLUMNAS["con"]].to_numpy(float), puntajes("final", idx_tr, idx_te)]
Xtr, ytr = X[idx_tr], y[idx_tr]
i_in, i_iv = mu.bloques_internos(xs[idx_tr], ys[idx_tr], semilla=SEMILLA)
p = dict(objective="l2", learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.7, reg_lambda=1.0, n_estimators=5000, random_state=SEMILLA, verbose=-1, n_jobs=4)
m = lgb.LGBMRegressor(**p).fit(Xtr[i_in], ytr[i_in], eval_set=[(Xtr[i_iv], ytr[i_iv])], eval_metric="l2",
                               callbacks=[lgb.early_stopping(100, verbose=False)])
p["n_estimators"] = max(int(m.best_iteration_ or 100), 50)
modelo = lgb.LGBMRegressor(**p).fit(Xtr, ytr)
paso(f"LightGBM final: {p['n_estimators']} árboles")
contrib_te = modelo.predict(X[idx_te], pred_contrib=True)
rng = np.random.default_rng(SEMILLA)
muestra = rng.choice(len(idx_tr), 6000, replace=False)
contrib_tr = modelo.predict(Xtr[muestra], pred_contrib=True)
pd.DataFrame(contrib_te[:, :-1], columns=nombres).assign(_base=contrib_te[:, -1], _pred=contrib_te.sum(1),
    property_id=g["property_id"].values[idx_te]).to_parquet(DATOS / "shap_prueba.parquet", index=False)
pd.DataFrame(contrib_tr[:, :-1], columns=nombres).assign(_base=contrib_tr[:, -1]).to_parquet(DATOS / "shap_entrenamiento.parquet", index=False)
pd.DataFrame(Xtr[muestra], columns=nombres).to_parquet(DATOS / "valores_entrenamiento.parquet", index=False)
pd.DataFrame(X[idx_te], columns=nombres).to_parquet(DATOS / "valores_prueba.parquet", index=False)
pd.DataFrame({"variable": nombres, "ganancia": modelo.booster_.feature_importance("gain")}).to_csv(DATOS / "importancia_ganancia.csv", index=False)
pd.DataFrame({"parametro": list(p), "valor": [str(v) for v in p.values()]}).to_csv(DATOS / "hiperparametros_lightgbm.csv", index=False)
paso("SHAP listo")

# ---------------------------------------------------------------------------------------------------------------------
# 2. Distancia al vecino de entrenamiento más cercano en cada esquema de validación (frente a la prueba real)
# ---------------------------------------------------------------------------------------------------------------------
xy = np.c_[xs, ys]
dist = {"Prueba real (Kaggle)": cKDTree(xy[idx_tr]).query(xy[idx_te])[0]}
for nombre, e in [("CV aleatoria", "aleatoria"), ("Bloques de 2 km", "bloques"), ("Clústeres k-means", "clusters")]:
    d = []
    for n, (esq, a, b, part) in PLIEGUES_CLASE.items():
        if esq == e:
            d.append(cKDTree(xy[a]).query(xy[b])[0])
    dist[nombre] = np.concatenate(d)
a, b = PLIEGUES["Chapinero"]
dist["Chapinero fuera"] = cKDTree(xy[a]).query(xy[b])[0]
fold_k, grupo_q, q_k, res_k, obj_k = es.knndm(xy[idx_tr], xy[idx_te], k=5)
dist["kNNDM"] = es.distancias_cv(xy[idx_tr], fold_k)
pd.concat([pd.DataFrame({"esquema": k, "distancia_m": v}) for k, v in dist.items()]).to_parquet(DATOS / "distancias_vecino.parquet", index=False)
paso("Distancias listas")

# ---------------------------------------------------------------------------------------------------------------------
# 3. Predicciones fuera de pliegue para errores y negocio
# ---------------------------------------------------------------------------------------------------------------------
rk = pd.read_parquet(DIR_MODELOS / "rk_knndm_oof_lightgbm_con.parquet")
oof = pd.DataFrame({"idx": rk["idx"], "esquema": "kNNDM", "pred": rk["base"], "real": rk["real"]})
L = pd.read_parquet(DIR_MODELOS / "cv_clase_oof_lightgbm_con.parquet")
oof = pd.concat([oof, L[L.esquema.isin(["clusters", "Chapinero (imitando el taller)"])][["idx", "esquema", "pred", "real"]]])
oof["localidad"] = g["localidad"].values[oof["idx"].values]
oof["area"] = g["area_imp"].values[oof["idx"].values]
oof["apartamento"] = g["es_apartamento"].values[oof["idx"].values]
oof.to_parquet(DATOS / "oof_lightgbm_con.parquet", index=False)
paso("Predicciones fuera de pliegue guardadas")

# ---------------------------------------------------------------------------------------------------------------------
# 4. Prueba: predicción final, coordenadas y variables fuera del rango del entrenamiento (p1-p99)
# ---------------------------------------------------------------------------------------------------------------------
final = pd.read_csv(DIR_PRED / "lightgbm_l2_eta005_hojas63_semillas5_catastro_x108.csv")
prueba = g.iloc[idx_te][["property_id", "lat", "lon", "localidad", "area_imp", "es_apartamento"]].copy()
prueba = prueba.merge(final.rename(columns={"price": "pred_final"}), on="property_id", how="left")
prueba["pred_final"] /= 1e6
VARS_RANGO = {"ln_dist_area_protegida_m": "Distancia a los cerros", "ln_n_embajada_1000m": "Embajadas a 1 km",
              "ln_n_hotel_1000m": "Hoteles a 1 km", "ln_n_oficina_1000m": "Oficinas a 1 km", "ln_vref_300m": "Valor del suelo a 300 m",
              "prop_estrato6_300m": "Manzanas estrato 6 a 300 m", "area_imp": "Área", "elevacion_m": "Elevación"}
fuera = {}
for v, et in VARS_RANGO.items():
    lo, hi = np.nanpercentile(g[v].values[idx_tr], [1, 99])
    val = g[v].values[idx_te]
    fuera[et] = (val < lo) | (val > hi)
    prueba[f"fuera_{v}"] = fuera[et]
prueba["n_fuera"] = np.sum(list(fuera.values()), axis=0)
prueba.to_parquet(DATOS / "prueba_mapa.parquet", index=False)
pd.Series({k: 100 * v.mean() for k, v in fuera.items()}).rename("pct_fuera_p1_p99").to_csv(DATOS / "pct_fuera_rango.csv")
ent = g.iloc[idx_tr][["lat", "lon", "localidad", "price"]].copy()
ent["precio_m2"] = g["price"].values[idx_tr] / 1e6 / g["area_imp"].values[idx_tr]
ent.to_parquet(DATOS / "entrenamiento_mapa.parquet", index=False)
paso("Datos de mapas y rango listos")
print(pd.Series({k: round(100 * v.mean(), 1) for k, v in fuera.items()}).to_string())
imp = pd.DataFrame({"variable": nombres, "shap_medio": np.abs(contrib_te[:, :-1]).mean(0)}).sort_values("shap_medio", ascending=False)
print(imp.head(30).round(1).to_string(index=False))
