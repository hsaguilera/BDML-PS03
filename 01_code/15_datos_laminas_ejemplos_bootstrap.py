"""15 · Datos de las láminas: aviso de ejemplo, cascada SHAP, calibración, maldición del ganador y bootstrap.

Qué hace (antes document/presentaciones/datos_ppt2.py)
    1. Un aviso real de Chapinero para ilustrar texto como datos, y verificación de que el texto no trae el precio.
    2. Cascada SHAP del apartamento del ejemplo de boosting (modelo entrenado sin Chapinero).
    3. Calibración por decil de la predicción en kNNDM y en Chapinero fuera.
    4. Chapinero fuera: predicho frente a real y error según el número de variables fuera del rango del entrenamiento.
    5. Maldición del ganador y 100 ofertas simuladas.
    6. Intervalos de confianza por bootstrap (Chapinero fuera, 307 avisos) del MAE y sus diferencias.

Salidas (00_data/processed/laminas/; se regeneran en cada corrida)
    ppt_texto_candidatos.parquet, ppt_cascada_ejemplo.json, ppt_calibracion.csv, ppt_chapinero_fuera.parquet,
    ppt_regla_rango.csv, ppt_maldicion.csv, ppt_cien_ofertas.csv, ppt_bootstrap.json y ppt_varios.json
Insumos: scripts 06 a 09 y 13. Tiempo: unos 2 minutos.
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

R = {}
nombres = COLUMNAS["con"] + ["puntaje_texto_ridge"]
P = dict(objective="l2", learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.7, reg_lambda=1.0, random_state=SEMILLA, verbose=-1, n_jobs=4)

# 1. Texto: ejemplo real y verificación de que el precio no aparece en el texto
kag = pd.concat([pd.read_csv(DIR_RAW / "train.csv"), pd.read_csv(DIR_RAW / "test.csv")])[["property_id", "title", "description"]]
d = g[["property_id", "localidad", "muestra", "price", "area_imp", "bedrooms", "banos_imp", "area_texto", "parqueaderos_texto"] +
      [c for c in g.columns if c.startswith("txt_")]].merge(kag, on="property_id", how="left")
punt = np.load(DIR_CACHE_TEXTO / "ridge_final.npy")
d["puntaje_millones"] = np.exp(punt)
desc = d["description"].fillna("").str.lower()
R["pct_menciona_millones_con_cifra"] = float(100 * desc.str.contains(r"\d[\d\.,]*\s*(millones|mill|mm)\b").mean())
R["pct_menciona_precio_palabra"] = float(100 * desc.str.contains(r"\bprecio\b|\bvalor\b").mean())
R["pct_mascara_x"] = float(100 * d["description"].fillna("").str.contains(r"\bX{2,}\b|\bX\b").mean())
c = d[(d["localidad"] == "Chapinero") & (d["muestra"] == "train")].copy()
t = c["description"].fillna("")
cond = (t.str.lower().str.contains("terraza") & t.str.lower().str.contains("parqueadero") & c["area_texto"].notna()
        & t.str.len().between(200, 900) & ~t.str.contains(r"\d{7,}|@|www|http", case=False))
c = c[cond].sort_values("price")
c.to_parquet(DATOS / "ppt_texto_candidatos.parquet", index=False)
R["n_candidatos_texto"] = int(len(c))

# 2. Cascada SHAP del apartamento del ejemplo de boosting (modelo sin Chapinero)
ej = json.load(open(DATOS / "ppt_ejemplo_boosting.json", encoding="utf-8"))
a, b = PLIEGUES["Chapinero"]
Xc = np.c_[g[COLUMNAS["con"]].to_numpy(float), puntajes("Chapinero", a, b)]
m_c = lgb.LGBMRegressor(**P, n_estimators=ej["parada"]).fit(Xc[a], y[a])
i_ej = int(np.where(g["property_id"].values == ej["property_id"])[0][0])
contrib = m_c.predict(Xc[[i_ej]], pred_contrib=True)[0]
cas = pd.Series(contrib[:-1], index=nombres).sort_values(key=np.abs, ascending=False)
json.dump({"base": float(contrib[-1]), "pred": float(contrib.sum()), "real": float(y[i_ej]),
           "contribuciones": {k: float(v) for k, v in cas.head(8).items()}, "resto": float(cas.iloc[8:].sum()),
           "n_resto": int(len(cas) - 8), "valores": {k: float(Xc[i_ej, nombres.index(k)]) for k in cas.head(8).index}},
          open(DATOS / "ppt_cascada_ejemplo.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# 3. Calibración por decil de la predicción
oof = pd.read_parquet(DATOS / "oof_lightgbm_con.parquet")
oof["pf"] = 654 + 1.08 * (oof["pred"] - 654)
cal = []
for esq, n in [("kNNDM", 10), ("Chapinero (imitando el taller)", 5)]:
    o = oof[oof["esquema"] == esq].copy()
    o["q"] = pd.qcut(o["pf"], n, labels=False)
    gq = o.groupby("q").agg(pred_media=("pf", "mean"), real_media=("real", "mean"), n=("real", "size")).reset_index()
    gq["esquema"] = esq
    cal.append(gq)
pd.concat(cal).to_csv(DATOS / "ppt_calibracion.csv", index=False)

# 4. Chapinero fuera: predicho frente a real; error según variables fuera del rango del entrenamiento sin Chapinero
ch = oof[oof["esquema"].str.startswith("Chapinero")].copy()
VARS = ["ln_dist_area_protegida_m", "ln_n_embajada_1000m", "ln_n_hotel_1000m", "ln_n_oficina_1000m", "ln_vref_300m",
        "prop_estrato6_300m", "area_imp", "elevacion_m"]
fuera = np.zeros(len(ch), int)
for v in VARS:
    lo, hi = np.nanpercentile(g[v].values[a], [1, 99])
    val = g[v].values[ch["idx"].values]
    fuera += ((val < lo) | (val > hi)).astype(int)
ch["n_fuera"] = fuera
ch["err_pct"] = 100 * (ch["pf"] - ch["real"]) / ch["real"]
ch["cat_fuera"] = np.where(ch["n_fuera"] >= 2, "Dos o más fuera", np.where(ch["n_fuera"] == 1, "Una fuera", "Dentro del rango"))
ch[["idx", "real", "pf", "n_fuera", "cat_fuera", "err_pct"]].to_parquet(DATOS / "ppt_chapinero_fuera.parquet", index=False)
reg = ch.groupby("cat_fuera").agg(n=("real", "size"), err_abs_pct_mediano=("err_pct", lambda s: s.abs().median()),
                                  sesgo_pct=("err_pct", "mean"), mae=("pf", lambda s: 0)).reset_index()
reg["mae"] = [np.mean(np.abs(ch.loc[ch["cat_fuera"] == k, "pf"] - ch.loc[ch["cat_fuera"] == k, "real"])) for k in reg["cat_fuera"]]
reg.to_csv(DATOS / "ppt_regla_rango.csv", index=False)
print(reg.round(1).to_string(index=False))

# 5. Maldición del ganador y 100 ofertas (oferta = (1 - m) predicción; el vendedor acepta si oferta >= (1 - d) valor)
k = oof[oof["esquema"] == "kNNDM"]
dist = []
for nombre, o in [("kNNDM", k), ("Chapinero fuera", ch)]:
    V, Pr = o["real"].to_numpy(), o["pf"].to_numpy()
    for m in (0.0, 0.10, 0.20):
        for dd in (0.05, 0.10):
            Of = (1 - m) * Pr
            acc = Of >= (1 - dd) * V
            dist.append({"validacion": nombre, "margen": m, "descuento": dd, "ofertas": 100, "aceptadas": 100 * acc.mean(),
                         "con_sobreprecio": 100 * (acc & (Of > V)).mean(),
                         "ganancia_por_100_ofertas": (V - Of)[acc].sum() / len(V) * 100,
                         "ganancia_media_por_compra": (V - Of)[acc].mean() if acc.any() else np.nan})
pd.DataFrame(dist).to_csv(DATOS / "ppt_cien_ofertas.csv", index=False)
V, Pr = k["real"].to_numpy(), k["pf"].to_numpy()
err = 100 * (Pr / V - 1)
acc = 0.9 * Pr >= 0.9 * V
bins = np.arange(-60, 65, 5)
h_todas = np.histogram(np.clip(err, -59.9, 59.9), bins)[0] / len(err) * 100
h_acep = np.histogram(np.clip(err[acc], -59.9, 59.9), bins)[0] / acc.sum() * 100
pd.DataFrame({"desde": bins[:-1], "hasta": bins[1:], "todas": h_todas, "aceptadas": h_acep}).to_csv(DATOS / "ppt_maldicion.csv", index=False)
R["sesgo_pct_todas"] = float(np.mean(err)); R["sesgo_pct_aceptadas"] = float(np.mean(err[acc]))

# 6. Bootstrap en Chapinero fuera (307 avisos): MAE con y sin estiramiento, y frente a otras familias
rng = np.random.default_rng(2026)
B = 2000
base = ch.set_index("idx")
fam = {}
for f in ["lightgbm", "catboost", "random_forest", "cart", "nn_atencion", "elastic_net", "ols"]:
    o = pd.read_parquet(DIR_MODELOS / f"cv_clase_oof_{f}_con.parquet")
    o = o[o["esquema"].str.startswith("Chapinero")].set_index("idx").loc[base.index]
    fam[f] = np.abs(o["pred"].to_numpy() - o["real"].to_numpy())
fam["lightgbm_estirado"] = np.abs(base["pf"].to_numpy() - base["real"].to_numpy())
n = len(base)
idx_b = rng.integers(0, n, (B, n))
ic = {}
for f, e in fam.items():
    mb = e[idx_b].mean(1)
    ic[f] = {"mae": float(e.mean()), "ic_bajo": float(np.percentile(mb, 2.5)), "ic_alto": float(np.percentile(mb, 97.5))}
for f in ["lightgbm_estirado", "catboost", "random_forest", "nn_atencion", "ols"]:
    dif = (fam[f] - fam["lightgbm"])[idx_b].mean(1)
    ic[f"dif_{f}_menos_lightgbm"] = {"dif": float((fam[f] - fam["lightgbm"]).mean()), "ic_bajo": float(np.percentile(dif, 2.5)),
                                    "ic_alto": float(np.percentile(dif, 97.5))}
json.dump(ic, open(DATOS / "ppt_bootstrap.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
R.update({"mape_chap": float(ch["err_pct"].abs().median())})
json.dump(R, open(DATOS / "ppt_varios.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
print(json.dumps(ic, ensure_ascii=False, indent=1))
print(pd.DataFrame(dist).round(1).to_string(index=False))
