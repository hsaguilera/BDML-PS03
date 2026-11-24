"""16 · Datos de las láminas: calibración, compras simuladas, bootstrap, tres apartamentos y Chapinero frente al resto.

Qué hace (antes document/presentaciones/datos_ppt3.py)
    1. Calibración por cuantil de la predicción, con y sin el ajuste ×1,08 (kNNDM y Chapinero fuera).
    2. Sobrestimación y subestimación de más de 10 % (también por quintil de la predicción).
    3. Compras simuladas: oferta (1 − m) × predicción; el dueño acepta si la oferta es al menos (1 − d) × su precio.
    4. Precisión requerida: modelo hipotético con el error logarítmico multiplicado por lambda.
    5. Bootstrap pareado (10.000 réplicas) en Chapinero fuera.
    6. Tres apartamentos de Chapinero con error típico: camino del boosting y cascada SHAP.
    7. Chapinero frente al resto: precios, precio por m², Catastro 2021 (estrato 6 y valor del suelo por manzana).
    8. Regla de rango con 7 variables; 9. términos del ridge de TF-IDF; 10. distribución del pronóstico.

Salidas (00_data/processed/laminas/; se versionan porque la sección 7 usa las capas de Catastro)
    ppt_calibracion2.csv, ppt_lados.csv, ppt_lados_quintil.csv, ppt_compras.csv, ppt_precision_requerida.csv,
    ppt_bootstrap2.json, ppt_tres_avisos.json, ppt_histograma_precios.csv, ppt_regla_rango2.csv,
    ppt_prueba_fuera7.parquet, ppt_palabras_ridge.csv y ppt_varios2.json

Recálculo
    Por defecto usa las salidas guardadas. Con PS3_RECALCULAR=02,16,17 descarga Catastro y las recalcula (unos 3
    minutos más la descarga).
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import json  # noqa: E402,F401
import os  # noqa: E402,F401
import re  # noqa: E402,F401
import sys  # noqa: E402,F401
import time  # noqa: E402,F401
from pathlib import Path  # noqa: E402,F401

from functions.laminas_entorno import DATOS  # noqa: E402
from functions.entorno_ps3 import existen, log, recalcular  # noqa: E402

# Este paso necesita las capas de Catastro (script 02, con internet). Por defecto se usan sus salidas guardadas.
SALIDAS = [DATOS / f for f in ("ppt_calibracion2.csv", "ppt_lados.csv", "ppt_lados_quintil.csv", "ppt_compras.csv",
                               "ppt_precision_requerida.csv", "ppt_bootstrap2.json", "ppt_tres_avisos.json",
                               "ppt_histograma_precios.csv", "ppt_regla_rango2.csv", "ppt_prueba_fuera7.parquet",
                               "ppt_palabras_ridge.csv", "ppt_varios2.json")]
if not recalcular("16") and existen(*SALIDAS):
    log("Se usan los datos guardados de este paso (necesita Catastro; para recalcular: PS3_RECALCULAR=02,16,17)")
    sys.exit(0)

# Base del modelo final (el mismo espacio de nombres que dejaba cv_clase.py en la carpeta original): g, y, idx_tr, idx_te,
# COLUMNAS, PLIEGUES, puntajes, mu (modelo_utils), vc (validacion_clase), np, pd, ...
from functions.flujo_final import *  # noqa: F401,F403,E402
from functions import knndm_kriging as es  # noqa: E402
from functions.entorno_ps3 import (ARCHIVO_LOCALIDADES, DIR_CACHE_TEXTO, DIR_CATASTRO, DIR_MODELOS, DIR_PRED, DIR_RAW,  # noqa: E402
                                   DIR_TAB, DIR_TEXTO)
from functions.laminas_entorno import DATOS  # noqa: E402  (00_data/processed/laminas/)

PLIEGUES_CLASE = pliegues_clase(("aleatoria",))
a, b = PLIEGUES["Chapinero"]          # pliegue "Chapinero fuera", como lo dejaba cv_clase.py
import geopandas as gpd  # noqa: E402
import lightgbm as lgb  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

R = {}
CENTRO, FACTOR = 654, 1.08
oof = pd.read_parquet(DATOS / "oof_lightgbm_con.parquet")
oof["pf"] = CENTRO + FACTOR * (oof["pred"] - CENTRO)
ESQ = {"kNNDM": oof[oof["esquema"] == "kNNDM"].copy(),
       "Chapinero fuera": oof[oof["esquema"].str.startswith("Chapinero")].copy()}

# 1. Calibración por cuantil de la predicción
cal = []
for nombre, o in ESQ.items():
    nq = 10 if nombre == "kNNDM" else 5
    for col, version in (("pred", "Sin ajuste"), ("pf", "Con ajuste x1,08")):
        q = pd.qcut(o[col], nq, labels=False)
        t = o.groupby(q).agg(pred_media=(col, "mean"), real_media=("real", "mean"), n=("real", "size")).reset_index(names="q")
        t["sesgo"] = t["pred_media"] - t["real_media"]
        t["esquema"], t["version"] = nombre, version
        cal.append(t)
cal = pd.concat(cal)
cal.to_csv(DATOS / "ppt_calibracion2.csv", index=False)
print(cal.round(1).to_string(index=False))

# 2. Mayor o menor precio: más de 10 % por encima o por debajo
lados = []
for nombre, o in ESQ.items():
    for col, version in (("pred", "Sin ajuste"), ("pf", "Con ajuste x1,08")):
        e = 100 * (o[col] / o["real"] - 1)
        lados.append({"esquema": nombre, "version": version, "sobre_10": float((e > 10).mean() * 100),
                      "dentro_10": float((e.abs() <= 10).mean() * 100), "bajo_10": float((e < -10).mean() * 100),
                      "error_abs_mediano": float(e.abs().median()), "mae": float(np.mean(np.abs(o[col] - o["real"]))),
                      "sesgo": float(np.mean(o[col] - o["real"]))})
lados = pd.DataFrame(lados)
lados.to_csv(DATOS / "ppt_lados.csv", index=False)
print(lados.round(1).to_string(index=False))
ch = ESQ["Chapinero fuera"]
e = 100 * (ch["pf"] / ch["real"] - 1)
qq = pd.qcut(ch["pf"], 5, labels=False)
lq = pd.DataFrame({"q": qq, "sobre": e > 10, "bajo": e < -10, "pf": ch["pf"]}).groupby("q").agg(
    pred_min=("pf", "min"), pred_max=("pf", "max"), sobre=("sobre", "mean"), bajo=("bajo", "mean")).reset_index()
lq[["sobre", "bajo"]] *= 100
lq.to_csv(DATOS / "ppt_lados_quintil.csv", index=False)
print(lq.round(1).to_string(index=False))

# 3. Compras simuladas (valor aproximado con el precio publicado; ofertas a dueños que no publican precio)
comp = []
for nombre, o in ESQ.items():
    V, Pr = o["real"].to_numpy(), o["pf"].to_numpy()
    for dd in (0.05, 0.10):
        for m in np.round(np.arange(0, 0.351, 0.05), 2):
            Of = (1 - m) * Pr
            acc = Of >= (1 - dd) * V
            pierde = acc & (Of > V)
            comp.append({"validacion": nombre, "descuento": dd, "margen": m, "aceptadas_pct": 100 * acc.mean(),
                         "pagan_de_mas_pct_ofertas": 100 * pierde.mean(),
                         "pagan_de_mas_pct_compras": 100 * pierde.sum() / max(acc.sum(), 1),
                         "ganancia_por_oferta": (V - Of)[acc].sum() / len(V),
                         "ganancia_por_compra": (V - Of)[acc].mean() if acc.any() else np.nan,
                         "perdida_media_si_paga_de_mas": (Of - V)[pierde].mean() if pierde.any() else np.nan,
                         "ganancia_media_si_gana": (V - Of)[acc & ~pierde].mean() if (acc & ~pierde).any() else np.nan})
comp = pd.DataFrame(comp)
comp.to_csv(DATOS / "ppt_compras.csv", index=False)
print(comp.round(1).to_string(index=False))

# 4. Precisión requerida (kNNDM y Chapinero fuera): error logarítmico multiplicado por lambda
prec = []
for nombre, o in ESQ.items():
    V, Pr = o["real"].to_numpy(), o["pf"].to_numpy()
    le = np.log(Pr / V)
    for lam in np.round(np.arange(1.0, 0.24, -0.05), 2):
        P_l = V * np.exp(lam * le)
        em = float(np.median(np.abs(P_l / V - 1)) * 100)
        for dd in (0.05, 0.10):
            mejor = (-np.inf, None)
            for m in np.arange(0, 0.5001, 0.005):
                Of = (1 - m) * P_l
                acc = Of >= (1 - dd) * V
                gan = (V - Of)[acc].sum() / len(V)
                if gan > mejor[0]:
                    mejor = (gan, m)
            prec.append({"validacion": nombre, "lambda": lam, "error_mediano_pct": em, "descuento": dd,
                         "margen_optimo": mejor[1], "ganancia_max_por_oferta": mejor[0]})
prec = pd.DataFrame(prec)
prec.to_csv(DATOS / "ppt_precision_requerida.csv", index=False)
for (nombre, dd), t in prec.groupby(["validacion", "descuento"]):
    t = t.sort_values("error_mediano_pct")
    pos = t[t["ganancia_max_por_oferta"] > 0]
    neg = t[t["ganancia_max_por_oferta"] <= 0]
    R[f"umbral_{nombre}_{dd}"] = {"rentable_hasta": float(pos["error_mediano_pct"].max()) if len(pos) else None,
                                  "no_rentable_desde": float(neg["error_mediano_pct"].min()) if len(neg) else None}
print(json.dumps({k: v for k, v in R.items() if k.startswith("umbral")}, indent=1))

# 5. Bootstrap pareado en Chapinero fuera
base = ch.set_index("idx")
err = {"lightgbm": np.abs(base["pred"] - base["real"]).to_numpy(), "lightgbm_x108": np.abs(base["pf"] - base["real"]).to_numpy()}
preds_fam = {}
for f in ["lightgbm", "catboost", "random_forest", "cart", "nn_atencion", "elastic_net", "ols"]:
    o = pd.read_parquet(DIR_MODELOS / f"cv_clase_oof_{f}_con.parquet")
    o = o[o["esquema"].str.startswith("Chapinero")].set_index("idx").loc[base.index]
    preds_fam[f] = o["pred"].to_numpy()
    if f != "lightgbm":
        err[f] = np.abs(o["pred"].to_numpy() - base["real"].to_numpy())
err["mezcla_40_60"] = np.abs(0.4 * preds_fam["lightgbm"] + 0.6 * preds_fam["catboost"] - base["real"].to_numpy())
R["mae_lightgbm_cv_clase"] = float(np.abs(preds_fam["lightgbm"] - base["real"].to_numpy()).mean())
rng = np.random.default_rng(2026)
B, n = 10000, len(base)
ib = rng.integers(0, n, (B, n))
bo = {}
for f, ef in err.items():
    mb = ef[ib].mean(1)
    bo[f] = {"mae": float(ef.mean()), "ic_bajo": float(np.percentile(mb, 2.5)), "ic_alto": float(np.percentile(mb, 97.5))}
    if f != "lightgbm":
        d_ = (ef - err["lightgbm"])[ib].mean(1)
        bo[f].update({"dif": float((ef - err["lightgbm"]).mean()), "dif_bajo": float(np.percentile(d_, 2.5)),
                      "dif_alto": float(np.percentile(d_, 97.5))})
json.dump(bo, open(DATOS / "ppt_bootstrap2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(pd.DataFrame(bo).T.round(1).to_string())

# 6. Tres apartamentos de Chapinero con error típico (modelo entrenado sin Chapinero, 1 semilla, parada de 241 árboles)
ej = json.load(open(DATOS / "ppt_ejemplo_boosting.json", encoding="utf-8"))
PARADA = int(ej["parada"])
nombres = COLUMNAS["con"] + ["puntaje_texto_ridge"]
P = dict(objective="l2", learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.7, reg_lambda=1.0, random_state=SEMILLA, verbose=-1, n_jobs=4)
a, b = PLIEGUES["Chapinero"]
Xc = np.c_[g[COLUMNAS["con"]].to_numpy(float), puntajes("Chapinero", a, b)]
m_c = lgb.LGBMRegressor(**P, n_estimators=PARADA).fit(Xc[a], y[a])
contrib = m_c.predict(Xc[b], pred_contrib=True)
base0 = float(contrib[0, -1])
pred_b = contrib.sum(1)
kag = pd.read_csv(DIR_RAW / "train.csv")[["property_id", "surface_total", "surface_covered", "bathrooms", "rooms", "description", "title"]]
info = g.iloc[b][["property_id", "area_imp", "bedrooms", "banos_imp", "es_apartamento"]].reset_index(drop=True)
info = info.merge(kag, on="property_id", how="left")
info["real"], info["pred"] = y[b], pred_b
info["err_pct"] = 100 * (info["pred"] / info["real"] - 1)
info["pos"] = np.arange(len(info))
med_abs = float(info["err_pct"].abs().median())
R["error_abs_mediano_ejemplos"] = med_abs
obs = info[(info["es_apartamento"] == 1) & info["bathrooms"].notna() & (info["surface_total"].notna() | info["surface_covered"].notna())]
info["pct_precio"] = info["real"].rank(pct=True) * 100
obs = obs.assign(pct_precio=info.loc[obs.index, "pct_precio"])
elegidos = []
for objetivo in (15, 50, 85):
    c = obs[(obs["pct_precio"] - objetivo).abs() <= 8].copy()
    c["dist"] = (c["err_pct"].abs() - med_abs).abs()
    elegidos.append(c.sort_values("dist").iloc[0])
pasos = list(range(0, PARADA + 1))
caminos = {}
for r in elegidos:
    fila = Xc[b][[int(r["pos"])]]
    caminos[r["property_id"]] = [base0] + [float(m_c.predict(fila, num_iteration=k)[0]) for k in pasos[1:]]
tres = []
for r in elegidos:
    i = int(r["pos"])
    cas = pd.Series(contrib[i, :-1], index=nombres).sort_values(key=np.abs, ascending=False)
    tres.append({"property_id": r["property_id"], "real": float(r["real"]), "pred": float(r["pred"]), "err_pct": float(r["err_pct"]),
                 "area": float(r["area_imp"]), "alcobas": float(r["bedrooms"]), "banos": float(r["banos_imp"]),
                 "camino": caminos[r["property_id"]],
                 "cascada": {k: float(v) for k, v in cas.head(7).items()}, "resto": float(cas.iloc[7:].sum()),
                 "valores": {k: float(Xc[b][i, nombres.index(k)]) for k in cas.head(7).index},
                 "titulo": str(r["title"])[:120]})
json.dump({"base": base0, "parada": PARADA, "error_abs_mediano": med_abs, "avisos": tres},
          open(DATOS / "ppt_tres_avisos.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for t in tres:
    print(t["property_id"], round(t["real"]), round(t["pred"]), round(t["err_pct"], 1), t["area"], t["alcobas"], t["banos"], t["titulo"][:60])
    print("   ", {k: round(v, 1) for k, v in t["cascada"].items()}, round(t["resto"], 1))

# 7. Chapinero frente al resto (entrenamiento) y Catastro 2021 por manzana
em = pd.read_parquet(DATOS / "entrenamiento_mapa.parquet")
em["grupo"] = np.where(em["localidad"] == "Chapinero", "Chapinero", "Resto de Bogotá")
pm = pd.read_parquet(DATOS / "prueba_mapa.parquet")
cortes = np.arange(300, 1651, 75)
hist = {}
for k_, s in [("Chapinero", em.loc[em["grupo"] == "Chapinero", "price"] / 1e6 if em["price"].max() > 1e5 else em.loc[em["grupo"] == "Chapinero", "price"]),
              ("Resto de Bogotá", em.loc[em["grupo"] == "Resto de Bogotá", "price"] / 1e6 if em["price"].max() > 1e5 else em.loc[em["grupo"] == "Resto de Bogotá", "price"]),
              ("Pronóstico para la prueba", pm["pred_final"])]:
    s = np.clip(s.to_numpy(float), 300, 1724.9)
    h = np.histogram(s, np.r_[cortes, 1725])[0]
    hist[k_] = (100 * h / h.sum()).tolist()
pd.DataFrame({"desde": np.r_[cortes], **hist}).to_csv(DATOS / "ppt_histograma_precios.csv", index=False)
precio = em["price"] / 1e6 if em["price"].max() > 1e5 else em["price"]
pm2 = em["precio_m2"] / 1e6 if em["precio_m2"].median() > 1e3 else em["precio_m2"]
for k_ in ("Chapinero", "Resto de Bogotá"):
    s = em["grupo"] == k_
    R[f"desc_{k_}"] = {"n": int(s.sum()), "mediana": float(precio[s].median()), "media": float(precio[s].mean()),
                       "pct_sobre_1200": float((precio[s] > 1200).mean() * 100), "m2_mediana": float(pm2[s].median())}
R["pronostico"] = {"n": int(len(pm)), "mediana": float(pm["pred_final"].median()), "media": float(pm["pred_final"].mean()),
                   "pct_sobre_1200": float((pm["pred_final"] > 1200).mean() * 100), "n_sobre_1650": int((pm["pred_final"] > 1650).sum()),
                   "pct_localidad_chapinero": float((pm["localidad"] == "Chapinero").mean() * 100),
                   "m2_mediana": float((pm["pred_final"] / pm["area_imp"]).median())}
loc = gpd.read_file(ARCHIVO_LOCALIDADES).to_crs(9377)
man = gpd.read_parquet(DIR_CATASTRO / "manzanas_geometria_2021.parquet").to_crs(9377)
est = pd.read_parquet(DIR_CATASTRO / "estratificacion_manzanas.parquet")
print("estratificación:", est.columns.tolist(), est.head(2).drop(columns=[c for c in est.columns if "geom" in c.lower()], errors="ignore").to_dict("records"))
cen = man.copy()
cen["geometry"] = man.geometry.representative_point()
cen = gpd.sjoin(cen, loc[["localidad", "geometry"]], how="left", predicate="within")
cen["chap"] = cen["localidad"] == "Chapinero"
vr = pd.read_parquet(DIR_CATASTRO / "valor_referencia_2019_2021.parquet")
vr = vr[vr["anio"] == 2021].groupby("MANCODIGO")["V_REF"].median()
cen = cen.merge(vr.rename("vref"), left_on="MANCODIGO", right_index=True, how="left")
col_est = [c for c in est.columns if "estrato" in c.lower()]
col_man = [c for c in est.columns if "manc" in c.lower() or "codigo" in c.lower()]
if col_est and col_man:
    e_ = est[[col_man[0], col_est[0]]].drop_duplicates(col_man[0]).rename(columns={col_man[0]: "MANCODIGO", col_est[0]: "estrato"})
    cen = cen.merge(e_, on="MANCODIGO", how="left")
    for k_, s in (("Chapinero", cen["chap"]), ("Resto de Bogotá", ~cen["chap"])):
        t = cen.loc[s & cen["estrato"].notna()]
        R[f"catastro_{k_}"] = {"manzanas_con_estrato": int(len(t)), "pct_estrato6": float((pd.to_numeric(t["estrato"], errors="coerce") == 6).mean() * 100),
                               "vref_mediana_millones": float(cen.loc[s & (cen["vref"] > 0), "vref"].median() / 1e6)}
print({k: v for k, v in R.items() if k.startswith(("desc_", "catastro_", "pronostico"))})

# 8. Regla de rango con 7 variables
VARS = ["ln_dist_area_protegida_m", "ln_n_embajada_1000m", "ln_n_hotel_1000m", "ln_n_oficina_1000m", "ln_vref_300m",
        "area_imp", "elevacion_m"]
tr_all = np.where(g["muestra"].values == "train")[0]
te_all = np.where(g["muestra"].values == "test")[0]


def cuenta_fuera(idx_ref, idx_eval):
    f = np.zeros(len(idx_eval), int)
    por_var = {}
    for v in VARS:
        lo, hi = np.nanpercentile(g[v].values[idx_ref], [1, 99])
        val = g[v].values[idx_eval]
        marca = (val < lo) | (val > hi)
        por_var[v] = float(marca.mean() * 100)
        f += marca.astype(int)
    return f, por_var


f_te, pv_te = cuenta_fuera(tr_all, te_all)
R["regla_prueba"] = {"dentro": int((f_te == 0).sum()), "una": int((f_te == 1).sum()), "dos_o_mas": int((f_te >= 2).sum()), "por_variable_pct": pv_te}
f_ch, pv_ch = cuenta_fuera(a, ch["idx"].to_numpy())
ch = ch.assign(n_fuera=f_ch, cat=np.where(f_ch >= 2, "Dos o más fuera", np.where(f_ch == 1, "Una fuera", "Dentro del rango")))
reg = ch.groupby("cat").apply(lambda t: pd.Series({"n": len(t), "mae": np.mean(np.abs(t["pf"] - t["real"])),
                                                    "error_abs_mediano": np.median(np.abs(t["pf"] / t["real"] - 1)) * 100,
                                                    "sesgo": np.mean(t["pf"] - t["real"]),
                                                    "sobre_10": np.mean(t["pf"] / t["real"] - 1 > 0.1) * 100})).reset_index()
reg.to_csv(DATOS / "ppt_regla_rango2.csv", index=False)
print(R["regla_prueba"]); print(reg.round(1).to_string(index=False)); print(pv_ch)
pm2_ = pm.copy()
pm2_["n_fuera7"] = 0
orden_te = pd.Series(np.arange(len(te_all)), index=g["property_id"].values[te_all])
pm2_["n_fuera7"] = f_te[orden_te.loc[pm2_["property_id"]].to_numpy()]
pm2_[["property_id", "n_fuera7"]].to_parquet(DATOS / "ppt_prueba_fuera7.parquet", index=False)
print("coincide con n_fuera de 8 variables:", float((np.minimum(pm2_["n_fuera7"], 2) == np.minimum(pm["n_fuera"], 2)).mean()))

# 9. Ridge de TF-IDF: términos con mayor y menor coeficiente (df >= 300)
voc = pd.read_csv(DIR_TEXTO / "vocabulario.csv")
rid = Ridge(alpha=1.0).fit(X_tfidf[tr_all], lp[tr_all])
voc["coef"] = rid.coef_[: len(voc)] if len(rid.coef_) >= len(voc) else np.r_[rid.coef_, np.full(len(voc) - len(rid.coef_), np.nan)]
v2 = voc[(voc["df"] >= 300)].dropna(subset=["coef"])
top = pd.concat([v2.nlargest(12, "coef").assign(lado="sube"), v2.nsmallest(12, "coef").assign(lado="baja")])
top.to_csv(DATOS / "ppt_palabras_ridge.csv", index=False)
print("tfidf", X_tfidf.shape, "vocabulario", len(voc))
print(top[["termino", "df", "coef", "lado"]].to_string(index=False))

# Aviso de ejemplo para el texto: candidatos de Chapinero con puntaje de texto cercano al precio
cand = pd.read_parquet(DATOS / "ppt_texto_candidatos.parquet")
cand["brecha"] = (cand["puntaje_millones"] / cand["price"] * (1e6 if cand["price"].max() > 1e5 else 1) - 1)
cand = cand.assign(abs_brecha=cand["brecha"].abs()).sort_values("abs_brecha")
for _, r in cand.head(6).iterrows():
    print("----", r["property_id"], r["price"], round(r["puntaje_millones"]), r["area_texto"], r["parqueaderos_texto"])
    print(str(r["title"])[:100]); print(str(r["description"])[:700])
json.dump(R, open(DATOS / "ppt_varios2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: v for k, v in R.items() if not k.startswith("umbral")}, ensure_ascii=False, indent=1, default=float)[:3000])
