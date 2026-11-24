"""11 · Dónde se equivoca el modelo y qué significa para comprar vivienda en Chapinero.

Qué hace (predicciones fuera de muestra del LightGBM con Catastro, con y sin el ajuste ×1,08)
    1. Sesgo por cuantil de la predicción (lo que ve la empresa) en kNNDM (deciles) y en Chapinero fuera (quintiles), y
       compresión de los precios altos (sesgo en los avisos de más de 1.000 millones).
    2. Avisos de prueba fuera del rango del entrenamiento (percentiles 1 a 99) en 7 variables, y error en Chapinero fuera
       según cuántas quedan fuera.
    3. A qué lado se equivoca: más de 10 % arriba (pagar de más) o abajo (perder el negocio), también por quintil, y mapa
       del error de cada aviso en Chapinero fuera.
    4. Qué mueve la predicción: valores SHAP del LightGBM de una semilla (contribuciones exactas de LightGBM), por
       variable y por bloque.
    5. Simulación de ofertas: la empresa ofrece (1 − m) × predicción y el dueño acepta si la oferta es al menos (1 − d)
       × su precio; ganancia por oferta, maldición del ganador y precisión necesaria para ganar con margen de 10 %.

Salidas
    02_outputs/figures/: errores_sesgo_por_cuantil, errores_chapinero_mapa, shap_importancia, negocio_maldicion_ganador,
        negocio_precision_necesaria
    02_outputs/tables/: errores_sesgo_por_cuantil, errores_precios_altos, errores_fuera_de_rango, errores_lados,
        errores_lados_quintil, shap_importancia_variables, shap_bloques, negocio_simulacion_ofertas,
        negocio_precision_necesaria
Tiempo: unos 2 minutos (ajusta un LightGBM de una semilla para los SHAP).
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd

from functions import flujo_final as F
from functions.entorno_ps3 import (ARCHIVO_LOCALIDADES, CRS_M, DIR_MODELOS, FACTOR, coma, exigir, guardar_figura,
                                   guardar_tabla, log, plt)

exigir(DIR_MODELOS / "rk_knndm_oof_lightgbm_con.parquet", DIR_MODELOS / "cv_clase_oof_lightgbm_con.parquet",
       como="Corra antes los scripts 07 y 08.")
g, y, idx_tr, idx_te = F.g, F.y, F.idx_tr, F.idx_te
rk = pd.read_parquet(DIR_MODELOS / "rk_knndm_oof_lightgbm_con.parquet")
oof_l = pd.read_parquet(DIR_MODELOS / "cv_clase_oof_lightgbm_con.parquet")
ch = oof_l[oof_l["esquema"].str.startswith("Chapinero")]
ESQ = {"kNNDM": pd.DataFrame({"idx": rk["idx"].values, "real": rk["real"].values, "pred": rk["base"].values}),
       "Chapinero fuera": ch[["idx", "real", "pred"]].reset_index(drop=True)}
for nombre, o in ESQ.items():
    o["pf"] = F.estirar(o["pred"], FACTOR)
    log(f"{nombre}: {len(o):,} avisos | MAE sin ajuste {coma(np.abs(o['pred'] - o['real']).mean())} | con ×1,08 "
        f"{coma(np.abs(o['pf'] - o['real']).mean())} | sesgo con ×1,08 {coma((o['pf'] - o['real']).mean())}")

# ------------------------------------------------------------------------------------------------------------------
# 1. Sesgo por cuantil de la predicción y compresión de los precios altos
# ------------------------------------------------------------------------------------------------------------------
filas = []
for nombre, o in ESQ.items():
    nq = 10 if nombre == "kNNDM" else 5
    for col, version in (("pred", "Sin ajuste"), ("pf", "Con ajuste ×1,08")):
        q = pd.qcut(o[col], nq, labels=False)
        t = o.groupby(q + 1).agg(pred_media=(col, "mean"), real_media=("real", "mean"), n=("real", "size")).reset_index(names="cuantil")
        t["sesgo"] = t["pred_media"] - t["real_media"]
        t["esquema"], t["version"] = nombre, version
        filas.append(t)
cal = pd.concat(filas)
guardar_tabla(cal, "errores_sesgo_por_cuantil", index=False, decimales=3)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
for ax, nombre in zip(axes, ESQ):
    for version, col in (("Sin ajuste", "#4C72B0"), ("Con ajuste ×1,08", "#DD8452")):
        t = cal[(cal["esquema"] == nombre) & (cal["version"] == version)]
        ax.plot(t["pred_media"], t["sesgo"], "o-", color=col, label=version)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title=f"{nombre} ({'deciles' if nombre == 'kNNDM' else 'quintiles'} de la predicción)",
           xlabel="Predicción media del grupo (millones)", ylabel="Sesgo (millones)")
    ax.legend(frameon=False)
fig.tight_layout()
guardar_figura(fig, "errores_sesgo_por_cuantil")

filas = []
for nombre, o in ESQ.items():
    for corte, et in ((-np.inf, "todos"), (1000, "precio real > 1.000")):
        m = o["real"] > corte
        filas.append({"esquema": nombre, "avisos": et, "n": int(m.sum()), "sesgo sin ajuste": (o["pred"] - o["real"])[m].mean(),
                      "sesgo con ×1,08": (o["pf"] - o["real"])[m].mean()})
guardar_tabla(pd.DataFrame(filas), "errores_precios_altos", index=False, decimales=3)

# ------------------------------------------------------------------------------------------------------------------
# 2. Avisos fuera del rango del entrenamiento
# ------------------------------------------------------------------------------------------------------------------
VARS = {"ln_dist_area_protegida_m": "Distancia a los cerros", "ln_n_embajada_1000m": "Embajadas a 1 km",
        "ln_n_hotel_1000m": "Hoteles a 1 km", "ln_n_oficina_1000m": "Oficinas a 1 km", "ln_vref_300m": "Valor del suelo a 300 m",
        "area_imp": "Área", "elevacion_m": "Elevación"}


def cuenta_fuera(idx_ref, idx_eval):
    n, por_var = np.zeros(len(idx_eval), int), {}
    for v, et in VARS.items():
        lo, hi = np.nanpercentile(g[v].values[idx_ref], [1, 99])
        val = g[v].values[idx_eval]
        marca = (val < lo) | (val > hi)
        por_var[et] = 100 * marca.mean()
        n += marca.astype(int)
    return n, por_var


n_te, pv = cuenta_fuera(idx_tr, idx_te)
o = ESQ["Chapinero fuera"]
n_ch, _ = cuenta_fuera(F.PLIEGUES["Chapinero"][0], o["idx"].to_numpy())
o["grupo de rango"] = np.where(n_ch >= 2, "Dos o más fuera", np.where(n_ch == 1, "Una fuera", "Dentro del rango"))
reg = o.groupby("grupo de rango").apply(lambda t: pd.Series({
    "n": len(t), "MAE": np.mean(np.abs(t["pf"] - t["real"])), "error abs. mediano (%)": 100 * np.median(np.abs(t["pf"] / t["real"] - 1)),
    "sesgo": np.mean(t["pf"] - t["real"]), "% más de 10 % arriba": 100 * np.mean(t["pf"] / t["real"] - 1 > 0.1)}))
fuera = pd.concat({"prueba: % fuera de p1 a p99 por variable": pd.Series(pv),
                   "prueba: % de avisos por número de variables fuera": pd.Series(
                       {"ninguna": 100 * np.mean(n_te == 0), "una": 100 * np.mean(n_te == 1), "dos o más": 100 * np.mean(n_te >= 2)})})
guardar_tabla(fuera.rename("valor").rename_axis(["medida", "categoria"]).to_frame().reset_index().merge(
    reg.reset_index().rename(columns={"grupo de rango": "categoria"}), on="categoria", how="outer"),
    "errores_fuera_de_rango", index=False, decimales=3)

# ------------------------------------------------------------------------------------------------------------------
# 3. ¿Más de 10 % arriba o abajo?
# ------------------------------------------------------------------------------------------------------------------
filas = []
for nombre, o_ in ESQ.items():
    for col, version in (("pred", "Sin ajuste"), ("pf", "Con ajuste ×1,08")):
        e = 100 * (o_[col] / o_["real"] - 1)
        filas.append({"esquema": nombre, "version": version, "% más de 10 % arriba": (e > 10).mean() * 100,
                      "% dentro de ±10 %": (e.abs() <= 10).mean() * 100, "% más de 10 % abajo": (e < -10).mean() * 100,
                      "error abs. mediano (%)": e.abs().median(), "MAE": np.abs(o_[col] - o_["real"]).mean()})
guardar_tabla(pd.DataFrame(filas), "errores_lados", index=False, decimales=3)
e = 100 * (o["pf"] / o["real"] - 1)
lq = pd.DataFrame({"quintil": pd.qcut(o["pf"], 5, labels=False) + 1, "arriba": e > 10, "abajo": e < -10, "pf": o["pf"]}).groupby("quintil").agg(
    **{"prediccion_minima": ("pf", "min"), "prediccion_maxima": ("pf", "max"), "% más de 10 % arriba": ("arriba", "mean"),
       "% más de 10 % abajo": ("abajo", "mean")})
lq[["% más de 10 % arriba", "% más de 10 % abajo"]] *= 100
guardar_tabla(lq, "errores_lados_quintil", decimales=3)

loc = gpd.read_file(ARCHIVO_LOCALIDADES).to_crs(CRS_M)
chap = loc[loc["localidad"] == "Chapinero"]
om = o.assign(x=F.xs[o["idx"]], yy=F.ys[o["idx"]], error_pct=100 * (o["pf"] / o["real"] - 1))
fig, axes = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={"width_ratios": [1, 1.2]})
sc = axes[0].scatter(om["x"], om["yy"], c=om["error_pct"].clip(-50, 50), cmap="PuOr_r", vmin=-50, vmax=50, s=14, edgecolor="grey", lw=0.2)
chap.boundary.plot(ax=axes[0], color="black", lw=1)
b_ = chap.total_bounds
axes[0].set_xlim(b_[0] - 300, b_[0] + 0.55 * (b_[2] - b_[0])); axes[0].set_ylim(b_[1] - 300, b_[3] + 800); axes[0].set_axis_off()
plt.colorbar(sc, ax=axes[0], shrink=0.7, label="error (%): morado arriba, naranja abajo")
axes[0].set_title("Chapinero fuera: error de cada aviso (×1,08)")
axes[1].scatter(om["real"], om["pf"], s=10, color="#4C72B0", alpha=0.7)
lim = [250, 2000]
axes[1].plot(lim, lim, color="black", lw=1)
axes[1].fill_between(lim, [0.9 * v for v in lim], [1.1 * v for v in lim], color="grey", alpha=0.15, label="±10 %")
axes[1].set(xlim=lim, ylim=lim, xlabel="Precio real (millones)", ylabel="Predicción con ×1,08 (millones)", title="Predicho frente a real")
axes[1].legend(frameon=False)
fig.tight_layout()
guardar_figura(fig, "errores_chapinero_mapa")

# ------------------------------------------------------------------------------------------------------------------
# 4. Valores SHAP del LightGBM (una semilla, todo el entrenamiento) en la prueba
# ------------------------------------------------------------------------------------------------------------------
NOMBRES = F.COLUMNAS["con"] + ["puntaje_texto_ridge"]
X = np.c_[g[F.COLUMNAS["con"]].to_numpy(float), F.puntajes("final", idx_tr, idx_te)]
Xtr, ytr = X[idx_tr], y[idx_tr]
i_in, i_iv = F.mu.bloques_internos(F.xs[idx_tr], F.ys[idx_tr], semilla=F.SEMILLA)
p = dict(objective="l2", learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.7, reg_lambda=1.0, n_estimators=5000, random_state=F.SEMILLA, verbose=-1, n_jobs=4)
m = lgb.LGBMRegressor(**p).fit(Xtr[i_in], ytr[i_in], eval_set=[(Xtr[i_iv], ytr[i_iv])], eval_metric="l2",
                               callbacks=[lgb.early_stopping(100, verbose=False)])
p["n_estimators"] = max(int(m.best_iteration_ or 100), 50)
contrib = lgb.LGBMRegressor(**p).fit(Xtr, ytr).predict(X[idx_te], pred_contrib=True)
shap_te = pd.DataFrame(contrib[:, :-1], columns=NOMBRES)
valores_te = pd.DataFrame(X[idx_te], columns=NOMBRES)
BLOQUE = {**{v: "Vivienda" for v in F.ESTRUCTURALES}, **{v: "Texto" for v in F.TEXTO + ["puntaje_texto_ridge"]},
          **{v: "Catastro" for v in F.CATASTRO}, **{v: "Entorno (OSM, GEE, coordenadas)" for v in F.EXTERNAS + F.EXTRA + F.COORD}}
media_abs = shap_te.abs().mean()
bloques = media_abs.groupby(BLOQUE).sum().sort_values(ascending=False)
guardar_tabla(pd.DataFrame({"millones": bloques, "%": 100 * bloques / bloques.sum()}).rename_axis("bloque"), "shap_bloques", decimales=3)
guardar_tabla(pd.DataFrame({"bloque": [BLOQUE[v] for v in media_abs.index], "phi_abs_medio_millones": media_abs}).sort_values(
    "phi_abs_medio_millones", ascending=False).rename_axis("variable"), "shap_importancia_variables", decimales=4)
log(f"SHAP ({p['n_estimators']} árboles): " + ", ".join(f"{k} {coma(100 * v / bloques.sum(), 0)} %" for k, v in bloques.items()))
top = media_abs.sort_values(ascending=False).head(15)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
t = top[::-1]
axes[0].barh(t.index, t.values, color="#4C72B0")
axes[0].set(xlabel="|φ| medio en la prueba (millones)", title="Las 15 variables que más mueven la predicción")
v = "ln_vref_300m"
axes[1].scatter(np.exp(valores_te[v]) / 1e6, shap_te[v], s=3, color="#DD8452", alpha=0.5, label="prueba (Chapinero)")
p99 = np.exp(np.nanpercentile(g[v].values[idx_tr], 99)) / 1e6
axes[1].axvline(p99, color="black", ls="--", lw=1)
axes[1].text(p99 * 1.02, axes[1].get_ylim()[0] * 0.9, "p99 del\nentrenamiento", fontsize=8)
axes[1].set(xlabel="Valor del suelo a 300 m (millones por m²)", ylabel="φ (millones)",
            title="El aporte del valor del suelo se aplana sobre el p99")
fig.tight_layout()
guardar_figura(fig, "shap_importancia")

# ------------------------------------------------------------------------------------------------------------------
# 5. Simulación de ofertas, maldición del ganador y precisión necesaria
# ------------------------------------------------------------------------------------------------------------------
filas = []
for nombre, o_ in ESQ.items():
    V, Pr = o_["real"].to_numpy(), o_["pf"].to_numpy()
    for d in (0.05, 0.10):
        for mg in np.round(np.arange(0, 0.351, 0.05), 2):
            Of = (1 - mg) * Pr
            acc = Of >= (1 - d) * V
            pierde = acc & (Of > V)
            filas.append({"validacion": nombre, "d": d, "m": mg, "% aceptadas": 100 * acc.mean(),
                          "% de las compras que pagan de más": 100 * pierde.sum() / max(acc.sum(), 1),
                          "ganancia por oferta": (V - Of)[acc].sum() / len(V), "ganancia por compra": (V - Of)[acc].mean()})
guardar_tabla(pd.DataFrame(filas), "negocio_simulacion_ofertas", index=False, decimales=4)
o_ = ESQ["Chapinero fuera"]
V, Of = o_["real"].to_numpy(), 0.9 * o_["pf"].to_numpy()
acc = Of >= 0.9 * V
log(f"Chapinero fuera, m = d = 10 %: de 100 ofertas se aceptan {coma(100 * acc.mean(), 0)}, {coma(100 * (acc & (Of > V)).mean(), 0)} "
    f"pagan de más; ganancia por oferta {coma((V - Of)[acc].sum() / len(V))} millones")

o_ = ESQ["kNNDM"]
err = 100 * (o_["pf"] / o_["real"] - 1)
acc = 0.9 * o_["pf"] >= 0.9 * o_["real"]
fig, ax = plt.subplots(figsize=(9, 4.5))
bins = np.arange(-60, 65, 5)
ax.hist(err.clip(-59.9, 59.9), bins, density=True, alpha=0.55, color="#4C72B0", label=f"todas las predicciones (sesgo {coma(err.mean())} %)")
ax.hist(err[acc].clip(-59.9, 59.9), bins, density=True, alpha=0.55, color="#DD8452",
        label=f"ofertas aceptadas (sesgo {coma(err[acc].mean())} %)")
ax.axvline(0, color="black", lw=0.8)
ax.set(xlabel="Error de la predicción (%)", ylabel="Densidad", title="Las ofertas aceptadas son las que el modelo sobrestimó")
ax.legend(frameon=False)
guardar_figura(fig, "negocio_maldicion_ganador")

filas = []
for nombre, o_ in ESQ.items():
    V, le = o_["real"].to_numpy(), np.log(o_["pf"].to_numpy() / o_["real"].to_numpy())
    for lam in np.round(np.arange(1.0, 0.19, -0.01), 2):
        P = V * np.exp(lam * le)
        em = 100 * np.median(np.abs(P / V - 1))
        for d in (0.05, 0.10):
            Of = 0.9 * P
            acc = Of >= (1 - d) * V
            filas.append({"validacion": nombre, "lambda": lam, "error mediano (%)": em, "d": d, "ganancia por oferta": (V - Of)[acc].sum() / len(V)})
prec = pd.DataFrame(filas)
umbral = []
for (nombre, d), t in prec.groupby(["validacion", "d"]):
    t = t.sort_values("error mediano (%)")
    s = np.sign(t["ganancia por oferta"].values)
    k = np.where((s[:-1] > 0) & (s[1:] <= 0))[0]
    if len(k):
        k = k[-1]
        x0, x1 = t["error mediano (%)"].values[k:k + 2]
        y0, y1 = t["ganancia por oferta"].values[k:k + 2]
        cruce = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
    else:
        cruce = np.nan
    umbral.append({"validacion": nombre, "d (el dueño acepta hasta)": f"{int(100 * d)} % menos",
                   "error mediano actual (%)": 100 * np.median(np.abs(ESQ[nombre]["pf"] / ESQ[nombre]["real"] - 1)),
                   "error mediano necesario (%)": cruce})
guardar_tabla(pd.DataFrame(umbral), "negocio_precision_necesaria", index=False, decimales=3)
fig, ax = plt.subplots(figsize=(8, 4.5))
for (nombre, d), t in prec.groupby(["validacion", "d"]):
    ax.plot(t["error mediano (%)"], t["ganancia por oferta"], label=f"{nombre}, d = {int(100 * d)} %")
ax.axhline(0, color="black", lw=0.8)
ax.set(xlabel="Error absoluto mediano del modelo simulado (%)", ylabel="Ganancia por oferta (millones)",
       title="Ofertas con margen de 10 %: ganancia según la precisión")
ax.legend(frameon=False)
guardar_figura(fig, "negocio_precision_necesaria")
log("Fin de errores y negocio")
