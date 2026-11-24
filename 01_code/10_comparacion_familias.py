"""10 · Comparación de las seis familias y de las seis maneras de validar (tablas y figuras de la comparación).

Qué hace (lee las predicciones fuera de pliegue de los scripts 07 y 08; no ajusta modelos)
    1. Un pliegue de validación de cada esquema en el mapa (CV aleatoria, bloques de 2 km, bloques con buffer de 300 m,
       clústeres k-means, kNNDM y Chapinero fuera) y la distancia de cada aviso validado a su vecino de entrenamiento
       más cercano, frente a la de la tarea real (prueba de Kaggle → entrenamiento).
    2. MAE y sesgo de las siete especificaciones por esquema, el orden de las familias con la validación estándar y con
       las espaciales (correlación de rangos) y la figura de las familias con tres validaciones.
    3. Bootstrap pareado (10.000 réplicas) del MAE en Chapinero fuera y de su diferencia con LightGBM.
    4. Calibración con Kaggle: qué validación ordena los tres envíos igual que la tabla privada.
    5. Pruebas que no entraron al modelo final: sin Catastro, mezcla con CatBoost y kriging de residuos.

Salidas
    02_outputs/figures/: validacion_pliegues_mapa, validacion_distancia_vecino, familias_validaciones
    02_outputs/tables/: validacion_distancia_vecino, familias_mae_por_esquema, familias_sesgo_por_esquema,
        familias_orden_por_esquema, familias_bootstrap_chapinero, calibracion_kaggle_envios, pruebas_descartadas
Tiempo: unos 2 minutos.
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from functions import flujo_final as F
from functions import knndm_kriging as es_mod
from functions.entorno_ps3 import (ARCHIVO_LOCALIDADES, CENTRO, CRS_M, DIR_MODELOS, DIR_PRED, DIR_TAB, FACTOR, coma, exigir,
                                   guardar_figura, guardar_tabla, log, plt)

exigir(DIR_MODELOS / "knndm_resumen.csv", DIR_MODELOS / "pesos_mezcla.csv", DIR_TAB / "validacion_mezcla_clusters.csv",
       DIR_PRED / "registro_envios_sebastian_vela.csv", como="Corra antes los scripts 07, 08 y 09.")
g, y, idx_tr, idx_te = F.g, F.y, F.idx_tr, F.idx_te
xy = np.c_[F.xs, F.ys]
loc = gpd.read_file(ARCHIVO_LOCALIDADES).to_crs(CRS_M)
chap = loc[loc["localidad"] == "Chapinero"]

# ------------------------------------------------------------------------------------------------------------------
# 1. Seis maneras de validar: un pliegue de cada una y la distancia al vecino de entrenamiento
# ------------------------------------------------------------------------------------------------------------------
fold_k, grupo_q, q_k, resumen_k, obj_k = es_mod.knndm(xy[idx_tr], xy[idx_te], k=5)
guardado = pd.read_csv(DIR_MODELOS / "knndm_resumen.csv")
assert np.allclose(pd.DataFrame(resumen_k)["W"].values, guardado["W"].values), "kNNDM no coincide con el script 08"
log(f"kNNDM elige q = {q_k}; coincide con 00_data/processed/modelos/knndm_resumen.csv")
pl = F.pliegues_clase(("aleatoria", "bloques", "bloques_buffer", "clusters"))
validacion = {"CV aleatoria": pl["aleatoria-0"][2], "Bloques de 2 km": pl["bloques-0"][2],
              "Bloques con buffer de 300 m": pl["bloques_buffer-0"][2], "Clústeres k-means": pl["clusters-0"][2],
              "kNNDM": idx_tr[fold_k == 0], "Chapinero fuera": F.PLIEGUES["Chapinero"][1]}
quitados_buffer = np.setdiff1d(idx_tr, np.r_[pl["bloques_buffer-0"][1], pl["bloques_buffer-0"][2]])
fig, axes = plt.subplots(2, 3, figsize=(16, 14))
b_ = np.r_[np.percentile(F.xs[idx_tr], 0.5) - 500, np.percentile(F.ys[idx_tr], 0.5) - 500,
           np.percentile(F.xs[idx_tr], 99.5) + 500, np.percentile(F.ys[idx_tr], 99.5) + 500]
for ax, (nombre, iv) in zip(axes.ravel(), validacion.items()):
    ax.scatter(F.xs[idx_tr], F.ys[idx_tr], s=0.3, color="#C8C8C8")
    if nombre.startswith("Bloques con"):
        ax.scatter(F.xs[quitados_buffer], F.ys[quitados_buffer], s=0.5, color="#E6A817", label="buffer (no entrena ni valida)")
        ax.legend(markerscale=10, loc="lower right", fontsize=8)
    ax.scatter(F.xs[iv], F.ys[iv], s=0.5, color="#4C72B0")
    chap.boundary.plot(ax=ax, color="black", lw=1)
    ax.set_xlim(b_[0], b_[2]); ax.set_ylim(b_[1], b_[3]); ax.set_axis_off()
    ax.set_title(f"{nombre} ({len(iv):,} validados)".replace(",", "."), fontsize=11)
fig.suptitle("Primer pliegue de validación de cada esquema (azul: validación; gris: entrenamiento)", fontsize=13, y=1.01)
fig.tight_layout()
guardar_figura(fig, "validacion_pliegues_mapa")


def dist_esquema(prefijo):
    return np.concatenate([cKDTree(xy[a]).query(xy[b])[0] for n, (e, a, b, _) in pl.items() if e == prefijo])


dist = {"Prueba real (Kaggle)": cKDTree(xy[idx_tr]).query(xy[idx_te])[0],
        "CV aleatoria": dist_esquema("aleatoria"), "Bloques de 2 km": dist_esquema("bloques"),
        "Bloques con buffer de 300 m": dist_esquema("bloques_buffer"), "Clústeres k-means": dist_esquema("clusters"),
        "kNNDM": es_mod.distancias_cv(xy[idx_tr], fold_k),
        "Chapinero fuera": cKDTree(xy[F.PLIEGUES["Chapinero"][0]]).query(xy[F.PLIEGUES["Chapinero"][1]])[0]}
tab_d = pd.DataFrame({k: {"% a 0 m (misma coordenada)": 100 * np.mean(v < 1), "mediana (m)": np.median(v),
                          "percentil 90 (m)": np.percentile(v, 90)} for k, v in dist.items()}).T
guardar_tabla(tab_d.rename_axis("esquema"), "validacion_distancia_vecino", decimales=2)
fig, ax = plt.subplots(figsize=(9, 5))
for k, v in dist.items():
    s = np.sort(np.maximum(v, 1))
    ax.plot(s, np.arange(1, len(s) + 1) / len(s), lw=2.5 if k.startswith("Prueba") else 1.3,
            color="black" if k.startswith("Prueba") else None, label=k)
ax.set_xscale("log")
ax.set(xlabel="Distancia al aviso de entrenamiento más cercano (m, escala log; 0 m se dibuja en 1 m)",
       ylabel="Proporción acumulada", title="Qué tan lejos queda lo validado de lo que el modelo vio")
ax.legend(frameon=False, fontsize=8)
guardar_figura(fig, "validacion_distancia_vecino")

# ------------------------------------------------------------------------------------------------------------------
# 2. Error por familia y esquema
# ------------------------------------------------------------------------------------------------------------------
FAMILIAS = ["lightgbm", "catboost", "random_forest", "cart", "nn_atencion", "elastic_net", "ols"]
ESQ = {"aleatoria": "CV aleatoria", "bloques": "Bloques de 2 km", "bloques_buffer": "Bloques con buffer",
       "clusters": "Clústeres k-means", "Chapinero (imitando el taller)": "Chapinero fuera"}
res = pd.concat([pd.read_csv(DIR_MODELOS / f"cv_clase_res_{f}_con.csv") for f in FAMILIAS])
res["esquema"] = res["esquema"].map(ESQ)
mae = res.pivot(index="modelo", columns="esquema", values="MAE")[list(ESQ.values())].sort_values("Chapinero fuera")
sesgo = res.pivot(index="modelo", columns="esquema", values="Sesgo")[list(ESQ.values())].loc[mae.index]
rk = pd.read_csv(DIR_MODELOS / "rk_knndm_res_lightgbm_con.csv").set_index("variante")
mae.loc["LightGBM", "kNNDM"] = rk.loc["sin kriging", "MAE kNNDM"]
guardar_tabla(mae, "familias_mae_por_esquema", decimales=3)
guardar_tabla(sesgo, "familias_sesgo_por_esquema", decimales=3)
log("MAE por especificación y esquema (las celdas vacías no se calcularon, por tiempo):\n" + mae.round(1).to_string())

registro = pd.read_csv(DIR_PRED / "registro_envios_sebastian_vela.csv", dtype={"sha256": str}, keep_default_na=False)
priv_01 = float(registro.loc[registro["n"] == 3, "kaggle_privado"].iloc[0])
fig, ax = plt.subplots(figsize=(10, 5.5))
colores = {"CV aleatoria": "#BBBBBB", "Clústeres k-means": "#4C72B0", "Chapinero fuera": "#DD8452"}
pos = np.arange(len(mae))
for k, (esq, col) in enumerate(colores.items()):
    ax.scatter(mae[esq], pos + (k - 1) * 0.18, color=col, s=55, label=esq, zorder=3)
ax.set_yticks(pos, mae.index)
ax.invert_yaxis()
ax.axvline(priv_01, color="black", ls="--", lw=1)
ax.text(priv_01 + 1, len(mae) - 0.6, f"Kaggle privado\ndel envío 01 ({coma(priv_01)})", fontsize=8)
ax.set(xlabel="Error absoluto medio (millones)", title="Las siete especificaciones con tres validaciones")
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="x", alpha=0.3)
guardar_figura(fig, "familias_validaciones")

comunes = mae[["CV aleatoria", "Clústeres k-means", "Chapinero fuera"]].dropna()
orden = comunes.rank().astype(int)
rho_ac = spearmanr(comunes["CV aleatoria"], comunes["Chapinero fuera"])[0]
guardar_tabla(orden.assign(**{"rho CV aleatoria frente a Chapinero fuera": rho_ac}), "familias_orden_por_esquema", decimales=3)
log(f"Ganador por esquema: { {c: mae[c].idxmin() for c in ['CV aleatoria', 'Clústeres k-means', 'Chapinero fuera']} } | "
    f"correlación de rangos CV aleatoria frente a Chapinero fuera: {coma(rho_ac, 2)}")

# ------------------------------------------------------------------------------------------------------------------
# 3. Bootstrap pareado en Chapinero fuera (307 avisos)
# ------------------------------------------------------------------------------------------------------------------
oofs = {f: pd.read_parquet(DIR_MODELOS / f"cv_clase_oof_{f}_con.parquet") for f in FAMILIAS}
base = oofs["lightgbm"][oofs["lightgbm"]["esquema"].str.startswith("Chapinero")].set_index("idx")
real = base["real"].to_numpy()
pred = {f: oofs[f][oofs[f]["esquema"].str.startswith("Chapinero")].set_index("idx").loc[base.index, "pred"].to_numpy()
        for f in FAMILIAS}
err = {"LightGBM": np.abs(pred["lightgbm"] - real),
       "LightGBM ×1,08": np.abs(CENTRO + FACTOR * (pred["lightgbm"] - CENTRO) - real)}
for f in FAMILIAS[1:]:
    err[F.NOMBRE[f]] = np.abs(pred[f] - real)
pesos = pd.read_csv(DIR_MODELOS / "pesos_mezcla.csv").set_index("catastro").loc["con"]
err[f"Mezcla {coma(pesos['peso_lightgbm'])} LightGBM + {coma(pesos['peso_catboost'])} CatBoost"] = np.abs(
    pesos["peso_lightgbm"] * pred["lightgbm"] + pesos["peso_catboost"] * pred["catboost"] - real)
rng = np.random.default_rng(2026)
ib = rng.integers(0, len(real), (10_000, len(real)))
filas = []
for k, e in err.items():
    mb = e[ib].mean(1)
    fila = {"especificacion": k, "MAE": e.mean(), "IC95 bajo": np.percentile(mb, 2.5), "IC95 alto": np.percentile(mb, 97.5)}
    if k != "LightGBM":
        d = (e - err["LightGBM"])[ib].mean(1)
        fila.update({"diferencia con LightGBM": (e - err["LightGBM"]).mean(), "diferencia IC95 bajo": np.percentile(d, 2.5),
                     "diferencia IC95 alto": np.percentile(d, 97.5)})
    filas.append(fila)
guardar_tabla(pd.DataFrame(filas), "familias_bootstrap_chapinero", index=False, decimales=3)

# ------------------------------------------------------------------------------------------------------------------
# 4. Calibración con Kaggle de los tres envíos del modelo final
# ------------------------------------------------------------------------------------------------------------------
r3 = registro[registro["familia"] == "Boosting"]
cal = pd.DataFrame({"envio": ["01 LightGBM", "02 LightGBM ×1,08", "03 Mezcla 0,4 LightGBM + 0,6 CatBoost"],
                    "MAE clústeres": r3["mae_cv_clusters"].astype(float).to_numpy(),
                    "MAE Chapinero fuera": r3["mae_chapinero_fuera"].astype(float).to_numpy(),
                    "Kaggle público": r3["kaggle_publico"].astype(float).to_numpy(),
                    "Kaggle privado": r3["kaggle_privado"].astype(float).to_numpy()}).set_index("envio")
for c in ["MAE clústeres", "MAE Chapinero fuera", "Kaggle privado"]:
    cal[f"orden {c}"] = cal[c].rank().astype(int)
guardar_tabla(cal, "calibracion_kaggle_envios", decimales=3)

# ------------------------------------------------------------------------------------------------------------------
# 5. Pruebas que no entraron al modelo final
# ------------------------------------------------------------------------------------------------------------------
filas = []
for conj, nom in (("sin", "sin Catastro"), ("con", "con Catastro")):
    for f in ("lightgbm", "catboost"):
        r = pd.read_csv(DIR_MODELOS / f"cv_clase_res_{f}_{conj}.csv").set_index("esquema")
        filas.append({"prueba": f"{F.NOMBRE[f]} {nom}", "MAE clústeres": r.loc["clusters", "MAE"],
                      "MAE Chapinero fuera": r.loc["Chapinero (imitando el taller)", "MAE"],
                      "sesgo Chapinero fuera": r.loc["Chapinero (imitando el taller)", "Sesgo"]})
mz = pd.read_csv(DIR_TAB / "validacion_mezcla_clusters.csv")
for conj in ("sin", "con"):
    r = mz[mz["catastro"] == conj].set_index("esquema")
    filas.append({"prueba": f"Mezcla LightGBM + CatBoost elegida en clústeres, {conj} Catastro", "MAE clústeres": r.loc["clusters", "MAE"],
                  "MAE Chapinero fuera": r.loc["Chapinero (imitando el taller)", "MAE"],
                  "sesgo Chapinero fuera": r.loc["Chapinero (imitando el taller)", "Sesgo"]})
for v in rk.index:
    filas.append({"prueba": f"LightGBM con Catastro, kNNDM, {v}", "MAE kNNDM": rk.loc[v, "MAE kNNDM"],
                  "MAE Chapinero dentro de kNNDM": rk.loc[v, "MAE Chapinero (307)"],
                  "sesgo Chapinero dentro de kNNDM": rk.loc[v, "Sesgo Chapinero"]})
guardar_tabla(pd.DataFrame(filas), "pruebas_descartadas", index=False, decimales=3)
log("Fin de la comparación de familias")
