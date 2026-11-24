"""09 · Modelo final (LightGBM con Catastro y texto), ajuste ×1,08, gemelos de texto y archivos enviados a Kaggle.

Qué hace
    1. Describe las 162 columnas del modelo por bloque (diccionario con faltantes, medianas y diferencia estandarizada
       entre prueba y entrenamiento) y mapea cuatro de ellas en la coordenada exacta.
    2. Especificación: LightGBM con pérdida L2 sobre el precio en millones (niveles), parada temprana con una partición
       interna por celdas de 2 km y promedio de 5 semillas; CatBoost RMSE (3 semillas) para la mezcla del envío 03.
    3. Validación del modelo elegido en los seis esquemas (scripts 07 y 08) y su brecha con Kaggle.
    4. Ajuste final 654 + b (ŷ − 654): rejilla de b en Chapinero fuera y clústeres (se usa b = 1,08).
    5. Gemelos de texto: si un aviso de prueba tiene el mismo texto normalizado que avisos de entrenamiento, la predicción
       es 0,3 × modelo + 0,7 × mediana del precio de esos avisos; se comprueba a posteriori en la CV por clústeres.
    6. Genera los tres envíos de este integrante y verifica su SHA-256 contra el registro
       02_outputs/predictions/registro_envios_sebastian_vela.csv (deben ser idénticos a los archivos subidos).

Salidas
    02_outputs/predictions/: lightgbm_l2_eta005_hojas63_semillas5_catastro.csv (envío 01),
        lightgbm_l2_eta005_hojas63_semillas5_catastro_x108.csv (envío 02) y mezcla_lightgbm040_catboost060_prof8_catastro.csv
        (envío 03)
    02_outputs/tables/: modelo_diccionario_variables, modelo_bloques_variables, modelo_hiperparametros,
        modelo_validacion_esquemas, modelo_factor_estiramiento, modelo_gemelos_peso, envios_verificacion_sha256
    02_outputs/figures/: modelo_mapas_variables, modelo_particion_interna, modelo_factor_estiramiento, modelo_pronostico_final
    00_data/processed/modelos/: final_pred_lightgbm_con.npy y final_pred_catboost_con.npy (predicciones sin ajustes)

Recálculo
    Por defecto lee las predicciones finales guardadas (vienen en el repositorio) y reconstruye los envíos. Con
    PS3_RECALCULAR=1 (o =09) reentrena LightGBM (5 semillas) y CatBoost (3 semillas) con todo el entrenamiento
    (unos 10 minutos en CPU).
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import hashlib

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from functions import flujo_final as F
from functions.entorno_ps3 import (ARCHIVO_LOCALIDADES, CENTRO, COLOR_TEST, COLOR_TRAIN, CRS_M, DIR_MODELOS, DIR_PRED,
                                   DIR_RAW, FACTOR, coma, existen, exigir, guardar_figura, guardar_tabla, log, plt,
                                   recalcular)

RECALCULAR = recalcular("09")
g, y, idx_tr, idx_te = F.g, F.y, F.idx_tr, F.idx_te
REGISTRO = DIR_PRED / "registro_envios_sebastian_vela.csv"
exigir(REGISTRO, DIR_MODELOS / "cv_clase_oof_lightgbm_con.parquet", DIR_MODELOS / "rk_knndm_res_lightgbm_con.csv",
       DIR_MODELOS / "pesos_mezcla.csv", como="Corra antes los scripts 07 y 08.")

# ------------------------------------------------------------------------------------------------------------------
# 1. Las 162 columnas por bloque
# ------------------------------------------------------------------------------------------------------------------
punt = F.puntajes("final", idx_tr, idx_te, RECALCULAR)[:, 0]     # ridge de texto con todo el entrenamiento
g["puntaje_texto_ridge"] = punt
NOMBRES = F.COLUMNAS["con"] + ["puntaje_texto_ridge"]
X = np.c_[g[F.COLUMNAS["con"]].to_numpy(float), punt]
OSM_BASE = [c for c in F.EXTERNAS if c.startswith("ln_") and c != "ln_poblacion_100m"]
BLOQUES = {
    "Vivienda (Kaggle completado con el texto)": (F.ESTRUCTURALES, "Kaggle + script 06"),
    "OpenStreetMap y distancias a centros": (OSM_BASE, "script 01 + functions/flujo_final.py"),
    "Google Earth Engine": ([c for c in F.EXTERNAS if c not in OSM_BASE], "script 01"),
    "Entorno adicional (OSM y avisos sin precio)": (F.EXTRA, "script 03"),
    "Catastro de Bogotá (manzana y 300 m)": (F.CATASTRO, "scripts 02 y 04"),
    "Texto (diccionarios, cantidades, LDA, LSA)": (F.TEXTO, "script 06"),
    "Coordenadas (km, EPSG:9377)": (F.COORD, "functions/flujo_final.py"),
    "Puntaje de texto (ridge fuera de pliegue)": (["puntaje_texto_ridge"], "functions/flujo_final.py"),
}
BLOQUE_DE = {v: b for b, (vs, _) in BLOQUES.items() for v in vs}
assert len(NOMBRES) == 162 and set(NOMBRES) == set(BLOQUE_DE)
guardar_tabla(pd.DataFrame([{"bloque": b, "variables": len(vs), "construidas_en": f} for b, (vs, f) in BLOQUES.items()]),
              "modelo_bloques_variables", index=False)
Xtr, Xte = X[idx_tr], X[idx_te]
dicc = pd.DataFrame({
    "bloque": [BLOQUE_DE[v] for v in NOMBRES],
    "% faltante entrenamiento": 100 * np.isnan(Xtr).mean(0), "% faltante prueba": 100 * np.isnan(Xte).mean(0),
    "mediana entrenamiento": np.nanmedian(Xtr, 0), "mediana prueba": np.nanmedian(Xte, 0),
    "d (prueba menos entrenamiento)": (np.nanmean(Xte, 0) - np.nanmean(Xtr, 0)) / np.nanstd(Xtr, 0),
}, index=pd.Index(NOMBRES, name="variable"))
guardar_tabla(dicc, "modelo_diccionario_variables", decimales=4)
log(f"Correlación del puntaje de texto fuera de pliegue con ln(precio) en entrenamiento: "
    f"{coma(np.corrcoef(punt[idx_tr], F.lp[idx_tr])[0, 1], 3)}")

loc = gpd.read_file(ARCHIVO_LOCALIDADES).to_crs(CRS_M)
chap = loc[loc["localidad"] == "Chapinero"]
g["vref_300m_mill"] = np.exp(g["ln_vref_300m"]) / 1e6
g["puntaje_mill"] = np.exp(g["puntaje_texto_ridge"])
g["dist_cerros_km"] = np.expm1(g["ln_dist_area_protegida_m"]) / 1000
EXT = np.r_[np.percentile(g["x"], 0.5) - 500, np.percentile(g["y"], 0.5) - 500,
            np.percentile(g["x"], 99.5) + 500, np.percentile(g["y"], 99.5) + 500]


def mapa_puntos(ax, var, titulo, cmap):
    u = g.groupby(["x", "y"], as_index=False)[var].mean().dropna().sample(frac=1, random_state=2026)
    sc = ax.scatter(u["x"], u["y"], c=u[var], s=2, cmap=cmap, linewidths=0,
                    vmin=u[var].quantile(.02), vmax=u[var].quantile(.98))
    loc.boundary.plot(ax=ax, color="grey", lw=0.3)
    chap.boundary.plot(ax=ax, color="black", lw=1.2)
    ax.set_xlim(EXT[0], EXT[2]); ax.set_ylim(EXT[1], EXT[3]); ax.set_axis_off()
    plt.colorbar(sc, ax=ax, shrink=0.6)
    ax.set_title(titulo, fontsize=10)


fig, axes = plt.subplots(1, 4, figsize=(20, 8))
for ax, (v_, t_, c_) in zip(axes, [("vref_300m_mill", "Valor del suelo a 300 m (millones por m²)", "magma"),
                                   ("prop_estrato6_300m", "Manzanas de estrato 6 a 300 m (proporción)", "viridis"),
                                   ("puntaje_mill", "Puntaje de texto (millones)", "cividis"),
                                   ("dist_cerros_km", "Distancia a los cerros (km)", "YlGnBu")]):
    mapa_puntos(ax, v_, t_, c_)
fig.suptitle("Cuatro de las 162 variables, en la coordenada exacta de cada aviso", fontsize=13)
fig.tight_layout()
guardar_figura(fig, "modelo_mapas_variables")

# ------------------------------------------------------------------------------------------------------------------
# 2. Especificación: hiperparámetros, partición interna y árboles de la semilla 2026
# ------------------------------------------------------------------------------------------------------------------
i_in, i_iv = F.mu.bloques_internos(F.xs[idx_tr], F.ys[idx_tr], semilla=F.SEMILLA)
p = dict(objective="l2", learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.7, reg_lambda=1.0, n_estimators=5000, random_state=F.SEMILLA, verbose=-1, n_jobs=3)
m = lgb.LGBMRegressor(**p).fit(Xtr[i_in], y[idx_tr][i_in], eval_set=[(Xtr[i_iv], y[idx_tr][i_iv])], eval_metric="l2",
                               callbacks=[lgb.early_stopping(100, verbose=False, first_metric_only=True)])
arboles = max(int(m.best_iteration_ or 100), 50)
guardar_tabla(pd.DataFrame([
    ("objective", "l2", "pérdida cuadrática: estima la media condicional"),
    ("learning_rate", "0,05", "tasa de aprendizaje η"),
    ("num_leaves", "63", "hojas máximas por árbol"),
    ("min_child_samples", "20", "avisos mínimos por hoja"),
    ("subsample / subsample_freq", "0,8 / 1", "80 % de los avisos en cada árbol"),
    ("colsample_bytree", "0,7", "70 % de las variables en cada árbol"),
    ("reg_lambda", "1,0", "penalidad L2 de los valores de las hojas"),
    ("n_estimators", f"parada temprana ({arboles} con la semilla 2026)", "hasta 5.000; paciencia de 100 árboles"),
    ("semillas", "5", "promedio de 2026 a 2030"),
], columns=["hiperparametro", "valor", "significado"]), "modelo_hiperparametros", index=False)
log(f"Partición interna: ajuste {len(i_in):,} | parada {len(i_iv):,} | árboles con la semilla 2026: {arboles}")
fig, ax = plt.subplots(figsize=(6, 7))
ax.scatter(F.xs[idx_tr][i_in], F.ys[idx_tr][i_in], s=0.5, color="#BBBBBB", label="ajuste interno")
ax.scatter(F.xs[idx_tr][i_iv], F.ys[idx_tr][i_iv], s=0.5, color=COLOR_TRAIN, label="parada temprana")
chap.boundary.plot(ax=ax, color="black", lw=1)
ax.set_xlim(EXT[0], EXT[2]); ax.set_ylim(EXT[1], EXT[3]); ax.set_axis_off()
ax.legend(markerscale=12, loc="lower right")
ax.set_title("Partición interna por celdas de 2 km (semilla 2026)")
guardar_figura(fig, "modelo_particion_interna")

# ------------------------------------------------------------------------------------------------------------------
# 3. Validación del modelo final y brecha con Kaggle
# ------------------------------------------------------------------------------------------------------------------
registro = pd.read_csv(REGISTRO, dtype={"sha256": str}, keep_default_na=False)
privado_01 = float(registro.loc[registro["archivo"] == "lightgbm_l2_eta005_hojas63_semillas5_catastro.csv", "kaggle_privado"].iloc[0])
res = pd.read_csv(DIR_MODELOS / "cv_clase_res_lightgbm_con.csv")
rk = pd.read_csv(DIR_MODELOS / "rk_knndm_res_lightgbm_con.csv").set_index("variante").loc["sin kriging"]
nombres_esq = {"aleatoria": "CV aleatoria (5 pliegues)", "bloques": "Bloques de 2 km",
               "bloques_buffer": "Bloques de 2 km con buffer de 300 m", "clusters": "Clústeres k-means (10)",
               "Chapinero (imitando el taller)": "Chapinero fuera (307 avisos)"}
val = res.assign(esquema=res["esquema"].map(nombres_esq)).set_index("esquema")[["n", "MAE", "RMSE", "Sesgo"]]
val.loc["kNNDM (q = 1.600)"] = [len(idx_tr), rk["MAE kNNDM"], rk["RMSE kNNDM"], rk["Sesgo"]]
val["Kaggle privado (envío 01) menos validación"] = privado_01 - val["MAE"]
guardar_tabla(val, "modelo_validacion_esquemas", decimales=3)

# ------------------------------------------------------------------------------------------------------------------
# 4. Ajuste ×1,08: rejilla del factor en Chapinero fuera y clústeres
# ------------------------------------------------------------------------------------------------------------------
oof = pd.read_parquet(DIR_MODELOS / "cv_clase_oof_lightgbm_con.parquet")
ch = oof[oof["esquema"].str.startswith("Chapinero")]
cl = oof[oof["esquema"] == "clusters"]
rejilla_b = pd.DataFrame([{"b": b, "MAE Chapinero fuera": np.abs(F.estirar(ch["pred"], b) - ch["real"]).mean(),
                           "sesgo Chapinero fuera": (F.estirar(ch["pred"], b) - ch["real"]).mean(),
                           "MAE clústeres": np.abs(F.estirar(cl["pred"], b) - cl["real"]).mean()}
                          for b in np.round(np.arange(1.00, 1.125, 0.01), 2)]).set_index("b")
guardar_tabla(rejilla_b, "modelo_factor_estiramiento", decimales=3)
log(f"Pendiente de la recta precio real sobre predicción en Chapinero fuera: {coma(np.polyfit(ch['pred'], ch['real'], 1)[0], 2)}")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(rejilla_b.index, rejilla_b["MAE Chapinero fuera"], "o-", color=COLOR_TEST)
ax[0].set(title="Chapinero fuera", xlabel="factor b", ylabel="MAE (millones)")
ax[1].plot(rejilla_b.index, rejilla_b["MAE clústeres"], "o-", color=COLOR_TRAIN)
ax[1].set(title="Clústeres k-means", xlabel="factor b", ylabel="MAE (millones)")
for a in ax:
    a.axvline(FACTOR, color="grey", ls="--", lw=1)
fig.suptitle(f"Error según el factor de estiramiento alrededor de {CENTRO} millones")
fig.tight_layout()
guardar_figura(fig, "modelo_factor_estiramiento")

# ------------------------------------------------------------------------------------------------------------------
# 5. Gemelos de texto: comprobación a posteriori en la CV por clústeres (no se usó para fijar 0,3 y 0,7)
# ------------------------------------------------------------------------------------------------------------------
log(f"Avisos de prueba con gemelo de texto en entrenamiento: {F.TIENE_GEMELO.sum()} ({coma(100 * F.TIENE_GEMELO.mean(), 2)} %)")
pliegues_cl = F.pliegues_clase(("clusters",))
partes = []
for n, (e, a, b, _) in pliegues_cl.items():
    if e != "clusters":
        continue
    gm = pd.Series(y[a], index=F.grupo[a]).groupby(level=0).median()
    pg = pd.Series(F.grupo[b]).map(gm).to_numpy()
    o = cl[cl["pliegue"] == n].set_index("idx").loc[b]
    mk = np.isfinite(pg)
    partes.append(pd.DataFrame({"pred": o["pred"].values[mk], "real": o["real"].values[mk], "gemelo": pg[mk]}))
gv = pd.concat(partes)
comp = pd.DataFrame({"peso_del_gemelo": np.round(np.arange(0, 1.01, 0.1), 1)})
comp["MAE"] = [np.abs((1 - w) * gv["pred"] + w * gv["gemelo"] - gv["real"]).mean() for w in comp["peso_del_gemelo"]]
comp["avisos"] = len(gv)
guardar_tabla(comp, "modelo_gemelos_peso", index=False, decimales=3)

# ------------------------------------------------------------------------------------------------------------------
# 6. Predicciones finales y envíos a Kaggle
# ------------------------------------------------------------------------------------------------------------------
ARCH_L, ARCH_C = DIR_MODELOS / "final_pred_lightgbm_con.npy", DIR_MODELOS / "final_pred_catboost_con.npy"
if RECALCULAR or not existen(ARCH_L, ARCH_C):
    for fam, semillas, archivo in (("lightgbm", 5, ARCH_L), ("catboost", 3, ARCH_C)):
        preds = Parallel(n_jobs=semillas)(delayed(F.mu.ajustar_predecir)(fam, X, y, F.xs, F.ys, F.lpm2, F.lp, F.grupo, None,
                                                                         idx_tr, idx_te, False, False, F.params_de(fam),
                                                                         F.SEMILLA + s) for s in range(semillas))
        np.save(archivo, np.mean(preds, axis=0))
        log(f"Modelo final {fam}: {semillas} semillas")
else:
    log("Se usan las predicciones finales guardadas (para reentrenar: PS3_RECALCULAR=09)")
L, C = np.load(ARCH_L), np.load(ARCH_C)
pesos = pd.read_csv(DIR_MODELOS / "pesos_mezcla.csv").set_index("catastro").loc["con"]
log(f"LightGBM sin ajustes: media {coma(L.mean())}, mediana {coma(np.median(L))} millones | pesos de la mezcla: "
    f"LightGBM {pesos['peso_lightgbm']:.1f}, CatBoost {pesos['peso_catboost']:.1f}")

plantilla = pd.read_csv(DIR_RAW / "submission_template.csv")
ENVIOS = {
    "lightgbm_l2_eta005_hojas63_semillas5_catastro.csv": L,
    "lightgbm_l2_eta005_hojas63_semillas5_catastro_x108.csv": F.estirar(L, FACTOR),
    "mezcla_lightgbm040_catboost060_prof8_catastro.csv": pesos["peso_lightgbm"] * L + pesos["peso_catboost"] * C,
}
filas = []
for archivo, pred in ENVIOS.items():
    envio = F.envio_kaggle(pred)
    assert len(envio) == len(plantilla) and set(envio["property_id"]) == set(plantilla["property_id"])
    contenido = envio.to_csv(index=False, lineterminator="\r\n")      # mismo formato (CRLF) que los archivos subidos
    (DIR_PRED / archivo).write_bytes(contenido.encode("utf-8"))
    sha = hashlib.sha256(contenido.encode("utf-8")).hexdigest()
    fila = registro[registro["archivo"] == archivo].iloc[0]
    filas.append({"archivo": archivo, "sha256_generado": sha, "sha256_subido": fila["sha256"], "identico": sha == fila["sha256"],
                  "precio_mediano_millones": envio["price"].median() / 1e6, "kaggle_publico": fila["kaggle_publico"],
                  "kaggle_privado": fila["kaggle_privado"]})
    log(f"{archivo}: SHA-256 {'idéntico al subido' if sha == fila['sha256'] else 'DISTINTO del subido'}")
verif = pd.DataFrame(filas)
guardar_tabla(verif, "envios_verificacion_sha256", index=False)
if not verif["identico"].all():
    log("AVISO: algún envío regenerado no coincide byte a byte con el subido (por ejemplo, tras reentrenar en otro equipo)")

final = F.estirar(L, FACTOR).copy()
final[F.TIENE_GEMELO] = 0.3 * final[F.TIENE_GEMELO] + 0.7 * F.PRECIO_GEMELO[F.TIENE_GEMELO]
fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.2, 1]})
bins = np.arange(250, 1901, 50)
axes[0].hist(y[idx_tr], bins, density=True, alpha=0.6, color=COLOR_TRAIN, label="Precio en entrenamiento (Bogotá)")
axes[0].hist(final, bins, density=True, alpha=0.6, color=COLOR_TEST, label="Pronóstico para la prueba (Chapinero)")
axes[0].set(xlabel="Millones de pesos", ylabel="Densidad", title="Distribución del precio y del pronóstico")
axes[0].legend(frameon=False)
pr = g.iloc[idx_te][["x", "y"]].assign(pred=final).groupby(["x", "y"], as_index=False)["pred"].mean()
sc = axes[1].scatter(pr["x"], pr["y"], c=pr["pred"], s=3, cmap="magma", vmin=pr["pred"].quantile(.02), vmax=pr["pred"].quantile(.98))
chap.boundary.plot(ax=axes[1], color="black", lw=1)
loc.boundary.plot(ax=axes[1], color="grey", lw=0.3)
b_ = chap.total_bounds
axes[1].set_xlim(b_[0] - 300, b_[0] + 0.55 * (b_[2] - b_[0])); axes[1].set_ylim(b_[1] - 300, b_[3] + 1500)
axes[1].set_axis_off()
plt.colorbar(sc, ax=axes[1], shrink=0.7, label="millones")
axes[1].set_title("Pronóstico final (envío 02) en la coordenada exacta (zona urbana)")
fig.tight_layout()
guardar_figura(fig, "modelo_pronostico_final")
log(f"Pronóstico final: mediana {coma(np.median(final))}, media {coma(final.mean())}, "
    f"{coma(100 * np.mean(final > 1200))} % por encima de 1.200 millones")
