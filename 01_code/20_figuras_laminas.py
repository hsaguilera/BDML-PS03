"""20 · Imágenes de las láminas: mapas, boosting con una variable, pliegues, dependencia SHAP y fórmulas.

Qué hace (antes document/presentaciones/figuras_ppt.py)
    Dibuja cada imagen al tamaño de su lugar en la lámina (13,33 × 7,5 pulgadas), con fuente Calibri si está instalada,
    texto de 12 puntos o más, coma decimal y mapas en la coordenada exacta de cada aviso.

Salidas (02_outputs/figures/laminas/, png a 300 ppp)
    portada_mapa, mapa_datos, mapa_valor_suelo, boosting_1d, pliegues, pronostico_mapa, errores_mapa, confianza_mapa,
    dependencia, dependencia_area, distancias, texto_zipf y las fórmulas f_mae, f_rosen, f_boosting, f_sesgo, f_shap,
    f_oferta, f_estiramiento, f_tfidf, f_ridge y f_atencion

Recálculo
    mapa_valor_suelo necesita las capas de Catastro (script 02, con internet): si no están, se conserva la imagen
    guardada. boosting_1d ajusta un modelo didáctico (precio frente al área): se rehace solo con PS3_RECALCULAR=20 o si
    falta la imagen.
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import json
import os
from pathlib import Path

import geopandas as gpd
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from functions.entorno_ps3 import ARCHIVO_LOCALIDADES, DIR_CATASTRO, DIR_RAW, DIR_TEXTO, existen, recalcular  # noqa: E402
from functions.laminas_entorno import DATOS, IMG as OUT  # noqa: E402

# entorno_ps3 fija un estilo para las figuras de análisis; aquí se vuelve a los valores por defecto de matplotlib para que
# las imágenes salgan igual que en la versión final de las láminas.
matplotlib.rcParams.update({k: matplotlib.rcParamsDefault[k] for k in ("figure.dpi", "savefig.dpi", "savefig.bbox")})

for _ruta in fm.findSystemFonts():                 # Calibri si está instalada (Windows); si no, la fuente por defecto
    if Path(_ruta).name.lower() in ("calibri.ttf", "calibril.ttf"):
        fm.fontManager.addfont(_ruta)
plt.rcParams.update({"font.family": "Calibri", "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "mathtext.fontset": "stixsans",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.edgecolor": "#5D6D7E", "text.color": "#1B2A41", "axes.labelcolor": "#1B2A41",
                     "xtick.color": "#5D6D7E", "ytick.color": "#5D6D7E", "figure.constrained_layout.use": True,
                     "legend.frameon": False, "axes.titleweight": "normal"})
AZUL, NARANJA, ROJO, GRIS, GRIS_CLARO, TINTA, VERDE = "#1F5FA8", "#D9822B", "#C0392B", "#7F8C8D", "#C8CDD0", "#1B2A41", "#2E8B57"
COMA = FuncFormatter(lambda v, p: f"{v:g}".replace(".", ",").replace("-", "−"))
MILES = FuncFormatter(lambda v, p: f"{v:,.0f}".replace(",", ".").replace("-", "−"))


def guardar(fig, nombre, transparente=False):
    fig.savefig(OUT / f"{nombre}.png", dpi=300, transparent=transparente)
    plt.close(fig)
    print("imagen", nombre)


def proy(lon, lat):
    s = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326).to_crs(9377)
    return s.x.to_numpy(), s.y.to_numpy()


def escala(ax, km, color=TINTA, pos=(0.06, 0.05)):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    xa, ya = x0 + pos[0] * (x1 - x0), y0 + pos[1] * (y1 - y0)
    ax.plot([xa, xa + km * 1000], [ya, ya], color=color, lw=2, solid_capstyle="butt")
    ax.text(xa + km * 500, ya + 0.015 * (y1 - y0), f"{km} km", ha="center", va="bottom", fontsize=12, color=color,
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))


loc = gpd.read_file(ARCHIVO_LOCALIDADES).to_crs(9377)
chap = loc[loc["localidad"] == "Chapinero"]
ent = pd.read_parquet(DATOS / "entrenamiento_mapa.parquet")
pru = pd.read_parquet(DATOS / "prueba_mapa.parquet")
xe, ye = proy(ent["lon"], ent["lat"])
xp, yp = proy(pru["lon"], pru["lat"])
en_chap = ent["localidad"].values == "Chapinero"
bx = chap.total_bounds
_ch = pru["localidad"].values == "Chapinero"
ZOOM = (np.percentile(xp[_ch], 0.5) - 400, np.percentile(xp[_ch], 99.5) + 400,
        np.percentile(yp[_ch], 0.5) - 300, np.percentile(yp[_ch], 99.5) + 300)
# Parte urbana de Chapinero: el polígono recortado a la envolvente de los avisos (más 250 m), para que en los mapas de toda
# la ciudad el borde no salga cortado al filo (disenio_v3_opus.md, M7).
_envolvente = gpd.GeoSeries(gpd.points_from_xy(np.r_[xe, xp], np.r_[ye, yp]), crs=9377).union_all().convex_hull.buffer(250)
chap_u = chap.geometry.intersection(_envolvente)

# 1. Portada: mapa sobre fondo oscuro (6,0 x 6,6)
fig, ax = plt.subplots(figsize=(5.6, 6.2), facecolor="#1B2A41")
ax.set_facecolor("#1B2A41")
loc.boundary.plot(ax=ax, color="#3B4C66", lw=0.4)
ax.scatter(xe, ye, s=0.25, color=GRIS_CLARO, alpha=0.45, rasterized=True)
ax.scatter(xp, yp, s=0.25, color=NARANJA, alpha=0.8, rasterized=True)
chap_u.boundary.plot(ax=ax, color="#F0F3F7", lw=1.0)
ax.set_xlim(xe.min() - 500, xe.max() + 500); ax.set_ylim(ye.min() - 500, ye.max() + 500)
ax.set_aspect("equal"); ax.axis("off")
guardar(fig, "portada_mapa")

# 2. Mapa de los datos (5,0 x 5,4)
fig, ax = plt.subplots(figsize=(5.0, 5.4))
loc.boundary.plot(ax=ax, color=GRIS_CLARO, lw=0.4)
ax.scatter(xe[~en_chap], ye[~en_chap], s=0.3, color=GRIS, alpha=0.45, rasterized=True)
ax.scatter(xp, yp, s=0.3, color=NARANJA, alpha=0.6, rasterized=True)
ax.scatter(xe[en_chap], ye[en_chap], s=3, color=AZUL, zorder=4)
chap_u.boundary.plot(ax=ax, color=TINTA, lw=1.0)
ax.set_xlim(xe.min() - 500, xe.max() + 500); ax.set_ylim(ye.min() - 500, ye.max() + 500)
ax.set_aspect("equal"); ax.axis("off"); escala(ax, 5, pos=(0.06, 0.93))
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=GRIS, label="Entrenamiento fuera de Chapinero: 38.337"),
                   Line2D([], [], marker="o", ls="", color=NARANJA, label="Prueba (Kaggle): 10.286"),
                   Line2D([], [], marker="o", ls="", color=AZUL, label="Entrenamiento en Chapinero: 307")],
          loc="lower right", fontsize=12, markerscale=2, frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)
guardar(fig, "mapa_datos")

# 3. Valor del suelo, Catastro 2021 (4,6 x 5,4). Necesita las capas de Catastro (script 02); si no están, se conserva
#    la imagen guardada.
if not existen(DIR_CATASTRO / "manzanas_geometria_2021.parquet", DIR_CATASTRO / "valor_referencia_2019_2021.parquet"):
    print("mapa_valor_suelo: sin las capas de Catastro se conserva la imagen guardada (PS3_RECALCULAR=02 las descarga)")
else:
    man = gpd.read_parquet(DIR_CATASTRO / "manzanas_geometria_2021.parquet")
    vr = pd.read_parquet(DIR_CATASTRO / "valor_referencia_2019_2021.parquet")
    vr = vr[vr["anio"] == 2021].groupby("MANCODIGO")["V_REF"].median()
    man = man.merge(vr.rename("vref"), left_on="MANCODIGO", right_index=True, how="inner").to_crs(9377)
    man = man[man["vref"] > 0]
    fig, ax = plt.subplots(figsize=(4.4, 5.4))
    norma = LogNorm(vmin=0.4e6, vmax=8e6)
    man.plot(ax=ax, column="vref", cmap="Blues", norm=norma, linewidth=0, rasterized=True)
    chap_u.boundary.plot(ax=ax, color=NARANJA, lw=1.4)
    ax.set_xlim(xe.min() - 300, xe.max() + 300); ax.set_ylim(ye.min() - 300, ye.max() + 300)
    ax.set_aspect("equal"); ax.axis("off"); escala(ax, 5)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norma, cmap="Blues"), ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, aspect=30)
    cb.set_ticks([0.5e6, 1e6, 2e6, 4e6, 8e6]); cb.set_ticklabels(["0,5", "1", "2", "4", "8"])
    cb.set_label("Valor del suelo 2021 (millones por m²)")
    guardar(fig, "mapa_valor_suelo")

# 4. Boosting paso a paso con una sola variable: precio frente al área (entrenamiento sin Chapinero). Ajusta un modelo
# didáctico; no se rehace salvo que se pida con PS3_RECALCULAR=20 o que falte la imagen (los datos y modelos quedaron
# cerrados con la versión final de las presentaciones).
if not recalcular("20") and (OUT / "boosting_1d.png").exists():
    print("boosting_1d: se conserva la imagen existente")
else:
  tv = pd.read_parquet(DIR_TEXTO / "texto_variables.parquet")[["property_id", "area_imp"]]
  tr = pd.read_csv(DIR_RAW / "train.csv")[["property_id", "price"]].merge(tv, on="property_id")
  tr = tr[tr["area_imp"].between(30, 400)].copy()
  tr["precio"] = tr["price"] / 1e6
  Xa, ya_ = tr[["area_imp"]].to_numpy(), tr["precio"].to_numpy()
  m1 = lgb.LGBMRegressor(n_estimators=241, learning_rate=0.05, num_leaves=4, min_child_samples=100, verbose=-1, random_state=2026).fit(Xa, ya_)
  rejilla = np.linspace(30, 400, 400)[:, None]
  rng = np.random.default_rng(1)
  mu_ = rng.choice(len(tr), 2500, replace=False)
  pasos = [(0, "0 árboles: el promedio"), (1, "1 árbol"), (20, "20 árboles"), (241, "241 árboles")]
  fig, axs = plt.subplots(1, 4, figsize=(12.3, 3.3), sharey=True)
  for ax, (k, tit) in zip(axs, pasos):
      ax.scatter(Xa[mu_, 0], ya_[mu_], s=4, color=GRIS_CLARO, alpha=0.7, rasterized=True)
      if k == 0:
          pred, pg = np.full(len(rejilla), ya_.mean()), np.full(len(ya_), ya_.mean())
      else:
          pred, pg = m1.predict(rejilla, num_iteration=k), m1.predict(Xa, num_iteration=k)
      ax.plot(rejilla[:, 0], pred, color=AZUL, lw=2.4, drawstyle="steps-mid")
      ax.set_title(f"{tit}\nerror medio: {np.mean(np.abs(pg - ya_)):.0f} millones".replace(".", ","), fontsize=13)
      ax.set_xlabel("Área (m²)")
      ax.xaxis.set_major_formatter(MILES); ax.yaxis.set_major_formatter(MILES)
  axs[0].set_ylabel("Precio (millones)")
  guardar(fig, "boosting_1d")

# 5. Pliegues de cada esquema en el mapa (12,3 x 3,5)
pl = pd.read_parquet(DATOS / "ppt_pliegues_mapa.parquet")
esq = ["CV aleatoria", "Bloques de 2 km", "Clústeres k-means", "kNNDM", "Chapinero fuera"]
fig, axs = plt.subplots(1, 5, figsize=(12.3, 3.5))
for ax, e in zip(axs, esq):
    d = pl[pl["esquema"] == e]
    xx, yy = proy(d["lon"], d["lat"])
    v = d["validacion"].to_numpy()
    ax.scatter(xx[~v], yy[~v], s=0.15, color=GRIS_CLARO, rasterized=True)
    tam_v = {"CV aleatoria": 0.15, "kNNDM": 0.3}.get(e, 0.6 if e != "Chapinero fuera" else 2)
    ax.scatter(xx[v], yy[v], s=tam_v, color=NARANJA, rasterized=True, zorder=3)
    chap_u.boundary.plot(ax=ax, color=TINTA, lw=0.6)
    ax.set_xlim(xe.min() - 500, xe.max() + 500); ax.set_ylim(ye.min() - 500, ye.max() + 500)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(e, fontsize=14)
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=NARANJA, label="Avisos que se validan en un pliegue"),
                    Line2D([], [], marker="o", ls="", color=GRIS_CLARO, label="Avisos con los que se entrena")],
           loc="outside lower center", ncol=2, fontsize=13, markerscale=3)
guardar(fig, "pliegues")

# 6. Pronóstico: precio por m2 predicho en Chapinero (4,6 x 5,4)
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
AZULES = LinearSegmentedColormap.from_list("azules", plt.cm.Blues(np.linspace(0.3, 1.0, 256)))
pm2 = pru["pred_final"] / pru["area_imp"]
ok = np.isfinite(pm2) & pru["area_imp"].between(20, 1000)
fig, ax = plt.subplots(figsize=(4.6, 5.4))
loc.boundary.plot(ax=ax, color=GRIS_CLARO, lw=0.4)
norma2 = Normalize(vmin=4, vmax=14)
sc = ax.scatter(xp[ok], yp[ok], c=pm2[ok], s=6, cmap=AZULES, norm=norma2, rasterized=True)
chap.boundary.plot(ax=ax, color=TINTA, lw=1.0)
ax.set_xlim(ZOOM[0], ZOOM[1]); ax.set_ylim(ZOOM[2], ZOOM[3])
ax.set_aspect("equal"); ax.axis("off"); escala(ax, 1, pos=(0.60, 0.04))
cb = fig.colorbar(sc, ax=ax, orientation="vertical", fraction=0.08, pad=0.02, aspect=30)
cb.set_label("Precio predicho por m² (millones)"); cb.ax.yaxis.set_major_formatter(COMA)
guardar(fig, "pronostico_mapa")

# 7. Errores con signo en Chapinero fuera (4,6 x 5,4)
ec = pd.read_parquet(DATOS / "ppt_errores_chapinero.parquet")
xc, yc_ = proy(ec["lon"], ec["lat"])
fig, ax = plt.subplots(figsize=(4.6, 5.4))
loc.boundary.plot(ax=ax, color=GRIS_CLARO, lw=0.4)
e_ = ec["error"].to_numpy()
tam = 6 + 60 * np.clip(np.abs(e_) / 400, 0, 1)
ax.scatter(xc[e_ > 0], yc_[e_ > 0], s=tam[e_ > 0], color=ROJO, alpha=0.75, label="Predice un precio\nmayor que el real")
ax.scatter(xc[e_ <= 0], yc_[e_ <= 0], s=tam[e_ <= 0], color=AZUL, alpha=0.75, label="Predice un precio\nmenor que el real")
chap.boundary.plot(ax=ax, color=TINTA, lw=1.0)
ax.set_xlim(ZOOM[0], ZOOM[1]); ax.set_ylim(ZOOM[2], ZOOM[3])
ax.set_aspect("equal"); ax.axis("off"); escala(ax, 1, pos=(0.60, 0.04))
fig.legend(loc="outside right center", fontsize=12, markerscale=1, title="Tamaño del punto:\ntamaño del error", title_fontsize=12, alignment="left")
guardar(fig, "errores_mapa")

# 8. Mapa de confianza (4,4 x 5,4)
catg = np.where(pru["n_fuera"] >= 2, 2, np.where(pru["n_fuera"] == 1, 1, 0))
fig, ax = plt.subplots(figsize=(4.4, 5.4))
loc.boundary.plot(ax=ax, color=GRIS_CLARO, lw=0.4)
for k2, c, lab in [(0, GRIS, "Ninguna variable\nfuera del rango"), (1, NARANJA, "Una variable fuera"), (2, ROJO, "Dos o más fuera")]:
    m_ = catg == k2
    ax.scatter(xp[m_], yp[m_], s=2.5, color=c, alpha=0.85, rasterized=True, label=f"{lab}: {m_.sum():,}".replace(",", "."))
chap.boundary.plot(ax=ax, color=TINTA, lw=1.0)
ax.set_xlim(ZOOM[0], ZOOM[1]); ax.set_ylim(ZOOM[2], ZOOM[3])
ax.set_aspect("equal"); ax.axis("off"); escala(ax, 1, pos=(0.60, 0.04))
fig.legend(loc="outside right center", fontsize=12, markerscale=4)
guardar(fig, "confianza_mapa")

# 9. Dependencia SHAP (apéndice, 12,3 x 3,6)
sh = pd.read_parquet(DATOS / "shap_prueba.parquet"); sh_tr = pd.read_parquet(DATOS / "shap_entrenamiento.parquet")
va_tr = pd.read_parquet(DATOS / "valores_entrenamiento.parquet"); va_te = pd.read_parquet(DATOS / "valores_prueba.parquet")
fig, axs = plt.subplots(1, 2, figsize=(12.3, 3.6))
for ax, v, etq, f in [(axs[0], "area_imp", "Área (m²)", lambda s: s),
                      (axs[1], "ln_vref_300m", "Valor del suelo a 300 m (millones por m², escala logarítmica)", lambda s: np.exp(s) / 1e6)]:
    ax.scatter(f(va_tr[v]), sh_tr[v], s=3, color=GRIS_CLARO, alpha=0.6, rasterized=True, label="Entrenamiento")
    ax.scatter(f(va_te[v]), sh[v], s=3, color=NARANJA, alpha=0.3, rasterized=True, label="Prueba (Chapinero)")
    ax.set_xlabel(etq); ax.axhline(0, color=GRIS, lw=0.6); ax.yaxis.set_major_formatter(MILES)
axs[0].set_xlim(0, 500); axs[0].set_ylabel("Aporte a la predicción (millones)")
axs[1].set_xscale("log"); axs[1].set_xlim(0.3, 12); axs[1].set_xticks([0.5, 1, 2, 4, 8]); axs[1].xaxis.set_major_formatter(COMA)
hi = 5.9  # percentil 99 del entrenamiento completo (datos_ppt3.py)
axs[1].axvline(hi, color=TINTA, lw=1.5); axs[1].text(hi * 1.07, -20, "percentil 99 del\nentrenamiento", ha="left", fontsize=12)
axs[0].legend(loc="upper left", markerscale=4)
guardar(fig, "dependencia")

# 9b. Solo el área (presentación de comparación, 6,1 x 3,5)
fig, ax = plt.subplots(figsize=(6.1, 3.5))
ax.scatter(va_tr["area_imp"], sh_tr["area_imp"], s=3, color=GRIS_CLARO, alpha=0.6, rasterized=True, label="Entrenamiento")
ax.scatter(va_te["area_imp"], sh["area_imp"], s=3, color=NARANJA, alpha=0.3, rasterized=True, label="Prueba (Chapinero)")
ax.set_xlim(0, 500); ax.axhline(0, color=GRIS, lw=0.6); ax.yaxis.set_major_formatter(MILES)
ax.set_xlabel("Área (m²)"); ax.set_ylabel("Aporte a la predicción (millones)")
ax.legend(loc="upper left", markerscale=4)
guardar(fig, "dependencia_area")

# 10. Distancias al vecino de entrenamiento (apéndice, 6,5 x 4,4)
dv = pd.read_parquet(DATOS / "distancias_vecino.parquet")
ESTILO = {"Prueba real (Kaggle)": (TINTA, 3.0), "CV aleatoria": (GRIS_CLARO, 2.0), "Bloques de 2 km": (GRIS, 1.6),
          "Clústeres k-means": ("#4D5656", 1.6), "Chapinero fuera": (NARANJA, 2.0), "kNNDM": (AZUL, 2.4)}
fig, ax = plt.subplots(figsize=(6.5, 4.4))
for e, (c, lw) in ESTILO.items():
    d = np.sort(np.maximum(dv.loc[dv["esquema"] == e, "distancia_m"].to_numpy(), 1))
    ax.plot(d, np.arange(1, len(d) + 1) / len(d) * 100, color=c, lw=lw, label=e,
            ls="--" if e == "Prueba real (Kaggle)" else "-")
ax.set_xscale("log"); ax.set_xlim(1, 8000); ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["0 a 1", "10", "100", "1.000"])
ax.set_xlabel("Distancia al aviso de entrenamiento más cercano (metros)"); ax.set_ylabel("Avisos acumulados (%)")
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), fontsize=12)
guardar(fig, "distancias")

# 10b. Ley de Zipf en los avisos (lámina de texto como datos, 5,8 x 1,75): frecuencia documental por rango del término
zf = pd.read_csv(DATOS / "ppt_texto_zipf.csv")
fig, ax = plt.subplots(figsize=(5.8, 1.75))
ax.plot(zf["rango"], zf["avisos"], color=AZUL, lw=2.2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([1, 10, 100, 1000, 10000]); ax.set_xticklabels(["1", "10", "100", "1.000", "10.000"])
ax.set_yticks([10, 100, 1000, 10000]); ax.set_yticklabels(["10", "100", "1.000", "10.000"])
ax.minorticks_off()
ax.set_xlabel("Rango del término (escala logarítmica)", fontsize=12); ax.set_ylabel("Avisos", fontsize=12)
guardar(fig, "texto_zipf")

# 11. Fórmulas (mathtext, transparentes)
FORM = {"f_mae": r"$\mathrm{MAE}=\dfrac{1}{n}\sum_{i=1}^{n}\left|\,\hat{y}_i-y_i\,\right|$",
        "f_rosen": r"$P_i=f\,(c_{i,1},\;c_{i,2},\;\ldots,\;c_{i,n})+u_i$",
        "f_boosting": r"$\hat{y}(x)=\bar{y}+\sum_{m=1}^{M}\eta\,f_m(x),\qquad \eta=0{,}05$",
        "f_sesgo": "CV aleatoria: $2\\sigma^2(1-\\rho)$\u2003\u2003lugar nuevo: $2\\sigma^2$",
        "f_shap": r"$\hat{y}(x)=\phi_0+\sum_{j=1}^{p}\,\phi_j\,(x)$",
        "f_oferta": r"$O_i=(1-m)\,\hat{y}_i\qquad \mathrm{acepta\ si}\ \ O_i\geq(1-d)\,V_i$",
        "f_estiramiento": r"$\tilde{y}=\bar{y}+1{,}08\,(\hat{y}-\bar{y}),\qquad \bar{y}=654$",
        "f_tfidf": r"$\mathrm{tfidf}(t,d)=\mathrm{tf}(t,d)\cdot\ln\dfrac{D}{\mathrm{df}(t)}$",
        "f_ridge": r"$\min_{a,\,b}\ \sum_i\left(\ln p_i-a-x_i'b\right)^2+\lambda\sum_j b_j^2$",
        "f_atencion": "Atención$(Q,K,V)=\\mathrm{softmax}\\left(\\dfrac{QK^{\\top}}{\\sqrt{d_k}}\\right)V$"}
for k, tex in FORM.items():
    fig = plt.figure(figsize=(8, 0.9))
    fig.text(0.0, 0.5, tex, fontsize=26, va="center", ha="left", color=TINTA)
    fig.savefig(OUT / f"{k}.png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
print("fórmulas listas")
