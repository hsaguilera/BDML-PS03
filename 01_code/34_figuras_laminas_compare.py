"""34 · Imágenes de las láminas de compare: fórmulas de cada familia y mapas de un pliegue de cada validación.

Qué hace (antes document/presentaciones/34_figuras_laminas_compare.py)
    Fórmulas con mathtext (MAE, descomposición del error, regresión lineal, Elastic Net, árbol, bosque, boosting, atención,
    residuo y optimismo) y los mapas de un pliegue de la CV aleatoria, los clústeres y Chapinero fuera, junto a la prueba.

Salidas (02_outputs/figures/laminas/compare_nuevo/, png; se versionan)
    c_arbol, c_atencion, c_boost, c_bosque, c_descomp, c_en, c_mae, c_ols, c_optimismo, c_residuo y mapas_validaciones
Insumos: scripts 01, 12 y 13 (ppt_pliegues_mapa.parquet).
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

from functions.entorno_ps3 import (ARCHIVO_AMENIDADES, ARCHIVO_LOCALIDADES, CENTRO, DIR_CACHE_TEXTO,  # noqa: E402,F401
                                   DIR_MODELOS, DIR_OSM, DIR_PRED, DIR_RAW, DIR_TEXTO, FACTOR, existen, log,
                                   recalcular)
from functions.laminas_entorno import DATOS, IMG  # noqa: E402,F401  (00_data/processed/laminas/ y 02_outputs/figures/laminas/)

import matplotlib  # noqa: E402

matplotlib.rcdefaults()  # entorno_ps3 fija un estilo para las figuras de análisis; aquí se usa el de las láminas

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = IMG / "compare_nuevo"
STORES = DIR_PROC  # carpeta de intermedios del repositorio
OUT.mkdir(parents=True, exist_ok=True)
for _ruta in fm.findSystemFonts():                 # Calibri si está instalada (Windows); si no, la por defecto
    if Path(_ruta).name.lower() in ("calibri.ttf", "calibril.ttf"):
        fm.fontManager.addfont(_ruta)
plt.rcParams.update({"font.family": "Calibri", "font.size": 13, "mathtext.fontset": "stixsans", "text.color": "#1B2A41"})
NARANJA, GRIS_CLARO, TINTA = "#D9822B", "#C8CDD0", "#1B2A41"

FORM = {
    "c_descomp": r"$\mathrm{E}\left[(y-\hat{f}(x))^2\right]=\mathrm{sesgo}^2+\mathrm{varianza}+\sigma^2$",
    "c_mae": r"$\mathrm{MAE}=\dfrac{1}{n}\sum_{i=1}^{n}\left|\hat{y}_i-y_i\right|$",
    "c_ols": r"$\min_{\beta}\ \sum_{i}\left(y_i-x_i'\beta\right)^2$",
    "c_en": r"$\min_{\beta}\ \dfrac{1}{2n}\sum_{i}\left(y_i-x_i'\beta\right)^2+\lambda\left[\alpha\sum_{j}|\beta_j|"
            r"+\dfrac{1-\alpha}{2}\sum_{j}\beta_j^{\,2}\right]$",
    "c_arbol": r"$\hat{f}(x)=\sum_{j=1}^{J}\gamma_j\,\mathbf{1}\left(x\in R_j\right)$",
    "c_bosque": r"$\hat{f}(x)=\dfrac{1}{B}\sum_{b=1}^{B}T_b(x),\quad B=300$",
    "c_boost": r"$\hat{f}_M(x)=\hat{f}_0+\eta\sum_{m=1}^{M}T_m(x)$",
    "c_residuo": r"$T_m\ \mathrm{ajusta}\ \ r_i=y_i-\hat{f}_{m-1}(x_i)$",
    "c_atencion": r"$\hat{y}_i=g\left(e_i,\ \sum_{j=1}^{20}a_{ij}\,v_j\right),\quad a_{ij}=\mathrm{softmax}_j"
                  r"\left(q_i\cdot k_j/\sqrt{d}\right)$",
    "c_optimismo": r"CV aleatoria: $2\sigma^2(1-\rho)$;   lugar nuevo: $2\sigma^2$",
}
for k, tex in FORM.items():
    fig = plt.figure(figsize=(11, 1.4))
    fig.text(0.0, 0.5, tex, fontsize=26, va="center", ha="left", color=TINTA)
    fig.savefig(OUT / f"{k}.png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
print("fórmulas listas")


def proy(lon, lat):
    s = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326).to_crs(9377)
    return s.x.to_numpy(), s.y.to_numpy()


loc = gpd.read_file(STORES / "localidades_bogota.geojson").to_crs(9377)
chap = loc[loc["localidad"] == "Chapinero"]
ent = pd.read_parquet(DATOS / "entrenamiento_mapa.parquet")
pru = pd.read_parquet(DATOS / "prueba_mapa.parquet")
xe, ye = proy(ent["lon"], ent["lat"])
xp, yp = proy(pru["lon"], pru["lat"])
env = gpd.GeoSeries(gpd.points_from_xy(np.r_[xe, xp], np.r_[ye, yp]), crs=9377).union_all().convex_hull.buffer(250)
chap_u = chap.geometry.intersection(env)
pl = pd.read_parquet(DATOS / "ppt_pliegues_mapa.parquet")
ESQ = [("CV aleatoria", "CV aleatoria: pliegue 1 de 5"), ("Clústeres k-means", "Clústeres: pliegue 1 de 10"),
       ("Chapinero fuera", "Chapinero fuera: 1 pliegue"), ("Prueba real (Kaggle)", "Prueba real de Kaggle")]
WF, HF = 5.3, 5.45
fig = plt.figure(figsize=(WF, HF))
W_, H_, ARR, ABJ = 2.55, 2.2, 0.32, 0.5
axs = [fig.add_axes([(0.05 + c * 2.65) / WF, (ABJ + (1 - f) * (H_ + ARR)) / HF, W_ / WF, H_ / HF])
       for f in range(2) for c in range(2)]
for ax, (e, tit) in zip(axs, ESQ):
    if e == "Prueba real (Kaggle)":
        ax.scatter(xe, ye, s=0.08, color=GRIS_CLARO, rasterized=True)
        ax.scatter(xp, yp, s=0.2, color=NARANJA, rasterized=True, zorder=3)
    else:
        d = pl[pl["esquema"] == e]
        xx, yy = proy(d["lon"], d["lat"])
        v = d["validacion"].to_numpy()
        ax.scatter(xx[~v], yy[~v], s=0.08, color=GRIS_CLARO, rasterized=True)
        tam = {"CV aleatoria": 0.1, "Chapinero fuera": 1.2}.get(e, 0.35)
        ax.scatter(xx[v], yy[v], s=tam, color=NARANJA, rasterized=True, zorder=3)
    chap_u.boundary.plot(ax=ax, color=TINTA, lw=0.6)
    ax.set_xlim(xe.min() - 500, xe.max() + 500); ax.set_ylim(ye.min() - 500, ye.max() + 500)
    ax.set_aspect("equal", adjustable="datalim"); ax.axis("off")
    ax.text(0.5, 1.02, tit, transform=ax.transAxes, ha="center", va="bottom", fontsize=13, color=TINTA)
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=NARANJA, label="Avisos que se predicen (validación o prueba)"),
                    Line2D([], [], marker="o", ls="", color=GRIS_CLARO, label="Avisos con los que se entrena")],
           loc="lower center", ncol=1, fontsize=13, markerscale=1.6, frameon=False, borderaxespad=0.1)
fig.savefig(OUT / "mapas_validaciones.png", dpi=300)
plt.close(fig)
print("mapas listos en", OUT)
