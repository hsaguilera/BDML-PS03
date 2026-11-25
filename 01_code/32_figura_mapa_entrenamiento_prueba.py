"""32 · Imagen de las láminas: mapa de entrenamiento (gris) y prueba (naranja) en la coordenada exacta de cada aviso.

Qué hace (antes document/presentaciones/32_figura_mapa_entrenamiento_prueba.py)
    Dibuja el mapa al tamaño de su lugar en la lámina, con el borde de Chapinero y letra de 12 puntos o más.

Salida (02_outputs/figures/laminas/, png a 300 ppp; se versiona)
    mapa_entrenamiento_prueba.png
Insumos: script 01 (localidades) y 12 (entrenamiento_mapa.parquet y prueba_mapa.parquet).
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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = IMG
STORES = DIR_PROC  # carpeta de intermedios del repositorio
for _ruta in fm.findSystemFonts():                 # Calibri si está instalada (Windows); si no, la por defecto
    if Path(_ruta).name.lower() in ("calibri.ttf", "calibril.ttf"):
        fm.fontManager.addfont(_ruta)
plt.rcParams.update({"font.family": "Calibri", "font.size": 13, "legend.fontsize": 12, "text.color": "#1B2A41",
                     "figure.constrained_layout.use": True})
NARANJA, GRIS, GRIS_CLARO, TINTA = "#D9822B", "#7F8C8D", "#C8CDD0", "#1B2A41"


def proy(lon, lat):
    s = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326).to_crs(9377)
    return s.x.to_numpy(), s.y.to_numpy()


loc = gpd.read_file(STORES / "localidades_bogota.geojson").to_crs(9377)
chap = loc[loc["localidad"] == "Chapinero"]
ent = pd.read_parquet(DATOS / "entrenamiento_mapa.parquet")
pru = pd.read_parquet(DATOS / "prueba_mapa.parquet")
xe, ye = proy(ent["lon"], ent["lat"])
xp, yp = proy(pru["lon"], pru["lat"])
n_chap = int((ent["localidad"].values == "Chapinero").sum())
# Parte urbana de Chapinero: el polígono recortado a la envolvente de los avisos (más 250 m), como en 20_figuras_laminas.py
env = gpd.GeoSeries(gpd.points_from_xy(np.r_[xe, xp], np.r_[ye, yp]), crs=9377).union_all().convex_hull.buffer(250)
chap_u = chap.geometry.intersection(env)


def miles(v):
    return f"{v:,}".replace(",", ".")


fig, ax = plt.subplots(figsize=(5.0, 5.4))
loc.boundary.plot(ax=ax, color=GRIS_CLARO, lw=0.4)
ax.scatter(xp, yp, s=0.4, color=NARANJA, alpha=0.6, rasterized=True)
ax.scatter(xe, ye, s=0.3, color=GRIS, alpha=0.5, rasterized=True)
chap_u.boundary.plot(ax=ax, color=TINTA, lw=1.0)
ax.set_xlim(xe.min() - 500, xe.max() + 500); ax.set_ylim(ye.min() - 500, ye.max() + 500)
ax.set_aspect("equal"); ax.axis("off")
x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
xa, ya = x0 + 0.06 * (x1 - x0), y0 + 0.93 * (y1 - y0)
ax.plot([xa, xa + 5000], [ya, ya], color=TINTA, lw=2, solid_capstyle="butt")
ax.text(xa + 2500, ya + 0.015 * (y1 - y0), "5 km", ha="center", va="bottom", fontsize=12, color=TINTA)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=GRIS,
                          label=f"Entrenamiento: {miles(len(ent))} ({n_chap} en Chapinero)"),
                   Line2D([], [], marker="o", ls="", color=NARANJA, label=f"Prueba: {miles(len(pru))}")],
          loc="lower right", fontsize=12, markerscale=2, frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)
fig.savefig(OUT / "mapa_entrenamiento_prueba.png", dpi=300)
print("imagen mapa_entrenamiento_prueba", len(ent), len(pru), n_chap)
