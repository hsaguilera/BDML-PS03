"""33 · Imágenes de las láminas de las etapas 2 a 4: fórmulas, mapas de los seis esquemas de validación y enjambre SHAP.

Qué hace (antes document/presentaciones/33_figuras_laminas_etapas2a4.py)
    Fórmulas con mathtext a 26 puntos y 300 ppp (la lámina las pone a 0,75 de su tamaño o más), mapas de un pliegue de cada
    esquema en 2 × 3 y el gráfico de enjambre de los valores SHAP, al tamaño exacto de su lugar en la lámina.

Salidas (02_outputs/figures/laminas/etapas2a4/, png; se versionan)
    f_actualiza, f_estira, f_ganancia, f_gemelo, f_hoja, f_inicio, f_objetivo, f_residuo, f_sesgo_a, f_sesgo_b,
    pliegues_6 y shap_enjambre
Insumos: scripts 01 y 12 (mapas, ppt_pliegues_mapa.parquet, shap_prueba.parquet y valores_prueba.parquet) y 13.
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

OUT = IMG / "etapas2a4"
STORES = DIR_PROC  # carpeta de intermedios del repositorio
OUT.mkdir(parents=True, exist_ok=True)
for _ruta in fm.findSystemFonts():                 # Calibri si está instalada (Windows); si no, la por defecto
    if Path(_ruta).name.lower() in ("calibri.ttf", "calibril.ttf"):
        fm.fontManager.addfont(_ruta)
plt.rcParams.update({"font.family": "Calibri", "font.size": 13, "mathtext.fontset": "stixsans", "text.color": "#1B2A41"})
NARANJA, GRIS_CLARO, TINTA = "#D9822B", "#C8CDD0", "#1B2A41"

# ------------------------------------------------------------------------------------------------------------------
# Fórmulas
# ------------------------------------------------------------------------------------------------------------------
FORM = {
    "f_objetivo": r"$\min_{\mathrm{cortes},\ w}\ \ \sum_{i}\left(r_i-w_{\,j(i)}\right)^2\ +\ \lambda\sum_{j=1}^{J}w_j^{\,2}$",
    "f_residuo": r"$r_i=y_i-\hat{y}_i^{\,(m-1)}$",
    "f_hoja": r"$w_j=\dfrac{S_j}{n_j+\lambda}$",
    "f_ganancia": r"$\mathrm{ganancia}=\dfrac{S_{\mathrm{sí}}^{\,2}}{n_{\mathrm{sí}}+\lambda}+\dfrac{S_{\mathrm{no}}^{\,2}}{n_{\mathrm{no}}+\lambda}"
                  r"-\dfrac{S^{\,2}}{n+\lambda}$",
    "f_actualiza": r"$\hat{y}^{\,(m)}(x)=\hat{y}^{\,(m-1)}(x)+\eta\,w_m(x)$",
    "f_inicio": r"$\hat{y}^{\,(0)}=\bar{y}=654{,}5$",
    "f_estira": r"$\tilde{y}=654+1{,}08\,(\hat{y}-654)$",
    "f_sesgo_a": r"CV aleatoria: $2\sigma^2(1-\rho)$",
    "f_sesgo_b": r"lugar nuevo: $2\sigma^2$",
    "f_gemelo": r"$\hat{y}^{\,\mathrm{final}}=0{,}3\,\tilde{y}+0{,}7\,p_{\mathrm{gemelo}}$",}
for k, tex in FORM.items():
    fig = plt.figure(figsize=(9, 1.2))
    fig.text(0.0, 0.5, tex, fontsize=26, va="center", ha="left", color=TINTA)
    fig.savefig(OUT / f"{k}.png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
print("fórmulas listas")


# ------------------------------------------------------------------------------------------------------------------
# Mapas de los pliegues en 2 x 3 (6,3 x 5,45 pulgadas), con la prueba real como sexto panel
# ------------------------------------------------------------------------------------------------------------------
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
ESQ = ["CV aleatoria", "Bloques de 2 km", "Clústeres k-means", "kNNDM", "Chapinero fuera", "Prueba real (Kaggle)"]
fig = plt.figure(figsize=(6.3, 5.45))
W_, H_, ARR, ABJ = 2.08, 2.12, 0.33, 0.55
axs = [fig.add_axes([(0.02 + c * 2.1) / 6.3, (ABJ + (1 - f) * (H_ + ARR)) / 5.45, W_ / 6.3, H_ / 5.45])
       for f in range(2) for c in range(3)]
for ax, e in zip(axs, ESQ):
    if e == "Prueba real (Kaggle)":
        ax.scatter(xe, ye, s=0.08, color=GRIS_CLARO, rasterized=True)
        ax.scatter(xp, yp, s=0.2, color=NARANJA, rasterized=True, zorder=3)
    else:
        d = pl[pl["esquema"] == e]
        xx, yy = proy(d["lon"], d["lat"])
        v = d["validacion"].to_numpy()
        ax.scatter(xx[~v], yy[~v], s=0.08, color=GRIS_CLARO, rasterized=True)
        tam = {"CV aleatoria": 0.1, "kNNDM": 0.2, "Chapinero fuera": 1.2}.get(e, 0.35)
        ax.scatter(xx[v], yy[v], s=tam, color=NARANJA, rasterized=True, zorder=3)
    chap_u.boundary.plot(ax=ax, color=TINTA, lw=0.6)
    ax.set_xlim(xe.min() - 500, xe.max() + 500); ax.set_ylim(ye.min() - 500, ye.max() + 500)
    ax.set_aspect("equal", adjustable="datalim"); ax.axis("off")
    ax.text(0.5, 1.02, e, transform=ax.transAxes, ha="center", va="bottom", fontsize=13, color=TINTA)
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=NARANJA, label="Avisos que se predicen (validación o prueba)"),
                    Line2D([], [], marker="o", ls="", color=GRIS_CLARO, label="Avisos con los que se entrena")],
           loc="lower center", ncol=1, fontsize=12, markerscale=1.6, frameon=False, borderaxespad=0.1)
fig.savefig(OUT / "pliegues_6.png", dpi=300)
plt.close(fig)
print("mapas listos en", OUT)


# ------------------------------------------------------------------------------------------------------------------
# Enjambre de valores SHAP (12,3 x 4,55 pulgadas): las 12 variables de mayor aporte medio en la prueba
# ------------------------------------------------------------------------------------------------------------------
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

NOMBRE = {"puntaje_texto_ridge": "Puntaje de precio del texto", "area_imp": "Área", "banos_imp": "Baños", "bedrooms": "Alcobas",
          "ln_dist_embajada_m": "Distancia a una embajada", "ln_vref_300m": "Valor del suelo a 300 m",
          "estrato_medio_300m": "Estrato medio a 300 m", "prop_apartamentos_500m": "Apartamentos a 500 m",
          "elevacion_m": "Elevación", "ln_dist_zonat_m": "Distancia a la Zona T", "lsa_09": "Componente 9 del texto (LSA)",
          "lsa_04": "Componente 4 del texto (LSA)", "log_area": "Logaritmo del área", "rooms": "Ambientes",
          "y_km": "Coordenada norte", "lsa_03": "Componente 3 del texto (LSA)", "parqueaderos_imp": "Parqueaderos", "ln_avaluo_ph_300m": "Avalúo a 300 m"}
sh = pd.read_parquet(DATOS / "shap_prueba.parquet")
va = pd.read_parquet(DATOS / "valores_prueba.parquet")
cols = [c for c in va.columns]
imp = sh[cols].abs().mean().sort_values(ascending=False)
top = list(imp.index[:12])
cmap = LinearSegmentedColormap.from_list("gris_azul", ["#C8CDD0", "#7FA7D4", "#1F5FA8", "#1B2A41"])
fig = plt.figure(figsize=(12.3, 4.15))
ax = fig.add_axes([0.25, 0.165, 0.66, 0.82])
rng = np.random.default_rng(2026)
for k, v in enumerate(top):
    yk = len(top) - 1 - k
    x = sh[v].to_numpy()
    val = va[v].to_numpy()
    ok = np.isfinite(val)
    rango = pd.Series(val[ok]).rank(pct=True).to_numpy()
    # jitter vertical proporcional a la densidad local, como un enjambre
    orden = np.argsort(x)
    dens = np.zeros(len(x))
    h_ = np.histogram(x, bins=120)
    idx_b = np.clip(np.digitize(x, h_[1]) - 1, 0, 119)
    dens = h_[0][idx_b] / h_[0].max()
    jit = (rng.random(len(x)) - 0.5) * 0.8 * np.sqrt(dens)
    ax.scatter(x[~ok], yk + jit[~ok], s=1.2, color="#E3E7EB", rasterized=True)
    ax.scatter(x[ok], yk + jit[ok], s=1.2, c=rango, cmap=cmap, vmin=0, vmax=1, rasterized=True)
    ax.text(-0.012, yk, f"{NOMBRE.get(v, v)} ({imp[v]:.0f})".replace(".", ","), transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=13, color=TINTA)
ax.axvline(0, color="#7F8C8D", lw=0.8)
ax.set_yticks([]); ax.set_ylim(-0.6, len(top) - 0.4)
ax.set_xlabel("Valor SHAP: lo que la variable suma o resta a la predicción del aviso (millones)", fontsize=13)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}".replace(",", ".").replace("-", "−")))
ax.tick_params(axis="x", labelsize=12)
for s_ in ("top", "right", "left"):
    ax.spines[s_].set_visible(False)
cax = fig.add_axes([0.935, 0.25, 0.012, 0.6])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax)
cb.set_ticks([0, 1]); cb.set_ticklabels(["bajo", "alto"]); cb.ax.tick_params(labelsize=12)
cb.set_label("Valor de la variable", fontsize=12)
fig.savefig(OUT / "shap_enjambre.png", dpi=300)
plt.close(fig)
print("enjambre listo")
