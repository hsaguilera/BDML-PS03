"""05 · Análisis espacial descriptivo de las variables brutas (Kaggle, Earth Engine y OpenStreetMap). No ajusta modelos.

Qué hace
    Responde la Etapa 1 del enunciado: cómo se distribuyen los precios y las variables en el espacio y en qué se diferencia
    Chapinero (la prueba) del resto de Bogotá (el entrenamiento).
    1. Mapas en la coordenada exacta de cada aviso (un punto por coordenada; si varios avisos la comparten, el promedio).
    2. Correlaciones de Pearson y Spearman de cada variable con el precio, el log-precio y el precio por m², aviso por
       aviso, en celdas de 500 m (para las amenidades de OSM) y por zona (Chapinero, vecinas y resto).
    3. Autocorrelación espacial: I de Moran global (10 vecinos, 199 permutaciones), correlograma por bandas de distancia y
       clústeres locales LISA del precio por m².
    4. Precisión de las coordenadas: una parte de los avisos está redondeada a 0,001 grados (malla de unos 110 m).

Salidas
    02_outputs/figures/: mapas_variables_kaggle, mapa_entrenamiento_prueba, mapas_capas_gee, mapas_objetos_osm,
        correlacion_variables_precio, matriz_correlacion_spearman, dispersion_celdas_500m, correlograma_precio,
        lisa_precio_m2, precision_coordenadas
    02_outputs/tables/: correlaciones_precio, correlaciones_por_zona, correlaciones_celdas_500m, moran_global,
        correlograma_precio, precision_coordenadas

Los mapas no llevan mapa base (descargarlo requiere internet). Con PS3_MAPA_BASE=1 se agrega el de Esri (contextily).
Tiempo: unos 2 minutos.
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from esda.moran import Moran, Moran_Local
from libpysal.weights import KNN
from scipy.stats import pearsonr, spearmanr

from functions.entorno_ps3 import (ARCHIVO_AMENIDADES, ARCHIVO_GEE, ARCHIVO_LOCALIDADES, COLOR_TRAIN, CRS_GEO, CRS_M,
                                   DIR_RAW, exigir, guardar_figura, guardar_tabla, log, plt)

warnings.filterwarnings("ignore")
SEMILLA = 2026
USAR_MAPA_BASE = os.environ.get("PS3_MAPA_BASE", "0") == "1"
VECINAS = ["Usaquén", "Barrios Unidos", "Teusaquillo", "Santa Fe"]
COLOR_PRUEBA = "#C44E52"
exigir(ARCHIVO_GEE, ARCHIVO_LOCALIDADES, ARCHIVO_AMENIDADES, como="Corra antes el script 01.")

ETIQUETAS = {
    "price": "Precio (COP)", "precio_mill": "Precio (millones)", "log_precio": "Logaritmo del precio",
    "precio_m2_mill": "Precio por m² (millones)", "surface_covered": "Área cubierta (m²)", "surface_total": "Área total (m²)",
    "rooms": "Ambientes", "bedrooms": "Alcobas", "bathrooms": "Baños", "es_apartamento": "Es apartamento (0/1)",
    "year": "Año de publicación", "month": "Mes de publicación",
    "elevacion_m": "Elevación (m)", "pendiente_grados": "Pendiente (°)", "ndvi_200m": "Vegetación NDVI (200 m)",
    "luces_nocturnas": "Luces nocturnas", "altura_construida_m": "Altura construida (m)",
    "poblacion_100m": "Población (celda de 100 m)",
}
CATEGORIAS = {"bar_cafe": "Bares y cafés", "centro_comercial": "Centros comerciales", "colegio": "Colegios",
              "estacion": "Estaciones de transporte", "hospital": "Hospitales y clínicas", "paradero": "Paraderos de bus",
              "parque": "Parques", "policia": "Policía", "restaurante": "Restaurantes", "supermercado": "Supermercados",
              "universidad": "Universidades", "via_principal": "Vías principales"}
for k, v in CATEGORIAS.items():
    ETIQUETAS[f"osm_{k}"] = f"OSM · {v} (número en la celda)"
for k in ["elevacion_m", "pendiente_grados", "ndvi_200m", "luces_nocturnas", "altura_construida_m", "poblacion_100m"]:
    ETIQUETAS[f"gee_{k}"] = f"GEE · {ETIQUETAS[k]} (promedio en la celda)"


def es(nombre):
    """Etiqueta en español de una variable (si no está en el diccionario, se devuelve igual)."""
    return ETIQUETAS.get(nombre, nombre)


# ------------------------------------------------------------------------------------------------------------------
# 0. Datos: Kaggle + Earth Engine (por coordenada exacta) + localidad
# ------------------------------------------------------------------------------------------------------------------
kag = pd.concat([pd.read_csv(DIR_RAW / "train.csv").assign(muestra="train"),
                 pd.read_csv(DIR_RAW / "test.csv").assign(muestra="test")], ignore_index=True)
gee = pd.read_parquet(ARCHIVO_GEE)
GEE = [c for c in gee.columns if c not in ("lat", "lon")]
df = kag.merge(gee, on=["lat", "lon"], how="left")
assert len(df) == len(kag) and df[GEE].notna().all().all()
df["precio_mill"] = df["price"] / 1e6
df["log_precio"] = np.log(df["price"])
area = df["surface_covered"].fillna(df["surface_total"])
df["precio_m2_mill"] = (df["precio_mill"] / area).where(area.between(15, 1000))
df["es_apartamento"] = (df["property_type"] == "Apartamento").astype(int)

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=CRS_GEO).to_crs(CRS_M)
localidades = gpd.read_file(ARCHIVO_LOCALIDADES).to_crs(CRS_M)
gdf = gdf.sjoin(localidades[["localidad", "geometry"]], how="left", predicate="within").drop(columns="index_right")
gdf = gdf[~gdf.index.duplicated()]
gdf["localidad"] = gdf["localidad"].fillna("Fuera de límites")
gdf["zona"] = np.select([gdf["localidad"] == "Chapinero", gdf["localidad"].isin(VECINAS)], ["Chapinero", "Vecinas"], "Resto")
gdf["x"], gdf["y"] = gdf.geometry.x, gdf.geometry.y
chap = localidades[localidades["localidad"] == "Chapinero"]
tr = gdf[gdf["muestra"] == "train"]
osm_geo = gpd.read_parquet(ARCHIVO_AMENIDADES).to_crs(CRS_M)
osm = osm_geo.copy()
osm["geometry"] = osm.geometry.representative_point()          # un punto por objeto (centro de áreas y líneas)
EXTENT = gdf.total_bounds + np.array([-1000, -1000, 1000, 1000])
log(f"Avisos: {len(gdf):,} (entrenamiento {len(tr):,}) | objetos OSM: {len(osm):,} | capas GEE: {len(GEE)}")


def mapa_base(ax, extent=EXTENT, contorno=True):
    """Límites de localidades, contorno de Chapinero y, si PS3_MAPA_BASE=1, mapa base de Esri."""
    localidades.boundary.plot(ax=ax, color="black", lw=0.3, alpha=0.5)
    if contorno:
        chap.boundary.plot(ax=ax, color="black", lw=1.6)
    ax.set_xlim(extent[0], extent[2]); ax.set_ylim(extent[1], extent[3])
    if USAR_MAPA_BASE:
        import contextily as cx
        try:
            cx.add_basemap(ax, crs=CRS_M, source=cx.providers.Esri.WorldGrayCanvas, attribution_size=4)
        except Exception as e:                       # sin internet: el mapa queda sin fondo
            log(f"Sin mapa base: {type(e).__name__}")
    ax.set_axis_off()


def puntomapa(ax, d, var, titulo, cmap="viridis", robusto=True, s=2.5):
    """Mapa en coordenadas exactas: un punto por coordenada con el promedio de sus avisos, en orden aleatorio."""
    u = d.groupby(["x", "y"], as_index=False).agg(valor=(var, "mean"), n=(var, "size")).dropna(subset=["valor"])
    u = u.sample(frac=1, random_state=SEMILLA)
    kw = dict(vmin=u["valor"].quantile(.02), vmax=u["valor"].quantile(.98)) if robusto else {}
    sc = ax.scatter(u["x"], u["y"], c=u["valor"], s=s, cmap=cmap, linewidths=0, **kw)
    mapa_base(ax)
    plt.colorbar(sc, ax=ax, shrink=0.55)
    ax.set_title(f"{titulo}\n({len(u):,} coordenadas exactas)", fontsize=10)


# ------------------------------------------------------------------------------------------------------------------
# 1 a 3. Mapas de Kaggle, Earth Engine y OpenStreetMap
# ------------------------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(22, 13))
paneles = [(tr, "precio_mill", "Precio (millones, entrenamiento)", "viridis", True),
           (tr, "precio_m2_mill", "Precio por m² (millones, entrenamiento)", "magma", True),
           (gdf, "surface_covered", "Área cubierta (m²)", "cividis", True),
           (gdf, "surface_total", "Área total (m²)", "cividis", True),
           (gdf, "bedrooms", "Alcobas", "YlGnBu", True),
           (gdf, "bathrooms", "Baños", "YlGnBu", True),
           (gdf, "es_apartamento", "Proporción de apartamentos", "PuRd", False),
           (gdf, "year", "Año de publicación (promedio)", "coolwarm", False)]
for ax, (d, v, t, cm, rob) in zip(axes.ravel(), paneles):
    puntomapa(ax, d, v, t, cm, robusto=rob)
fig.suptitle("Variables de Kaggle en coordenadas exactas (un punto por coordenada)", fontweight="bold", fontsize=14)
fig.tight_layout()
guardar_figura(fig, "mapas_variables_kaggle")

fig, ax = plt.subplots(figsize=(8, 10))
tr.plot(ax=ax, markersize=0.4, color=COLOR_TRAIN, alpha=0.4, label=f"Entrenamiento (n={len(tr):,})")
gdf[gdf["muestra"] == "test"].plot(ax=ax, markersize=0.4, color=COLOR_PRUEBA, alpha=0.4,
                                   label=f"Prueba (n={(gdf['muestra'] == 'test').sum():,})")
mapa_base(ax)
ax.legend(markerscale=12, loc="lower right", frameon=True)
ax.set_title("Ubicación de los avisos: entrenamiento frente a prueba (Chapinero)")
guardar_figura(fig, "mapa_entrenamiento_prueba")

fig, axes = plt.subplots(2, 3, figsize=(18, 13))
titulos = {"elevacion_m": "Elevación (m.s.n.m.)", "pendiente_grados": "Pendiente (grados)",
           "ndvi_200m": "NDVI en 200 m (vegetación)", "luces_nocturnas": "Luces nocturnas (nW/cm²/sr)",
           "altura_construida_m": "Altura construida (m)", "poblacion_100m": "Población por celda de 100 m"}
for ax, v, cm in zip(axes.ravel(), GEE, ["terrain", "magma", "Greens", "inferno", "viridis", "cividis"]):
    puntomapa(ax, gdf, v, titulos[v], cm)
fig.suptitle("Capas de Google Earth Engine en la coordenada exacta de cada aviso", fontweight="bold", fontsize=14)
fig.tight_layout()
guardar_figura(fig, "mapas_capas_gee")

cats = osm_geo["categoria"].value_counts().index.tolist()
fig, axes = plt.subplots(3, 4, figsize=(22, 19))
for ax, cat in zip(axes.ravel(), cats):
    capa = osm_geo[osm_geo["categoria"] == cat].cx[EXTENT[0]:EXTENT[2], EXTENT[1]:EXTENT[3]]
    if cat == "parque":
        capa[capa.geom_type.isin(["Polygon", "MultiPolygon"])].plot(ax=ax, color="#2E8B57", lw=0)
        pts_ = capa[~capa.geom_type.isin(["Polygon", "MultiPolygon"])]
        ax.scatter(pts_.geometry.x, pts_.geometry.y, s=3, color="#2E8B57")
    elif cat == "via_principal":
        capa.plot(ax=ax, color="#B22222", lw=0.6)
    else:
        p = capa.geometry.representative_point()
        ax.scatter(p.x, p.y, s=3, color="#3B4CC0", alpha=0.7)
    mapa_base(ax)
    ax.set_title(f"{CATEGORIAS[cat]} (n={len(capa):,})", fontsize=11)
fig.suptitle("Objetos de OpenStreetMap en su ubicación exacta (contorno negro: Chapinero)", fontweight="bold", fontsize=15)
fig.tight_layout()
guardar_figura(fig, "mapas_objetos_osm")

# ------------------------------------------------------------------------------------------------------------------
# 4 y 5. Correlación con el precio (aviso por aviso, entrenamiento) y por zona
# ------------------------------------------------------------------------------------------------------------------
VARS = ["surface_total", "surface_covered", "rooms", "bedrooms", "bathrooms", "es_apartamento", "year"] + GEE
OBJETIVOS = {"precio_mill": "precio", "log_precio": "log-precio", "precio_m2_mill": "precio/m²"}


def correlaciones(d, x, yv):
    ok = d[[x, yv]].dropna()
    if len(ok) < 10 or ok[x].nunique() < 2:
        return np.nan, np.nan, len(ok)
    return pearsonr(ok[x], ok[yv])[0], spearmanr(ok[x], ok[yv])[0], len(ok)


filas = []
for v in VARS:
    fila = {"variable": v}
    for yv, nombre in OBJETIVOS.items():
        p, s, n = correlaciones(tr, v, yv)
        fila[f"Pearson {nombre}"], fila[f"Spearman {nombre}"] = p, s
        if yv == "precio_mill":
            fila["n"] = n
    filas.append(fila)
corr_precio = pd.DataFrame(filas).set_index("variable").sort_values("Spearman precio", key=np.abs, ascending=False)
guardar_tabla(corr_precio.assign(nombre=corr_precio.index.map(es)), "correlaciones_precio")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
orden = corr_precio.sort_values("Spearman precio").index
for ax, nombre in zip(axes, ["precio", "precio/m²"]):
    d = corr_precio.loc[orden]
    pos = np.arange(len(d))
    ax.barh(pos - 0.2, d[f"Pearson {nombre}"], 0.4, label="Pearson", color="#8C8C8C")
    ax.barh(pos + 0.2, d[f"Spearman {nombre}"], 0.4, label="Spearman", color="#4C72B0")
    ax.set_yticks(pos, [es(v) for v in d.index])
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title=f"Correlación con el {nombre}", xlabel="Coeficiente")
axes[0].legend(frameon=False)
fig.suptitle("Correlación de las variables brutas con el precio (entrenamiento, aviso por aviso)", fontweight="bold")
fig.tight_layout()
guardar_figura(fig, "correlacion_variables_precio")

M = tr[["precio_mill", "precio_m2_mill"] + VARS].corr(method="spearman")
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(M)), [es(v) for v in M.columns], rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(len(M)), [es(v) for v in M.index], fontsize=8)
for i in range(len(M)):
    for j in range(len(M)):
        if np.isfinite(M.iloc[i, j]):
            ax.text(j, i, f"{M.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(M.iloc[i, j]) > 0.6 else "black")
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Matriz de correlación de Spearman (entrenamiento)")
guardar_figura(fig, "matriz_correlacion_spearman")

por_zona = {}
for z in ["Chapinero", "Vecinas", "Resto"]:
    d = tr[tr["zona"] == z]
    por_zona[(z, "precio")] = {v: correlaciones(d, v, "precio_mill")[1] for v in VARS}
    por_zona[(z, "precio/m²")] = {v: correlaciones(d, v, "precio_m2_mill")[1] for v in VARS}
    por_zona[(z, "n")] = {v: correlaciones(d, v, "precio_mill")[2] for v in VARS}
guardar_tabla(pd.DataFrame(por_zona), "correlaciones_por_zona")

# ------------------------------------------------------------------------------------------------------------------
# 6. Celdas de 500 m: precio frente a amenidades de OSM y capas de Earth Engine
# ------------------------------------------------------------------------------------------------------------------
TAM = 500


def celda(x, yv):
    return (np.floor(x / TAM).astype(int)).astype(str) + "_" + (np.floor(yv / TAM).astype(int)).astype(str)


tr_c = tr.assign(celda=celda(tr["x"].values, tr["y"].values))
osm_c = osm.assign(celda=celda(osm.geometry.x.values, osm.geometry.y.values))
gdf_c = gdf.assign(celda=celda(gdf["x"].values, gdf["y"].values))
agg = tr_c.groupby("celda").agg(avisos=("price", "size"), precio_mediano=("precio_mill", "median"),
                                precio_m2_mediano=("precio_m2_mill", "median"))
agg = agg[agg["avisos"] >= 5]
agg = agg.join(osm_c.groupby(["celda", "categoria"]).size().unstack(fill_value=0).add_prefix("osm_"), how="left").fillna(0)
agg = agg.join(gdf_c.groupby("celda")[GEE].mean().add_prefix("gee_"), how="left")
VARS_CELDA = [c for c in agg.columns if c.startswith(("osm_", "gee_"))]
corr_celda = pd.DataFrame({
    "Spearman precio mediano": [spearmanr(agg[v], agg["precio_mediano"])[0] for v in VARS_CELDA],
    "Spearman precio/m² mediano": [spearmanr(agg[v], agg["precio_m2_mediano"], nan_policy="omit")[0] for v in VARS_CELDA],
    "Pearson precio/m² mediano": [agg[[v, "precio_m2_mediano"]].dropna().corr().iloc[0, 1] for v in VARS_CELDA],
}, index=VARS_CELDA).sort_values("Spearman precio/m² mediano", key=np.abs, ascending=False)
guardar_tabla(corr_celda.assign(nombre=corr_celda.index.map(es)), "correlaciones_celdas_500m")
log(f"Celdas de 500 m con al menos 5 avisos de entrenamiento: {len(agg):,}")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, v in zip(axes.ravel(), corr_celda.index[:6]):
    xv = agg[v]
    ax.scatter(np.log1p(xv) if v.startswith("osm_") else xv, agg["precio_m2_mediano"], s=8, alpha=0.5, color="#4C72B0")
    ax.set(xlabel=f"log(1 + {es(v)})" if v.startswith("osm_") else es(v), ylabel="Precio por m² mediano (millones)",
           title=f"{es(v)}\nρ = {corr_celda.loc[v, 'Spearman precio/m² mediano']:.2f}")
fig.suptitle("Celdas de 500 m: las variables más correlacionadas con el precio por m²", fontweight="bold")
fig.tight_layout()
guardar_figura(fig, "dispersion_celdas_500m")


# ------------------------------------------------------------------------------------------------------------------
# 7. Autocorrelación espacial: I de Moran, correlograma y LISA
# ------------------------------------------------------------------------------------------------------------------
def moran_variable(d, var, k=10):
    u = d.groupby(["x", "y"], as_index=False)[var].mean().dropna()
    w = KNN.from_array(u[["x", "y"]].to_numpy(), k=k)
    w.transform = "r"
    mi = Moran(u[var].to_numpy(), w, permutations=199)
    return {"I de Moran": mi.I, "p-valor": mi.p_sim, "ubicaciones": len(u)}


moran = {v: moran_variable(tr, v) for v in ["precio_mill", "log_precio", "precio_m2_mill"]}
for v in ["surface_covered", "bedrooms", "bathrooms", "es_apartamento"] + GEE:
    moran[v] = moran_variable(gdf, v)
moran = pd.DataFrame(moran).T.sort_values("I de Moran", ascending=False)
guardar_tabla(moran.assign(nombre=moran.index.map(es)), "moran_global")


def correlograma(d, var, bordes_km=(0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 15), n=4000):
    u = d.groupby(["x", "y"], as_index=False)[var].mean().dropna()
    u = u.sample(min(n, len(u)), random_state=SEMILLA)
    z = ((u[var] - u[var].mean()) / u[var].std()).to_numpy()
    xy = u[["x", "y"]].to_numpy()
    i, j = np.triu_indices(len(z), 1)
    dist = np.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1]) / 1000
    prod = z[i] * z[j]
    filas_ = []
    for a, b in zip(bordes_km[:-1], bordes_km[1:]):
        m = (dist > a) & (dist <= b)
        filas_.append({"desde_km": a, "hasta_km": b, "centro_km": (a + b) / 2, "I": prod[m].mean(), "pares": int(m.sum())})
    return pd.DataFrame(filas_)


cg = {v: correlograma(tr, v) for v in ["log_precio", "precio_m2_mill"]}
guardar_tabla(pd.concat({k: v.set_index("centro_km")[["I", "pares"]] for k, v in cg.items()}, axis=1), "correlograma_precio")
fig, ax = plt.subplots(figsize=(9, 4.5))
for v, c, nombre in [("log_precio", "#4C72B0", "log-precio"), ("precio_m2_mill", "#8172B3", "precio por m²")]:
    ax.plot(cg[v]["centro_km"], cg[v]["I"], "o-", color=c, label=nombre)
ax.axhline(0, color="black", lw=0.8)
ax.set(title="Correlograma espacial: la correlación de precios cae con la distancia",
       xlabel="Distancia entre ubicaciones (km)", ylabel="Correlación (I de Moran por banda)")
ax.legend(frameon=False)
guardar_figura(fig, "correlograma_precio")

u = tr.groupby(["x", "y"], as_index=False)["precio_m2_mill"].mean().dropna()
w = KNN.from_array(u[["x", "y"]].to_numpy(), k=10); w.transform = "r"
lisa = Moran_Local(np.log(u["precio_m2_mill"].to_numpy()), w, permutations=199, seed=SEMILLA)
etiquetas = {1: "Alto-Alto", 2: "Bajo-Alto", 3: "Bajo-Bajo", 4: "Alto-Bajo"}
u["lisa"] = np.where(lisa.p_sim < 0.05, pd.Series(lisa.q).map(etiquetas).values, "No significativo")
colores = {"Alto-Alto": "#d7191c", "Bajo-Bajo": "#2c7bb6", "Alto-Bajo": "#fdae61", "Bajo-Alto": "#abd9e9",
           "No significativo": "#dddddd"}
fig, ax = plt.subplots(figsize=(8, 10))
for cat_, col in colores.items():
    s = u[u["lisa"] == cat_]
    ax.scatter(s["x"], s["y"], s=3 if cat_ != "No significativo" else 1, color=col, label=f"{cat_} (n={len(s):,})")
mapa_base(ax)
ax.legend(markerscale=5, loc="lower right", frameon=True, title="LISA (5 %)")
ax.set_title("Clústeres locales del precio por m² (entrenamiento, ubicaciones únicas)")
guardar_figura(fig, "lisa_precio_m2")


# ------------------------------------------------------------------------------------------------------------------
# 8. Precisión de las coordenadas: malla de unos 110 m
# ------------------------------------------------------------------------------------------------------------------
def n_decimales(v):
    return len(f"{v:.10f}".rstrip("0").split(".")[1])


gdf["decimales"] = np.maximum(gdf["lat"].map(n_decimales), gdf["lon"].map(n_decimales))
gdf["coord_redondeada"] = gdf["decimales"] <= 3
resumen_prec = gdf.groupby("coord_redondeada").agg(
    avisos=("property_id", "size"),
    coordenadas_unicas=("x", lambda s: gdf.loc[s.index].groupby(["x", "y"]).ngroups),
    pct_prueba=("muestra", lambda s: 100 * (s == "test").mean()),
    pct_chapinero=("zona", lambda s: 100 * (s == "Chapinero").mean()),
    precio_mediano_entrenamiento=("precio_mill", "median"))
resumen_prec["avisos_por_coordenada"] = resumen_prec["avisos"] / resumen_prec["coordenadas_unicas"]
resumen_prec.index = resumen_prec.index.map({True: "Redondeada (3 decimales o menos)", False: "Precisa (más de 3 decimales)"})
guardar_tabla(resumen_prec.rename_axis("tipo_de_coordenada"), "precision_coordenadas")

zoom = gdf[gdf["zona"] == "Chapinero"].total_bounds + np.array([-800, -500, 800, 500])
fig, ax = plt.subplots(figsize=(8, 10))
u = gdf.drop_duplicates(["x", "y"])
for redondeada, col, et in [(False, "#4C72B0", "Precisa (más de 3 decimales)"),
                            (True, "#C44E52", "Redondeada a 0,001° (malla de unos 110 m)")]:
    s = u[u["coord_redondeada"] == redondeada]
    ax.scatter(s["x"], s["y"], s=4, color=col, label=f"{et}: {len(s):,} coordenadas")
mapa_base(ax, zoom)
ax.legend(loc="lower left", frameon=True, markerscale=3, fontsize=8)
ax.set_title("Precisión de las coordenadas en Chapinero y alrededores")
guardar_figura(fig, "precision_coordenadas")
log("Fin del análisis espacial descriptivo")
