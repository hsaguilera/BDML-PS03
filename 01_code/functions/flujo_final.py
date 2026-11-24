"""Base del modelo final: 161 variables por aviso, pesos adversariales, puntaje de texto y pliegues de validación.

Es la parte de datos del código que produjo los envíos a Kaggle (pronostico_rapido.py hasta la marca "FIN DATOS" y la
preparación de cv_clase.py en la carpeta original), con las rutas de la plantilla del curso. Se quitaron solo piezas que
no llegan al modelo final y no cambian ninguna de sus cifras: la validación V3 del paso M4, el Elastic Net de texto de
Gentzkow, el puntaje de BETO ajustado y la ablación de escenarios.

Al importarlo se construye la base (unos 30 segundos). Lo usan los scripts 07 a 11:

    from functions import flujo_final as F
    F.g, F.y, F.idx_tr, F.idx_te, F.COLUMNAS["con"], F.puntajes(...), F.pliegues_clase(...), F.oof_clase(...)

Insumos (00_data/ y 00_data/processed/): train.csv y test.csv de Kaggle; gee_features.parquet y localidades_bogota.geojson
(script 01); osm/amenidades_osm.parquet (01); variables_externas_extra.parquet (03); variables_catastro.parquet (04);
texto/texto_variables.parquet, texto/tfidf.npz y texto/dtm_filas.parquet (06); texto/beto_componentes.parquet (06, GPU).
"""
import warnings

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

from functions import modelo_utils as mu
from functions import validacion_clase as vc
from functions.entorno_ps3 import (ARCHIVO_AMENIDADES, ARCHIVO_CATASTRO, ARCHIVO_EXTERNAS_EXTRA, ARCHIVO_GEE,
                                   ARCHIVO_LOCALIDADES, CENTRO, CRS_GEO, CRS_M, DIR_CACHE_TEXTO, DIR_RAW, DIR_TEXTO,
                                   exigir, log)

warnings.filterwarnings("ignore")
SEMILLA = 2026
VECINAS = ["Usaquén", "Barrios Unidos", "Teusaquillo", "Santa Fe"]
NOMBRE = {"lightgbm": "LightGBM", "catboost": "CatBoost", "nn_atencion": "Red con atención", "nn": "Red densa 256-128",
          "ols": "Regresión lineal", "elastic_net": "Elastic Net", "cart": "CART", "random_forest": "Random Forest"}

exigir(DIR_RAW / "train.csv", DIR_RAW / "test.csv", ARCHIVO_GEE, ARCHIVO_LOCALIDADES, ARCHIVO_AMENIDADES,
       ARCHIVO_EXTERNAS_EXTRA, ARCHIVO_CATASTRO, DIR_TEXTO / "texto_variables.parquet", DIR_TEXTO / "tfidf.npz",
       DIR_TEXTO / "dtm_filas.parquet", DIR_TEXTO / "beto_componentes.parquet",
       como="Corra antes los scripts 01 a 06 (01_code/00_rundirectory.R los corre en orden).")

# ------------------------------------------------------------------------------------------------------------------
# 1. Base: Kaggle + Earth Engine + texto + BETO + localidades
# ------------------------------------------------------------------------------------------------------------------
kag = pd.concat([pd.read_csv(DIR_RAW / "train.csv").assign(muestra="train"),
                 pd.read_csv(DIR_RAW / "test.csv").assign(muestra="test")], ignore_index=True)
gee = pd.read_parquet(ARCHIVO_GEE)
GEE = [c for c in gee.columns if c not in ("lat", "lon")]
texto = pd.read_parquet(DIR_TEXTO / "texto_variables.parquet")
beto = pd.read_parquet(DIR_TEXTO / "beto_componentes.parquet")
base = kag.merge(gee, on=["lat", "lon"], how="left").merge(texto, on="property_id", how="left").merge(beto, on="property_id", how="left")
g = gpd.GeoDataFrame(base, geometry=gpd.points_from_xy(base["lon"], base["lat"]), crs=CRS_GEO).to_crs(CRS_M)
loc = gpd.read_file(ARCHIVO_LOCALIDADES).to_crs(CRS_M)
g = g.sjoin(loc[["localidad", "geometry"]], how="left", predicate="within").drop(columns="index_right")
g = g[~g.index.duplicated()].copy()
g["localidad"] = g["localidad"].fillna("Fuera de límites")
g["x"], g["y"] = g.geometry.x, g.geometry.y

g["es_apartamento"] = (g["property_type"] == "Apartamento").astype(int)
g["tiempo"] = g["year"] + (g["month"] - 1) / 12
g["log_area"] = np.log(g["area_imp"])
g["area_desde_texto"] = (g["area_fuente"] == "texto").astype(int)
g["area_falta"] = (g["area_fuente"] == "faltante").astype(int)
g["banos_desde_texto"] = (g["banos_fuente"] == "texto").astype(int)
g["area_inconsistente"] = (g["surface_covered"] > g["surface_total"]).astype(int)
ESTRUCTURALES = ["es_apartamento", "bedrooms", "rooms", "banos_imp", "area_imp", "log_area", "parqueaderos_imp",
                 "piso_texto", "estrato_texto", "antiguedad_texto", "tiempo", "area_desde_texto", "area_falta",
                 "banos_desde_texto", "area_inconsistente"]

# ------------------------------------------------------------------------------------------------------------------
# 2. Variables externas desde OSM: distancias y conteos como en la clase de datos espaciales
# ------------------------------------------------------------------------------------------------------------------
osm = gpd.read_parquet(ARCHIVO_AMENIDADES).to_crs(CRS_M)
ubic = g.drop_duplicates(["x", "y"])[["x", "y", "geometry"]].reset_index(drop=True)


def distancia(capa):
    """Distancia (m) de cada ubicación única al objeto más cercano de la capa (st_nearest_feature + st_distance)."""
    j = gpd.sjoin_nearest(ubic[["geometry"]], capa[["geometry"]], how="left", distance_col="d")
    return j[~j.index.duplicated()]["d"].values


def conteo(capa, r):
    """Número de objetos de la capa a <= r metros (st_is_within_distance + lengths)."""
    p = capa.geometry.representative_point()
    return cKDTree(np.c_[p.x, p.y]).query_ball_point(np.c_[ubic["x"], ubic["y"]], r=r, return_length=True)


def categoria(nombre):
    return osm[osm["categoria"] == nombre]


parques = categoria("parque")[lambda d: d.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
parques["area_parque_m2"] = parques.area
j = gpd.sjoin_nearest(ubic[["geometry"]], parques[["geometry", "area_parque_m2"]], how="left", distance_col="d")
j = j[~j.index.duplicated()]
ubic["dist_parque_m"], ubic["area_parque_m2"] = j["d"].values, j["area_parque_m2"].values
ubic["n_restaurantes_2km"] = conteo(categoria("restaurante"), 2000)
ubic["dist_parque_grande_m"] = distancia(parques[parques["area_parque_m2"] >= 10_000])
ubic["n_restaurantes_500m"] = conteo(categoria("restaurante"), 500)
ubic["n_bares_cafes_500m"] = conteo(categoria("bar_cafe"), 500)
est = categoria("estacion")
red = (est["network"].fillna("") + " " + est["operator"].fillna("") + " " + est["nombre"].fillna("")).str.lower()
ubic["dist_transmilenio_m"] = distancia(est[~red.str.contains("metro|regiotram|ferrocarril|sitp|terminal|invias|turistren|parada")])
ubic["dist_via_principal_m"] = distancia(categoria("via_principal"))
ubic["dist_universidad_m"] = distancia(categoria("universidad"))
ubic["n_colegios_1km"] = conteo(categoria("colegio"), 1000)
ubic["dist_centro_comercial_m"] = distancia(categoria("centro_comercial"))
ubic["n_supermercados_1km"] = conteo(categoria("supermercado"), 1000)
ubic["dist_hospital_m"] = distancia(categoria("hospital"))
ubic["dist_policia_m"] = distancia(categoria("policia"))
ubic["n_paraderos_500m"] = conteo(categoria("paradero"), 500)
for nombre, (lon, lat) in {"p93": (-74.0485, 4.6766), "zonat": (-74.0537, 4.6669),
                           "centro_internacional": (-74.0700, 4.6150)}.items():
    punto = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs=CRS_GEO).to_crs(CRS_M).iloc[0]
    ubic[f"dist_{nombre}_m"] = ubic.geometry.distance(punto)
OSM_VARS = [c for c in ubic.columns if c.startswith(("dist_", "n_", "area_parque"))]
g = g.merge(pd.DataFrame(ubic.drop(columns="geometry")), on=["x", "y"], how="left")
g.index = range(len(g))
for v in OSM_VARS + ["poblacion_100m"]:
    g[f"ln_{v}"] = np.log1p(g[v])
EXTERNAS = [f"ln_{v}" for v in OSM_VARS] + [v for v in GEE if v != "poblacion_100m"] + ["ln_poblacion_100m"]

# Variables externas adicionales (script 03): oficinas, hoteles, bancos, gimnasios, embajadas, conteos a 300 m y 1 km,
# TransMilenio a 1 km, área de parques a 500 m, distancia a los cerros, elevación relativa, entorno de avisos (sin precio)
# y bandera de coordenada redondeada.
extra = pd.read_parquet(ARCHIVO_EXTERNAS_EXTRA)
g = g.merge(extra, on="property_id", how="left")
g.index = range(len(g))
EXTRA = []
for v in [c for c in extra.columns if c != "property_id"]:
    if v.startswith(("n_", "dist_", "area_")):
        g[f"ln_{v}"] = np.log1p(g[v])
        EXTRA.append(f"ln_{v}")
    else:
        EXTRA.append(v)
TEXTO = [c for c in texto.columns if c.startswith(("txt_", "topico_", "lsa_"))] + \
        ["alcobas_texto", "banos_texto", "parqueaderos_texto", "area_texto"]
BETO = [c for c in beto.columns if c.startswith("beto_pc_")]
g["x_km"], g["y_km"] = g["x"] / 1000, g["y"] / 1000
COORD = ["x_km", "y_km"]

# ------------------------------------------------------------------------------------------------------------------
# 3. Objetivo, índices, pliegue de Chapinero y pesos adversariales (solo para el MAE ponderado)
# ------------------------------------------------------------------------------------------------------------------
es_tr = (g["muestra"] == "train").values
y = g["price"].to_numpy() / 1e6                    # precio en millones (NaN en prueba)
lp = np.log(y)
lpm2 = np.log(y / g["area_imp"].where(g["area_imp"].between(15, 1000)).to_numpy())
xs, ys = g["x"].to_numpy(), g["y"].to_numpy()
grupo = pd.factorize(g["grupo_texto"])[0]          # avisos con el mismo texto normalizado
idx_tr, idx_te = np.where(es_tr)[0], np.where(~es_tr)[0]
PLIEGUES = {z: (np.setdiff1d(idx_tr, np.where(g["localidad"].values == z)[0]),
                np.intersect1d(idx_tr, np.where(g["localidad"].values == z)[0])) for z in ["Chapinero"] + VECINAS}

COMPOS = ESTRUCTURALES + [c for c in TEXTO if c.startswith(("txt_", "topico_"))] + BETO
Xc = g[COMPOS].to_numpy(float)
zc = (~es_tr).astype(int)
p_adv = np.zeros(len(g))
for a, b in KFold(5, shuffle=True, random_state=SEMILLA).split(Xc):
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=SEMILLA, verbose=-1, n_jobs=8).fit(Xc[a], zc[a])
    p_adv[b] = clf.predict_proba(Xc[b])[:, 1]
pt = np.clip(p_adv[idx_tr], 1e-4, 1 - 1e-4)
w = pt / (1 - pt) * (len(idx_tr) / len(idx_te))
w = np.minimum(w, np.quantile(w, 0.99))
PESOS = np.zeros(len(g))
PESOS[idx_tr] = w / w.mean()
AUC_ADVERSARIAL = roc_auc_score(zc, p_adv)

# ------------------------------------------------------------------------------------------------------------------
# 4. Puntaje de texto supervisado fuera de pliegue (Gentzkow et al. 2019, ec. 2): ridge de ln(precio) sobre TF-IDF.
#    Filas de entrenamiento: predicción fuera de pliegue con GroupKFold por grupo de texto (los duplicados nunca se
#    predicen a sí mismos). Filas de validación o prueba: ajuste con todo el entrenamiento del pliegue.
# ------------------------------------------------------------------------------------------------------------------
X_tfidf = sp.load_npz(DIR_TEXTO / "tfidf.npz").tocsr()
filas_dtm = pd.read_parquet(DIR_TEXTO / "dtm_filas.parquet")["property_id"].values
orden_dtm = pd.Series(np.arange(len(filas_dtm)), index=filas_dtm).loc[g["property_id"].values].values
X_tfidf = X_tfidf[orden_dtm]


def puntaje_texto(idx_fit, idx_pred, alpha=1.0):
    out = np.full(len(g), np.nan)
    for a, b in GroupKFold(5).split(idx_fit, groups=grupo[idx_fit]):
        m = Ridge(alpha=alpha).fit(X_tfidf[idx_fit[a]], lp[idx_fit[a]])
        out[idx_fit[b]] = m.predict(X_tfidf[idx_fit[b]])
    out[idx_pred] = Ridge(alpha=alpha).fit(X_tfidf[idx_fit], lp[idx_fit]).predict(X_tfidf[idx_pred])
    return out


_CACHE = {}


def puntajes(zona, a, b, recalcular=False):
    """Puntaje ridge del pliegue `zona` (columna n × 1). Los pliegues son deterministas, así que el puntaje se guarda en
    00_data/processed/cache_puntajes_texto/ y se reutiliza; con recalcular=True se vuelve a calcular."""
    if zona not in _CACHE:
        archivo = DIR_CACHE_TEXTO / f"ridge_{zona}.npy"
        if archivo.exists() and not recalcular:
            _CACHE[zona] = np.load(archivo)
        else:
            _CACHE[zona] = puntaje_texto(a, b)
            np.save(archivo, _CACHE[zona])
    return _CACHE[zona][:, None]


H_BASE = g[["log_area", "bedrooms", "banos_imp", "es_apartamento"]].to_numpy(float)   # para la red con atención

# ------------------------------------------------------------------------------------------------------------------
# 5. Variables de Catastro y conjuntos de columnas
# ------------------------------------------------------------------------------------------------------------------
cat_ = pd.read_parquet(ARCHIVO_CATASTRO)
_orden = g["property_id"].values.copy()
g = g.merge(cat_, on="property_id", how="left")
assert (g["property_id"].values == _orden).all()
g.index = range(len(g))
CATASTRO = [c for c in cat_.columns if c != "property_id"]
SIN = ESTRUCTURALES + EXTERNAS + EXTRA + TEXTO + COORD
COLUMNAS = {"sin": SIN, "con": SIN + CATASTRO}
log(f"Base: {len(g):,} avisos ({len(idx_tr):,} de entrenamiento y {len(idx_te):,} de prueba) | variables: estructurales "
    f"{len(ESTRUCTURALES)}, externas {len(EXTERNAS)} + adicionales {len(EXTRA)}, texto {len(TEXTO)}, coordenadas "
    f"{len(COORD)}, Catastro {len(CATASTRO)} (+ puntaje de texto) | AUC adversarial {AUC_ADVERSARIAL:.3f}")


# ------------------------------------------------------------------------------------------------------------------
# 6. Pliegues de validación (validacion_clase.py) y predicciones fuera de pliegue
# ------------------------------------------------------------------------------------------------------------------
def pliegues_clase(esquemas=("aleatoria", "bloques", "bloques_buffer", "clusters")):
    """{nombre: (esquema, idx_entrenamiento, idx_validación, partición interna)} para los esquemas pedidos y el pliegue
    "Chapinero (imitando el taller)", que se entrena sin Chapinero y predice sus 307 avisos de entrenamiento."""
    loc_tr = g["localidad"].values[idx_tr]
    pl = {}
    for e in esquemas:
        for k, (a, b) in enumerate(vc.pliegues_externos(e, xs[idx_tr], ys[idx_tr], loc_tr)):
            ia, ib = idx_tr[a], idx_tr[b]
            pl[f"{e}-{k}"] = (e, ia, ib, vc.particion_interna(e, xs[ia], ys[ia], g["localidad"].values[ia], SEMILLA + k))
    a, b = PLIEGUES["Chapinero"]
    pl["Chapinero"] = ("Chapinero (imitando el taller)", a, b,
                       vc.particion_interna("localidades", xs[a], ys[a], g["localidad"].values[a], SEMILLA))
    return pl


def params_de(fam, part=None):
    p = {} if part is None else {"particion_interna": part}
    if fam == "nn_atencion":
        p["H_base"] = H_BASE
    return p


def oof_clase(fam, conj, pliegues, n_jobs=6, recalcular_puntajes=False):
    """Predicciones fuera de pliegue de la familia `fam` con el conjunto de variables `conj` ("sin" o "con" Catastro).
    En cada pliegue se recalcula el puntaje de texto solo con su entrenamiento."""
    X = g[COLUMNAS[conj]].to_numpy(float)
    tareas = (delayed(mu.ajustar_predecir)(fam, np.c_[X, puntajes(n, a, b, recalcular_puntajes)], y, xs, ys, lpm2, lp,
                                           grupo, None, a, b, False, False, params_de(fam, part), SEMILLA)
              for n, (e, a, b, part) in pliegues.items())
    preds = Parallel(n_jobs=n_jobs)(tareas)
    return pd.concat([pd.DataFrame({"esquema": e, "pliegue": n, "idx": b, "pred": pr, "real": y[b], "peso": PESOS[b]})
                      for (n, (e, a, b, part)), pr in zip(pliegues.items(), preds)], ignore_index=True)


def metricas(o, pred=None):
    """MAE, RMSE y sesgo por esquema sobre las predicciones reunidas (como en la lectura de validación espacial)."""
    pred = o["pred"].to_numpy() if pred is None else np.asarray(pred)
    filas = []
    for e in list(dict.fromkeys(o["esquema"])):
        m = (o["esquema"] == e).values
        err = pred[m] - o["real"].values[m]
        fila = {"esquema": e, "n": int(m.sum()), "MAE": np.abs(err).mean(), "RMSE": np.sqrt((err ** 2).mean()),
                "Sesgo": err.mean()}
        if e.startswith("Chapinero"):
            fila["MAE ponderado"] = np.average(np.abs(err), weights=o["peso"].values[m])
        filas.append(fila)
    return filas


# ------------------------------------------------------------------------------------------------------------------
# 7. Posproceso de los envíos: ajuste ×1,08 y gemelos de texto
# ------------------------------------------------------------------------------------------------------------------
def estirar(pred, factor):
    """654 + b (ŷ − 654): corrige la compresión de las predicciones hacia la media."""
    return CENTRO + factor * (np.asarray(pred) - CENTRO)


_gem = pd.Series(y[idx_tr], index=grupo[idx_tr]).groupby(level=0).median()
PRECIO_GEMELO = pd.Series(grupo[idx_te]).map(_gem).to_numpy()   # mediana del precio de los avisos con el mismo texto
TIENE_GEMELO = np.isfinite(PRECIO_GEMELO)


def envio_kaggle(pred):
    """Archivo de Kaggle: gemelos de texto (0,3 modelo + 0,7 precio del gemelo) y precio en pesos redondeado."""
    pred = np.array(pred, dtype=float)
    pred[TIENE_GEMELO] = 0.3 * pred[TIENE_GEMELO] + 0.7 * PRECIO_GEMELO[TIENE_GEMELO]
    return pd.DataFrame({"property_id": g["property_id"].values[idx_te], "price": np.round(pred * 1e6, 0)})
