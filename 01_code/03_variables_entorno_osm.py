"""03 · Variables externas adicionales: entorno de OpenStreetMap, cerros, elevación relativa y entorno de los avisos.

Qué hace (26 variables por aviso, todas en EPSG:9377, en metros)
    1. Lee de nuevo el extracto de OSM (pyosmium) para categorías que el script 01 no extrae: oficinas, edificios de
       oficinas, hoteles, bancos, gimnasios, embajadas y áreas protegidas (reserva de los cerros orientales).
    2. Conteos a 300 m y 1 km (robustos a la malla de unos 110 m de las coordenadas redondeadas), estaciones de
       TransMilenio a 1 km y área de parques a 500 m.
    3. Distancia a las áreas protegidas de 100 ha o más y elevación relativa al entorno (elevación del punto menos la
       mediana de las coordenadas a 1 km o menos; elevación de Earth Engine, script 01).
    4. Entorno de los avisos sin usar el precio (sin fuga): número de avisos a 500 m, proporción de apartamentos y área
       mediana a 500 m.
    5. Bandera de coordenada redondeada a 0,001 grados.
    Las 18 variables de OSM de la clase (distancia al parque y su área, restaurantes a 2 km, TransMilenio, etc.) y las
    distancias a los centros de empleo se calculan en functions/flujo_final.py, a partir de osm/amenidades_osm.parquet.

Salidas (00_data/processed/)
    variables_externas_extra.parquet   property_id y las 26 variables
    osm/amenidades_osm_extra.parquet   objetos nuevos de OSM (categoría, lon, lat); solo al recalcular

Recálculo
    Por defecto usa variables_externas_extra.parquet (viene en el repositorio). Con PS3_RECALCULAR=1 (o =03) la
    reconstruye; necesita osm/Bogota.osm.pbf, que descarga el script 01 al recalcular (no se versiona, 21 MB).
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from functions.entorno_ps3 import (ARCHIVO_AMENIDADES, ARCHIVO_EXTERNAS_EXTRA, ARCHIVO_GEE, ARCHIVO_PBF, CRS_GEO, CRS_M,
                                   DIR_OSM, DIR_RAW, exigir, existen, log, recalcular)

RECALCULAR = recalcular("03")


def clasificar(tags):
    """Categoría nueva de un objeto de OSM según sus etiquetas (None si no interesa)."""
    if "office" in tags:
        return "embajada" if tags.get("office") == "diplomatic" else "oficina"
    if tags.get("amenity") == "embassy":
        return "embajada"
    if tags.get("amenity") == "bank":
        return "banco"
    if tags.get("tourism") == "hotel":
        return "hotel"
    if tags.get("leisure") == "fitness_centre" or tags.get("sport") == "fitness":
        return "gimnasio"
    if tags.get("building") in ("office", "commercial"):
        return "edificio_oficinas"
    return None


def construir():
    import osmium
    import shapely.wkb

    class Lector(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.fab = osmium.geom.WKBFactory()
            self.filas, self.protegidas = [], []

        def node(self, n):
            c = clasificar(n.tags)
            if c and n.location.valid():
                self.filas.append((c, n.location.lon, n.location.lat))

        def area(self, a):
            tags = a.tags
            if tags.get("boundary") == "protected_area" or tags.get("leisure") == "nature_reserve":
                try:
                    self.protegidas.append((tags.get("name", ""), shapely.wkb.loads(self.fab.create_multipolygon(a), hex=True)))
                except Exception:          # polígono roto en el extracto: se descarta
                    pass
                return
            c = clasificar(tags)
            if c:
                try:
                    geom = shapely.wkb.loads(self.fab.create_multipolygon(a), hex=True)
                    p = geom.representative_point()
                    self.filas.append((c, p.x, p.y))
                except Exception:          # polígono roto en el extracto: se descarta
                    pass

    lector = Lector()
    lector.apply_file(str(ARCHIVO_PBF), locations=True)
    extra = gpd.GeoDataFrame(pd.DataFrame(lector.filas, columns=["categoria", "lon", "lat"]),
                             geometry=gpd.points_from_xy([f[1] for f in lector.filas], [f[2] for f in lector.filas]),
                             crs=CRS_GEO).to_crs(CRS_M)
    prot = gpd.GeoDataFrame({"nombre": [p[0] for p in lector.protegidas]}, geometry=[p[1] for p in lector.protegidas],
                            crs=CRS_GEO).to_crs(CRS_M)
    prot = prot[prot.area >= 1_000_000]                    # áreas protegidas de 100 ha o más (cerros, humedales grandes)
    log("Objetos nuevos de OSM: " + ", ".join(f"{k} {v:,}" for k, v in extra["categoria"].value_counts().items())
        + f" | áreas protegidas de 100 ha o más: {len(prot)}")
    extra.drop(columns="geometry").to_parquet(DIR_OSM / "amenidades_osm_extra.parquet", index=False)

    # --- Ubicaciones únicas (entrenamiento y prueba) ---
    kag = pd.concat([pd.read_csv(DIR_RAW / "train.csv"), pd.read_csv(DIR_RAW / "test.csv")], ignore_index=True)
    g = gpd.GeoDataFrame(kag[["property_id", "lat", "lon", "property_type", "surface_covered", "surface_total"]],
                         geometry=gpd.points_from_xy(kag["lon"], kag["lat"]), crs=CRS_GEO).to_crs(CRS_M)
    g["x"], g["y"] = g.geometry.x.round(3), g.geometry.y.round(3)
    ubic = g.drop_duplicates(["x", "y"])[["x", "y", "lat", "lon", "geometry"]].reset_index(drop=True)
    P = np.c_[ubic["x"], ubic["y"]]
    log(f"Ubicaciones únicas: {len(ubic):,}")

    def conteo(xy, r):
        return cKDTree(xy).query_ball_point(P, r=r, return_length=True)

    def xy_de(capa):
        p = capa.geometry if (capa.geom_type == "Point").all() else capa.geometry.representative_point()
        return np.c_[p.x, p.y]

    osm = gpd.read_parquet(ARCHIVO_AMENIDADES).to_crs(CRS_M)

    def categoria(nombre):
        return osm[osm["categoria"] == nombre]

    for c, radios in {"oficina": (500, 1000), "edificio_oficinas": (500,), "hotel": (1000,), "banco": (500,),
                      "gimnasio": (1000,), "embajada": (1000,)}.items():
        xy = np.c_[extra.loc[extra["categoria"] == c].geometry.x, extra.loc[extra["categoria"] == c].geometry.y]
        for r in radios:
            ubic[f"n_{c}_{r}m"] = conteo(xy, r) if len(xy) else 0
    emb = extra[extra["categoria"] == "embajada"]
    ubic["dist_embajada_m"] = cKDTree(np.c_[emb.geometry.x, emb.geometry.y]).query(P)[0]
    for c in ("restaurante", "bar_cafe", "supermercado", "paradero", "colegio"):
        xy = xy_de(categoria(c))
        ubic[f"n_{c}_300m"] = conteo(xy, 300)
        ubic[f"n_{c}_1km"] = conteo(xy, 1000)
    est = categoria("estacion")
    red = (est["network"].fillna("") + " " + est["operator"].fillna("") + " " + est["nombre"].fillna("")).str.lower()
    tm = est[~red.str.contains("metro|regiotram|ferrocarril|sitp|terminal|invias|turistren|parada")]
    ubic["n_transmilenio_1km"] = conteo(xy_de(tm), 1000)
    parques = categoria("parque")[lambda d: d.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    parques["area"] = parques.area
    vec = cKDTree(xy_de(parques)).query_ball_point(P, r=500)
    ubic["area_parques_500m_ha"] = [parques["area"].values[v].sum() / 1e4 for v in vec]
    ubic["dist_area_protegida_m"] = ubic.geometry.distance(prot.union_all()).values

    # Elevación relativa al entorno (capa de Earth Engine en las coordenadas de los avisos)
    gee = pd.read_parquet(ARCHIVO_GEE)
    ubic = ubic.merge(gee[["lat", "lon", "elevacion_m"]], on=["lat", "lon"], how="left")
    vec = cKDTree(P).query_ball_point(P, r=1000)
    elev = ubic["elevacion_m"].to_numpy()
    ubic["elevacion_relativa_1km"] = elev - np.array([np.nanmedian(elev[v]) for v in vec])
    ubic = ubic.drop(columns="elevacion_m")

    # Entorno de avisos (sin precio): densidad, proporción de apartamentos y área mediana a 500 m
    xy_av = np.c_[g["x"], g["y"]]
    es_apto = (g["property_type"] == "Apartamento").to_numpy(float)
    area = g["surface_covered"].fillna(g["surface_total"]).to_numpy(float)
    vec = cKDTree(xy_av).query_ball_point(P, r=500)
    ubic["n_avisos_500m"] = [len(v) for v in vec]
    ubic["prop_apartamentos_500m"] = [es_apto[v].mean() for v in vec]
    ubic["area_mediana_500m"] = [np.nanmedian(area[v]) if np.isfinite(area[v]).any() else np.nan for v in vec]

    out = g[["property_id", "x", "y", "lat", "lon"]].merge(ubic.drop(columns=["geometry", "lat", "lon"]), on=["x", "y"],
                                                         how="left")
    entero = lambda v: np.isclose((v * 1000) % 1, 0, atol=1e-6) | np.isclose((v * 1000) % 1, 1, atol=1e-6)
    out["coord_redondeada"] = (entero(out["lat"]) & entero(out["lon"])).astype(int)
    out = out.drop(columns=["x", "y", "lat", "lon"])
    out.to_parquet(ARCHIVO_EXTERNAS_EXTRA, index=False)


if RECALCULAR:
    exigir(ARCHIVO_PBF, ARCHIVO_AMENIDADES, ARCHIVO_GEE,
           como="Descargue el extracto de OSM con PS3_RECALCULAR=01 (script 01) antes de recalcular este paso.")
    construir()
elif not existen(ARCHIVO_EXTERNAS_EXTRA):
    if existen(ARCHIVO_PBF):
        construir()
    else:
        exigir(ARCHIVO_EXTERNAS_EXTRA, como="Viene en el repositorio; para reconstruirla corra 01 y 03 con PS3_RECALCULAR=01,03.")
else:
    log("Se usa la base guardada (para reconstruirla: PS3_RECALCULAR=03, con osm/Bogota.osm.pbf descargado)")

out = pd.read_parquet(ARCHIVO_EXTERNAS_EXTRA)
log(f"{ARCHIVO_EXTERNAS_EXTRA.as_posix()}: {len(out):,} avisos y {out.shape[1] - 1} variables")
print(out.drop(columns="property_id").describe().T[["mean", "50%", "max"]].round(2).to_string(), flush=True)
