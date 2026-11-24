"""01 · Descarga de las fuentes externas: OpenStreetMap, límites de las localidades y Google Earth Engine.

Qué hace
    1. OpenStreetMap: descarga el extracto de Bogotá de BBBike (Bogota.osm.pbf) y extrae, con pyosmium, los objetos de
       12 categorías de la wiki Map_features (parques, restaurantes, bares y cafés, universidades, colegios, centros
       comerciales, supermercados, hospitales, policía, estaciones de transporte, paraderos y vías principales).
    2. Localidades: polígonos de las 19 localidades urbanas de Bogotá desde Nominatim (osmnx.geocode_to_gdf).
    3. Earth Engine: valor de 6 capas satelitales en cada coordenada única de los avisos (elevación y pendiente SRTM,
       NDVI de Sentinel-2 en 200 m, luces nocturnas VIIRS, altura construida y población de GHSL).

Salidas (00_data/processed/)
    osm/amenidades_osm.parquet   objetos de OSM: categoría, tipo de geometría, nombre, red, operador y geometría
    localidades_bogota.geojson   localidad y polígono (EPSG:4326)
    gee_features.parquet         lat, lon y las 6 capas por coordenada única
    osm/Bogota.osm.pbf           extracto de OSM (solo al recalcular; no se versiona)

Recálculo
    Por defecto no usa internet: verifica que las tres salidas existan (vienen en el repositorio) y resume su contenido.
    Con PS3_RECALCULAR=1 (o =01) las vuelve a descargar. Requiere internet y, para Earth Engine, credenciales propias
    (`earthengine authenticate`) y un proyecto de Google Cloud registrado en Earth Engine, que se indica con la variable
    PS3_GEE_PROYECTO (sin valor por defecto: el ID de su proyecto de Google Cloud registrado en Earth Engine). No se guardan credenciales
    en el repositorio. Aviso: BBBike actualiza el extracto cada semana y Nominatim puede cambiar los polígonos, así que
    una descarga nueva puede diferir levemente de la usada en los envíos (extracto descargado el 23-nov-2026).
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import os
import urllib.request

import geopandas as gpd
import pandas as pd

from functions.entorno_ps3 import (ARCHIVO_AMENIDADES, ARCHIVO_GEE, ARCHIVO_LOCALIDADES, ARCHIVO_PBF, CRS_GEO, DIR_RAW,
                                   exigir, existen, log, recalcular)

RECALCULAR = recalcular("01")
URL_PBF = "https://download.bbbike.org/osm/bbbike/Bogota/Bogota.osm.pbf"
LOCALIDADES = ["Usaquén", "Chapinero", "Santa Fe", "San Cristóbal", "Usme", "Tunjuelito", "Bosa", "Kennedy", "Fontibón",
               "Engativá", "Suba", "Barrios Unidos", "Teusaquillo", "Los Mártires", "Antonio Nariño", "Puente Aranda",
               "La Candelaria", "Rafael Uribe Uribe", "Ciudad Bolívar"]
CAPAS_GEE = ["elevacion_m", "pendiente_grados", "ndvi_200m", "luces_nocturnas", "altura_construida_m", "poblacion_100m"]

# Ubicaciones únicas de los avisos (entrenamiento y prueba): muchos avisos comparten coordenadas
avisos = pd.concat([pd.read_csv(DIR_RAW / "train.csv"), pd.read_csv(DIR_RAW / "test.csv")], ignore_index=True)
ubic = avisos.drop_duplicates(["lat", "lon"])[["lat", "lon"]].reset_index(drop=True)
log(f"{len(avisos):,} avisos | {len(ubic):,} coordenadas únicas")


# ------------------------------------------------------------------------------------------------------------------
# 1. OpenStreetMap
# ------------------------------------------------------------------------------------------------------------------
def descargar_osm():
    import osmium
    import shapely.wkb

    if not ARCHIVO_PBF.exists() or RECALCULAR:
        urllib.request.urlretrieve(URL_PBF, ARCHIVO_PBF)
    log(f"{ARCHIVO_PBF.name}: {ARCHIVO_PBF.stat().st_size / 1e6:.1f} MB")

    filtros = {
        "parque": lambda t: t.get("leisure") == "park",
        "restaurante": lambda t: t.get("amenity") == "restaurant",
        "bar_cafe": lambda t: t.get("amenity") in {"bar", "pub", "cafe", "nightclub"},
        "universidad": lambda t: t.get("amenity") == "university",
        "colegio": lambda t: t.get("amenity") == "school",
        "centro_comercial": lambda t: t.get("shop") == "mall",
        "supermercado": lambda t: t.get("shop") == "supermarket",
        "hospital": lambda t: t.get("amenity") in {"hospital", "clinic"},
        "policia": lambda t: t.get("amenity") == "police",
        "estacion": lambda t: t.get("public_transport") == "station" or t.get("amenity") == "bus_station",
        "paradero": lambda t: t.get("highway") == "bus_stop",
    }
    vias_principales = {"motorway", "trunk", "primary"}

    class Recolector(osmium.SimpleHandler):
        """Guarda (categoría, tipo, name, network, operator, geometría WKB) de los objetos que cumplen los filtros."""

        def __init__(self):
            super().__init__()
            self.fab = osmium.geom.WKBFactory()
            self.filas = []

        def _guardar(self, tags, wkb, tipo):
            t = {k: v for k, v in tags}
            for cat, f in filtros.items():
                if f(t):
                    self.filas.append((cat, tipo, t.get("name"), t.get("network"), t.get("operator"), wkb))

        def node(self, n):
            if len(n.tags) and n.location.valid():
                if any(k in n.tags for k in ("leisure", "amenity", "shop", "public_transport", "highway")):
                    self._guardar(n.tags, self.fab.create_point(n), "punto")

        def way(self, w):
            if w.tags.get("highway") in vias_principales:
                try:
                    self.filas.append(("via_principal", "linea", w.tags.get("name"), None, None,
                                       self.fab.create_linestring(w)))
                except Exception:          # vía sin geometría válida en el extracto: se descarta
                    pass

        def area(self, a):
            if any(k in a.tags for k in ("leisure", "amenity", "shop", "public_transport")):
                try:
                    self._guardar(a.tags, self.fab.create_multipolygon(a), "area")
                except Exception:          # polígono roto: se descarta (equivale a validar geometrías en clase)
                    pass

    rec = Recolector()
    rec.apply_file(str(ARCHIVO_PBF), locations=True)
    osm = pd.DataFrame(rec.filas, columns=["categoria", "tipo", "nombre", "network", "operator", "wkb"])
    osm = gpd.GeoDataFrame(osm.drop(columns="wkb"), geometry=[shapely.wkb.loads(b, hex=True) for b in osm["wkb"]],
                           crs=CRS_GEO)
    osm["geometry"] = osm.geometry.make_valid()
    osm = osm[~osm.geometry.is_empty]
    osm.to_parquet(ARCHIVO_AMENIDADES)


# ------------------------------------------------------------------------------------------------------------------
# 2. Localidades (Nominatim). El código original de esta descarga se perdió; esta versión la reconstruye con la misma
#    herramienta (osmnx.geocode_to_gdf) y deja las columnas del archivo original: osm_id, localidad y geometría.
# ------------------------------------------------------------------------------------------------------------------
def descargar_localidades():
    import osmnx as ox

    partes = []
    for nombre in LOCALIDADES:
        d = ox.geocode_to_gdf(f"Localidad {nombre}, Bogotá, Colombia")
        partes.append(d[["osm_id", "geometry"]].assign(localidad=nombre).iloc[:1])
    loc = gpd.GeoDataFrame(pd.concat(partes, ignore_index=True), crs=CRS_GEO)[["osm_id", "localidad", "geometry"]]
    loc.to_file(ARCHIVO_LOCALIDADES, driver="GeoJSON")


# ------------------------------------------------------------------------------------------------------------------
# 3. Google Earth Engine
# ------------------------------------------------------------------------------------------------------------------
def descargar_gee():
    import ee

    # PS3_GEE_PROYECTO: el ID de su proyecto de Google Cloud registrado en Earth Engine (por ejemplo,
    # "mi-proyecto-earth-engine"). Defínalo como variable de entorno o escríbalo aquí, en lugar de "".
    proyecto = os.environ.get("PS3_GEE_PROYECTO", "")
    if not proyecto:
        raise SystemExit("Defina PS3_GEE_PROYECTO con el ID de su proyecto de Google Cloud registrado en Earth Engine.")
    ee.Initialize(project=proyecto)            # usa las credenciales locales de `earthengine authenticate`
    bbox = ee.Geometry.Rectangle([-74.25, 4.45, -73.95, 4.85])
    srtm = ee.Image("USGS/SRTMGL1_003")

    def mascara_nubes(img):
        scl = img.select("SCL")              # 3 = sombra de nube; 8 a 10 = nubes y cirros
        return img.updateMask(scl.neq(3).And(scl.lt(8).Or(scl.gt(10))))

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(bbox)
          .filterDate("2020-01-01", "2021-12-31").filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
          .map(mascara_nubes).median())
    pila = ee.Image.cat([
        srtm.rename("elevacion_m"),
        ee.Terrain.slope(srtm).rename("pendiente_grados"),
        s2.normalizedDifference(["B8", "B4"]).reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.circle(200, "meters"))
          .rename("ndvi_200m"),
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate("2020-01-01", "2020-12-31")
          .select("avg_rad").mean().rename("luces_nocturnas"),
        ee.Image("JRC/GHSL/P2023A/GHS_BUILT_H/2018").select("built_height").rename("altura_construida_m"),
        ee.Image("JRC/GHSL/P2023A/GHS_POP/2020").select("population_count").rename("poblacion_100m"),
    ]).clip(bbox)

    filas, tam = [], 4000                    # lotes para no superar el límite de elementos por consulta
    for i in range(0, len(ubic), tam):
        lote = ubic.iloc[i:i + tam]
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lo, la]), {"i": int(k)})
                                   for k, lo, la in zip(lote.index, lote["lon"], lote["lat"])])
        res = pila.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=10).getInfo()
        filas += [f["properties"] for f in res["features"]]
        log(f"Earth Engine, lote {i // tam + 1}: {len(res['features'])} puntos")
    valores = pd.DataFrame(filas).set_index("i").sort_index()
    ubic.join(valores)[["lat", "lon"] + CAPAS_GEE].to_parquet(ARCHIVO_GEE, index=False)


if RECALCULAR:
    log("PS3_RECALCULAR: se descargan de nuevo OSM, localidades y Earth Engine (requiere internet y credenciales)")
    descargar_osm()
    descargar_localidades()
    descargar_gee()
elif not existen(ARCHIVO_AMENIDADES, ARCHIVO_LOCALIDADES, ARCHIVO_GEE):
    exigir(ARCHIVO_AMENIDADES, ARCHIVO_LOCALIDADES, ARCHIVO_GEE,
           como="Vienen en el repositorio; para descargarlas corra este script con PS3_RECALCULAR=01.")
else:
    log("Se usan las fuentes ya descargadas (para descargarlas de nuevo: PS3_RECALCULAR=01)")

# --- Resumen de lo que queda para los pasos siguientes ------------------------------------------------------------
osm = gpd.read_parquet(ARCHIVO_AMENIDADES)
log(f"OpenStreetMap: {len(osm):,} objetos | " + ", ".join(f"{k} {v:,}" for k, v in osm["categoria"].value_counts().items()))
loc = gpd.read_file(ARCHIVO_LOCALIDADES)
log(f"Localidades: {len(loc)} polígonos")
gee = pd.read_parquet(ARCHIVO_GEE)
assert len(ubic.merge(gee, on=["lat", "lon"])) == len(ubic), "Hay coordenadas sin valores de Earth Engine"
log(f"Earth Engine: {len(gee):,} coordenadas con {gee.shape[1] - 2} capas ({', '.join(CAPAS_GEE)})")
