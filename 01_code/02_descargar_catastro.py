"""02 · Descarga de las capas por manzana de Catastro Bogotá (UAECD), años 2019 a 2021 (los años de los avisos).

Qué hace
    Consulta el servicio ArcGIS público de Catastro Bogotá (sin credenciales) y guarda cuatro capas:
    - catastro/valorreferencia (capa 0): mediana del valor comercial del terreno por m² de cada manzana (V_REF), por año;
      geometría de 2021 y atributos de 2019, 2020 y 2021;
    - catastro/medavalcatphmz (capa 0): mediana del avalúo catastral por m² de los predios en propiedad horizontal;
    - catastro/alturamedia (capa 0): número de pisos ponderado por manzana;
    - ordenamientoterritorial/estratificacion (capa 1): estrato socioeconómico de cada manzana.

Salidas (00_data/processed/catastro/, unos 58 MB; no se versionan)
    manzanas_geometria_2021.parquet, estratificacion_manzanas.parquet, valor_referencia_2019_2021.parquet,
    avaluo_ph_2019_2021.parquet, altura_media_2019_2021.parquet

Recálculo
    Por defecto no descarga nada: el script 04 usa 00_data/processed/variables_catastro.parquet, que viene en el
    repositorio. Con PS3_RECALCULAR=1 (o =02) descarga las capas (requiere internet; unos 5 minutos).
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import pandas as pd

from functions.entorno_ps3 import ARCHIVO_CATASTRO, DIR_CATASTRO, log, recalcular

RECALCULAR = recalcular("02")
SERVICIO = "https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services"
ANIOS = "(DATE '2019-01-01', DATE '2020-01-01', DATE '2021-01-01')"
ARCHIVOS = ["manzanas_geometria_2021.parquet", "estratificacion_manzanas.parquet", "valor_referencia_2019_2021.parquet",
            "avaluo_ph_2019_2021.parquet", "altura_media_2019_2021.parquet"]


def consultar(url, params):
    """POST al servicio ArcGIS con hasta 5 intentos (el servicio a veces corta la conexión)."""
    for intento in range(5):
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode(params).encode(),
                                         headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:                    # reintento con espera creciente
            time.sleep(3 * (intento + 1))
    raise RuntimeError(f"Falló la consulta {url} (desplazamiento {params.get('resultOffset')})")


def descargar(capa, where, campos, geometria, pagina=2000):
    url = f"{SERVICIO}/{capa}/query"
    n = consultar(url, {"where": where, "returnCountOnly": "true", "f": "json"})["count"]
    base = {"where": where, "outFields": campos, "returnGeometry": "true" if geometria else "false",
            "orderByFields": "OBJECTID", "resultRecordCount": pagina}
    if geometria:
        base.update({"f": "geojson", "outSR": 4326, "geometryPrecision": 6})
    else:
        base["f"] = "json"
    with ThreadPoolExecutor(4) as ex:
        partes = list(ex.map(lambda o: consultar(url, {**base, "resultOffset": o}), range(0, n, pagina)))
    if geometria:
        datos = gpd.GeoDataFrame.from_features([f for p in partes for f in p["features"]], crs="EPSG:4326")
    else:
        datos = pd.DataFrame([f["attributes"] for p in partes for f in p["features"]])
    log(f"{capa} [{where}]: {len(datos):,} de {n:,} registros")
    return datos


if RECALCULAR:
    geo_vref = descargar("catastro/valorreferencia/MapServer/0", "ANO = DATE '2021-01-01'", "OBJECTID,MANCODIGO", True)
    geo_vref.to_parquet(DIR_CATASTRO / "manzanas_geometria_2021.parquet")
    estr = descargar("ordenamientoterritorial/estratificacion/MapServer/1", "1=1", "OBJECTID,CODIGO_MANZANA,ESTRATO", True)
    estr.to_parquet(DIR_CATASTRO / "estratificacion_manzanas.parquet")
    for capa, campos, fecha, nombre in [
            ("catastro/valorreferencia/MapServer/0", "MANCODIGO,V_REF,ANO", "ANO", "valor_referencia"),
            ("catastro/medavalcatphmz/MapServer/0", "MANCODIGO,AV_CAT_PH,VIGENCIA,PREDIOS", "VIGENCIA", "avaluo_ph"),
            ("catastro/alturamedia/MapServer/0", "MANCODIGO,ALTURA,ANO", "ANO", "altura_media")]:
        d = descargar(capa, f"{fecha} IN {ANIOS}", campos, False)
        d["anio"] = pd.to_datetime(d[fecha], unit="ms").dt.year
        d.drop(columns=fecha).to_parquet(DIR_CATASTRO / f"{nombre}_2019_2021.parquet", index=False)
    log("Descarga completa: " + ", ".join(f"{a} {(DIR_CATASTRO / a).stat().st_size / 1e6:.1f} MB" for a in ARCHIVOS))
else:
    hay = [a for a in ARCHIVOS if (DIR_CATASTRO / a).exists()]
    log(f"Sin descarga (PS3_RECALCULAR=02 para descargar). Capas de Catastro presentes: {len(hay)} de {len(ARCHIVOS)}. "
        f"El script 04 usa {ARCHIVO_CATASTRO.as_posix()}" + (" (existe)" if ARCHIVO_CATASTRO.exists() else " (FALTA)"))
