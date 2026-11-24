"""04 · Variables de Catastro Bogotá por manzana para cada aviso (mismo año del aviso: 2019, 2020 o 2021).

Qué hace (11 variables)
    - Manzana del aviso: la que contiene el punto; si cae en la vía, la más cercana a 100 m o menos
      (dist_manzana_m guarda esa distancia).
    - De su manzana y de su año: ln del valor de referencia del suelo por m² (V_REF), ln del avalúo catastral por m² en
      propiedad horizontal (AV_CAT_PH), pisos ponderados y estrato (0 = manzana sin estrato residencial).
    - Entorno a 300 m (robusto al redondeo de las coordenadas a una malla de unos 110 m): medianas de V_REF, AV_CAT_PH y
      pisos, estrato medio, proporción de manzanas de estrato 6 y número de manzanas, entre las manzanas cuyo centroide
      está a 300 m o menos.
    Si falta el valor del año del aviso, se usa el del año más cercano disponible (2019 a 2021).

Salida (00_data/processed/)
    variables_catastro.parquet   property_id y las 11 variables

Recálculo
    Por defecto usa variables_catastro.parquet (viene en el repositorio). Con PS3_RECALCULAR=1 (o =04) la reconstruye a
    partir de las capas que descarga el script 02 (PS3_RECALCULAR=02,04).
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from functions.entorno_ps3 import ARCHIVO_CATASTRO, CRS_M, DIR_CATASTRO, DIR_RAW, exigir, existen, log, recalcular

RECALCULAR = recalcular("04")
ANIOS = [2019, 2020, 2021]
VARS = ["ln_vref_manzana", "ln_vref_300m", "ln_avaluo_ph_manzana", "ln_avaluo_ph_300m", "pisos_manzana", "pisos_300m",
        "estrato_manzana", "estrato_medio_300m", "prop_estrato6_300m", "n_manzanas_300m", "dist_manzana_m"]
CAPAS = [DIR_CATASTRO / a for a in ("manzanas_geometria_2021.parquet", "estratificacion_manzanas.parquet",
                                    "valor_referencia_2019_2021.parquet", "avaluo_ph_2019_2021.parquet",
                                    "altura_media_2019_2021.parquet")]


def construir():
    kag = pd.concat([pd.read_csv(DIR_RAW / "train.csv"), pd.read_csv(DIR_RAW / "test.csv")], ignore_index=True)
    av = gpd.GeoDataFrame(kag[["property_id", "year"]], geometry=gpd.points_from_xy(kag["lon"], kag["lat"]),
                          crs="EPSG:4326").to_crs(CRS_M)
    man = gpd.read_parquet(CAPAS[0]).to_crs(CRS_M)[["MANCODIGO", "geometry"]]
    man = man.drop_duplicates("MANCODIGO").reset_index(drop=True)
    estr = gpd.read_parquet(CAPAS[1]).to_crs(CRS_M)[["ESTRATO", "geometry"]]
    vref, avph, alt = (pd.read_parquet(c) for c in CAPAS[2:])
    log(f"Manzanas con geometría: {len(man):,} | estratificación: {len(estr):,} | V_REF: {len(vref):,} | "
        f"AV_CAT_PH: {len(avph):,} | ALTURA: {len(alt):,}")

    def tabla_anual(d, campo):
        """Matriz manzana × año con el valor del campo; huecos con el año más cercano disponible."""
        t = d.pivot_table(index="MANCODIGO", columns="anio", values=campo, aggfunc="median").reindex(columns=ANIOS)
        t = t.reindex(man["MANCODIGO"])
        for a in ANIOS:
            orden = sorted(ANIOS, key=lambda b: (abs(b - a), b))
            col = t[a].copy()
            for b in orden[1:]:
                col = col.fillna(t[b])
            t[a] = col
        return t.to_numpy(float)                          # filas alineadas con `man`

    V = {"vref": tabla_anual(vref, "V_REF"), "avaluo_ph": tabla_anual(avph, "AV_CAT_PH"), "pisos": tabla_anual(alt, "ALTURA")}

    # --- Manzana del aviso (contiene el punto; si no, la más cercana a 100 m o menos) ---
    ubic = av.drop_duplicates(subset=None).copy()
    ubic["x"], ubic["y"] = ubic.geometry.x.round(2), ubic.geometry.y.round(2)
    u = ubic.drop_duplicates(["x", "y"])[["x", "y", "geometry"]].reset_index(drop=True)
    j = gpd.sjoin_nearest(u, man.reset_index().rename(columns={"index": "i_man"}), how="left", max_distance=100,
                          distance_col="d")
    j = j[~j.index.duplicated()]
    u["i_man"], u["dist_manzana_m"] = j["i_man"].values, j["d"].values
    je = gpd.sjoin_nearest(u[["geometry"]], estr, how="left", max_distance=100, distance_col="d")
    je = je[~je.index.duplicated()]
    u["estrato_manzana"] = je["ESTRATO"].values
    log(f"Ubicaciones únicas: {len(u):,} | con manzana a 100 m o menos: {u['i_man'].notna().mean():.1%} | "
        f"dentro de la manzana: {(u['dist_manzana_m'] == 0).mean():.1%} | con estrato: {u['estrato_manzana'].notna().mean():.1%}")

    # --- Entorno a 300 m (centroides de manzana) ---
    cen = man.geometry.centroid
    vec = cKDTree(np.c_[cen.x, cen.y]).query_ball_point(np.c_[u["x"], u["y"]], r=300)
    cen_e = estr.geometry.centroid
    est_v = estr["ESTRATO"].to_numpy(float)
    vec_e = cKDTree(np.c_[cen_e.x, cen_e.y]).query_ball_point(np.c_[u["x"], u["y"]], r=300)
    res_e = [est_v[v][est_v[v] > 0] for v in vec_e]
    u["estrato_medio_300m"] = [r.mean() if len(r) else np.nan for r in res_e]
    u["prop_estrato6_300m"] = [(r == 6).mean() if len(r) else np.nan for r in res_e]
    u["n_manzanas_300m"] = [len(v) for v in vec]

    filas = av.assign(x=av.geometry.x.round(2), y=av.geometry.y.round(2)).merge(u.drop(columns="geometry"), on=["x", "y"],
                                                                                how="left")
    ia = filas["i_man"].to_numpy()
    col_anio = filas["year"].map({a: k for k, a in enumerate(ANIOS)}).to_numpy()
    tiene = np.isfinite(ia)
    iu = filas.merge(u.reset_index()[["x", "y", "index"]], on=["x", "y"], how="left")["index"].to_numpy()
    for nombre, M in V.items():
        v_man = np.full(len(filas), np.nan)
        v_man[tiene] = M[ia[tiene].astype(int), col_anio[tiene]]
        filas[f"{nombre}_manzana"] = v_man
        med = np.full((len(u), len(ANIOS)), np.nan)       # mediana a 300 m por ubicación única y año
        for k in range(len(ANIOS)):
            col = M[:, k]
            med[:, k] = [np.nanmedian(col[v]) if len(v) and np.isfinite(col[v]).any() else np.nan for v in vec]
        filas[f"{nombre}_300m"] = med[iu, col_anio]
    for c in ["vref_manzana", "vref_300m", "avaluo_ph_manzana", "avaluo_ph_300m"]:
        filas[f"ln_{c}"] = np.log(filas[c].where(filas[c] > 0))
    filas[["property_id"] + VARS].to_parquet(ARCHIVO_CATASTRO, index=False)


if RECALCULAR:
    exigir(*CAPAS, como="Descargue las capas con PS3_RECALCULAR=02 (script 02) antes de recalcular este paso.")
    construir()
elif not existen(ARCHIVO_CATASTRO):
    if existen(*CAPAS):
        construir()
    else:
        exigir(ARCHIVO_CATASTRO, como="Viene en el repositorio; para reconstruirla corra 02 y 04 con PS3_RECALCULAR=02,04.")
else:
    log("Se usa la base guardada (para reconstruirla: PS3_RECALCULAR=02,04)")

out = pd.read_parquet(ARCHIVO_CATASTRO)
log(f"{ARCHIVO_CATASTRO.as_posix()}: {len(out):,} avisos y {out.shape[1] - 1} variables")
print(out[VARS].describe().T[["count", "mean", "50%", "min", "max"]].round(2).to_string(), flush=True)
