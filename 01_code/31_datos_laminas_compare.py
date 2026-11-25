"""31 · Datos de las láminas de compare: error, sesgo, bootstrap por bloques, riesgo y rangos de las seis familias.

Qué hace (antes document/presentaciones/31_datos_laminas_compare.py)
    Con las predicciones fuera de pliegue guardadas (script 07): MAE y sesgo por familia y validación; bootstrap por bloques
    espaciales (celdas de 1 km y 500 m) de la diferencia con LightGBM; avisos más de 10 % arriba y abajo del precio; sesgo
    por quintil de la predicción; error según las variables fuera del rango del entrenamiento; correlación de los errores
    entre familias; y correlación de rangos entre validaciones.

Salida (00_data/processed/laminas/; se versiona porque la lee el script 22)
    compare_nuevo.json
Insumos: scripts 07, 12 (entrenamiento_mapa.parquet) y 16 (ppt_chapinero_fuera.parquet). Tiempo: unos minutos.
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

from functions.entorno_ps3 import (ARCHIVO_AMENIDADES, ARCHIVO_LOCALIDADES, CENTRO, DIR_CACHE_TEXTO,  # noqa: E402,F401
                                   DIR_MODELOS, DIR_OSM, DIR_PRED, DIR_RAW, DIR_TEXTO, FACTOR, existen, log,
                                   recalcular)
from functions.laminas_entorno import DATOS, IMG  # noqa: E402,F401  (00_data/processed/laminas/ y 02_outputs/figures/laminas/)

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.stats import spearmanr

D = DATOS
STORES = DIR_PROC  # carpeta de intermedios del repositorio
FAM = ["lightgbm", "catboost", "random_forest", "cart", "nn_atencion", "elastic_net", "ols"]
CH = "Chapinero (imitando el taller)"
B = 10_000
rng = np.random.default_rng(2026)

oof = {f: pd.read_parquet(DIR_MODELOS / f"cv_clase_oof_{f}_con.parquet") for f in FAM}
mapa = pd.read_parquet(D / "entrenamiento_mapa.parquet")
xs, ys = Transformer.from_crs(4326, 9377, always_xy=True).transform(mapa["lon"].to_numpy(), mapa["lat"].to_numpy())
R = {"familias": FAM}


def tabla(esq):
    """Predicciones de las familias que tienen el esquema, alineadas por idx (avisos comunes)."""
    base = None
    for f in FAM:
        o = oof[f]
        o = o[o["esquema"] == esq]
        if o.empty:
            continue
        o = o[["idx", "real", "pred"]].rename(columns={"pred": f})
        base = o if base is None else base.merge(o[["idx", f]], on="idx", how="inner")
    return base.sort_values("idx").reset_index(drop=True)


# 1. MAE y sesgo
R["mae"], R["sesgo"] = {}, {}
for esq, nom in (("aleatoria", "aleatoria"), ("clusters", "clusters"), (CH, "chapinero")):
    t = tabla(esq)
    R["mae"][nom] = {f: float(np.mean(np.abs(t[f] - t["real"]))) for f in FAM if f in t}
    R["sesgo"][nom] = {f: float(np.mean(t[f] - t["real"])) for f in FAM if f in t}
    R[f"n_{nom}"] = int(len(t))


# 2. Bootstrap por bloques espaciales
def celdas(idx, tam):
    cx = np.floor(xs[idx] / tam).astype(np.int64)
    cy = np.floor(ys[idx] / tam).astype(np.int64)
    return pd.factorize(cx * 1_000_003 + cy)[0]


def boot_bloques(t, tam, b=B):
    c = celdas(t["idx"].to_numpy(), tam)
    k = c.max() + 1
    err = {f: np.abs(t[f].to_numpy() - t["real"].to_numpy()) for f in FAM if f in t}
    # sumas y conteos por celda: una réplica sortea k celdas con reemplazo
    cnt = np.bincount(c, minlength=k).astype(float)
    sums = {f: np.bincount(c, weights=e, minlength=k) for f, e in err.items()}
    draws = rng.integers(0, k, (b, k))
    w = np.zeros((b, k))
    for j in range(k):
        w[:, j] = (draws == j).sum(1)
    n_b = w @ cnt
    mae_b = {f: (w @ s) / n_b for f, s in sums.items()}
    out = {"celdas": int(k), "avisos": int(len(t))}
    for f in err:
        if f == "lightgbm":
            continue
        d = mae_b[f] - mae_b["lightgbm"]
        out[f] = {"dif": float(err[f].mean() - err["lightgbm"].mean()), "bajo": float(np.percentile(d, 2.5)),
                  "alto": float(np.percentile(d, 97.5))}
    out["ic_mae"] = {f: [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))] for f, m in mae_b.items()}
    return out


tch = tabla(CH)
R["bloques_chapinero"] = {"1000": boot_bloques(tch, 1000), "500": boot_bloques(tch, 500)}
tcl = tabla("clusters")
R["bloques_clusters"] = {"1000": boot_bloques(tcl, 1000, b=2000)}
tal = tabla("aleatoria")
R["bloques_aleatoria"] = {"1000": boot_bloques(tal, 1000, b=2000)}

# 3. Riesgo para la empresa y dónde fallan, en Chapinero fuera
fuera = pd.read_parquet(D / "ppt_chapinero_fuera.parquet")[["idx", "n_fuera"]]
tch = tch.merge(fuera, on="idx", how="left")
assert tch["n_fuera"].notna().all()
cat = np.where(tch["n_fuera"] >= 2, "dos_o_mas", np.where(tch["n_fuera"] == 1, "una", "dentro"))
R["riesgo"], R["quintil"], R["rango"] = {}, {}, {}
real = tch["real"].to_numpy()
for f in FAM:
    p = tch[f].to_numpy()
    r_ = p / real - 1
    R["riesgo"][f] = {"sobre_10": float(np.mean(r_ > 0.10) * 100), "bajo_10": float(np.mean(r_ < -0.10) * 100),
                      "dentro_10": float(np.mean(np.abs(r_) <= 0.10) * 100),
                      "error_abs_mediano_pct": float(np.median(np.abs(r_)) * 100)}
    q = pd.qcut(p, 5, labels=False)
    R["quintil"][f] = [{"q": int(k), "pred_media": float(p[q == k].mean()), "real_media": float(real[q == k].mean()),
                        "sesgo": float((p - real)[q == k].mean()), "n": int((q == k).sum())} for k in range(5)]
    R["rango"][f] = {c_: {"n": int((cat == c_).sum()), "mae": float(np.mean(np.abs(p - real)[cat == c_])),
                          "sesgo": float(np.mean((p - real)[cat == c_]))} for c_ in ("dentro", "una", "dos_o_mas")}

# 4. Errores que se mueven juntos
E = np.c_[tuple(tch[f].to_numpy() - real for f in FAM)]
C = np.corrcoef(E.T)
off = C[np.triu_indices(len(FAM), 1)]
R["corr_errores"] = {"min": float(off.min()), "max": float(off.max()), "media": float(off.mean()),
                     "matriz": {f: {g: float(C[i, j]) for j, g in enumerate(FAM)} for i, f in enumerate(FAM)}}
R["todas_subestiman_pct"] = float(np.mean((E < 0).all(1)) * 100)
R["todas_sobreestiman_pct"] = float(np.mean((E > 0).all(1)) * 100)
R["real_media_chapinero"] = float(real.mean())

# 5. Orden entre validaciones
seis = [f for f in FAM if f != "catboost"]
m = R["mae"]
R["spearman"] = {
    "aleatoria_chapinero": float(spearmanr([m["aleatoria"][f] for f in seis], [m["chapinero"][f] for f in seis])[0]),
    "aleatoria_clusters": float(spearmanr([m["aleatoria"][f] for f in seis], [m["clusters"][f] for f in seis])[0]),
    "clusters_chapinero": float(spearmanr([m["clusters"][f] for f in seis], [m["chapinero"][f] for f in seis])[0])}
R["optimismo"] = {f: {"dif": m["chapinero"][f] - m["aleatoria"][f], "razon": m["chapinero"][f] / m["aleatoria"][f]}
                  for f in seis}

json.dump(R, open(D / "compare_nuevo.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: R[k] for k in ("mae", "sesgo", "spearman", "optimismo", "riesgo", "todas_subestiman_pct",
                                    "todas_sobreestiman_pct")}, indent=1, ensure_ascii=False))
print(json.dumps(R["bloques_chapinero"], indent=1))
print(json.dumps(R["bloques_clusters"]["1000"], indent=1)[:1500])
print(json.dumps(R["rango"], indent=1))
print(json.dumps(R["corr_errores"]["min"]), R["corr_errores"]["max"], R["corr_errores"]["media"])
