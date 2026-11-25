"""28 · Datos de las láminas de las etapas 2 a 4: variables, primer corte, pliegues, pronóstico y ajustes.

Qué hace (antes document/presentaciones/28_datos_laminas_etapas2a4.py)
    Con la base del modelo final (sin validar ni entrenar): 1. tipos de las 162 variables; 2. tres avisos de ejemplo con
    sus variables; 3. búsqueda del primer corte con los 38.644 avisos (ganancia de LightGBM para pérdida L2 y lambda = 1);
    4. tamaños de los pliegues de cada esquema; 5. predicción final antes y después del ×1,08 y de los gemelos de texto,
    con la comprobación contra la huella SHA-256 del envío 02; 6. rejilla del factor y peso del gemelo (comprobaciones).

Salida (00_data/processed/laminas/; se versiona porque la leen los scripts 21 y 22)
    etapas2a4.json
Insumos: scripts 06 a 09 y 16 (ppt_tres_avisos.json). Tiempo: unos 2 minutos.
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

from functions.entorno_ps3 import (ARCHIVO_AMENIDADES, ARCHIVO_LOCALIDADES, CENTRO, DIR_CACHE_TEXTO,  # noqa: E402,F401
                                   DIR_MODELOS, DIR_OSM, DIR_PRED, DIR_RAW, DIR_TEXTO, FACTOR, existen, log,
                                   recalcular)
from functions.laminas_entorno import DATOS, IMG  # noqa: E402,F401  (00_data/processed/laminas/ y 02_outputs/figures/laminas/)

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.stdout.reconfigure(encoding="utf-8")
from functions import flujo_final as F  # noqa: E402  (base del modelo final: F.g, F.y, F.idx_tr, F.COLUMNAS, ...)
F.PLIEGUES_CLASE = F.pliegues_clase(("aleatoria", "bloques", "bloques_buffer", "clusters"))
g, y, idx_tr, idx_te = F.g, F.y, F.idx_tr, F.idx_te
NOMBRES = F.COLUMNAS["con"] + ["puntaje_texto_ridge"]
punt = np.load(DIR_CACHE_TEXTO / "ridge_final.npy")
X = np.c_[g[F.COLUMNAS["con"]].to_numpy(float), punt]
assert X.shape[1] == 162
R = {}

# 1. Tipos de variables
Xtr, Xte = X[idx_tr], X[idx_te]
binaria = [bool(np.isin(np.unique(X[:, j][np.isfinite(X[:, j])]), [0, 1]).all()) for j in range(162)]
falta_tr = np.isnan(Xtr).any(0)
falta_te = np.isnan(Xte).any(0)
R["variables"] = {"n": 162, "binarias": int(sum(binaria)), "continuas": int(162 - sum(binaria)),
                  "con_faltantes_entrenamiento": int(falta_tr.sum()), "con_faltantes_prueba": int(falta_te.sum()),
                  "con_faltantes_alguna": int((falta_tr | falta_te).sum()),
                  "celdas_faltantes_entrenamiento_pct": float(100 * np.isnan(Xtr).mean()),
                  "binarias_nombres": [NOMBRES[j] for j in range(162) if binaria[j]]}

# 2. Tres avisos de ejemplo (los de la ilustración de la Etapa 2) con variables legibles
tres = json.load(open(DATOS / "ppt_tres_avisos.json", encoding="utf-8"))
pos = pd.Series(np.arange(len(g)), index=g["property_id"].values)
COLS_EJ = ["area_imp", "banos_imp", "es_apartamento", "estrato_texto", "ln_dist_transmilenio_m", "ln_vref_300m",
           "txt_ascensor", "txt_chimenea", "txt_terraza_balcon", "txt_parqueadero", "txt_gimnasio", "puntaje_texto_ridge"]
ej = []
for a in tres["avisos"]:
    i = int(pos[a["property_id"]])
    fila = {"property_id": a["property_id"], "precio": float(y[i]), "localidad": str(g["localidad"].values[i]),
            "es_entrenamiento": bool(i in set(idx_tr))}
    for c in COLS_EJ:
        v = float(X[i, NOMBRES.index(c)])
        fila[c] = None if not np.isfinite(v) else v
    fila["dist_transmilenio_m"] = float(np.expm1(fila["ln_dist_transmilenio_m"]))
    fila["vref_300m_mill"] = None if fila["ln_vref_300m"] is None else float(np.exp(fila["ln_vref_300m"]) / 1e6)
    fila["puntaje_mill"] = float(np.exp(fila["puntaje_texto_ridge"]))
    ej.append(fila)
R["ejemplos"] = ej

# 3. Búsqueda del primer corte (pérdida L2: gradiente ŷ − y, hessiana 1; la hoja vale S/(n + λ))
LAMBDA, MIN_HOJA = 1.0, 20
r = y[idx_tr] - y[idx_tr].mean()
S, N = r.sum(), len(r)
base = S ** 2 / (N + LAMBDA)
sse_total = float((r ** 2).sum())


def mejor_corte(x):
    ok = np.isfinite(x)
    xs_, rs_ = x[ok], r[ok]
    s_nan, n_nan = r[~ok].sum(), int((~ok).sum())
    o = np.argsort(xs_, kind="mergesort")
    xs_, rs_ = xs_[o], rs_[o]
    cs = np.cumsum(rs_)
    distintos = np.where(np.diff(xs_) > 0)[0]          # cortes entre valores distintos
    mejor = (-np.inf, None, None)
    for lado_nan in ("izq", "der"):
        sl = cs[distintos] + (s_nan if lado_nan == "izq" else 0)
        nl = distintos + 1 + (n_nan if lado_nan == "izq" else 0)
        sr, nr = S - sl, N - nl
        valido = (nl >= MIN_HOJA) & (nr >= MIN_HOJA)
        if not valido.any():
            continue
        gan = sl ** 2 / (nl + LAMBDA) + sr ** 2 / (nr + LAMBDA) - base
        gan = np.where(valido, gan, -np.inf)
        k = int(np.argmax(gan))
        if gan[k] > mejor[0]:
            umbral = (xs_[distintos[k]] + xs_[distintos[k] + 1]) / 2
            mejor = (float(gan[k]), float(umbral), {"n_si": int(nl[k]), "n_no": int(nr[k]), "faltantes_a": lado_nan,
                                                    "media_si": float(sl[k] / nl[k] + y[idx_tr].mean()),
                                                    "media_no": float(sr[k] / nr[k] + y[idx_tr].mean())})
    return mejor


cortes = []
for j, nom in enumerate(NOMBRES):
    gan, umb, info = mejor_corte(Xtr[:, j])
    if info is None:
        continue
    cortes.append({"variable": nom, "ganancia": gan, "reduccion_pct": 100 * gan / sse_total, "umbral": umb, **info})
cortes.sort(key=lambda d: -d["ganancia"])
R["primer_corte"] = {"sse_total": sse_total, "media": float(y[idx_tr].mean()), "top": cortes[:10],
                     "n_variables_probadas": len(cortes)}
print("primer corte:", [(c["variable"], round(c["reduccion_pct"], 1), round(c["umbral"], 3)) for c in cortes[:8]])

# 4. Pliegues de cada esquema
pl = {}
for nom, (esq, a, b, (i_in, i_iv)) in F.PLIEGUES_CLASE.items():
    pl.setdefault(esq, []).append((len(a), len(b), len(i_iv)))
R["pliegues"] = {e: {"pliegues": len(v), "validados": int(sum(t[1] for t in v)), "ent_medio": float(np.mean([t[0] for t in v])),
                     "ent_min": int(min(t[0] for t in v)), "ent_max": int(max(t[0] for t in v)),
                     "val_min": int(min(t[1] for t in v)), "val_max": int(max(t[1] for t in v)),
                     "parada_media": float(np.mean([t[2] for t in v]))} for e, v in pl.items()}
i_in, i_iv = F.mu.bloques_internos(F.xs[idx_tr], F.ys[idx_tr], semilla=F.SEMILLA)
R["parada_final"] = {"ajuste": int(len(i_in)), "parada": int(len(i_iv))}
from scipy.spatial import cKDTree  # noqa: E402
dist = {}
for nom, (esq, a, b, _) in F.PLIEGUES_CLASE.items():
    d_, _ = cKDTree(np.c_[F.xs[a], F.ys[a]]).query(np.c_[F.xs[b], F.ys[b]], k=1)
    dist.setdefault(esq, []).append(d_)
R["distancia_vecino"] = {e: {"mediana": float(np.median(np.concatenate(v))), "pct_0m": float(100 * np.mean(np.concatenate(v) < 1))}
                         for e, v in dist.items()}
print(R["distancia_vecino"])
print(R["pliegues"], R["parada_final"])

# 5. Predicción final y envío 02
L = np.load(DIR_MODELOS / "final_pred_lightgbm_con.npy")
gem = pd.Series(y[idx_tr], index=F.grupo[idx_tr]).groupby(level=0).median()
pg = pd.Series(F.grupo[idx_te]).map(gem).to_numpy()
tiene = np.isfinite(pg)
est = CENTRO + FACTOR * (L - CENTRO)
fin = est.copy()
fin[tiene] = 0.3 * est[tiene] + 0.7 * pg[tiene]
envio = pd.DataFrame({"property_id": g["property_id"].values[idx_te], "price": np.round(fin * 1e6, 0)})
sha = hashlib.sha256(envio.to_csv(index=False).encode("utf-8")).hexdigest()
reg = pd.read_csv(DIR_PRED / "registro_envios_sebastian_vela.csv")
sha02 = reg.loc[reg["archivo"].astype(str).str.contains("catastro_x108"), "sha256"].iloc[0]


def resumen(p):
    return {"media": float(p.mean()), "mediana": float(np.median(p)), "min": float(p.min()), "max": float(p.max()),
            "pct_sobre_1200": float(100 * np.mean(p > 1200)), "n_sobre_1650": int((p > 1650).sum()),
            "n_bajo_300": int((p < 300).sum())}


R["final"] = {"lightgbm": resumen(L), "estirada": resumen(est), "envio": resumen(fin), "n_gemelos": int(tiene.sum()),
              "pct_gemelos": float(100 * tiene.mean()), "sha_coincide": bool(sha == sha02), "n_prueba": int(len(idx_te)),
              "media_entrenamiento": float(y[idx_tr].mean()), "centro": CENTRO, "factor": FACTOR,
              "cambio_estirado_medio_abs": float(np.mean(np.abs(est - L)))}
print("final:", R["final"])

# 6. Rejilla de b y peso del gemelo (comprobaciones del cuaderno 02)
oof = pd.read_parquet(DIR_MODELOS / "cv_clase_oof_lightgbm_con.parquet")
ch = oof[oof["esquema"].str.startswith("Chapinero")]
cl = oof[oof["esquema"] == "clusters"]
fil = []
for b in np.round(np.arange(1.00, 1.125, 0.01), 2):
    f = lambda o: CENTRO + b * (o["pred"] - CENTRO)  # noqa: E731
    fil.append({"b": float(b), "mae_chapinero": float(np.abs(f(ch) - ch["real"]).mean()),
                "sesgo_chapinero": float((f(ch) - ch["real"]).mean()), "mae_clusters": float(np.abs(f(cl) - cl["real"]).mean())})
R["rejilla_b"] = fil
R["pendiente_chapinero"] = float(np.polyfit(ch["pred"], ch["real"], 1)[0])
pls = {n_: v for n_, v in F.PLIEGUES_CLASE.items() if v[0] == "clusters"}
filas = []
for n_, (e, a, b, _) in pls.items():
    gm = pd.Series(y[a], index=F.grupo[a]).groupby(level=0).median()
    pgv = pd.Series(F.grupo[b]).map(gm).to_numpy()
    o = cl[cl["pliegue"] == n_].set_index("idx").loc[b]
    m = np.isfinite(pgv)
    filas.append(pd.DataFrame({"pred": o["pred"].values[m], "real": o["real"].values[m], "gemelo": pgv[m]}))
gv = pd.concat(filas)
R["gemelos_clusters"] = {"n": int(len(gv)), "mae": {f"{w:.1f}": float(np.abs((1 - w) * gv["pred"] + w * gv["gemelo"] - gv["real"]).mean())
                                                     for w in np.round(np.arange(0, 1.01, 0.1), 1)}}
print("gemelos:", R["gemelos_clusters"])

# 7. Métricas de cada validación (MAE, RMSE, R², sesgo) y correlación del error con el de sus 10 vecinos
rk = pd.read_parquet(DIR_MODELOS / "rk_knndm_oof_lightgbm_con.parquet")
conj = {e: (o_["pred"].to_numpy(), o_["real"].to_numpy(), o_["idx"].to_numpy()) for e, o_ in oof.groupby("esquema")}
conj["knndm"] = (rk["base"].to_numpy(), rk["real"].to_numpy(), rk["idx"].to_numpy())
pch, rch, ich = conj["Chapinero (imitando el taller)"]
conj["Chapinero x1,08"] = (CENTRO + FACTOR * (pch - CENTRO), rch, ich)
arbol_xy = cKDTree(np.c_[F.xs[idx_tr], F.ys[idx_tr]])
met = {}
for e, (pr, re_, ix) in conj.items():
    err = pr - re_
    fila = {"n": int(len(err)), "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "r2": float(1 - (err ** 2).sum() / ((re_ - re_.mean()) ** 2).sum()), "sesgo": float(err.mean())}
    if len(err) == len(idx_tr):
        res = pd.Series(err, index=ix).reindex(idx_tr).to_numpy()      # residuo por aviso de entrenamiento
        _, vec = arbol_xy.query(np.c_[F.xs[idx_tr], F.ys[idx_tr]], k=11)
        fila["cor_vecinos10"] = float(np.corrcoef(res, res[vec[:, 1:]].mean(1))[0, 1])
    met[e] = fila
R["metricas"] = met
print("métricas:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in met.items()})
R["dispersion_chapinero"] = {"pred_x108": [float(v) for v in conj["Chapinero x1,08"][0]], "real": [float(v) for v in rch]}

json.dump(R, open(DATOS / "etapas2a4.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("listo: laminas/etapas2a4.json")
