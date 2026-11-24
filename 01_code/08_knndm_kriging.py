"""08 · Validación kNNDM y regression kriging de los residuos del LightGBM con Catastro.

Qué hace (functions/knndm_kriging.py)
    1. kNNDM (Milà et al. 2022; Linnenbrink et al. 2024): busca pliegues cuya distribución de distancias "aviso validado
       → aviso de entrenamiento más cercano" se parezca a la de la tarea real "aviso de prueba → aviso de entrenamiento
       más cercano". Para cada q de una rejilla agrupa las coordenadas de entrenamiento en q grupos con k-means, los
       reparte en 5 pliegues y mide la distancia de Wasserstein; elige q = 1.600.
    2. Regression kriging (Hengl, Heuvelink y Rossiter 2007): ŷ + kriging simple local (32 vecinos) de los residuos, con
       variograma exponencial con pepita, en niveles y en relativo (log). Sin fuga: los residuos que se interpolan hacia
       el pliegue k salen de modelos entrenados sin el par de pliegues {j, k} (10 modelos más los 5 base).
    Resultado: el kriging no mejora el MAE (a las distancias de kNNDM el residuo es casi todo ruido sin estructura), así
    que el modelo final no lo usa. Las predicciones base fuera de pliegue alimentan el análisis de errores (script 11).

Salidas
    00_data/processed/modelos/: knndm_resumen.csv, rk_knndm_res_lightgbm_con.csv, rk_variogramas_lightgbm_con.csv,
        rk_knndm_oof_lightgbm_con.parquet
    02_outputs/tables/: knndm_rejilla.csv, kriging_residuos_knndm.csv, kriging_variogramas.csv

Recálculo
    Por defecto lee los resultados guardados (vienen en el repositorio). Con PS3_RECALCULAR=1 (o =08) ajusta los 15
    LightGBM (unos 10 minutos con 5 procesos; PS3_N_JOBS fija los procesos).
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from functions import knndm_kriging as es
from functions.entorno_ps3 import DIR_MODELOS, existen, guardar_tabla, log, recalcular

RECALCULAR = recalcular("08")
N_JOBS = int(os.environ.get("PS3_N_JOBS", "5"))
K = 5
SALIDAS = {"resumen": DIR_MODELOS / "knndm_resumen.csv", "res": DIR_MODELOS / "rk_knndm_res_lightgbm_con.csv",
           "variogramas": DIR_MODELOS / "rk_variogramas_lightgbm_con.csv",
           "oof": DIR_MODELOS / "rk_knndm_oof_lightgbm_con.parquet"}

if RECALCULAR or not existen(*SALIDAS.values()):
    from functions import flujo_final as F
    g, y, idx_tr, idx_te, mu = F.g, F.y, F.idx_tr, F.idx_te, F.mu
    xy = np.c_[F.xs, F.ys]
    fold_tr, grupo_q, q_elegido, resumen_k, objetivo = es.knndm(xy[idx_tr], xy[idx_te], k=K)
    GRUPOQ = np.full(len(g), -1)
    GRUPOQ[idx_tr] = grupo_q
    pd.DataFrame(resumen_k).to_csv(SALIDAS["resumen"], index=False)
    log(f"kNNDM: q = {q_elegido} | prueba → entrenamiento: mediana {objetivo['mediana_m']:.0f} m, p90 {objetivo['p90_m']:.0f} m, "
        f"{objetivo['pct_0m']:.0f} % a 0 m | " + ", ".join(f"pliegue {k}: {(fold_tr == k).sum():,}" for k in range(K)))
    es_chap = g["localidad"].values == "Chapinero"

    def interna(A, semilla):
        """Partición interna de la parada temprana: el 20 % de los grupos kNNDM del entrenamiento del pliegue."""
        grupos = np.unique(GRUPOQ[A])
        parada = np.random.default_rng(semilla).choice(grupos, size=max(1, int(0.2 * len(grupos))), replace=False)
        m = np.isin(GRUPOQ[A], parada)
        return np.where(~m)[0], np.where(m)[0]

    # --- 5 modelos base (sin el pliegue k) y 10 modelos sin el par {j, k} ---
    X = g[F.COLUMNAS["con"]].to_numpy(float)
    tareas = [(f"knndm-{k}", idx_tr[fold_tr != k], idx_tr[fold_tr == k]) for k in range(K)]
    for j in range(K):
        for k in range(j + 1, K):
            tareas.append((f"knndm-par-{j}{k}", idx_tr[(fold_tr != j) & (fold_tr != k)], idx_tr[(fold_tr == j) | (fold_tr == k)]))
    gen = (delayed(mu.ajustar_predecir)("lightgbm", np.c_[X, F.puntajes(n, A, B, RECALCULAR)], y, F.xs, F.ys, F.lpm2, F.lp,
                                        F.grupo, None, A, B, False, False, F.params_de("lightgbm", interna(A, F.SEMILLA + i)),
                                        F.SEMILLA)
           for i, (n, A, B) in enumerate(tareas))
    preds = Parallel(n_jobs=N_JOBS)(gen)
    base = np.full(len(g), np.nan)
    par = {}
    for (n, A, B), p in zip(tareas, preds):
        if "-par-" in n:
            par[n[-2:]] = pd.Series(p, index=B)
        else:
            base[B] = p

    # --- Kriging de los residuos hacia cada pliegue ---
    rk = {"niveles": np.full(len(g), np.nan), "relativo": np.full(len(g), np.nan)}
    vgs = []
    for k in range(K):
        B = idx_tr[fold_tr == k]
        src, pr = [], []
        for j in range(K):
            if j == k:
                continue
            filas = idx_tr[fold_tr == j]
            src.append(filas)
            pr.append(par[f"{min(j, k)}{max(j, k)}"].loc[filas].to_numpy())
        src, pr = np.concatenate(src), np.concatenate(pr)
        r_niv = y[src] - pr
        vg = es.ajustar_variograma(xy[src], r_niv)
        rk["niveles"][B] = base[B] + es.kriging_simple(xy[src], r_niv, xy[B], vg)
        r_rel = np.log(y[src]) - np.log(np.maximum(pr, 50))
        vg_r = es.ajustar_variograma(xy[src], r_rel)
        rk["relativo"][B] = base[B] * np.exp(es.kriging_simple(xy[src], r_rel, xy[B], vg_r))
        vgs.append({"pliegue": k, "c0": vg["c0"], "c1": vg["c1"], "alcance_m": vg["a"], "c0_rel": vg_r["c0"],
                    "c1_rel": vg_r["c1"], "alcance_rel_m": vg_r["a"]})

    def resumen(pred, nombre):
        e = pred[idx_tr] - y[idx_tr]
        ch = es_chap[idx_tr]
        alto = y[idx_tr] > 1000
        return {"variante": nombre, "MAE kNNDM": np.abs(e).mean(), "RMSE kNNDM": np.sqrt((e ** 2).mean()), "Sesgo": e.mean(),
                "MAE ponderado": np.average(np.abs(e), weights=F.PESOS[idx_tr]), "MAE Chapinero (307)": np.abs(e[ch]).mean(),
                "Sesgo Chapinero": e[ch].mean(), "Sesgo > 1.000 M": e[alto].mean()}

    pd.DataFrame([resumen(base, "sin kriging"), resumen(rk["niveles"], "RK en niveles"),
                  resumen(rk["relativo"], "RK relativo (log)")]).assign(modelo="LightGBM", catastro="con").to_csv(
        SALIDAS["res"], index=False)
    pd.DataFrame(vgs).to_csv(SALIDAS["variogramas"], index=False)
    pd.DataFrame({"idx": idx_tr, "fold": fold_tr, "real": y[idx_tr], "base": base[idx_tr], "rk_niveles": rk["niveles"][idx_tr],
                  "rk_relativo": rk["relativo"][idx_tr], "peso": F.PESOS[idx_tr]}).to_parquet(SALIDAS["oof"], index=False)
else:
    log("Se usan los resultados guardados de kNNDM y kriging (para recalcularlos: PS3_RECALCULAR=08)")

res = pd.read_csv(SALIDAS["res"])
vgs = pd.read_csv(SALIDAS["variogramas"])
guardar_tabla(pd.read_csv(SALIDAS["resumen"]), "knndm_rejilla", index=False, decimales=3)
guardar_tabla(res, "kriging_residuos_knndm", index=False, decimales=3)
guardar_tabla(vgs, "kriging_variogramas", index=False, decimales=3)
log(f"Variogramas (niveles): pepita ≈ {vgs.c0.mean():.0f}, meseta parcial ≈ {vgs.c1.mean():.0f}, alcance ≈ {vgs.alcance_m.mean():.0f} m")
print(res.round(1).to_string(index=False), flush=True)
