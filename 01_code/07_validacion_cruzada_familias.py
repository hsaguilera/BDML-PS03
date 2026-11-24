"""07 · Validación cruzada de las seis familias del taller con esquemas aleatorios y espaciales.

Qué hace
    Ajusta siete especificaciones (Regresión lineal, Elastic Net, CART, Random Forest, LightGBM, CatBoost y la red con
    atención a los vecinos) con las mismas 161 variables con Catastro más el puntaje de texto, y LightGBM y CatBoost
    también sin Catastro. Cada una se valida fuera de pliegue con los esquemas de la lectura "Validación cruzada
    espacial" (v = 5, semilla 2026; functions/validacion_clase.py):
        aleatoria (vfold_cv), bloques de 2 km (spatial_block_cv), bloques de 2 km con buffer de 300 m,
        clústeres k-means de las coordenadas con v = 10 (spatial_clustering_cv) y
        "Chapinero fuera": se entrena sin Chapinero y se predicen sus 307 avisos (imita la tarea del taller).
    Dentro de cada pliegue se recalcula el puntaje de texto solo con el entrenamiento del pliegue, y la parada temprana
    (o la elección de la complejidad) usa una partición interna con la misma lógica del esquema externo.
    Por tiempo, no todas las especificaciones se validan con todos los esquemas (tabla ESPECIFICACIONES).
    Al final elige los pesos de la mezcla convexa LightGBM + CatBoost (rejilla de paso 0,1) con los clústeres; con
    Catastro da 0,4 y 0,6, los pesos del envío 03 (script 09).

Salidas
    00_data/processed/modelos/: cv_clase_oof_<familia>_<con|sin>.parquet (predicciones fuera de pliegue),
        cv_clase_res_<familia>_<con|sin>.csv (MAE, RMSE y sesgo por esquema) y pesos_mezcla.csv
    02_outputs/tables/: validacion_familias_esquemas.csv (todas las especificaciones y esquemas, formato largo) y
        validacion_mezcla_clusters.csv

Recálculo
    Por defecto lee las predicciones fuera de pliegue guardadas (vienen en el repositorio). Con PS3_RECALCULAR=1 (o =07)
    las vuelve a calcular: unas 2 horas en CPU con 6 procesos (Random Forest y CatBoost son los más lentos).
    PS3_ESPECIFICACIONES=ols_con,cart_con limita el cálculo a esas especificaciones; PS3_N_JOBS fija los procesos.
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import itertools
import os

import numpy as np
import pandas as pd

from functions.entorno_ps3 import DIR_MODELOS, existen, guardar_tabla, log, recalcular

RECALCULAR = recalcular("07")
N_JOBS = int(os.environ.get("PS3_N_JOBS", "6"))
TODOS = ("aleatoria", "bloques", "bloques_buffer", "clusters")
ESPECIFICACIONES = {                      # especificación: esquemas (además de "Chapinero fuera", que va en todas)
    "lightgbm_con": TODOS, "lightgbm_sin": TODOS,
    "catboost_con": ("clusters",), "catboost_sin": ("clusters",),
    "random_forest_con": ("aleatoria", "clusters"), "cart_con": ("aleatoria", "clusters"),
    "nn_atencion_con": ("aleatoria", "clusters"), "elastic_net_con": ("aleatoria", "clusters"),
    "ols_con": ("aleatoria", "clusters"),
}
NOMBRE = {"lightgbm": "LightGBM", "catboost": "CatBoost", "nn_atencion": "Red con atención", "ols": "Regresión lineal",
          "elastic_net": "Elastic Net", "cart": "CART", "random_forest": "Random Forest"}
FAMILIA_TALLER = {"lightgbm": "Boosting", "catboost": "Boosting", "nn_atencion": "Red neuronal", "ols": "Regresión lineal",
                  "elastic_net": "Elastic Net", "cart": "CART", "random_forest": "Random Forest"}
pedidas = [e.strip() for e in os.environ.get("PS3_ESPECIFICACIONES", "").split(",") if e.strip()]


def archivos(espec):
    return DIR_MODELOS / f"cv_clase_oof_{espec}.parquet", DIR_MODELOS / f"cv_clase_res_{espec}.csv"


por_calcular = [e for e in ESPECIFICACIONES if (not pedidas or e in pedidas) and (RECALCULAR or not existen(*archivos(e)))]
if por_calcular:
    from functions import flujo_final as F         # construye la base (unos 30 segundos)
    for espec in por_calcular:
        fam, conj = espec.rsplit("_", 1)
        pliegues = F.pliegues_clase(ESPECIFICACIONES[espec])
        log(f"Validando {NOMBRE[fam]} {conj} Catastro en {len(pliegues)} pliegues ...")
        o = F.oof_clase(fam, conj, pliegues, n_jobs=N_JOBS, recalcular_puntajes=RECALCULAR)
        res = pd.DataFrame(F.metricas(o)).assign(modelo=NOMBRE[fam], catastro=conj)
        o.to_parquet(archivos(espec)[0], index=False)
        res.to_csv(archivos(espec)[1], index=False)
        log(f"{NOMBRE[fam]} · {conj} Catastro: " + " | ".join(f"{r.esquema} MAE {r.MAE:.1f} (sesgo {r.Sesgo:.0f})"
                                                            for r in res.itertuples()))
else:
    log("Se usan las predicciones fuera de pliegue guardadas (para recalcularlas: PS3_RECALCULAR=07)")

# --- Tabla larga con todas las especificaciones ---------------------------------------------------------------------
filas = []
for espec in ESPECIFICACIONES:
    fam, conj = espec.rsplit("_", 1)
    r = pd.read_csv(archivos(espec)[1]).assign(especificacion=espec, familia_taller=FAMILIA_TALLER[fam])
    filas.append(r)
tabla = pd.concat(filas, ignore_index=True)[["familia_taller", "modelo", "catastro", "especificacion", "esquema", "n",
                                             "MAE", "RMSE", "Sesgo", "MAE ponderado"]]
guardar_tabla(tabla, "validacion_familias_esquemas", index=False, decimales=3)

# --- Mezcla convexa LightGBM + CatBoost con pesos elegidos en los clústeres ------------------------------------------
filas, pesos = [], []
for conj in ("sin", "con"):
    fams = ["lightgbm", "catboost"]
    com = None                                     # solo los pliegues que tienen los dos modelos
    for f in fams:
        o = pd.read_parquet(archivos(f"{f}_{conj}")[0])[["esquema", "pliegue", "idx", "real", "peso", "pred"]].rename(columns={"pred": f})
        com = o if com is None else com.merge(o[["pliegue", "idx", f]], on=["pliegue", "idx"], how="inner")
    m = (com["esquema"] == "clusters").values
    P = com[fams].to_numpy()
    rejilla = [np.array(w) for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=len(fams)) if abs(sum(w) - 1) < 1e-9]
    w = min(rejilla, key=lambda w_: np.abs(P[m] @ w_ - com["real"].values[m]).mean())
    pesos.append({"catastro": conj, **{f"peso_{f}": round(float(x), 1) for f, x in zip(fams, w)}})
    pred = P @ w
    for e in list(dict.fromkeys(com["esquema"])):
        mm = (com["esquema"] == e).values
        err = pred[mm] - com["real"].values[mm]
        fila = {"modelo": "Mezcla", "catastro": conj, "esquema": e, "n": int(mm.sum()), "MAE": np.abs(err).mean(),
                "RMSE": np.sqrt((err ** 2).mean()), "Sesgo": err.mean()}
        if e.startswith("Chapinero"):
            fila["MAE ponderado"] = np.average(np.abs(err), weights=com["peso"].values[mm])
        filas.append(fila)
    log(f"Mezcla {conj} Catastro, pesos elegidos con clústeres: LightGBM {w[0]:.1f}, CatBoost {w[1]:.1f}")
pd.DataFrame(pesos).to_csv(DIR_MODELOS / "pesos_mezcla.csv", index=False)
guardar_tabla(pd.DataFrame(filas).merge(pd.DataFrame(pesos), on="catastro"), "validacion_mezcla_clusters", index=False,
              decimales=3)
print(tabla.pivot_table(index=["modelo", "catastro"], columns="esquema", values="MAE").round(1).to_string(), flush=True)
