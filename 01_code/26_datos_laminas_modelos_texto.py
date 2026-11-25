"""26 · Datos de las láminas: cifras para explicar los modelos del texto (TF-IDF, LSA, LDA y ridge).

Qué hace (antes document/presentaciones/26_datos_laminas_modelos_texto.py)
    Describe la matriz TF-IDF, recalcula la SVD truncada de 50 componentes con la misma semilla solo para describirla
    (proporción de varianza explicada y términos de los primeros ejes) y comprueba que coincide con las variables lsa_
    guardadas; mide la calidad fuera de pliegue del puntaje ridge del modelo final (correlación, R² y error en millones) y
    su valor en el aviso de ejemplo. No reemplaza ningún archivo del modelo.

Salida (00_data/processed/laminas/; se versiona porque la leen los scripts 21 y 22)
    ppt_modelos_texto.json
Insumos: script 06 (texto/tfidf.npz, dtm_conteos.npz, vocabulario.csv, dtm_filas.parquet y texto_variables.parquet).
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
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

TXT = DIR_TEXTO
SEMILLA = 2026
PID = "84a9da9d84b975bcd0954c36"

X = sp.load_npz(TXT / "tfidf.npz").tocsr()
C = sp.load_npz(TXT / "dtm_conteos.npz").tocsr()
voc = pd.read_csv(TXT / "vocabulario.csv")
filas = pd.read_parquet(TXT / "dtm_filas.parquet")["property_id"].values
N, P = X.shape
R = {"N": int(N), "P": int(P), "nnz": int(X.nnz), "densidad_pct": float(100 * X.nnz / (N * P)),
     "terminos_por_aviso_mediana": float(np.median(np.diff(X.indptr)))}

# 1. Una fila de la matriz: el aviso del ejemplo
i = int(np.where(filas == PID)[0][0])
fila = X[i]
cnt = C[i]
tf_sub = {}
for j, c in zip(cnt.indices, cnt.data):
    tf_sub[j] = 1 + np.log(c)
w_sin_norma = {j: tf_sub[j] * voc.loc[j, "idf"] for j in tf_sub}
norma = float(np.sqrt(sum(v ** 2 for v in w_sin_norma.values())))
R["aviso"] = {"terminos": int(fila.nnz), "norma": norma}
elegidos = ["refugi", "chic refugi", "chimene", "gym", "terraz panoram", "parqueader", "apartament", "alcob"]
tabla = []
for t in elegidos:
    j = voc.index[voc["termino"] == t]
    if len(j) == 0:
        continue
    j = int(j[0])
    tabla.append({"termino": t, "bigrama": bool(voc.loc[j, "bigrama"]), "conteo": int(C[i, j]),
                  "df": int(voc.loc[j, "df"]), "idf": float(voc.loc[j, "idf"]),
                  "peso_sin_normalizar": float(w_sin_norma.get(j, 0.0)), "peso": float(X[i, j])})
R["aviso"]["tabla"] = tabla

# 2. LSA: SVD truncada con la misma semilla; energía y términos con más carga; comprobación contra lsa_ guardadas
svd = TruncatedSVD(n_components=50, random_state=SEMILLA)
Z = svd.fit_transform(X)
energia = svd.singular_values_ ** 2 / sp.linalg.norm(X, "fro") ** 2
R["lsa"] = {"energia_pct_por_componente": [float(100 * e) for e in energia],
            "energia_pct_acumulada_50": float(100 * energia.sum())}
tv = pd.read_parquet(TXT / "texto_variables.parquet").set_index("property_id").loc[filas]
corr = [float(abs(np.corrcoef(Z[:, k], tv[f"lsa_{k:02d}"].values)[0, 1])) for k in range(50)]
R["lsa"]["corr_min_con_guardadas"] = min(corr)
comp = []
for k in range(4):
    v = svd.components_[k]
    pos = [voc.loc[j, "termino"] for j in np.argsort(-v)[:6]]
    neg = [voc.loc[j, "termino"] for j in np.argsort(v)[:6]]
    comp.append({"k": k, "positivos": pos, "negativos": neg, "valor_aviso": float(Z[i, k])})
R["lsa"]["componentes"] = comp

# Proporción de varianza explicada por cada componente (explained_variance_ratio_ de la misma SVD)
R["lsa"]["pve_pct_por_componente"] = [float(100 * e) for e in svd.explained_variance_ratio_]
R["lsa"]["pve_pct_acumulada_50"] = float(100 * svd.explained_variance_ratio_.sum())

# 3. Ridge: calidad fuera de pliegue en el entrenamiento (puntaje guardado del modelo final)
kag = pd.concat([pd.read_csv(DIR_RAW / "train.csv").assign(m="train"),
                 pd.read_csv(DIR_RAW / "test.csv").assign(m="test")], ignore_index=True)
print("filas Kaggle", len(kag))
# Puntaje ridge del modelo final, en el orden de la base (entrenamiento y luego prueba de Kaggle): en el entrenamiento
# es fuera de pliegue (GroupKFold por grupo de texto); se compara con ln(precio en millones)
pr = np.load(DIR_CACHE_TEXTO / "ridge_final.npy").astype(float)
es_tr = (kag["m"] == "train").values
lp = np.log(kag["price"].values / 1e6)
a_, b_ = pr[es_tr], lp[es_tr]
i_k = int(np.where(kag["property_id"].values == PID)[0][0])
R["ridge"] = {"corr_oof": float(np.corrcoef(a_, b_)[0, 1]),
              "r2_oof": float(1 - np.sum((b_ - a_) ** 2) / np.sum((b_ - b_.mean()) ** 2)),
              "mae_oof_millones": float(np.mean(np.abs(np.exp(a_) - np.exp(b_)))),
              "puntaje_aviso": float(pr[i_k]), "puntaje_aviso_millones": float(np.exp(pr[i_k]))}
json.dump(R, open(DATOS / "ppt_modelos_texto.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: R[k] for k in ("N", "P", "nnz", "densidad_pct", "terminos_por_aviso_mediana")}, ensure_ascii=False))
print("aviso", R["aviso"]["terminos"], round(norma, 3))
for r in tabla:
    print(r)
print("LSA energía 50:", round(R["lsa"]["energia_pct_acumulada_50"], 2), "| primeras:",
      [round(e, 2) for e in R["lsa"]["energia_pct_por_componente"][:5]], "| corr mínima con guardadas:",
      round(R["lsa"]["corr_min_con_guardadas"], 4))
for c in comp:
    print(c)
