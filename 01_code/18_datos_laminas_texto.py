"""18 · Datos de las láminas de texto como datos (Lecture 15; Gentzkow y otros, 2019).

Qué hace (antes document/presentaciones/datos_ppt5.py; no ajusta modelos: cuenta frecuencias)
    Lee la base de texto del script 06 (matriz de conteos, vocabulario, tópicos LDA, auditorías de limpieza y de
    ortografía) y la base de los mapas del script 12.

Salidas (00_data/processed/laminas/; se regeneran en cada corrida)
    ppt_texto_zipf.csv (frecuencia documental por rango del término), ppt_texto_frecuentes.csv (palabras y bigramas más
    frecuentes), ppt_texto_chapinero.csv (términos que más distinguen la prueba del entrenamiento fuera de Chapinero),
    ppt_texto_topicos.csv (tópicos LDA, peso en Chapinero y en el resto, correlación con el precio) y
    ppt_texto_limpieza.json (conteos y ejemplos de la limpieza y de la corrección con BETO)
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import json
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from nltk.stem.snowball import SnowballStemmer
from scipy.stats import spearmanr

from functions.entorno_ps3 import DIR_RAW, DIR_TAB, DIR_TEXTO
from functions.laminas_entorno import DATOS as D


dtm = sp.load_npz(DIR_TEXTO / "dtm_conteos.npz").tocsc()
voc = pd.read_csv(DIR_TEXTO / "vocabulario.csv")
filas = pd.read_parquet(DIR_TEXTO / "dtm_filas.parquet")["property_id"].astype(str).to_numpy()
assert dtm.shape == (len(filas), len(voc))
tr = pd.read_csv(DIR_RAW / "train.csv")[["property_id", "price", "lat", "lon"]]
te = pd.read_csv(DIR_RAW / "test.csv")[["property_id"]]
ent = pd.read_parquet(D / "entrenamiento_mapa.parquet")  # mismo orden que train.csv; se comprueba con lat, lon y precio
assert len(ent) == len(tr) and np.allclose(ent["lat"], tr["lat"]) and np.allclose(ent["lon"], tr["lon"])
loc_tr = dict(zip(tr["property_id"].astype(str), ent["localidad"]))
es_te = np.isin(filas, te["property_id"].astype(str).to_numpy())
es_tr_resto = np.array([(p in loc_tr) and (loc_tr[p] != "Chapinero") for p in filas])
n_te, n_resto, n_tot = int(es_te.sum()), int(es_tr_resto.sum()), len(filas)
print("avisos:", n_tot, "| prueba:", n_te, "| entrenamiento fuera de Chapinero:", n_resto)

presencia = (dtm > 0).astype(np.int8).tocsc()
df_tot = np.asarray(presencia.sum(axis=0)).ravel()
df_te = np.asarray(presencia[es_te].sum(axis=0)).ravel()
df_resto = np.asarray(presencia[es_tr_resto].sum(axis=0)).ravel()

# Forma escrita más común de cada raíz, contada en el texto normalizado (después de la limpieza y de la ortografía).
st = SnowballStemmer("spanish")
txt = pd.read_parquet(DIR_TEXTO / "texto_normalizado.parquet")["texto_normalizado"].fillna("")
conteo = Counter()
for t in txt:
    conteo.update(re.findall(r"[a-zñ0-9]+", t.lower()))
formas = defaultdict(Counter)
for tok, c in conteo.items():
    formas[st.stem(tok)][tok] += c


def legible(termino):
    partes = str(termino).split(" ")
    return " ".join(formas[p_].most_common(1)[0][0] if formas.get(p_) else p_ for p_ in partes)


# 1. Ley de Zipf: frecuencia documental por rango (puntos espaciados en escala logarítmica).
orden = np.sort(df_tot)[::-1]
rangos = np.unique(np.round(np.logspace(0, np.log10(len(orden)), 300)).astype(int))
pd.DataFrame({"rango": rangos, "avisos": orden[rangos - 1]}).to_csv(D / "ppt_texto_zipf.csv", index=False)

# 2. Más frecuentes: palabras (sin cifras sueltas) y bigramas, en % de avisos.
v = voc.assign(df_tot=df_tot, pct=100 * df_tot / n_tot)
es_num = v["termino"].astype(str).str.fullmatch(r"[0-9 ]+")
uni = v[(~v["bigrama"]) & (~es_num)].nlargest(15, "df_tot").assign(tipo="palabra")
bi = v[(v["bigrama"]) & (~es_num)].nlargest(10, "df_tot").assign(tipo="bigrama")
frec = pd.concat([uni, bi])
frec["legible"] = frec["termino"].map(legible)
frec[["tipo", "termino", "legible", "df_tot", "pct"]].to_csv(D / "ppt_texto_frecuentes.csv", index=False)

# 3. Qué distingue el texto de Chapinero: razón de proporciones de avisos, con suavizado, entre términos frecuentes.
p_te = (df_te + 1) / (n_te + 2)
p_resto = (df_resto + 1) / (n_resto + 2)
v["razon"] = p_te / p_resto
v["pct_chapinero"], v["pct_resto"] = 100 * df_te / n_te, 100 * df_resto / n_resto
cand = v[(df_tot >= 300) & (~es_num)].copy()
mas = cand.nlargest(12, "razon").assign(lado="más en Chapinero")
menos = cand.nsmallest(12, "razon").assign(lado="más en el resto de Bogotá")
dist = pd.concat([mas, menos])
dist["legible"] = dist["termino"].map(legible)
dist[["lado", "termino", "legible", "pct_chapinero", "pct_resto", "razon"]].to_csv(D / "ppt_texto_chapinero.csv", index=False)

# 4. Tópicos LDA (K = 10 elegido por perplejidad en el cuaderno 01): palabras, peso en Chapinero y relación con el precio.
lda = pd.read_csv(DIR_TAB / "texto_lda_topicos.csv", index_col=0)
tv = pd.read_parquet(DIR_TEXTO / "texto_variables.parquet")
tv["property_id"] = tv["property_id"].astype(str)
top_cols = [f"topico_{k:02d}" for k in range(10)]
m_te = tv[tv["property_id"].isin(set(filas[es_te]))][top_cols].mean()
m_resto = tv[tv["property_id"].isin(set(filas[es_tr_resto]))][top_cols].mean()
trp = tr.assign(property_id=tr["property_id"].astype(str)).merge(tv[["property_id"] + top_cols], on="property_id")
filas_top = []
for k, c in enumerate(top_cols):
    rho = spearmanr(trp[c], trp["price"]).statistic
    fila = lda.iloc[k]
    filas_top.append({"topico": k, "probables": ", ".join(legible(w.strip()) for w in str(fila.iloc[0]).split(",")[:6]),
                      "distintivas": ", ".join(legible(w.strip()) for w in str(fila.iloc[1]).split(",")[:6]),
                      "media_chapinero": m_te[c], "media_resto": m_resto[c], "rho_precio": rho})
pd.DataFrame(filas_top).to_csv(D / "ppt_texto_topicos.csv", index=False)

# 5. Limpieza: reparación de la codificación (la ñ y las tildes se perdieron en origen) y corrección con BETO.
rep = pd.read_csv(DIR_TAB / "texto_correcciones_letras_perdidas.csv")
beto = pd.read_csv(DIR_TEXTO / "auditoria_ortografia_beto.csv")
acep = beto[beto["aceptada"]]
json.dump({"reparaciones": int(len(rep)),
           "ejemplos_reparacion": rep.head(6)[["original", "corregido", "df_original"]].values.tolist(),
           "beto_candidatas": int(len(beto)), "beto_aceptadas": int(len(acep)),
           "ejemplos_beto": acep.sort_values("df_correccion", ascending=False)
           .drop_duplicates("correccion").head(6)[["original", "correccion"]].values.tolist(),
           "avisos": n_tot, "terminos": int(len(voc)), "unigramas": int((~voc["bigrama"]).sum()),
           "bigramas": int(voc["bigrama"].sum())},
          open(D / "ppt_texto_limpieza.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(frec[["tipo", "legible", "pct"]].round(1).to_string(index=False))
print(dist[["lado", "legible", "pct_chapinero", "pct_resto", "razon"]].round(2).to_string(index=False))
print(pd.DataFrame(filas_top).round(3).to_string(index=False))
