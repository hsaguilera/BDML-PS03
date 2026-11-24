"""06 · Texto como datos: del título y la descripción de cada aviso a variables para el modelo.

Qué hace (Gentzkow, Kelly y Taddy 2019; Ash y Hansen 2023; Lecture 15 y lectura Texto como datos del curso)
    1. Normalización: quita las máscaras de números de Kaggle ("X"), repara entidades HTML partidas, separa dígitos y
       letras, unifica abreviaturas del dominio (apto, mts, hab, garaje), convierte números escritos a dígitos y repara
       letras perdidas (ñ y vocales tildadas) con una regla de frecuencias del corpus.
    2. Corrección ortográfica con el modelo de lenguaje enmascarado de BETO (por defecto se aplican las 732 correcciones
       aceptadas en la ejecución original, guardadas en texto/auditoria_ortografia_beto.csv).
    3. Firmas de agentes: borra nombres y apellidos de las listas del DNP (ConTexto) solo tras expresiones de contacto.
    4. Huella del texto (MD5) para agrupar duplicados: los avisos con el mismo texto quedan juntos en la validación y
       los de prueba con gemelo en entrenamiento se corrigen en el envío (script 10).
    5. Diccionarios con manejo de negación (8 indicadores de clase y 10 extensiones) y cantidades extraídas del texto
       (área, alcobas, baños, parqueaderos, piso, estrato, antigüedad), que completan los faltantes de Kaggle.
    6. Matriz documento-término (stopwords del DNP y NLTK, stemming Snowball, unigramas y bigramas con df de 5 o más),
       TF-IDF, 50 componentes LSA y tópicos LDA (K elegido por perplejidad fuera de muestra).
    7. Embeddings de BETO congelado (dccuchile/bert-base-spanish-wwm-uncased) y 32 componentes principales.

Salidas
    00_data/processed/texto/: texto_variables.parquet (93 columnas por aviso), dtm_conteos.npz, tfidf.npz,
        dtm_filas.parquet, vocabulario.csv y texto_normalizado.parquet (se regeneran en cada corrida; no se versionan
        por tamaño)
    02_outputs/tables/: texto_correcciones_letras_perdidas, texto_vocabulario_etapas, texto_duplicados,
        texto_diccionarios, texto_cantidades, texto_lda_perplejidad, texto_lda_topicos, texto_informatividad,
        texto_beto_informatividad
    02_outputs/figures/: texto_vocabulario_zipf, texto_lsa_lda, texto_informatividad, texto_mapas, texto_beto

Recálculo
    Por defecto todo se calcula en CPU (unos 15 minutos), salvo tres pasos que leen su resultado guardado: la elección
    de K del LDA (cinco LDA, unos 25 minutos; texto/lda_perplejidad.csv), la corrección ortográfica con BETO (aplica las
    correcciones de texto/auditoria_ortografia_beto.csv) y los componentes de BETO (texto/beto_componentes.parquet).
    Con PS3_RECALCULAR=1 (o =06) también se recalculan: BETO se descarga de Hugging Face (internet) y usa la GPU si
    existe (en CPU los embeddings tardan mucho más). Los embeddings completos (texto/beto_embeddings.npy,
    48.930 × 768, 144 MB) no se versionan: se regeneran así.
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import hashlib
import re
import unicodedata
import warnings
from collections import Counter, defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
from nltk.stem import SnowballStemmer
from scipy.stats import spearmanr
from sklearn.decomposition import PCA, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GroupShuffleSplit

from functions.entorno_ps3 import (ARCHIVO_LOCALIDADES, COLOR_TRAIN, CRS_GEO, CRS_M, DIR_RAW, DIR_TEXTO, exigir,
                                   guardar_figura, guardar_tabla, log, plt, recalcular)

warnings.filterwarnings("ignore")
RECALCULAR = recalcular("06")
SEMILLA = 2026
MODELO_BETO = "dccuchile/bert-base-spanish-wwm-uncased"
COLOR_PRUEBA = "#C44E52"
exigir(ARCHIVO_LOCALIDADES, DIR_TEXTO / "contexto_dnp" / "lista_sw_es.txt", DIR_TEXTO / "stopwords_nltk_es.txt",
       como="Corra antes el script 01; las listas de stopwords vienen en el repositorio.")
if not RECALCULAR:
    exigir(DIR_TEXTO / "auditoria_ortografia_beto.csv", DIR_TEXTO / "beto_componentes.parquet",
           como="Vienen en el repositorio; para regenerarlos con BETO corra este script con PS3_RECALCULAR=06.")

# ------------------------------------------------------------------------------------------------------------------
# 0. Datos: un documento por aviso (título y descripción)
# ------------------------------------------------------------------------------------------------------------------
kag = pd.concat([pd.read_csv(DIR_RAW / "train.csv").assign(muestra="train"),
                 pd.read_csv(DIR_RAW / "test.csv").assign(muestra="test")], ignore_index=True)
kag["precio_mill"] = kag["price"] / 1e6
gdf = gpd.GeoDataFrame(kag, geometry=gpd.points_from_xy(kag["lon"], kag["lat"]), crs=CRS_GEO).to_crs(CRS_M)
gdf["x"], gdf["y"] = gdf.geometry.x, gdf.geometry.y
localidades = gpd.read_file(ARCHIVO_LOCALIDADES).to_crs(CRS_M)
chap = localidades[localidades["localidad"] == "Chapinero"]
EXTENT = gdf.total_bounds + np.array([-1000, -1000, 1000, 1000])
es_tr = (gdf["muestra"] == "train").values

corpus = (gdf["title"].fillna("") + " . " + gdf["description"].fillna("")).astype(str)
n_palabras = corpus.str.split().str.len()
log(f"Documentos: {len(corpus):,} | palabras por documento: mediana {n_palabras.median():.0f}, "
    f"p90 {n_palabras.quantile(.9):.0f}, máximo {n_palabras.max():,}")


# ------------------------------------------------------------------------------------------------------------------
# 1. Normalización
# ------------------------------------------------------------------------------------------------------------------
def sin_tildes(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


NUMEROS = {"un": 1, "una": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
           "ocho": 8, "nueve": 9, "diez": 10}
ORDINALES = {"primer": 1, "primero": 1, "segundo": 2, "tercer": 3, "tercero": 3, "cuarto": 4, "quinto": 5,
             "sexto": 6, "septimo": 7, "octavo": 8, "noveno": 9, "decimo": 10}
MAPA_DOMINIO = [  # (patrón, reemplazo): abreviaturas y variantes del dominio inmobiliario
    (r"\b(?:m|mt|mts)\s*2\b|\bmts\b|\bmetros? cuadrados?\b", " m2 "),
    (r"\baptos?\b|\bapartamentos?\b", " apartamento "),
    (r"\bhabs?\b|\bhabitaciones?\b|\bcuartos\b|\bcuarto\b(?!\s+(?:piso|nivel|util))|\bdormitorios?\b", " alcoba "),
    (r"\balcobas\b", " alcoba "),
    (r"\bparq\b|\bparqueaderos?\b|\bgarajes?\b|\bgarages?\b|\bparqueos?\b", " parqueadero "),
    (r"\badmon\b|\badmin\b|\badmi\b|\badministracion\b", " administracion "),
    (r"\bbanos\b", " bano "),
]
RE_MASCARA = re.compile(r"\bX+\b")                                    # máscaras de números de la competencia
RE_WWW = re.compile(r"\bwww(?:\s+[a-z0-9]+){1,3}?\s+com(?:\s+co)?\b|https?://\S+|\S+@\S+")
RE_HTML = re.compile(r"<[^>]+>|&nbsp;|&amp;|&#?\w+;")
RE_ENT_TILDE = re.compile(r"&([a-zA-Z])(?:tilde|acute|grave|uml|circ);")
RE_ORDINAL = re.compile(r"\b(" + "|".join(ORDINALES) + r")\s+(piso|nivel)\b")
RE_NUMEROS = re.compile(r"\b(" + "|".join(NUMEROS) + r")\b(?=\s+(?:alcoba|bano|parqueadero|piso|nivel|ambientes?))")
FUNCIONALES = {"de", "la", "el", "en", "y", "a", "con", "los", "las", "del", "para", "por", "un", "una", "al", "su", "sus",
               "se", "que", "o", "es", "muy", "mas", "sin", "no"}


def reparar_entidades(t):
    """'ba ntilde o' -> 'bano'; 'balc oacute n' -> 'balcon'; 'de aacute rea' -> 'de area'."""
    pal = t.split()
    out, i = [], 0
    while i < len(pal):
        m = re.fullmatch(r"([a-z])(acute|tilde|grave|uml|circ)", pal[i])
        if not m:
            out.append(pal[i]); i += 1
            continue
        letra = m.group(1)
        izq = out.pop() if out and out[-1].isalpha() and out[-1] not in FUNCIONALES else ""
        der = ""
        if i + 1 < len(pal) and pal[i + 1].isalpha() and (pal[i + 1] not in FUNCIONALES or len(pal[i + 1]) <= 2):
            der = pal[i + 1]; i += 1
        out.append(izq + letra + der)
        i += 1
    return " ".join(out)


def normalizar(texto_):
    t = RE_MASCARA.sub(" ", str(texto_))                             # antes de pasar a minúsculas (las máscaras son "X")
    t = RE_ENT_TILDE.sub(r"\1", t)
    t = RE_HTML.sub(" ", t)
    t = sin_tildes(t.lower())
    t = RE_WWW.sub(" ", t)
    t = re.sub(r"\bbr\b", " ", t)
    t = reparar_entidades(t)
    t = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", t)       # 3alcobas -> 3 alcobas (regex de clase)
    t = RE_ORDINAL.sub(lambda m: f"{m.group(2)} {ORDINALES[m.group(1)]}", t)
    for patron, reemplazo in MAPA_DOMINIO:
        t = re.sub(patron, reemplazo, t)
    t = RE_NUMEROS.sub(lambda m: str(NUMEROS[m.group(1)]), t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


texto = corpus.map(normalizar)

# Regla de las letras perdidas: t -> argmax df(s) entre las palabras s que resultan de insertar n, a, e, i, o o u,
# si df(s) >= max(20, 2 df(t))
PERDIDAS = "naeiou"
PROTEGIDAS = {"vistas", "nos", "com", "star", "sal", "chic", "cer"}   # falsos positivos detectados en la auditoría
df_tok = Counter(w for doc in texto for w in set(re.findall(r"[a-z]{3,}", doc)))
mapa_letras = {}
for t, f in df_tok.items():
    if t in PROTEGIDAS:
        continue
    cand = max({t[:k] + c + t[k:] for k in range(len(t) + 1) for c in PERDIDAS}, key=lambda s: df_tok.get(s, 0))
    if df_tok.get(cand, 0) >= max(20, 2 * f):
        mapa_letras[t] = cand
for t in mapa_letras:                                 # cadenas: balc -> balcn -> balcon
    vistos = {t}
    while mapa_letras[t] in mapa_letras and mapa_letras[t] not in vistos:
        vistos.add(mapa_letras[t])
        mapa_letras[t] = mapa_letras[mapa_letras[t]]
re_letras = re.compile(r"\b(" + "|".join(map(re.escape, sorted(mapa_letras, key=len, reverse=True))) + r")\b")
texto = texto.str.replace(re_letras, lambda m: mapa_letras[m.group(1)], regex=True)


def remapear(t):                                      # después de corregir: habitacin -> habitacion -> alcoba
    for patron, reemplazo in MAPA_DOMINIO:
        t = re.sub(patron, reemplazo, t)
    t = RE_NUMEROS.sub(lambda m: str(NUMEROS[m.group(1)]), t)
    return re.sub(r"\s+", " ", t).strip()


texto = texto.map(remapear)
auditoria_corr = pd.DataFrame({"original": list(mapa_letras), "corregido": list(mapa_letras.values()),
                               "df_original": [df_tok[t] for t in mapa_letras],
                               "df_corregido": [df_tok.get(c, 0) for c in mapa_letras.values()]}
                              ).sort_values("df_original", ascending=False)
guardar_tabla(auditoria_corr, "texto_correcciones_letras_perdidas", index=False)
log(f"Formas corregidas por la regla de letras perdidas: {len(mapa_letras):,}")


# ------------------------------------------------------------------------------------------------------------------
# 2. Corrección ortográfica con el modelo de lenguaje enmascarado de BETO (solo al recalcular; si no, la auditoría)
# ------------------------------------------------------------------------------------------------------------------
def vocabulario_beto(tokenizador):
    """Vocabulario de BETO y formas con tilde de las palabras que en nuestro texto la perdieron (sección 10.3)."""
    voc = tokenizador.get_vocab()
    acentuadas = {}
    for w in sorted((w for w in voc if not w.startswith("##") and w.isalpha()), key=voc.get):
        s = sin_tildes(w)
        if s != w and s not in voc and s not in acentuadas:
            acentuadas[s] = w
    acentuadas.update({"ano": "año", "anos": "años", "bano": "baño", "banos": "baños"})
    return voc, acentuadas


def correccion_beto(texto_):
    """Palabras raras fuera del vocabulario de BETO, candidatas a distancia de Damerau-Levenshtein de 1 o 2, elección
    con el contexto del aviso (MLM) y aceptación conservadora. Devuelve la auditoría de todas las candidatas."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok_beto = AutoTokenizer.from_pretrained(MODELO_BETO)
    voc_beto, acentuadas = vocabulario_beto(tok_beto)
    forma_beto = lambda w: w if w in voc_beto else acentuadas.get(w)  # noqa: E731

    def borrados(w, n):
        res, frente = {w}, {w}
        for _ in range(n):
            frente = {x[:i] + x[i + 1:] for x in frente for i in range(len(x))}
            res |= frente
        return res

    def damerau_levenshtein(a, b):
        m, n = len(a), len(b)
        D = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            D[i][0] = i
        for j in range(n + 1):
            D[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
                if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                    D[i][j] = min(D[i][j], D[i - 2][j - 2] + 1)
        return D[m][n]

    df_pal = Counter(w for doc in texto_ for w in set(re.findall(r"[a-z]{3,}", doc)))
    sospechosas = [w for w, f in df_pal.items() if f < 5 and len(w) >= 4 and forma_beto(w) is None]
    indice = defaultdict(set)
    for v in (w for w, f in df_pal.items() if f >= 20 and forma_beto(w) is not None):
        for b in borrados(v, 2):
            indice[b].add(v)
    candidatas = {}
    for w in sospechosas:
        n = 1 if len(w) < 6 else 2
        c = {v for v in set().union(*(indice.get(b, set()) for b in borrados(w, n))) if damerau_levenshtein(w, v) <= n}
        if c:
            candidatas[w] = c
    contextos = defaultdict(list)
    for pal in texto_.str.split().tolist():
        for w in set(pal) & candidatas.keys():
            if len(contextos[w]) < 2:
                p = pal.index(w)
                ventana = ([acentuadas.get(x, x) for x in pal[max(0, p - 40):p]] + [tok_beto.mask_token]
                           + [acentuadas.get(x, x) for x in pal[p + 1:p + 41]])
                contextos[w].append(" ".join(ventana))
    items = [(w, c) for w, cs in contextos.items() for c in cs]
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    mlm = AutoModelForMaskedLM.from_pretrained(MODELO_BETO).to(dispositivo).eval()
    puntajes = defaultdict(lambda: defaultdict(list))
    with torch.inference_mode(), torch.autocast(device_type=dispositivo, dtype=torch.float16, enabled=dispositivo == "cuda"):
        for i in range(0, len(items), 128):
            lote = items[i:i + 128]
            enc = tok_beto([c for _, c in lote], padding=True, truncation=True, max_length=128, return_tensors="pt").to(dispositivo)
            logp = torch.log_softmax(mlm(**enc).logits.float(), dim=-1)
            for fila, col in (enc["input_ids"] == tok_beto.mask_token_id).nonzero().tolist():
                w = lote[fila][0]
                for v in candidatas[w]:
                    puntajes[w][v].append(logp[fila, col, voc_beto[forma_beto(v)]].item())
    filas = []
    for w, ps in puntajes.items():
        v, s = max(((v, float(np.mean(x))) for v, x in ps.items()), key=lambda t: t[1])
        dist = damerau_levenshtein(w, v)
        solo_plural = w == v + "s" or v == w + "s"
        acepta = (not solo_plural) and (s >= -3 or (s >= -5 and dist == 1))
        filas.append({"original": w, "correccion": v, "log_prob": s, "distancia": dist, "df_original": df_pal[w],
                      "df_correccion": df_pal[v], "aceptada": acepta})
    log(f"BETO: {len(sospechosas):,} palabras sospechosas, {len(candidatas):,} con candidatas, {len(items):,} contextos "
        f"evaluados en {dispositivo}")
    return pd.DataFrame(filas).sort_values("log_prob", ascending=False)


if RECALCULAR:
    auditoria_ort = correccion_beto(texto)
    auditoria_ort.to_csv(DIR_TEXTO / "auditoria_ortografia_beto.csv", index=False)
else:
    auditoria_ort = pd.read_csv(DIR_TEXTO / "auditoria_ortografia_beto.csv")
mapa_ort = dict(auditoria_ort.loc[auditoria_ort["aceptada"], ["original", "correccion"]].values)
re_ort = re.compile(r"\b(" + "|".join(map(re.escape, sorted(mapa_ort, key=len, reverse=True))) + r")\b")
antes_ort = texto.copy()
texto = texto.str.replace(re_ort, lambda m: mapa_ort[m.group(1)], regex=True)
log(f"Correcciones ortográficas de BETO aceptadas: {len(mapa_ort):,} formas | avisos modificados: {(antes_ort != texto).sum():,}")


# ------------------------------------------------------------------------------------------------------------------
# 3. Firmas de agentes (listas de nombres y apellidos de ConTexto, DNP, solo tras expresiones de contacto)
# ------------------------------------------------------------------------------------------------------------------
def cargar_lista(nombre):
    crudo = (DIR_TEXTO / "contexto_dnp" / f"lista_{nombre}.txt").read_bytes()
    for cod in ("utf-8", "latin-1"):
        try:
            txt = crudo.decode(cod)
            break
        except UnicodeDecodeError:
            continue
    return {sin_tildes(linea.strip().lower()) for linea in txt.splitlines() if linea.strip()}


NOMBRES_DNP = (cargar_lista("nombres_ambos") | cargar_lista("nombres_hombres") | cargar_lista("nombres_mujeres")
               | cargar_lista("apellidos"))
NOMBRES_DNP -= cargar_lista("sw_es") | FUNCIONALES | {"casa", "sala", "mayor", "hermoso", "linda", "bella", "real", "alta",
                                                       "chico", "cabrera", "rosales", "retiro", "andino", "castillo", "prado"}
RE_CONTACTO = re.compile(r"\b(contactame|contactenos|contacteme|contactarse con|contactar a|comunicate con|comuniquese con|"
                         r"comunicarse con|llamar a|llamar al|llama a|al celular|asesora?|agente|informes con|pregunta por|"
                         r"preguntar por|escribir a|atendido por)\b((?:\s+[a-z]+){1,4})")


def quitar_firma(t):
    def limpiar(m):
        return " " + m.group(1) + " " + " ".join(w for w in m.group(2).split() if w not in NOMBRES_DNP) + " "
    return RE_CONTACTO.sub(limpiar, t)


antes = texto.copy()
texto = texto.map(quitar_firma).str.replace(r"\s+", " ", regex=True).str.strip()
log(f"Avisos con firma de agente eliminada: {(antes != texto).sum():,}")

# ------------------------------------------------------------------------------------------------------------------
# 4. Duplicados de texto (huella MD5 del texto normalizado)
# ------------------------------------------------------------------------------------------------------------------
huella = texto.map(lambda s: hashlib.md5(s.encode()).hexdigest())
gdf["grupo_texto"] = huella.values
huellas_tr = set(huella[es_tr])
guardar_tabla(pd.Series({
    "Documentos": len(texto),
    "Textos distintos (tras normalizar)": huella.nunique(),
    "% de avisos de entrenamiento con texto repetido": 100 * huella[es_tr].duplicated(keep=False).mean(),
    "% de avisos de prueba con texto repetido (dentro de prueba)": 100 * huella[~es_tr].duplicated(keep=False).mean(),
    "% de avisos de prueba con gemelo exacto en entrenamiento": 100 * huella[~es_tr].isin(huellas_tr).mean(),
}, name="valor").rename_axis("indicador").to_frame(), "texto_duplicados", decimales=2)

# ------------------------------------------------------------------------------------------------------------------
# 5. Diccionarios con negación y cantidades extraídas del texto
# ------------------------------------------------------------------------------------------------------------------
NEGACION = r"\b(?:sin|no\s+(?:tiene|cuenta\s+con|dispone\s+de|hay|incluye))\s+(?:(?:un|una|el|la|los|las)\s+)?(?:{p})"
CARACTERISTICAS = {
    # Ocho indicadores de la clase
    "parqueadero": r"parqueadero", "piscina": r"piscina|jacuzzi", "ascensor": r"ascensor",
    "terraza_balcon": r"terraza|balcon", "vigilancia": r"vigilancia|seguridad|porteria|portero",
    "remodelado": r"remodelad|remodelo\b|renovad|(?<!requiere )(?<!necesita )(?<!para )remodelacion",
    "cocina_integral": r"cocinas?\s+integral",
    "zona_social": r"zonas?\s+social(?:es)?|salon(?:es)?\s+(?:social(?:es)?|comunal(?:es)?|de\s+reuniones)",
    # Extensiones
    "chimenea": r"chimenea", "vista": r"vista|panoramic", "estudio": r"estudio", "apartaestudio": r"apartaestudio|aparta estudio|loft",
    "duplex_penthouse": r"duplex|penthouse|pent house", "exterior": r"exterior", "gimnasio": r"gimnasio|gym",
    "deposito": r"deposito|bodega", "conjunto": r"conjunto cerrado|conjunto", "nuevo_estrenar": r"para estrenar|estrenar|sobre planos|proyecto nuevo",
}
for nombre, patron in CARACTERISTICAS.items():
    limpio = texto.str.replace(NEGACION.format(p=patron), " ", regex=True)
    gdf[f"txt_{nombre}"] = limpio.str.contains(rf"\b(?:{patron})", regex=True).astype(int).values
gdf["txt_para_remodelar"] = texto.str.contains(r"para remodelar|requiere remodela|necesita remodela|para reformar").astype(int).values
VARS_DICC = [c for c in gdf.columns if c.startswith("txt_")]
negadas = {n: int(texto.str.contains(rf"\b(?:{p})").sum() - gdf[f"txt_{n}"].sum()) for n, p in CARACTERISTICAS.items()}
tab_dicc = pd.DataFrame({"% entrenamiento": 100 * gdf.loc[es_tr, VARS_DICC].mean(),
                         "% prueba": 100 * gdf.loc[~es_tr, VARS_DICC].mean(),
                         "menciones negadas descartadas": pd.Series({f"txt_{k}": v for k, v in negadas.items()})}).fillna(0)
guardar_tabla(tab_dicc.rename_axis("indicador").sort_values("% prueba", ascending=False), "texto_diccionarios", decimales=2)


def extraer(serie, patron, minimo, maximo):
    """Primer número capturado por el patrón que cae en el rango plausible [minimo, maximo]; NA si no hay."""
    encontrados = serie.str.findall(patron)

    def elegir(lista):
        for x in lista:
            x = x[0] if isinstance(x, tuple) else x
            try:
                v = float(x)
            except ValueError:
                continue
            if minimo <= v <= maximo:
                return v
        return np.nan
    return encontrados.map(elegir)


area_expl = extraer(texto, r"\barea\s*(?:total|construida|privada|util|de)?\s*(?:de|es)?\s*(?:aprox\w*)?\s*(\d{2,4})\b", 20, 1000)
area_gen = extraer(texto, r"\b(\d{2,4})\s*m2\b", 20, 1000)
EXTRAIDAS = {
    "area_texto": area_expl.fillna(area_gen),
    "alcobas_texto": extraer(texto, r"\b(\d{1,2})\s*alcoba\b", 1, 10),
    "banos_texto": extraer(texto, r"\b(\d{1,2})\s*bano\b", 1, 10),
    "parqueaderos_texto": extraer(texto, r"\b(\d{1,2})\s*parqueadero\b", 1, 6),
    "piso_texto": extraer(texto, r"\b(?:piso|nivel)\s*(\d{1,2})\b(?!\s*(?:alcoba|bano|parqueadero|m2|ambiente))", 1, 40),
    "estrato_texto": extraer(texto, r"\bestrato\s*(\d)\b", 1, 6),
    "antiguedad_texto": extraer(texto, r"\b(\d{1,2})\s*anos?\s*(?:de\s*)?(?:construid|antig|de uso)", 0, 80),
}
for k, v in EXTRAIDAS.items():
    gdf[k] = v.values
VARS_NUM_TXT = list(EXTRAIDAS)

# Imputación con el texto (primero el dato estructurado, luego el texto), con la fuente de cada valor
area_estr = gdf["surface_covered"].fillna(gdf["surface_total"]).where(lambda a: a.between(15, 1000))
gdf["area_imp"] = area_estr.fillna(gdf["area_texto"])
gdf["area_fuente"] = np.select([area_estr.notna(), gdf["area_texto"].notna()], ["estructurado", "texto"], "faltante")
gdf["banos_imp"] = gdf["bathrooms"].fillna(gdf["banos_texto"])
gdf["banos_fuente"] = np.select([gdf["bathrooms"].notna(), gdf["banos_texto"].notna()], ["estructurado", "texto"], "faltante")
gdf["parqueaderos_imp"] = gdf["parqueaderos_texto"].fillna(gdf["txt_parqueadero"])   # mención sin número -> 1; sin mención -> 0


def concordancia(txt, estr, tol=0.10):
    ok = txt.notna() & estr.notna()
    return ok.sum(), 100 * ((txt[ok] - estr[ok]).abs() / estr[ok] <= tol).mean() if ok.any() else np.nan


auditoria = pd.DataFrame([{"cantidad": var, "avisos con ambos datos": concordancia(gdf[var], estr, tol)[0],
                           "% que coincide con Kaggle": concordancia(gdf[var], estr, tol)[1]}
                          for var, estr, tol in [("area_texto", area_estr, .10), ("banos_texto", gdf["bathrooms"], 0),
                                                 ("alcobas_texto", gdf["bedrooms"], 0)]]).set_index("cantidad")
cobertura = pd.DataFrame({"% entrenamiento": 100 * gdf.loc[es_tr, VARS_NUM_TXT].notna().mean(),
                          "% prueba": 100 * gdf.loc[~es_tr, VARS_NUM_TXT].notna().mean(),
                          "mediana": gdf[VARS_NUM_TXT].median()}).join(auditoria)
guardar_tabla(cobertura.rename_axis("cantidad"), "texto_cantidades", decimales=2)

# ------------------------------------------------------------------------------------------------------------------
# 6. Representación: tokens, stopwords, stemming, n-gramas, DTM, TF-IDF, LSA y LDA
# ------------------------------------------------------------------------------------------------------------------
STOP_NLTK = (DIR_TEXTO / "stopwords_nltk_es.txt").read_text(encoding="utf-8").split()   # copia de nltk stopwords("spanish")
STOP = (cargar_lista("sw_es") | {sin_tildes(w) for w in STOP_NLTK} | {"ser", "estar", "haber", "tener"}) \
       - {"no", "sin", "ni", "estado", "estados"}
stemmer = SnowballStemmer("spanish")
DOMINIO = ["vender", "venta", "vende", "vendo", "inmobiliaria", "inmueble", "propiedad", "ofrecer", "oferta", "ofrece",
           "informacion", "contactar", "contacto", "contar", "cuenta", "constar", "consta", "bogota"]
RAICES_DOMINIO = {stemmer.stem(w) for w in DOMINIO}
_raiz = {}


def tokens(doc):
    out = []
    for w in re.findall(r"[a-z]{2,}|\b\d{1,2}\b", doc):
        if w in STOP:
            continue
        if w not in _raiz:
            _raiz[w] = w if w.isdigit() else stemmer.stem(w)
        if _raiz[w] in RAICES_DOMINIO:
            continue
        out.append(_raiz[w])
    return out


crudos = Counter(w for doc in texto for w in set(re.findall(r"[a-z]{2,}|\b\d{1,2}\b", doc)))
sin_stop = {w: f for w, f in crudos.items() if w not in STOP}
raices = {(w if w.isdigit() else stemmer.stem(w)) for w in sin_stop} - RAICES_DOMINIO
vectorizador = CountVectorizer(tokenizer=tokens, lowercase=False, token_pattern=None, ngram_range=(1, 2), min_df=5, max_df=0.9)
C = vectorizador.fit_transform(texto).tocsr()
vocab = np.array(vectorizador.get_feature_names_out())
es_bigrama = np.char.count(vocab.astype(str), " ") > 0
guardar_tabla(pd.Series({
    "1. Términos distintos (palabras y números de 1 o 2 dígitos, tras normalizar)": len(crudos),
    "2. Sin stopwords (DNP y NLTK)": len(sin_stop),
    "3. Con stemming Snowball y sin palabras de dominio": len(raices),
    "4. Unigramas y bigramas con df de 5 o más y df/N menor que 0,9": C.shape[1],
    "   de ellos, unigramas": int((~es_bigrama).sum()),
    "   de ellos, bigramas": int(es_bigrama.sum()),
}, name="terminos").rename_axis("etapa").to_frame(), "texto_vocabulario_etapas")
log(f"Matriz documento-término C: {C.shape[0]:,} × {C.shape[1]:,}")

df_term = np.asarray((C > 0).sum(axis=0)).ravel()
orden = np.argsort(-df_term)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].loglog(np.arange(1, len(orden) + 1), np.sort(df_term)[::-1], color="#4C72B0")
axes[0].set(title="Frecuencia documental de los términos (ley de Zipf)", xlabel="Rango del término",
            ylabel="Documentos que lo contienen")
top = pd.Series(df_term[orden[:25]], index=vocab[orden[:25]])[::-1]
axes[1].barh(top.index, 100 * top.values / C.shape[0], color="#4C72B0")
axes[1].set(title="25 términos más frecuentes (raíces)", xlabel="% de avisos")
fig.tight_layout()
guardar_figura(fig, "texto_vocabulario_zipf")

tfidf = TfidfTransformer(sublinear_tf=True)
X_tfidf = tfidf.fit_transform(C).tocsr()

K_LSA = 50
svd = TruncatedSVD(n_components=K_LSA, random_state=SEMILLA)
Z = svd.fit_transform(X_tfidf)
VARS_LSA = [f"lsa_{k:02d}" for k in range(K_LSA)]
gdf[VARS_LSA] = Z
ve_lsa = svd.singular_values_ ** 2 / sp.linalg.norm(X_tfidf, "fro") ** 2

uni = ~es_bigrama
C_uni = C[:, uni]
vocab_uni = vocab[uni]
a_idx, b_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEMILLA).split(C_uni, groups=gdf["grupo_texto"]))
ARCHIVO_PERPLEJIDAD = DIR_TEXTO / "lda_perplejidad.csv"
if RECALCULAR or not ARCHIVO_PERPLEJIDAD.exists():
    # Elección de K: cinco LDA sobre el 80 % de los avisos (unos 25 minutos en CPU); por defecto se lee el resultado
    perplejidad = {}
    for K in (5, 10, 15, 20, 25):                  # K del LDA por perplejidad en un 20 % de avisos retenidos
        m = LatentDirichletAllocation(n_components=K, learning_method="online", batch_size=2048, max_iter=15,
                                      random_state=SEMILLA, n_jobs=4).fit(C_uni[a_idx])
        perplejidad[K] = m.perplexity(C_uni[b_idx])
        log(f"LDA con K = {K}: perplejidad fuera de muestra {perplejidad[K]:.1f}")
    perplejidad = pd.Series(perplejidad, name="perplejidad_fuera_de_muestra").rename_axis("K")
    perplejidad.to_frame().to_csv(ARCHIVO_PERPLEJIDAD)
else:
    perplejidad = pd.read_csv(ARCHIVO_PERPLEJIDAD, index_col="K")["perplejidad_fuera_de_muestra"]
    log("LDA: se usa la perplejidad guardada de K = 5 a 25 (PS3_RECALCULAR=06 la recalcula)")
K_LDA = int(perplejidad.idxmin())
guardar_tabla(perplejidad.to_frame(), "texto_lda_perplejidad", decimales=3)

lda = LatentDirichletAllocation(n_components=K_LDA, learning_method="online", batch_size=2048, max_iter=15,
                                random_state=SEMILLA, n_jobs=4)
theta = lda.fit_transform(C_uni)
VARS_LDA = [f"topico_{k:02d}" for k in range(K_LDA)]
gdf[VARS_LDA] = theta
beta = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
m_doc = np.asarray(C_uni.sum(axis=1)).ravel()
con = m_doc > 0
p_bar = np.asarray((sp.diags(1 / m_doc[con]) @ C_uni[con]).mean(axis=0)).ravel()
lift = beta / np.maximum(p_bar, 1e-12)            # lift_kj = beta_kj / p̄_j (Gentzkow et al., p. 556)
topicos = pd.DataFrame({f"Tópico {k}": [", ".join(vocab_uni[np.argsort(-beta[k])[:8]]),
                                         ", ".join(vocab_uni[np.argsort(-lift[k] * (p_bar > 1e-4))[:8]])]
                        for k in range(K_LDA)}, index=["más probables (β)", "más distintivas (lift)"]).T
guardar_tabla(topicos.rename_axis("topico"), "texto_lda_topicos")
log(f"LDA: K = {K_LDA} tópicos")

fig, axes = plt.subplots(1, 3, figsize=(19, 4.5))
axes[0].plot(np.arange(1, K_LSA + 1), 100 * np.cumsum(ve_lsa), "o-", ms=3, color="#4C72B0")
axes[0].set(title="LSA: energía acumulada capturada (σ²ₖ / ‖X‖²_F)", xlabel="Número de componentes K", ylabel="%")
axes[1].plot(perplejidad.index, perplejidad.values, "o-", color="#8172B3")
axes[1].axvline(K_LDA, color="gray", ls="--", lw=1)
axes[1].set(title="LDA: perplejidad fuera de muestra según K", xlabel="Número de tópicos K", ylabel="Perplejidad")
prev = pd.DataFrame({"Entrenamiento": theta[es_tr].mean(axis=0), "Prueba": theta[~es_tr].mean(axis=0)},
                    index=[f"T{k}" for k in range(K_LDA)]) * 100
prev.plot.bar(ax=axes[2], color=[COLOR_TRAIN, COLOR_PRUEBA], width=0.8)
axes[2].set(title="LDA: peso promedio de cada tópico", ylabel="% del texto", xlabel="")
axes[2].tick_params(axis="x", rotation=0)
axes[2].legend(frameon=False)
fig.tight_layout()
guardar_figura(fig, "texto_lsa_lda")

# ------------------------------------------------------------------------------------------------------------------
# Guardado de la base de texto (la usa functions/flujo_final.py)
# ------------------------------------------------------------------------------------------------------------------
COLS_GUARDAR = (["property_id", "grupo_texto"] + VARS_DICC + VARS_NUM_TXT
                + ["area_imp", "area_fuente", "banos_imp", "banos_fuente", "parqueaderos_imp"] + VARS_LDA + VARS_LSA)
pd.DataFrame(gdf[COLS_GUARDAR]).to_parquet(DIR_TEXTO / "texto_variables.parquet", index=False)
sp.save_npz(DIR_TEXTO / "dtm_conteos.npz", C.tocsr())
sp.save_npz(DIR_TEXTO / "tfidf.npz", X_tfidf.tocsr())
pd.DataFrame({"termino": vocab, "bigrama": es_bigrama, "df": df_term, "idf": tfidf.idf_}).to_csv(
    DIR_TEXTO / "vocabulario.csv", index=False)
gdf[["property_id"]].to_parquet(DIR_TEXTO / "dtm_filas.parquet", index=False)
# Texto normalizado de cada aviso (lo leen las láminas de texto como datos, script 18; al recalcular, la sección 7 agrega
# la versión con tildes restauradas que entra a BETO)
pd.DataFrame({"property_id": gdf["property_id"].values, "texto_normalizado": texto.values}).to_parquet(
    DIR_TEXTO / "texto_normalizado.parquet", index=False)
log(f"Base de texto guardada en {DIR_TEXTO.as_posix()}/: {len(COLS_GUARDAR) - 2} variables para {len(gdf):,} avisos")


# ------------------------------------------------------------------------------------------------------------------
# Mapas e informatividad de las variables de texto (descriptivo)
# ------------------------------------------------------------------------------------------------------------------
def puntomapa(ax, d, var, titulo, cmap="viridis", robusto=True, s=2.5):
    """Un punto por coordenada exacta con el promedio de sus avisos, en orden aleatorio; contorno de Chapinero."""
    u = d.groupby(["x", "y"], as_index=False).agg(valor=(var, "mean")).dropna(subset=["valor"])
    u = u.sample(frac=1, random_state=SEMILLA)
    kw = dict(vmin=u["valor"].quantile(.02), vmax=u["valor"].quantile(.98)) if robusto else {}
    sc = ax.scatter(u["x"], u["y"], c=u["valor"], s=s, cmap=cmap, linewidths=0, **kw)
    localidades.boundary.plot(ax=ax, color="black", lw=0.3, alpha=0.5)
    chap.boundary.plot(ax=ax, color="black", lw=1.6)
    ax.set_xlim(EXTENT[0], EXTENT[2]); ax.set_ylim(EXTENT[1], EXTENT[3]); ax.set_axis_off()
    plt.colorbar(sc, ax=ax, shrink=0.55)
    ax.set_title(f"{titulo}\n({len(u):,} coordenadas exactas)", fontsize=10)


ETIQUETAS = {**{f"txt_{k}": f"Menciona: {k.replace('_', ' ')}" for k in list(CARACTERISTICAS) + ["para_remodelar"]},
             "area_texto": "Área extraída del texto", "alcobas_texto": "Alcobas (texto)", "banos_texto": "Baños (texto)",
             "parqueaderos_texto": "Parqueaderos (texto)", "piso_texto": "Piso (texto)", "estrato_texto": "Estrato (texto)",
             "antiguedad_texto": "Antigüedad (texto)", "area_imp": "Área imputada (estructura y luego texto)",
             "banos_imp": "Baños imputados", "parqueaderos_imp": "Parqueaderos imputados",
             **{f"topico_{k:02d}": f"Tópico LDA {k}" for k in range(K_LDA)}, **{f"lsa_{k:02d}": f"Componente LSA {k}" for k in range(10)}}
gdf["precio_m2_imp"] = (gdf["precio_mill"] / gdf["area_imp"]).where(gdf["area_imp"].between(15, 1000))
tr_t = gdf[es_tr]
VARS_TXT = VARS_DICC + VARS_NUM_TXT + ["area_imp", "banos_imp", "parqueaderos_imp"] + VARS_LDA + VARS_LSA[:10]
info = pd.DataFrame({
    "ρ precio": [spearmanr(tr_t[v], tr_t["precio_mill"], nan_policy="omit")[0] for v in VARS_TXT],
    "ρ precio/m²": [spearmanr(tr_t[v], tr_t["precio_m2_imp"], nan_policy="omit")[0] for v in VARS_TXT],
    "n entrenamiento": [int(tr_t[v].notna().sum()) for v in VARS_TXT],
    "d (prueba − entrenamiento)": [(gdf.loc[~es_tr, v].mean() - tr_t[v].mean()) / tr_t[v].std() for v in VARS_TXT],
}, index=VARS_TXT)
guardar_tabla(info.assign(nombre=[ETIQUETAS.get(v, v) for v in info.index]).rename_axis("variable"), "texto_informatividad")

top = info.reindex(info["ρ precio"].abs().sort_values(ascending=False).index[:20])[::-1]
fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
axes[0].barh([ETIQUETAS.get(v, v) for v in top.index], top["ρ precio"], color=np.where(top["ρ precio"] > 0, "#55A868", "#C44E52"))
axes[0].axvline(0, color="black", lw=0.8)
axes[0].set(title="Correlación de Spearman con el precio (entrenamiento)", xlabel="ρ")
axes[1].barh([ETIQUETAS.get(v, v) for v in top.index], top["d (prueba − entrenamiento)"], color="#8172B3")
axes[1].axvline(0, color="black", lw=0.8)
for v in (-0.5, 0.5):
    axes[1].axvline(v, color="gray", ls="--", lw=0.8)
axes[1].set(title="Diferencia estandarizada prueba − entrenamiento", xlabel="d")
fig.suptitle("Las 20 variables de texto más correlacionadas con el precio", fontweight="bold")
fig.tight_layout()
guardar_figura(fig, "texto_informatividad")

tmax = int(np.argmax(np.abs([spearmanr(tr_t[f"topico_{k:02d}"], tr_t["precio_mill"])[0] for k in range(K_LDA)])))
gdf["area_desde_texto"] = (gdf["area_fuente"] == "texto").astype(float)
fig, axes = plt.subplots(1, 3, figsize=(20, 8))
puntomapa(axes[0], gdf, "area_desde_texto", "Proporción de avisos cuya área viene del texto", "PuRd", robusto=False)
puntomapa(axes[1], gdf, "area_imp", "Área imputada (estructura y luego texto, m²)", "cividis")
puntomapa(axes[2], gdf, f"topico_{tmax:02d}", f"Peso del tópico LDA {tmax} (el más ligado al precio)", "viridis")
fig.suptitle("Variables construidas desde el texto, en la coordenada exacta de cada aviso", fontweight="bold", fontsize=13)
fig.tight_layout()
guardar_figura(fig, "texto_mapas")

# ------------------------------------------------------------------------------------------------------------------
# 7. BETO congelado: embeddings (media de la última capa) y 32 componentes principales
# ------------------------------------------------------------------------------------------------------------------
VARS_BETO = [f"beto_pc_{k:02d}" for k in range(32)]
if RECALCULAR:
    import torch
    from transformers import AutoModel, AutoTokenizer

    MAX_TOKENS = 256
    tok = AutoTokenizer.from_pretrained(MODELO_BETO)
    _, acentuadas = vocabulario_beto(tok)           # el texto no se traduce: se restauran las tildes que BETO conoce
    texto_beto = texto.str.replace(r"\b[a-z]+\b", lambda m: acentuadas.get(m.group(0), m.group(0)), regex=True)
    pd.DataFrame({"property_id": gdf["property_id"].values, "texto_normalizado": texto.values,
                  "texto_beto": texto_beto.values}).to_parquet(DIR_TEXTO / "texto_normalizado.parquet", index=False)
    longitudes = np.array([len(x) for x in tok(texto_beto.tolist(), add_special_tokens=True, truncation=False)["input_ids"]])
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    beto = AutoModel.from_pretrained(MODELO_BETO).to(dispositivo).eval()
    docs = texto_beto.tolist()
    orden_lon = np.argsort(longitudes)              # lotes de longitud parecida: menos relleno
    E = np.zeros((len(docs), beto.config.hidden_size), dtype=np.float32)
    with torch.inference_mode(), torch.autocast(device_type=dispositivo, dtype=torch.float16, enabled=dispositivo == "cuda"):
        for i in range(0, len(docs), 128):
            lote = orden_lon[i:i + 128]
            enc = tok([docs[j] for j in lote], padding=True, truncation=True, max_length=MAX_TOKENS,
                      return_tensors="pt").to(dispositivo)
            H = beto(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(H.dtype)
            E[lote] = ((H * m).sum(1) / m.sum(1)).float().cpu().numpy()     # mean pooling
    np.save(DIR_TEXTO / "beto_embeddings.npy", E)
    log(f"Embeddings de BETO: {E.shape[0]:,} × {E.shape[1]} en {dispositivo} | truncados a {MAX_TOKENS} sub-palabras: "
        f"{100 * (longitudes > MAX_TOKENS).mean():.1f} %")
    pca_beto = PCA(n_components=32, random_state=SEMILLA)
    P = pca_beto.fit_transform(E)
    pd.concat([gdf[["property_id"]].reset_index(drop=True), pd.DataFrame(P, columns=VARS_BETO)], axis=1).to_parquet(
        DIR_TEXTO / "beto_componentes.parquet", index=False)
    info_beto = pd.DataFrame({
        "% varianza": 100 * pca_beto.explained_variance_ratio_,
        "ρ precio": [spearmanr(P[es_tr, k], gdf.loc[es_tr, "precio_mill"])[0] for k in range(32)],
        "ρ precio/m²": [spearmanr(P[es_tr, k], gdf.loc[es_tr, "precio_m2_imp"], nan_policy="omit")[0] for k in range(32)],
        "d (prueba − entrenamiento)": (P[~es_tr].mean(0) - P[es_tr].mean(0)) / P[es_tr].std(0),
    }, index=VARS_BETO).assign(nombre=[f"Componente BETO {k}" for k in range(32)])
    info_beto.to_csv(DIR_TEXTO / "beto_informatividad.csv")
else:
    comp = pd.read_parquet(DIR_TEXTO / "beto_componentes.parquet").set_index("property_id").loc[gdf["property_id"].values]
    P = comp[VARS_BETO].to_numpy()
    info_beto = pd.read_csv(DIR_TEXTO / "beto_informatividad.csv", index_col=0)
    log("BETO: se usan los 32 componentes guardados (PS3_RECALCULAR=06 recalcula los embeddings)")
guardar_tabla(info_beto.rename_axis("componente"), "texto_beto_informatividad")

fig, axes = plt.subplots(1, 3, figsize=(19, 4.5))
axes[0].plot(np.arange(1, 33), np.cumsum(info_beto["% varianza"]), "o-", ms=3, color="#4C72B0")
axes[0].set(title="BETO: varianza acumulada (PCA)", xlabel="Componentes", ylabel="%")
axes[1].bar(np.arange(32), info_beto["ρ precio"], color=np.where(info_beto["ρ precio"] > 0, "#55A868", "#C44E52"))
axes[1].set(title="Correlación de Spearman con el precio (entrenamiento)", xlabel="Componente", ylabel="ρ")
gdf["beto_pc_mas_ligado"] = P[:, int(np.argmax(np.abs(info_beto["ρ precio"].values)))]
puntomapa(axes[2], gdf, "beto_pc_mas_ligado", "Componente BETO más ligado al precio", "coolwarm")
fig.tight_layout()
guardar_figura(fig, "texto_beto")
log("Fin de texto como datos")
