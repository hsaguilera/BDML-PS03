"""Etapas 2 a 4 reescritas para un público no experto: algoritmo, validación, pronóstico, negocio y fuentes.

Reemplaza las láminas 24 a 45 de la versión anterior (de «Cada árbol hace preguntas de sí o no sobre el aviso» a las
fuentes). No hay apéndice: lo que servía de las láminas 36 a 44 quedó en el cuerpo.

Orden (16 láminas):
  Etapa 2. Algoritmo: qué recibe y qué devuelve; un árbol de decisión; cómo elige cada pregunta; boosting; función
  objetivo; hiperparámetros; número de árboles (parada temprana); qué mueve la predicción (SHAP).
  Etapa 3. Validación: qué es validar y por qué el azar engaña; seis formas de esconder avisos; resultado y qué decidió
  cada validación.
  Etapa 4. Pronóstico y negocio: del modelo al archivo de Kaggle; dónde se equivoca; la trampa de Zillow.
  Recomendación. Fuentes.

Cifras: 00_data/processed/laminas/*.csv|json (generados por los scripts 12 a 19 y 24 a 27 y 28_datos_laminas_etapas2a4.py), 00_data/processed/modelos/cv_clase_res_*.csv,
00_data/processed/modelos/rk_knndm_res_lightgbm_con.csv y 00_data/processed/laminas/hiperparametros_lightgbm.csv. Las comparaciones de
la pérdida y de las hojas vienen de la bitácora del proyecto (pasos 11 y 12) y se escriben tal cual. Solo lee; no cambia datos ni
modelos. No modifica laminas_comunes.py ni ppt_lib.py: importa de ellos.

Uso: from laminas_etapas2a4 import etapas2a4; laminas = etapas2a4(deck)  # lista de (lámina, notas)
"""
import json
import math

import numpy as np
import pandas as pd
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

from functions.laminas_comunes import D, IMG, STORES, n
from functions.ppt_lib import (_ejes, _estilo_grafico, _lineas, barras, circulo, etiquetas_puntos, forma, imagen, linea, lineas_xy,
                     nodo, ocultar_leyenda, rgb, tabla, tarjeta, texto, waffle)

IMG2 = IMG / "etapas2a4"
ETQ2, ETQ3, ETQ4 = "Etapa 2. Algoritmo", "Etapa 3. Validación", "Etapa 4. Pronóstico y negocio"
SEC2, SEC3, SEC4 = "Algoritmo", "Validación", "Pronóstico y negocio"
MENOS = '0;"−"0'
FIN = {}  # y final del contenido principal de cada lámina (control de calidad)


def _leer(nombre):
    return json.load(open(D / nombre, encoding="utf-8"))


# ------------------------------------------------------------------------------------------------------------------
# Datos
# ------------------------------------------------------------------------------------------------------------------
E = _leer("etapas2a4.json")
TRES = _leer("ppt_tres_avisos.json")
ARBOL = _leer("ppt_arbol2.json")
EJB = _leer("ppt_ejemplo_boosting.json")
BO = _leer("ppt_bootstrap2.json")
V2 = _leer("ppt_varios2.json")
HIPER = pd.read_csv(D / "hiperparametros_lightgbm.csv").set_index("parametro")["valor"]
ENV = pd.read_csv(D / "envios_equipo.csv").set_index("archivo")
CLAS = pd.read_csv(D / "clasificacion.csv")
CURVA = pd.read_csv(D / "ppt_curva_aprendizaje.csv")
HIST = pd.read_csv(D / "ppt_histograma_precios.csv")
CAL = pd.read_csv(D / "ppt_calibracion2.csv")
LADOS = pd.read_csv(D / "ppt_lados.csv")
LQ = pd.read_csv(D / "ppt_lados_quintil.csv")
COMP = pd.read_csv(D / "ppt_compras.csv")
GP = pd.read_csv(D / "ppt_ganancia_precision.csv")
REG = pd.read_csv(D / "ppt_regla_rango2.csv").set_index("cat")
SHAP_IMP = pd.read_csv(D / "shap_importancia.csv", index_col=0).iloc[:, 0]
SHAP_BLQ = pd.read_csv(D / "shap_bloques.csv", index_col=0).iloc[:, 0]
CV_CON = pd.read_csv(STORES / "cv_clase_res_lightgbm_con.csv").set_index("esquema")
CV_SIN = pd.read_csv(STORES / "cv_clase_res_lightgbm_sin.csv").set_index("esquema")
CV_FAM = pd.read_csv(D / "cv_familias.csv")
RK = pd.read_csv(STORES / "rk_knndm_res_lightgbm_con.csv").set_index("variante").loc["sin kriging"]

P01 = ENV.loc["envio_01_lightgbm_con_catastro", "privado"]
U01 = ENV.loc["envio_01_lightgbm_con_catastro", "publico"]
P02 = ENV.loc["envio_02_lightgbm_con_catastro_x108", "privado"]
U02 = ENV.loc["envio_02_lightgbm_con_catastro_x108", "publico"]
CH = "Chapinero (imitando el taller)"
MAE = {"aleatoria": CV_CON.loc["aleatoria", "MAE"], "bloques": CV_CON.loc["bloques", "MAE"],
       "buffer": CV_CON.loc["bloques_buffer", "MAE"], "clusters": CV_CON.loc["clusters", "MAE"],
       "knndm": RK["MAE kNNDM"], "chapinero": CV_CON.loc[CH, "MAE"]}
assert abs(MAE["chapinero"] - BO["lightgbm"]["mae"]) < 1e-6
N_ARB = int(HIPER["n_estimators"])
FINAL = E["final"]
PF = E["parada_final"]
PL = E["pliegues"]
DV = E["distancia_vecino"]
LADO = {(r.esquema, r.version): r for r in LADOS.itertuples()}
L_CH, L_CH0 = LADO[("Chapinero fuera", "Con ajuste x1,08")], LADO[("Chapinero fuera", "Sin ajuste")]
L_KN = LADO[("kNNDM", "Con ajuste x1,08")]
REJ_B = pd.DataFrame(E["rejilla_b"]).set_index("b")


def _fam(f, esq):
    r = CV_FAM[(CV_FAM["familia"] == f) & (CV_FAM["esquema"] == esq)]
    return float(r["mae"].iloc[0]) if len(r) else None


# ------------------------------------------------------------------------------------------------------------------
# Utilidades de diseño
# ------------------------------------------------------------------------------------------------------------------
def _nueva(deck, titulo, etiqueta, seccion, oscura=False):
    """Lámina nueva con el título en una sola línea: 26 puntos si cabe; si no, 24 o 22."""
    tam = next((t for t in (26, 24, 22) if _lineas(titulo, 12.1, t) == 1), 22)
    return deck.nueva(titulo, etiqueta=etiqueta, seccion=seccion, size_titulo=tam, oscura=oscura)


def _alto_celda(t, ancho, size):
    return _lineas(str(t), max(ancho - 0.17, 0.3) * 0.97, size) * 1.21 * size / 72 + 0.08


def _tabla(s, x, y, filas, anchos, size=12, minimo=0.3, colores_col=None):
    """Tabla con altos de fila medidos con Calibri para que nada se encime. Devuelve la y final."""
    gf = tabla(s, x, y, sum(anchos), filas, anchos, size=size, alto_fila=minimo, colores_col=colores_col)
    altos = [max(minimo, max(_alto_celda(c[0] if isinstance(c, tuple) else c, a, size) for c, a in zip(f, anchos)))
             for f in filas]
    for i, h in enumerate(altos):
        gf.table.rows[i].height = Inches(h)
    gf.height = Inches(sum(altos))
    return y + sum(altos)


def _formula(s, nombre, x, y, escala=0.75, alt=""):
    """Imagen de fórmula a `escala` de su tamaño (26 puntos a 300 dpi): 0,75 deja la letra en 19,5 puntos."""
    from PIL import Image
    w_px, h_px = Image.open(IMG2 / f"{nombre}.png").size
    w, h = w_px / 300 * escala, h_px / 300 * escala
    imagen(s, IMG2 / f"{nombre}.png", x, y, w=w, alt=alt or nombre)
    return w, h


# ------------------------------------------------------------------------------------------------------------------
# Etapa 2. Algoritmo
# ------------------------------------------------------------------------------------------------------------------
ETQ2G = "Etapa 2. Cómo funciona el algoritmo"
ETQ2M = "Etapa 2. El modelo estimado"
# Ejemplo ilustrativo con números inventados: cuatro avisos, una variable (el área) y una tasa de 0,1 para que se vea el paso
JUG = [("A", 60, 300), ("B", 80, 500), ("C", 120, 700), ("D", 150, 900)]
ETA_J = 0.1


def _jug_pred(m):
    """Predicción del ejemplo tras m árboles de una pregunta (área ≤ 100): se acerca a 400 y 800 en pasos de 10 %."""
    lado = [-1, -1, 1, 1]
    return [600 + s_ * 200 * (1 - (1 - ETA_J) ** m) for s_ in lado]


def _general_arbol(deck):
    s = _nueva(deck, "Un árbol de decisión: preguntas de sí o no que terminan en hojas", ETQ2G, SEC2)
    texto(s, 0.55, 1.3, 4.0, 0.4, "Ejemplo ilustrativo, con números inventados", size=13, color="muted")
    filas = [("Aviso", "Área (m²)", "Precio (millones)")] + [(a, str(ar), n(p, 0)) for a, ar, p in JUG]
    filas.append(("Promedio", "", "600"))
    _tabla(s, 0.55, 1.75, filas, [1.2, 1.3, 1.5], size=13, minimo=0.4)
    # Árbol del ejemplo
    cx = 6.6
    nodo(s, cx - 1.45, 1.75, 2.9, 0.62, ["¿Área ≤ 100 m²?"], relleno="tinte", borde="gris", sizes=(15,),
         colores=("tinta",), margen=0.05)
    hojas = [(cx - 2.35, "sí", "Hoja 1: avisos A y B", "predice 400"), (cx + 0.15, "no", "Hoja 2: avisos C y D", "predice 800")]
    for xh, rama, t1, t2 in hojas:
        linea(s, cx, 2.37, xh + 1.1, 3.35, color="gris", ancho=1.5)
        texto(s, (cx + xh + 1.1) / 2 + (-0.55 if rama == "sí" else 0.1), 2.7, 0.45, 0.3, rama, size=13, color="muted",
              align="right" if rama == "sí" else "left")
        nodo(s, xh, 3.35, 2.2, 0.8, [t1, t2], relleno="azul_claro", borde="azul", sizes=(13, 13), colores=("tinta", "azul"),
             margen=0.05)
    texto(s, 4.6, 4.35, 4.1, 1.1, "Cada hoja predice el promedio de sus avisos: A y B, (300 + 500)/2 = 400; C y D, "
          "(700 + 900)/2 = 800.", size=13, color="tinta")
    # Cómo eligió la pregunta
    filas2 = [("Pregunta probada", "Suma de errores al cuadrado"), ("Ninguna: todos a 600", "200.000"),
              ("¿Área ≤ 70 m²?", "80.000"), (("¿Área ≤ 100 m²?", "azul"), ("40.000: la elegida", "azul")),
              ("¿Área ≤ 135 m²?", "80.000")]
    texto(s, 0.55, 4.35, 3.9, 0.62, "Cómo eligió la pregunta: prueba los umbrales posibles y se queda con el de menor error",
          size=13, color="muted")
    _tabla(s, 0.55, 4.95, filas2, [2.0, 2.0], size=12, minimo=0.34)
    tarjeta(s, 9.0, 1.3, 3.8, 3.55, "", [
        "Árbol de decisión: una secuencia de preguntas de sí o no sobre las variables del aviso.",
        "Pregunta o corte: una variable y un umbral, como área ≤ 100 m².",
        "Rama: la respuesta, sí o no.",
        "Hoja: el grupo de avisos al final de las ramas; todos reciben el mismo valor."], tipo="definicion")
    tarjeta(s, 9.0, 5.0, 3.8, 1.95, "", [
        "Un árbol grande memoriza el ruido y uno pequeño es tosco: aquí A y B reciben 400 aunque valen 300 y 500. El boosting "
        "suma muchos árboles pequeños."], tipo="recordar")
    FIN[id(s)] = 6.95
    notas = ("Antes de ver nuestro modelo, la idea con un ejemplo inventado de cuatro avisos: dos pequeños, de 300 y 500 "
             "millones, y dos grandes, de 700 y 900. Un árbol de decisión es una secuencia de preguntas de sí o no. Este árbol "
             "tiene una sola pregunta, o corte: ¿el área es de 100 metros o menos? Las respuestas, sí o no, son las ramas, y "
             "al final de cada rama hay una hoja: un grupo de avisos que recibe el mismo valor, el promedio de sus precios. "
             "¿Por qué esa pregunta y no otra? Porque el árbol prueba todos los umbrales posibles y mide cuánto se equivoca "
             "con cada uno: sin pregunta, prediciendo 600 para todos, la suma de errores al cuadrado es 200.000; con área de "
             "100 o menos baja a 40.000, la menor. El problema: un solo árbol es tosco, A y B reciben 400 aunque valen 300 y "
             "500, y si lo hacemos muy grande memoriza el ruido. La solución es sumar muchos árboles pequeños: el boosting.")
    return s, notas


def _general_boosting(deck):
    s = _nueva(deck, "Boosting: sumar muchos árboles pequeños que corrigen el error poco a poco", ETQ2G, SEC2)
    pasos = [("0", "Predecir el promedio para todos los avisos."),
             ("1", "Calcular el residuo de cada aviso: precio real menos predicción."),
             ("2", "Un árbol pequeño aprende los residuos: cada hoja da una corrección."),
             ("3", "Sumar la corrección multiplicada por una tasa pequeña, η."),
             ("4", "Repetir con los nuevos residuos, cientos de veces.")]
    wp, gap = 2.3, 0.19
    for k, (num, t_) in enumerate(pasos):
        x0 = 0.55 + k * (wp + gap)
        forma(s, MSO_SHAPE.ROUNDED_RECTANGLE, x0, 1.3, wp, 1.0, relleno="tinte", linea="gris_claro", radio=0.08)
        circulo(s, x0 + 0.1, 1.4, 0.36, f"{num}", relleno="navy", size=13)
        texto(s, x0 + 0.52, 1.34, wp - 0.6, 0.94, t_, size=13, color="tinta")
        if k < len(pasos) - 1:
            linea(s, x0 + wp, 1.8, x0 + wp + gap, 1.8, color="gris", ancho=1.5)
    p1, p2, p50 = _jug_pred(1), _jug_pred(2), _jug_pred(50)
    res0 = [p - 600 for _, _, p in JUG]
    hoja1 = [-200, -200, 200, 200]
    res1 = [p - q for (_, _, p), q in zip(JUG, p1)]
    enc = ("Aviso", "Precio real", "Paso 0: predicción", "Paso 1: residuo", "Paso 2: hoja del árbol 1",
           "Paso 3: predicción tras 1 árbol", "Paso 4: nuevo residuo", "Tras 2 árboles", "Tras 50 árboles")
    filas = [enc]
    for k, (a, _, p) in enumerate(JUG):
        filas.append((a, n(p, 0), "600", n(res0[k], 0, True), n(hoja1[k], 0, True), n(p1[k], 0), n(res1[k], 0, True),
                      n(p2[k], 0), n(p50[k], 0)))
    texto(s, 0.55, 2.42, 12.25, 0.58, "Ejemplo ilustrativo con los cuatro avisos de la lámina anterior y η = 0,1 (en nuestro "
          "modelo, 0,05). El árbol 1 pregunta ¿área ≤ 100 m²? y cada hoja vale el promedio de los residuos de sus avisos.",
          size=13, color="muted")
    y1 = _tabla(s, 0.55, 3.05, filas, [0.9, 1.15, 1.3, 1.25, 1.45, 1.6, 1.4, 1.5, 1.7], size=13, minimo=0.36,
                colores_col={5: "azul", 7: "azul", 8: "azul"})
    texto(s, 0.55, y1 + 0.15, 7.4, 0.62, "Predicción = promedio + η × (hoja del árbol 1 + hoja del árbol 2 + … + hoja del "
          "árbol M)", size=16, color="navy")
    texto(s, 0.55, y1 + 0.8, 7.4, 6.95 - (y1 + 0.8), "Paso 3, para A: 600 + 0,1 × (−200) = 580. Las correcciones se achican "
          "en cada vuelta y A y B se acercan a 400: separar A de B exige otra pregunta, sobre otra variable. Por eso nuestro "
          "modelo usa 162 variables y árboles de hasta 63 hojas.", size=13, color="tinta")
    tarjeta(s, 8.2, y1 + 0.15, 4.6, 6.95 - (y1 + 0.15), "", [
        "Residuo: lo que falta por explicar, el precio real menos la predicción actual.",
        "Tasa de aprendizaje η: la fracción de cada corrección que se suma. Pasos cortos evitan aprender el ruido; a cambio se "
        "necesitan muchos árboles.",
        "LightGBM: una implementación rápida del boosting (Ke y otros, 2017)."], tipo="definicion", maximo=13)
    FIN[id(s)] = 6.95
    notas = ("Así funciona el boosting, con los mismos cuatro avisos. Paso cero: se predice el promedio, 600, para todos. Paso "
             "uno: se calcula el residuo, lo que falta por explicar: A vale 300, así que su residuo es menos 300; D, más 300. "
             "Paso dos: un árbol pequeño aprende esos residuos; con la pregunta del área, la hoja de A y B vale menos 200 y la de "
             "C y D, más 200. Paso tres: no se suma la corrección completa sino una fracción, la tasa de aprendizaje eta; con "
             "0,1, A pasa de 600 a 580. Paso cuatro: se recalculan los residuos y se repite. Tras dos árboles A va en 562; tras "
             "50, en 401. La predicción final es el promedio más eta por la suma de las hojas de todos los árboles. Los pasos "
             "cortos evitan aprender el ruido, y otras preguntas sobre otras variables corrigen lo que una sola no puede. "
             "LightGBM es una implementación rápida de esta idea; es la que usamos.")
    return s, notas


def _entrada(deck):
    s = _nueva(deck, "Nuestro modelo recibe 162 variables por aviso y devuelve un precio en millones", ETQ2M, SEC2)
    texto(s, 0.55, 1.3, 12.25, 0.62, "El algoritmo es LightGBM, un boosting de árboles de decisión (Ke y otros, 2017). Recibe una "
          "tabla con una fila por aviso y una columna por variable, y devuelve un número por aviso: su precio predicho en "
          "millones de pesos.", size=15, color="tinta")
    ej = E["ejemplos"]
    enc = ("Aviso", "Área (m²)", "Baños", "Apartamento (sí = 1)", "Terraza o balcón en el texto (sí = 1)", "Estrato escrito en el texto",
           "Suelo a 300 m (mill./m²)", "Puntaje de precio del texto (mill.)", "Otras 155 variables", "Precio (millones)")
    filas = [enc]
    for k, e in enumerate(ej, 1):
        filas.append((f"{k}. Chapinero", n(e["area_imp"], 0), n(e["banos_imp"], 0), n(e["es_apartamento"], 0),
                      n(e["txt_terraza_balcon"], 0), "vacío" if e["estrato_texto"] is None else n(e["estrato_texto"], 0),
                      n(e["vref_300m_mill"], 1), n(e["puntaje_mill"], 0), "…", n(e["precio"], 0)))
    filas.append(("38.644 avisos de entrenamiento", "", "", "", "", "", "", "", "", "conocido: se aprende de él"))
    filas.append(("10.286 avisos de prueba", "", "", "", "", "", "", "", "", ("vacío: se predice", "naranja")))
    anchos = [2.25, 0.8, 0.7, 1.15, 1.35, 1.15, 1.1, 1.1, 0.95, 1.7]
    y1 = _tabla(s, 0.55, 2.0, filas, anchos, size=12, minimo=0.3)
    texto(s, 0.55, y1 + 0.05, 12.25, 0.3, "Avisos de ejemplo: los tres de Chapinero que se siguen en las láminas siguientes. Las "
          "distancias, los conteos, el suelo y el puntaje entran en logaritmo; aquí van en su unidad.", size=12, color="muted")
    v = E["variables"]
    yc, hc, wc = y1 + 0.45, 6.95 - (y1 + 0.45), 2.95
    tarjeta(s, 0.55, yc, wc, hc, "Lo que se predice", [
        "El precio del aviso en millones de pesos, sin logaritmo, porque Kaggle mide el error en pesos. Para Kaggle, la "
        "predicción se multiplica por un millón (Etapa 4)."], tipo="definicion")
    tarjeta(s, 0.55 + (wc + 0.15), yc, wc, hc, "Sin escalar", [
        "Un árbol solo pregunta si un valor es menor o mayor que un umbral. Cambiar la unidad o tomar logaritmo no cambia el "
        "orden de los avisos, así que no cambia las preguntas."], tipo="neutra")
    tarjeta(s, 0.55 + 2 * (wc + 0.15), yc, wc, hc, "Faltantes: se dejan vacíos", [
        f"{v['con_faltantes_entrenamiento']} de las 162 variables tienen vacíos ({n(v['celdas_faltantes_entrenamiento_pct'], 0)} % "
        "de las celdas). En cada pregunta se prueba mandar los vacíos a un lado y al otro, y gana el lado que más baja el "
        "error."], tipo="recordar")
    tarjeta(s, 0.55 + 3 * (wc + 0.15), yc, wc, hc, "Continuas y de sí o no", [
        f"{v['continuas']} variables son continuas, como el área o las distancias, y {v['binarias']} son de sí o no: 1 si el "
        "texto dice terraza, 0 si no. Entran igual; en una de sí o no solo cabe la pregunta 0 contra 1."], tipo="neutra")
    FIN[id(s)] = 6.95
    notas = ("Pasemos de la idea general a nuestro modelo, empezando por lo más concreto: qué le entra y qué le sale. Le entra una tabla: cada fila es un aviso y "
             "cada columna una de las 162 variables de la Etapa 1. Aquí van tres avisos de Chapinero que vamos a seguir: "
             "área, baños, si es apartamento, si el texto menciona terraza, el estrato que a veces escribe el aviso, el valor "
             "del suelo alrededor y el puntaje de precio del texto. La última columna es el precio: en los 38.644 avisos de "
             "entrenamiento lo conocemos y el modelo aprende de él; en los 10.286 de prueba está vacío y es lo que hay que "
             "predecir. Cuatro detalles de cómo entran los datos. Uno: predecimos el precio en millones, sin logaritmo, "
             "porque Kaggle mide el error en pesos. Dos: no hay que escalar nada, porque un árbol solo compara con umbrales. "
             f"Tres: los vacíos se dejan vacíos; {v['con_faltantes_entrenamiento']} variables los tienen, y el algoritmo "
             "decide en cada pregunta hacia qué lado mandarlos. Cuatro: las variables de sí o no entran igual que las "
             "continuas. Lo que sale es un número por aviso: su precio predicho en millones.")
    return s, notas


CAM = _leer("etapas2a4_camino.json")


def _arbol(deck):
    s = _nueva(deck, "Un aviso real recorre 241 árboles: del promedio, 653, a una predicción de 783", ETQ2M, SEC2)
    av2 = TRES["avisos"][1]
    ruta = CAM["ruta_arbol1"]

    def pregunta(p_):
        v_, u, x = p_["variable"], p_["umbral"], p_["valor_aviso"]
        if v_ == "puntaje_texto_ridge":
            return f"Puntaje de precio del texto ≤ {n(math.exp(u), 0)} millones", n(math.exp(x), 0)
        if v_ == "bedrooms":
            return f"Alcobas ≤ {n(u, 1)}", n(x, 0)
        if v_ == "lsa_04":
            return f"Componente 4 del texto (LSA) ≤ {n(u, 3)}", n(x, 3)
        if v_ == "area_texto":
            return f"Área escrita en el texto ≤ {n(u, 1)} m²", "vacío" if x is None else n(x, 0)
        if v_ == "ln_dist_zonat_m":
            return f"Distancia a la Zona T ≤ {n(math.expm1(u) / 1000, 1)} km", f"{n(math.expm1(x) / 1000, 1)} km"
        return v_, "" if x is None else n(x, 2)

    # Columna izquierda: el camino del aviso por el árbol 1
    texto(s, 0.55, 1.3, 6.3, 0.62, f"Árbol 1: las {len(ruta)} preguntas que responde el apartamento de {n(av2['area'], 0)} m² "
          "(modelo entrenado sin Chapinero: por eso su puntaje vale 814 y no 841)", size=13, color="muted")
    enc_y = 1.95
    texto(s, 0.55, enc_y, 3.55, 0.3, "Pregunta o corte", size=12, color="muted")
    texto(s, 4.15, enc_y, 1.15, 0.3, "Respuesta", size=12, color="muted")
    texto(s, 5.3, enc_y, 1.6, 0.3, "Otra rama", size=12, color="muted")
    y0, paso, hq = 2.3, 0.49, 0.38
    for k, p_ in enumerate(ruta):
        y = y0 + k * paso
        q, x = pregunta(p_)
        nodo(s, 0.55, y, 3.5, hq, [q], relleno="tinte", borde="gris_claro", sizes=(12,), colores=("tinta",), margen=0.06)
        if k < len(ruta):
            linea(s, 0.75, y + hq, 0.75, y + paso, color="gris", ancho=1.25)
        texto(s, 4.15, y, 1.15, hq, f"{p_['respuesta']}: {x}", size=12, color="azul", anchor="middle")
        linea(s, 4.05, y + hq / 2, 4.12, y + hq / 2, color="gris_claro", ancho=1.0)
        texto(s, 5.3, y, 1.6, hq, f"{n(p_['n_otra'], 0)} avisos", size=12, color="muted", anchor="middle")
    yh = y0 + len(ruta) * paso
    s1 = CAM["suma_por_arbol"][0]
    nodo(s, 0.55, yh, 3.5, 0.55, [f"Hoja: {n(CAM['n_hoja_arbol1'], 0)} avisos", f"suma {n(s1, 1, signo=True)} millones"],
         relleno="azul_claro", borde="azul", sizes=(13, 13), colores=("azul", "azul"), margen=0.06)
    texto(s, 4.15, yh, 2.75, 0.55, f"El árbol 1 ve {n(CAM['n_arbol1'], 0)} avisos (el 80 % sorteado) y tiene 63 hojas.",
          size=12, color="muted", anchor="middle")
    # Columna derecha: lo que suma cada árbol y la predicción acumulada
    xr, wr = 7.2, 5.65
    suma = CAM["suma_por_arbol"]
    cam = CAM["camino"]
    arb = list(range(1, len(suma) + 1))
    texto(s, xr, 1.3, wr, 0.35, "Lo que suma cada uno de los 241 árboles (millones)", size=13, color="muted")
    ch1 = lineas_xy(s, xr - 0.1, 1.6, wr + 0.1, 1.5, [
        dict(nombre="Suma", x=arb, y=suma, color="azul", ancho=1.25),
        dict(nombre="Cero", x=[0, len(suma)], y=[0, 0], color="gris", ancho=1.0)],
        leyenda=False, minx=0, maxx=250, miny=-10, maxy=15, fmt_x="0", fmt_y=MENOS, unidad_x=50, unidad_y=5)
    _ = ch1
    texto(s, xr, 3.15, wr, 0.35, f"Predicción acumulada: parte del promedio, {n(cam[0], 0)} (millones)", size=13, color="muted")
    ch2 = lineas_xy(s, xr - 0.1, 3.45, wr + 0.1, 1.95, [
        dict(nombre="Predicción", x=list(range(len(cam))), y=cam, color="azul", ancho=2.5),
        dict(nombre="Precio real", x=[0, len(cam) - 1], y=[av2["real"]] * 2, color="tinta", ancho=1.5, guiones=True)],
        leyenda=False, minx=0, maxx=250, miny=550, maxy=850, fmt_x="0", fmt_y="#,##0", titulo_x="Número de árboles",
        unidad_x=50, unidad_y=50)
    etiquetas_puntos(ch2, [[None] * (len(cam) - 1) + [f"{n(cam[-1], 0)}"],
                           [None, f"precio real: {n(av2['real'], 0)}"]], size=12)
    ch2.plots[0].series[0].points[len(cam) - 1].data_label.position = XL_LABEL_POSITION.ABOVE
    ch2.plots[0].series[1].points[1].data_label.position = XL_LABEL_POSITION.BELOW
    pos = sum(1 for v in suma if v > 0)
    texto(s, xr, 5.42, wr, 0.5, f"Predicción = promedio {n(cam[0], 1)} + la suma de sus 241 hojas = {n(cam[-1], 1)}. {pos} "
          f"árboles suman y {len(suma) - pos} restan.", size=12, color="tinta")
    tarjeta(s, xr, 5.95, wr, 1.0, "", [
        "Pregunta o corte: una variable y un umbral. Rama: la respuesta, sí o no. Hoja: el grupo final de avisos; su valor "
        "se suma a la predicción de todos ellos."], tipo="definicion", maximo=13)
    FIN[id(s)] = 6.95
    notas = ("Esta lámina muestra el algoritmo completo con un solo aviso: el apartamento de Chapinero de 125 metros, con un "
             "modelo entrenado sin Chapinero, así que nunca lo vio. A la izquierda, su paso por el primer árbol. Cada caja es "
             "una pregunta, o corte: una variable y un umbral. La respuesta es la rama que toma: su puntaje de precio del texto, "
             f"calculado sin los avisos de Chapinero, es {n(CAM['valores']['puntaje_mill'], 0)} millones, así que a la primera "
             "pregunta, si es 687 o menos, responde no; sigue preguntando por el mismo "
             "puntaje, por las alcobas, por un componente del texto, por el área escrita, que en este aviso está vacía y por "
             "eso sigue por el lado sí, y por la distancia a la Zona T. A la derecha de cada pregunta, cuántos avisos se fueron "
             f"por la otra rama. Tras 8 preguntas llega a una hoja, un grupo de {n(CAM['n_hoja_arbol1'], 0)} avisos, que le suma "
             f"{n(s1, 1)} millones. Eso se repite en 241 árboles, cada uno con sus propias preguntas. Arriba a la derecha, lo "
             "que suma cada árbol: los primeros suben la predicción y después hay correcciones en los dos sentidos, cada vez "
             f"más pequeñas. Abajo, la predicción acumulada: parte del promedio, {n(cam[0], 0)}, sube hasta {n(max(cam), 0)} y los "
             f"últimos árboles la bajan a {n(cam[-1], 0)}, frente a un precio real de {n(av2['real'], 0)}.")
    return s, notas


NOMBRE_VAR = {"puntaje_texto_ridge": "Puntaje de precio del texto", "banos_imp": "Baños", "area_imp": "Área", "log_area": "Logaritmo del área",
              "parqueaderos_imp": "Parqueaderos", "parqueaderos_texto": "Parqueaderos en el texto",
              "lsa_08": "Componente 8 del texto (LSA)", "prop_estrato6_300m": "Manzanas de estrato 6 a 300 m",
              "bedrooms": "Alcobas", "ln_dist_zonat_m": "Distancia a la Zona T"}


def _umbral(c):
    v_, u = c["variable"], c["umbral"]
    if v_ == "puntaje_texto_ridge":
        return f"≤ {n(math.exp(u), 0)} millones"
    if v_ == "area_imp":
        return f"≤ {n(u, 1)} m²"
    if v_ == "prop_estrato6_300m":
        return f"≤ {n(100 * u, 0)} %"
    if v_ == "ln_dist_zonat_m":
        return f"≤ {n(math.expm1(u) / 1000, 1)} km"
    if v_ == "lsa_08":
        return f"≤ {n(u, 3)}"
    return f"≤ {n(u, 1)}"


def _corte(deck):
    s = _nueva(deck, "En nuestros datos, la mejor primera pregunta es el puntaje de precio del texto", ETQ2M, SEC2)
    pc = E["primer_corte"]
    # Sin las copias exactas de otra variable (logaritmo del área y parqueaderos del texto dan la misma división)
    top = [c for c in pc["top"] if c["variable"] not in ("log_area", "parqueaderos_texto")][:8]
    texto(s, 0.55, 1.3, 6.3, 0.62, "Ejemplo real: la primera pregunta con los 38.644 avisos. Cuánto baja el error cuadrático "
          "con el mejor umbral de cada variable (% del total)", size=13, color="muted")
    cats = [f"{NOMBRE_VAR.get(c['variable'], c['variable'])} {_umbral(c)}" for c in top]
    ch = barras(s, 0.45, 1.95, 6.4, 3.55, cats, [("Baja", [c["reduccion_pct"] for c in top],
                                                 ["azul"] + ["gris_claro"] * (len(top) - 1))],
                minimo=0, maximo=60, gap=35, grid=False)
    etiquetas_puntos(ch, [[f"{n(c['reduccion_pct'], 1)} %" for c in top]])
    g0 = top[0]
    area = next(c for c in top if c["variable"] == "area_imp")
    texto(s, 0.55, 5.55, 6.3, 1.4, [
        f"Gana el puntaje de precio del texto: separa {n(g0['n_si'], 0)} avisos con precio medio de {n(g0['media_si'], 0)} millones de "
        f"{n(g0['n_no'], 0)} con {n(g0['media_no'], 0)}. En el área, los avisos sin área quedan del lado sí: así se reparten "
        "los vacíos.",
        "Ejemplo sin el sorteo de avisos y variables que hace cada árbol del modelo."], size=13, color="tinta", espacio=4)
    _ = area
    x0, w0 = 7.15, 5.65
    texto(s, x0, 1.3, w0, 0.85, "1. Para cada variable prueba los umbrales posibles. LightGBM agrupa cada variable en hasta 255 "
          "intervalos (un histograma) y solo prueba sus bordes: así es rápido.", size=14, color="tinta")
    texto(s, x0, 2.2, w0, 0.6, "2. Mide cada umbral con su ganancia: cuánto baja el error cuadrático si se hace ese corte.",
          size=14, color="tinta")
    _, hg = _formula(s, "f_ganancia", x0 + 0.2, 2.82, escala=0.75, alt="Ganancia de un corte")
    texto(s, x0, 2.82 + hg + 0.08, w0, 0.85, "S: suma de los residuos de un lado (precio menos lo que ya se predice). n: avisos "
          "de ese lado. λ = 1: penalización que frena a las hojas con pocos avisos.", size=13, color="muted")
    y3 = 2.82 + hg + 0.98
    texto(s, x0, y3, w0, 0.62, "3. Se queda con la variable y el umbral de mayor ganancia. Los vacíos se prueban a los dos "
          "lados.", size=14, color="tinta")
    texto(s, x0, y3 + 0.68, w0, 0.85, "4. Crece por hojas: el siguiente corte va en la hoja que más gana, hasta 63 hojas y sin "
          "dejar hojas de menos de 20 avisos. Otros algoritmos crecen nivel por nivel.", size=14, color="tinta")
    y5 = y3 + 1.6
    texto(s, x0, y5 + 0.12, 2.45, 0.45, "Valor de cada hoja:", size=14, color="tinta")
    wh, hh = _formula(s, "f_hoja", x0 + 2.4, y5, escala=0.72, alt="Valor de una hoja")
    texto(s, x0 + 2.45 + wh + 0.1, y5 + 0.12, w0 - (2.45 + wh + 0.1), 0.65, "y entra multiplicado por η = 0,05.", size=14,
          color="tinta")
    FIN[id(s)] = max(y5 + hh, 6.95)
    notas = ("En los cuatro avisos de juguete el árbol eligió su pregunta probando umbrales; aquí está lo mismo con nuestros datos. "
             "Aquí está el ejemplo real de la primera pregunta con los 38.644 avisos: para cada variable buscamos su mejor "
             "umbral y medimos cuánto baja el error cuadrático, es decir, la suma de los errores al cuadrado. Gana el puntaje "
             f"del texto, que solo con esa pregunta baja el error en {n(g0['reduccion_pct'], 0)} %; después vienen los baños y "
             "el área. La regla tiene cuatro pasos. Uno: para cada variable prueba los umbrales; LightGBM, para ser rápido, "
             "agrupa los valores en hasta 255 intervalos y prueba solo sus bordes. Dos: mide cada umbral con la ganancia de "
             "la fórmula, que es cuánto baja el error; S es la suma de los residuos, es decir, de lo que falta por explicar, "
             "y lambda es una penalización. Tres: se queda con la mejor; los avisos con el dato vacío se prueban a los dos "
             "lados. Cuatro: LightGBM crece por hojas: el siguiente corte va donde más baja el error, hasta 63 hojas y sin "
             "hojas de menos de 20 avisos. El valor de cada hoja es el promedio de sus residuos, encogido un poco por lambda, "
             "y se multiplica por la tasa de aprendizaje de 0,05, que explicamos ahora.")
    return s, notas


def _boosting(deck):
    s = _nueva(deck, "Datos reales y una sola variable: cada árbol acerca la predicción a los precios", ETQ2G, SEC2)
    imagen(s, IMG / "boosting_1d.png", 0.55, 1.3, w=12.3,
           alt="Cuatro paneles: precio frente al área con la predicción del boosting tras 0, 1, 20 y 241 árboles")
    texto(s, 0.55, 4.75, 7.6, 2.2, [
        "Puntos grises: avisos de entrenamiento fuera de Chapinero. Línea azul: la predicción según el área, tras 0, 1, 20 y "
        "241 árboles de 4 hojas, con η = 0,05.",
        "Con un árbol la línea casi no se mueve, porque solo se suma el 5 % de cada corrección. A los 20 árboles aparece la "
        "forma: más área, más precio. A los 241 la escalera sigue a los datos y el error absoluto medio en el entrenamiento "
        "baja de 247 a 161 millones."], size=14, color="tinta", espacio=8)
    tarjeta(s, 8.45, 4.75, 4.35, 2.2, "", [
        "Cada escalón es una pregunta sobre el área. Sumar sin fin bajaría más el error de entrenamiento pero aprendería el "
        "ruido: la parada temprana decide cuándo parar. Nuestro modelo hace lo mismo con 162 variables y hasta 63 hojas."],
        tipo="neutra", maximo=15)
    FIN[id(s)] = 6.95
    notas = ("Ahora la misma idea con datos reales, para verla: miles de avisos de entrenamiento y una sola variable, el área. "
             "Con cero árboles se predice el promedio para todos, una línea plana, con un error medio de 247 millones. El primer "
             "árbol casi no mueve la línea, porque solo se suma el 5 % de su corrección. A los 20 árboles ya aparece la forma: "
             "los apartamentos grandes valen más. A los 241 la escalera sigue los datos y el error baja a 161. Cada escalón es "
             "una pregunta sobre el área. Si siguiéramos sumando, la escalera perseguiría cada punto, es decir, el ruido; por "
             "eso hay que decidir cuándo parar. Nuestro modelo hace exactamente esto, pero con 162 variables a la vez.")
    return s, notas


def _objetivo(deck):
    s = _nueva(deck, "Lo que minimiza nuestro modelo: error cuadrático más una penalización", ETQ2M, SEC2)
    _, hf = _formula(s, "f_objetivo", 0.75, 1.35, escala=0.8, alt="Función objetivo de cada árbol")
    y = 1.35 + hf + 0.12
    par = [[("rᵢ: residuo del aviso i, su precio menos lo que predicen los árboles anteriores, en millones de pesos. ", {}),
            ("wⱼ: valor de la hoja j; j(i) es la hoja donde cae el aviso i. J: hojas del árbol, hasta 63.", {})],
           [("Primer término, la pérdida L2: ", {"color": "azul"}),
            ("la suma de los errores al cuadrado. El valor de hoja que la minimiza es el promedio de sus residuos: por eso el "
             "modelo apunta a la media.", {})],
           [("Segundo término, la regularización L2 con λ = 1: ", {"color": "azul"}),
            ("castiga los valores de hoja grandes. La hoja queda en S/(n + 1): con 20 avisos, el 95 % del promedio; con "
             "1.000, casi el 100 %. Frena sobre todo a las hojas pequeñas.", {})],
           [("Todo el modelo: ", {"color": "azul"}),
            (f"cada árbol resuelve este problema sobre los residuos que dejó el anterior; al sumar los {N_ARB} árboles con "
             "η = 0,05, baja paso a paso la suma de (y − ŷ)² en los 38.644 avisos. Los demás frenos son los hiperparámetros: "
             "hojas, avisos por hoja, sorteos y parada temprana.", {})]]
    texto(s, 0.55, y, 6.6, 6.95 - y, par, size=14, color="tinta", espacio=8)
    tarjeta(s, 7.45, 1.35, 5.35, 2.05, "Por qué error cuadrático si Kaggle mide el absoluto", [
        "El absoluto apunta a la mediana y el cuadrático a la media. Chapinero es más caro que el entrenamiento y la media se "
        "queda menos corta: entrenando sin Chapinero, el error en sus 307 avisos fue 150,1 con L2 y 160,7 con L1 (variables "
        "anteriores)."], tipo="recordar")
    tarjeta(s, 7.45, 3.52, 5.35, 1.3, "Por qué el precio y no su logaritmo", [
        "Kaggle mide el error en pesos y el logaritmo comprime los precios altos de Chapinero. Logaritmo y precio por m² "
        "se probaron y no ganaron."], tipo="neutra")
    tarjeta(s, 7.45, 4.95, 5.35, 2.0, "Parámetros e hiperparámetros", [
        f"Parámetros: lo que el algoritmo aprende de los datos al minimizar: los cortes y los valores de las hojas. Con {N_ARB} "
        f"árboles de hasta 63 hojas son hasta {n(N_ARB * 62, 0)} cortes y {n(N_ARB * 63, 0)} valores por corrida.",
        "Hiperparámetros: lo que se fija antes de entrenar, como λ, η o las 63 hojas (lámina siguiente)."], tipo="definicion")
    FIN[id(s)] = 6.95
    notas = ("¿Qué optimiza exactamente nuestro modelo? Cada árbol elige sus cortes y los valores de sus hojas para minimizar "
             "esta función. El primer término es la pérdida L2: la suma de los errores al cuadrado, en millones de pesos, "
             "entre el residuo de cada aviso y el valor de la hoja donde cae. El segundo término es la regularización L2: "
             "lambda, igual a 1, castiga los valores de hoja grandes. En la práctica, la hoja vale la suma de sus residuos "
             "dividida por el número de avisos más uno: con 20 avisos queda en el 95 % del promedio, con mil casi no cambia. "
             "Como cada árbol corrige al anterior, el modelo completo va bajando la suma de errores al cuadrado de los 38.644 "
             "avisos. ¿Por qué cuadrático si Kaggle mide el absoluto? Porque el cuadrático apunta a la media y el absoluto a la "
             "mediana, y en Chapinero, que es más caro, la media se quedó menos corta: 150,1 frente a 160,7 millones al "
             "predecir Chapinero sin haberlo visto, con las variables de entonces. Y una distinción que usaremos: los "
             "parámetros, cortes y valores de hojas, los aprende el algoritmo; los hiperparámetros los fijamos nosotros.")
    return s, notas


def _hiperparametros(deck):
    s = _nueva(deck, "Hiperparámetros: qué controla cada uno, su valor y cómo se eligió", ETQ2M, SEC2)
    filas = [("Hiperparámetro (nombre en LightGBM)", "Qué controla", "Valor", "Cómo se eligió"),
             ("Pérdida (objective)", "qué error se minimiza", "L2: error cuadrático",
              "Comparada en validación espacial con L1 (error absoluto), logaritmo y precio por m²: L2 tuvo menos error y "
              "sesgo en Chapinero."),
             ("Tasa de aprendizaje η (learning_rate)", "cuánto pesa cada árbol nuevo", n(float(HIPER["learning_rate"]), 2),
              "Fijada a mano, sin búsqueda (por defecto, 0,1); la parada temprana compensa con más árboles."),
             ("Número de árboles (n_estimators)", "cuántos árboles se suman", f"{N_ARB} con la semilla 2026",
              "Parada temprana: hasta 5.000; para si el error en el 20 % de las celdas de 2 km no baja en 100 árboles."),
             ("Hojas por árbol (num_leaves)", "qué tan complejo es cada árbol", f"hasta {int(HIPER['num_leaves'])}",
              "Comparado en validación espacial con 31 hojas y 50 avisos por hoja: casi igual (150,1 frente a 148,8 "
              "en Chapinero); se dejó 63."),
             ("Avisos por hoja (min_child_samples)", "tamaño mínimo de una hoja", str(int(HIPER["min_child_samples"])),
              "Valor por defecto de LightGBM; comparado con 50 en la misma prueba."),
             ("Muestreo de avisos (subsample)", "parte de los avisos por árbol",
              f"{n(100 * float(HIPER['subsample']), 0)} %, sorteo por árbol", "Fijado a mano, sin búsqueda (por defecto, 100 %)."),
             ("Muestreo de variables (colsample_bytree)", "parte de las variables por árbol",
              f"{n(100 * float(HIPER['colsample_bytree']), 0)} %", "Fijado a mano, sin búsqueda (por defecto, 100 %)."),
             ("Penalización λ (reg_lambda)", "cuánto se encogen las hojas", n(float(HIPER["reg_lambda"]), 0),
              "Fijada a mano, sin búsqueda (por defecto, 0)."),
             ("Semillas (random_state)", "los sorteos de avisos y variables", "5 corridas promediadas",
              "Fijado a mano: semillas 2026 a 2030; promediar reduce la variación del sorteo."),
             ("Resto (max_depth, max_bin, reg_alpha, min_split_gain)", "profundidad, intervalos, L1 y ganancia mínima",
              "sin límite, 255, 0 y 0", "Por defecto de LightGBM; no se cambiaron.")]
    anchos = [3.05, 2.45, 2.0, 4.75]
    y1 = _tabla(s, 0.55, 1.3, filas, anchos, size=12, minimo=0.3)
    texto(s, 0.55, y1 + 0.06, 12.25, max(6.95 - y1 - 0.06, 0.5), "Validación espacial de la pérdida y las hojas: sin Chapinero, cada localidad fuera y bloques de 4 km, con "
          "variables anteriores. El ridge del texto usa α = 1, el de menor error entre 0,03 y 100. Después de la competencia, "
          "una búsqueda al azar con validación espacial comprobó estos valores (lámina siguiente).", size=12, color="muted")
    FIN[id(s)] = y1
    notas = ("Esta es la ficha técnica, con honestidad sobre cómo se eligió cada valor. La pérdida cuadrática y las 63 hojas "
             "se compararon con validación espacial, con un conjunto anterior de variables: la pérdida absoluta subestimaba más "
             "a Chapinero, y 31 hojas con 50 avisos por hoja daba casi lo mismo, así que dejamos 63. El número de árboles es el "
             f"único que el modelo final elige con sus datos, con parada temprana: {N_ARB} en la corrida de la semilla 2026. "
             "La tasa de aprendizaje de 0,05, los sorteos del 80 % de avisos y 70 % de variables, lambda igual a 1 y las cinco "
             "semillas se fijaron a mano, sin búsqueda. Los demás quedaron en los valores por defecto de LightGBM. Antes del "
             "envío no hubo una búsqueda sistemática con la validación final; la hicimos después, como comprobación, y la "
             "mostramos en la lámina siguiente. El ridge "
             "del texto usa alfa igual a 1; lo fijamos y después comprobamos que es el de menor error entre 0,03 y 100.")
    return s, notas


def _busqueda(deck):
    B = _leer("busqueda_hiperparametros.json")
    R_ = pd.DataFrame(B["resultados"])
    R_["id"] = range(len(R_))
    ev = R_.dropna(subset=["mae_chapinero"])
    fin = R_.iloc[0]
    rk_ch = int((ev["mae_chapinero"] < fin["mae_chapinero"]).sum()) + 1
    s = _nueva(deck, f"Búsqueda al azar posterior: la configuración final, puesto {rk_ch} de {len(ev)} en Chapinero fuera",
               ETQ2M, SEC2)
    hay_cl = "mae_clusters" in R_ and R_["mae_clusters"].notna().sum() >= 2
    texto(s, 0.55, 1.3, 6.4, 0.62, f"Chapinero fuera: error absoluto medio de las {len(ev) - 1} combinaciones sorteadas y de la "
          "del modelo final (en azul), de menor a mayor (millones)", size=13, color="muted")
    ev_o = ev.sort_values("mae_chapinero")
    ch = barras(s, 0.45, 1.9, 6.5, 2.45, [" " * (k + 1) for k in range(len(ev_o))],
                [("MAE", list(ev_o["mae_chapinero"]), ["azul" if i_ == 0 else "gris_claro" for i_ in ev_o["id"]])],
                horizontal=False, etiquetas=False, valor_visible=True, gap=30, fmt="0",
                minimo=math.floor(ev_o["mae_chapinero"].min() / 10) * 10 - 10,
                maximo=math.ceil(ev_o["mae_chapinero"].max() / 10) * 10, grid=True)
    _ = ch
    top = ev_o.head(6)
    filas = [("Combinación", "η", "Hojas", "Avisos por hoja", "Avisos por árbol", "Variables por árbol", "λ",
              "Chapinero fuera", "Clústeres")]

    def fila(r_, nombre):
        return (nombre, n(r_["learning_rate"], 3), str(int(r_["num_leaves"])), str(int(r_["min_child_samples"])),
                f"{n(100 * r_['subsample'], 0)} %", f"{n(100 * r_['colsample_bytree'], 0)} %", n(r_["reg_lambda"], 1),
                n(r_["mae_chapinero"]), "no evaluado" if pd.isna(r_.get("mae_clusters", np.nan)) else n(r_["mae_clusters"]))

    filas.append(tuple((c, "azul") for c in fila(fin, f"Modelo final (puesto {rk_ch})")))
    for k, (_, r_) in enumerate(top.iterrows(), 1):
        if int(r_["id"]) != 0 and len(filas) < 6:
            filas.append(fila(r_, f"Puesto {k}"))
    y1 = _tabla(s, 0.55, 4.45, filas, [2.35, 0.8, 0.8, 1.3, 1.3, 1.45, 0.75, 1.75, 1.75], size=12, minimo=0.28)
    ic = BO["lightgbm"]
    lin = [f"Chapinero fuera va de {n(ev['mae_chapinero'].min())} a {n(ev['mae_chapinero'].max())} millones; la configuración "
           f"final da {n(fin['mae_chapinero'])} (puesto {rk_ch} de {len(ev)})."]
    if hay_cl:
        cl = R_.dropna(subset=["mae_clusters"])
        rk_cl = int((cl["mae_clusters"] < fin["mae_clusters"]).sum()) + 1
        lin.append(f"Clústeres va de {n(cl['mae_clusters'].min())} a {n(cl['mae_clusters'].max())}; la final da "
                   f"{n(fin['mae_clusters'])} (puesto {rk_cl} de {len(cl)}).")
    lin.append(f"Con 307 avisos, el intervalo de 95 % de Chapinero fuera mide unos {n(ic['ic_alto'] - ic['ic_bajo'], 0)} millones "
               "de ancho: diferencias de pocos millones entre combinaciones están dentro del ruido.")
    tarjeta(s, 7.15, 1.3, 5.65, 3.0, "", lin, tipo="neutra", maximo=14)
    texto(s, 0.55, y1 + 0.05, 12.25, 6.95 - y1 - 0.05, f"Rangos sorteados: η 0,02 a 0,1; hojas 15 a 127; avisos por hoja 10 a 100; avisos por árbol 60 a 100 %; "
          f"variables por árbol 40 a 100 %; λ 0 a 10. Una semilla (2026); parada temprana como en el modelo final.",
          size=12, color="muted")
    FIN[id(s)] = 6.95
    notas = ("Los hiperparámetros del modelo final se fijaron antes de una búsqueda sistemática. Para comprobar si fue una buena "
             f"elección, hicimos después una búsqueda al azar: {len(ev) - 1} combinaciones sorteadas en rangos amplios de la "
             "tasa de aprendizaje, las hojas, los avisos por hoja, los sorteos y la penalización, cada una evaluada con la misma "
             "validación espacial que usamos para medir el riesgo, dejar Chapinero fuera"
             + (", y con los clústeres de la ciudad" if hay_cl else "") +
             f". La configuración final queda en el puesto {rk_ch} de {len(ev)}. Las diferencias entre las mejores son de pocos "
             "millones, menores que la incertidumbre de medir con 307 avisos. No cambiamos el modelo enviado: la búsqueda es un "
             "diagnóstico hecho después de la competencia.")
    return s, notas


def _parada(deck):
    s = _nueva(deck, "La parada temprana eligió 252 árboles: parar cuando el error deja de bajar", ETQ2M, SEC2)
    cv = CURVA
    i_par = int(np.where(cv["arboles"] == TRES["parada"])[0][0])
    imin = int(cv["mae_chapinero"].idxmin())
    texto(s, 0.55, 1.3, 6.5, 0.4, "Error absoluto medio (millones) según el número de árboles", size=13, color="muted")
    ch = lineas_xy(s, 0.45, 1.65, 6.6, 3.85, [
        dict(nombre="Chapinero", x=list(cv["arboles"]), y=list(cv["mae_chapinero"]), color="naranja", ancho=2.5),
        dict(nombre="Entrenamiento", x=list(cv["arboles"]), y=list(cv["mae_entrenamiento"]), color="gris", ancho=2.0),
        dict(nombre="Parada", x=[TRES["parada"]] * 2, y=[0, 320], color="gris", ancho=1.5, guiones=True)],
        leyenda=False, minx=0, maxx=800, miny=0, maxy=320, fmt_x="0", fmt_y="0", titulo_x="Número de árboles",
        titulo_y="Error absoluto medio", unidad_x=200, unidad_y=50)
    k = 599
    etiquetas_puntos(ch, [[None] * k + ["Chapinero, 307 avisos que no vio"] + [None] * (len(cv) - k - 1),
                          [None] * k + ["Entrenamiento"] + [None] * (len(cv) - k - 1),
                          [None, f"Parada: {TRES['parada']}"]], size=12)
    ch.plots[0].series[0].points[k].data_label.position = XL_LABEL_POSITION.ABOVE
    ch.plots[0].series[1].points[k].data_label.position = XL_LABEL_POSITION.BELOW
    ch.plots[0].series[2].points[1].data_label.position = XL_LABEL_POSITION.RIGHT
    texto(s, 0.55, 5.55, 6.5, 1.4, "Modelo de ilustración: la misma especificación con una semilla, entrenada sin los 307 avisos "
          f"de Chapinero. Paró en {TRES['parada']} árboles sin mirar Chapinero; ahí su error en Chapinero es "
          f"{n(cv['mae_chapinero'][i_par])}. El mínimo, {n(cv['mae_chapinero'][imin])}, llega en el árbol "
          f"{int(cv['arboles'][imin])}, pero elegirlo mirando Chapinero sería usar la misma prueba para decidir y para medir.",
          size=12, color="muted")
    x0, w0 = 7.35, 5.45
    tarjeta(s, x0, 1.3, w0, 1.25, "Parada temprana", [
        "Sumar árboles mientras baje el error en avisos que no se usan para ajustar, y parar cuando deje de bajar. Es una "
        "forma de regularizar."], tipo="definicion")
    pasos = [f"Se sortea el 20 % de las celdas de 2 km del entrenamiento: sus {n(PF['parada'], 0)} avisos quedan aparte y los "
             f"otros {n(PF['ajuste'], 0)} ajustan (semilla 2026).",
             "Se agregan árboles de uno en uno y se mide el error cuadrático en los avisos apartados.",
             f"Si en 100 árboles seguidos no baja, se para y se toma el mejor número: {N_ARB} con la semilla 2026.",
             "Se entrena de nuevo con los 38.644 avisos y ese número de árboles. Cada una de las 5 corridas repite los pasos con "
             "su semilla y sus celdas."]
    y = 2.7
    for i, p_ in enumerate(pasos, 1):
        circulo(s, x0, y + 0.02, 0.36, i, relleno="navy", size=13)
        alto = _lineas(p_, w0 - 0.55, 13) * 1.21 * 13 / 72 + 0.06
        texto(s, x0 + 0.5, y, w0 - 0.5, alto, p_, size=13, color="tinta")
        y += max(alto, 0.4) + 0.1
    tarjeta(s, x0, y + 0.05, w0, 6.95 - y - 0.05, "Por qué celdas y no avisos al azar", [
        "Con avisos al azar, el vecino de la misma coordenada quedaría a los dos lados y el error apartado premiaría "
        "memorizar el ruido de cada lugar."], tipo="recordar")
    FIN[id(s)] = 6.95
    notas = ("El hiperparámetro que el modelo sí elige con sus datos es el número de árboles, con parada temprana. La idea: seguir sumando árboles mientras el error baje en avisos que el modelo no usa para "
             "ajustar, y parar cuando deje de bajar. En el modelo final apartamos el 20 % de las celdas de 2 kilómetros: unos "
             "7.800 avisos. Sumamos árboles de uno en uno y, si en 100 árboles el error de esos avisos no baja, paramos: con la "
             f"semilla 2026 fueron {N_ARB}. Luego reentrenamos con todos los avisos y ese número. La gráfica muestra por qué "
             "importa: con un modelo de ilustración entrenado sin Chapinero, el error de entrenamiento, en gris, sigue bajando "
             "sin fin, pero el de Chapinero, en naranja, se aplana: más árboles solo memorizan. Apartamos celdas y no avisos al "
             "azar para que un aviso y su vecino de la misma coordenada no queden a los dos lados.")
    return s, notas


NOMBRE_SHAP = {"puntaje_texto_ridge": "Puntaje de precio del texto", "area_imp": "Área", "banos_imp": "Baños", "bedrooms": "Alcobas",
               "ln_dist_embajada_m": "Distancia a una embajada", "ln_vref_300m": "Valor del suelo a 300 m",
               "estrato_medio_300m": "Estrato medio a 300 m", "prop_apartamentos_500m": "Apartamentos a 500 m",
               "elevacion_m": "Elevación", "ln_dist_zonat_m": "Distancia a la Zona T",
               "lsa_09": "Componente 9 del texto (LSA)", "lsa_04": "Componente 4 del texto (LSA)", "log_area": "Logaritmo del área"}
SHP = pd.read_parquet(D / "shap_prueba.parquet")
VALP = pd.read_parquet(D / "valores_prueba.parquet")


def _shap_importancia(deck):
    s = _nueva(deck, "Qué variables pesan más: la importancia según los valores SHAP", ETQ2M, SEC2)
    top = SHAP_IMP.head(10)
    texto(s, 0.55, 1.3, 6.3, 0.62, "Importancia: promedio, en los 10.286 avisos de prueba, del valor SHAP sin signo (millones)",
          size=13, color="muted")
    ch = barras(s, 0.45, 1.9, 6.4, 4.4, [NOMBRE_SHAP.get(k_, k_) for k_ in top.index], [("SHAP", list(top.values), "azul")],
                minimo=0, maximo=175, gap=35, grid=False)
    etiquetas_puntos(ch, [[n(v, 0) for v in top.values]])
    texto(s, 0.55, 6.3, 6.3, 0.68, "Puntaje de precio del texto: el precio que sugieren las palabras del aviso, aprendido "
          "con un ridge de otros avisos, sin ver su propio precio. Modelo final, semilla 2026; valores exactos con TreeSHAP.",
          size=12, color="navy")
    tarjeta(s, 7.15, 1.3, 5.65, 2.55, "Valor SHAP", [
        "Reparte la predicción de cada aviso entre sus 162 variables: φ₀ es la predicción promedio y cada φⱼ lo que la "
        "variable j suma o resta para ese aviso. Los aportes suman exactamente la predicción (Lundberg y Lee, 2017; "
        "Shapley, 1953)."], tipo="definicion")
    from PIL import Image
    w_px, h_px = Image.open(IMG / "f_shap.png").size
    hs = 0.75
    imagen(s, IMG / "f_shap.png", 7.3, 3.95, h=hs, alt="Descomposición SHAP de la predicción")
    tot = SHAP_BLQ.sum()
    orden = [("Texto", "Texto"), ("Vivienda", "Vivienda"), ("Entorno (OSM, GEE, coordenadas)", "Entorno y ubicación"),
             ("Catastro", "Catastro")]
    texto(s, 7.15, 4.8, 5.65, 0.35, "Por bloques de variables (% de la importancia total)", size=13, color="muted")
    chb = barras(s, 7.05, 5.1, 5.75, 1.85, [e for _, e in orden], [("Bloque", [100 * SHAP_BLQ[k] / tot for k, _ in orden],
                                                                    "azul")], minimo=0, maximo=60, gap=35, grid=False)
    etiquetas_puntos(chb, [[f"{n(100 * SHAP_BLQ[k] / tot, 0)} %" for k, _ in orden]])
    FIN[id(s)] = 6.95
    notas = ("Ya sabemos cómo predice el modelo; ahora, qué aprendió. Para eso usamos los valores SHAP. Toman la predicción de "
             "un aviso y la reparten entre sus variables: parten de la predicción promedio y cada variable suma o resta su "
             "parte, de modo que todo suma exactamente la predicción. La importancia de una variable es el promedio de esos "
             "aportes sin signo en los avisos de prueba. El puntaje de precio del texto, el precio que sugieren las palabras del "
             "aviso según el ridge, es, por lejos, lo que más mueve la predicción: "
             f"{n(SHAP_IMP.iloc[0], 0)} millones en promedio. Le siguen el área, los baños y las alcobas, y después variables de "
             "ubicación, como la distancia a una embajada, el valor del suelo y el estrato alrededor. Por bloques, el texto "
             "pesa casi la mitad. Ojo: el puntaje de precio del texto también recoge tamaño y barrio, así que parte de su peso es "
             "ubicación y tamaño dichos con palabras.")
    return s, notas


def _shap_enjambre(deck):
    s = _nueva(deck, "Cómo pesa cada variable: el gráfico de enjambre de los valores SHAP", ETQ2M, SEC2)
    imagen(s, IMG2 / "shap_enjambre.png", 0.5, 1.3, w=12.3, alt="Enjambre de valores SHAP de las 12 variables más importantes")
    p = SHP["puntaje_texto_ridge"]
    tarjeta(s, 0.55, 5.55, 4.0, 1.4, "", [
        "Cada punto es un aviso de prueba: a lo ancho, su valor SHAP; el color, el valor de la variable. Entre paréntesis, la "
        "importancia."], tipo="definicion", maximo=13)
    tarjeta(s, 4.7, 5.55, 4.35, 1.4, "Lo que muestra", [
        f"Un puntaje de precio del texto alto suma hasta {n(p.max(), 0)} millones y uno bajo resta hasta {n(-p.min(), 0)}. Más área, "
        "baños y alcobas suben el precio; estar cerca de una embajada también."], tipo="neutra", maximo=13)
    tarjeta(s, 9.2, 5.55, 3.6, 1.4, "Lo que no es", [
        "Un efecto causal: describe cómo reparte el modelo, y entre variables parecidas el aporte se divide."],
        tipo="recordar", maximo=13)
    FIN[id(s)] = 6.95
    notas = ("Este es el gráfico de enjambre. Cada fila es una variable, ordenada por importancia, y cada punto es uno de los "
             "10.286 avisos de prueba. La posición a lo ancho es su valor SHAP: a la derecha de cero, la variable sube la "
             "predicción de ese aviso; a la izquierda, la baja. El color es el valor de la variable en ese aviso: gris si es "
             "bajo, azul oscuro si es alto. Así se lee la dirección: en el puntaje de precio del texto, los puntos oscuros están a la "
             "derecha, es decir, un texto que suena caro sube el precio, hasta unos 500 millones. En área, baños y alcobas, "
             "más es más caro. En la distancia a una embajada, los puntos claros, los cercanos, están a la derecha: estar "
             "cerca sube el precio, que es la geografía de Chapinero. No son efectos causales: es cómo el modelo reparte.")
    return s, notas


def _shap_cascada(deck):
    s = _nueva(deck, "Una predicción explicada paso a paso: la cascada de un apartamento", ETQ2M, SEC2)
    av2 = TRES["avisos"][1]
    casc = dict(av2["cascada"])
    area_tot = casc.pop("area_imp", 0) + casc.pop("log_area", 0)
    pasos_c = [("puntaje_texto_ridge", casc.pop("puntaje_texto_ridge")), ("area", area_tot)]
    pasos_c += sorted(casc.items(), key=lambda kv: -abs(kv[1]))
    pasos_c.append(("resto", av2["resto"]))

    def etiqueta_var(k_, val):
        if k_ == "banos_imp":
            return f"Baños ({n(val, 0)})"
        if k_ == "area":
            return f"Área ({n(av2['area'], 0)} m²) y su logaritmo"
        if k_ == "ln_vref_300m":
            return f"Suelo a 300 m ({n(math.exp(val) / 1e6, 1)} mill./m²)"
        if k_ == "resto":
            return "Otras 155 variables"
        return NOMBRE_SHAP.get(k_, k_)

    X0C, WC, LAB = 0.55, 8.1, 3.3
    xmin, xmax = 550, 850
    esc = lambda v: X0C + LAB + (v - xmin) / (xmax - xmin) * (WC - LAB - 0.2)  # noqa: E731
    texto(s, X0C, 1.3, WC, 0.4, f"Un apartamento de Chapinero de {n(av2['area'], 0)} m², {n(av2['alcobas'], 0)} alcobas y "
          f"{n(av2['banos'], 0)} baños (millones de pesos)", size=13, color="muted")
    filas_c = [("Promedio sin Chapinero", TRES["base"], "total")]
    for k_, d_ in pasos_c:
        filas_c.append((etiqueta_var(k_, av2["valores"].get(k_, 0)), d_, "paso"))
    filas_c.append(("Predicción", TRES["base"] + sum(d for _, d in pasos_c), "total"))
    y0c, hc = 2.1, 0.42
    acum = TRES["base"]
    for i, (et, val, tipo) in enumerate(filas_c):
        yy = y0c + i * hc
        texto(s, X0C, yy + 0.02, LAB - 0.1, hc - 0.04, et, size=13, color="tinta", align="right", anchor="middle")
        if tipo == "total":
            forma(s, MSO_SHAPE.RECTANGLE, esc(xmin), yy + 0.08, esc(val) - esc(xmin), hc - 0.16, relleno="navy")
            if i == 0:
                texto(s, esc(val) - 0.75, yy + 0.02, 0.67, hc - 0.04, n(val, 0), size=13, color="blanco", align="right",
                      anchor="middle")
            else:
                texto(s, esc(val) + 0.05, yy + 0.02, 0.9, hc - 0.04, n(val, 0), size=13, color="navy", anchor="middle")
        else:
            a_, b_ = sorted([acum, acum + val])
            col = "azul" if val > 0 else "rojo"
            forma(s, MSO_SHAPE.RECTANGLE, esc(a_), yy + 0.08, max(esc(b_) - esc(a_), 0.02), hc - 0.16, relleno=col)
            texto(s, esc(b_) + 0.05, yy + 0.02, 0.9, hc - 0.04, n(val, 1, signo=True), size=13, color=col, anchor="middle")
            acum += val
    yb = y0c + len(filas_c) * hc
    xr = esc(av2["real"])
    linea(s, xr, y0c - 0.05, xr, yb, color="tinta", ancho=1.5)
    texto(s, xr - 1.2, y0c - 0.38, 2.4, 0.3, f"precio real: {n(av2['real'], 0)}", size=12, color="tinta", align="center")
    for v in (600, 700, 800):
        texto(s, esc(v) - 0.4, yb + 0.06, 0.8, 0.3, n(v, 0), size=12, color="muted", align="center")
        linea(s, esc(v), yb + 0.02, esc(v), yb + 0.08, color="gris")
    linea(s, esc(xmin), yb + 0.02, esc(xmax), yb + 0.02, color="gris_claro")
    texto(s, X0C + LAB, yb + 0.34, WC - LAB, 0.3, "millones; el eje empieza en 550", size=12, color="muted", align="center")
    tarjeta(s, 9.0, 1.3, 3.8, 2.1, "", [
        "Arranca en la predicción promedio; cada barra suma (azul) o resta (rojo) lo que aporta una variable, y la última es "
        "la predicción. Modelo entrenado sin los 307 avisos de Chapinero."], tipo="definicion")
    tarjeta(s, 9.0, 3.55, 3.8, 1.85, "Lo que muestra", [
        f"El texto le suma {n(av2['cascada']['puntaje_texto_ridge'], 0)} millones y la predicción, {n(av2['pred'], 0)}, queda "
        f"{n(av2['pred'] - av2['real'], 0)} sobre el precio real: comprar con ella sería pagar de más."], tipo="cuidado")
    tarjeta(s, 9.0, 5.55, 3.8, 1.4, "Por qué el área resta", [
        "El puntaje de precio del texto ya recoge parte del tamaño: entre variables que van juntas, el aporte se reparte."],
        tipo="neutra")
    FIN[id(s)] = 6.95
    notas = ("La cascada explica una sola predicción. Este apartamento de Chapinero, de 125 metros, cuesta 680 millones. El "
             "modelo, entrenado sin Chapinero, arranca en el promedio, 653 millones. El puntaje de precio del texto le suma "
             f"{n(av2['cascada']['puntaje_texto_ridge'], 0)}: su descripción suena a un apartamento caro. El área le resta, "
             "porque el texto ya había contado parte del tamaño; los cuatro baños suman; el valor del suelo alrededor suma "
             f"un poco. Al final predice {n(av2['pred'], 0)}, más de 100 millones por encima del precio. Si la empresa comprara "
             "con esa predicción, pagaría de más: es el error caro, y la cascada muestra de dónde viene.")
    return s, notas


def _shap_dependencia(deck):
    s = _nueva(deck, "Dependencia: cómo cambia el aporte del área y del suelo con su valor", ETQ2M, SEC2)
    imagen(s, IMG / "dependencia.png", 0.5, 1.3, w=12.3, alt="Valor SHAP del área y del valor del suelo según su valor")
    pp = 100 * float((VALP["ln_vref_300m"] > math.log(5.9e6)).mean())
    tarjeta(s, 0.55, 5.05, 4.0, 1.9, "Gráfico de dependencia", [
        "Para una variable, cada punto es un aviso: a lo ancho, el valor de la variable; a lo alto, su valor SHAP. Muestra la "
        "forma que aprendió el modelo."], tipo="definicion")
    tarjeta(s, 4.7, 5.05, 4.0, 1.9, "Área", [
        "El aporte sube con el área hasta unos 250 m² y luego se aplana. En la prueba, en naranja, pesa más en los dos "
        "sentidos: el área se cruza con la ubicación."], tipo="neutra")
    tarjeta(s, 8.85, 5.05, 3.95, 1.9, "Valor del suelo", [
        f"Pasado el percentil 99 del entrenamiento, 5,9 millones por m², el aporte deja de subir: un árbol no extrapola. Ahí "
        f"está el {n(pp, 0)} % de la prueba."], tipo="cuidado")
    FIN[id(s)] = 6.95
    notas = ("Los gráficos de dependencia muestran la forma que aprendió el modelo para una variable. En gris, avisos de "
             "entrenamiento; en naranja, de prueba. A la izquierda, el área: los pequeños restan, el aporte sube hasta unos 250 "
             "metros y luego se aplana. En la prueba el área pesa más en los dos sentidos: el modelo combina tamaño y "
             "ubicación. A la derecha, el valor del suelo alrededor, en escala logarítmica: sube el precio hasta unos 5 "
             "millones por metro cuadrado y después se aplana. La línea es el percentil 99 del entrenamiento. Un árbol no "
             "puede inventar un corte donde no tuvo datos: todo lo que está por encima recibe lo mismo. Y más de la mitad de "
             "la prueba está ahí. Es una causa probable de que el modelo subestime Chapinero, como veremos en la Etapa 4.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# Etapa 3. Validación
# ------------------------------------------------------------------------------------------------------------------
def _validar(deck):
    s = _nueva(deck, "Validar: esconder avisos con precio, predecirlos y medir el error", ETQ3, SEC3)
    texto(s, 0.55, 1.3, 6.2, 1.0, "Validación cruzada (CV): se parte el entrenamiento en grupos llamados pliegues. Se entrena "
          "con todos menos uno, se predice el que quedó fuera y se repite con cada pliegue: cada aviso se predice una vez sin "
          "haberlo visto.", size=14, color="tinta")
    xb, yb, wb, hb = 1.85, 2.45, 4.6, 0.3
    for k in range(5):
        y = yb + k * (hb + 0.1)
        texto(s, 0.55, y - 0.02, 1.25, hb + 0.04, f"Pliegue {k + 1}", size=12, color="muted", anchor="middle")
        for j in range(5):
            forma(s, MSO_SHAPE.RECTANGLE, xb + j * wb / 5, y, wb / 5 - 0.04, hb,
                  relleno="naranja" if j == k else "gris_claro")
    yl = yb + 5 * (hb + 0.1) + 0.02
    forma(s, MSO_SHAPE.RECTANGLE, xb, yl + 0.06, 0.22, 0.18, relleno="naranja")
    texto(s, xb + 0.28, yl, 1.7, 0.3, "se predice", size=12, color="tinta")
    forma(s, MSO_SHAPE.RECTANGLE, xb + 2.0, yl + 0.06, 0.22, 0.18, relleno="gris_claro")
    texto(s, xb + 2.28, yl, 2.2, 0.3, "se entrena", size=12, color="tinta")
    from PIL import Image
    w_px, h_px = Image.open(IMG / "f_mae.png").size
    hm = 0.85
    imagen(s, IMG / "f_mae.png", 0.6, 4.95, h=hm, alt="Fórmula del error absoluto medio")
    texto(s, 0.6 + w_px / h_px * hm + 0.2, 4.95, 6.1 - (w_px / h_px * hm + 0.25), 0.9, "ŷᵢ: predicción; yᵢ: precio real; n: "
          "avisos predichos.", size=12, color="muted")
    texto(s, 0.55, 5.9, 6.2, 1.05, "Error absoluto medio (MAE): el promedio, en millones, de cuánto se aleja cada predicción de "
          "su precio, sin importar si es hacia arriba o hacia abajo. Es la medida de Kaggle. Se calcula con las predicciones "
          "de todos los pliegues juntas.", size=13, color="tinta")
    tarjeta(s, 7.0, 1.3, 5.8, 2.75, "Por qué el azar engaña con datos espaciales", [
        f"En la CV al azar, el 64 % de los avisos validados tiene un aviso de entrenamiento en la misma coordenada (avisos "
        "repetidos o coordenadas compartidas), y el error de esos pares tiene correlación de 0,31: el modelo se apoya en el "
        "vecino y parece mejor de lo que es.",
        "En la prueba real solo el 18 % tiene un vecino así, y Chapinero casi no está en el entrenamiento."], tipo="cuidado")
    wa, _ = _formula(s, "f_sesgo_a", 7.1, 4.17, escala=0.68, alt="Error cuadrático esperado con un vecino en el entrenamiento")
    _formula(s, "f_sesgo_b", 7.1 + wa + 0.45, 4.17, escala=0.68, alt="Error cuadrático esperado en un lugar nuevo")
    texto(s, 7.0, 4.72, 5.8, 0.85, "Si el error de dos vecinos tiene varianza σ² y correlación ρ, copiar al vecino da un error "
          "cuadrático esperado de 2σ²(1 − ρ); en un lugar nuevo, 2σ². La CV al azar promete 2σ²ρ de menos (Roberts y otros, 2017).",
          size=12, color="muted")
    tarjeta(s, 7.0, 5.62, 5.8, 1.33, "Igual en todos los esquemas", [
        "Mismo LightGBM y mismas 162 variables. En cada pliegue el puntaje de precio del texto se recalcula sin los avisos validados, y la "
        "parada temprana aparta avisos como el esquema: al azar o por lugar."], tipo="neutra")
    FIN[id(s)] = 6.95
    notas = ("Validar es esconder avisos cuyo precio sí conocemos, predecirlos como si no lo conociéramos y medir el error. En la "
             "validación cruzada se parte el entrenamiento en pliegues; se entrena con todos menos uno, se predice el que quedó "
             "fuera, y se rota: cada aviso se predice una vez sin haberlo visto. El error que usamos es el de Kaggle, el error "
             "absoluto medio: cuántos millones, en promedio, se aleja la predicción del precio. El problema es cómo armar los "
             "pliegues. Si se sortean los avisos al azar, dos de cada tres avisos validados tienen otro de entrenamiento en la "
             "misma coordenada, a veces el mismo inmueble publicado dos veces, y sus errores se parecen: el modelo se apoya en "
             "el vecino y la validación promete un error que en Chapinero no vamos a tener. La fórmula lo resume. Para que la comparación sea justa, en todos los esquemas el puntaje de precio del texto se recalcula dentro de "
             "cada pliegue y la parada temprana aparta avisos con la misma lógica del esquema.")
    return s, notas


def _esquemas(deck):
    s = _nueva(deck, "Seis formas de esconder avisos: del azar a dejar Chapinero fuera", ETQ3, SEC3)
    imagen(s, IMG2 / "pliegues_6.png", 0.45, 1.3, w=6.3, alt="Mapas de un pliegue de cada esquema y de la prueba real")

    def d_(e):
        return f"{n(DV[e]['mediana'], 0)} m"

    filas = [("Esquema", "Cómo se arma cada pliegue", "Vecino mediano"),
             ("CV aleatoria", f"{PL['aleatoria']['pliegues']} pliegues: cada aviso se sortea a uno.", d_("aleatoria")),
             ("Bloques de 2 km", f"{PL['bloques']['pliegues']} pliegues: una cuadrícula de 2 km; cada cuadro se sortea a un "
                                 "pliegue.", d_("bloques")),
             ("Bloques con buffer de 300 m", "Los mismos; además se quitan del entrenamiento los avisos a 300 m o menos de "
                                             "los validados.", d_("bloques_buffer")),
             ("Clústeres k-means", f"{PL['clusters']['pliegues']} pliegues: k-means agrupa las coordenadas en 10 zonas; cada "
                                   "zona es un pliegue.", d_("clusters")),
             ("kNNDM", "5 pliegues hechos con 1.600 grupos de k-means; 1.600 se eligió para que la distancia al vecino de "
                       "entrenamiento se parezca a la de la prueba.", "79 m"),
             ("Chapinero fuera", "1 pliegue: se entrena con los 38.337 avisos de fuera de Chapinero y se predicen sus 307.",
              d_(CH)),
             (("Prueba real (Kaggle)", "naranja"), ("10.286 avisos, el 94 % en la localidad de Chapinero.", "naranja"),
              ("96 m", "naranja"))]
    anchos = [1.55, 3.35, 0.95]
    y1 = _tabla(s, 6.95, 1.3, filas, anchos, size=12, minimo=0.3)
    texto(s, 6.95, y1 + 0.06, 5.85, 6.95 - y1 - 0.06, "Vecino mediano: distancia mediana de cada aviso que se predice a su "
          "aviso de entrenamiento más cercano. Todos, salvo Chapinero fuera, predicen los 38.644 avisos una vez. Sorteos con "
          "semilla 2026. kNNDM: emparejamiento de distancias al vecino más cercano (Milà y otros, 2022; Linnenbrink y otros, 2024). El buffer no se "
          "dibuja.", size=12, color="muted")
    FIN[id(s)] = max(y1, 6.75)
    notas = ("Probamos seis formas de armar los pliegues, de la más optimista a la más exigente. En los mapas, en naranja, los "
             "avisos que se predicen en un pliegue, y en gris, con los que se entrena. La CV aleatoria sortea avisos sueltos: "
             "el vecino más cercano está, en la mediana, en la misma coordenada. Los bloques sortean cuadros de 2 kilómetros, "
             "y la variante con buffer además quita del entrenamiento lo que esté a 300 metros de lo validado. Los clústeres "
             "parten la ciudad en 10 zonas con k-means. kNNDM arma pliegues para que la distancia al vecino de entrenamiento se "
             "parezca a la de la prueba real, que es de 96 metros. Y Chapinero fuera imita el reto: entrenamos sin los 307 "
             "avisos de Chapinero y los predecimos. El último mapa es la prueba real de Kaggle, para comparar: se parece más a "
             "Chapinero fuera que a cualquier otro.")
    return s, notas


def _resultado_validacion(deck):
    s = _nueva(deck, "Resultado: solo dejar Chapinero fuera anticipa el error de Kaggle", ETQ3, SEC3)
    esq = [("CV aleatoria", MAE["aleatoria"]), ("kNNDM", MAE["knndm"]), ("Bloques de 2 km", MAE["bloques"]),
           ("Bloques con buffer de 300 m", MAE["buffer"]), ("Clústeres k-means", MAE["clusters"]),
           ("Chapinero fuera", MAE["chapinero"]), ("Kaggle, tabla privada", P01), ("Kaggle, tabla pública", U01)]
    texto(s, 0.55, 1.3, 6.2, 0.62, "Error absoluto medio del mismo LightGBM, sin el ajuste ×1,08, en cada validación y en "
          "Kaggle (millones)", size=13, color="muted")
    cols = ["gris_claro"] * 5 + ["azul", "naranja", "naranja"]
    ch = barras(s, 0.45, 1.9, 6.4, 3.8, [e for e, _ in esq], [("MAE", [v for _, v in esq], cols)], minimo=0, maximo=200, gap=35,
                grid=False)
    etiquetas_puntos(ch, [[n(v) for _, v in esq]])
    ic = BO["lightgbm"]
    texto(s, 0.55, 5.75, 6.3, 1.2, [
        f"La CV aleatoria promete un error {n(100 * (1 - MAE['aleatoria'] / P01), 0)} % menor que el de Kaggle. Chapinero fuera "
        f"queda a {n(P01 - MAE['chapinero'], 0)} millones de la tabla privada.",
        f"Con 307 avisos, su intervalo de 95 % va de {n(ic['ic_bajo'], 0)} a {n(ic['ic_alto'], 0)} (bootstrap de 10.000 "
        "réplicas, que trata los avisos como independientes)."], size=12, color="tinta", espacio=4)
    cat_cl = CV_SIN.loc["clusters", "MAE"] - CV_CON.loc["clusters", "MAE"]
    cat_ch = CV_SIN.loc[CH, "MAE"] - CV_CON.loc[CH, "MAE"]
    filas = [("Decisión", "Validación que la decidió", "Resultado"),
             ("Familia de algoritmo", "CV aleatoria, clústeres y Chapinero fuera",
              f"gana el boosting en las tres: LightGBM {n(MAE['chapinero'])} en Chapinero fuera, frente a "
              f"{n(_fam('random_forest', 'Chapinero fuera'))} del Random Forest"),
             ("Variables de Catastro", "clústeres y Chapinero fuera", f"bajan el error {n(cat_cl)} y {n(cat_ch)} millones"),
             ("Pérdida y hojas", "validación espacial anterior", "L2 y 63 hojas (lámina de hiperparámetros)"),
             ("Número de árboles", "celdas de 2 km dentro del entrenamiento", f"{N_ARB} con la semilla 2026"),
             ("Ajuste ×1,08", "Chapinero fuera y avisos a 1 km o menos de la prueba",
              f"{n(MAE['chapinero'])} a {n(BO['lightgbm_x108']['mae'])}; optimista, porque esa prueba lo eligió"),
             ("Error que se espera en Chapinero", "Chapinero fuera", f"{n(MAE['chapinero'])}; Kaggle dio {n(P01)}")]
    anchos = [1.75, 1.95, 2.15]
    y1 = _tabla(s, 6.95, 1.3, filas, anchos, size=12, minimo=0.3)
    tarjeta(s, 6.95, y1 + 0.15, 5.85, 6.95 - y1 - 0.15, "", [
        "Todas las validaciones eligen el mismo algoritmo; lo que cambia es el nivel del error. El riesgo de negocio se mide "
        "con Chapinero fuera."], tipo="clave")
    FIN[id(s)] = 6.95
    notas = (f"Este es el hallazgo de validación. Con el mismo modelo, la CV aleatoria promete {n(MAE['aleatoria'])} millones de "
             f"error; Kaggle dio {n(P01)}, casi el doble. A medida que la validación se aleja en el espacio, el error sube, y "
             f"la única que llega cerca de Kaggle es dejar Chapinero fuera: {n(MAE['chapinero'])}. Con solo 307 avisos, su "
             f"intervalo va de {n(ic['ic_bajo'], 0)} a {n(ic['ic_alto'], 0)}. A la derecha, qué decidió cada validación: todas "
             "eligen el boosting, así que la validación al azar no se equivoca de ganador, se equivoca en el nivel del error. "
             "Catastro bajó el error en clústeres y en Chapinero fuera. La pérdida y las hojas se compararon antes, con otra "
             "validación espacial; el número de árboles lo decide la parada temprana; y el ajuste por 1,08 se eligió mirando "
             "Chapinero fuera, así que su mejora ahí es optimista: Kaggle la confirmó después. Por eso medimos el riesgo de "
             "negocio con Chapinero fuera.")
    return s, notas


MET = E["metricas"]
NOM_ESQ = [("aleatoria", "CV aleatoria"), ("bloques", "Bloques de 2 km"), ("bloques_buffer", "Bloques con buffer de 300 m"),
           ("clusters", "Clústeres k-means"), ("knndm", "kNNDM"), (CH, "Chapinero fuera"),
           ("Chapinero x1,08", "Chapinero fuera, con ×1,08")]


def _metricas(deck):
    s = _nueva(deck, "Cuatro medidas del error en cada validación: MAE, RMSE, R² y sesgo", ETQ3, SEC3)
    filas = [("Validación", "Avisos predichos", "MAE (millones)", "RMSE (millones)", "R²", "Sesgo (millones)")]
    for k, nom in NOM_ESQ:
        m = MET[k]
        filas.append((nom, n(m["n"], 0), n(m["mae"]), n(m["rmse"]), n(m["r2"], 2), n(m["sesgo"], 1, signo=True)))
    filas.append((("Kaggle, tabla privada (envío 02)", "naranja"), ("80 % de 10.286", "naranja"), (n(P02, 1), "naranja"),
                  ("no se publica", "naranja"), ("no se publica", "naranja"), ("no se publica", "naranja")))
    y1 = _tabla(s, 0.55, 1.3, filas, [3.3, 1.75, 1.75, 1.75, 1.55, 2.15], size=13, minimo=0.36)
    w4 = (12.25 - 3 * 0.15) / 4
    defs = [("MAE", "Error absoluto medio: el promedio de |predicción − precio|. Es la medida de Kaggle.", "definicion"),
            ("RMSE", "Raíz del error cuadrático medio: castiga más los errores grandes; siempre es mayor o igual al MAE.",
             "neutra"),
            ("R²", "Parte de la variación del precio de los avisos predichos que explica el modelo: 1 − Σ(y − ŷ)² / Σ(y − ȳ)².",
             "neutra"),
            ("Sesgo", "Promedio de predicción − precio: negativo si predice de menos en promedio.", "recordar")]
    for i, (t_, c_, tp) in enumerate(defs):
        tarjeta(s, 0.55 + i * (w4 + 0.15), y1 + 0.2, w4, 6.95 - y1 - 0.2, t_, [c_], tipo=tp)
    FIN[id(s)] = 6.95
    ch, ch8 = MET[CH], MET["Chapinero x1,08"]
    notas = ("Además del error absoluto medio de Kaggle, miramos otras tres medidas en cada validación. El RMSE, la raíz del "
             "error cuadrático medio, castiga más los errores grandes; el R² dice qué parte de la variación del precio explica "
             "el modelo; y el sesgo dice si en promedio predice de más o de menos. Las tres cuentan la misma historia: al "
             f"alejar la validación en el espacio, el MAE sube de {n(MET['aleatoria']['mae'])} a {n(ch['mae'])} y el RMSE de "
             f"{n(MET['aleatoria']['rmse'])} a {n(ch['rmse'])}. El R² de Chapinero fuera, {n(ch['r2'], 2)}, no es bajo, pero "
             "se mide contra la variación de sus propios precios, que es mayor: para comparar esquemas sirven más el MAE y el "
             f"RMSE. Y el sesgo: solo en Chapinero el modelo predice de menos en promedio, {n(ch['sesgo'], 0)} millones; el "
             f"ajuste por 1,08 lo reduce a {n(ch8['sesgo'], 0)}. Kaggle solo publica el MAE.")
    return s, notas


def _predicho_real(deck):
    s = _nueva(deck, "Predicho frente a real: los 307 avisos de Chapinero que el modelo no vio", ETQ3, SEC3)
    dsp = E["dispersion_chapinero"]
    pr, re_ = np.array(dsp["pred_x108"]), np.array(dsp["real"])
    texto(s, 0.55, 1.3, 6.6, 0.4, "Chapinero fuera, con el ajuste ×1,08 (millones)", size=13, color="muted")
    ch = lineas_xy(s, 0.45, 1.65, 6.7, 5.3, [
        dict(nombre="Avisos", x=list(pr), y=list(re_), color="azul", linea=False, marcador=True, tam=5),
        dict(nombre="Predicción igual al precio", x=[200, 1800], y=[200, 1800], color="gris", ancho=1.5, guiones=True)],
        leyenda=False, minx=200, maxx=1800, miny=200, maxy=1800, fmt_x="#,##0", fmt_y="#,##0", titulo_x="Predicción",
        titulo_y="Precio real", unidad_x=400, unidad_y=400)
    med = np.median(pr)
    bajo, alto = pr <= med, pr > med
    mae_b, mae_a = np.abs(pr - re_)[bajo].mean(), np.abs(pr - re_)[alto].mean()
    arriba = 100 * np.mean(re_ > pr)
    tarjeta(s, 7.45, 1.3, 5.35, 1.65, "", [
        "Cada punto es un aviso. Sobre la línea punteada, la predicción acierta; arriba de ella, el precio real es mayor y el modelo "
        "predijo de menos; abajo, predijo de más."], tipo="definicion")
    tarjeta(s, 7.45, 3.1, 5.35, 2.1, "Lo que muestra", [
        f"La nube sigue la línea: ordena bien (correlación de rangos de 0,87; R² de {n(MET['Chapinero x1,08']['r2'], 2)}). "
        f"Pero el {n(arriba, 0)} % de los puntos queda arriba: predice de menos más veces que de más."], tipo="neutra")
    tarjeta(s, 7.45, 5.35, 5.35, 1.6, "El error crece con el precio", [
        f"Error absoluto medio de {n(mae_b, 0)} millones en la mitad de menor predicción y de {n(mae_a, 0)} en la mitad de "
        "mayor predicción."], tipo="recordar")
    FIN[id(s)] = 6.95
    notas = ("Esta es la gráfica clásica de predicho frente a real, con los 307 avisos de Chapinero que el modelo no vio. Si el "
             "modelo fuera perfecto, todos los puntos estarían sobre la línea punteada. La nube sigue la línea: el modelo ordena "
             "bien los avisos de barato a caro. Pero hay más puntos arriba que abajo: el precio real suele ser mayor que el "
             f"predicho, el {n(arriba, 0)} % de las veces. Y la nube se abre a la derecha: en los avisos caros el error en "
             f"millones es mayor, {n(mae_a, 0)} frente a {n(mae_b, 0)}. Eso importa para el negocio, porque los errores grandes "
             "en millones están justo en las viviendas caras.")
    return s, notas


def _vecinos(deck):
    s = _nueva(deck, "Los errores se parecen entre vecinos: queda algo del lugar sin explicar", ETQ3, SEC3)
    texto(s, 0.55, 1.3, 4.6, 0.4, "Los 307 avisos de Chapinero fuera", size=13, color="muted")
    imagen(s, IMG / "errores_mapa.png", 0.55, 1.55, w=4.6, alt="Mapa de los 307 avisos de Chapinero con el signo del error")
    filas = [("Validación", "Correlación del error de un aviso con el promedio del de sus 10 vecinos")]
    for k, nom in NOM_ESQ[:5]:
        filas.append((nom, n(MET[k]["cor_vecinos10"], 2)))
    y1 = _tabla(s, 5.55, 1.3, filas, [3.0, 4.25], size=13, minimo=0.36)
    tarjeta(s, 5.55, y1 + 0.2, 7.25, 1.75, "", [
        "Para cada aviso se toma su error y el promedio del error de sus 10 avisos de entrenamiento más cercanos. Si el lugar "
        "no importara después del modelo, la correlación sería cercana a cero."], tipo="definicion")
    rk = pd.read_csv(STORES / "rk_knndm_res_lightgbm_con.csv").set_index("variante")
    tarjeta(s, 5.55, y1 + 2.1, 7.25, 6.95 - (y1 + 2.1), "Lo que muestra", [
        "Con validación al azar es 0,10: el vecino está en el entrenamiento y el modelo ya lo usó. Con validación espacial es "
        "0,38 a 0,44: queda un componente del lugar. Interpolar esos errores entre vecinos (kriging) no ayudó: kNNDM pasó de "
        f"{n(rk.loc['sin kriging', 'MAE kNNDM'])} a {n(rk.loc['RK en niveles', 'MAE kNNDM'])}."], tipo="neutra")
    FIN[id(s)] = 6.95
    notas = ("Un diagnóstico más del error: ¿se parecen los errores de avisos vecinos? En el mapa, los 307 avisos de Chapinero: "
             "rojo si el modelo predijo de más, azul si predijo de menos, y más grande cuanto mayor el error. Hay errores de los "
             "dos signos por todo Chapinero, pero en grupos. La tabla lo mide: para cada aviso, la correlación entre su error y "
             "el promedio del error de sus 10 vecinos. Con validación al azar casi no hay correlación, porque el vecino estaba "
             "en el entrenamiento. Con validación espacial sube a cerca de 0,4: queda algo del lugar que las 162 variables no "
             "recogen. Intentamos aprovecharlo interpolando los errores entre vecinos, con kriging, y no mejoró: el error es "
             "casi todo ruido de cada aviso.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# Etapa 4. Pronóstico y negocio
# ------------------------------------------------------------------------------------------------------------------
def _pronostico(deck):
    s = _nueva(deck, "Del modelo al pronóstico: cinco corridas, un ajuste y los gemelos de texto", ETQ4, SEC4)
    lgb_ = FINAL["lightgbm"]
    rb = REJ_B
    pasos = [
        ("Cinco corridas", "El modelo se entrena 5 veces con los 38.644 avisos, incluidos los 307 de Chapinero, con semillas "
                           "2026 a 2030; cada corrida elige su número de árboles.", None),
        ("Promedio", f"Para cada aviso de prueba, ŷ es el promedio de las 5 predicciones: mediana de {n(lgb_['mediana'])} "
                     f"millones, de {n(lgb_['min'])} a {n(lgb_['max'])}.", None),
        ("Ajuste ×1,08", f"El modelo comprime los precios hacia el centro: en Chapinero fuera, cuando la predicción sube "
                         f"1 millón, el precio real sube {n(E['pendiente_chapinero'], 2)} en promedio. Se aleja cada predicción de 654, el "
                         "promedio del entrenamiento. 1,08 se eligió con Chapinero fuera y la zona a 1 km de la prueba.",
         "f_estira"),
        ("Gemelos de texto", f"{FINAL['n_gemelos']} avisos de prueba ({n(FINAL['pct_gemelos'], 1)} %) repiten la descripción de "
                             "avisos del entrenamiento: se usa 0,3 del modelo y 0,7 de la mediana del precio de su gemelo. "
                             "Regla fijada a mano.", "f_gemelo"),
        ("Archivo para Kaggle", f"Se multiplica por 1.000.000 y se redondea: {n(FINAL['n_prueba'], 0)} precios en pesos. Tabla "
                                f"privada: {n(P02, 2)} millones de error; sin el ×1,08, {n(P01, 2)}.", None)]
    x0, w0, y = 0.55, 6.6, 1.3
    hs = [0.95, 0.86, 1.4, 1.08, 0.86]
    for i, ((tit, cuerpo, form), h) in enumerate(zip(pasos, hs), 1):
        forma(s, MSO_SHAPE.ROUNDED_RECTANGLE, x0 + 0.5, y, w0 - 0.5, h, relleno="tinte", linea="gris_claro", radio=0.06)
        circulo(s, x0, y + h / 2 - 0.18, 0.36, i, relleno="navy", size=13)
        if i < len(pasos):
            linea(s, x0 + 0.18, y + h / 2 + 0.18, x0 + 0.18, y + h + 0.1 + hs[i] / 2 - 0.18, color="gris_claro", ancho=1.5)
        texto(s, x0 + 0.62, y + 0.04, 1.75, 0.35, tit, size=14, color="navy")
        if form:
            _formula(s, form, x0 + 2.3, y + 0.02, escala=0.68, alt=tit)
        texto(s, x0 + 0.62, y + 0.38, w0 - 0.75, h - 0.4, cuerpo, size=12, color="tinta")
        y += h + 0.1
    FIN[id(s)] = y - 0.1
    texto(s, 7.45, 1.3, 5.35, 0.4, "Distribución del precio (% de los avisos)", size=13, color="muted")
    cent = list(HIST["desde"] + 37.5)
    lineas_xy(s, 7.35, 1.65, 5.5, 3.1, [
        dict(nombre=f"Pronóstico, prueba ({n(FINAL['n_prueba'], 0)})", x=cent, y=list(HIST["Pronóstico para la prueba"]),
             color="naranja", ancho=3),
        dict(nombre="Chapinero, entrenamiento (307)", x=cent, y=list(HIST["Chapinero"]), color="azul", ancho=2),
        dict(nombre="Resto de Bogotá (38.337)", x=cent, y=list(HIST["Resto de Bogotá"]), color="gris", ancho=2)],
        minx=300, maxx=1725, miny=0, fmt_x="#,##0", fmt_y="0", titulo_x="Precio (millones)", unidad_x=300,
        pos_leyenda="arriba")
    fin = FINAL["envio"]
    stats = [(n(fin["mediana"], 0), f"millones: mediana del pronóstico (media de {n(fin['media'], 0)})"),
             (f"{n(fin['pct_sobre_1200'], 0)} %", "de los avisos con pronóstico mayor a 1.200 millones"),
             (str(fin["n_sobre_1650"]), f"pronósticos sobre 1.650 millones, el máximo del entrenamiento "
                                        f"({FINAL['lightgbm']['n_sobre_1650']} sin el ×1,08)")]
    for i, (v_, e_) in enumerate(stats):
        x_ = 7.45 + i * 1.8
        texto(s, x_, 4.9, 1.75, 0.6, v_, size=28, color="naranja")
        texto(s, x_, 5.5, 1.7, 1.45, e_, size=12, color="muted")
    notas = ("¿Cómo se pasa del modelo al archivo que subimos? Cinco pasos. Uno: entrenamos el modelo cinco veces con todos los "
             "avisos de entrenamiento, incluidos los 307 de Chapinero, cambiando la semilla de los sorteos. Dos: promediamos "
             "las cinco predicciones. Tres: un ajuste. Todas las validaciones espaciales mostraban que el modelo comprime: "
             "predice de menos lo caro y de más lo barato. Por eso alejamos cada predicción del promedio de 654 millones, "
             "multiplicando la distancia por 1,08; ese factor se eligió mirando Chapinero fuera y la zona cercana a la prueba. "
             f"Cuatro: {FINAL['n_gemelos']} avisos de prueba tienen exactamente la misma descripción que avisos del "
             "entrenamiento, probablemente el mismo inmueble; ahí pesamos 70 % el precio del gemelo y 30 % el modelo. Cinco: "
             f"pasamos a pesos. Con el ajuste, el error en Kaggle bajó de {n(P01, 1)} a {n(P02, 1)}. A la derecha, el pronóstico: "
             f"mediana de {n(fin['mediana'], 0)} millones, por encima del resto de Bogotá y parecido a los avisos de Chapinero "
             "que sí teníamos. Parecerse en la distribución no es acertar aviso por aviso: eso lo vemos ahora.")
    return s, notas


def _calibracion(deck):
    s = _nueva(deck, "Calibración: en Chapinero predice de menos en todos los niveles; en Bogotá, no", ETQ4, SEC4)
    cc = CAL[CAL["esquema"] == "Chapinero fuera"]
    sa, ca = cc[cc["version"] == "Sin ajuste"]["sesgo"].tolist(), cc[cc["version"] == "Con ajuste x1,08"]["sesgo"].tolist()
    texto(s, 0.55, 1.3, 6.0, 0.62, "Chapinero fuera (307 avisos): predicción menos precio, promedio por quintil de la "
          "predicción (millones)", size=13, color="muted")
    ch = barras(s, 0.45, 1.9, 6.1, 3.3, ["1 (más barato)", "2", "3", "4", "5 (más caro)"],
                [("Sin ajuste", sa, "gris_claro"), ("Con ajuste ×1,08", ca, "navy")], horizontal=False, minimo=-120, maximo=40,
                unidad=40, leyenda=True, gap=60, overlap=-5, valor_visible=True, fmt=MENOS, etiquetas_abajo=True)
    etiquetas_puntos(ch, [[n(v, 0) for v in sa], [n(v, 0) for v in ca]])
    kk = CAL[CAL["esquema"] == "kNNDM"]
    sk, ck = kk[kk["version"] == "Sin ajuste"]["sesgo"].tolist(), kk[kk["version"] == "Con ajuste x1,08"]["sesgo"].tolist()
    texto(s, 6.85, 1.3, 5.95, 0.62, "Bogotá, kNNDM (38.644 avisos): lo mismo, por decil de la predicción (millones)", size=13,
          color="muted")
    ch2 = barras(s, 6.75, 1.9, 6.1, 3.3, [str(i) for i in range(1, 11)], [("Sin ajuste", sk, "gris_claro"),
                                                                          ("Con ajuste ×1,08", ck, "navy")],
                 horizontal=False, minimo=-120, maximo=40, unidad=40, leyenda=True, gap=50, overlap=-5, valor_visible=True,
                 fmt=MENOS, etiquetas_abajo=True, etiquetas=False)
    _ = ch2
    tarjeta(s, 0.55, 5.35, 4.0, 1.6, "Calibración", [
        "Comparar, por grupos de la predicción, lo predicho con lo real. Un modelo calibrado da barras cercanas a cero."],
        tipo="definicion")
    tarjeta(s, 4.7, 5.35, 4.0, 1.6, "Por qué agrupar por la predicción", [
        "La empresa solo ve la predicción. Agrupar por el precio real mostraría un sesgo falso por regresión a la media."],
        tipo="neutra")
    tarjeta(s, 8.85, 5.35, 3.95, 1.6, "Lo que muestra", [
        f"En Bogotá, sin ajuste, de {n(min(sk), 0)} a {n(max(sk), 0, True)}: calibrado. En Chapinero falta una prima de "
        f"ubicación. El ×1,08 la reduce arriba, pero en Bogotá sube los deciles altos ({n(ck[-2], 0, True)} y "
        f"{n(ck[-1], 0, True)})."], tipo="recordar")
    FIN[id(s)] = 6.95
    notas = ("Ahora, dónde se equivoca. Primero, en promedio: la calibración compara, por grupos de la predicción, lo predicho "
             "con lo real. Agrupamos por la predicción porque es lo único que la empresa ve. A la izquierda, Chapinero fuera: "
             "en los cinco quintiles la barra es negativa, el modelo predice de menos en todos los niveles de precio. Es una "
             "prima de ubicación de Chapinero que el modelo no capta, y encaja con el valor del suelo que se aplana fuera del "
             "rango. A la derecha, el mismo gráfico para toda Bogotá con kNNDM: sin ajuste las barras están cerca de cero, el "
             "modelo está calibrado en la ciudad. El ajuste por 1,08 corrige parte de Chapinero, pero en Bogotá sobrestima las "
             "predicciones más altas. Por eso, para ofertar, no lo recomendamos.")
    return s, notas


def _lados(deck):
    s = _nueva(deck, "Predecir de más o de menos: el 27 % queda más de 10 % arriba del precio", ETQ4, SEC4)
    texto(s, 0.55, 1.3, 7.4, 0.4, "Avisos según su error, con el ajuste ×1,08 (%)", size=13, color="muted")
    cd = CategoryChartData()
    cd.categories = ["Chapinero fuera (307)", "Bogotá, kNNDM (38.644)"]
    cd.add_series("Menor que el real en más de 10 %", [L_CH.bajo_10, L_KN.bajo_10])
    cd.add_series("Dentro de 10 %", [L_CH.dentro_10, L_KN.dentro_10])
    cd.add_series("Mayor que el real en más de 10 %", [L_CH.sobre_10, L_KN.sobre_10])
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_STACKED_100, Inches(0.45), Inches(1.65), Inches(7.6), Inches(2.3), cd)
    chb = gf.chart
    _estilo_grafico(chb, 12)
    chb.has_title = False
    chb.has_legend = True
    chb.legend.position = XL_LEGEND_POSITION.TOP
    chb.legend.include_in_layout = False
    chb.legend.font.size = Pt(12)
    chb.plots[0].gap_width = 40
    for ser, c in zip(chb.plots[0].series, ["azul", "gris_claro", "rojo"]):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = rgb(c)
    _ejes(chb, valor_visible=False, grid=False, invertir_cat=True)
    etiquetas_puntos(chb, [[f"{n(L_CH.bajo_10, 0)} %", f"{n(L_KN.bajo_10, 0)} %"],
                           [f"{n(L_CH.dentro_10, 0)} %", f"{n(L_KN.dentro_10, 0)} %"],
                           [f"{n(L_CH.sobre_10, 0)} %", f"{n(L_KN.sobre_10, 0)} %"]], color="tinta",
                     posicion=XL_LABEL_POSITION.CENTER)
    for i in range(2):
        for j in (0, 2):
            for r in chb.plots[0].series[j].points[i].data_label.text_frame.paragraphs[0].runs:
                r.font.color.rgb = rgb("blanco")
    texto(s, 0.55, 4.1, 7.4, 0.4, "Chapinero fuera: % de avisos con error mayor a 10 %, por quintil de la predicción (millones)",
          size=13, color="muted")
    cats = [f"{n(a_, 0)} a {n(b_, 0)}" for a_, b_ in zip(LQ["pred_min"], LQ["pred_max"])]
    ch = barras(s, 0.45, 4.45, 7.6, 2.5, cats, [("Predice de menos", list(LQ["bajo"]), "azul"),
                                                ("Predice de más", list(LQ["sobre"]), "rojo")],
                horizontal=False, minimo=0, maximo=50, leyenda=True, gap=60, overlap=-5, grid=False)
    etiquetas_puntos(ch, [[n(v, 0) for v in LQ["bajo"]], [n(v, 0) for v in LQ["sobre"]]])
    tarjeta(s, 8.3, 1.3, 4.5, 1.95, f"Predice de más: {n(L_CH.sobre_10, 0)} % de Chapinero", [
        "Si se compra con la predicción, se paga de más: pérdida de capital, costos de tenencia y de reventa. Sin el ajuste, "
        f"{n(L_CH0.sobre_10, 0)} %."], tipo="cuidado")
    tarjeta(s, 8.3, 3.4, 4.5, 1.6, f"Predice de menos: {n(L_CH.bajo_10, 0)} %", [
        f"La oferta queda corta y el dueño no vende: se pierde el negocio, no capital. Sin el ajuste, {n(L_CH0.bajo_10, 0)} %."],
        tipo="definicion")
    tarjeta(s, 8.3, 5.15, 4.5, 1.8, "Lo que muestra", [
        f"El riesgo de pagar de más está en todos los niveles: de {n(LQ['sobre'].min(), 0)} % a {n(LQ['sobre'].max(), 0)} % "
        f"por quintil. Error porcentual mediano: {n(L_CH.error_abs_mediano, 1)} % ({n(L_CH0.error_abs_mediano, 1)} % sin el "
        "ajuste)."], tipo="neutra")
    FIN[id(s)] = 6.95
    notas = ("El promedio esconde la dispersión. Separemos los dos errores, porque no cuestan lo mismo. En Chapinero, con el "
             f"modelo final, el {n(L_CH.sobre_10, 0)} % de los avisos queda más de 10 % por encima de su precio: ahí la empresa "
             f"pagaría de más. El {n(L_CH.bajo_10, 0)} % queda más de 10 % por debajo: ahí perdería el negocio, pero no plata. "
             "Abajo, por quintil de la predicción: el riesgo de pagar de más aparece en todos los niveles de precio, no en un "
             f"segmento que se pueda evitar. El error porcentual mediano es {n(L_CH.error_abs_mediano, 1)} %: la mitad de los "
             "avisos yerra más que eso.")
    return s, notas


def _fuera_rango(deck):
    s = _nueva(deck, "Fuera del rango del entrenamiento: más error en millones, no en porcentaje", ETQ4, SEC4)
    rp = V2["regla_prueba"]
    imagen(s, IMG / "confianza_mapa.png", 0.55, 1.4, w=4.4,
           alt="Mapa de los avisos de prueba según cuántas variables quedan fuera del rango del entrenamiento")
    pv = rp["por_variable_pct"]
    VN = {"ln_vref_300m": "Valor del suelo a 300 m", "ln_n_embajada_1000m": "Embajadas a 1 km", "elevacion_m": "Elevación",
          "ln_dist_area_protegida_m": "Distancia a un área protegida grande", "ln_n_oficina_1000m": "Oficinas a 1 km", "area_imp": "Área",
          "ln_n_hotel_1000m": "Hoteles a 1 km"}
    orden = sorted(pv, key=pv.get, reverse=True)
    texto(s, 5.2, 1.3, 7.6, 0.62, "Avisos de prueba con la variable fuera del rango del entrenamiento: bajo el percentil 1 o "
          "sobre el 99 (%)", size=13, color="muted")
    ch = barras(s, 5.1, 1.9, 7.7, 2.3, [VN[k_] for k_ in orden], [("fuera", [pv[k_] for k_ in orden], "naranja")],
                minimo=0, maximo=75, gap=30, grid=False)
    etiquetas_puntos(ch, [[n(pv[k_], 1) for k_ in orden]])
    filas = [("Variables fuera del rango", "Avisos de prueba", "MAE en Chapinero fuera (avisos de los 307)",
              "Error porcentual mediano")]
    for cat, nom, k in (("Dentro del rango", "Ninguna", "dentro"), ("Una fuera", "Una", "una"), ("Dos o más fuera", "Dos o más",
                                                                                                "dos_o_mas")):
        filas.append((nom, n(rp[k], 0), f"{n(REG.loc[cat, 'mae'])} ({int(REG.loc[cat, 'n'])})",
                      f"{n(REG.loc[cat, 'error_abs_mediano'], 0)} %"))
    y1 = _tabla(s, 5.2, 4.3, filas, [1.9, 1.5, 2.6, 1.6], size=13, minimo=0.36)
    tarjeta(s, 5.2, y1 + 0.15, 7.6, 6.95 - y1 - 0.15, None, [
        f"El {n(100 * rp['dos_o_mas'] / FINAL['n_prueba'], 0)} % de la prueba tiene dos o más variables fuera del rango. Esos "
        "avisos son más caros y yerran más en millones, pero no en porcentaje: el rango sirve para priorizar la revisión, no "
        "para confiar a ciegas en el resto."], tipo="neutra", maximo=14)
    FIN[id(s)] = 6.95
    notas = ("Un árbol no extrapola, así que preguntamos si salir del rango del entrenamiento sirve como alarma. El mapa marca "
             f"los avisos de prueba según cuántas variables quedan fuera del rango: el {n(100 * rp['dos_o_mas'] / FINAL['n_prueba'], 0)} "
             "% tiene dos o más, sobre todo el valor del suelo, las embajadas y la elevación, en el norte y junto a los cerros. "
             "Con los 307 avisos de Chapinero fuera, clasificados igual, esos avisos tienen el mayor error en millones, porque "
             "son más caros, pero en porcentaje el error es parecido en los tres grupos. Conclusión: el rango ayuda a decidir "
             "qué revisar primero, pero estar dentro del rango no hace confiable a un aviso.")
    return s, notas


def _zillow(deck):
    s = _nueva(deck, "La trampa de Zillow: el dueño acepta cuando el modelo sobrestima", ETQ4, SEC4)

    def compra(m, dd):
        return COMP[(COMP["validacion"] == "Chapinero fuera") & np.isclose(COMP["descuento"], dd) &
                    np.isclose(COMP["margen"], m)].iloc[0]

    def equilibrio(dd):
        t = GP[(GP["validacion"] == "Chapinero fuera") & np.isclose(GP["descuento"], dd)].sort_values("error_mediano_pct")
        x, y = t["error_mediano_pct"].to_numpy(), t["ganancia_por_oferta"].to_numpy()
        i = int(np.where((y[:-1] > 0) & (y[1:] <= 0))[0][0])
        return x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1])

    c10, c20 = compra(0.10, 0.10), compra(0.20, 0.10)
    eq10, eq5 = equilibrio(0.10), equilibrio(0.05)
    acep, pierde = int(round(c10["aceptadas_pct"])), int(round(c10["pagan_de_mas_pct_ofertas"]))
    gana = acep - pierde
    texto(s, 0.55, 1.3, 4.6, 0.4, "100 ofertas en Chapinero fuera, con margen de 10 %", size=13, color="muted")
    waffle(s, 0.6, 1.8, 0.36, ["rojo"] * pierde + ["azul"] * gana + ["gris_claro"] * (100 - acep), gap=0.06)
    ley = [("rojo", f"{pierde} compran y pagan de más: {n(c10['perdida_media_si_paga_de_mas'], 0)} millones en promedio"),
           ("azul", f"{gana} compran y ganan: {n(c10['ganancia_media_si_gana'], 0)} millones en promedio"),
           ("gris_claro", f"{100 - acep} no aceptan")]
    for i, (c, t) in enumerate(ley):
        yl = 6.08 + i * 0.29
        forma(s, MSO_SHAPE.RECTANGLE, 0.6, yl + 0.06, 0.18, 0.18, relleno=c)
        texto(s, 0.85, yl, 4.45, 0.3, t, size=12, color="tinta", anchor="middle")
    _formula_vieja = IMG / "f_oferta.png"
    from PIL import Image
    w_px, h_px = Image.open(_formula_vieja).size
    imagen(s, _formula_vieja, 5.45, 1.32, h=0.46, alt="Regla de oferta y aceptación")
    texto(s, 5.35, 1.85, 7.45, 1.35, [
        [("Oᵢ: oferta; m: margen de la empresa; ŷᵢ: predicción; d: descuento que acepta el dueño; Vᵢ: valor de la vivienda. "
          "Con m = d = 10 %, el dueño solo acepta si la predicción supera el valor: ", {}),
         ("justo donde el modelo sobrestimó.", {"color": "rojo"})],
        f"Resultado: {n(c10['ganancia_por_oferta'], 1)} millones por oferta hecha, aceptada o no (−13,2 sin el ×1,08). Con "
        f"margen de 20 % se compran {n(c20['aceptadas_pct'], 0)} de cada 100 y aún se pierden "
        f"{n(-c20['ganancia_por_oferta'], 1)} millones por oferta."], size=13, color="tinta", espacio=4)
    g10 = GP[(GP["validacion"] == "Chapinero fuera") & np.isclose(GP["descuento"], 0.10)].sort_values("error_mediano_pct")
    g05 = GP[(GP["validacion"] == "Chapinero fuera") & np.isclose(GP["descuento"], 0.05)].sort_values("error_mediano_pct")
    err = L_CH.error_abs_mediano
    ch = lineas_xy(s, 5.25, 3.3, 4.45, 3.65, [
        dict(nombre="Dueños aceptan 10 % menos", x=list(g10["error_mediano_pct"]), y=list(g10["ganancia_por_oferta"]),
             color="azul", ancho=2.5),
        dict(nombre="Dueños aceptan 5 % menos", x=list(g05["error_mediano_pct"]), y=list(g05["ganancia_por_oferta"]),
             color="naranja", ancho=2.5),
        dict(nombre="Hoy", x=[err, err], y=[-30, 30], color="gris", ancho=1.25, guiones=True),
        dict(nombre="Equilibrio", x=[2, 14], y=[0, 0], color="gris_claro", ancho=1.0)],
        leyenda=True, minx=2, maxx=14, miny=-30, maxy=30, fmt_x="0", fmt_y=MENOS, titulo_x="Error mediano simulado (%)",
        titulo_y="Ganancia por oferta (millones)", unidad_x=2, unidad_y=10, pos_leyenda="arriba")
    etiquetas_puntos(ch, [None, None, [None, f"hoy: {n(err, 1)} %"], None], size=12)
    ocultar_leyenda(ch, [2, 3])
    ch.plots[0].series[2].points[1].data_label.position = XL_LABEL_POSITION.LEFT
    tarjeta(s, 9.9, 3.35, 2.9, 1.75, "Precisión necesaria", [
        f"Con margen de 10 %, la regla deja de perder solo si el error mediano baja a cerca de {n(eq10, 0)} % (o {n(eq5, 0)} % "
        "si los dueños aceptan solo 5 % menos)."], tipo="recordar")
    tarjeta(s, 9.9, 5.22, 2.9, 1.73, "Supuestos", [
        "El valor de cada vivienda es su precio publicado; el dueño solo ve la oferta, como en Zillow Offers. La curva reduce "
        "todos los errores en la misma proporción."], tipo="neutra")
    FIN[id(s)] = 6.95
    notas = ("Ahora traducimos el error a plata. Supongamos que la empresa ofrece 10 % menos que la predicción y que el dueño "
             "acepta si la oferta es al menos el 90 % de lo que vale su vivienda. Entonces el dueño solo acepta cuando la "
             f"predicción está por encima del valor real, es decir, justo cuando el modelo se equivocó hacia arriba. De 100 "
             f"ofertas en Chapinero, se aceptan {acep}: en {pierde} la empresa paga de más, en promedio "
             f"{n(c10['perdida_media_si_paga_de_mas'], 0)} millones, y solo en {gana} gana. En total pierde "
             f"{n(-c10['ganancia_por_oferta'], 1)} millones por oferta. Subir el margen reduce la pérdida pero también las "
             f"compras. La curva dice cuánta precisión haría falta: un error mediano de cerca de {n(eq10, 0)} %, frente al "
             f"{n(err, 1)} % de hoy. Para ofertar conviene la predicción sin el 1,08, que pierde menos. Es la trampa de Zillow: "
             "con un modelo así, los vendedores eligen las ofertas que le convienen a ellos.")
    return s, notas, eq10


def _recomendacion(deck, eq10):
    s = _nueva(deck, "Usar el modelo para filtrar y negociar, no para ofertar sin tasación", "Recomendación", "Recomendación",
               oscura=True)
    err = L_CH.error_abs_mediano
    col = [("Usarlo para", "azul_claro", "azul", "0,87", "correlación de rangos (Spearman) entre predicción y precio en los "
            "307 avisos de Chapinero fuera", [
                f"Ordenar y filtrar avisos y dar una referencia de precio (error mediano de {n(err, 1)} %).",
                "Usar la brecha entre predicción y precio pedido como alerta, no como prueba de una ganga: con ese error, una "
                "brecha grande puede ser error del modelo."]),
           ("No usarlo para", "rojo_claro", "rojo", f"{n(err, 1)} %", "de error mediano al predecir Chapinero sin haberlo "
            "visto", [
                "Ofertas automáticas: los dueños aceptan cuando el modelo sobrestima y la regla pierde dinero.",
                "Confiar en un aviso por estar dentro del rango del entrenamiento: en porcentaje, yerra igual.",
                "Ofertar con el ×1,08: sube las predicciones altas y pierde más (−14,8 frente a −13,2 millones por oferta)."]),
           ("Cómo hacerlo", "tinte", "navy", f"{n(eq10, 0)} %", "de error mediano haría falta para no perder con margen de 10 %",
            ["Toda alerta pasa por comparables y tasación antes de ofertar.",
             "La oferta se ancla en el menor valor entre la predicción con margen y el precio pedido.",
             "Siguiente paso: predecir cuantiles para ofertar de forma conservadora y sumar variables de imágenes satelitales "
             "con una red convolucional preentrenada, que no necesitan precios de Chapinero."])]
    for i, (tit, fondo, borde, cifra_, etq_, items) in enumerate(col):
        x0 = 0.55 + i * 4.13
        forma(s, MSO_SHAPE.ROUNDED_RECTANGLE, x0, 1.4, 3.95, 5.5, relleno=fondo, radio=0.04)
        texto(s, x0 + 0.2, 1.5, 3.6, 0.5, tit, size=20, color=borde)
        texto(s, x0 + 0.2, 1.98, 3.6, 0.75, cifra_, size=40, color=borde)
        texto(s, x0 + 0.2, 2.75, 3.55, 0.75, etq_, size=13, color="muted")
        linea(s, x0 + 0.2, 3.55, x0 + 3.75, 3.55, color="gris_claro")
        texto(s, x0 + 0.2, 3.65, 3.6, 3.2, items, size=14, color="tinta", espacio=8)
    FIN[id(s)] = 6.9
    notas = ("Nuestra recomendación. Sí usar el modelo para ordenar y filtrar avisos y como referencia de precio: ordena bien, "
             "con una correlación de rangos de 0,87 entre predicción y precio en Chapinero. No usarlo para ofertar de forma "
             f"automática: con un error mediano de {n(err, 1)} %, los dueños aceptan justo cuando el modelo se equivoca hacia "
             "arriba y la empresa pierde plata; tampoco confiar en un aviso solo porque se parece al entrenamiento, ni ofertar "
             "con el ajuste por 1,08. Cómo hacerlo: toda alerta pasa por comparables y tasación, y la oferta se ancla en el "
             "menor valor entre la predicción con margen y el precio pedido. El siguiente paso es predecir cuantiles para "
             "ofertar de forma conservadora y sumar variables de imágenes satelitales, que no necesitan precios de Chapinero.")
    return s, notas


def _fuentes(deck):
    s = _nueva(deck, "Fuentes", "Fuentes", "Fuentes")
    refs = ["Datos: Properati (avisos de la competencia de Kaggle); Catastro de Bogotá, UAECD (valor de referencia del "
            "suelo, avalúos, pisos y estratificación); OpenStreetMap (extracto de BBBike); Google Earth Engine (SRTM, "
            "Sentinel-2, VIIRS y GHSL).",
            "Ash, E. y Hansen, S. (2023). Text algorithms in economics. Annual Review of Economics, 15, 659-688.",
            "Bird, S., Klein, E. y Loper, E. (2009). Natural Language Processing with Python. O'Reilly.",
            "Blei, D. M., Ng, A. Y. y Jordan, M. I. (2003). Latent Dirichlet allocation. Journal of Machine Learning Research, 3, "
            "993-1022.",
            "Cañete, J. y otros (2020). Spanish pre-trained BERT model and evaluation data. PML4DC, ICLR.",
            "Departamento Nacional de Planeación, Unidad de Científicos de Datos (2020). ConTexto: librería de procesamiento de "
            "texto en español. Licencia MIT.",
            "Gentzkow, M., Kelly, B. y Taddy, M. (2019). Text as data. Journal of Economic Literature, 57(3), 535-574.",
            "James, G., Witten, D., Hastie, T. y Tibshirani, R. (2021). An Introduction to Statistical Learning, segunda edición. "
            "Springer, caps. 8 y 10.",
            "Ke, G. y otros (2017). LightGBM: a highly efficient gradient boosting decision tree. NeurIPS 30.",
            "Linnenbrink, J., Milà, C., Ludwig, M. y Meyer, H. (2024). kNNDM CV. Geoscientific Model Development, 17, 5897-5912.",
            "Lundberg, S. M. y Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS 30.",
            "Milà, C., Mateu, J., Pebesma, E. y Meyer, H. (2022). Nearest neighbour distance matching LOO CV. Methods in Ecology "
            "and Evolution, 13(6), 1304-1316.",
            "Roberts, D. R. y otros (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or "
            "phylogenetic structure. Ecography, 40(8), 913-929.",
            "Rosen, S. (1974). Hedonic prices and implicit markets. Journal of Political Economy, 82(1), 34-55.",
            "Shapley, L. S. (1953). A value for n-person games. Contributions to the Theory of Games II, 307-317.",
            "Stanford GSB Insights (2021). Flip Flop: Why Zillow's algorithmic home buying venture imploded."]
    texto(s, 0.55, 1.3, 12.25, 5.65, refs, size=13, color="tinta", espacio=3)
    FIN[id(s)] = 6.95
    notas = "Referencias citadas en la presentación."
    return s, notas


def resultados_mejor_modelo(deck):
    """Lámina de resultados del mejor modelo (va como lámina 2 de best, justo después de la portada)."""
    from scipy.stats import spearmanr
    s = _nueva(deck, "El modelo ordena bien Chapinero, pero no basta para ofertar sin tasación", "Resultados", "Resultados")
    texto(s, 0.55, 1.28, 12.25, 0.62, "Mejor modelo: LightGBM, un boosting de árboles de decisión con 162 variables por aviso, "
          "promedio de 5 corridas, ajuste ×1,08 y gemelos de texto. Medido en los 307 avisos de Chapinero, predichos con un "
          "modelo entrenado sin ellos.", size=14, color="tinta")
    dsp = E["dispersion_chapinero"]
    pr, re_ = np.array(dsp["pred_x108"]), np.array(dsp["real"])
    rho = spearmanr(pr, re_).correlation
    c10 = COMP[(COMP["validacion"] == "Chapinero fuera") & np.isclose(COMP["descuento"], 0.10) &
               np.isclose(COMP["margen"], 0.10)].iloc[0]
    ic = BO["lightgbm"]
    pos = CLAS.sort_values("privado").reset_index(drop=True)
    puesto = int(pos.index[pos["equipo"] == "Grupo_02"][0]) + 1
    stats = [(n(MAE["chapinero"]), "azul", f"millones de error absoluto medio por aviso (intervalo de 95 %: {n(ic['ic_bajo'], 0)} "
              f"a {n(ic['ic_alto'], 0)}); con el ajuste ×1,08, {n(BO['lightgbm_x108']['mae'])}"),
             (f"{n(L_CH.error_abs_mediano, 1)} %", "azul", "de error mediano en porcentaje del precio: la mitad de los avisos "
              "yerra menos que eso"),
             (n(rho, 2), "azul", "de correlación de rangos entre predicción y precio: ordena bien los avisos de barato a caro"),
             (f"{n(L_CH.sobre_10, 0)} %", "rojo", "de los avisos queda más de 10 % por encima de su precio: comprar con esa "
              "predicción es pagar de más"),
             (n(c10["ganancia_por_oferta"], 1), "rojo", "millones por oferta si se oferta en automático 10 % por debajo de la "
              "predicción: los dueños aceptan cuando sobrestima"),
             (n(P02, 1), "navy", f"millones de error absoluto medio en Kaggle, tabla privada: puesto {puesto} de {len(CLAS)} "
              "equipos")]
    for k, (v_, c_, e_) in enumerate(stats):
        x0, y0 = 0.55 + (k % 3) * 2.35, 2.05 + (k // 3) * 1.75
        texto(s, x0, y0, 2.25, 0.7, v_, size=34, color=c_)
        texto(s, x0, y0 + 0.66, 2.2, 1.05, e_, size=12, color="muted")
    texto(s, 7.75, 1.98, 5.05, 0.35, "Predicho, con el ajuste ×1,08, frente a real (millones)", size=13, color="muted")
    lineas_xy(s, 7.65, 2.3, 5.2, 3.3, [
        dict(nombre="Avisos", x=list(pr), y=list(re_), color="azul", linea=False, marcador=True, tam=4),
        dict(nombre="Predicción igual al precio", x=[200, 1800], y=[200, 1800], color="gris", ancho=1.5, guiones=True)],
        leyenda=False, minx=200, maxx=1800, miny=200, maxy=1800, fmt_x="#,##0", fmt_y="#,##0", titulo_x="Predicción",
        titulo_y="Precio real", unidad_x=400, unidad_y=400)
    tarjeta(s, 0.55, 5.7, 12.25, 1.25, "¿Se puede confiar en el modelo para comprar?", [
        "Para ordenar, filtrar y negociar, sí. Para ofertar en automático, no: con este error, los dueños aceptan justo "
        "cuando el modelo sobrestima. Toda compra pasa por comparables y tasación."], tipo="neutra", maximo=16)
    FIN[id(s)] = 6.95
    notas = ("Empezamos por el resultado. Nuestro mejor modelo es un LightGBM, un boosting de árboles de decisión, con 162 "
             "variables por aviso. Como la prueba de Kaggle no trae precios, lo medimos así: entrenamos sin los 307 avisos de "
             "Chapinero que tiene el entrenamiento y los predijimos. Se equivoca en promedio "
             f"{n(MAE['chapinero'], 0)} millones por aviso; en porcentaje, la mitad de los avisos yerra menos de "
             f"{n(L_CH.error_abs_mediano, 0)} %. Ordena bien: la correlación de rangos entre predicción y precio es "
             f"{n(rho, 2)}. Pero el riesgo es claro: el {n(L_CH.sobre_10, 0)} % de los avisos queda más de 10 % por encima de "
             "su precio, y si la empresa ofertara en automático perdería unos "
             f"{n(-c10['ganancia_por_oferta'], 0)} millones por oferta, porque los dueños aceptan justo cuando el modelo "
             f"sobrestima. La respuesta: sirve para ordenar, filtrar y negociar, no para ofertar sin tasación. En Kaggle "
             f"quedamos en el puesto {puesto} de {len(CLAS)}, con {n(P02, 1)} millones. El resto de la presentación muestra "
             "cómo llegamos a esto.")
    return s, notas


def etapas2a4(deck):
    """Crea las láminas de las etapas 2 a 4, la recomendación y las fuentes; devuelve una lista de (lámina, notas)."""
    out = []
    for f in (_general_arbol, _general_boosting, _boosting, _entrada, _corte, _objetivo, _hiperparametros, _busqueda, _parada, _arbol,
              _shap_importancia, _shap_enjambre,
              _shap_cascada, _shap_dependencia, _validar, _esquemas, _resultado_validacion, _metricas, _predicho_real, _vecinos,
              _pronostico, _calibracion, _lados, _fuera_rango):
        out.append(f(deck))
    s, notas, eq10 = _zillow(deck)
    out.append((s, notas))
    out.append(_recomendacion(deck, eq10))
    out.append(_fuentes(deck))
    for s, notas in out:
        deck.notas(s, notas)
    return out
