"""22 · Construye 03_slides/compare_equipo_02.pptx (Algorithm Comparison) con python-pptx.

Cifras: 00_data/processed/laminas/ (cv_familias.csv, envios_equipo.csv, ppt_bootstrap2.json, ppt_lados.csv; scripts 12 a
19) y 00_data/processed/modelos/cv_clase_oof_*_{sin,con}.parquet (script 07); imágenes: 02_outputs/figures/laminas/. Correcciones de los revisores (revision/informes/) aplicadas, incluidas las de
la v3 (requisitos_v3_opus.md, cifras_v3_opus.md y disenio_v3_opus.md). Las láminas de datos y la tabla de familias del
apéndice vienen de laminas_comunes.py, las mismas de best.

Antes document/presentaciones/construir_compare_ppt.py. Las referencias a revision/informes/ son a los informes de revisión de
las láminas, que no forman parte del repositorio. Exportar a PDF: script 23 (opcional, Windows con PowerPoint).
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

from functions.laminas_comunes import (chapinero_otro_mercado, dependencia_espacial, error_por_familia, texto_a_matriz,
                             texto_apendice, variables_hedonicas)
from functions.ppt_lib import (Deck, _ejes, _estilo_grafico, barras, etiquetas_puntos, flecha, forma, imagen, linea, lineas_xy, nodo,
                     rgb, tabla, tarjeta, texto)

from functions.entorno_ps3 import DIR_MODELOS  # noqa: E402
from functions.laminas_entorno import DATOS as D, IMG, SALIDA_PPT as OUT  # noqa: E402  (SALIDA_PPT: otra carpeta de salida)
# SALIDA_PPT permite guardar versiones de trabajo en otra carpeta (por ejemplo, mientras PowerPoint tiene abiertas las finales).


def n(x, d=1, signo=False):
    s = f"{abs(x):,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if x < 0 and float(f"{abs(x):.{d}f}") != 0:
        return "−" + s
    return ("+" + s) if signo and float(f"{abs(x):.{d}f}") != 0 else s


cvf = pd.read_csv(D / "cv_familias.csv")
env = pd.read_csv(D / "envios_equipo.csv")
bo = json.load(open(D / "ppt_bootstrap2.json", encoding="utf-8"))
lados = pd.read_csv(D / "ppt_lados.csv")
M = {(r.familia, r.esquema): r.mae for r in cvf.itertuples()}
S = {(r.familia, r.esquema): r.sesgo for r in cvf.itertuples()}
AL, CL, CH = "CV aleatoria", "Clústeres k-means", "Chapinero fuera"
FAM = [("lightgbm", "LightGBM"), ("catboost", "CatBoost"), ("random_forest", "Random Forest"), ("cart", "CART"),
       ("nn_atencion", "Red con atención"), ("elastic_net", "Elastic Net"), ("ols", "Regresión lineal")]
priv = dict(zip(env["archivo"], env["privado"]))
publ = dict(zip(env["archivo"], env["publico"]))
P01, P02 = priv["envio_01_lightgbm_con_catastro"], priv["envio_02_lightgbm_con_catastro_x108"]
U01, U02 = publ["envio_01_lightgbm_con_catastro"], publ["envio_02_lightgbm_con_catastro_x108"]
PREVIO = env[env["orden"] < 25]["privado"].min()
SOBRE_SIN = lados[(lados["esquema"] == "Chapinero fuera") & (lados["version"] == "Sin ajuste")]["sobre_10"].iloc[0]
SOBRE_CON = lados[(lados["esquema"] == "Chapinero fuera") & (lados["version"] == "Con ajuste x1,08")]["sobre_10"].iloc[0]
cat = {}
for f in ("lightgbm", "catboost"):
    for c in ("sin", "con"):
        o = pd.read_parquet(DIR_MODELOS / f"cv_clase_oof_{f}_{c}.parquet")
        o = o[o["esquema"].str.startswith("Chapinero")]
        cat[(f, c)] = float(np.mean(np.abs(o["pred"] - o["real"])))
rej = env[(env["autor"] == "César Vallejo") & (env["familia"] == "Boosting")]
assert len(rej) == 16
deck = Deck("Algorithm Comparison")
NOTAS = {}


def nota(t):
    NOTAS[len(deck.pies) - 1] = t


# ------------------------------------------------------------------------------------------------------------------
# 1. Portada
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva(oscura=True)
texto(s, 0.7, 0.9, 6.3, 0.4, "Problem Set 3. Making Money with ML?", size=16, color="cielo")
texto(s, 0.7, 1.45, 6.4, 1.6, "¿Por qué ganó el boosting?", size=40, color="blanco")
texto(s, 0.7, 3.0, 6.3, 1.6, "Algorithm Comparison: seis familias y siete modelos bajo las mismas reglas para predecir "
      "precios en Chapinero", size=19, color="cielo")
linea(s, 0.7, 4.95, 3.2, 4.95, color="naranja", ancho=2.5)
texto(s, 0.7, 5.15, 6.3, 1.0, ["Equipo 02: Jhon Sebastián Vela Salcedo, Harold Santiago Aguilera Mulfo y César Augusto "
                               "Vallejo Quiroga", "25 de noviembre de 2026"], size=15, color="blanco", espacio=8)
imagen(s, IMG / "portada_mapa.png", 7.45, 0.45, h=5.95, alt="Mapa de Bogotá con los avisos de entrenamiento y de prueba")
texto(s, 7.45, 6.45, 5.4, 0.5, "Gris: avisos de entrenamiento. Naranja: avisos de prueba (94 % en la localidad de "
      "Chapinero).", size=12, color="cielo")
nota("Comparamos seis familias de algoritmos para predecir el precio de vivienda en Chapinero, con las mismas variables y las "
     "mismas validaciones. La pregunta es por qué gana el boosting y qué se sacrifica con cada alternativa.")

# ------------------------------------------------------------------------------------------------------------------
# 2. El mejor modelo
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("El mejor modelo: boosting de árboles con texto, entorno y Catastro", etiqueta="El mejor modelo",
               seccion="Mejor modelo")
tarjeta(s, 0.55, 1.5, 5.9, 2.45, "El modelo ganador", [
    "Boosting de árboles con LightGBM, pérdida cuadrática sobre el precio en millones de pesos y 162 variables; el número "
    "de árboles se elige con una partición espacial del propio entrenamiento y se promedian 5 corridas con distinta "
    "semilla.",
    "Ajuste final ỹ: la distancia a la media del entrenamiento (ȳ = 654) se multiplica por 1,08, porque el modelo se quedaba "
    "corto en los avisos caros. Gemelos de texto: en 78 avisos de prueba cuya descripción repite la de uno del "
    "entrenamiento, la predicción pesa 0,7 el precio de ese aviso y 0,3 la del modelo."],
    tipo="definicion", size_titulo=16)
imagen(s, IMG / "f_estiramiento.png", 0.7, 4.1, w=4.6, alt="Fórmula del ajuste final alrededor de la media")
filas = [("Medida", "Error absoluto medio, MAE (millones de pesos)"),
         ("Kaggle, tabla privada (80 %)", f"{n(P02)}: segundo de seis equipos"), ("Kaggle, tabla pública (20 %)", n(U02)),
         ("Chapinero fuera: entrenar sin los 307 avisos de Chapinero y predecirlos",
          f"{n(M[('lightgbm', CH)])} (140,8 con el ×1,08, optimista: el 1,08 se eligió aquí)")]
tabla(s, 0.55, 4.85, 5.9, filas, [3.2, 2.7], size=13, alto_fila=0.5)
texto(s, 6.9, 1.4, 5.9, 0.4, "Kaggle, tabla privada (millones)", size=13, color="muted")
ch = barras(s, 6.8, 1.75, 6.0, 3.35, ["Mejor envío previo del equipo", "Envío 01: LightGBM con Catastro", "Envío 02: con ajuste ×1,08"],
            [("Privado", [PREVIO, P01, P02], ["gris_claro", "azul_suave", "azul"])], minimo=0, maximo=230, gap=40, grid=False)
etiquetas_puntos(ch, [[n(PREVIO), n(P01), n(P02)]])
tarjeta(s, 6.8, 5.3, 6.0, 1.65, "La pregunta de esta presentación", [
    "Una empresa quiere comprar vivienda en Chapinero; si el modelo predice por encima del precio, paga de más. ¿Por qué este "
    "modelo le gana a las otras familias, y qué se sacrifica con cada alternativa?"], tipo="neutra", size_titulo=16)
nota(f"Pensemos en una empresa que compra vivienda en Chapinero: predecir de menos le cuesta un negocio; predecir de más le "
     "hace pagar caro. Empezamos por el mejor modelo: LightGBM con 162 variables, incluidas 11 de Catastro, y un ajuste final; "
     "si la descripción de un aviso de prueba repite la de uno del entrenamiento, la predicción se apoya sobre todo en el "
     "precio de ese aviso. En la tabla "
     f"privada de Kaggle obtuvo {n(P02)} millones de error absoluto medio, que abreviamos MAE, segundo de seis equipos, y bajó "
     f"{n(PREVIO - P02)} millones frente al mejor envío previo del equipo. Para entender por qué le gana a las demás familias, "
     "primero veamos los datos que reciben todas. Si preguntan: el 140,8 con el ajuste es algo optimista, porque el factor "
     "se eligió con esa misma validación; Kaggle lo confirmó fuera de muestra.")

# ------------------------------------------------------------------------------------------------------------------
# 2b y 2c. Los datos que reciben todas las familias (las mismas láminas de best)
# ------------------------------------------------------------------------------------------------------------------
s, t_ = chapinero_otro_mercado(deck, "Datos", "Datos",
                               contexto="Solo 307 de los 38.644 avisos de entrenamiento están en Chapinero. ")
nota(t_ + " Esto anticipa una diferencia entre familias: los árboles no extrapolan y las lineales sí.")
s, t_ = texto_a_matriz(deck, "Datos", "Datos")
nota(t_)
s, t_ = variables_hedonicas(deck, "Datos", "Datos", nota_extra="Todas las familias reciben estas mismas 162 variables.",
                            texto_extra="Todas las familias reciben estas mismas 162 variables.")
nota(t_)

# ------------------------------------------------------------------------------------------------------------------
# 3. Cómo predice cada familia (esquemas con formas)
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Cómo predice cada familia: seis familias, siete modelos", etiqueta="Seis familias", seccion="Seis familias")
COLW, GAP = 2.32, 0.15
desc = [("Regresión lineal y Elastic Net", "Una suma ponderada de las variables. Elastic Net encoge y selecciona los "
         "coeficientes (lasso y ridge).", f"{n(M[('ols', CH)])} y {n(M[('elastic_net', CH)])}"),
        ("CART", "Un solo árbol de regresión: reglas de sí o no que parten el mercado en hojas.", n(M[("cart", CH)])),
        ("Random Forest", "Muchos árboles con muestras y variables al azar; promedia sus predicciones.", n(M[("random_forest", CH)])),
        ("Boosting: LightGBM y CatBoost", "Árboles en secuencia: cada uno corrige el residuo de los anteriores.",
         f"{n(M[('lightgbm', CH)])} y {n(M[('catboost', CH)])}"),
        ("Red con atención", "Capas de neuronas; la atención pondera precio y características de 20 vecinos de "
         "entrenamiento.", n(M[("nn_atencion", CH)]))]


def mini_arbol(cx, y, w, h, color="azul", hojas=True):
    forma(s, MSO_SHAPE.ROUNDED_RECTANGLE, cx - w * 0.18, y, w * 0.36, h * 0.24, relleno=color, radio=0.2)
    for dx in (-0.25, 0.25):
        hx = cx + dx * w
        linea(s, cx, y + h * 0.24, hx, y + h * 0.5, color="gris", ancho=1)
        forma(s, MSO_SHAPE.ROUNDED_RECTANGLE, hx - w * 0.15, y + h * 0.5, w * 0.3, h * 0.2, relleno=color, radio=0.2)
        if hojas:
            for dx2 in (-0.11, 0.11):
                lx = hx + dx2 * w
                linea(s, hx, y + h * 0.7, lx, y + h * 0.86, color="gris", ancho=0.75)
                forma(s, MSO_SHAPE.OVAL, lx - w * 0.04, y + h * 0.86, w * 0.08, w * 0.08, relleno="gris_claro")


for i, (tit, txt, mae_txt) in enumerate(desc):
    x0 = 0.55 + i * (COLW + GAP)
    forma(s, MSO_SHAPE.ROUNDED_RECTANGLE, x0, 1.45, COLW, 5.5, relleno="tinte", radio=0.04)
    texto(s, x0 + 0.12, 1.55, COLW - 0.24, 0.65, tit, size=15, color="navy")
    ya, ha = 2.3, 1.9
    cx = x0 + COLW / 2
    if i == 0:
        linea(s, x0 + 0.3, ya + ha, x0 + COLW - 0.2, ya + ha, color="gris")
        linea(s, x0 + 0.3, ya + 0.1, x0 + 0.3, ya + ha, color="gris")
        pts = [(0.1, 0.25), (0.22, 0.3), (0.3, 0.45), (0.42, 0.42), (0.55, 0.6), (0.62, 0.52), (0.75, 0.72), (0.85, 0.8)]
        for px, py in pts:
            forma(s, MSO_SHAPE.OVAL, x0 + 0.3 + px * (COLW - 0.6) - 0.05, ya + ha - py * (ha - 0.2) - 0.05, 0.1, 0.1, relleno="gris")
        linea(s, x0 + 0.35, ya + ha - 0.2 * (ha - 0.2), x0 + COLW - 0.25, ya + ha - 0.88 * (ha - 0.2), color="azul", ancho=2.5)
    elif i == 1:
        mini_arbol(cx, ya + 0.1, COLW - 0.3, ha - 0.1)
    elif i == 2:
        for k in range(3):
            mini_arbol(x0 + 0.42 + k * 0.74, ya + 0.05, 0.75, 1.15, hojas=False)
        for k in range(3):
            linea(s, x0 + 0.42 + k * 0.74, ya + 0.95, cx, ya + 1.45, color="gris", ancho=1)
        nodo(s, cx - 0.62, ya + 1.45, 1.24, 0.42, ["promedio"], relleno="azul_claro", borde="azul", sizes=(12,), colores=("azul",))
    elif i == 3:
        for k in range(3):
            mini_arbol(x0 + 0.42 + k * 0.74, ya + 0.05, 0.75, 1.15, color=["azul_suave", "azul", "navy"][k], hojas=False)
            if k < 2:
                texto(s, x0 + 0.67 + k * 0.74, ya + 0.3, 0.3, 0.35, "+", size=16, color="navy", align="center")
        linea(s, x0 + 0.3, ya + 1.55, x0 + COLW - 0.3, ya + 1.55, color="navy", ancho=1.5)
        texto(s, x0 + 0.2, ya + 1.6, COLW - 0.4, 0.35, "cada árbol corrige al anterior", size=12, color="muted", align="center")
    else:
        capas = [4, 3, 1]
        pos = {}
        for c_, m_ in enumerate(capas):
            for r_ in range(m_):
                pos[(c_, r_)] = (x0 + 0.35 + c_ * 0.62, ya + 0.25 + (r_ + (4 - m_) / 2) * 0.36)
        for c_ in range(2):
            for r_ in range(capas[c_]):
                for r2 in range(capas[c_ + 1]):
                    (a1, b1), (a2, b2) = pos[(c_, r_)], pos[(c_ + 1, r2)]
                    linea(s, a1 + 0.08, b1 + 0.08, a2 + 0.08, b2 + 0.08, color="gris_claro", ancho=0.75)
        for (c_, r_), (px, py) in pos.items():
            forma(s, MSO_SHAPE.OVAL, px, py, 0.16, 0.16, relleno="azul" if c_ < 2 else "navy")
        for k in range(4):
            forma(s, MSO_SHAPE.OVAL, x0 + COLW - 0.45, ya + 0.2 + k * 0.38, 0.2, 0.2, relleno="naranja")
            linea(s, x0 + COLW - 0.45, ya + 0.3 + k * 0.38, pos[(1, 1)][0] + 0.16, pos[(1, 1)][1] + 0.08, color="naranja", ancho=0.75)
        texto(s, x0 + COLW - 0.75, ya + 1.65, 0.7, 0.3, "vecinos", size=12, color="naranja", align="center")
    texto(s, x0 + 0.12, 4.35, COLW - 0.24, 1.55, txt, size=14, color="tinta")
    texto(s, x0 + 0.12, 5.95, COLW - 0.24, 0.35, "MAE en Chapinero fuera", size=12, color="muted")
    texto(s, x0 + 0.12, 6.25, COLW - 0.24, 0.55, mae_txt, size=20 if " y " in mae_txt else 24,
          color="azul" if i == 3 else "navy")
nota("Antes de comparar, qué hace cada familia. La regresión lineal suma variables con pesos fijos; Elastic Net además encoge "
     "y selecciona. CART es un solo árbol de reglas. Random Forest promedia muchos árboles distintos. El boosting pone árboles "
     "en secuencia, cada uno corrigiendo el error del anterior; lo corrimos con LightGBM y con CatBoost. La red con atención "
     "aprende capas de neuronas y mira los precios de los vecinos. Abajo, el error de cada una al predecir Chapinero sin "
     "haberlo visto.")

# ------------------------------------------------------------------------------------------------------------------
# 4. Las mismas reglas
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Seis familias bajo las mismas reglas", etiqueta="Seis familias", seccion="Seis familias")
filas = [("Familia", "Especificación", "Búsqueda de hiperparámetros"),
         ("Regresión lineal", "mínimos cuadrados con las 162 variables estandarizadas e indicadores de faltantes", "no tiene"),
         ("Elastic Net", "lasso y ridge combinados", "rejilla de 4 valores de la mezcla por 30 de λ, con 3 particiones "
                                                     "espaciales internas"),
         ("CART", "un árbol de regresión, mínimo 20 avisos por hoja", "profundidad en {6, 8, 10, 12, 15}, elegida en la "
                                                                     "partición interna"),
         ("Random Forest", "300 árboles, 30 % de las variables en cada corte, mínimo 5 por hoja",
          "valores fijos de uso común, sin búsqueda"),
         ("Boosting", "LightGBM y CatBoost", "árboles por parada temprana (partición interna); en LightGBM, además, variantes "
                                             "de objetivo, hojas y mínimo por hoja comparadas con validación espacial"),
         ("Red con atención", "codificador de 256 y 128 neuronas más atención sobre 20 vecinos de entrenamiento",
          "épocas por parada temprana (partición interna)")]
tabla(s, 0.55, 1.45, 12.25, filas, [2.3, 5.6, 4.35], size=13, alto_fila=0.52)
tarjeta(s, 0.55, 5.3, 5.95, 1.65, "Las mismas reglas para todas", [
    "Mismas 162 variables (la red suma precios de 20 vecinos) y pliegues; pérdida cuadrática salvo la red (absoluta); "
    "puntaje de texto recalculado en cada pliegue. Hiperparámetros y parada temprana (parar si el error deja de bajar) "
    "se eligen dentro de cada pliegue, con una partición interna hecha como la externa: al azar en la CV aleatoria y por "
    "lugar en las espaciales (Elastic Net, por bloques)."], tipo="definicion")
tarjeta(s, 6.65, 5.3, 6.15, 1.65, "Tres validaciones", [
    "CV aleatoria: pliegues al azar, con vecinos en el entrenamiento. Clústeres k-means: pliegues por 10 zonas de la ciudad "
    "(k-means sobre las coordenadas). Chapinero fuera: dejar una localización por fuera (Lecture 14), entrenando sin los 307 "
    "avisos de Chapinero y prediciéndolos. Es la principal, porque imita la prueba: una zona que el modelo no vio."],
    tipo="neutra")
nota("Para que la comparación sea justa, todas las familias usan las mismas 162 variables y los mismos pliegues. Los "
     "hiperparámetros que se afinan, como λ, la profundidad o el número de árboles o épocas, se eligen dentro de cada pliegue "
     "con una partición interna que repite la lógica del esquema externo, como en la lectura de validación cruzada "
     "espacial: al azar en la CV aleatoria y por lugar en las espaciales; Elastic Net usa siempre tres particiones por "
     "bloques. Por eso la CV aleatoria, al elegir hiperparámetros, premia al modelo que memoriza el ruido local (Lecture 14). "
     "Random Forest usa valores fijos, sin búsqueda. Dos diferencias: "
     "la red usa además los precios de sus vecinos y una pérdida absoluta. La rejilla explícita está en Elastic Net, 4 por 30, y "
     "en CART, 5 profundidades; en LightGBM comparamos variantes.")

# ------------------------------------------------------------------------------------------------------------------
# 5. Resultado
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Resultado: el boosting gana en las tres validaciones", etiqueta="Resultado", seccion="Resultado")
cats = [nom for _, nom in FAM]
ser = []
for esq, c in ((AL, "gris_claro"), (CL, "gris"), (CH, "naranja")):
    ser.append((esq, [M.get((f, esq), None) for f, _ in FAM], c))
cd = CategoryChartData()
cd.categories = cats
for nom, vals, _ in ser:
    cd.add_series(nom, vals)
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.45), Inches(1.45), Inches(7.9), Inches(5.05), cd)
ch = gf.chart
_estilo_grafico(ch, 12)
ch.has_title = False
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.TOP
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(12)
ch.plots[0].gap_width, ch.plots[0].overlap = 45, -5
for sr, (_, vals, c) in zip(ch.plots[0].series, ser):
    sr.format.fill.solid()
    sr.format.fill.fore_color.rgb = rgb(c)
_ejes(ch, valor_visible=False, minimo=0, maximo=230, invertir_cat=True, grid=False)
etiquetas_puntos(ch, [[n(v) if v is not None else None for v in vals] for _, vals, _ in ser], size=12,
                 posicion=XL_LABEL_POSITION.OUTSIDE_END)
texto(s, 0.55, 6.6, 7.7, 0.35, "MAE por validación, millones. CatBoost no se corrió con la CV aleatoria.", size=12, color="muted")
tarjeta(s, 8.6, 1.5, 4.2, 2.35, "Lectura", [
    f"El boosting es el mejor en las tres validaciones: LightGBM en la aleatoria y en Chapinero fuera, CatBoost en clústeres "
    f"({n(M[('catboost', CL)])} frente a {n(M[('lightgbm', CL)])})."], tipo="neutra")
tarjeta(s, 8.6, 4.0, 4.2, 1.35, "Distancia a las demás", [
    "En Chapinero fuera, LightGBM le saca 6 millones a CatBoost, 15 a Random Forest y 41 a 42 a los lineales."], tipo="neutra")
tarjeta(s, 8.6, 5.5, 4.2, 1.45, "Optimismo de la CV aleatoria", [
    f"Promete un error {n(100 * (1 - M[('lightgbm', AL)] / P01), 0)} % menor que Kaggle ({n(M[('lightgbm', AL)])} frente a "
    f"{n(P01)})."], tipo="cuidado")
nota("Con las mismas reglas, el boosting gana en las tres validaciones: LightGBM en la aleatoria y en la que deja Chapinero "
     "fuera, y CatBoost, por menos de un millón, en clústeres. Los lineales quedan últimos. Fíjense en el nivel: con la CV "
     "aleatoria todos parecen mucho mejores de lo que serán en Kaggle.")

# ------------------------------------------------------------------------------------------------------------------
# 6. ¿Son reales las diferencias?
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("¿Son reales las diferencias? Sí, frente a cada familia", etiqueta="Resultado", seccion="Resultado")
texto(s, 0.55, 1.4, 8.2, 0.6, "Diferencia de MAE frente a LightGBM en Chapinero fuera (millones), con intervalo de confianza de "
      "95 % por bootstrap pareado (10.000 réplicas de los 307 avisos)", size=13, color="muted")
filas_f = [("CatBoost", "catboost"), ("Random Forest", "random_forest"), ("CART", "cart"), ("Red con atención", "nn_atencion"),
           ("Elastic Net", "elastic_net"), ("Regresión lineal", "ols")]
X0F, WF, LABF = 0.55, 8.2, 2.1
vmin, vmax = -5, 60
esc = lambda v: X0F + LABF + (v - vmin) / (vmax - vmin) * (WF - LABF - 1.9)  # noqa: E731
y0f, hf = 2.3, 0.6
for i, (nom, f) in enumerate(filas_f):
    yy = y0f + i * hf
    b = bo[f]
    texto(s, X0F, yy, LABF - 0.1, hf, nom, size=14, color="tinta", align="right", anchor="middle")
    linea(s, esc(b["dif_bajo"]), yy + hf / 2, esc(b["dif_alto"]), yy + hf / 2, color="gris", ancho=2.5)
    forma(s, MSO_SHAPE.OVAL, esc(b["dif"]) - 0.09, yy + hf / 2 - 0.09, 0.18, 0.18, relleno="azul")
    texto(s, esc(vmax) + 0.1, yy, 1.85, hf, f"{n(b['dif'], 1, True)} ({n(b['dif_bajo'], 1)} a {n(b['dif_alto'], 1)})",
          size=13, color="tinta", anchor="middle")
yb = y0f + len(filas_f) * hf
linea(s, esc(0), y0f - 0.1, esc(0), yb, color="navy", ancho=1.25)
linea(s, esc(vmin), yb, esc(vmax), yb, color="gris_claro")
for v in (0, 10, 20, 30, 40, 50, 60):
    texto(s, esc(v) - 0.35, yb + 0.05, 0.7, 0.3, n(v, 0), size=12, color="muted", align="center")
texto(s, esc(0) + 0.05, y0f - 0.45, 3.0, 0.3, "0: igual que LightGBM", size=12, color="navy")
tarjeta(s, 9.0, 1.5, 3.8, 2.6, "Lectura", [
    "El boosting con LightGBM supera a las otras cinco familias: todas las diferencias son positivas y ningún intervalo "
    "incluye el cero."], tipo="clave")
tarjeta(s, 9.0, 4.25, 3.8, 2.7, "La menor distancia", [
    "Es con CatBoost, la otra implementación de boosting: el salto grande es entre familias.",
    "El bootstrap trata los avisos como independientes; con vecinos que comparten ruido, los intervalos reales son algo más "
    "anchos, y el de CatBoost es el más frágil."], tipo="neutra")
nota("Con solo 307 avisos de Chapinero, ¿son reales estas diferencias? Remuestreamos los mismos avisos 10.000 veces y "
     "comparamos cada familia con LightGBM sobre los mismos avisos. Todas pierden y ningún intervalo toca el cero. La "
     "distancia menor es con CatBoost, que también es boosting: 6,3 millones.")

# ------------------------------------------------------------------------------------------------------------------
# 7. Orden con validación estándar y espacial (gráfico de puestos con formas)
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Coinciden en el ganador; la red pierde el segundo puesto sin Chapinero",
               etiqueta="Validación estándar frente a espacial", seccion="Validación estándar frente a espacial")
ESQ = [AL, CL, CH]
XS = [2.75, 4.6, 6.45]
rangos = {}
for esq in ESQ:
    vals = sorted([(M[(f, esq)], f) for f, _ in FAM if (f, esq) in M])
    for r_, (_, f) in enumerate(vals, 1):
        rangos[(f, esq)] = r_
ypos = lambda r: 2.15 + 0.6 * (r - 1)  # noqa: E731
for x_, e in zip(XS, ESQ):
    texto(s, x_ - 0.9, 1.45, 1.8, 0.4, e, size=14, color="navy", align="center")
ESTILO = {"lightgbm": ("azul", 3.5), "catboost": ("azul_suave", 2.5), "nn_atencion": ("naranja", 3.5)}
orden_dibujo = [f for f, _ in FAM if f not in ESTILO] + [f for f in ("catboost", "nn_atencion", "lightgbm")]
nombres = dict(FAM)
for f in orden_dibujo:
    color, ancho = ESTILO.get(f, ("gris_claro", 2.0))
    pts = [(x_, ypos(rangos[(f, e)])) for x_, e in zip(XS, ESQ) if (f, e) in rangos]
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        linea(s, x1, y1, x2, y2, color=color, ancho=ancho)
for f in orden_dibujo:
    color, _ = ESTILO.get(f, ("gris", 2.0))
    c_txt = color if f in ESTILO else "muted"
    for x_, e in zip(XS, ESQ):
        if (f, e) not in rangos:
            continue
        y_ = ypos(rangos[(f, e)])
        forma(s, MSO_SHAPE.OVAL, x_ - 0.08, y_ - 0.08, 0.16, 0.16, relleno=color if f in ESTILO else "gris")
        if e == AL:
            texto(s, 0.45, y_ - 0.17, x_ - 0.6, 0.34, f"{nombres[f]} {n(M[(f, e)])}", size=13, color=c_txt, align="right",
                  anchor="middle")
        elif e == CL:
            # El valor va pegado a la derecha de su punto, sobre fondo blanco (disenio_v3_opus.md, M1).
            forma(s, MSO_SHAPE.RECTANGLE, x_ + 0.11, y_ - 0.13, 0.56, 0.26, relleno="blanco")
            texto(s, x_ + 0.11, y_ - 0.13, 0.56, 0.26, n(M[(f, e)]), size=12, color=c_txt, anchor="middle", margen=0.02)
        else:
            texto(s, x_ + 0.15, y_ - 0.17, 2.1, 0.34, f"{nombres[f]} {n(M[(f, e)])}", size=13, color=c_txt, anchor="middle")
texto(s, 0.55, 6.4, 8.0, 0.55, "Puesto de cada familia (1 arriba) y su MAE en millones. CatBoost no se corrió con la CV "
      "aleatoria. Estándar: CV aleatoria; espaciales: clústeres k-means y Chapinero fuera.", size=12, color="muted")
tarjeta(s, 8.85, 1.5, 3.95, 1.45, "Sí para el ganador", [
    f"El boosting es primero en las tres (CatBoost supera a LightGBM en clústeres por {n(M[('lightgbm', CL)] - M[('catboost', CL)])})."],
    tipo="clave")
tarjeta(s, 8.85, 3.07, 3.95, 1.73, "No para el resto", [
    f"La red con atención es segunda en la CV aleatoria ({n(M[('nn_atencion', AL)])}); al dejar Chapinero fuera "
    f"({n(M[('nn_atencion', CH)])}) queda detrás de Random Forest y casi empatada con CART."], tipo="recordar")
tarjeta(s, 8.85, 4.92, 3.95, 2.03, "Por qué cae la red", [
    "La atención promedia los precios de 20 vecinos: aprovecha que lo cercano se parece (Lecture 14). Con pliegues aleatorios "
    "esos vecinos están a metros; sin Chapinero, los más cercanos están a unos 550 m y en otro mercado. La pérdida absoluta, "
    "además, apunta a "
    "la mediana, que en precios sesgados queda bajo la media."],
    tipo="aclaracion")
nota("¿La validación estándar y las espaciales eligen lo mismo? Para el ganador, sí: el boosting queda primero en las tres. Para el resto, no. La "
     "red con atención es segunda con pliegues aleatorios y queda detrás de Random Forest y CART cuando tiene que predecir Chapinero sin "
     "haberlo visto, porque depende de los precios de vecinos cercanos que allí no tiene.")

# ------------------------------------------------------------------------------------------------------------------
# 8. Kaggle
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("¿Qué validación refleja Kaggle? En el nivel, Chapinero fuera", etiqueta="Validación estándar frente a espacial",
               seccion="Validación estándar frente a espacial")
texto(s, 0.55, 1.4, 6.0, 0.4, "Mismo modelo (LightGBM, envío 01): validación frente a Kaggle", size=13, color="muted")
esq = [("CV aleatoria", M[("lightgbm", AL)]), ("Clústeres k-means", M[("lightgbm", CL)]), ("Chapinero fuera", M[("lightgbm", CH)]),
       ("Kaggle, tabla privada", P01), ("Kaggle, tabla pública", U01)]
ch = barras(s, 0.45, 1.75, 6.1, 2.8, [e for e, _ in esq], [("v", [v for _, v in esq], ["gris_claro", "gris", "naranja", "navy", "navy"])],
            minimo=0, maximo=200, gap=35, grid=False)
etiquetas_puntos(ch, [[n(v) for _, v in esq]])
mejores = (env[env["orden"] < 28].sort_values("privado").groupby("familia").head(1).sort_values("privado"))
nomf = {"Ensamble": "Ensamble (superlearner)"}
texto(s, 6.85, 1.4, 6.0, 0.4, "Mejor envío del equipo en Kaggle por familia (tabla privada)", size=13, color="muted")
ch = barras(s, 6.75, 1.75, 6.1, 2.8, [nomf.get(f, f) for f in mejores["familia"]],
            [("v", list(mejores["privado"]), ["azul" if f == "Boosting" else "gris_claro" for f in mejores["familia"]])],
            minimo=0, maximo=300, gap=35, grid=False)
etiquetas_puntos(ch, [[n(v) for v in mejores["privado"]]])
tarjeta(s, 0.55, 4.75, 4.0, 2.2, "El nivel", [
    f"Chapinero fuera queda a {n(P01 - M[('lightgbm', CH)], 0)} millones de Kaggle; la CV aleatoria, a "
    f"{n(P01 - M[('lightgbm', AL)], 0)}, por dos razones. Dependencia: el 64 % de sus avisos validados tiene uno de "
    "entrenamiento en la misma coordenada; si el ruido de dos vecinos tiene correlación ρ, se subestima el error cuadrático "
    f"en 2σ²ρ (Lecture 14). Mercado: los clústeres ya separan los vecinos y aún quedan {n(P01 - M[('lightgbm', CL)], 0)} "
    "por debajo de Kaggle."],
    tipo="clave")
tarjeta(s, 4.7, 4.75, 4.0, 2.2, "El orden entre familias", [
    "En el orden, Kaggle coincide con la CV aleatoria, pero cada envío usó variables distintas: ese orden no sirve para "
    "elegir la validación."], tipo="neutra")
tarjeta(s, 8.85, 4.75, 3.95, 2.2, "Comparación imperfecta", [
    "La red de Kaggle es la de clase (dos capas de 10 neuronas) y el ensamble combina varios modelos. Elastic Net no se envió "
    "a Kaggle: solo se evaluó en estas validaciones, donde empata con la regresión lineal."], tipo="recordar")
nota(f"¿Qué validación se parece a Kaggle? En el nivel, la que deja Chapinero fuera: {n(M[('lightgbm', CH)])} frente a {n(P01)} "
     f"en la tabla privada y {n(U01)} en la pública, mientras la CV aleatoria promete {n(M[('lightgbm', AL)])}. En el orden entre "
     "familias, Kaggle coincide con la CV aleatoria, pero esa comparación es imperfecta: nuestros envíos por familia usaron "
     "variables distintas y la red de Kaggle no es la red con atención. El enunciado pedía envíos de las seis familias y "
     "Elastic Net no se envió; la evaluamos con las mismas variables y pliegues que las demás: 185,2 en Chapinero fuera, igual "
     "que la regresión lineal, así que no habría cambiado el orden. Por qué la CV aleatoria es tan optimista: por dos "
     "razones. El ruido que comparten los vecinos no está en las variables, y una validación que deja vecinos en el "
     "entrenamiento lo descuenta del error (Lecture 14 y la lectura de validación cruzada espacial). Además, no tiene que "
     f"predecir otro mercado: los clústeres, sin vecinos cercanos, dan {n(M[('lightgbm', CL)], 0)}, todavía "
     f"{n(P01 - M[('lightgbm', CL)], 0)} por debajo de Kaggle; como advierten Ash y Hansen, el desempeño en los datos "
     "conocidos puede no representar el de un dominio nuevo.")

# ------------------------------------------------------------------------------------------------------------------
# 9. Por qué gana el boosting
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Por qué gana el boosting: forma flexible y corrección en secuencia", etiqueta="Por qué gana el boosting",
               seccion="Por qué gana el boosting")
texto(s, 0.55, 1.4, 5.8, 0.4, "Chapinero fuera, MAE (millones)", size=13, color="muted")
ch = barras(s, 0.45, 1.75, 5.9, 3.0, ["Un árbol (CART)", "Bosque promediado (Random Forest)", "Árboles en secuencia (LightGBM)"],
            [("v", [M[("cart", CH)], M[("random_forest", CH)], M[("lightgbm", CH)]], ["gris_claro", "gris", "azul"])],
            horizontal=False, minimo=0, maximo=200, gap=50, grid=False)
etiquetas_puntos(ch, [[n(M[("cart", CH)]), n(M[("random_forest", CH)]), n(M[("lightgbm", CH)])]])
texto(s, 6.8, 1.4, 6.0, 0.4, "Aporte del área a la predicción de LightGBM, valores de Shapley (millones; Lecture 13)", size=12,
      color="muted")
imagen(s, IMG / "dependencia_area.png", 6.95, 1.75, w=5.6, alt="Aporte SHAP del área en entrenamiento y en la prueba de Chapinero")
tarjeta(s, 0.55, 5.1, 5.8, 1.85, "Un árbol, un bosque y árboles en secuencia", [
    "Según ISLR (cap. 8), un árbol solo tiene mucha varianza y el bosque la reduce promediando árboles perturbados. El "
    "boosting, al corregir en secuencia los residuos con árboles simples, apunta sobre todo al sesgo. La varianza no se "
    "midió. El sesgo medio (predicción menos precio) solo mide el nivel: baja de "
    f"{n(S[('cart', CH)], 0)} con un árbol a {n(S[('random_forest', CH)], 0)} con el bosque y "
    f"{n(S[('lightgbm', CH)], 0)} con el boosting, pero los lineales tienen {n(S[('ols', CH)], 0)} y el mayor error."],
    tipo="clave")
tarjeta(s, 6.7, 5.1, 6.1, 1.85, "No linealidad e interacciones", [
    "El valor de Shapley del área es lo que el área suma o resta a la predicción de cada aviso frente al promedio; no es un "
    "efecto causal ni el precio implícito de Rosen. Se curva, y a igual área la prueba de Chapinero (naranja) recibe más que "
    "el entrenamiento (gris): una interacción con la ubicación. Un modelo lineal no "
    f"capta esas formas: {n(M[('ols', CH)])} frente a {n(M[('lightgbm', CH)])}."], tipo="definicion")
nota("Dos razones. Primera, la forma: el precio no crece en línea recta con el área, y en Chapinero la curva es "
     "más extrema; los lineales no pueden seguirla. Segunda, el error de aproximación y la varianza: un árbol solo es muy "
     "variable; el bosque promedia y baja la varianza; el boosting además corrige en secuencia lo que dejan los árboles "
     "anteriores. Ojo con dos palabras: promediar árboles baja la varianza; ajustar residuos, como hace el boosting, baja "
     "sobre todo el sesgo. El bosque también tiene menos sesgo que el árbol solo, probablemente porque sus árboles son más "
     "profundos (mínimo 5 avisos por hoja frente a 20). La partición interna la usan todas las familias que se "
     "afinan, así que no explica la diferencia. "
     "Ojo: el aporte del área no es el precio implícito del metro cuadrado de Rosen; es cómo el modelo reparte su predicción, y "
     "no es causal. El sesgo medio mide solo el nivel: los lineales lo tienen bajo porque extrapolan, pero fallan la forma, "
     "y por eso su error es el mayor. La tasa de aprendizaje del boosting es η = 0,05, con parada temprana.")

# ------------------------------------------------------------------------------------------------------------------
# 10. Todos subestiman Chapinero
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Todas las familias subestiman Chapinero", etiqueta="Dónde fallan", seccion="Dónde fallan")
texto(s, 0.55, 1.4, 7.6, 0.4, "Chapinero fuera: sesgo medio, predicción menos precio real (millones)", size=13, color="muted")
ses = [S[(f, CH)] for f, _ in FAM]
ch = barras(s, 0.45, 1.75, 7.7, 4.6, [nom for _, nom in FAM], [("Sesgo", ses, "azul")], minimo=-160, maximo=0, gap=40, grid=False)
ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.HIGH
etiquetas_puntos(ch, [[f"{n(v, 0)} (MAE {n(M[(f, CH)])})" for v, (f, _) in zip(ses, FAM)]])
texto(s, 0.55, 6.45, 7.6, 0.5, "Barras hacia la izquierda: el modelo predice por debajo del precio real.", size=12, color="muted")
tarjeta(s, 8.4, 1.5, 4.4, 1.7, "Lineales", [
    f"Menor sesgo medio ({n(S[('ols', CH)], 0)}) pero mayor error ({n(M[('ols', CH)], 1)}): extrapolan en línea recta y "
    "aciertan mejor el nivel, pero no la forma, porque el precio no crece en línea recta con las variables."], tipo="neutra")
tarjeta(s, 8.4, 3.35, 4.4, 1.75, "Árboles y red", [
    f"Predicen como en el borde del entrenamiento y subestiman más; la red con atención es la más sesgada "
    f"({n(S[('nn_atencion', CH)], 0)})."], tipo="cuidado")
tarjeta(s, 8.4, 5.25, 4.4, 1.7, "Para la empresa", [
    "Predecir de menos cuesta un negocio; predecir de más hace pagar caro. Aun con sesgo medio negativo, LightGBM sobrevalora "
    f"en más de 10 % a uno de cada cuatro avisos de Chapinero ({n(SOBRE_SIN, 0)} %; {n(SOBRE_CON, 0)} % con el ×1,08)."],
    tipo="recordar")
nota("Ninguna familia acierta el nivel de Chapinero: todas predicen por debajo, en promedio. Los lineales tienen el menor sesgo medio "
     "porque extrapolan en línea recta, pero su error total es el mayor. Árboles y red predicen como en el borde de lo que "
     "conocen; buena parte del entorno de Chapinero queda fuera del rango del entrenamiento. Para la empresa, subestimar es el "
     "error barato, pero eso es en promedio: uno de cada cuatro avisos queda más de 10 % por encima del precio real.")

# ------------------------------------------------------------------------------------------------------------------
# 11. Las variables pesan dentro del boosting
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Entre familias pesó el algoritmo; dentro del boosting, los datos", etiqueta="Qué movió el resultado",
               seccion="Qué movió el resultado")
texto(s, 0.55, 1.4, 5.9, 0.4, "Chapinero fuera, sin y con las 11 variables de Catastro (millones)", size=13, color="muted")
ch = barras(s, 0.45, 1.75, 6.0, 3.3, ["LightGBM", "CatBoost"],
            [("Sin Catastro", [cat[("lightgbm", "sin")], cat[("catboost", "sin")]], "gris_claro"),
             ("Con Catastro", [cat[("lightgbm", "con")], cat[("catboost", "con")]], "azul")],
            horizontal=False, minimo=0, maximo=200, leyenda=True, gap=60, overlap=-5, grid=False)
etiquetas_puntos(ch, [[n(cat[("lightgbm", "sin")]), n(cat[("catboost", "sin")])], [n(cat[("lightgbm", "con")]), n(cat[("catboost", "con")])]])
texto(s, 6.85, 1.4, 6.0, 0.4, "Kaggle, tabla pública (millones)", size=13, color="muted")
nuevos = env[env["orden"] >= 25]
ch = lineas_xy(s, 6.75, 1.75, 6.05, 2.95, [
    dict(nombre="Rejilla previa: 16 configuraciones de boosting, con las variables iniciales", x=list(rej["publico"]),
         y=[1.9 + 0.25 * (k % 2) for k in range(len(rej))], color="gris", linea=False, marcador=True, tam=9),
    dict(nombre="Envíos con las 162 variables, Catastro incluido", x=list(nuevos["publico"]), y=[1] * len(nuevos), color="azul",
         linea=False, marcador=True, tam=11)],
    leyenda=True, minx=145, maxx=195, miny=0.3, maxy=2.7, fmt_x="0", fmt_y="0",
    titulo_x="Error absoluto medio en la tabla pública (millones)",
    unidad_x=10, unidad_y=1, pos_leyenda="arriba")
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
texto(s, 6.85, 4.73, 6.0, 0.55, "Cada punto es un envío a Kaggle; la altura no significa nada. Los grises usan LightGBM, "
      "XGBoost (otra versión de boosting) o CatBoost.", size=12, color="muted")
tarjeta(s, 0.55, 5.3, 6.0, 1.65, "Comparación controlada", [
    "Mismos pliegues y variables, en Chapinero fuera: las 11 variables de Catastro bajaron el error "
    f"{n(cat[('lightgbm', 'sin')] - cat[('lightgbm', 'con')])} millones con LightGBM y "
    f"{n(cat[('catboost', 'sin')] - cat[('catboost', 'con')])} con CatBoost; cambiar LightGBM por CatBoost lo sube "
    f"{n(cat[('catboost', 'con')] - cat[('lightgbm', 'con')])}. Pesan más los datos que la implementación."], tipo="clave")
tarjeta(s, 6.85, 5.3, 5.95, 1.65, "Entre familias y en Kaggle", [
    f"Del lineal al boosting, el error de Chapinero fuera bajó {n(M[('ols', CH)] - M[('lightgbm', CH)], 0)} millones. En "
    f"Kaggle, las 16 configuraciones de una rejilla previa de boosting variaron "
    f"{n(rej['publico'].max() - rej['publico'].min(), 0)} millones entre sí, y los envíos con las 162 variables quedaron "
    f"{n(rej['publico'].min() - U01, 0)} por debajo de la mejor: varios cambios a la vez, no un experimento."], tipo="neutra")
nota("Entre familias pesó el algoritmo: pasar de un lineal al boosting baja el error en 42 millones. Dentro del boosting, "
     "pesaron más los datos: agregar 11 variables de Catastro bajó el error en Chapinero en 9 millones con LightGBM y en 11 "
     "con CatBoost, más que los 6 que separan a LightGBM de CatBoost. En Kaggle, las 16 configuraciones de la rejilla previa "
     "de boosting quedaron en una franja de 12 millones, todas lejos de los envíos con las 162 variables; son varios cambios a la vez, no un "
     "experimento controlado. XGBoost es otra implementación de boosting.")

# ------------------------------------------------------------------------------------------------------------------
# 12. Más complejo no siempre es mejor
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Añadir complejidad al LightGBM no ayudó; describir mejor Chapinero, sí", etiqueta="Qué movió el resultado",
               seccion="Qué movió el resultado")
filas = [("Idea", "Validación con que se probó", "Chapinero fuera", "Kaggle, tabla privada"),
         ("Variables de Catastro", ("clústeres: −2,4", "azul"), ("−9,0", "azul"), "en todos los envíos"),
         ("Ajuste ×1,08", ("kNNDM: +0,5; clústeres: −0,9", "tinta"), ("−2,9 (elegido con esta validación)", "azul"), (f"{n(P02 - P01, 1)}", "azul")),
         ("Mezcla 40 % LightGBM y 60 % CatBoost", ("clústeres: −1,4 (pesos elegidos aquí)", "azul"),
          (f"{n(bo['mezcla_40_60']['dif'], 1, True)}", "rojo"),
          (f"{n(priv['envio_03_mezcla_lightgbm_catboost'] - P01, 1, True)}", "rojo")),
         ("BETO ajustado como variable", ("zona cercana: −1,1", "azul"), ("+2,6", "rojo"), "no se envió"),
         ("Kriging de los residuos", ("kNNDM: +3,0", "rojo"), ("no se corrió; los 307 avisos de Chapinero en kNNDM: +2,4", "rojo"), "no se envió"),
         ("Red con atención en vez de LightGBM", ("clústeres: +3,8", "rojo"), ("+32,9", "rojo"), "no se envió")]
tabla(s, 0.55, 1.45, 12.25, filas, [3.7, 3.4, 3.2, 1.95], size=14, alto_fila=0.52)
texto(s, 0.55, 5.2, 12.25, 0.75, "Cambio en el MAE al agregar cada idea al LightGBM, o al cambiarlo por la red (millones). "
      "Azul: mejora; rojo: empeora; negro: depende de la validación o sin cifra. kNNDM: pliegues con la misma distancia al entrenamiento que la "
      "prueba. Zona cercana: avisos de entrenamiento a 1 km o menos de la prueba. BETO: modelo de lenguaje BERT en español, "
      "probado antes de agregar Catastro. Kriging: interpolación espacial de los residuos.", size=12, color="muted")
tarjeta(s, 0.55, 6.0, 12.25, 0.95, None, [
    "La columna clave es Chapinero fuera: solo Catastro y el ajuste la mejoran. La mezcla y BETO mejoraron la validación con "
    "que se probaron y empeoraron Chapinero; el kriging empeoró kNNDM. Lo anticipa la clase: que una red pueda representar "
    "cualquier función no garantiza que generalice (Lecture 12); entre modelos parecidos conviene el más simple (ISLR, "
    "10.6); y, como plantean Ash y Hansen (2023), lo complejo puede rendir menos cuando la prueba viene de otra "
    "distribución."], tipo="neutra",
    maximo=15)
nota("Cada cifra es el cambio en el MAE al agregar la idea; negativo, en azul, es mejor. Miren la columna de Chapinero fuera: "
     "solo los datos de Catastro y el ajuste la mejoran. Probamos ideas más complejas: mezclar con CatBoost, sumar BETO ajustado como variable, krigear los residuos y la red con "
     "atención. Varias mejoraban la validación general pero empeoraban Chapinero, que es lo que pide el negocio. Lo que sí "
     "funcionó fue agregar datos que describen ese mercado. ISLR muestra en Hitters que una red no le gana a la regresión con "
     "lasso y recomienda la navaja de Occam: entre modelos parecidos, el más simple.")

# ------------------------------------------------------------------------------------------------------------------
# 13. Compromisos
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("El boosting gana en error; los lineales, en sesgo medio", etiqueta="Compromisos", seccion="Compromisos")
filas = [("Familia", "MAE en Chapinero fuera", "Sesgo medio", "Qué se gana y qué se pierde", "Fuera del rango"),
         ("LightGBM", n(M[("lightgbm", CH)]), n(S[("lightgbm", CH)], 0), "el menor error; se explica con valores de Shapley (SHAP)",
          "predice como en el borde"),
         ("CatBoost", n(M[("catboost", CH)]), n(S[("catboost", CH)], 0), "el mejor en clústeres (124,1); 6 millones peor en Chapinero",
          "predice como en el borde"),
         ("Random Forest", n(M[("random_forest", CH)]), n(S[("random_forest", CH)], 0),
          "robusto y sin afinar; reduce varianza, pero no corrige el error como el boosting", "predice como en el borde"),
         ("CART", n(M[("cart", CH)]), n(S[("cart", CH)], 0), "reglas que se leen a simple vista; mucha varianza", "una hoja por zona"),
         ("Red con atención", n(M[("nn_atencion", CH)]), n(S[("nn_atencion", CH)], 0), "usa los precios vecinos; falla en zonas nuevas",
          "depende de los vecinos"),
         ("Elastic Net", n(M[("elastic_net", CH)]), n(S[("elastic_net", CH)], 0), "coeficientes y selección de variables; forma lineal",
          "extrapola en línea recta"),
         ("Regresión lineal", n(M[("ols", CH)]), n(S[("ols", CH)], 0), "coeficientes directos; forma lineal", "extrapola en línea recta")]
tabla(s, 0.55, 1.45, 12.25, filas, [2.2, 1.7, 1.0, 4.85, 2.5], size=13, alto_fila=0.55)
tarjeta(s, 0.55, 6.0, 12.25, 0.95, "Lectura para el negocio", [
    "El menor sesgo de los lineales no compensa: su error es 42 millones mayor. Y aun con sesgo negativo, el ganador sobrevalora "
    "en más de 10 % a uno de cada cuatro avisos: se usa con margen y tasación humana."], tipo="neutra", maximo=15)
nota("Cada familia tiene un costo. El boosting da el menor error pero, como todos los árboles, no extrapola. Los lineales "
     "extrapolan y tienen menos sesgo, pero un error mucho mayor. La red depende de tener vecinos parecidos.")

# ------------------------------------------------------------------------------------------------------------------
# 14. Conclusión
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Gana el boosting, pero ninguna familia corrige el nivel de Chapinero", oscura=True, etiqueta="Conclusión",
               seccion="Conclusión")
bloques = [("Por qué ganó el boosting", "tinte", "navy",
            "Capta no linealidades e interacciones de ubicación y tamaño, y corrige en secuencia el error que deja un árbol "
            "solo. Ganó en las tres validaciones (LightGBM en dos, CatBoost en clústeres) y en Kaggle, aunque allí cada "
            "familia usó variables distintas."),
           ("Qué movió el resultado", "tinte", "navy",
            "Elegir el boosting (15 a 42 millones menos que las otras familias en Chapinero fuera) y, dentro del boosting, "
            "las variables que describen Chapinero (Catastro, −9 a −11) más que la implementación (6)."),
           ("Lo que ninguna familia resuelve", "rojo_claro", "rojo",
            "Todas subestiman Chapinero: sus entornos (valor del suelo, embajadas) quedan fuera del rango del entrenamiento. "
            "Es sobre todo un límite de los datos."),
           ("Para la empresa", "tinte", "navy",
            "Usar LightGBM para ordenar y filtrar avisos y como punto de partida de la negociación, con margen y tasación "
            "humana; no para ofertar de forma automática.")]
for i, (tit, fondo, borde, t) in enumerate(bloques):
    x0, y0 = 0.55 + (i % 2) * 6.2, 1.5 + (i // 2) * 2.7
    forma(s, MSO_SHAPE.ROUNDED_RECTANGLE, x0, y0, 6.0, 2.5, relleno=fondo, radio=0.05)
    texto(s, x0 + 0.25, y0 + 0.15, 5.5, 0.45, tit, size=19, color=borde)
    texto(s, x0 + 0.25, y0 + 0.7, 5.5, 1.7, t, size=15, color="tinta")
nota("En resumen: el boosting gana porque capta formas no lineales y corrige el error paso a paso. Entre familias pesó el "
     "algoritmo; dentro del boosting, los datos. Ninguna familia resuelve que Chapinero esté fuera del rango del "
     "entrenamiento. Y para la empresa, el modelo sirve para filtrar y negociar, no para comprar solo.")

# ------------------------------------------------------------------------------------------------------------------
# Apéndice
# ------------------------------------------------------------------------------------------------------------------
s, t_ = error_por_familia(deck, "Apéndice", "Apéndice")
nota(t_)

s = deck.nueva("La red con atención", etiqueta="Apéndice", seccion="Apéndice")
cajas = [("162 variables del aviso", "tinte", "navy"), ("Codificador: 256 y 128 neuronas, normalización por lotes y dropout",
                                                       "azul_claro", "azul"),
         ("Atención sobre 20 vecinos de entrenamiento: pesa su precio, precio por m² y características según parecido y "
          "distancia", "ambar_claro", "ambar"), ("Cabeza de 128 neuronas: precio predicho", "tinte", "navy")]
for i, (t, fondo, borde) in enumerate(cajas):
    x0 = 0.55 + i * 3.1
    nodo(s, x0, 1.8, 2.8, 1.45, [t], relleno=fondo, borde=borde, sizes=(14,), colores=("tinta",))
    if i < 3:
        linea(s, x0 + 2.82, 2.52, x0 + 3.08, 2.52, color="gris", ancho=1.5)
imagen(s, IMG / "f_atencion.png", 0.65, 3.5, h=0.62, alt="Fórmula de la atención")
texto(s, 6.4, 3.45, 6.4, 0.75, "Q resume el aviso, K describe a cada vecino y V lleva su precio; los pesos suman uno, así "
      "que la red usa un promedio ponderado de la vecindad (Vaswani y otros, 2017).", size=12, color="muted")
tarjeta(s, 0.55, 4.4, 6.0, 2.55, "Entrenamiento", [
    "Pérdida absoluta (L1) y AdamW. Regularización de la Lecture 12: decaimiento de pesos (sin penalizar sesgos ni la "
    "normalización), dropout, normalización por lotes y parada temprana con la partición interna.",
    "Los vecinos excluyen la misma coordenada y los avisos duplicados por texto."], tipo="definicion")
tarjeta(s, 6.8, 4.4, 6.0, 2.55, "Fuentes de la idea", [
    "Atención de Vaswani y otros (2017), adaptada a precios de vivienda como en Viana y Barbosa (2021): la predicción combina "
    "el aviso con un resumen ponderado de su vecindad."], tipo="neutra")
nota("Detalle de la red con atención para preguntas.")

s, t_ = texto_apendice(deck, "Apéndice", "Apéndice")
nota(t_)
s, t_ = dependencia_espacial(deck, "Apéndice", "Apéndice")
nota(t_)

s = deck.nueva("Fuentes", etiqueta="Apéndice", seccion="Apéndice")
refs = ["Datos: Properati (avisos de la competencia de Kaggle del curso); Catastro de Bogotá; OpenStreetMap; Google Earth Engine "
        "(SRTM, Sentinel-2, VIIRS y GHSL).",
        "Ash, E. y Hansen, S. (2023). Text algorithms in economics. Annual Review of Economics, 15, 659-688.",
        "Blei, D. M., Ng, A. Y. y Jordan, M. I. (2003). Latent Dirichlet allocation. Journal of Machine Learning Research, 3, "
        "993-1022.",
        "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.",
        "Cañete, J. y otros (2020). Spanish pre-trained BERT model and evaluation data. PML4DC, ICLR.",
        "Gentzkow, M., Kelly, B. y Taddy, M. (2019). Text as data. Journal of Economic Literature, 57(3), 535-574.",
        "Hengl, T., Heuvelink, G. y Rossiter, D. (2007). About regression-kriging. Computers & Geosciences, 33(10), 1301-1315.",
        "James, G., Witten, D., Hastie, T. y Tibshirani, R. (2021). An Introduction to Statistical Learning, segunda edición. "
        "Springer, caps. 6, 8 y 10.",
        "Ke, G. y otros (2017). LightGBM: a highly efficient gradient boosting decision tree. NeurIPS 30.",
        "Linnenbrink, J., Milà, C., Ludwig, M. y Meyer, H. (2024). kNNDM CV. Geoscientific Model Development, 17, 5897-5912.",
        "Lundberg, S. M. y Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS 30.",
        "Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. Biometrika, 37(1/2), 17-23.",
        "Prokhorenkova, L. y otros (2018). CatBoost: unbiased boosting with categorical features. NeurIPS 31.",
        "Roberts, D. R. y otros (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or "
        "phylogenetic structure. Ecography, 40(8), 913-929.",
        "Rosen, S. (1974). Hedonic prices and implicit markets. Journal of Political Economy, 82(1), 34-55.",
        "Tobler, W. R. (1970). A computer movie simulating urban growth in the Detroit region. Economic Geography, 46, "
        "234-240.",
        "Vaswani, A. y otros (2017). Attention is all you need. NeurIPS 30.",
        "Viana, D. y Barbosa, L. (2021). Attention-based spatial interpolation for house price prediction. SIGSPATIAL.",
        "Zou, H. y Hastie, T. (2005). Regularization and variable selection via the elastic net. JRSS B, 67(2), 301-320.",
        "Material del curso: Lectures 12 a 15; Validación cruzada espacial (Rojas y Sarmiento-Barbieri, 2026); Texto como datos "
        "(material de clase); enunciado del Problem Set 3."]
texto(s, 0.55, 1.45, 12.25, 5.5, refs, size=12, color="tinta", espacio=4)
nota("Referencias citadas.")

for i, sl in enumerate(deck.prs.slides):
    if i in NOTAS:
        deck.notas(sl, NOTAS[i])
deck.guardar(OUT / "compare_equipo_02.pptx")
