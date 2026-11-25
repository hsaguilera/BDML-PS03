"""Láminas propias de compare (Algorithm Comparison): portada, resumen del mejor modelo y cuerpo de la comparación.

Uso:
    from laminas_compare_nuevo import compare_portada_y_resumen, compare_cuerpo
    inicio = compare_portada_y_resumen(deck)   # [(lámina, notas)]: portada y lámina 2
    cuerpo = compare_cuerpo(deck)              # [(lámina, notas)]: de la 26 al final (sin apéndice)

Las láminas 3 a 25 (Etapa 1) no se tocan: van entre las dos llamadas.

Orden del cuerpo (23 láminas):
  Cómo se compara: criterios (error fuera de muestra, sesgo y varianza, flexibilidad).
  Seis familias: cómo predice cada una; lineales; CART y Random Forest; boosting; red con atención; hiperparámetros.
  Reglas y validación: las mismas reglas; tres validaciones.
  Resultado: error por validación; flexibilidad y optimismo; ¿son reales las diferencias?; estándar frente a espacial;
  qué validación refleja Kaggle.
  Por qué gana el boosting: forma no lineal; promediar frente a corregir.
  Dónde fallan: subestiman Chapinero; riesgo de pagar de más.
  Qué movió el resultado: algoritmo frente a datos; complejidad que no ayudó.
  Compromisos; conclusión; fuentes.

Cifras: laminas/cv_familias.csv, laminas/compare_nuevo.json (31_datos_laminas_compare.py, solo lectura), laminas/ppt_bootstrap2.json,
laminas/envios_equipo.csv, laminas/clasificacion.csv, laminas/ppt_lados.csv, laminas/etapas2a4.json, laminas/distancias_vecino.parquet,
00_data/processed/modelos/cv_clase_res_*.csv, 00_data/processed/modelos/cv_clase_oof_{lightgbm,catboost}_{sin,con}.parquet y 00_data/processed/modelos/rk_knndm_res_lightgbm_con.csv.
Hiperparámetros: functions/modelo_utils.py (ajustar_predecir y red con atención) y functions/validacion_clase.py. Solo lee; no
cambia datos ni modelos. No modifica laminas_comunes.py ni ppt_lib.py: importa de ellos.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE, XL_TICK_LABEL_POSITION
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

from functions.laminas_comunes import D, IMG, STORES, n
from functions.ppt_lib import (_ejes, _estilo_grafico, _lineas, barras, etiquetas_puntos, forma, imagen, linea, nodo, rgb, tabla,
                     tarjeta, texto)

IMG3 = IMG / "compare_nuevo"
FIN = {}  # y final del contenido principal de cada lámina (control de calidad)

# ------------------------------------------------------------------------------------------------------------------
# Datos
# ------------------------------------------------------------------------------------------------------------------
CVF = pd.read_csv(D / "cv_familias.csv")
M = {(r.familia, r.esquema): r.mae for r in CVF.itertuples()}
S = {(r.familia, r.esquema): r.sesgo for r in CVF.itertuples()}
AL, CL, CH = "CV aleatoria", "Clústeres k-means", "Chapinero fuera"
FAM = [("lightgbm", "LightGBM"), ("catboost", "CatBoost"), ("random_forest", "Random Forest"), ("cart", "CART"),
       ("nn_atencion", "Red con atención"), ("elastic_net", "Elastic Net"), ("ols", "Regresión lineal")]
NOM = dict(FAM)
R = json.load(open(D / "compare_nuevo.json", encoding="utf-8"))
BO = json.load(open(D / "ppt_bootstrap2.json", encoding="utf-8"))
E24 = json.load(open(D / "etapas2a4.json", encoding="utf-8"))
ENV = pd.read_csv(D / "envios_equipo.csv")
PRIV = dict(zip(ENV["archivo"], ENV["privado"]))
PUBL = dict(zip(ENV["archivo"], ENV["publico"]))
P01, P02 = PRIV["envio_01_lightgbm_con_catastro"], PRIV["envio_02_lightgbm_con_catastro_x108"]
U01, U02 = PUBL["envio_01_lightgbm_con_catastro"], PUBL["envio_02_lightgbm_con_catastro_x108"]
PREVIO = ENV[ENV["orden"] < 25]["privado"].min()
CLAS = pd.read_csv(D / "clasificacion.csv").sort_values("privado").reset_index(drop=True)
PUESTO = int(CLAS.index[CLAS["equipo"] == "Grupo_02"][0]) + 1
ORDINAL = {1: "primero", 2: "segundo", 3: "tercero", 4: "cuarto", 5: "quinto", 6: "sexto"}
CARD = {5: "cinco", 6: "seis", 7: "siete"}
LADOS = pd.read_csv(D / "ppt_lados.csv")
SOBRE_SIN = LADOS[(LADOS["esquema"] == CH) & (LADOS["version"] == "Sin ajuste")]["sobre_10"].iloc[0]
DIST = pd.read_parquet(D / "distancias_vecino.parquet")
PCT0 = {e: 100 * float((g_["distancia_m"] == 0).mean()) for e, g_ in DIST.groupby("esquema")}
MED = {e: float(g_["distancia_m"].median()) for e, g_ in DIST.groupby("esquema")}
PL = E24["pliegues"]
RES = {(f, c): pd.read_csv(STORES / f"cv_clase_res_{f}_{c}.csv").set_index("esquema")
       for f in ("lightgbm", "catboost") for c in ("sin", "con")}
CHS = "Chapinero (imitando el taller)"
CAT = {(f, c): RES[(f, c)].loc[CHS, "MAE"] for f in ("lightgbm", "catboost") for c in ("sin", "con")}
LGB = RES[("lightgbm", "con")]["MAE"]
RK = pd.read_csv(STORES / "rk_knndm_res_lightgbm_con.csv").set_index("variante")
REJ = ENV[(ENV["autor"] == "César Vallejo") & (ENV["familia"] == "Boosting")]
assert len(REJ) == 16
MEJOR_FAM = ENV[ENV["orden"] < 28].sort_values("privado").groupby("familia").head(1).sort_values("privado")
BL = R["bloques_chapinero"]["1000"]
BLC = R["bloques_clusters"]["1000"]
RIESGO = R["riesgo"]

ETQ = {"comp": "Cómo se compara", "fam": "Seis familias", "reg": "Reglas y validación", "res": "Resultado",
       "kag": "Validación estándar frente a espacial", "por": "Por qué gana el boosting", "fal": "Dónde fallan",
       "mov": "Qué movió el resultado", "tra": "Compromisos", "con": "Conclusión", "fue": "Fuentes"}


# ------------------------------------------------------------------------------------------------------------------
# Utilidades de diseño
# ------------------------------------------------------------------------------------------------------------------
def _nueva(deck, titulo, clave, oscura=False):
    """Lámina nueva con el título en una sola línea: 26 puntos si cabe; si no, 24 o 22."""
    tam = next((t for t in (26, 24, 22) if _lineas(titulo, 12.1, t) == 1), 22)
    return deck.nueva(titulo, etiqueta=ETQ[clave], seccion=ETQ[clave], size_titulo=tam, oscura=oscura)


def _alto_celda(t, ancho, size):
    return _lineas(str(t), max(ancho - 0.17, 0.3) * 0.97, size) * 1.21 * size / 72 + 0.08


def _tabla(s, x, y, filas, anchos, size=13, minimo=0.32, colores_col=None):
    """Tabla con altos de fila medidos con Calibri para que nada se encime. Devuelve la y final."""
    gf = tabla(s, x, y, sum(anchos), filas, anchos, size=size, alto_fila=minimo, colores_col=colores_col)
    altos = [max(minimo, max(_alto_celda(c[0] if isinstance(c, tuple) else c, a, size) for c, a in zip(f, anchos)))
             for f in filas]
    for i, h in enumerate(altos):
        gf.table.rows[i].height = Inches(h)
    gf.height = Inches(sum(altos))
    return y + sum(altos)


def _formula(s, nombre, x, y, escala=0.75, alt=""):
    """Imagen de fórmula a `escala` de su tamaño (26 puntos a 300 dpi): 0,68 o más deja los subíndices en 12 puntos o más."""
    from PIL import Image
    w_px, h_px = Image.open(IMG3 / f"{nombre}.png").size
    w, h = w_px / 300 * escala, h_px / 300 * escala
    imagen(s, IMG3 / f"{nombre}.png", x, y, w=w, alt=alt or nombre)
    return w, h


def _lineas_cat(s, x, y, w, h, categorias, series, minimo, maximo, unidad, size=12, leyenda=True):
    """Gráfico de líneas con categorías; series: lista de (nombre, valores, color, ancho, guiones)."""
    cd = CategoryChartData()
    cd.categories = categorias
    for nom, vals, *_ in series:
        cd.add_series(nom, vals)
    gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    _estilo_grafico(ch, size)
    ch.has_title = False
    ch.has_legend = leyenda
    if leyenda:
        ch.legend.position = XL_LEGEND_POSITION.TOP
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(size)
    for ser, (_, _, color, ancho, guiones) in zip(ch.plots[0].series, series):
        ser.smooth = False
        ser.format.line.color.rgb = rgb(color)
        ser.format.line.width = Pt(ancho)
        if guiones:
            ser.format.line.dash_style = MSO_LINE.DASH
        ser.marker.style = XL_MARKER_STYLE.CIRCLE
        ser.marker.size = 8
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.rgb = rgb(color)
        ser.marker.format.line.fill.background()
    _ejes(ch, valor_visible=True, fmt="0", minimo=minimo, maximo=maximo, unidad=unidad, grid=True, size=size)
    ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    return ch


def _mini_arbol(s, cx, y, w, h, color="azul", hojas=True):
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


# ------------------------------------------------------------------------------------------------------------------
# Portada y lámina 2
# ------------------------------------------------------------------------------------------------------------------
def _portada(deck):
    s = deck.nueva(oscura=True)
    texto(s, 0.7, 0.9, 6.3, 0.4, "Problem Set 3. Making Money with ML?", size=16, color="cielo")
    texto(s, 0.7, 1.4, 6.4, 1.7, "Qué modelo predice mejor los precios de vivienda en Chapinero", size=30, color="blanco")
    texto(s, 0.7, 3.25, 6.3, 1.5, "Algorithm Comparison: seis familias de algoritmos con las mismas variables, los mismos "
          "pliegues y la misma medida de error", size=19, color="cielo")
    linea(s, 0.7, 4.95, 3.2, 4.95, color="naranja", ancho=2.5)
    texto(s, 0.7, 5.15, 6.3, 1.0, ["Equipo 02: Jhon Sebastián Vela Salcedo, Harold Santiago Aguilera Mulfo y César Augusto "
                                   "Vallejo Quiroga", "25 de noviembre de 2026"], size=15, color="blanco", espacio=8)
    imagen(s, IMG / "portada_mapa.png", 7.45, 0.45, h=5.95, alt="Mapa de Bogotá con los avisos de entrenamiento y de prueba")
    texto(s, 7.45, 6.45, 5.4, 0.5, "Gris: avisos de entrenamiento. Naranja: avisos de prueba (94 % en la localidad de "
          "Chapinero).", size=12, color="cielo")
    notas = ("Comparamos seis familias de algoritmos para predecir el precio de la vivienda en Chapinero: regresión lineal, "
             "Elastic Net, un árbol de decisión, Random Forest, boosting y redes neuronales. Todas reciben las mismas "
             "variables, se prueban con los mismos pliegues y se miden con el mismo error. La pregunta es cuál predice mejor, "
             "por qué, y qué se sacrifica con cada alternativa. En el mapa, en gris, los avisos con los que se aprende; en "
             "naranja, los que hay que predecir, casi todos en Chapinero.")
    return s, notas


def _mejor_modelo(deck):
    s = deck.nueva("El mejor modelo: boosting de árboles (LightGBM) con texto, entorno y Catastro", etiqueta="El mejor modelo",
                   seccion="El mejor modelo", size_titulo=next(t for t in (26, 24, 22) if _lineas(
                       "El mejor modelo: boosting de árboles (LightGBM) con texto, entorno y Catastro", 12.1, t) == 1))
    tarjeta(s, 0.55, 1.3, 6.1, 2.35, "Qué es", [
        "Árboles de decisión sumados en secuencia: cada uno corrige el error que dejaron los anteriores.",
        "Recibe 162 variables del aviso, su texto y su entorno (mapa, satélite y Catastro) y predice el precio en millones "
        "de pesos, minimizando el error cuadrático.",
        "Tasa de aprendizaje 0,05; hasta 63 hojas por árbol; 80 % de los avisos y 70 % de las variables en cada árbol; 252 "
        "árboles por parada temprana (semilla 2026); promedio de 5 corridas."],
        tipo="definicion")
    tarjeta(s, 0.55, 3.77, 6.1, 1.45, "Dos ajustes al final", [
        "La distancia a la media del entrenamiento (654) se multiplica por 1,08, porque se quedaba corto en los avisos "
        "caros. En 78 avisos de prueba cuya descripción repite la de uno del entrenamiento, pesa 0,7 el precio de ese aviso."],
        tipo="recordar")
    filas = [("Medida", "Error absoluto medio, MAE (millones)"),
             ("Kaggle, tabla privada (80 % de la prueba)", f"{n(P02)}: {ORDINAL[PUESTO]} de {CARD[len(CLAS)]} equipos"),
             ("Kaggle, tabla pública (20 %)", n(U02)),
             ("Chapinero fuera, sin los dos ajustes",
              f"{n(M[('lightgbm', CH)])} (intervalo de 95 %: {n(BO['lightgbm']['ic_bajo'], 0)} a "
              f"{n(BO['lightgbm']['ic_alto'], 0)})")]
    y1 = _tabla(s, 0.55, 5.37, filas, [2.75, 3.35], size=12, minimo=0.3)
    texto(s, 6.95, 1.3, 5.85, 0.62, "Chapinero fuera: MAE de las siete especificaciones, con las mismas 162 variables y los "
          "mismos pliegues (millones)", size=13, color="muted")
    vals = [M[(f, CH)] for f, _ in FAM]
    cols = ["azul", "azul_suave"] + ["gris_claro"] * 5
    ch = barras(s, 6.85, 1.9, 5.95, 3.45, [nom for _, nom in FAM], [("MAE", vals, cols)], minimo=0, maximo=215, gap=35,
                grid=False)
    etiquetas_puntos(ch, [[n(v) for v in vals]])
    tarjeta(s, 6.95, 5.45, 5.85, 1.5, "La pregunta de esta presentación", [
        "Por qué el boosting le gana a las otras cinco familias con las mismas reglas, si esa ventaja es real y qué se "
        "sacrifica con cada alternativa."], tipo="neutra")
    FIN[id(s)] = y1
    notas = ("Empezamos por el resultado. El mejor modelo es boosting de árboles, con la librería LightGBM: suma árboles de "
             "decisión en secuencia, y cada uno corrige el error que dejaron los anteriores. Recibe 162 variables, entre "
             "ellas el texto del aviso, el entorno en el mapa y el satélite, y 11 variables del Catastro de Bogotá. Al final "
             "estiramos un 8 % sus predicciones alrededor de la media, porque se quedaba corto en los avisos caros, y para 78 "
             "avisos cuya descripción repite la de uno del entrenamiento usamos sobre todo el precio de ese gemelo. En la "
             f"tabla privada de Kaggle obtuvo un error absoluto medio de {n(P02)} millones, segundo de seis equipos, y bajó "
             f"{n(PREVIO - P02)} millones frente al mejor envío previo del equipo. La validación que imita la tarea, "
             f"entrenar sin Chapinero y predecirlo, le da {n(M[('lightgbm', CH)])} millones. A la derecha está la "
             "comparación que vamos a explicar: con las mismas variables y los mismos pliegues, el boosting, en azul, tiene "
             "el menor error; CatBoost es otra versión de boosting. La pregunta es por qué, si la ventaja es real y qué "
             "se pierde con cada alternativa.")
    return s, notas


def compare_portada_y_resumen(deck):
    """Portada y lámina 2 (resumen del mejor modelo). Devuelve [(lámina, notas), (lámina, notas)] y pone las notas."""
    out = [_portada(deck), _mejor_modelo(deck)]
    for s, notas in out:
        deck.notas(s, notas)
    return out


# ------------------------------------------------------------------------------------------------------------------
# Cómo se compara
# ------------------------------------------------------------------------------------------------------------------
def _criterios(deck):
    s = _nueva(deck, "Cómo comparamos: el error en avisos que el modelo no vio", "comp")
    tarjeta(s, 0.55, 1.3, 6.1, 1.85, "Error fuera de muestra (MAE)", [
        "Promedio, en millones, de cuánto se aleja la predicción ŷ del precio real y en avisos que el modelo no usó para "
        "aprender, sin importar si es hacia arriba o hacia abajo. Es la medida de Kaggle."], tipo="definicion")
    _formula(s, "c_mae", 0.75, 3.25, escala=0.7, alt="Fórmula del error absoluto medio")
    tarjeta(s, 0.55, 3.95, 6.1, 1.5, "Sesgo, varianza y ruido", [
        "Sesgo: error sistemático de una forma demasiado rígida. Varianza: cuánto cambia la predicción si cambian los "
        "avisos de entrenamiento. σ²: ruido que ningún modelo puede predecir."], tipo="definicion")
    _formula(s, "c_descomp", 0.75, 5.55, escala=0.72, alt="Descomposición del error en sesgo, varianza y ruido")
    texto(s, 0.55, 6.0, 6.1, 0.95, "La descomposición es la del error cuadrático; el MAE responde a la misma tensión. Más "
          "flexibilidad baja el sesgo y sube la varianza (Hastie y otros, 2009; James y otros, 2021).", size=12, color="muted")
    tarjeta(s, 6.95, 1.3, 5.85, 1.9, "Flexibilidad, hiperparámetros y regularización", [
        "Los hiperparámetros fijan cuán flexible es un modelo (cuánto se encogen los coeficientes, la profundidad de un "
        "árbol, el número de árboles o de épocas). No se aprenden del ajuste: se eligen mirando el error en avisos "
        "apartados. Regularizar es restringir la flexibilidad a propósito."], tipo="definicion")
    filas = [("Criterio", "Qué se mira en cada familia"),
             ("Error", "MAE en tres validaciones; la principal, Chapinero fuera"),
             ("Diferencia real", "intervalo de 95 % de la diferencia con el mejor, por remuestreo"),
             ("Sesgo medio", "si predice de más o de menos en promedio (predicción menos precio)"),
             ("Riesgo de pagar de más", "porcentaje de avisos con predicción más de 10 % sobre el precio"),
             ("Interpretabilidad", "si se puede leer cómo llega a cada predicción"),
             ("Fuera del rango conocido", "qué hace con valores que no vio al entrenar")]
    y1 = _tabla(s, 6.95, 3.4, filas, [2.15, 3.7], size=13, minimo=0.32)
    FIN[id(s)] = max(y1, 6.95)
    notas = ("Antes de comparar, cómo comparamos. Lo que importa es el error en avisos que el modelo no vio: dentro de la "
             "muestra cualquier modelo flexible parece bueno. Usamos el error absoluto medio, MAE, en millones, que es la "
             "medida de Kaggle. Ese error tiene tres partes. El sesgo es el error de una forma demasiado rígida, como una "
             "recta donde el precio se curva. La varianza es cuánto cambia la predicción si cambian los avisos con que se "
             "entrena, el problema de un modelo que memoriza. Y el ruido nadie lo predice. Un modelo más flexible baja el "
             "sesgo pero sube la varianza; los hiperparámetros, como cuánto encoger los coeficientes, la profundidad de un "
             "árbol o cuántos árboles sumar, fijan ese punto, y se eligen mirando el error en avisos apartados. A la "
             "derecha, lo que vamos a mirar de cada familia: el error, si la diferencia es real, el sesgo, el riesgo de "
             "pagar de más, si se entiende cómo predice y qué hace fuera de lo que conoce.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# Seis familias
# ------------------------------------------------------------------------------------------------------------------
def _como_predice(deck):
    s = _nueva(deck, "Cómo predice cada familia y qué controla su flexibilidad", "fam")
    COLW, GAP = 2.32, 0.15
    desc = [("Regresión lineal y Elastic Net", "Una suma ponderada de las variables. Elastic Net además encoge los "
             "coeficientes hacia cero y puede anular algunos.", "cuánto se encoge (λ) y la mezcla de lasso y ridge (α)"),
            ("CART: un árbol", "Preguntas de sí o no que parten los avisos en hojas; predice el promedio de la hoja.",
             "profundidad y avisos mínimos por hoja"),
            ("Random Forest", "300 árboles, cada uno con avisos y variables sorteados; promedia sus predicciones.",
             "variables por corte y avisos por hoja"),
            ("Boosting: LightGBM y CatBoost", "Árboles pequeños en secuencia: cada uno corrige el error que dejaron los "
             "anteriores.", "número de árboles, tasa de aprendizaje y tamaño de cada árbol"),
            ("Red con atención", "Capas de neuronas que transforman las variables; la atención pondera el precio de 20 "
             "vecinos.", "capas y neuronas, épocas, dropout y decaimiento de pesos")]
    for i, (tit, txt, ctrl) in enumerate(desc):
        x0 = 0.55 + i * (COLW + GAP)
        forma(s, MSO_SHAPE.ROUNDED_RECTANGLE, x0, 1.3, COLW, 5.65, relleno="tinte", radio=0.04)
        texto(s, x0 + 0.12, 1.4, COLW - 0.24, 0.65, tit, size=15, color="navy")
        ya, ha = 2.15, 1.75
        cx = x0 + COLW / 2
        if i == 0:
            linea(s, x0 + 0.3, ya + ha, x0 + COLW - 0.2, ya + ha, color="gris")
            linea(s, x0 + 0.3, ya + 0.1, x0 + 0.3, ya + ha, color="gris")
            pts = [(0.1, 0.25), (0.22, 0.3), (0.3, 0.45), (0.42, 0.42), (0.55, 0.6), (0.62, 0.52), (0.75, 0.72), (0.85, 0.8)]
            for px, py in pts:
                forma(s, MSO_SHAPE.OVAL, x0 + 0.3 + px * (COLW - 0.6) - 0.05, ya + ha - py * (ha - 0.2) - 0.05, 0.1, 0.1,
                      relleno="gris")
            linea(s, x0 + 0.35, ya + ha - 0.2 * (ha - 0.2), x0 + COLW - 0.25, ya + ha - 0.88 * (ha - 0.2), color="azul",
                  ancho=2.5)
        elif i == 1:
            _mini_arbol(s, cx, ya + 0.1, COLW - 0.3, ha - 0.1)
        elif i == 2:
            for k in range(3):
                _mini_arbol(s, x0 + 0.42 + k * 0.74, ya + 0.05, 0.75, 1.1, hojas=False)
            for k in range(3):
                linea(s, x0 + 0.42 + k * 0.74, ya + 0.9, cx, ya + 1.3, color="gris", ancho=1)
            nodo(s, cx - 0.62, ya + 1.3, 1.24, 0.42, ["promedio"], relleno="azul_claro", borde="azul", sizes=(12,),
                 colores=("azul",))
        elif i == 3:
            for k in range(3):
                _mini_arbol(s, x0 + 0.42 + k * 0.74, ya + 0.05, 0.75, 1.1, color=["azul_suave", "azul", "navy"][k], hojas=False)
                if k < 2:
                    texto(s, x0 + 0.67 + k * 0.74, ya + 0.3, 0.3, 0.35, "+", size=16, color="navy", align="center")
            texto(s, x0 + 0.15, ya + 1.25, COLW - 0.3, 0.45, "cada árbol corrige al anterior", size=12, color="muted",
                  align="center")
        else:
            capas = [4, 3, 1]
            pos = {}
            for c_, m_ in enumerate(capas):
                for r_ in range(m_):
                    pos[(c_, r_)] = (x0 + 0.35 + c_ * 0.62, ya + 0.2 + (r_ + (4 - m_) / 2) * 0.34)
            for c_ in range(2):
                for r_ in range(capas[c_]):
                    for r2 in range(capas[c_ + 1]):
                        (a1, b1), (a2, b2) = pos[(c_, r_)], pos[(c_ + 1, r2)]
                        linea(s, a1 + 0.08, b1 + 0.08, a2 + 0.08, b2 + 0.08, color="gris_claro", ancho=0.75)
            for (c_, r_), (px, py) in pos.items():
                forma(s, MSO_SHAPE.OVAL, px, py, 0.16, 0.16, relleno="azul" if c_ < 2 else "navy")
            for k in range(4):
                forma(s, MSO_SHAPE.OVAL, x0 + COLW - 0.45, ya + 0.15 + k * 0.34, 0.2, 0.2, relleno="gris")
                linea(s, x0 + COLW - 0.45, ya + 0.25 + k * 0.34, pos[(1, 1)][0] + 0.16, pos[(1, 1)][1] + 0.08, color="gris",
                      ancho=0.75)
            texto(s, x0 + COLW - 0.95, ya + 1.5, 0.9, 0.3, "vecinos", size=12, color="muted", align="center")
        texto(s, x0 + 0.12, 4.05, COLW - 0.24, 1.45, txt, size=14, color="tinta")
        texto(s, x0 + 0.12, 5.5, COLW - 0.24, 0.3, "Su flexibilidad depende de", size=12, color="muted")
        texto(s, x0 + 0.12, 5.8, COLW - 0.24, 1.1, ctrl, size=13, color="azul")
    FIN[id(s)] = 6.95
    notas = ("Primero, qué hace cada familia. La regresión lineal suma las variables con pesos fijos; Elastic Net hace lo "
             "mismo pero encoge esos pesos hacia cero, y puede anular algunos. CART es un solo árbol de decisión: hace "
             "preguntas de sí o no, por ejemplo si el área pasa de cierto valor, y en cada hoja predice el promedio de sus "
             "avisos. Random Forest entrena 300 árboles distintos, cada uno con avisos y variables sorteados, y promedia. El "
             "boosting suma árboles pequeños en secuencia: cada uno corrige lo que los anteriores dejaron mal; lo corrimos "
             "con dos librerías, LightGBM y CatBoost. La red con atención aprende capas de neuronas y además mira el precio "
             "de los 20 vecinos más cercanos. Abajo, en azul, lo que controla qué tan flexible es cada una: son los "
             "hiperparámetros que vamos a ver.")
    return s, notas


def _lineales(deck):
    s = _nueva(deck, "Regresión lineal y Elastic Net: una suma ponderada, con o sin encoger", "fam")
    texto(s, 0.55, 1.3, 6.2, 0.4, "Regresión lineal: el menor error cuadrático dentro de la muestra", size=14,
          color="navy")
    _formula(s, "c_ols", 0.75, 1.75, escala=0.72, alt="Mínimos cuadrados")
    texto(s, 0.55, 2.4, 6.2, 0.4, "Elastic Net: además castiga el tamaño de los coeficientes",
          size=14, color="navy")
    w_, h_ = _formula(s, "c_en", 0.6, 2.85, escala=0.68, alt="Elastic Net")
    texto(s, 0.55, 2.85 + h_ + 0.05, 6.2, 1.0, ["λ: cuánto se encoge; con λ = 0 es la regresión lineal.",
                                               "α: la mezcla. Con α = 1 es lasso, que anula coeficientes; con α = 0 es "
                                               "ridge, que los encoge sin anularlos."], size=13, color="tinta", espacio=3)
    tarjeta(s, 0.55, 4.95, 6.2, 2.0, "Cómo se eligen λ y α", [
        "Rejilla de 4 valores de α (0,1; 0,5; 0,9 y 1) por 30 de λ. Dentro del entrenamiento de cada pliegue se prueban con "
        "3 particiones que apartan el 20 % de las celdas de 2 km, y se queda la pareja de menor error: validación cruzada en "
        "dos dimensiones (Zou y Hastie, 2005)."], tipo="definicion")
    tarjeta(s, 7.0, 1.3, 5.8, 1.85, "Cómo se preparan los datos", [
        "Los faltantes se llenan con la mediana del entrenamiento y se agrega una marca de faltante. Después cada variable "
        "se estandariza (media 0, desviación 1): con penalización, la escala cambiaría cuánto se encoge cada coeficiente."],
        tipo="neutra")
    tarjeta(s, 7.0, 3.3, 5.8, 1.85, "Qué se gana", [
        "Cada coeficiente se lee directamente, y fuera del rango conocido la predicción sigue la recta. Con 38.644 avisos y "
        "162 variables hay mucha información por coeficiente: la varianza ya es baja y encoger casi no cambia el resultado."],
        tipo="clave")
    tarjeta(s, 7.0, 5.3, 5.8, 1.65, "Lo que no puede", [
        "Solo suma efectos: el área pesa lo mismo en un apartamento pequeño que en uno grande, y en cualquier barrio. Curvas e "
        "interacciones habría que construirlas a mano."], tipo="recordar")
    FIN[id(s)] = 6.95
    notas = ("La regresión lineal busca los pesos que hacen más pequeño el error cuadrático en la muestra. Elastic Net agrega "
             "un castigo al tamaño de los pesos: lambda dice cuánto castigar y alfa mezcla dos castigos, el de lasso, que "
             "lleva coeficientes a cero, y el de ridge, que los encoge sin anularlos. Encoger mete un poco de sesgo a cambio "
             "de menos varianza. Elegimos lambda y alfa con una rejilla de 4 por 30 valores, probando cada pareja en "
             "celdas de 2 kilómetros apartadas dentro del entrenamiento. Como la penalización depende de la escala, primero "
             "estandarizamos las variables, y los vacíos se llenan con la mediana más una marca de faltante. La ventaja: se "
             "lee cada coeficiente y extrapola en línea recta. El límite: solo suma efectos, sin curvas ni interacciones. Y "
             "con tantos avisos por variable, encoger casi no cambia nada, como veremos en los resultados.")
    return s, notas


def _arboles(deck):
    s = _nueva(deck, "CART y Random Forest: un árbol de reglas y el promedio de 300 árboles", "fam")
    texto(s, 0.55, 1.3, 6.0, 0.4, "CART: un árbol de regresión", size=16, color="navy")
    w_, h_ = _formula(s, "c_arbol", 0.75, 1.75, escala=0.7, alt="Predicción de un árbol")
    texto(s, 0.55, 1.75 + h_ + 0.05, 6.0, 1.7, "Las regiones R son las hojas y γ, el promedio del precio en cada una. Para crecer, prueba "
          "cada variable y cada punto de corte, se queda con el que más baja el error cuadrático, parte los avisos en dos y "
          "repite en cada rama (partición binaria recursiva; Breiman y otros, 1984).", size=14, color="tinta")
    tarjeta(s, 0.55, 3.75, 6.0, 1.75, "Flexibilidad", [
        "Profundidad máxima elegida entre 6, 8, 10, 12 y 15 con la partición interna de cada pliegue; al menos 20 avisos por "
        "hoja. Más profundo: menos sesgo y más varianza."], tipo="definicion")
    texto(s, 6.95, 1.3, 5.85, 0.4, "Random Forest: muchos árboles distintos, promediados", size=16, color="navy")
    w2, h2 = _formula(s, "c_bosque", 7.15, 1.75, escala=0.7, alt="Promedio de árboles del bosque")
    texto(s, 6.95, 1.75 + h2 + 0.05, 5.85, 1.7, "Cada árbol T aprende de una muestra con reemplazo de los avisos (bootstrap) y "
          "en cada corte solo puede usar el 30 % de las variables, sorteadas (48 de 162): los árboles se parecen menos entre "
          "sí y su promedio cancela más ruido (Breiman, 2001).", size=14, color="tinta")
    tarjeta(s, 6.95, 3.75, 5.85, 1.75, "Flexibilidad", [
        "300 árboles sin límite de profundidad y al menos 5 avisos por hoja: valores fijos de uso común, sin búsqueda."],
        tipo="definicion")
    tarjeta(s, 0.55, 5.65, 12.25, 1.3, "La diferencia entre los dos", [
        "Un árbol profundo memoriza su muestra: tiene mucha varianza. Promediar árboles poco parecidos baja la varianza "
        "sin subir el sesgo, y el bosque puede usar árboles más profundos que un árbol solo."], tipo="clave")
    FIN[id(s)] = 6.95
    notas = ("Un árbol de regresión parte el mercado en hojas. Para cada corte prueba todas las variables y todos los puntos "
             "posibles y se queda con el que más reduce el error; luego repite en cada rama. Predice el promedio de la hoja. "
             "Su flexibilidad es la profundidad: lo elegimos entre 6 y 15 niveles mirando el error en una parte apartada del "
             "entrenamiento, con al menos 20 avisos por hoja. Random Forest entrena 300 árboles, cada uno con una muestra "
             "sorteada de avisos, y en cada corte solo deja mirar el 30 % de las variables, también sorteado. Así los "
             "árboles son distintos entre sí, y al promediarlos el ruido de cada uno se cancela. Sus valores los fijamos sin "
             "búsqueda. La clave: un árbol solo tiene mucha varianza; el promedio la baja sin agregar sesgo.")
    return s, notas


def _boosting(deck):
    s = _nueva(deck, "Boosting: árboles pequeños en secuencia que corrigen el error anterior", "fam")
    w_, h_ = _formula(s, "c_boost", 0.75, 1.3, escala=0.7, alt="Suma de árboles del boosting")
    w2, h2 = _formula(s, "c_residuo", 0.75, 1.3 + h_ + 0.1, escala=0.7, alt="Cada árbol ajusta el residuo")
    y0 = 1.3 + h_ + h2 + 0.2
    texto(s, 0.55, y0, 5.9, 1.6, "Empieza en el promedio del precio y suma árboles uno a uno. Cada árbol nuevo aprende el residuo r, "
          "el error que dejaron los anteriores, y entra multiplicado por un paso pequeño η, la tasa de aprendizaje: aprende despacio "
          "(Friedman, 2001; Ke y otros, 2017; Prokhorenkova y otros, 2018).", size=13, color="tinta")
    tarjeta(s, 0.55, 4.25, 5.9, 1.25, "Parada temprana", [
        "Sumar árboles mientras baje el error en avisos apartados del entrenamiento y parar cuando deja de bajar: regulariza."],
        tipo="definicion")
    tarjeta(s, 0.55, 5.65, 5.9, 1.3, "Frente al bosque", [
        "El bosque promedia árboles independientes y baja la varianza; el boosting suma correcciones y baja sobre todo el "
        "sesgo. Su riesgo es sobreajustar si suma demasiados."], tipo="clave")
    filas = [("Aspecto", "LightGBM", "CatBoost"),
             ("Forma de cada árbol", "crece por la hoja que más baja el error; hasta 63 hojas y 20 avisos por hoja",
              "simétrico: la misma pregunta en todo un nivel; profundidad 8"),
             ("Tasa de aprendizaje η", "0,05", "0,05"),
             ("Penalización de las hojas", "λ = 1", "λ = 3"),
             ("Muestreo en cada árbol", "80 % de los avisos y 70 % de las variables", "el que trae por defecto"),
             ("Número de árboles", "parada temprana: hasta 5.000; para si 100 seguidos no mejoran",
              "parada temprana: hasta 4.000; para si 150 seguidos no mejoran"),
             ("Pérdida", "error cuadrático", "error cuadrático")]
    y1 = _tabla(s, 6.75, 1.3, filas, [1.9, 2.25, 1.9], size=12, minimo=0.3)
    tarjeta(s, 6.75, y1 + 0.15, 6.05, 6.95 - y1 - 0.15, "Cómo se eligieron", [
        "El número de árboles, con la partición interna de cada pliegue; después se reentrena con todo el entrenamiento "
        "del pliegue. Lo demás se fijó a mano. En LightGBM, la pérdida y las hojas se compararon antes en validación "
        "espacial con variables anteriores: 31 hojas y 50 avisos por hoja dieron 148,8 en Chapinero fuera, frente a 150,1 con "
        "63 y 20, casi lo mismo."], tipo="recordar")
    FIN[id(s)] = 6.95
    notas = ("El boosting empieza prediciendo el promedio y suma árboles pequeños. Cada árbol nuevo aprende el error que "
             "dejaron los anteriores, el residuo, y entra con un paso pequeño, la tasa de aprendizaje de 0,05: el modelo "
             "aprende despacio. Mientras el bosque promedia árboles independientes para bajar la varianza, el boosting suma "
             "correcciones y baja sobre todo el sesgo; por eso hay que pararlo a tiempo. La parada temprana suma árboles "
             "mientras el error baje en avisos apartados y para cuando deja de bajar. Lo corrimos con dos librerías. "
             "LightGBM hace crecer cada árbol por la hoja que más ayuda, con hasta 63 hojas; CatBoost hace árboles "
             "simétricos de profundidad 8. El número de árboles se eligió con la partición interna; lo demás lo fijamos a "
             "mano, y en LightGBM comparamos antes la pérdida y el número de hojas: casi no cambiaba.")
    return s, notas


def _red(deck):
    s = _nueva(deck, "Red con atención: capas de neuronas que además miran el precio de 20 vecinos", "fam")
    cajas = [("162 variables del aviso, estandarizadas", "tinte", "navy"),
             ("Codificador: 256 y 128 neuronas, normalización por lotes y dropout", "azul_claro", "azul"),
             ("Atención: pondera precio, precio por m² y características de 20 vecinos según parecido y distancia",
              "ambar_claro", "ambar"), ("Cabeza de 128 neuronas: precio predicho", "tinte", "navy")]
    for i, (t, fondo, borde) in enumerate(cajas):
        x0 = 0.55 + i * 3.1
        nodo(s, x0, 1.3, 2.8, 1.3, [t], relleno=fondo, borde=borde, sizes=(14,), colores=("tinta",))
        if i < 3:
            linea(s, x0 + 2.82, 1.95, x0 + 3.08, 1.95, color="gris", ancho=1.5)
    w_, h_ = _formula(s, "c_atencion", 0.65, 2.75, escala=0.68, alt="Fórmula de la atención")
    texto(s, 0.55, 2.75 + h_ + 0.02, 12.25, 0.62, "e resume el aviso; q y k comparan el aviso con cada vecino; v lleva "
          "el precio y las características del vecino. Los pesos a suman uno: la red usa un promedio ponderado de la "
          "vecindad (Vaswani y otros, 2017; Viana y Barbosa, 2021).", size=12, color="muted")
    tarjeta(s, 0.55, 4.45, 4.0, 2.5, "Regularización", [
        "Dropout de 0,2 (apaga neuronas al azar al entrenar), normalización por lotes y decaimiento de pesos de 0,0001 "
        "sin penalizar sesgos; parada temprana."], tipo="definicion")
    tarjeta(s, 4.7, 4.45, 4.0, 2.5, "Cómo se eligió", [
        "Las épocas, con parada temprana en la partición interna: hasta 200 y para si 12 seguidas no mejoran. Lo demás, a "
        "mano: pérdida absoluta, AdamW con tasa 0,001, lotes de 512, atención de dimensión 32."], tipo="neutra")
    tarjeta(s, 8.85, 4.45, 3.95, 2.5, "Vecinos sin fuga", [
        "Solo del entrenamiento del pliegue, sin la misma coordenada ni avisos con el mismo texto. La red enviada a Kaggle "
        f"fue otra, más simple: dos capas de 10 neuronas ({n(PRIV['nn_clase_2x10_D_coordenadas'])} en la privada)."],
        tipo="recordar")
    FIN[id(s)] = 6.95
    notas = ("La sexta familia es una red neuronal. Las 162 variables pasan por dos capas de 256 y 128 neuronas que las "
             "transforman; luego un mecanismo de atención mira los 20 avisos de entrenamiento más cercanos y decide cuánto "
             "pesa cada uno según qué tan parecido y qué tan cerca está; los pesos suman uno, así que es un promedio "
             "ponderado de la vecindad, con su precio. Una última capa da el precio. Como las redes tienden a sobreajustar, "
             "la regularizamos con dropout, que apaga neuronas al azar mientras entrena, normalización por lotes, "
             "decaimiento de pesos y parada temprana. Las épocas se eligen con la partición interna; el resto lo fijamos a "
             "mano. Los vecinos salen solo del entrenamiento del pliegue, sin copias del mismo aviso. La red que subimos a "
             "Kaggle fue una más sencilla, de dos capas de 10 neuronas.")
    return s, notas


def _hiperparametros(deck):
    s = _nueva(deck, "Hiperparámetros de cada familia: qué fija su flexibilidad y cómo se eligió", "fam")
    filas = [("Modelo", "Qué fija su flexibilidad", "Valores", "Cómo se eligió"),
             ("Regresión lineal", "nada: usa las 162 variables", "no tiene", "no hay nada que elegir"),
             ("Elastic Net", "λ (cuánto encoge) y α (mezcla)", "30 valores de λ por 4 de α (0,1; 0,5; 0,9; 1)",
              "validación cruzada interna con 3 particiones de celdas de 2 km"),
             ("CART", "profundidad; avisos por hoja", "6, 8, 10, 12 o 15 niveles; 20 por hoja",
              "profundidad con la partición interna; 20, fijo"),
             ("Random Forest", "árboles, variables por corte, avisos por hoja", "300; 30 %; 5", "fijos, sin búsqueda"),
             ("LightGBM", "árboles, η, hojas, λ, muestreos", "hasta 5.000 (parada temprana); 0,05; 63; 1; 80 % y 70 %",
              "árboles con la partición interna; el resto a mano"),
             ("CatBoost", "árboles, η, profundidad, λ", "hasta 4.000 (parada temprana); 0,05; 8; 3",
              "árboles con la partición interna; el resto a mano"),
             ("Red con atención", "épocas, neuronas, dropout, decaimiento", "hasta 200 (parada temprana); 256 y 128; 0,2; 0,0001",
              "épocas con la partición interna; el resto a mano")]
    y1 = _tabla(s, 0.55, 1.3, filas, [1.85, 3.2, 3.6, 3.6], size=13, minimo=0.34)
    tarjeta(s, 0.55, y1 + 0.15, 6.05, 6.95 - y1 - 0.15, "Partición interna", [
        "Dentro del entrenamiento de cada pliegue se aparta una parte con la misma lógica de la validación: 20 % al azar en "
        "la CV aleatoria, un clúster en la de clústeres y una localidad entera en Chapinero fuera. Así se elige sin mirar "
        "los avisos que se validan."], tipo="definicion")
    tarjeta(s, 6.75, y1 + 0.15, 6.05, 6.95 - y1 - 0.15, "Limitación", [
        "Solo Elastic Net y CART tienen rejilla; en el boosting y la red solo se eligió el número de árboles o de épocas. "
        "Afinar más acortaría distancias, pero en LightGBM cambiar las hojas movió 1,3 millones y entre familias hay de 6 "
        "a 42."], tipo="recordar")
    FIN[id(s)] = 6.95
    notas = ("Esta es la ficha de las siete especificaciones. La regresión lineal no tiene nada que elegir. Elastic Net "
             "prueba 120 parejas de lambda y alfa con tres particiones espaciales. CART elige su profundidad entre cinco "
             "valores. Random Forest usa valores fijos de uso común. En el boosting y en la red, lo que se elige con datos es "
             "cuándo parar: el número de árboles o de épocas; lo demás se fijó a mano. Todo lo que se elige con datos usa "
             "una partición interna que imita la validación: si validamos dejando Chapinero fuera, la parte apartada para "
             "elegir es otra localidad entera. Lo decimos de frente: no hubo una rejilla completa para el boosting ni la red. "
             "Pero lo que movían esos ajustes, cerca de un millón, es poco frente a lo que separa a las familias.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# Reglas y validación
# ------------------------------------------------------------------------------------------------------------------
def _reglas(deck):
    s = _nueva(deck, "Las mismas reglas para las seis familias", "reg")
    filas = [("Regla", "Cómo se cumple"),
             ("Mismas variables", "las mismas 162 en las siete especificaciones: 21 del aviso, 84 del texto, 39 del mapa, 7 "
                                  "del satélite y 11 de Catastro"),
             ("Mismos pliegues", "los mismos sorteos (semilla 2026) para todas: cada aviso se predice con modelos que no lo "
                                 "vieron"),
             ("Misma medida", "MAE en millones sobre todas las predicciones fuera de pliegue juntas; además, el sesgo medio"),
             ("Sin fuga del precio", "el puntaje de precio del texto se recalcula en cada pliegue sin los avisos validados"),
             ("Misma lógica para elegir", "lo que se elige con datos usa una partición interna hecha como la validación externa")]
    y1 = _tabla(s, 0.55, 1.3, filas, [2.6, 9.65], size=15, minimo=0.5)
    tarjeta(s, 0.55, y1 + 0.2, 6.05, 6.95 - y1 - 0.2, "Lo que no es igual, y por qué", [
        "Lineales y red llenan los faltantes con la mediana y estandarizan; los árboles usan los datos tal cual. La red usa "
        "además el precio de sus 20 vecinos y la pérdida absoluta. Elastic Net elige siempre con celdas de 2 km."],
        tipo="recordar")
    tarjeta(s, 6.75, y1 + 0.2, 6.05, 6.95 - y1 - 0.2, "Qué se corrió", [
        "Las siete especificaciones con clústeres y Chapinero fuera; todas menos CatBoost con la CV aleatoria (por tiempo). "
        "LightGBM, además, con bloques de 2 km, bloques con buffer de 300 m y kNNDM."], tipo="neutra")
    FIN[id(s)] = 6.95
    notas = ("Para que la comparación sea justa, las reglas son las mismas. Las siete especificaciones reciben exactamente las "
             "mismas 162 variables, se prueban con los mismos pliegues, sorteados con la misma semilla, y se miden igual: el "
             "error absoluto medio de todas las predicciones fuera de pliegue. El puntaje de precio que sale del texto se "
             "vuelve a calcular en cada pliegue sin los avisos que se validan, para que ningún modelo vea el precio por la "
             "puerta de atrás. Y todo lo que se elige con datos usa una partición interna hecha como la validación. Hay "
             "diferencias necesarias: los lineales y la red no aceptan vacíos, así que se llenan y se estandarizan; la red usa "
             "el precio de sus vecinos. CatBoost no se corrió con la validación aleatoria por tiempo.")
    return s, notas


def _validaciones(deck):
    s = _nueva(deck, "Tres validaciones: del azar a predecir una zona que el modelo no vio", "reg")
    imagen(s, IMG3 / "mapas_validaciones.png", 0.45, 1.3, w=5.3, alt="Mapas de un pliegue de cada validación y de la prueba real")
    filas = [("Validación", "Cómo se arma cada pliegue", "Pliegues", "Vecino mediano"),
             ("CV aleatoria", "cada aviso se sortea a uno de los pliegues", str(PL["aleatoria"]["pliegues"]),
              f"0 m ({n(PCT0['CV aleatoria'], 0)} % en la misma coordenada)"),
             ("Clústeres k-means", "k-means agrupa las coordenadas en 10 zonas; cada zona es un pliegue",
              str(PL["clusters"]["pliegues"]), f"{n(MED['Clústeres k-means'], 0)} m"),
             ("Chapinero fuera", f"se entrena con los {n(PL[CHS]['ent_medio'], 0)} avisos de fuera de Chapinero y se predicen "
                                 "sus 307", "1", f"{n(MED['Chapinero fuera'], 0)} m"),
             (("Prueba real (Kaggle)", "naranja"), ("10.286 avisos, el 94 % en la localidad de Chapinero", "naranja"),
              ("", "naranja"), (f"{n(MED['Prueba real (Kaggle)'], 0)} m ({n(PCT0['Prueba real (Kaggle)'], 0)} % en la misma "
                                "coordenada)", "naranja"))]
    y1 = _tabla(s, 6.0, 1.3, filas, [1.55, 2.6, 0.95, 1.7], size=12, minimo=0.3)
    texto(s, 6.0, y1 + 0.05, 6.8, 0.62, "Vecino mediano: distancia típica entre un aviso que se predice y su aviso de "
          "entrenamiento más cercano. Las dos primeras predicen los 38.644 avisos una vez.", size=12, color="muted")
    tarjeta(s, 6.0, y1 + 0.8, 6.8, 6.95 - y1 - 0.8, "Por qué Chapinero fuera es la principal", [
        "Imita la tarea: predecir una zona cara que el modelo casi no vio. La CV aleatoria deja vecinos en el entrenamiento "
        "y premia memorizar lo local; los clústeres quitan los vecinos, pero sus zonas se parecen más al resto de la ciudad "
        "que Chapinero."],
        tipo="clave")
    FIN[id(s)] = 6.95
    notas = ("Probamos cada familia de tres formas. En los mapas, en naranja, los avisos que se predicen en un pliegue, y en "
             "gris, con los que se entrena. La validación cruzada aleatoria sortea avisos sueltos en 5 pliegues: casi dos de "
             "cada tres avisos validados tienen otro de entrenamiento en la misma coordenada, a veces el mismo inmueble "
             "publicado dos veces. Los clústeres agrupan la ciudad en 10 zonas con k-means, un método que junta puntos "
             "cercanos, y cada zona es un pliegue: el vecino de entrenamiento queda a casi un kilómetro. Chapinero fuera "
             "entrena con toda la ciudad menos Chapinero y predice sus 307 avisos. El último mapa es la prueba real de "
             "Kaggle. Usamos Chapinero fuera como principal porque es la que imita la tarea. Con LightGBM probamos además "
             "bloques de 2 kilómetros, un buffer de 300 metros y kNNDM; esos resultados están en el otro deck.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# Resultado
# ------------------------------------------------------------------------------------------------------------------
def _resultado(deck):
    s = _nueva(deck, "Resultado: el boosting tiene el menor error en las tres validaciones", "res")
    texto(s, 0.55, 1.3, 7.8, 0.4, "MAE por validación (millones); CatBoost no se corrió con la CV aleatoria", size=13,
          color="muted")
    cd = CategoryChartData()
    cd.categories = [nom for _, nom in FAM]
    ser = [(AL, "gris_claro"), (CL, "azul_suave"), (CH, "navy")]
    for esq, _ in ser:
        cd.add_series(esq, [M.get((f, esq), None) for f, _ in FAM])
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.45), Inches(1.65), Inches(7.9), Inches(5.3), cd)
    ch = gf.chart
    _estilo_grafico(ch, 12)
    ch.has_title = False
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.TOP
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(12)
    ch.plots[0].gap_width, ch.plots[0].overlap = 45, -5
    for sr, (_, c) in zip(ch.plots[0].series, ser):
        sr.format.fill.solid()
        sr.format.fill.fore_color.rgb = rgb(c)
    _ejes(ch, valor_visible=False, minimo=0, maximo=230, invertir_cat=True, grid=False)
    etiquetas_puntos(ch, [[n(M[(f, e)]) if (f, e) in M else None for f, _ in FAM] for e, _ in ser], size=12,
                     posicion=XL_LABEL_POSITION.OUTSIDE_END)
    tarjeta(s, 8.6, 1.3, 4.2, 1.85, "", [
        f"El boosting es el mejor en las tres: LightGBM en la CV aleatoria y en Chapinero fuera; CatBoost en clústeres "
        f"({n(M[('catboost', CL)])} frente a {n(M[('lightgbm', CL)])})."], tipo="neutra")
    d = {f: M[(f, CH)] - M[("lightgbm", CH)] for f, _ in FAM}
    tarjeta(s, 8.6, 3.3, 4.2, 1.75, "Distancias en Chapinero fuera", [
        f"LightGBM le saca {n(d['catboost'], 0)} millones a CatBoost, {n(d['random_forest'], 0)} a Random Forest, "
        f"{n(d['cart'], 0)} a CART, {n(d['nn_atencion'], 0)} a la red y {n(d['elastic_net'], 0)} a "
        f"{n(d['ols'], 0)} a los lineales."], tipo="neutra")
    tarjeta(s, 8.6, 5.2, 4.2, 1.75, "Elastic Net y la regresión lineal empatan", [
        f"{n(M[('elastic_net', CH)])} frente a {n(M[('ols', CH)])}: con tantos avisos por variable, encoger los coeficientes "
        "había poca varianza que quitar."], tipo="recordar")
    FIN[id(s)] = 6.95
    notas = (f"Este es el resultado. Con las mismas reglas, el boosting es el mejor en las tres validaciones: LightGBM en la "
             f"aleatoria y en Chapinero fuera, y CatBoost, por menos de un millón, en clústeres. En Chapinero fuera, que es la "
             f"que imita la tarea, LightGBM se equivoca {n(M[('lightgbm', CH)])} millones en promedio; le siguen CatBoost, "
             f"Random Forest, el árbol solo y la red, y al final los dos lineales, con unos 185. Elastic Net y la regresión "
             "lineal empatan: con 38.644 avisos para 162 variables, la varianza de los coeficientes ya es pequeña y "
             "encogerlos no ayuda. Fíjense también en el nivel: con la validación aleatoria todos parecen mucho mejores. A "
             "eso vamos ahora.")
    return s, notas


def _flexibilidad(deck):
    s = _nueva(deck, "Más flexible no siempre es mejor cuando hay que predecir una zona nueva", "res")
    orden = [("elastic_net", "Elastic Net"), ("ols", "Regresión lineal"), ("cart", "CART"),
             ("random_forest", "Random Forest"), ("lightgbm", "LightGBM"), ("nn_atencion", "Red con atención")]
    texto(s, 0.55, 1.3, 7.0, 0.62, "MAE (millones) de menos a más flexible, de izquierda a derecha (orden aproximado)",
          size=13, color="muted")
    ser = [(AL, [M[(f, AL)] for f, _ in orden], "gris", 2.25, False), (CL, [M[(f, CL)] for f, _ in orden], "azul_suave", 2.25, False),
           (CH, [M[(f, CH)] for f, _ in orden], "navy", 3.0, False)]
    ch = _lineas_cat(s, 0.45, 1.85, 7.2, 4.4, [nom for _, nom in orden], ser, minimo=50, maximo=200, unidad=50)
    etiquetas_puntos(ch, [[n(v) for v in vals] for _, vals, *_ in ser], size=12)
    for k, sr in enumerate(ch.plots[0].series):
        for i in range(len(orden)):
            sr.points[i].data_label.position = XL_LABEL_POSITION.ABOVE if k == 2 else XL_LABEL_POSITION.BELOW
    texto(s, 0.55, 6.3, 7.0, 0.65, "Elastic Net va primero porque encoger los coeficientes la hace un poco menos flexible "
          "que la regresión lineal; el orden entre bosque, boosting y red es aproximado.", size=12, color="muted")
    op = R["optimismo"]
    tarjeta(s, 7.95, 1.3, 4.85, 1.75, "La forma de U", [
        "Con la CV aleatoria el error cae casi siempre al ganar flexibilidad. Al predecir Chapinero, la curva baja hasta el boosting y "
        "vuelve a subir con la red, la más flexible: es la forma que anticipa el dilema entre sesgo y varianza."], tipo="neutra")
    tarjeta(s, 7.95, 3.2, 4.85, 2.15, "Por qué la CV aleatoria premia lo local", [
        f"El {n(PCT0['CV aleatoria'], 0)} % de sus avisos validados tiene uno de entrenamiento en la misma coordenada y "
        "sus errores tienen correlación de 0,31. Si dos vecinos tienen error de varianza σ² y correlación ρ, copiar al "
        "vecino da 2σ²(1 − ρ) en vez de 2σ²: promete 2σ²ρ de menos (Roberts y otros, 2017)."], tipo="cuidado")
    tarjeta(s, 7.95, 5.5, 4.85, 1.45, "Cuánto sube el error sin Chapinero", [
        f"Red ×{n(op['nn_atencion']['razon'], 1)}; LightGBM ×{n(op['lightgbm']['razon'], 1)}; Random Forest "
        f"×{n(op['random_forest']['razon'], 1)}; CART y lineales ×{n(op['cart']['razon'], 1)}. Pierde más quien más usaba "
        "lo cercano."], tipo="recordar")
    FIN[id(s)] = 6.95
    notas = ("Esta gráfica es la del sesgo y la varianza, con nuestros datos. De izquierda a derecha, las familias de menos "
             "a más flexibles. En gris, la validación aleatoria: el error cae casi todo el camino, como si más flexible "
             "siempre fuera mejor. En azul oscuro, Chapinero fuera: baja hasta el boosting y vuelve a subir con la red. Ahí "
             "más flexibilidad ya no reduce el sesgo; agrega varianza. Por qué la validación aleatoria engaña: dos de cada "
             "tres avisos que valida tienen un aviso de entrenamiento en la misma coordenada, y sus errores se parecen. Un "
             "modelo flexible aprende ese ruido local y la validación lo premia. La cuenta: si el error de dos vecinos tiene "
             "correlación rho, copiar al vecino promete un error cuadrático menor en dos sigma cuadrado rho. Por eso, al "
             "dejar Chapinero fuera, el error de la red se duplica, y el del boosting y el bosque sube 70 u 80 %, más que "
             "el del árbol solo y los lineales.")
    return s, notas


def _diferencias(deck):
    s = _nueva(deck, "¿Son reales las diferencias? Sí: ningún intervalo toca el cero", "res")
    texto(s, 0.55, 1.3, 8.2, 0.62, "Chapinero fuera: diferencia de MAE frente a LightGBM sobre los mismos 307 avisos "
          "(millones), con su intervalo de confianza de 95 %", size=13, color="muted")
    filas_f = [("catboost", "CatBoost"), ("random_forest", "Random Forest"), ("cart", "CART"), ("nn_atencion", "Red con atención"),
               ("elastic_net", "Elastic Net"), ("ols", "Regresión lineal")]
    X0F, WF, LABF = 0.55, 8.2, 2.0
    vmin, vmax = -5, 60
    ancho_txt = 2.0

    def esc(v):
        return X0F + LABF + (v - vmin) / (vmax - vmin) * (WF - LABF - ancho_txt)

    y0f, hf = 2.45, 0.6
    for i, (f, nom) in enumerate(filas_f):
        yy = y0f + i * hf
        b, bb = BO[f], BL[f]
        texto(s, X0F, yy, LABF - 0.1, hf, nom, size=14, color="tinta", align="right", anchor="middle")
        linea(s, esc(bb["bajo"]), yy + hf * 0.68, esc(bb["alto"]), yy + hf * 0.68, color="azul", ancho=3)
        linea(s, esc(b["dif_bajo"]), yy + hf * 0.36, esc(b["dif_alto"]), yy + hf * 0.36, color="gris", ancho=3)
        forma(s, MSO_SHAPE.OVAL, esc(b["dif"]) - 0.09, yy + hf / 2 - 0.09, 0.18, 0.18, relleno="navy")
        texto(s, esc(vmax) + 0.1, yy, ancho_txt - 0.05, hf,
              [[(f"{n(b['dif'], 1, True)} ", {"color": "navy"}), (f"({n(b['dif_bajo'], 1)} a {n(b['dif_alto'], 1)})", {"color": "gris"})],
               [(f"celdas: {n(bb['bajo'], 1)} a {n(bb['alto'], 1)}", {"color": "azul"})]],
              size=12, color="tinta", anchor="middle", espacio=0)
    yb = y0f + len(filas_f) * hf
    linea(s, esc(0), y0f - 0.1, esc(0), yb, color="navy", ancho=1.25)
    linea(s, esc(vmin), yb, esc(vmax), yb, color="gris_claro")
    for v in (0, 10, 20, 30, 40, 50, 60):
        texto(s, esc(v) - 0.25, yb + 0.05, 0.5, 0.3, n(v, 0), size=12, color="muted", align="center")
    texto(s, esc(0) + 0.05, y0f - 0.42, 3.0, 0.3, "0: igual que LightGBM", size=12, color="navy")
    texto(s, esc(vmax) + 0.1, y0f - 0.42, ancho_txt, 0.3, "Diferencia (intervalos)", size=12, color="muted")
    yl = yb + 0.42
    linea(s, 0.7, yl + 0.15, 1.2, yl + 0.15, color="gris", ancho=3)
    texto(s, 1.3, yl, 3.3, 0.3, "avisos remuestreados uno a uno", size=12, color="tinta")
    linea(s, 4.7, yl + 0.15, 5.2, yl + 0.15, color="azul", ancho=3)
    texto(s, 5.3, yl, 3.5, 0.3, f"celdas de 1 km remuestreadas ({BL['celdas']})", size=12, color="tinta")
    tarjeta(s, 9.0, 1.3, 3.8, 2.1, "Bootstrap pareado", [
        "Se sortean 10.000 veces, con reemplazo, los 307 avisos, y en cada sorteo se calcula la diferencia de MAE sobre los "
        "mismos avisos. El intervalo deja fuera el 2,5 % de cada lado."], tipo="definicion")
    tarjeta(s, 9.0, 3.55, 3.8, 1.75, "Por celdas de 1 km", [
        "Los vecinos comparten error, así que también se sortean celdas enteras (Künsch, 1989): los intervalos se "
        "ensanchan, pero ninguno toca el cero."], tipo="neutra")
    tarjeta(s, 9.0, 5.45, 3.8, 1.5, "En clústeres", [
        f"La ventaja de CatBoost ({n(-BLC['catboost']['dif'], 1)}) va de {n(-BLC['catboost']['alto'], 1)} a "
        f"{n(-BLC['catboost']['bajo'], 1)} por celdas de 1 km ({BLC['celdas']} celdas): real, pero pequeña."],
        tipo="recordar")
    FIN[id(s)] = 6.95
    notas = ("Con solo 307 avisos de Chapinero, ¿son reales estas diferencias? Hicimos un bootstrap pareado: sorteamos 10.000 "
             "veces los mismos 307 avisos, con reemplazo, y en cada sorteo medimos cuánto peor le va a cada familia que a "
             "LightGBM en esos mismos avisos. La línea gris es el intervalo de 95 %. Como los avisos vecinos comparten "
             "error, repetimos sorteando celdas enteras de un kilómetro, en azul: los intervalos se ensanchan, sobre todo "
             "el de la red, pero ninguno toca el cero. La distancia más pequeña es con CatBoost, la otra versión de boosting: "
             "unos 6 millones, con intervalo de 1 a 14. El salto grande es entre familias. Y en clústeres, donde gana "
             "CatBoost, su ventaja es real pero de menos de un millón.")
    return s, notas


def _estandar_espacial(deck):
    s = _nueva(deck, "Validación estándar y espacial: el mismo ganador, distinto orden detrás", "kag")
    ESQ = [AL, CL, CH]
    XS = [2.75, 4.6, 6.45]
    rangos = {}
    for esq in ESQ:
        vals = sorted([(M[(f, esq)], f) for f, _ in FAM if (f, esq) in M])
        for r_, (_, f) in enumerate(vals, 1):
            rangos[(f, esq)] = r_

    def ypos(r):
        return 2.0 + 0.6 * (r - 1)

    for x_, e in zip(XS, ESQ):
        texto(s, x_ - 0.9, 1.3, 1.8, 0.4, e, size=14, color="navy", align="center")
    ESTILO = {"lightgbm": ("azul", 3.5), "catboost": ("azul_suave", 2.5), "nn_atencion": ("ambar", 3.5)}
    orden_dibujo = [f for f, _ in FAM if f not in ESTILO] + ["catboost", "nn_atencion", "lightgbm"]
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
                texto(s, 0.45, y_ - 0.17, x_ - 0.6, 0.34, f"{NOM[f]} {n(M[(f, e)])}", size=13, color=c_txt, align="right",
                      anchor="middle")
            elif e == CL:
                forma(s, MSO_SHAPE.RECTANGLE, x_ + 0.11, y_ - 0.13, 0.56, 0.26, relleno="blanco")
                texto(s, x_ + 0.11, y_ - 0.13, 0.56, 0.26, n(M[(f, e)]), size=12, color=c_txt, anchor="middle", margen=0.02)
            else:
                texto(s, x_ + 0.15, y_ - 0.17, 2.1, 0.34, f"{NOM[f]} {n(M[(f, e)])}", size=13, color=c_txt, anchor="middle")
    texto(s, 0.55, 6.3, 8.0, 0.65, "Puesto de cada especificación (1 arriba) y su MAE en millones. CatBoost no se corrió con "
          "la CV aleatoria. Estándar: CV aleatoria; espaciales: clústeres k-means y Chapinero fuera.", size=12, color="muted")
    sp = R["spearman"]
    tarjeta(s, 8.85, 1.3, 3.95, 1.5, "Sí para el ganador", [
        f"El boosting es primero en las tres (CatBoost supera a LightGBM en clústeres por "
        f"{n(M[('lightgbm', CL)] - M[('catboost', CL)])})."], tipo="clave")
    tarjeta(s, 8.85, 2.95, 3.95, 1.9, "No para el resto", [
        f"Correlación de los puestos de las seis familias: {n(sp['aleatoria_chapinero'], 2)} entre la CV aleatoria y "
        f"Chapinero fuera, {n(sp['clusters_chapinero'], 2)} entre clústeres y Chapinero fuera (Spearman)."],
        tipo="recordar")
    tarjeta(s, 8.85, 5.0, 3.95, 1.95, "Por qué cae la red", [
        f"Es segunda en la CV aleatoria ({n(M[('nn_atencion', AL)])}) porque promedia precios de vecinos que allí están a "
        f"metros; sin Chapinero, el vecino mediano queda a {n(MED['Chapinero fuera'], 0)} m y en otro mercado, y queda "
        "detrás de Random Forest y de CART."], tipo="neutra")
    FIN[id(s)] = 6.95
    notas = ("¿La validación estándar y las espaciales eligen lo mismo? Para el ganador, sí: el boosting queda primero en las "
             "tres. Para el resto, no del todo. La red con atención, en ámbar, es segunda con pliegues aleatorios "
             "y queda detrás de Random Forest y de CART cuando tiene que predecir Chapinero sin haberlo visto, "
             "porque depende de los precios de vecinos cercanos que allí no tiene. Medido con la correlación de los puestos, "
             "la aleatoria y Chapinero fuera coinciden en 0,83; los clústeres y Chapinero fuera solo en 0,54. Conclusión: la "
             "validación aleatoria no se equivoca de ganador, pero sí exagera a los modelos que se apoyan en lo local.")
    return s, notas


def _kaggle(deck):
    s = _nueva(deck, "¿Qué validación refleja Kaggle? En el nivel, Chapinero fuera", "kag")
    texto(s, 0.55, 1.3, 6.0, 0.4, "El mismo LightGBM (envío 01) en cada validación y en Kaggle", size=13, color="muted")
    esq = [("CV aleatoria", M[("lightgbm", AL)]), ("Clústeres k-means", M[("lightgbm", CL)]), ("Chapinero fuera", M[("lightgbm", CH)]),
           ("Kaggle, tabla privada", P01), ("Kaggle, tabla pública", U01)]
    ch = barras(s, 0.45, 1.65, 6.1, 2.85, [e for e, _ in esq], [("v", [v for _, v in esq],
                                                                ["gris_claro", "azul_suave", "navy", "naranja", "naranja"])],
                minimo=0, maximo=200, gap=35, grid=False)
    etiquetas_puntos(ch, [[n(v) for _, v in esq]])
    nomf = {"Ensamble": "Ensamble de modelos", "Red neuronal": "Red neuronal (2 capas de 10)"}
    texto(s, 6.85, 1.3, 6.0, 0.4, "Mejor envío del equipo a Kaggle por familia (tabla privada)", size=13, color="muted")
    ch = barras(s, 6.75, 1.65, 6.1, 2.85, [nomf.get(f, f) for f in MEJOR_FAM["familia"]],
                [("v", list(MEJOR_FAM["privado"]), ["azul" if f == "Boosting" else "gris_claro" for f in MEJOR_FAM["familia"]])],
                minimo=0, maximo=300, gap=35, grid=False)
    etiquetas_puntos(ch, [[n(v) for v in MEJOR_FAM["privado"]]])
    tarjeta(s, 0.55, 4.7, 4.0, 2.25, "El nivel", [
        f"Chapinero fuera queda a {n(P01 - M[('lightgbm', CH)], 0)} millones de la tabla privada; los clústeres, a "
        f"{n(P01 - M[('lightgbm', CL)], 0)}; la CV aleatoria, a {n(P01 - M[('lightgbm', AL)], 0)}: promete un error "
        f"{n(100 * (1 - M[('lightgbm', AL)] / P01), 0)} % menor."], tipo="clave")
    tarjeta(s, 4.7, 4.7, 4.0, 2.25, "El orden entre familias", [
        "En Kaggle el boosting también gana, pero cada envío usó variables distintas, así que ese orden no sirve para elegir "
        "la validación."], tipo="neutra")
    tarjeta(s, 8.85, 4.7, 3.95, 2.25, "Comparación imperfecta", [
        "La red enviada no es la red con atención y el ensamble combina varios modelos. Elastic Net no se envió: solo se "
        "evaluó en las validaciones, donde empata con la regresión lineal."], tipo="recordar")
    FIN[id(s)] = 6.95
    notas = (f"¿Qué validación se parece a Kaggle? En el nivel, la que deja Chapinero fuera: {n(M[('lightgbm', CH)])} frente a "
             f"{n(P01)} en la tabla privada y {n(U01)} en la pública. Los clústeres ya quitan a los vecinos y aun así quedan "
             f"{n(P01 - M[('lightgbm', CL)], 0)} millones por debajo: no basta con alejarse, hay que predecir otro mercado, "
             f"el de Chapinero, más caro. La validación aleatoria promete {n(M[('lightgbm', AL)])}, casi la mitad. A la "
             "derecha, el mejor envío de cada familia en Kaggle: el boosting también gana, pero esa comparación es "
             "imperfecta, porque cada envío usó variables distintas. El taller pedía envíos de las seis familias y Elastic "
             "Net no se envió; la evaluamos con las mismas reglas que las demás y empata con la regresión lineal, así que no "
             "habría cambiado el orden.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# Por qué gana el boosting
# ------------------------------------------------------------------------------------------------------------------
def _forma(deck):
    s = _nueva(deck, "Por qué gana el boosting (1): el precio no crece en línea recta", "por")
    texto(s, 0.55, 1.3, 6.9, 0.62, "Aporte del área a la predicción de LightGBM (valores de Shapley, millones), en el "
          "entrenamiento (gris) y en la prueba de Chapinero (naranja)", size=13, color="muted")
    imagen(s, IMG / "dependencia_area.png", 0.6, 1.95, w=6.8, alt="Aporte SHAP del área en entrenamiento y en la prueba")
    tarjeta(s, 7.75, 1.3, 5.05, 1.85, "Valor de Shapley (SHAP)", [
        "Cuánto suma o resta una variable a la predicción de un aviso frente al promedio (Lundberg y Lee, 2017). Reparte "
        "la predicción del modelo: no es un efecto causal ni el precio implícito de Rosen."], tipo="definicion")
    tarjeta(s, 7.75, 3.3, 5.05, 2.0, "Lo que el modelo aprendió", [
        "El aporte del área baja hasta unos 70 m², sube con fuerza entre 100 y 250 m² y luego se aplana. Y a igual área, "
        "la prueba de Chapinero recibe más que el entrenamiento: el tamaño pesa distinto según el lugar, una interacción."],
        tipo="neutra")
    tarjeta(s, 7.75, 5.45, 5.05, 1.5, "Lo que un modelo lineal no puede", [
        f"Impone una recta y la misma pendiente en toda la ciudad: {n(M[('ols', CH)])} frente a {n(M[('lightgbm', CH)])} en "
        "Chapinero fuera."], tipo="clave")
    FIN[id(s)] = 6.95
    notas = ("Primera razón: la forma. Esta gráfica muestra cuánto aporta el área a la predicción de LightGBM, aviso por aviso, "
             "medido con valores de Shapley: lo que el área suma o resta frente al promedio. La curva no es una recta: "
             "casi no cambia en los apartamentos pequeños, sube con fuerza entre 100 y 250 metros y luego se aplana. Y los "
             "puntos naranja, los avisos de Chapinero que hay que predecir, quedan por encima de los grises con la misma "
             "área: el tamaño vale más según dónde esté. Eso es una interacción entre tamaño y ubicación. Un árbol la capta "
             "sola, porque pregunta por el área dentro de cada zona; un modelo lineal impone una recta con la misma "
             "pendiente en toda la ciudad, y por eso se equivoca 42 millones más. Ojo: esto describe al modelo, no es un "
             "efecto causal.")
    return s, notas


def _promediar_corregir(deck):
    s = _nueva(deck, "Por qué gana el boosting (2): promediar baja la varianza; corregir baja el sesgo", "por")
    texto(s, 0.55, 1.3, 6.0, 0.4, "Chapinero fuera (millones)", size=13, color="muted")
    cats = ["Un árbol (CART)", "Bosque promediado (Random Forest)", "Árboles en secuencia (LightGBM)"]
    fams = ["cart", "random_forest", "lightgbm"]
    cd = CategoryChartData()
    cd.categories = cats
    cd.add_series("MAE", [M[(f, CH)] for f in fams])
    cd.add_series("Sesgo medio (en valor absoluto)", [abs(S[(f, CH)]) for f in fams])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.45), Inches(1.65), Inches(6.1), Inches(3.6), cd)
    ch = gf.chart
    _estilo_grafico(ch, 12)
    ch.has_title = False
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.TOP
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(12)
    ch.plots[0].gap_width, ch.plots[0].overlap = 60, -5
    for sr, c in zip(ch.plots[0].series, ["navy", "gris"]):
        sr.format.fill.solid()
        sr.format.fill.fore_color.rgb = rgb(c)
    _ejes(ch, valor_visible=False, minimo=0, maximo=200, grid=False)
    etiquetas_puntos(ch, [[n(M[(f, CH)]) for f in fams], [n(abs(S[(f, CH)]), 0) for f in fams]],
                     posicion=XL_LABEL_POSITION.OUTSIDE_END)
    texto(s, 0.55, 5.3, 6.0, 0.6, "Sesgo medio: promedio de la predicción menos el precio; los tres predicen de menos en "
          "Chapinero.", size=12, color="muted")
    tarjeta(s, 0.55, 5.95, 6.0, 1.0, None, [
        "La varianza de cada familia no se midió por separado: el error total y el sesgo sí."], tipo="recordar")
    tarjeta(s, 6.85, 1.3, 5.95, 1.6, "Un árbol", [
        f"Reglas que se leen a simple vista, pero inestables: cambiar unos avisos cambia los cortes. Error "
        f"{n(M[('cart', CH)])} y sesgo {n(S[('cart', CH)], 0)}."], tipo="neutra")
    tarjeta(s, 6.85, 3.05, 5.95, 1.75, "Un bosque", [
        f"Promediar 300 árboles distintos baja la varianza: {n(M[('cart', CH)] - M[('random_forest', CH)], 1)} millones menos. "
        "Sus árboles son más profundos (5 avisos por hoja frente a 20), lo que también baja el sesgo."], tipo="neutra")
    tarjeta(s, 6.85, 4.95, 5.95, 2.0, "Árboles en secuencia", [
        f"Cada árbol ataca el error que quedó: el sesgo baja a {n(S[('lightgbm', CH)], 0)} y el error, otros "
        f"{n(M[('random_forest', CH)] - M[('lightgbm', CH)], 1)} millones. La tasa de 0,05 y la parada temprana evitan que "
        "esa corrección se vuelva memorizar."], tipo="clave")
    FIN[id(s)] = 6.95
    notas = ("Segunda razón: cómo combina los árboles. Un árbol solo es fácil de leer pero inestable: si cambian unos pocos "
             "avisos, cambian los cortes, y eso es varianza. El bosque promedia 300 árboles distintos, y el promedio "
             "cancela buena parte de esa inestabilidad: baja 15 millones. Además sus árboles son más profundos, así que "
             "también se queda menos corto. El boosting no promedia: suma árboles que atacan el error que quedó, y por eso "
             "reduce el sesgo, de menos 82 a menos 52 millones, y el error otros 15 millones. Lo que impide que esa "
             "corrección se convierta en memorizar es aprender despacio y parar a tiempo. Una aclaración honesta: medimos el "
             "error total y el sesgo; la varianza no la medimos por separado.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# Dónde fallan
# ------------------------------------------------------------------------------------------------------------------
def _subestiman(deck):
    s = _nueva(deck, "Dónde fallan: todas subestiman Chapinero, y más los avisos que predicen caros", "fal")
    texto(s, 0.55, 1.3, 6.0, 0.62, "Chapinero fuera: sesgo medio, predicción menos precio real (millones)", size=13,
          color="muted")
    ses = [S[(f, CH)] for f, _ in FAM]
    ch = barras(s, 0.45, 1.75, 6.0, 3.6, [nom for _, nom in FAM], [("Sesgo", ses, "azul")], minimo=-160,
                maximo=0, gap=40, grid=False)
    ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.HIGH
    etiquetas_puntos(ch, [[n(v, 0) for v in ses]])
    texto(s, 0.55, 5.35, 6.0, 0.35, "Barras hacia la izquierda: predice por debajo del precio real.", size=12, color="muted")
    q = R["quintil"]
    texto(s, 6.85, 1.3, 5.95, 0.62, "Sesgo medio por quintil de la propia predicción, Chapinero fuera (millones)", size=13,
          color="muted")
    sel = [("lightgbm", "LightGBM", "azul", 3.0, False), ("random_forest", "Random Forest", "gris", 2.25, False),
           ("nn_atencion", "Red con atención", "ambar", 2.25, False), ("ols", "Regresión lineal", "navy", 2.25, True)]
    cats = ["1 (más baratos)", "2", "3", "4", "5 (más caros)"]
    ser = [(nom, [x["sesgo"] for x in q[f]], c, a, g) for f, nom, c, a, g in sel]
    ch2 = _lineas_cat(s, 6.75, 1.75, 6.05, 3.6, cats, ser, minimo=-200, maximo=100, unidad=50)
    texto(s, 6.85, 5.35, 5.95, 0.35, "Quintil 5: el 20 % de avisos con la predicción más alta de cada familia.", size=12,
          color="muted")
    ce = R["corr_errores"]
    tarjeta(s, 0.55, 5.8, 6.0, 1.15, None, [
        f"Los lineales tienen el menor sesgo medio ({n(S[('ols', CH)], 0)}) porque extrapolan en línea recta, pero no el "
        "menor error: sobrevaloran los avisos que predicen baratos y subvaloran los que predicen caros."], tipo="neutra")
    tarjeta(s, 6.85, 5.8, 5.95, 1.15, None, [
        f"Los errores de las siete se mueven juntos (correlación de {n(ce['min'], 2)} a {n(ce['max'], 2)}); el "
        f"{n(R['todas_subestiman_pct'], 0)} % de los avisos lo subestiman todas: es un límite de los datos."],
        tipo="recordar")
    FIN[id(s)] = 6.95
    notas = ("Ninguna familia acierta el nivel de Chapinero: todas predicen de menos en promedio, desde menos 34 millones los "
             "lineales hasta menos 116 la red. A la derecha, el sesgo según cuánto predice cada modelo: en todas, los avisos "
             "que el modelo cree más caros son los que más subestima; los precios altos quedan comprimidos. Los lineales "
             "tienen el menor sesgo promedio, pero es un promedio engañoso: sobrevaloran los baratos y subvaloran los caros. "
             "Y los errores de las siete familias se mueven juntos: el 37 % de los avisos lo subestiman todas. Cuando todos "
             "los algoritmos fallan en los mismos avisos, el problema no es el algoritmo: es que el entrenamiento casi no "
             "tiene avisos como los de Chapinero.")
    return s, notas


def _riesgo(deck):
    s = _nueva(deck, "Para la empresa: un sesgo medio bajo no es un riesgo bajo", "fal")
    texto(s, 0.55, 1.3, 7.6, 0.62, "Chapinero fuera: porcentaje de avisos según cuánto se aleja la predicción del precio real",
          size=13, color="muted")
    orden = [f for f, _ in FAM]
    cd = CategoryChartData()
    cd.categories = [NOM[f] for f in orden]
    partes = [("Más de 10 % arriba: paga de más", "sobre_10", "rojo"), ("Dentro de ±10 %", "dentro_10", "azul"),
              ("Más de 10 % abajo: pierde el negocio", "bajo_10", "ambar")]
    for nom, k, _ in partes:
        cd.add_series(nom, [RIESGO[f][k] for f in orden])
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_STACKED_100, Inches(0.45), Inches(1.75), Inches(7.8), Inches(4.6), cd)
    ch = gf.chart
    _estilo_grafico(ch, 12)
    ch.has_title = False
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.TOP
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(12)
    ch.plots[0].gap_width, ch.plots[0].overlap = 45, 100
    for sr, (_, _, c) in zip(ch.plots[0].series, partes):
        sr.format.fill.solid()
        sr.format.fill.fore_color.rgb = rgb(c)
    _ejes(ch, valor_visible=False, invertir_cat=True, grid=False)
    etiquetas_puntos(ch, [[f"{n(RIESGO[f][k], 0)} %" for f in orden] for _, k, _ in partes], color="blanco")
    texto(s, 0.55, 6.4, 7.7, 0.55, "Precio real: el del aviso. Error relativo: predicción entre precio menos uno. Los 307 avisos "
          "de Chapinero, predichos sin haberlo visto.", size=12, color="muted")
    r = RIESGO
    tarjeta(s, 8.5, 1.3, 4.3, 1.95, "Lineales: el menor sesgo, el mayor riesgo", [
        f"Sobrevaloran más de 10 % el {n(r['ols']['sobre_10'], 0)} % ({n(r['elastic_net']['sobre_10'], 0)} % Elastic Net) "
        f"de los avisos, frente a {n(r['lightgbm']['sobre_10'], 0)} % de LightGBM."], tipo="cuidado")
    tarjeta(s, 8.5, 3.4, 4.3, 1.75, "Red: casi no paga de más, pero pierde negocios", [
        f"Sobrevalora el {n(r['nn_atencion']['sobre_10'], 0)} % y subvalora el {n(r['nn_atencion']['bajo_10'], 0)} %."],
        tipo="recordar")
    tarjeta(s, 8.5, 5.3, 4.3, 1.65, "LightGBM: el más preciso", [
        f"El {n(r['lightgbm']['dentro_10'], 0)} % queda dentro de ±10 %, la mayor proporción; aun así, uno de cada cuatro "
        "avisos queda más de 10 % arriba."], tipo="neutra")
    FIN[id(s)] = 6.95
    notas = ("Para la empresa, el error que cuesta plata es predecir de más: si el modelo sobrevalora, ofrece de más y "
             "compra caro. Cada barra reparte los 307 avisos de Chapinero en tres grupos: en rojo, predicción más de 10 % "
             "arriba del precio; en azul, dentro de más o menos 10 %; en ámbar, más de 10 % abajo. Los lineales, que tenían "
             "el menor sesgo promedio, son los más peligrosos: sobrevaloran a cuatro de cada diez avisos. La red casi no "
             "sobrevalora, pero subvalora a más de la mitad: con ella la empresa perdería muchos negocios. LightGBM es el "
             "que más avisos deja cerca del precio, pero uno de cada cuatro queda más de 10 % arriba: por eso se usa con "
             "margen y con tasación, no para ofertar sola.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# Qué movió el resultado
# ------------------------------------------------------------------------------------------------------------------
def _algoritmo_datos(deck):
    s = _nueva(deck, "Qué movió el resultado: entre familias, el algoritmo; dentro del boosting, los datos", "mov")
    texto(s, 0.55, 1.3, 5.9, 0.4, "Chapinero fuera, sin y con las 11 variables de Catastro (millones)", size=13, color="muted")
    ch = barras(s, 0.45, 1.65, 6.0, 3.3, ["LightGBM", "CatBoost"],
                [("Sin Catastro", [CAT[("lightgbm", "sin")], CAT[("catboost", "sin")]], "gris_claro"),
                 ("Con Catastro", [CAT[("lightgbm", "con")], CAT[("catboost", "con")]], "azul")],
                horizontal=False, minimo=0, maximo=200, leyenda=True, gap=60, overlap=-5, grid=False)
    etiquetas_puntos(ch, [[n(CAT[("lightgbm", "sin")]), n(CAT[("catboost", "sin")])],
                          [n(CAT[("lightgbm", "con")]), n(CAT[("catboost", "con")])]])
    texto(s, 6.85, 1.3, 6.0, 0.4, "Chapinero fuera, cada familia con las mismas variables (millones)", size=13, color="muted")
    fams = ["ols", "cart", "random_forest", "lightgbm"]
    ch2 = barras(s, 6.75, 1.65, 6.05, 3.3, [NOM[f] for f in fams], [("MAE", [M[(f, CH)] for f in fams],
                                                                    ["gris_claro", "gris_claro", "gris_claro", "azul"])],
                 horizontal=False, minimo=0, maximo=200, gap=60, grid=False)
    etiquetas_puntos(ch2, [[n(M[(f, CH)]) for f in fams]])
    tarjeta(s, 0.55, 5.15, 6.0, 1.8, "Dentro del boosting", [
        "Con los mismos pliegues, las 11 variables de Catastro bajaron el error "
        f"{n(CAT[('lightgbm', 'sin')] - CAT[('lightgbm', 'con')])} millones con LightGBM y "
        f"{n(CAT[('catboost', 'sin')] - CAT[('catboost', 'con')])} con CatBoost; cambiar de librería lo mueve "
        f"{n(CAT[('catboost', 'con')] - CAT[('lightgbm', 'con')])}. Pesan más los datos que la implementación."], tipo="clave")
    tarjeta(s, 6.85, 5.15, 5.95, 1.8, "Entre familias", [
        f"Pasar de la regresión lineal al boosting baja el error {n(M[('ols', CH)] - M[('lightgbm', CH)], 0)} millones con "
        f"las mismas variables. En Kaggle, 16 configuraciones previas de boosting variaron "
        f"{n(REJ['publico'].max() - REJ['publico'].min(), 0)} millones entre sí en la tabla pública."], tipo="neutra")
    FIN[id(s)] = 6.95
    notas = (f"¿Qué pesó más, el algoritmo o los datos? Depende de dónde se mire. Entre familias pesó el algoritmo: con las "
             f"mismas 162 variables, pasar de la regresión lineal al boosting baja el error en "
             f"{n(M[('ols', CH)] - M[('lightgbm', CH)], 0)} millones. Dentro del boosting pesaron más los datos: agregar 11 "
             f"variables de Catastro, como el valor del suelo y el estrato de la manzana, bajó el error en Chapinero en "
             f"{n(CAT[('lightgbm', 'sin')] - CAT[('lightgbm', 'con')], 0)} millones con LightGBM y en "
             f"{n(CAT[('catboost', 'sin')] - CAT[('catboost', 'con')], 0)} con CatBoost, más que los "
             f"{n(CAT[('catboost', 'con')] - CAT[('lightgbm', 'con')], 0)} que separan a las dos librerías. En Kaggle, 16 "
             f"configuraciones de boosting de una rejilla anterior, con otras variables, quedaron en una franja de "
             f"{n(REJ['publico'].max() - REJ['publico'].min(), 0)} millones: afinar movió poco.")
    return s, notas


def _complejidad(deck):
    s = _nueva(deck, "Añadir complejidad al LightGBM no ayudó; describir mejor Chapinero, sí", "mov")
    rk = RK
    filas = [("Idea", "Validación con que se probó", "Chapinero fuera", "Kaggle, tabla privada"),
             ("Variables de Catastro", (f"clústeres: {n(LGB['clusters'] - RES[('lightgbm', 'sin')].loc['clusters', 'MAE'], 1)}",
                                        "azul"),
              (n(CAT[("lightgbm", "con")] - CAT[("lightgbm", "sin")], 1), "azul"), "en todos los envíos"),
             ("Ajuste ×1,08", "kNNDM: +0,5; clústeres: −0,9", ("−2,9 (elegido mirando esta validación)", "azul"),
              (n(P02 - P01, 1), "azul")),
             ("Mezcla 40 % LightGBM y 60 % CatBoost", ("clústeres: −1,4 (pesos elegidos aquí)", "azul"),
              (n(BO["mezcla_40_60"]["dif"], 1, True), "rojo"),
              (n(PRIV["envio_03_mezcla_lightgbm_catboost"] - P01, 1, True), "rojo")),
             ("BETO ajustado como variable", ("zona cercana: −1,1", "azul"), ("+2,6", "rojo"), "no se envió"),
             ("Kriging de los residuos", (f"kNNDM: {n(rk.loc['RK en niveles', 'MAE kNNDM'] - rk.loc['sin kriging', 'MAE kNNDM'], 1, True)}",
                                         "rojo"),
              (f"en kNNDM: {n(rk.loc['RK en niveles', 'MAE Chapinero (307)'] - rk.loc['sin kriging', 'MAE Chapinero (307)'], 1, True)}",
               "rojo"), "no se envió"),
             ("Red con atención en vez de LightGBM", (f"clústeres: {n(M[('nn_atencion', CL)] - M[('lightgbm', CL)], 1, True)}", "rojo"),
              (n(M[("nn_atencion", CH)] - M[("lightgbm", CH)], 1, True), "rojo"), "no se envió")]
    y1 = _tabla(s, 0.55, 1.3, filas, [3.7, 3.4, 3.2, 1.95], size=14, minimo=0.46)
    texto(s, 0.55, y1 + 0.08, 12.25, 1.0, "Cambio en el MAE al agregar cada idea al LightGBM, o al cambiarlo por la red "
          "(millones). Azul: mejora; rojo: empeora; negro: depende de la validación. kNNDM: pliegues con la misma distancia "
          "al entrenamiento que la prueba. Zona cercana: avisos de entrenamiento a 1 km o menos de la prueba. BETO: modelo de "
          "lenguaje en español, probado antes de agregar Catastro. Kriging: interpolación espacial de los residuos.",
          size=12, color="muted")
    tarjeta(s, 0.55, 5.75, 12.25, 1.2, None, [
        "La columna clave es Chapinero fuera: solo Catastro y el ajuste la mejoran. Que un modelo pueda representar casi "
        "cualquier función no garantiza que generalice; entre modelos parecidos conviene el más simple (James y otros, "
        "2021), y lo complejo puede rendir menos cuando la prueba viene de otra distribución (Ash y Hansen, 2023)."],
        tipo="neutra")
    FIN[id(s)] = 6.95
    notas = ("Cada cifra es el cambio en el error al agregar una idea al LightGBM; negativo, en azul, es mejor. Miren la "
             "columna de Chapinero fuera: solo los datos de Catastro y el ajuste por 1,08 la mejoran, y el 1,08 se eligió "
             "mirando esa misma validación, así que su mejora es optimista; Kaggle la confirmó. Las ideas más complejas, "
             "mezclar con CatBoost, sumar BETO ajustado, interpolar los residuos con kriging o cambiar a la red con atención, "
             "mejoraban a veces la validación con que se probaron, pero empeoraban Chapinero, que es lo que pide el negocio. "
             "Lo que sí funcionó fue agregar datos que describen ese mercado.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# Compromisos y cierre
# ------------------------------------------------------------------------------------------------------------------
def _compromisos(deck):
    s = _nueva(deck, "Compromisos: qué se gana y qué se pierde con cada familia", "tra")
    filas = [("Modelo", "MAE", "Sesgo", "Más de 10 % arriba", "Se puede leer", "Fuera del rango conocido", "Qué hay que afinar")]
    info = {"lightgbm": ("con valores de Shapley", "predice como en el borde", "árboles; parar a tiempo"),
            "catboost": ("con valores de Shapley", "predice como en el borde", "árboles; parar a tiempo"),
            "random_forest": ("importancia de variables", "predice como en el borde", "casi nada"),
            "cart": ("sí: reglas a simple vista", "una hoja por zona", "profundidad"),
            "nn_atencion": ("poco", "depende de los vecinos", "épocas y arquitectura"),
            "elastic_net": ("sí: coeficientes", "extrapola en línea recta", "λ y α"),
            "ols": ("sí: coeficientes", "extrapola en línea recta", "nada")}
    for f, nom in FAM:
        filas.append((nom, n(M[(f, CH)]), n(S[(f, CH)], 0), f"{n(RIESGO[f]['sobre_10'], 0)} %") + info[f])
    y1 = _tabla(s, 0.55, 1.3, filas, [1.85, 0.85, 0.8, 1.25, 2.4, 2.55, 2.55], size=13, minimo=0.4)
    texto(s, 0.55, y1 + 0.05, 12.25, 0.4, "Chapinero fuera. MAE y sesgo en millones; sesgo: predicción menos precio real.",
          size=12, color="muted")
    tarjeta(s, 0.55, y1 + 0.55, 12.25, 6.95 - y1 - 0.55, "Para el negocio", [
        "El boosting da el menor error y la mayor proporción de avisos dentro de ±10 %, y se explica aviso por "
        "aviso con valores de Shapley; a cambio, como todo árbol, no extrapola. Los lineales se leen fácil, pero su sesgo "
        "bajo esconde el mayor riesgo. La red depende de tener vecinos parecidos."], tipo="neutra")
    FIN[id(s)] = 6.95
    notas = ("Cada familia tiene un costo. El boosting da el menor error y se puede explicar aviso por aviso con valores de "
             "Shapley, pero, como todos los árboles, no extrapola: fuera de lo que conoce predice como en el borde. Random "
             "Forest casi no hay que afinarlo, pero queda 15 millones detrás. El árbol solo se lee a simple vista, a costa "
             "de 30 millones más de error. La red depende de tener vecinos parecidos y sobrevalora poco, pero pierde "
             "negocios. Los lineales se leen y extrapolan, pero son los que más sobrevaloran. Para esta empresa, que pierde "
             "plata cuando paga de más, el boosting es el mejor compromiso.")
    return s, notas


def _conclusion(deck):
    s = _nueva(deck, "Gana el boosting, pero ninguna familia corrige el nivel de Chapinero", "con", oscura=True)
    bloques = [("Por qué ganó el boosting", "tinte", "navy",
                "Capta curvas e interacciones de tamaño y ubicación que los lineales no pueden, y corrige en secuencia el "
                "sesgo que deja un árbol solo. Ganó en las tres validaciones, y la ventaja es real: ningún intervalo toca "
                "el cero."),
               ("Qué enseña la validación", "tinte", "navy",
                "La CV aleatoria elige el mismo ganador, pero promete casi la mitad del error y exagera a los modelos que "
                "se apoyan en lo cercano. Dejar Chapinero fuera anticipa el nivel de Kaggle."),
               ("Lo que ninguna familia resuelve", "ambar_claro", "ambar",
                "Todas subestiman Chapinero y fallan en los mismos avisos: su entorno queda fuera de lo que el entrenamiento "
                "conoce. Lo movieron los datos (Catastro), no un algoritmo más complejo."),
               ("Para la empresa", "tinte", "navy",
                "Usar LightGBM para ordenar avisos y como punto de partida de la negociación, con margen y tasación: aun "
                "así, uno de cada cuatro avisos queda más de 10 % arriba.")]
    for i, (tit, fondo, borde, t) in enumerate(bloques):
        x0, y0 = 0.55 + (i % 2) * 6.2, 1.4 + (i // 2) * 2.8
        forma(s, MSO_SHAPE.ROUNDED_RECTANGLE, x0, y0, 6.0, 2.6, relleno=fondo, radio=0.05)
        texto(s, x0 + 0.25, y0 + 0.15, 5.5, 0.45, tit, size=19, color=borde)
        texto(s, x0 + 0.25, y0 + 0.7, 5.5, 1.8, t, size=17, color="tinta")
    FIN[id(s)] = 6.95
    notas = ("En resumen. El boosting gana porque capta formas no lineales y la interacción entre tamaño y ubicación, y "
             "porque corrige paso a paso el error que deja un árbol; la ventaja es real. La validación aleatoria elige el "
             "mismo ganador, pero promete la mitad del error que vimos en Kaggle; dejar Chapinero fuera es la que anticipa "
             "el nivel. Ninguna familia resuelve que Chapinero esté fuera de lo que el entrenamiento conoce: todas lo "
             "subestiman, y lo que más ayudó fueron datos nuevos, no algoritmos más complejos. Para la empresa: el modelo "
             "sirve para filtrar y negociar, no para comprar solo.")
    return s, notas


def _fuentes(deck):
    s = _nueva(deck, "Fuentes", "fue")
    refs = ["Datos: Properati (avisos de la competencia de Kaggle); Catastro de Bogotá, UAECD; OpenStreetMap; Google Earth "
            "Engine (SRTM, Sentinel-2, VIIRS y GHSL).",
            "Ash, E. y Hansen, S. (2023). Text algorithms in economics. Annual Review of Economics, 15, 659-688.",
            "Bird, S., Klein, E. y Loper, E. (2009). Natural Language Processing with Python. O'Reilly.",
            "Blei, D. M., Ng, A. Y. y Jordan, M. I. (2003). Latent Dirichlet allocation. Journal of Machine Learning Research, 3, "
            "993-1022.",
            "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.",
            "Breiman, L., Friedman, J. H., Olshen, R. A. y Stone, C. J. (1984). Classification and Regression Trees. Wadsworth.",
            "Cañete, J. y otros (2020). Spanish pre-trained BERT model and evaluation data. PML4DC, ICLR.",
            "Departamento Nacional de Planeación, Unidad de Científicos de Datos (2020). ConTexto: librería de procesamiento de "
            "texto en español.",
            "Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of Statistics, 29(5), "
            "1189-1232.",
            "Gentzkow, M., Kelly, B. y Taddy, M. (2019). Text as data. Journal of Economic Literature, 57(3), 535-574.",
            "Hastie, T., Tibshirani, R. y Friedman, J. (2009). The Elements of Statistical Learning, segunda edición. Springer.",
            "James, G., Witten, D., Hastie, T. y Tibshirani, R. (2021). An Introduction to Statistical Learning, segunda edición. "
            "Springer.",
            "Ke, G. y otros (2017). LightGBM: a highly efficient gradient boosting decision tree. NeurIPS 30.",
            "Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. Annals of Statistics, "
            "17(3), 1217-1241.",
            "Lundberg, S. M. y Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS 30.",
            "Prokhorenkova, L. y otros (2018). CatBoost: unbiased boosting with categorical features. NeurIPS 31.",
            "Roberts, D. R. y otros (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or "
            "phylogenetic structure. Ecography, 40(8), 913-929.",
            "Rosen, S. (1974). Hedonic prices and implicit markets. Journal of Political Economy, 82(1), 34-55.",
            "Stanford GSB Insights (2021). Flip Flop: Why Zillow's algorithmic home buying venture imploded.",
            "Vaswani, A. y otros (2017). Attention is all you need. NeurIPS 30.",
            "Viana, D. y Barbosa, L. (2021). Attention-based spatial interpolation for house price prediction. ACM SIGSPATIAL.",
            "Zou, H. y Hastie, T. (2005). Regularization and variable selection via the elastic net. Journal of the Royal "
            "Statistical Society B, 67(2), 301-320."]
    mitad = 11
    texto(s, 0.55, 1.3, 6.0, 5.65, refs[:mitad], size=12, color="tinta", espacio=4)
    texto(s, 6.8, 1.3, 6.0, 5.65, refs[mitad:], size=12, color="tinta", espacio=4)
    FIN[id(s)] = 6.95
    notas = "Referencias citadas en la presentación, incluidas las de la parte de datos."
    return s, notas


def compare_cuerpo(deck):
    """Cuerpo de compare (de la lámina 26 al final, sin apéndice). Devuelve [(lámina, notas)] y pone las notas."""
    out = [f(deck) for f in (_criterios, _como_predice, _lineales, _arboles, _boosting, _red, _hiperparametros, _reglas,
                             _validaciones, _resultado, _flexibilidad, _diferencias, _estandar_espacial, _kaggle, _forma,
                             _promediar_corregir, _subestiman, _riesgo, _algoritmo_datos, _complejidad, _compromisos,
                             _conclusion, _fuentes)]
    for s, notas in out:
        deck.notas(s, notas)
    return out
