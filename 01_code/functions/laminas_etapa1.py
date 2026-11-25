"""Etapa 1 reescrita para un público no experto: láminas del problema, la hoja de ruta y los datos (antes 2 a 23).

Orden: problema de negocio; hoja de ruta; mapa y precio (define entrenamiento y prueba); fuente 1a y 1b (avisos); apertura
de las técnicas de texto; pasos y modelos del texto; un aviso real regla por regla; BETO; listas de palabras;
fuente 2a y 2b (indicadores y cifras); TF-IDF; LSA; LDA; ridge; fuente 2c (tópicos, componentes y puntaje, como cierre del
texto); fuentes 3a a 3c (OpenStreetMap); fuente 4 (satélite); fuente 5 (Catastro); 162 variables y precios hedónicos.

Atiende las revisiones de claridad y de cifras de la Etapa 1. Solo lee archivos de
00_data/processed/laminas/ y 02_outputs/figures/laminas/; no cambia datos ni modelos. No modifica laminas_comunes.py ni ppt_lib.py: importa de ellos.

Uso: from laminas_etapa1 import etapa1; laminas = etapa1(deck)  # lista de (lámina, notas)
"""
import json
import math

import pandas as pd
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

from functions.laminas_comunes import (D, IMG, TOPICOS, _area_manual, _arista, _caja_dag, _caja_texto, _negativo, _pct,
                             _saltar_etiquetas, n)
from functions.ppt_lib import _lineas, barras, chips, etiquetas_puntos, forma, imagen, linea, lineas_xy, rgb, tabla, tarjeta, texto

ETQ_PROBLEMA, ETQ_RUTA, ETQ_DATOS = "El problema de negocio", "Hoja de ruta", "Etapa 1. Datos"
SEC_PROBLEMA, SEC_RUTA, SEC_DATOS = "El problema de negocio", "Hoja de ruta", "Datos"

# Variables por fuente (iguales en la hoja de ruta, la barra de cada tabla y la lámina de cierre)
FUENTES = [("Avisos", 21), ("Texto", 84), ("Mapa", 39), ("Satélite", 7), ("Catastro", 11)]

# Términos del ridge en palabras completas (las raíces vienen de laminas/ppt_palabras_ridge.csv)
LEG_RIDGE = {"3 parqueader": "3 parqueaderos", "cas": "casa", "ascensor priv": "ascensor privado",
             "4 parqueader": "4 parqueaderos", "carolin": "Carolina", "aprovech excelent": "aproveche excelente",
             "2 parqueader": "2 parqueaderos", "chic": "Chicó", "apartaestudi": "apartaestudio",
             "parqueader": "parqueadero", "1 parqueader": "1 parqueadero", "habit": "habitación", "cedrit": "Cedritos",
             "sal comedor": "sala comedor", "1": "el número 1", "ban": "baño"}
FIN = {}  # y final del contenido principal de cada lámina (control de calidad)
NOMBRE_TEMA = {**TOPICOS, 1: "Conjunto con zonas comunes", 3: "Avisos del portal Houm"}
TILDES = {"bano": "baño", "salon": "salón", "barbara": "bárbara", "anos": "años", "contactanos": "contáctanos"}


# ------------------------------------------------------------------------------------------------------------------
# Utilidades de diseño
# ------------------------------------------------------------------------------------------------------------------
def _nueva(deck, titulo, etiqueta=ETQ_DATOS, seccion=SEC_DATOS):
    """Lámina nueva con el título en una sola línea: 26 puntos si cabe; si no, 24 o 22."""
    tam = next((t for t in (26, 24, 22) if _lineas(titulo, 12.1, t) == 1), 22)
    return deck.nueva(titulo, etiqueta=etiqueta, seccion=seccion, size_titulo=tam)


def _lineas_celda(t, ancho, size):
    """Renglones de un texto en una celda, contando también las palabras más largas que el ancho (se parten)."""
    from functions.ppt_lib import _medidor, limpia
    f = _medidor()
    if not f:
        return _lineas(t, ancho, size)
    esc = size / 72 / 200
    esp = f.getlength(" ") * esc
    n_lin, actual = 1, 0.0
    for palabra in limpia(t).split(" "):
        w_ = f.getlength(palabra) * esc
        if w_ > ancho:  # la palabra se parte en varios renglones
            n_lin += (0 if actual == 0 else 1) + int(w_ // ancho)
            actual = w_ % ancho
        elif actual == 0:
            actual = w_
        elif actual + esp + w_ <= ancho:
            actual += esp + w_
        else:
            n_lin, actual = n_lin + 1, w_
    return n_lin


def _alto_celda(t, ancho, size):
    return _lineas_celda(str(t), max(ancho - 0.17, 0.3) * 0.98, size) * 1.21 * size / 72 + 0.075


def _tabla(s, x, y, filas, anchos, size=12, grupos=None, total=False, colores=None, minimo=0.27, alto_encabezado=None):
    """Tabla con altos de fila calculados para que nada se encime.

    grupos: lista de (fila_inicial, fila_final) cuya primera columna va unida. total: la última fila es el total (texto
    de la columna 1 unido hasta la última). colores: {(fila, columna): color}. Devuelve la y final."""
    w = sum(anchos)
    gf = tabla(s, x, y, w, filas, anchos, size=size, alto_fila=minimo)
    t = gf.table
    nc = len(anchos)
    en_grupo = {}
    for a_, b_ in (grupos or []):
        if b_ > a_:
            for r_ in range(a_, b_ + 1):
                en_grupo[r_] = (a_, b_)
    altos = []
    for i, fila in enumerate(filas):
        cols = range(nc)
        if i in en_grupo:
            cols = range(1, nc)
        if total and i == len(filas) - 1:
            h = max(_alto_celda(fila[0], anchos[0], size), _alto_celda(fila[1], sum(anchos[1:]), size))
        else:
            h = max(_alto_celda(fila[j], anchos[j], size) for j in cols)
        altos.append(max(minimo, h))
    if alto_encabezado:
        altos[0] = max(altos[0], alto_encabezado)
    # La primera columna unida debe caber en la suma de su grupo
    for a_, b_ in (grupos or []):
        if b_ > a_:
            falta = _alto_celda(filas[a_][0], anchos[0], size) - sum(altos[a_:b_ + 1])
            if falta > 0:
                altos[b_] += falta
    for i, h in enumerate(altos):
        t.rows[i].height = Inches(h)
    # Fondo por grupo y celdas unidas
    for g_, (a_, b_) in enumerate(grupos or []):
        fondo = rgb("FFFFFF" if g_ % 2 else "F7F9FB")
        for r_ in range(a_, b_ + 1):
            for c_ in range(nc):
                t.cell(r_, c_).fill.solid()
                t.cell(r_, c_).fill.fore_color.rgb = fondo
        if b_ > a_:
            t.cell(a_, 0).merge(t.cell(b_, 0))
    if total:
        r = len(filas) - 1
        for c_ in range(nc):
            t.cell(r, c_).fill.solid()
            t.cell(r, c_).fill.fore_color.rgb = rgb("E8EEF5")
            for par in t.cell(r, c_).text_frame.paragraphs:
                for run in par.runs:
                    run.font.color.rgb = rgb("navy")
        t.cell(r, 1).merge(t.cell(r, nc - 1))
    for (r_, c_), col in (colores or {}).items():
        for run in t.cell(r_, c_).text_frame.paragraphs[0].runs:
            run.font.color.rgb = rgb(col)
    return y + sum(altos)


def _barra(s, k):
    """Barra de las 162 variables por fuente, con todos los tramos rotulados; resalta la fuente k (0 a 4)."""
    texto(s, 0.55, 6.6, 2.05, 0.36, "Variables por fuente:", size=12, color="muted", anchor="middle")
    x0, ancho = 2.6, 10.2
    fijos = {3: 1.1, 4: 1.25}
    resto = ancho - sum(fijos.values())
    suma = sum(c for i, (_, c) in enumerate(FUENTES) if i not in fijos)
    x = x0
    for i, (nombre, c) in enumerate(FUENTES):
        w = fijos.get(i, resto * c / suma)
        relleno = "azul" if i == k else ("gris" if i < k else "gris_claro")
        forma(s, MSO_SHAPE.RECTANGLE, x, 6.6, w - 0.04, 0.36, relleno=relleno)
        texto(s, x, 6.6, w - 0.04, 0.36, f"{nombre} {c}", size=12, color="blanco" if i <= k else "tinta", align="center",
              anchor="middle")
        x += w


def _cuadro(s, x, y, w, h, titulo, detalle, modelo=True):
    """Caja de salida (azul claro) o de paso (gris), con título y una o dos líneas de detalle."""
    caja = forma(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, relleno="azul_claro" if modelo else "tinte",
                 linea="azul_suave" if modelo else "gris_claro", radio=0.12)
    texto(s, x + 0.12, y + 0.03, w - 0.24, 0.28, titulo, size=13, color="azul" if modelo else "navy")
    texto(s, x + 0.12, y + 0.3, w - 0.24, h - 0.32, detalle, size=12, color="tinta")
    return caja


def _tres_tarjetas(s, que_es, como, datos, h=2.45, titulo_como="Cómo se construye", maximo=14):
    for x, (titulo, cuerpo, tipo) in zip((0.55, 4.7, 8.85), (("Qué es", que_es, "definicion"), (titulo_como, como, "neutra"),
                                                            ("Cómo queda en los datos", datos, "clave"))):
        tarjeta(s, x, 1.33, 3.95, h, titulo, cuerpo, tipo=tipo, maximo=maximo)


def _si(p):
    """Proporción de una variable de sí o no como porcentaje de avisos con sí."""
    return _pct(100 * p)


# ------------------------------------------------------------------------------------------------------------------
# 1. El problema de negocio
# ------------------------------------------------------------------------------------------------------------------
def _problema(deck):
    pron = json.loads((D / "ppt_varios2.json").read_text(encoding="utf-8"))["pronostico"]
    s = _nueva(deck, "Para la empresa, pagar de más cuesta más que dejar de comprar", ETQ_PROBLEMA, SEC_PROBLEMA)
    empresa = _caja_texto(s, 0.55, 1.45, 3.35, 3.35, "La empresa", [
        "Quiere comprar la mayor cantidad de viviendas en Chapinero, en Bogotá, gastando lo menos posible.",
        "Decide cuánto ofrecer con el precio que calcula un modelo: un programa que aprende de avisos con precio y "
        "predice el de los que no lo traen."])
    mas = _caja_texto(s, 4.45, 1.45, 4.35, 2.45, "Predecir de más", [
        "Ofrece de más y compra caro: pierde plata al revender y paga los costos de tener la vivienda mientras la vende.",
        "Zillow, una empresa de Estados Unidos que compraba casas guiada por un modelo, cerró ese negocio con pérdidas de "
        "cerca de 500 millones de dólares."], relleno="rojo_claro", borde="rojo", color_titulo="rojo")
    menos = _caja_texto(s, 4.45, 4.05, 4.35, 1.2, "Predecir de menos", [
        "La oferta queda corta y el dueño no vende: se pierde un negocio, pero no plata."],
        relleno="azul_claro", borde="azul", color_titulo="azul")
    meta = _caja_texto(s, 9.35, 1.45, 3.45, 3.8, "Qué debe lograr el modelo", [
        f"Predecir el precio de 10.286 avisos que no lo traen, el {n(pron['pct_localidad_chapinero'], 0)} % en la "
        "localidad de Chapinero, aprendiendo de 38.644 avisos de Bogotá que sí lo traen. Solo 307 de esos están en "
        "Chapinero.",
        "Y mostrar hacia dónde se equivoca, porque predecir de más es el error caro."])
    for a, b in [(empresa, mas), (empresa, menos), (mas, meta), (menos, meta)]:
        _arista(s, a, b)
    texto(s, 0.55, 5.5, 12.25, 1.45, [
        "Avisos: anuncios de venta de vivienda publicados en Properati, un portal de internet. Todos los precios van en "
        "millones de pesos colombianos.",
        "Kaggle, la plataforma de la competencia del curso, califica con el error absoluto medio: cuánto se equivoca el "
        "modelo en promedio, en millones, sin importar si es hacia arriba o hacia abajo. Trata igual los dos errores: es un "
        "resultado más, no el objetivo."], size=14, color="tinta", espacio=8)
    notas = ("Empecemos por el problema. Una empresa quiere comprar muchas viviendas en Chapinero gastando lo menos posible, y "
             "decide cuánto ofrecer con el precio que calcula un modelo. Un modelo, para esta charla, es un programa que "
             "aprende de avisos que sí traen precio y predice el de los que no. Los dos errores no cuestan lo mismo. Si el "
             "modelo predice de más, la empresa compra caro: pierde plata al revender y además paga los costos de tener la "
             "vivienda mientras la vende. Eso le pasó a Zillow, que compraba casas guiada por un modelo y cerró ese negocio "
             "con pérdidas de cerca de 500 millones de dólares. Si predice de menos, el dueño no vende y se pierde un "
             "negocio, pero no plata. El reto es difícil: hay que predecir 10.286 avisos, casi todos de Chapinero, "
             "aprendiendo de 38.644 avisos de Bogotá de los que solo 307 están en Chapinero. Kaggle, la plataforma de la "
             "competencia, mide el error absoluto medio: cuántos millones se equivoca en promedio el modelo, hacia arriba o "
             "hacia abajo. Como trata igual los dos errores, para nosotros es un resultado más, no el objetivo.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# 2. Hoja de ruta
# ------------------------------------------------------------------------------------------------------------------
def _ruta(deck):
    s = _nueva(deck, "De los avisos al pronóstico: el camino completo", ETQ_RUTA, SEC_RUTA)
    cols = [(0.55, 2.05), (2.95, 2.35), (5.65, 1.55), (7.55, 2.0), (9.9, 2.9)]
    for nombre, (x, w) in zip(["Fuentes", "Variables", "Base única", "Modelos y validación", "Resultado"], cols):
        texto(s, x - 0.1, 1.3, w + 0.2, 0.34, nombre, size=13, color="muted", align="center")
    X, Wc = (lambda c: cols[c][0]), (lambda c: cols[c][1])
    fuentes = [_caja_dag(s, X(0), 1.72 + i * 0.96, Wc(0), 0.84, t, d) for i, (t, d) in enumerate([
        ("Avisos de Properati", "38.644 de entrenamiento y 10.286 de prueba"),
        ("Texto del aviso", "título y descripción"),
        ("OpenStreetMap", "mapa abierto: comercio, transporte, parques"),
        ("Imágenes de satélite", "terreno, vegetación y luces de noche"),
        ("Catastro de Bogotá", "registro oficial: suelo, avalúo, estrato")])]
    variables = [_caja_dag(s, X(1), y, Wc(1), 1.0, t, d) for y, (t, d) in zip([1.72, 2.92, 4.12, 5.4], [
        ("Avisos (21)", "vivienda, fecha, ubicación y avisos vecinos; el texto completa área y baños"),
        ("Texto (84)", "palabras clave, cifras escritas, temas y un puntaje de precio"),
        ("Mapa y satélite (46)", "lugares y distancias alrededor del aviso; terreno y luces"),
        ("Catastro (11)", "suelo, avalúo, pisos y estrato de la manzana y a 300 m")])]
    union = _caja_dag(s, X(2), 3.4, Wc(2), 1.3, "162 variables", "las mismas para todos los modelos",
                      relleno="azul_claro", borde="azul", color_titulo="azul")
    familias = _caja_dag(s, X(3), 2.45, Wc(3), 1.15, "Seis tipos de modelo", "desde una regresión lineal hasta árboles "
                         "y una red neuronal")
    validacion = _caja_dag(s, X(3), 4.5, Wc(3), 1.15, "Seis formas de probar", "desde apartar avisos al azar hasta "
                           "esconder todo Chapinero")
    yE, hE = [1.72, 2.9, 4.08, 5.26], 1.05
    gana = _caja_dag(s, X(4), yE[0], Wc(4), hE, "Gana LightGBM (árboles)", "al predecir Chapinero sin "
                     "haberlo visto, yerra en promedio 143,8 millones por aviso")
    final = _caja_dag(s, X(4), yE[1], Wc(4), hE, "Modelo final", "aprende de los 38.644 avisos (5 repeticiones); "
                      "ajustes: estira 8 % (×1,08) y gemelos de texto")
    prono = _caja_dag(s, X(4), yE[2], Wc(4), hE, "Pronóstico", "precio de los 10.286 avisos de prueba; en Kaggle, "
                      "error de 150,8 millones: segundo de seis equipos", relleno="verde_claro", borde="verde",
                      color_titulo="verde")
    uso = _caja_dag(s, X(4), yE[3], Wc(4), hE, "Uso para la empresa", "ordena bien los avisos de barato a caro; ofertar "
                    "en automático pierde 14,8 millones por oferta")
    for o, d in [(0, 0), (1, 0), (1, 1), (2, 2), (3, 2), (4, 3)]:
        _arista(s, fuentes[o], variables[d])
    for v in variables:
        _arista(s, v, union)
    for d in (familias, validacion):
        _arista(s, union, d)
        _arista(s, d, gana)
    for a, b in [(gana, final), (final, prono), (prono, uso)]:
        _arista(s, a, b, 2, 0, recta=True)
    texto(s, 0.55, 6.42, 12.25, 0.6, "Variable: cada dato que describe un aviso, como el área o la distancia a un parque; "
          "entre paréntesis, cuántas hay. Entrenamiento: avisos con precio, de los que el modelo aprende. Prueba: avisos sin "
          "precio, los que hay que predecir. Se lee de izquierda a derecha.", size=12, color="navy")
    notas = ("Este es el camino completo, de izquierda a derecha. Antes, tres palabras que vamos a usar todo el tiempo. Una "
             "variable es cada dato que describe un aviso, como el área o la distancia a un parque. El entrenamiento son los "
             "avisos que traen precio, de los que el modelo aprende; la prueba son los que no lo traen y hay que predecir. "
             "Partimos de cinco fuentes: los avisos de Properati, su texto, OpenStreetMap, que es un mapa abierto de "
             "internet, imágenes de satélite y el Catastro de Bogotá. Con ellas armamos 162 variables, contadas por fuente: "
             "21 de los avisos, 84 del texto, 46 del mapa y el satélite, y 11 de Catastro. El texto también ayuda al bloque "
             "de avisos: completa el área y los baños que faltan y aporta parqueaderos, piso, estrato y antigüedad. Las 162 "
             "variables son las mismas para todos los modelos. Comparamos seis tipos de modelo y los probamos de seis formas, "
             "desde apartar avisos al azar hasta esconder todo Chapinero. Ganó LightGBM, un modelo de árboles de decisión: al "
             "predecir Chapinero sin haberlo visto se equivoca en promedio 143,8 millones por aviso. El modelo final aprende "
             "de los 38.644 avisos y promedia cinco repeticiones, con dos ajustes que explicamos más adelante: estira sus "
             "precios un 8 % alrededor del promedio, el ×1,08, y en los 78 avisos de prueba cuya descripción repite la "
             "de un aviso de entrenamiento, los gemelos de texto, usa sobre todo el precio de ese gemelo. En Kaggle quedó con un error de 150,8 millones, segundo de "
             "seis equipos. Para la empresa sirve para ordenar avisos de barato a caro, pero ofertar en automático pierde "
             "14,8 millones por oferta. El resto de la presentación recorre este camino.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# 3. Entrenamiento y prueba: mapa e histograma del precio
# ------------------------------------------------------------------------------------------------------------------
def _mapa_precio(deck):
    import numpy as np
    desc = json.loads((D / "ppt_descripcion.json").read_text(encoding="utf-8"))
    pron = json.loads((D / "ppt_varios2.json").read_text(encoding="utf-8"))["pronostico"]
    ent = pd.read_parquet(D / "entrenamiento_mapa.parquet")
    precio = ent["price"].to_numpy() / 1e6
    bordes = np.arange(300, 1651, 50)
    cuenta, _ = np.histogram(precio, bins=bordes)
    med, prom = float(np.median(precio)), float(precio.mean())
    s = _nueva(deck, "Entrenamos con Bogotá, casi sin Chapinero, para predecir Chapinero")
    imagen(s, IMG / "mapa_entrenamiento_prueba.png", 0.55, 1.28, w=5.0,
           alt="Mapa de Bogotá con los avisos de entrenamiento en gris y los de prueba en naranja, casi todos en Chapinero")
    texto(s, 0.55, 6.7, 5.1, 0.3, "Borde negro: localidad de Chapinero.", size=12, color="muted")
    stats = [(n(desc["n_train"], 0), "avisos de entrenamiento (2019 a 2021): traen precio y el modelo aprende de ellos; "
                                     "solo 307 en Chapinero"),
             (n(desc["n_test"], 0), f"avisos de prueba: sin precio, son los que hay que predecir; el "
                                    f"{n(pron['pct_localidad_chapinero'], 0)} % en la localidad de Chapinero"),
             ("300 a 1.650", "millones de pesos: precio más bajo y más alto del entrenamiento"),
             (f"{n(desc['pct_apto_test'], 0)} %", f"de la prueba son apartamentos, frente a {n(desc['pct_apto_train'], 0)} % "
                                                  "del entrenamiento")]
    for i, (v, e) in enumerate(stats):
        x0, y0 = 5.9 + (i % 2) * 3.5, 1.3 + (i // 2) * 1.32
        texto(s, x0, y0, 3.4, 0.55, v, size=28, color="azul")
        texto(s, x0, y0 + 0.52, 3.35, 0.78, e, size=12, color="muted")
    texto(s, 5.9, 3.98, 6.9, 0.32, f"Precio de los {n(len(precio), 0)} avisos de entrenamiento (millones de pesos)", size=13,
          color="navy")
    texto(s, 5.9, 4.28, 6.9, 0.5, f"Línea continua: mediana, {n(med, 0)} millones: la mitad de los avisos cuesta menos. "
          f"Línea punteada: promedio, {n(prom, 0)} millones.", size=12, color="muted")
    gx, gy, gw, gh = 5.8, 4.72, 7.05, 2.28
    ch = barras(s, gx, gy, gw, gh, [n(b, 0) for b in bordes[:-1]], [("Entrenamiento", [int(c) for c in cuenta], "gris")],
                horizontal=False, fmt="#,##0", etiquetas=False, gap=8, valor_visible=True, titulo_valor="Avisos",
                maximo=6000, unidad=2000, titulo_cat="Precio del aviso; cada barra agrupa 50 millones")
    _saltar_etiquetas(ch, 4)
    lx, ly, lw, lh = 0.12, 0.05, 0.85, 0.62
    _area_manual(ch, lx, ly, lw, lh)
    x0, y0, ancho, alto = gx + lx * gw, gy + ly * gh, lw * gw, lh * gh
    for valor, nombre, color, guiones, lado in ((med, "Mediana", "navy", False, "izq"),
                                                (prom, "Promedio", "azul", True, "der")):
        xv = x0 + (valor - bordes[0]) / (bordes[-1] - bordes[0]) * ancho
        ln = linea(s, xv, y0, xv, y0 + alto, color=color, ancho=2.25)
        if guiones:
            ln.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        if lado == "izq":
            texto(s, xv - 1.55, y0 - 0.02, 1.5, 0.3, f"{nombre}: {n(valor, 0)}", size=12, color="tinta", align="right")
        else:
            texto(s, xv + 0.05, y0 - 0.02, 1.6, 0.3, f"{nombre}: {n(valor, 0)}", size=12, color="tinta")
    notas = ("Los datos son avisos de Properati de 2019 a 2021, en dos grupos. En gris, los 38.644 de entrenamiento: traen el "
             "precio y el modelo aprende de ellos; solo 307 están en Chapinero, cuyo borde es la línea negra. En naranja, los "
             "10.286 de prueba: no traen precio, son los que hay que predecir, y casi todos están en Chapinero. Cada punto "
             "está en su coordenada exacta. El histograma muestra cómo se reparte el precio del entrenamiento, en millones de "
             "pesos: va de 300 a 1.650, se concentra entre 300 y 600 y tiene una cola larga hacia los precios altos. Dos "
             f"medidas del centro: la mediana, {n(med, 0)} millones, es el precio que deja la mitad de los avisos por debajo; "
             f"el promedio, {n(prom, 0)}, queda a la derecha porque los avisos caros lo jalan hacia arriba. Y la prueba es "
             "casi toda de apartamentos.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# 4 y 5. Fuente 1: avisos de Properati
# ------------------------------------------------------------------------------------------------------------------
ENC6 = ("Dato original (% con dato: entrenamiento y prueba)", "Variable", "Cómo se construye", "Con dato, entrenamiento",
        "Con dato, prueba", "Promedio o % con sí")
ANCHOS6 = [2.6, 2.4, 3.9, 1.25, 1.05, 1.05]


def _filas_agrupadas(grupos):
    filas, tramos = [], []
    for col, vars_ in grupos:
        a_ = len(filas) + 1
        for j, f in enumerate(vars_):
            filas.append((col if j == 0 else "",) + tuple(f))
        tramos.append((a_, len(filas)))
    return filas, tramos


def _lamina_fuente(deck, titulo, base, grupos, pie, k, en_tabla, acum, encabezado=None, anchos=None, size_base=13,
                   alto_base=0.3, minimo=0.27):
    encabezado, anchos = encabezado or ENC6, anchos or ANCHOS6
    s = _nueva(deck, titulo)
    texto(s, 0.55, 1.22, 12.25, alto_base, base, size=size_base, color="navy")
    filas, tramos = _filas_agrupadas(grupos)
    nc = len(encabezado)
    total = ("Total", f"{en_tabla} variables en esta tabla; acumulado: {acum} de 162") + ("",) * (nc - 2)
    if acum == 10:
        total = ("Total", "10 variables en esta tabla; acumulado: 10 de 162. El precio no se cuenta: es lo que se predice")\
            + ("",) * (nc - 2)
    filas = [encabezado] + filas + [total]
    y_tabla = 1.25 + alto_base + 0.08
    y_fin = _tabla(s, 0.55, y_tabla, filas, anchos, grupos=tramos, total=True, minimo=minimo)
    texto(s, 0.55, max(y_fin + 0.04, 6.02), 12.25, 0.5, pie, size=12, color="muted")
    _barra(s, k)
    FIN[id(s)] = y_fin  # para el control de calidad
    return s


def _fuente1(deck):
    R = json.load(open(D / "ppt_fuentes.json", encoding="utf-8"))
    est, eg, bg = R["avisos_variables"], R["avisos_variables_grupo"], R["avisos_brutas_no_nulos_grupo"]

    def par(c):
        e_, p_ = _pct(bg[c]["entrenamiento"]), _pct(bg[c]["prueba"])
        return e_ if e_ == p_ else f"{e_} y {p_}"

    def fv(c, nombre, como, d=1, si=False, prom=None):
        p = est[c]["promedio"]
        val = prom if prom is not None else (_si(p) if si else n(p, d))
        return nombre, como, _pct(eg[c]["entrenamiento"]), _pct(eg[c]["prueba"]), val

    titulo_desc = _pct(min(bg["title"]["entrenamiento"], bg["description"]["entrenamiento"]))
    base = ("Properati es un portal de vivienda en internet. Trae 16 datos por aviso; en los 10.286 de prueba el precio "
            "viene vacío.")
    pie_comun = ("Con dato: % de avisos que traen la variable. Promedio de los 48.930 avisos; el del precio, solo de los "
                 "38.644 de entrenamiento. Las variables de sí o no valen 1 o 0: su promedio es el % de avisos con sí.")
    pie_b = ("Con dato: % de avisos que traen la variable; promedio de los 48.930 avisos. Mediana: el valor que deja la "
             "mitad de los avisos por debajo.")
    lam = []
    grupos_a = [
        (f"precio ({par('price')})", [("Precio (millones de pesos)", "el precio en millones; es lo que se predice",
                                       "100 %", "0 %", n(R["precio_promedio_entrenamiento"], 1))]),
        (f"tipo de inmueble ({par('property_type')})",
         [fv("es_apartamento", "Es apartamento", "sí si es apartamento; no si es casa", si=True)]),
        (f"alcobas ({par('bedrooms')})", [fv("bedrooms", "Alcobas", "tal como viene")]),
        (f"ambientes ({par('rooms')})", [fv("rooms", "Ambientes", "tal como viene; falta en casi la mitad")]),
        (f"área cubierta ({par('surface_covered')}) y área total ({par('surface_total')})", [
            fv("area_imp", "Área (m²)", "la cubierta; si falta, la total; si tampoco, la del texto. Solo de 15 a 1.000 m²"),
            fv("log_area", "Logaritmo del área", f"el área en escala comprimida; "
                                                 f"{n(est['log_area']['promedio'], 2)} equivale a unos "
                                                 f"{n(math.exp(est['log_area']['promedio']), 0)} m²", 2),
            fv("area_desde_texto", "Área tomada del texto", "sí si el área salió del texto", si=True),
            fv("area_falta", "Área faltante", "sí si no hay área ni en los datos ni en el texto", si=True),
            fv("area_inconsistente", "Área inconsistente", "sí si el área cubierta supera la total", si=True)]),
        (f"baños ({par('bathrooms')})", [
            fv("banos_imp", "Baños", "el dato; si falta, la cifra del texto"),
            fv("banos_desde_texto", "Baños tomados del texto", "sí si los baños salieron del texto", si=True)])]
    s = _lamina_fuente(deck, "Fuente 1a. Avisos de Properati: el precio que se predice y la vivienda", base, grupos_a,
                       pie_comun + " Logaritmo: escala comprimida; pasar de 50 a 100 m² pesa igual que pasar de 100 a 200.",
                       0, 10, 10)
    lam.append((s, "La primera fuente son los avisos de Properati, un portal de vivienda. Cada fila de la tabla es una "
                   "variable: a la izquierda, el dato original que trae la base y en qué porcentaje de avisos viene; al "
                   "lado, cómo construimos la variable, en cuántos avisos tiene dato y su promedio. La primera fila es el "
                   "precio: es lo que se predice, por eso no cuenta entre las 162; está en todo el entrenamiento y viene "
                   "vacío en la prueba. Alcobas y tipo de inmueble vienen completos. El área es el gran problema: cada "
                   "dato de superficie viene en uno de cada cinco avisos, y alguno de los dos en casi uno de cada tres, "
                   "el 31 % del entrenamiento. Con la regla de tomar la cubierta, luego la total y luego el área escrita en "
                   "el texto, la tenemos en el 54 % de todos los avisos, y tres marcas de sí o no dicen de dónde salió o si "
                   "falta. También usamos el logaritmo del área, una escala comprimida: pasar de 50 a 100 metros pesa lo "
                   "mismo que pasar de 100 a 200. Los baños pasan de 74 % a 85 % con dato gracias al texto."))
    grupos_b = [
        (f"título y descripción ({titulo_desc})", [
            fv("parqueaderos_imp", "Parqueaderos", "el número escrito; nombrados sin número, 1; si no, 0"),
            fv("piso_texto", "Piso (número)", "el piso escrito en el texto"),
            fv("estrato_texto", "Estrato", "el estrato escrito en el texto"),
            fv("antiguedad_texto", "Antigüedad (años)", "los años de construido escritos en el texto")]),
        (f"mes y año ({par('month')})", [
            fv("tiempo", "Fecha del aviso", "año con el mes como decimal: julio de 2021 es 2021,5",
               prom=f"{est['tiempo']['promedio']:.1f}".replace(".", ","))]),
        (f"latitud y longitud ({par('lat')})", [
            fv("x_km", "Coordenada este (km)", "posición oeste a este, en el sistema oficial de Colombia", prom="no aplica"),
            fv("y_km", "Coordenada norte (km)", "posición sur a norte, en el mismo sistema", prom="no aplica"),
            fv("coord_redondeada", "Ubicación aproximada", "sí si latitud y longitud traen 3 decimales o menos: una malla "
                                                           "de unos 110 m", si=True)]),
        ("avisos vecinos, sin usar su precio", [
            fv("n_avisos_500m", "Avisos a 500 m", "cuántos avisos hay a 500 m o menos"),
            fv("prop_apartamentos_500m", "Apartamentos a 500 m", "% de apartamentos entre esos avisos", si=True),
            fv("area_mediana_500m", "Área típica a 500 m (m²)", "área mediana de esos avisos: la mitad tiene menos")])]
    s = _lamina_fuente(deck, "Fuente 1b. Avisos de Properati: texto, fecha, ubicación y avisos vecinos", base, grupos_b,
                       pie_b, 0, 11, 21)
    lam.append((s, "Seguimos con Properati. Del título y la descripción salen los parqueaderos, que casi siempre se pueden "
                   "leer, y el piso, el estrato y la antigüedad, que aparecen en pocos avisos. La fecha se vuelve un número: "
                   "julio de 2021 es 2021,5. La latitud y la longitud se pasan a kilómetros en el sistema oficial de "
                   "coordenadas de Colombia; su promedio no dice nada, por eso no lo mostramos. Además marcamos las "
                   "ubicaciones aproximadas: en el 43 % de los avisos la latitud y la longitud vienen con tres decimales o "
                   "menos, así que el punto cae en una malla de unos 110 metros. Y con todos los avisos vecinos, sin usar su "
                   "precio, describimos el entorno a 500 metros: cuántos avisos hay, qué parte son apartamentos y el área "
                   "típica, la mediana, que es el valor que deja la mitad por debajo. En total, 21 variables de esta "
                   "fuente."))
    return lam


# ------------------------------------------------------------------------------------------------------------------
# 6. Apertura: las técnicas que convierten el texto en variables
# ------------------------------------------------------------------------------------------------------------------
def _apertura_clase(deck):
    s = _nueva(deck, "Del texto a variables: nueve técnicas y dónde entra cada una")
    texto(s, 0.55, 1.22, 12.25, 0.45, "Cada aviso es un documento y los 48.930 avisos son el corpus. Del texto salen 84 de las "
          "162 variables.", size=13, color="navy")
    filas = [("Técnica", "Qué es, en palabras simples", "Ejemplo con avisos", "Dónde entra"),
             ("Expresiones regulares", "patrones para encontrar palabras y números en el texto",
              "sin ascensor no cuenta como ascensor; 86 m2 da el área", "19 indicadores de sí o no y 4 cifras escritas"),
             ("Normalización, palabras vacías y palabras de dominio", "dejar el texto parejo y quitar las palabras que no "
              "distinguen un aviso de otro", "balcón queda balcon; salen de, la, que y vender; se quedan no, sin y ni",
              "limpieza del texto"),
             ("Token y stemming (Snowball)", "cada palabra es un token; el stemming la recorta a su raíz para juntar sus "
              "variantes", "apartamento y apartamentos quedan en apartament", "limpieza del texto"),
             ("Bolsa de palabras, bigramas y matriz documento-término", "contar cuántas veces sale cada palabra y cada par "
              "de palabras seguidas (bigrama), sin mirar el orden; se quitan los términos raros",
              "una fila por aviso y un conteo por término, como chimenea o sala comedor",
              "tabla de 48.930 avisos por 44.888 términos"),
             ("TF-IDF", "dar más peso a lo que se repite en el aviso y casi nadie más dice",
              "refugio pesa 0,27; apartamento, 0,05", "insumo del LSA y del ridge"),
             ("PCA del texto (LSA) y screeplot", "resumir miles de términos en pocos números; el screeplot muestra "
              "cuánto resume cada uno", "un componente separa casas de apartamentos en edificio", "50 variables"),
             ("Modelo de tópicos (LDA)", "descubrir temas: grupos de palabras que suelen ir juntas",
              "tema de casas; tema de cercanía a comercio", "10 variables"),
             ("Regresión penalizada (ridge, pariente del LASSO)", "aprender del precio qué palabras van con avisos caros o "
              "baratos", "3 parqueaderos sube el puntaje; apartaestudio lo baja", "1 variable: puntaje de precio del texto"),
             ("BETO, embeddings contextuales y distancia de edición", "modelo de lenguaje que entiende cada palabra por las "
              "que la rodean", "apartmamento se corrige a apartamento", "corrector de ortografía; como variable no entró")]
    y_fin = _tabla(s, 0.55, 1.66, filas, [3.0, 3.75, 3.35, 2.15], minimo=0.3)
    texto(s, 0.55, min(max(y_fin + 0.05, 6.35), 6.62), 12.25, 0.36, "Las tres primeras filas preparan el texto; las demás lo vuelven "
          "números. Solo la regresión penalizada mira el precio (supervisada); las demás no (no supervisadas).", size=13,
          color="navy")
    FIN[id(s)] = y_fin
    notas = ("Antes de entrar al texto, el mapa de las técnicas con las que lo convertimos en variables, y dónde entra "
             "cada una. Cada aviso es un documento y los 48.930 avisos son el corpus. Con expresiones regulares, que "
             "son patrones de búsqueda, leemos indicadores de sí o no y cifras, cuidando la negación: sin ascensor no es "
             "ascensor. Después limpiamos el texto: lo normalizamos, quitamos palabras vacías y palabras de dominio, que son "
             "las que usa todo aviso de venta, y recortamos cada palabra, o token, a su raíz con el stemming de Snowball. "
             "Con eso armamos la bolsa de palabras: contamos palabras y bigramas, que son pares de palabras seguidas, en la "
             "matriz documento-término, sin los términos raros. La ponderamos con TF-IDF. De ahí salen tres resúmenes: el "
             "PCA del texto, que aquí se llama LSA y se lee con el screeplot; los tópicos de LDA; y una regresión "
             "penalizada, un ridge, pariente del LASSO, la única que mira el precio. BETO, un modelo de embeddings "
             "contextuales, lo usamos como corrector de ortografía junto con la distancia de edición; "
             "como variable lo probamos y no entró.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# 7. Pasos y modelos del texto
# ------------------------------------------------------------------------------------------------------------------
def _pasos_modelos(deck):
    lim = json.load(open(D / "ppt_texto_limpieza.json", encoding="utf-8"))
    M = json.load(open(D / "ppt_modelos_texto.json", encoding="utf-8"))
    pesos = {r["termino"]: r["peso"] for r in M["aviso"]["tabla"]}
    s = _nueva(deck, "Del texto crudo a 84 variables: pasos, modelos y lo que queda en los datos")
    pasos = [
        ("1. Normalizar", "reglas fijas", False,
         "minúsculas, sin tildes ni cifras tapadas con X; números en letras: segundo piso pasa a piso 2"),
        ("2. Reparar letras perdidas", "regla: la palabra más usada", False,
         f"repone la letra perdida en los datos: bao pasa a bano (baño sin tilde); {n(lim['reparaciones'], 0)} "
         "palabras reparadas"),
        ("3. Corregir la ortografía", "modelo de lenguaje BETO", True,
         f"la palabra más probable en su contexto: apartmamento pasa a apartamento; {n(lim['beto_aceptadas'], 0)} "
         "corregidas"),
        ("4. Quitar firmas de agentes", "listas de nombres del DNP", True,
         "borra el nombre del agente tras contáctame o asesora; 1.424 avisos (2,9 %)"),
        ("5. Quitar palabras vacías", "listas del DNP y de NLTK", True,
         "quita 327 palabras como de, la o que, y de dominio como vender; deja no, sin y ni (negación)"),
        ("6. Recortar a la raíz", "stemming (Snowball)", False,
         "apartamento y apartamentos cuentan como un término, apartament; quedan 11.949 raíces"),
        ("7. Contar palabras y pares", "bolsa de palabras", False,
         f"cuántas veces sale cada palabra y cada par seguido (sala comedor), si están en 5 avisos o más: matriz "
         f"documento-término de {n(M['N'], 0)} avisos por {n(M['P'], 0)} términos"),
        ("8. Dar peso a lo raro", "peso TF-IDF", True,
         f"pesa más lo que se repite en el aviso y casi nadie dice: refugio {n(pesos['refugi'], 2)}; apartamento "
         f"{n(pesos['apartament'], 2)}"),
    ]
    x0, w, y = 0.55, 7.55, 1.3
    cajas, centros = [], []
    for titulo, herramienta, es_modelo, detalle in pasos:
        lineas_det = _lineas(detalle, w - 0.3, 12)
        h = 0.3 + lineas_det * 0.205 + 0.07
        caja = forma(s, MSO_SHAPE.ROUNDED_RECTANGLE, x0, y, w, h, relleno="tinte",
                     linea="azul_suave" if es_modelo else "gris_claro", radio=0.1)
        texto(s, x0 + 0.12, y + 0.02, 4.2, 0.28, titulo, size=13, color="navy")
        texto(s, x0 + w - 3.3, y + 0.02, 3.18, 0.28, herramienta, size=12, color="azul" if es_modelo else "muted",
              align="right")
        texto(s, x0 + 0.12, y + 0.28, w - 0.24, h - 0.3, detalle, size=12, color="tinta")
        cajas.append(caja)
        centros.append(y + h / 2)
        y += h + 0.06
    FIN[id(s)] = y
    xr, wr = 8.45, 4.35
    tarjeta(s, xr, 1.3, wr, 1.5, "84 variables entran al modelo final (LightGBM)", [
        "Los cuatro grupos se calculan por separado y entran juntos, con las otras 78 variables.",
        "BETO, las listas y TF-IDF solo preparan el texto: no son variables."], tipo="clave", maximo=13)
    a = _cuadro(s, xr, 2.9, wr, 0.7, "Indicadores y cifras: 23 variables",
                "con el texto del paso 4; reglas que no miran el precio", modelo=False)
    texto(s, xr, 3.66, wr, 0.95, "No supervisado: no mira el precio. Supervisado: aprende del precio. DNP: "
          "Departamento Nacional de Planeación. NLTK: herramienta de análisis de texto. Cada uno tiene su lámina.", size=12,
          color="muted")
    b = _cuadro(s, xr, 4.62, wr, 0.74, "LDA: 10 variables", "con los conteos del paso 7: agrupa las palabras "
                "en 10 temas; no supervisado")
    c = _cuadro(s, xr, 5.42, wr, 0.74, "LSA: 50 variables", "con los pesos del paso 8: resume la tabla en 50 "
                "números por aviso; no supervisado")
    d = _cuadro(s, xr, 6.22, wr, 0.74, "Ridge: 1 variable", "con los pesos del paso 8 y el precio: puntaje de "
                "precio; supervisado")
    for o, d_ in ((cajas[3], a), (cajas[6], b), (cajas[7], c), (cajas[7], d)):
        _arista(s, o, d_)
    notas = ("Este es el camino completo del texto. A la izquierda, ocho pasos; a la derecha de cada uno, la herramienta que "
             "usa; en azul, las que vienen de un modelo o de una lista externa. Los cuatro primeros limpian: normalizamos con "
             "reglas fijas y borramos las cifras tapadas con X en los datos de la competencia; reponemos la ñ y las vocales "
             "que se perdieron en los datos; BETO corrige la ortografía según el contexto; y con las listas de nombres del "
             "Departamento Nacional de Planeación borramos la firma del agente. Sobre ese texto limpio se leen 19 "
             "indicadores y 4 cifras: 23 variables. Los pasos 5 a 7 preparan el conteo: quitamos palabras vacías, recortamos "
             "cada palabra a su raíz, con stemming y no con lematización, porque predice igual de bien "
             "a una fracción del costo, y contamos palabras y pares en la matriz documento-término. Con esos conteos, LDA da 10 "
             "variables de temas. El paso 8, TF-IDF, da más peso a lo raro; de esa tabla salen las 50 variables de LSA y, "
             "con el precio del entrenamiento, el puntaje del ridge. Son ramas paralelas: ninguna alimenta a otra. LDA y LSA "
             "no miran el precio, son no supervisados; el ridge sí, es supervisado. Las 84 entran juntas al modelo final, "
             "con las otras 78 variables. BETO, las listas y TF-IDF no son variables: solo preparan el texto.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# 8. Un aviso real, regla por regla
# ------------------------------------------------------------------------------------------------------------------
def _aviso_real(deck):
    import re as _re
    cand = pd.read_parquet(D / "ppt_texto_candidatos.parquet").set_index("property_id")
    R = json.load(open(D / "ppt_fuentes.json", encoding="utf-8"))
    av = cand.loc["84a9da9d84b975bcd0954c36"]
    descr = str(av["description"]).strip()
    resaltar = ["86 m2", "segundo piso", "exterior", "balcon", "2 habitaciones", "chimenea", "parqueadero", "deposito",
                "gym", "salon comunal", "terraza panoramica", "bano privado", "bano social"]
    patron = _re.compile("|".join(_re.escape(r) for r in sorted(resaltar + ["informes X"], key=len, reverse=True)))
    partes, pos = [], 0
    for m in patron.finditer(descr):
        if m.start() > pos:
            partes.append((descr[pos:m.start()], {"color": "tinta"}))
        partes.append((m.group(0), {"color": "naranja" if m.group(0) == "informes X" else "azul"}))
        pos = m.end()
    if pos < len(descr):
        partes.append((descr[pos:], {"color": "tinta"}))
    s = _nueva(deck, "Un aviso real: qué sacamos de su texto y con qué regla")
    texto(s, 0.55, 1.2, 12.25, 0.34, f"Aviso de Chapinero de julio de 2021, precio publicado de {n(av['price'] / 1e6, 0)} "
          "millones. Sus datos traen alcobas (2) y ambientes (2), pero no área ni baños.", size=13, color="navy")
    forma(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.55, 1.54, 12.25, 0.68, relleno="tinte", linea="gris_claro", radio=0.05)
    texto(s, 0.68, 1.57, 12.0, 0.64, [[(f"{str(av['title'])}. ", {"color": "muted"})] + partes], size=12)
    texto(s, 0.55, 2.23, 12.25, 0.3, "Así viene este aviso en los datos: sin tildes ni ñ (bano es baño). Gris: título. Azul: lo "
          "que se lee. Naranja: cifra tapada en los datos de la competencia.", size=12, color="muted")
    ind = "terraza o balcón, chimenea, vista, exterior, gimnasio, depósito, zona social, parqueadero"
    filas = [("En el aviso", "Regla", "Variable que resulta", "Valor"),
             ("86 m2", "expresión regular: cifra antes de m2 (20 a 1.000)", "Área escrita y, como faltaba, Área", "86 m²"),
             ("2 habitaciones", "sinónimo (paso 1): pasa a 2 alcoba; luego, la cifra antes de alcoba (1 a 10)",
              "Alcobas escritas", "2"),
             ("segundo piso", "número en letras (paso 1): pasa a piso 2; luego, la cifra después de piso (1 a 40)", "Piso",
              "2"),
             ("parqueadero", "se nombra sin cifra", "Parqueaderos", "1"),
             ("bano privado, bano social", "no hay cifra antes de baño", "Baños escritos y Baños quedan sin dato",
              "sin dato"),
             ("balcon, terraza, chimenea, panoramica, exterior, gym, deposito, salon comunal, parqueadero",
              "palabra de las listas de indicadores, sin negación", f"8 indicadores: {ind}", "sí, cada uno"),
             ("no dice ascensor", "sin mención (igual si dijera sin ascensor)", "Ascensor y los otros 10 indicadores", "no"),
             ("informes X", "cifra tapada en los datos de la competencia", "se borra y no se usa", "no aplica"),
             ("todo el texto", "modelo de temas (LDA; ver su lámina)", "Tema 8: apartamento y servicios del edificio",
              "0,98 de 1"),
             ("todo el texto", "precio que sugieren sus palabras (ridge; ver su lámina)",
              "Puntaje de precio del texto", f"{n(av['puntaje_millones'], 0)} millones")]
    colores = {(r_, 0): ("naranja" if filas[r_][0] == "informes X" else "azul") for r_ in range(1, len(filas))}
    y_fin = _tabla(s, 0.55, 2.52, filas, [3.55, 3.2, 4.3, 1.2], colores=colores, minimo=0.26)
    ar = R["avisos_variables_grupo"]
    texto(s, 0.55, max(y_fin + 0.06, 6.5), 12.25, 0.36,
          f"Gracias al texto, en el entrenamiento los avisos con área pasan de 31 % a {n(ar['area_imp']['entrenamiento'], 0)} "
          f"% y los avisos con baños, de {n(R['avisos_brutas_no_nulos_grupo']['bathrooms']['entrenamiento'], 0)} % a "
          f"{n(ar['banos_imp']['entrenamiento'], 0)} %.", size=14, color="verde")
    FIN[id(s)] = y_fin
    notas = ("Ahora sigamos un aviso real de Chapinero, de punta a punta. Sus datos no traen área ni baños. Así viene el texto: "
             "sin tildes ni ñ, por eso dice bano. Con expresiones regulares, la regla del área busca una "
             "cifra antes de m2 y encuentra 86: esa es su área, y una marca recuerda que salió del texto. Con listas fijas, "
             "sin ningún modelo, habitaciones se vuelve alcoba y segundo piso se vuelve piso 2; después se toma la cifra "
             "junto a alcoba o a piso. El parqueadero, mencionado sin cifra, cuenta como uno. Los baños no tienen cifra, así "
             "que siguen sin dato. Ocho palabras de las listas de indicadores prenden sus marcas de sí; el ascensor no se "
             "menciona y queda en no, igual que si dijera sin ascensor. La X es una cifra tapada en los datos de la "
             "competencia y se borra. Con todo el texto, el modelo de temas dice que el aviso habla casi por completo del "
             "tema 8, apartamento y servicios del edificio, y el ridge, que veremos más adelante, le da un puntaje de "
             f"{n(av['puntaje_millones'], 0)} millones frente a un precio publicado de {n(av['price'] / 1e6, 0)}. Abajo, lo "
             "más importante: usamos la descripción para llenar datos que faltan; así, la parte "
             "del entrenamiento con área pasa de 31 % a 53 %, y con baños, de 74 % a 85 %.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# 9. BETO
# ------------------------------------------------------------------------------------------------------------------
def _beto(deck):
    lim = json.load(open(D / "ppt_texto_limpieza.json", encoding="utf-8"))
    s = _nueva(deck, "BETO: un modelo de lenguaje que corrige la ortografía según el contexto")
    _tres_tarjetas(s,
                   ["BETO es un modelo de lenguaje en español (Cañete y otros, 2020), entrenado con unos 3.000 millones de "
                    "palabras. Si se tapa una palabra, calcula qué tan probable es cada palabra posible en ese lugar, "
                    "mirando las palabras de los dos lados.",
                    "Es un modelo de embeddings contextuales: una palabra se entiende por las que la rodean."],
                   ["1. Sospechosas: palabras de 4 letras o más, en menos de 5 avisos y que BETO no conoce.",
                    "2. Candidatas: palabras frecuentes (20 avisos o más) a distancia de edición 1, o 2 si la sospechosa "
                    "tiene 6 letras o más: letras cambiadas, de más, de menos o volteadas.",
                    "3. Se tapa la palabra en hasta 2 avisos, con 40 palabras a cada lado, y BETO elige la candidata más "
                    "probable."],
                   ["Se acepta si su nota es −5 o mejor cuando cambia una letra, y −3 o mejor cuando cambian dos; nunca si "
                    "solo agrega o quita una s.",
                    f"De {n(lim['beto_candidatas'], 0)} palabras revisadas se corrigieron {n(lim['beto_aceptadas'], 0)}. No "
                    "crea variables: corrige el texto antes de leer cifras e indicadores y de contar."],
                   h=2.45, titulo_como="Cómo lo usamos", maximo=13)
    texto(s, 0.55, 3.88, 12.25, 0.32, f"Ejemplos del registro de las {n(lim['beto_candidatas'], 0)} palabras revisadas",
          size=13, color="navy")
    filas = [("Palabra en el aviso", "Candidata", "Letras de diferencia",
              "Nota de encaje en la frase (0 es perfecto; más negativa, peor)", "Decisión"),
             ("sauiza", "suiza", "1", "−0,002", "aceptada"),
             ("cocona", "cocina", "1", "−3,03", "aceptada: a una letra basta −5"),
             ("tcon", "con", "1", "−3,76", "aceptada"),
             ("apartmamento", "apartamento", "1", "−3,78", "aceptada"),
             ("intetior", "interior", "1", "−7,66", "rechazada: encaja poco en su frase"),
             ("certicado", "certificado", "2", "−9,60", "rechazada: encaja poco en su frase"),
             ("juegoss", "juegos", "1", "−0,76", "rechazada: solo quita una s")]
    y_fin = _tabla(s, 0.55, 4.2, filas, [2.2, 2.0, 1.55, 3.6, 2.9], minimo=0.26)
    texto(s, 0.55, max(y_fin + 0.05, 6.6), 12.25, 0.36, "La nota es el logaritmo de la probabilidad que da BETO: −3 equivale a "
          "5 % y −5, a menos de 1 %. Preferimos no corregir a corregir mal.", size=12, color="muted")
    FIN[id(s)] = y_fin
    notas = ("El primer modelo es BETO, un modelo de lenguaje en español entrenado con unos 3.000 millones de palabras. Es un "
             "modelo de embeddings contextuales: entiende cada palabra por las que la rodean. Su "
             "habilidad básica es adivinar una palabra tapada a partir del contexto, y la usamos como corrector. Tomamos las "
             "palabras raras que BETO no conoce; buscamos palabras frecuentes a distancia de edición uno o dos, es decir, que "
             "difieran en una o dos letras; tapamos la palabra en el aviso y le preguntamos a BETO cuál candidata encaja "
             "mejor. Cada candidata recibe una nota: cero es encaje perfecto y entre más negativa, peor. Somos "
             "conservadores: aceptamos con −5 o mejor si cambia una letra, y con −3 o mejor si cambian dos. Por eso cocona "
             "pasa a cocina, pero intetior no se corrige: interior es la palabra correcta, pero en esa frase encajó poco y "
             "preferimos no corregir a corregir mal. De 3.652 palabras revisadas corregimos 732. BETO no crea variables: "
             "cambia el texto antes de contar. BETO tiene otros dos usos, como variables o con ajuste "
             "fino; los probamos y no entraron al modelo final.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# 10. Listas del DNP y de NLTK
# ------------------------------------------------------------------------------------------------------------------
def _listas(deck):
    s = _nueva(deck, "Listas de palabras que se quitan del texto: de dónde salen y cómo se usan")
    tarjeta(s, 0.55, 1.33, 3.95, 1.75, "ConTexto, del DNP", "Herramienta de análisis de texto en español del Departamento "
            "Nacional de Planeación (DNP). Usamos sus listas de palabras vacías y de nombres.",
            tipo="definicion", maximo=14)
    tarjeta(s, 4.7, 1.33, 3.95, 1.75, "NLTK", "Herramienta clásica de análisis de texto (Bird, Klein y Loper, 2009). Usamos "
            "su lista de palabras vacías en español.", tipo="definicion", maximo=14)
    tarjeta(s, 8.85, 1.33, 3.95, 1.75, "Cómo queda en los datos", [
        "No crean variables: deciden qué palabras salen del texto.",
        "Las firmas de agentes se borran en 1.424 avisos (2,9 %), y las 327 palabras vacías nunca entran a la tabla de "
        "conteos."], tipo="clave", maximo=14)
    filas = [("Lista", "De dónde sale", "Palabras", "Ejemplos", "Cómo se usa"),
             ("Palabras vacías del DNP", "ConTexto (DNP)", "350; sin tildes, 329 distintas", "de, la, que, además, aunque, "
              "sino", "base de la lista que usamos"),
             ("Palabras vacías de NLTK", "NLTK, lista en español", "313; sin tildes, 306, todas ya en la del DNP",
              "de, la, que, el, en, y", "se unen con la del DNP: siguen siendo 329"),
             ("Lista que usamos", "las dos, en minúsculas y sin tildes", "327",
              "se agregan ser, haber y tener (estar ya estaba): 332; se sacan no, sin, ni, estado y estados: 327",
              "se quita cada palabra que esté en ella al contar (paso 5); no, sin y ni se quedan para la negación"),
             ("Palabras de dominio (las que usa todo aviso de venta)", "lista del equipo", "18",
              "vender, venta, inmueble, oferta, contacto, bogota", "se quitan sus raíces al contar"),
             ("Nombres y apellidos del DNP", "ConTexto: 3 listas de nombres y 1 de apellidos",
              "sin tildes, distintos: nombres 3.572, 6.439 y 6.316; apellidos 2.388", "nombres de pila y apellidos comunes",
              "se borran solo en las 4 palabras tras contáctame, asesora u otra expresión de contacto (paso 4); no los "
              "que son barrios, como Rosales o Chicó")]
    y_fin = _tabla(s, 0.55, 3.22, filas, [2.15, 2.15, 2.25, 2.85, 2.85], minimo=0.3)
    FIN[id(s)] = y_fin
    notas = ("Dos herramientas externas deciden qué palabras salen del texto. La primera es ConTexto, la herramienta de texto "
             "en español del Departamento Nacional de Planeación; usamos sus listas. La segunda es "
             "NLTK, la herramienta clásica de análisis de texto. Las dos traen listas de palabras vacías, como de, la o que, "
             "que no distinguen un aviso de otro. La del DNP tiene 329 palabras distintas sin tildes; la de NLTK no agrega "
             "ninguna nueva. A esa lista le agregamos ser, haber y tener, y le sacamos no, sin y "
             "ni, para poder leer la negación, y estado y estados: quedan 327. También quitamos 18 palabras de dominio, las "
             "que usa todo aviso de venta, como vender o inmueble. Además, el DNP tiene listas de nombres y apellidos: con "
             "ellas borramos la firma del agente, pero solo justo después de expresiones como contáctame o asesora, y sin "
             "tocar nombres que también son barrios, como Rosales o Chicó. Pasó en 1.424 avisos. Ninguna de estas listas crea "
             "variables: solo limpian el texto antes de contar.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# 11 y 12. Fuente 2a y 2b: indicadores y cifras
# ------------------------------------------------------------------------------------------------------------------
def _fuente2ab(deck):
    V = json.load(open(D / "ppt_variables.json", encoding="utf-8"))["variables"]

    def ind(c, nombre, palabras):
        return nombre, palabras, _si(V[c]["promedio"])

    def ind6(c, nombre, palabras):
        return nombre, palabras, _pct(V[c]["entrenamiento"]), _pct(V[c]["prueba"]), _si(V[c]["promedio"])

    def cif(c, nombre, como):
        return nombre, como, _pct(V[c]["entrenamiento"]), _pct(V[c]["prueba"]), n(V[c]["promedio"], 1)

    lam = []
    base = ("Un indicador vale sí (1) si el aviso nombra la característica y no (0) si no la nombra o la niega: sin ascensor "
            "cuenta como no. Se busca con expresiones regulares en el título y la descripción de los 48.930 avisos. Todos "
            "tienen dato en todos los avisos.")
    grupos = [
        ("8 indicadores básicos de vivienda", [
            ind("txt_parqueadero", "Menciona parqueadero", "parqueadero"),
            ind("txt_piscina", "Piscina", "piscina o jacuzzi"),
            ind("txt_ascensor", "Ascensor", "ascensor"),
            ind("txt_terraza_balcon", "Terraza o balcón", "terraza o balcón"),
            ind("txt_vigilancia", "Vigilancia", "vigilancia, seguridad, portería o portero"),
            ind("txt_remodelado", "Remodelado", "remodelado, renovado o remodelación"),
            ind("txt_cocina_integral", "Cocina integral", "cocina integral"),
            ind("txt_zona_social", "Zona social", "zona social, o salón social, comunal o de reuniones")]),
        ("Indicadores que agregó el equipo", [
            ind("txt_chimenea", "Chimenea", "chimenea"),
            ind("txt_vista", "Vista", "vista o panorámica"),
            ind("txt_estudio", "Estudio", "estudio"),
            ind("txt_apartaestudio", "Apartaestudio", "apartaestudio, aparta estudio o loft")])]
    s = _lamina_fuente(deck, "Fuente 2a. Texto del aviso: marcas de sí o no por palabra clave, sin contar las negadas", base,
                       grupos, "Negaciones que se leen: sin, no tiene, no cuenta con, no dispone de, no hay y no incluye. El "
                               "parqueadero se menciona en 7 de cada 10 avisos; la piscina, en menos de 1 de cada 10.",
                       1, 12, 33, encabezado=("Dato original", "Variable", "Palabras que busca en el texto",
                                              "% de avisos con sí"),
                       anchos=[3.2, 2.6, 4.9, 1.55], size_base=13, alto_base=0.5)
    lam.append((s, "La segunda fuente es el texto del aviso: 84 variables. Empezamos por los indicadores: marcas de sí o no. "
                   "Cada uno vale sí si el aviso nombra la característica, buscada con expresiones regulares, y la mención "
                   "no está negada: no tiene ascensor no cuenta como ascensor. Ocho son indicadores básicos de "
                   "vivienda y cuatro los agregamos nosotros. A la derecha, qué tan común es cada uno: el parqueadero "
                   "se menciona en siete de cada diez avisos; la piscina, en menos de uno de cada diez. Todos tienen dato en "
                   "todos los avisos, porque un aviso que no nombra algo queda en no."))
    grupos = [
        ("Indicadores que agregó el equipo", [
            ind6("txt_duplex_penthouse", "Dúplex o penthouse", "dúplex, penthouse o pent house"),
            ind6("txt_exterior", "Exterior (da a la calle)", "exterior"),
            ind6("txt_gimnasio", "Gimnasio", "gimnasio o gym"),
            ind6("txt_deposito", "Depósito", "depósito o bodega"),
            ind6("txt_conjunto", "Conjunto cerrado", "conjunto o conjunto cerrado"),
            ind6("txt_nuevo_estrenar", "Nuevo", "para estrenar, sobre planos o proyecto nuevo"),
            ind6("txt_para_remodelar", "Para remodelar", "para remodelar o reformar, requiere o necesita remodelación")]),
        ("Cifras escritas en el texto", [
            cif("alcobas_texto", "Alcobas escritas", "el número antes de alcoba; se acepta de 1 a 10"),
            cif("banos_texto", "Baños escritos", "el número antes de baño; se acepta de 1 a 10"),
            cif("parqueaderos_texto", "Parqueaderos escritos", "el número antes de parqueadero; se acepta de 1 a 6"),
            cif("area_texto", "Área escrita (m²)", "el número tras área o antes de m2; se acepta de 20 a 1.000")])]
    s = _lamina_fuente(deck, "Fuente 2b. Texto del aviso: más marcas de sí o no y las cifras escritas",
                       "Mismo texto y mismas reglas; 18 de los 19 indicadores leen la negación. Las cifras también se leen "
                       "con expresiones regulares.", grupos,
                       "Una cifra queda sin dato si el texto no trae un número creíble. Las cifras completan el área y los "
                       "baños de la fuente 1 y además entran solas al modelo; parqueaderos, piso, estrato y antigüedad "
                       "salen del texto, pero se cuentan en la fuente 1.", 1, 11, 44,
                       encabezado=("Dato original", "Variable", "Qué busca en el texto", "Con dato, entrenamiento",
                                   "Con dato, prueba", "Promedio o % con sí"),
                       anchos=[2.3, 2.35, 4.2, 1.25, 1.05, 1.1])
    lam.append((s, "Siguen los demás indicadores y las cuatro cifras que leemos en el texto: alcobas, baños, parqueaderos y "
                   "área. Un detalle: el indicador para remodelar no pasa por la regla de negación; los otros 18 sí. En las "
                   "cifras solo se acepta un número en un rango creíble, por eso tienen avisos sin dato: el área escrita "
                   "aparece en uno de cada tres. Estas cifras sirven dos veces: completan el área y los baños de la fuente "
                   "1 y además entran solas al modelo."))
    return lam


# ------------------------------------------------------------------------------------------------------------------
# 13 a 16. TF-IDF, LSA, LDA y ridge
# ------------------------------------------------------------------------------------------------------------------
def _tfidf(deck):
    M = json.load(open(D / "ppt_modelos_texto.json", encoding="utf-8"))
    s = _nueva(deck, "TF-IDF: cuánto pesa cada término en cada aviso")
    _tres_tarjetas(s,
                   ["Frecuencia en el aviso (TF) por rareza en la base (IDF). Un término pesa más si se "
                    "repite en el aviso y si aparece en pocos avisos. Si una palabra está en todos los avisos, no distingue "
                    "a ninguno: apartamento pesa poco."],
                   ["En palabras simples: TF crece si el término se repite en el aviso; IDF, si pocos avisos lo usan; el "
                    "peso es TF por IDF.",
                    "TF = 1 + ln(veces en el aviso); IDF = 1 + ln((1 + N)/(1 + avisos con el término)), con N = "
                    f"{n(M['N'], 0)}: variante suavizada de la de clase (ln: logaritmo).",
                    "Al final, cada aviso se reescala para que uno largo no pese más (en el ejemplo, se divide por "
                    f"{n(M['aviso']['norma'], 2)})."],
                   [f"Una tabla de {n(M['N'], 0)} avisos por {n(M['P'], 0)} términos, casi toda en ceros: un aviso típico usa "
                    f"{n(M['terminos_por_aviso_mediana'], 0)} términos.",
                    "No entra como variables: es el insumo del LSA y del ridge. Se calcula con todos los avisos, de "
                    "entrenamiento y de prueba; se puede porque no mira el precio."], h=2.42, maximo=13)
    av = M["aviso"]
    t = {r["termino"]: r for r in av["tabla"]}
    texto(s, 0.55, 3.85, 12.25, 0.32, f"Una fila de la tabla, el aviso del ejemplo ({av['terminos']} términos). Refugio, que "
          f"usan {n(t['refugi']['df'], 0)} avisos, pesa unas cinco veces más que apartamento, que usan "
          f"{n(t['apartament']['df'], 0)}.", size=13, color="navy")
    nombres = {"refugi": "refugi (refugio)", "chic refugi": "chic refugi (chicó refugio)",
               "terraz panoram": "terraz panoram (terraza panorámica)", "gym": "gym", "chimene": "chimene (chimenea)",
               "alcob": "alcob (alcoba)", "apartament": "apartament (apartamento)", "parqueader": "parqueader (parqueadero)"}
    filas = [("Término (raíz o par de raíces)", "Veces en el aviso", "Avisos con el término", "TF", "IDF", "TF × IDF",
              "Peso final")]
    for r in sorted(av["tabla"], key=lambda r: -r["peso"]):
        filas.append((nombres.get(r["termino"], r["termino"]), str(r["conteo"]), n(r["df"], 0),
                      n(1 + math.log(r["conteo"]), 2), n(r["idf"], 2), n(r["peso_sin_normalizar"], 2), n(r["peso"], 2)))
    y_fin = _tabla(s, 0.55, 4.2, filas, [3.45, 1.55, 2.05, 0.9, 1.0, 1.3, 2.0], minimo=0.26)
    FIN[id(s)] = y_fin
    notas = ("TF-IDF no se estima: es una forma de dar peso. Cada celda de la tabla de conteos dice cuántas veces aparece un "
             "término en un aviso; TF-IDF lo convierte en un peso. TF crece si el término se repite en el aviso, IDF crece "
             "si pocos avisos lo usan, y el peso es el producto. Es la idea de clase: si una palabra está en todos los "
             "documentos no distingue a ninguno. Miren la fila del aviso del ejemplo. Refugio aparece dos veces y solo en 277 "
             "avisos: TF de 1,69, IDF de 6,17, producto 10,45; al reescalar la fila, dividiendo por 38,47, queda en 0,27. "
             "Apartamento también aparece dos veces, pero está en 40.371 avisos: su IDF es 1,19 y su peso final 0,05. Así, "
             "lo que distingue al aviso pesa unas cinco veces más que lo que todos dicen. La tabla es casi toda ceros: un "
             "aviso usa unos 75 de los 44.888 términos. No entra como variables; alimenta al LSA y al ridge.")
    return s, notas


def _lsa(deck):
    M = json.load(open(D / "ppt_modelos_texto.json", encoding="utf-8"))
    L = M["lsa"]
    e = L["pve_pct_por_componente"]
    s = _nueva(deck, "LSA: la tabla de pesos resumida en 50 componentes")
    _tres_tarjetas(s,
                   ["En palabras simples: LSA, el análisis semántico latente, resume los 44.888 términos de la tabla en 50 "
                    "números por aviso. Cada número, un componente, mide cuánto usa el aviso un grupo de palabras que suelen "
                    "ir juntas. Es el PCA aplicado al texto (técnicamente, una SVD truncada que no resta el "
                    "promedio)."],
                   ["1. Se parte de la tabla TF-IDF completa.",
                    "2. El programa busca los 50 componentes que más diferencian a los avisos.",
                    "3. El número, 50, lo fijamos; no probamos otros.",
                    "4. Cada aviso recibe 50 números: su posición en cada componente."],
                   ["50 variables, del componente 1 (lsa_00) al 50 (lsa_49), sin faltantes.",
                    f"Cuánto resume: los 50 recogen el {n(L['pve_pct_acumulada_50'], 1)} % de las diferencias entre avisos "
                    "(varianza explicada). Es poco: los avisos usan palabras muy variadas; por eso además usamos el "
                    "ridge, que trabaja con la tabla completa."], h=2.3, maximo=13)
    acum_e = [sum(e[:k + 1]) for k in range(len(e))]
    for x_, tit_, ys_, ymax_, uy_ in ((0.55, "A. Varianza explicada por componente (%): screeplot", e, 1.5, 0.5),
                                      (3.85, "B. Varianza explicada acumulada (%)", acum_e, 15, 5)):
        texto(s, x_, 3.72, 3.2, 0.5, tit_, size=12, color="navy")
        lineas_xy(s, x_ - 0.1, 4.22, 3.3, 2.25, [dict(nombre="v", x=list(range(1, 51)), y=ys_, color="azul", ancho=1.5,
                                                      marcador="circulo", tam=4)],
                  leyenda=False, minx=0, maxx=50, miny=0, maxy=ymax_, fmt_x="0", fmt_y="0.0" if ymax_ < 5 else "0",
                  titulo_x="Componente", unidad_x=10, unidad_y=uy_)
    texto(s, 0.55, 6.52, 6.55, 0.45, f"Ningún componente domina: cada uno aporta menos de 1,5 % y los 50 juntos, "
          f"{n(L['pve_pct_acumulada_50'], 1)} %.", size=12, color="muted")
    comp = {c["k"]: c for c in L["componentes"]}
    filas = [("Componente", "Varianza", "Qué separa (valores altos frente a bajos)", "Aviso del ejemplo"),
             ("1 (lsa_00)", f"{n(e[0], 1)} %", "vocabulario común de un aviso de apartamento",
              f"{_negativo(comp[0]['valor_aviso'])} (alto)"),
             ("2 (lsa_01)", f"{n(e[1], 1)} %", "firma de una agente frente a descripción típica",
              _negativo(comp[1]["valor_aviso"])),
             ("3 (lsa_02)", f"{n(e[2], 1)} %", "dos plantillas de agencia, una frente a otra", _negativo(comp[2]["valor_aviso"])),
             ("4 (lsa_03)", f"{n(e[3], 1)} %", "casas y comercio frente a apartamento en edificio",
              f"{_negativo(comp[3]['valor_aviso'])} (apartamento)")]
    texto(s, 7.25, 3.72, 5.55, 0.3, "Los cuatro primeros componentes y el aviso del ejemplo", size=12, color="navy")
    y_fin = _tabla(s, 7.25, 4.04, filas, [1.15, 0.95, 2.25, 1.2], minimo=0.3)
    FIN[id(s)] = y_fin
    notas = ("LSA resume la tabla de pesos. Es el PCA aplicado al texto: busca las direcciones en las que más "
             "difieren los avisos y se queda con 50. Cada una, un componente, es una mezcla de términos que suelen ir juntos: "
             "el primero junta el vocabulario común de un aviso de apartamento; otros separan plantillas de agencias, o casas "
             "y locales frente a apartamentos en edificio. Cada aviso recibe 50 números, su posición en cada componente, y "
             "esas son 50 variables. El aviso del ejemplo tiene 0,27 en el primero, más que el promedio de 0,18, porque es "
             "una descripción típica, y −0,11 en el cuarto: está del lado de apartamento en edificio. ¿Cuánto resume? Con la "
             "varianza explicada y el screeplot, el gráfico A. Los 50 componentes recogen el 12,8 % de las "
             "diferencias entre avisos. El primero aporta casi nada, 0,4 %, porque LSA no resta el promedio y ese componente "
             "recoge el vocabulario que comparten todos; el que más aporta es el segundo, 1,3 %. Es poco porque los avisos "
             "usan palabras muy variadas; por eso además usamos el ridge, que trabaja con la tabla completa. El número 50 lo "
             "fijamos; no hay un criterio objetivo para elegirlo, y la curva no muestra un codo claro. El "
             "criterio de Kaiser no aplica aquí, porque supone variables estandarizadas.")
    return s, notas


def _lda(deck):
    lim = json.load(open(D / "ppt_texto_limpieza.json", encoding="utf-8"))
    top = pd.read_csv(D / "ppt_texto_topicos.csv")
    s = _nueva(deck, "LDA: cada aviso como una mezcla de 10 tópicos (temas)")
    _tres_tarjetas(s,
                   ["En palabras simples: LDA, la asignación latente de Dirichlet, supone que cada aviso mezcla unos pocos "
                    "temas y que cada tema usa más ciertas palabras. Mirando qué palabras aparecen juntas, descubre los "
                    "temas y cuánto de cada tema tiene cada aviso (Blei, Ng y Jordan, 2003)."],
                   [f"1. Insumo: los conteos de las {n(lim['unigramas'], 0)} palabras sueltas, sin pares.",
                    "2. Probamos de 5 a 25 temas; con 10, el modelo predijo mejor las palabras de avisos que no vio (menor "
                    "perplejidad, 327,0); entre 5 y 20 casi no cambia.",
                    "3. Ajuste final con los 48.930 avisos."],
                   ["10 variables (topico_00 a topico_09): la parte del aviso que habla de cada tema; suman 1. El aviso "
                    "del ejemplo pesa 0,98 en el tema 8.",
                    "La prueba habla mucho más de apartamentos en edificio (tema 8) y casi nada de casas (tema 7)."],
                   h=2.05, maximo=13)
    filas = [("Tema", "Nombre que le puso el equipo", "Palabras más probables", "Prueba", "Entrenamiento",
              "Relación con el precio")]
    for _, r in top.iterrows():
        k = int(r["topico"])
        pal = [TILDES.get(p, p) for p in str(r["probables"]).split(", ")[:5]]
        filas.append((str(k), NOMBRE_TEMA[k], ", ".join(pal), f"{n(100 * r['media_chapinero'], 0)} %",
                      f"{n(100 * r['media_resto'], 0)} %", _negativo(r["rho_precio"])))
    y_fin = _tabla(s, 0.55, 3.48, filas, [0.65, 3.3, 3.9, 1.1, 1.5, 1.8], minimo=0.24)
    k8, k7, k5 = (top.set_index("topico").loc[k] for k in (8, 7, 5))
    texto(s, 0.55, y_fin + 0.03, 12.25, 0.48, "Prueba y entrenamiento (fuera de Chapinero): parte del aviso que habla del "
          "tema, en promedio. Relación con el precio: correlación de rangos (Spearman, de −1 a 1) en el entrenamiento; "
          "positiva si a más tema, más precio. Co, id y fr son restos de plantillas.", size=12, color="muted")
    FIN[id(s)] = y_fin
    notas = ("LDA es un modelo de tópicos, o temas. Supone que cada aviso se escribe mezclando unos pocos temas y que cada "
             "tema usa más ciertas palabras; mirando qué palabras aparecen juntas, descubre los temas y la mezcla de cada "
             "aviso. Probamos de 5 a 25 temas y nos quedamos con 10, el que mejor predijo las palabras de avisos que el "
             "modelo no vio, aunque la diferencia es pequeña. Cada aviso recibe 10 números que suman 1: qué parte habla de "
             "cada tema. Los nombres los pusimos nosotros. La tabla muestra que la prueba habla mucho más de apartamentos en "
             "edificio y casi nada de casas, y que el tema de cercanía a comercio es el más ligado a precios bajos: su "
             f"relación con el precio es {_negativo(k5['rho_precio'])}. Ojo: ordenamos las palabras por probabilidad, por eso "
             "apartamento se repite en casi todos los temas; ordenarlas por lift ayudaría a leerlos mejor.")
    return s, notas


def _ridge(deck):
    M = json.load(open(D / "ppt_modelos_texto.json", encoding="utf-8"))
    G = json.load(open(D / "ppt_graficas_clase.json", encoding="utf-8"))
    rid = pd.read_csv(D / "ppt_palabras_ridge.csv")
    dist = pd.read_csv(D / "ppt_texto_chapinero.csv").set_index("termino")
    Rr = M["ridge"]
    mejor = min(G["palabras"], key=lambda r: r["mse"])
    alfas = [r["alfa"] for r in G["palabras"]]
    s = _nueva(deck, "Ridge: un puntaje de precio aprendido de las palabras")
    _tres_tarjetas(s,
                   ["En palabras simples: el ridge da a cada término un número, su coeficiente, que dice cuánto sube o baja "
                    "el precio cuando aparece; el puntaje de un aviso suma los de sus términos.",
                    "Es una regresión penalizada, pariente del LASSO: el LASSO lleva a cero los términos sin poder "
                    "predictivo; el ridge los encoge sin anularlos, para que un término raro no explique un precio por "
                    "casualidad (Gentzkow y otros, 2019)."],
                   ["1. Sin trampa (fuera de pliegue): el entrenamiento se parte en 5 grupos, con textos repetidos juntos; "
                    "el puntaje de cada grupo sale de un ridge que aprendió de los otros 4; el de la prueba, de uno con "
                    "todo el entrenamiento.",
                    f"2. Penalización (λ) fija en {n(mejor['alfa'], 0)}. Con esos 5 grupos (validación cruzada) vimos "
                    f"después que da el menor error entre {n(min(alfas), 2)} y {n(max(alfas), 0)}.",
                    "3. Al probar el modelo, se recalcula sin los avisos de prueba."],
                   ["1 variable: el precio que sugieren las palabras, en logaritmo.",
                    f"Sin ver el precio de cada aviso, se parece mucho al precio real: correlación de {n(Rr['corr_oof'], 2)} "
                    "con el logaritmo del precio (1 sería acierto perfecto; 0, ninguna relación).",
                    f"Aviso del ejemplo: {n(Rr['puntaje_aviso_millones'], 0)} millones; pedía 749."], h=2.58, maximo=13)
    sube = rid[rid["lado"] == "sube"].sort_values("coef", ascending=False).head(8)
    baja = rid[rid["lado"] == "baja"].sort_values("coef").head(8)
    texto(s, 0.55, 3.96, 8.1, 0.3, "Términos que más suben y más bajan el puntaje, entre los que aparecen en 300 avisos o más",
          size=12, color="navy")
    for x0, d_, col, tit in ((0.45, sube, "azul", "Suben el puntaje"), (4.5, baja, "ambar", "Bajan el puntaje")):
        texto(s, x0 + 0.1, 4.22, 3.9, 0.28, tit, size=12, color="muted")
        ch = barras(s, x0, 4.44, 4.05, 1.95, [LEG_RIDGE.get(t, t) for t in d_["termino"]],
                    [("coef", [abs(v) for v in d_["coef"]], col)], minimo=0, maximo=2.4, gap=35, grid=False)
        etiquetas_puntos(ch, [[_negativo(v) for v in d_["coef"]]])
    texto(s, 0.55, 6.42, 8.1, 0.55, "Coeficiente sin unidad: importan el signo y el orden. Carolina junta un barrio y la firma "
          "de una agente; parqueadero, baño y el número 1 suelen indicar uno solo.", size=12, color="muted")
    filas = [("Barrio", "Prueba", "Entrenamiento, fuera de Chapinero")]
    for t_, nombre in (("rosal", "Rosales"), ("chic nort", "Chicó Norte"), ("virrey", "Virrey"), ("parqu 93", "Parque 93"),
                       ("chic reserv", "Chicó Reservado"), ("cabrer", "Cabrera")):
        filas.append((nombre, f"{n(dist.loc[t_, 'pct_chapinero'], 1)} %", f"{n(dist.loc[t_, 'pct_resto'], 2)} %"))
    texto(s, 8.85, 3.96, 3.95, 0.3, "Barrios de Chapinero: % de avisos que los nombran", size=12, color="navy")
    y_fin = _tabla(s, 8.85, 4.26, filas, [1.55, 0.8, 1.6], minimo=0.25)
    texto(s, 8.85, y_fin + 0.04, 3.95, 0.7, "Los nombran muy pocos avisos de entrenamiento: su coeficiente se aprende "
          "de muy pocos ejemplos.", size=12, color="muted")
    FIN[id(s)] = y_fin
    notas = ("El ridge es el modelo del texto que más aporta. Es una regresión del logaritmo del precio sobre los 44.888 pesos "
             "de TF-IDF: a cada término le da un coeficiente que dice cuánto sube o baja el precio cuando aparece, y el "
             "puntaje de un aviso suma los de sus términos. Es la misma idea del LASSO que usan Gentzkow y Shapiro con "
             "los discursos del Congreso, con una diferencia: el LASSO lleva a cero los términos que no predicen; el ridge "
             "los encoge sin anularlos, para que un término raro no explique un precio por casualidad. La fuerza de esa "
             "penalización, lambda, la dejamos fija en 1; después comprobamos con validación cruzada que 1 da el menor error "
             "entre 0,03 y 100. Lo más importante es cómo se calcula: sin trampa, fuera de pliegue. Partimos el entrenamiento "
             "en 5 grupos, con los textos repetidos siempre juntos, y el puntaje de cada aviso lo calcula un ridge que no vio "
             "su precio. Así el puntaje tiene correlación 0,81 con el precio sin hacer trampa. Los coeficientes se leen bien: "
             "3 o 4 parqueaderos, casa, ascensor privado o Chicó suben el puntaje; apartaestudio, un solo parqueadero o "
             "Cedritos lo bajan. Carolina mezcla el barrio La Carolina y la firma de una agente. Y a la derecha: los barrios "
             "de Chapinero los nombran muy pocos avisos del entrenamiento, así que su coeficiente se aprende de muy pocos "
             "ejemplos. A Rosales y Cabrera el ridge sí les da una prima alta; a Chicó Norte o al Parque 93, casi nada.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# 17. Fuente 2c: temas, componentes y puntaje (cierre del texto)
# ------------------------------------------------------------------------------------------------------------------
def _fuente2c(deck):
    V = json.load(open(D / "ppt_variables.json", encoding="utf-8"))["variables"]
    top = pd.read_csv(D / "ppt_texto_topicos.csv").set_index("topico")
    filas_t = []
    for k in range(10):
        pal = [TILDES.get(p, p) for p in str(top.loc[k, "probables"]).split(", ")[:3]]
        filas_t.append((f"Tema {k}. {NOMBRE_TEMA[k]}", "palabras más probables: " + ", ".join(pal),
                        _pct(100 * V[f"topico_{k:02d}"]["promedio"])))
    pr = V["puntaje_ridge"]["promedio"]
    grupos = [("Conteos de palabras: LDA con 10 temas", filas_t),
              ("Tabla TF-IDF", [("50 componentes (LSA)", "50 números que resumen las palabras del aviso", "sin unidad")]),
              ("Tabla TF-IDF y precio del entrenamiento", [
                  ("Puntaje de precio del texto (ridge)", "precio que sugieren las palabras del aviso, calculado sin usar su propio "
                                                "precio", f"{n(pr, 2)}: unos {n(math.exp(pr), 0)} millones")])]
    s = _lamina_fuente(deck, "Fuente 2c. Texto del aviso: temas, resumen en 50 números y puntaje de precio",
                       "Las 61 variables de los modelos del texto, hechas con la tabla de 48.930 avisos por 44.888 términos; "
                       "todas tienen dato.", grupos,
                       "Promedio de los temas: parte del aviso que habla de cada tema; los 10 temas de un aviso suman 100 %. "
                       "El puntaje se guarda en logaritmo; su promedio equivale a unos 610 millones.", 1, 61, 105,
                       encabezado=("Dato original", "Variable", "Qué mide", "Promedio"), anchos=[3.0, 3.95, 3.95, 1.35],
                       minimo=0.25)
    notas = ("Cerramos el texto con las 61 variables que salen de los modelos. Diez son los temas de LDA: cada aviso recibe "
             "qué parte habla de cada tema, y las palabras dicen de qué trata; por ejemplo, el tema 0 junta Chapinero y "
             "Carolina, y el tema 8, el más común, describe apartamentos con servicios del edificio. Cincuenta son los "
             "componentes de LSA, que resumen las palabras del aviso en 50 números sin unidad. Y una es el puntaje de precio "
             "del ridge, calculado sin usar el precio del propio aviso: es la variable de texto que más aporta al modelo. "
             "Con esto llevamos 105 de las 162 variables.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# 18 a 22. OpenStreetMap, satélite y Catastro
# ------------------------------------------------------------------------------------------------------------------
def _externas(deck):
    R7 = json.load(open(D / "ppt_variables.json", encoding="utf-8"))
    V = R7["variables"]
    osm_o = {**R7["osm_objetos"], **R7.get("osm_extra_objetos", {})}
    total_osm = sum(R7["osm_objetos"].values()) + sum(R7.get("osm_extra_objetos", {}).values())
    F = json.load(open(D / "ppt_fuentes.json", encoding="utf-8"))["osm_conteos"]

    def prom(c, d=1, escala="original", factor=1.0):
        e = V[c]
        if escala == "geometrica":
            return math.exp(e["promedio"]) * factor
        if escala == "original" and e["promedio_original"] is not None:
            return e["promedio_original"] * factor
        return e["promedio"] * factor

    def fv(c, nombre, como, d=1, **k):
        return nombre, como, n(prom(c, d, **k), d)

    def fv6(c, nombre, como, d=1, si=False, **k):
        val = _si(prom(c, d, **k)) if si else n(prom(c, d, **k), d)
        return nombre, como, _pct(V[c]["entrenamiento"]), _pct(V[c]["prueba"]), val

    def o(cat):
        return n(osm_o[cat], 0)

    enc4 = ("Dato original (lugares en el mapa)", "Variable", "Cómo se construye", "Promedio")
    an4 = [3.7, 3.0, 4.2, 1.35]
    base_osm = (f"Base: OpenStreetMap, el mapa abierto de internet: unos {n(round(total_osm, -3), 0)} lugares de Bogotá y "
                "las áreas protegidas. Todas tienen dato en todos los avisos.")
    cerca = "cuántos hay a {} o menos del aviso"
    lam = []
    acum = 105
    paginas = [
        ("Fuente 3a. OpenStreetMap: comercio y transporte alrededor del aviso", [
            (f"restaurantes ({o('restaurante')})", [
                fv("ln_n_restaurante_300m", "Restaurantes a 300 m", cerca.format("300 m")),
                fv("ln_n_restaurantes_500m", "Restaurantes a 500 m", cerca.format("500 m")),
                fv("ln_n_restaurante_1km", "Restaurantes a 1 km", cerca.format("1 km")),
                fv("ln_n_restaurantes_2km", "Restaurantes a 2 km", cerca.format("2 km"))]),
            (f"bares, cafés y discotecas ({o('bar_cafe')})", [
                fv("ln_n_bar_cafe_300m", "Bares y cafés a 300 m", cerca.format("300 m")),
                fv("ln_n_bares_cafes_500m", "Bares y cafés a 500 m", cerca.format("500 m")),
                fv("ln_n_bar_cafe_1km", "Bares y cafés a 1 km", cerca.format("1 km"))]),
            (f"supermercados ({o('supermercado')})", [
                fv("ln_n_supermercado_300m", "Supermercados a 300 m", cerca.format("300 m")),
                fv("ln_n_supermercados_1km", "Supermercados a 1 km", cerca.format("1 km")),
                fv("ln_n_supermercado_1km", "Supermercados a 1 km (repetida)", "copia exacta de la fila anterior que quedó "
                                                                              "por error; no agrega información")]),
            (f"paraderos de bus ({o('paradero')})", [
                fv("ln_n_paradero_300m", "Paraderos a 300 m", cerca.format("300 m")),
                fv("ln_n_paraderos_500m", "Paraderos a 500 m", cerca.format("500 m")),
                fv("ln_n_paradero_1km", "Paraderos a 1 km", cerca.format("1 km"))])],
         "Promedio: número de lugares alrededor de un aviso. El modelo usa estas cuentas en escala comprimida (logaritmo de 1 "
         "más la cuenta): pasar de 1 a 2 pesa más que pasar de 100 a 101.",
         "La tercera fuente es OpenStreetMap, el mapa abierto de internet: 39 variables en tres láminas. De Bogotá tomamos "
         "unos 29.000 lugares, como restaurantes o paraderos, y alrededor de cada aviso contamos cuántos hay a distintas "
         "distancias, de 300 metros a 2 kilómetros. El modelo usa estas cuentas en escala comprimida: pasar de uno a dos "
         "restaurantes cerca pesa más que pasar de cien a ciento uno. Una aclaración: supermercados a 1 kilómetro quedó dos "
         "veces, con la misma cuenta; es una copia que no agrega información, y lo decimos."),
        ("Fuente 3b. OpenStreetMap: colegios, TransMilenio, oficinas y embajadas", [
            (f"colegios ({o('colegio')})", [
                fv("ln_n_colegio_300m", "Colegios a 300 m", cerca.format("300 m")),
                fv("ln_n_colegios_1km", "Colegios a 1 km", cerca.format("1 km")),
                fv("ln_n_colegio_1km", "Colegios a 1 km (repetida)", "copia exacta de la fila anterior que quedó por error; "
                                                                     "no agrega información")]),
            (f"estaciones de transporte ({o('estacion')})", [
                fv("ln_dist_transmilenio_m", "Distancia a TransMilenio (m)", "a la estación más cercana; no cuenta metro, "
                                                                             "tren ni SITP", 0),
                fv("ln_n_transmilenio_1km", "TransMilenio a 1 km", "estaciones a 1 km o menos")]),
            (f"oficinas ({o('oficina')}) y edificios de oficinas o comercio ({o('edificio_oficinas')})", [
                fv("ln_n_oficina_500m", "Oficinas a 500 m", cerca.format("500 m")),
                fv("ln_n_oficina_1000m", "Oficinas a 1 km", cerca.format("1 km")),
                fv("ln_n_edificio_oficinas_500m", "Edificios de oficinas o comercio a 500 m", cerca.format("500 m"))]),
            (f"hoteles ({o('hotel')}), bancos ({o('banco')}) y gimnasios ({o('gimnasio')})", [
                fv("ln_n_hotel_1000m", "Hoteles a 1 km", cerca.format("1 km")),
                fv("ln_n_banco_500m", "Bancos a 500 m", cerca.format("500 m")),
                fv("ln_n_gimnasio_1000m", "Gimnasios a 1 km", cerca.format("1 km"))]),
            (f"embajadas ({o('embajada')})", [
                fv("ln_n_embajada_1000m", "Embajadas a 1 km", cerca.format("1 km")),
                fv("ln_dist_embajada_m", "Distancia a una embajada (m)", "a la embajada más cercana", 0)])],
         f"Oficinas, hoteles y embajadas se concentran en Chapinero: a 1 km de un aviso de prueba hay, en mediana, "
         f"{n(F['n_embajada_1000m']['prueba'], 0)} embajadas, frente a {n(F['n_embajada_1000m']['entrenamiento_resto'], 0)} "
         "en el entrenamiento fuera de Chapinero. El modelo usa cuentas y distancias en logaritmo.",
         "Siguen colegios, TransMilenio, oficinas, hoteles, bancos, gimnasios y embajadas. Estas últimas las agregamos "
         "pensando en Chapinero, donde se concentran: a un kilómetro de un aviso de prueba hay muchas más oficinas, hoteles "
         "y embajadas que en el resto de Bogotá. Para TransMilenio solo contamos sus estaciones, no metro, tren ni SITP. "
         "Colegios a 1 kilómetro también quedó repetida."),
        ("Fuente 3c. OpenStreetMap: distancias a parques, vías, servicios y referencias", [
            ("parques dibujados con su área", [
                fv("ln_dist_parque_m", "Distancia al parque (m)", "al parque más cercano", 0),
                fv("ln_area_parque_m2", "Área de ese parque (m²)", "área del parque más cercano", 0),
                fv("ln_dist_parque_grande_m", "Distancia a un parque grande (m)", "al parque más cercano de 1 hectárea o "
                                                                                  "más", 0),
                fv("ln_area_parques_500m_ha", "Parques a 500 m (ha)", "área de los parques con centro a 500 m o menos")]),
            (f"vías principales: autopistas y avenidas ({o('via_principal')} tramos)", [
                fv("ln_dist_via_principal_m", "Distancia a vía principal (m)", "al tramo más cercano", 0)]),
            (f"universidades ({o('universidad')}), centros comerciales ({o('centro_comercial')}), hospitales y clínicas "
             f"({o('hospital')}) y policía ({o('policia')})", [
                fv("ln_dist_universidad_m", "Distancia a universidad (m)", "a la más cercana", 0),
                fv("ln_dist_centro_comercial_m", "Distancia a centro comercial (m)", "al más cercano", 0),
                fv("ln_dist_hospital_m", "Distancia a hospital (m)", "al más cercano", 0),
                fv("ln_dist_policia_m", "Distancia a policía (m)", "a la más cercana", 0)]),
            ("áreas protegidas de 100 hectáreas o más", [
                fv("ln_dist_area_protegida_m", "Distancia a un área protegida grande (m)", "a la más cercana: los cerros "
                                                                                           "orientales o un humedal "
                                                                                           "grande", 0)]),
            ("tres lugares de referencia", [
                fv("ln_dist_p93_m", "Al Parque 93 (m)", "en línea recta hasta ese lugar", 0),
                fv("ln_dist_zonat_m", "A la Zona T (m)", "en línea recta hasta ese lugar", 0),
                fv("ln_dist_centro_internacional_m", "Al Centro Internacional (m)", "en línea recta hasta ese lugar", 0)])],
         "Promedio en metros, m² o hectáreas (1 ha = 10.000 m²). El modelo usa las distancias en escala comprimida "
         "(logaritmo): pasar de 100 a 200 m pesa más que pasar de 5 a 5,1 km.",
         "Por último, distancias: al parque más cercano y a uno grande, a la vía principal, a universidades, centros "
         "comerciales, hospitales y policía, a un área protegida grande, que en Bogotá son los cerros orientales o un humedal "
         "grande, y a tres lugares de referencia: el Parque 93, la Zona T y el Centro Internacional. Las distancias van en "
         "metros y el modelo usa su logaritmo, porque pasar de 100 a 200 metros pesa más que pasar de 5 a 5,1 kilómetros."),
    ]
    for titulo, grupos, pie, nota_ in paginas:
        en_tabla = sum(len(v) for _, v in grupos)
        acum += en_tabla
        s = _lamina_fuente(deck, titulo, base_osm, grupos, pie, 2, en_tabla, acum, encabezado=enc4, anchos=an4,
                           minimo=0.255)
        lam.append((s, nota_))

    grupos = [
        ("altura del terreno de la NASA (SRTM), en cuadros de 30 m", [
            fv("elevacion_m", "Elevación (m)", "altura sobre el nivel del mar en el punto del aviso", 0),
            fv("pendiente_grados", "Pendiente (grados)", "qué tan inclinado es el terreno, con el mismo mapa"),
            fv("elevacion_relativa_1km", "Elevación relativa (m)", "cuántos metros más alto está el aviso que la mediana de "
                                                                   "las ubicaciones a 1 km")]),
        ("imágenes del satélite europeo Sentinel-2, 2020 y 2021, sin nubes", [
            fv("ndvi_200m", "Vegetación (índice NDVI)", "qué tan verde se ve desde el satélite a 200 m del aviso: cerca de "
                                                        "0, cemento o suelo; cerca de 1, vegetación densa", 2)]),
        ("luces de noche del satélite VIIRS, 2020", [
            fv("luces_nocturnas", "Luces nocturnas (brillo)", "brillo de noche visto desde el satélite, promedio de los "
                                                              "meses de 2020; más alto, más iluminado")]),
        ("construcciones y población de la Comisión Europea (GHSL), en cuadros de 100 m", [
            fv("altura_construida_m", "Altura construida (m)", "altura media de lo construido, en 2018"),
            fv("ln_poblacion_100m", "Población (personas)", "personas que viven en el cuadro de 100 m del aviso, en 2020",
               0)])]
    s = _lamina_fuente(deck, "Fuente 4. Imágenes de satélite: 7 variables del terreno y de lo construido",
                       "Base: cinco productos de satélite, bajados con Google Earth Engine y leídos en el punto de cada "
                       "aviso. Todas tienen dato.",
                       grupos, "La población entra al modelo en escala comprimida (logaritmo); su promedio va en personas. "
                               "El brillo de las luces no tiene una unidad sencilla: sirve para comparar avisos.", 3, 7, 151,
                       encabezado=("Dato original", "Variable", "Cómo se construye", "Promedio"),
                       anchos=[3.7, 2.8, 4.4, 1.35])
    tarjeta(s, 0.55, 4.95, 12.25, 0.9, "", [
        "Google Earth Engine es la plataforma de Google para bajar y procesar imágenes de satélite. La elevación relativa "
        "dice si el aviso está más alto que sus vecinos, como pasa al pie de los cerros."], tipo="definicion", maximo=14)
    lam.append((s, "La cuarta fuente son las imágenes de satélite, que bajamos con Google Earth Engine: cinco productos y "
                   "siete variables. Del mapa de altura de la NASA salen la elevación, la pendiente y la elevación relativa, "
                   "que dice cuántos metros más alto está el aviso que la mediana de las ubicaciones a un kilómetro, como "
                   "pasa al pie de los cerros. Sentinel-2 da la vegetación: un índice que va de cerca de cero en el cemento "
                   "a cerca de uno en la vegetación densa. VIIRS da el brillo de las luces de noche, y la Comisión Europea, "
                   "la altura de lo construido y la población. Todas tienen dato en todos los avisos."))

    grupos = [
        ("valor de referencia del suelo, por manzana y año", [
            fv6("ln_vref_manzana", "Suelo, manzana (millones por m²)", "valor mediano del terreno por m²", 2,
                escala="geometrica", factor=1e-6),
            fv6("ln_vref_300m", "Suelo a 300 m (millones por m²)", "mediana de las manzanas a 300 m o menos", 2,
                escala="geometrica", factor=1e-6)]),
        ("avalúo catastral de los predios en propiedad horizontal", [
            fv6("ln_avaluo_ph_manzana", "Avalúo, manzana (millones por m²)", "avalúo por m² de sus apartamentos", 2,
                escala="geometrica", factor=1e-6),
            fv6("ln_avaluo_ph_300m", "Avalúo a 300 m (millones por m²)", "mediana entre las manzanas a 300 m", 2,
                escala="geometrica", factor=1e-6)]),
        ("altura de las manzanas, en pisos", [
            fv6("pisos_manzana", "Pisos en la manzana", "número de pisos de la manzana, como lo publica Catastro",
                escala="modelo"),
            fv6("pisos_300m", "Pisos a 300 m", "mediana entre las manzanas a 300 m", escala="modelo")]),
        ("estratificación por manzana", [
            fv6("estrato_manzana", "Estrato de la manzana", "de 1 a 6; 0 si no es residencial", escala="modelo"),
            fv6("estrato_medio_300m", "Estrato medio a 300 m", "promedio entre las manzanas a 300 m", escala="modelo"),
            fv6("prop_estrato6_300m", "Estrato 6 a 300 m", "% de manzanas de estrato 6 a 300 m", si=True,
                escala="modelo")]),
        ("geometría de las manzanas, 2021", [
            fv6("n_manzanas_300m", "Manzanas a 300 m", "número de manzanas con centro a 300 m o menos", 0, escala="modelo"),
            fv6("dist_manzana_m", "Distancia a la manzana (m)", "0 si cae dentro; si no, la más cercana (hasta 100 m)",
                escala="modelo")])]
    s = _lamina_fuente(deck, "Fuente 5. Catastro: 11 variables del valor del suelo y de la manzana",
                       "Base: mapas de Catastro Bogotá por manzana, de 2019 a 2021; cada aviso toma su manzana y el año del "
                       "aviso.", grupos,
                       "Suelo: lo que vale el terreno, sin la construcción. Avalúo: valor que Catastro da a los "
                       "apartamentos; falta donde no hay. Ambos en millones de pesos por m², con su valor típico (promedio "
                       "geométrico); el modelo usa su logaritmo.", 4, 11, 162,
                       encabezado=("Dato original", "Variable", "Cómo se construye", "Con dato, entrenamiento",
                                   "Con dato, prueba", "Promedio o %"), anchos=[2.45, 2.8, 3.6, 1.25, 1.05, 1.1])
    lam.append((s, "La quinta fuente es Catastro, el registro oficial de predios de Bogotá, con mapas por manzana de 2019 a "
                   "2021. Cada aviso toma la manzana que lo contiene, o la más cercana si cae en la calle, y el valor del año "
                   "del aviso. De ahí salen el valor del suelo, que es lo que vale el terreno sin la construcción; el avalúo "
                   "de los apartamentos; los pisos y el estrato, en la manzana y en un radio de 300 metros. El avalúo de la "
                   "manzana es el que más falta, porque no todas las manzanas tienen apartamentos. Con esto completamos las "
                   "162 variables."))
    return lam


# ------------------------------------------------------------------------------------------------------------------
# 23. 162 variables y precios hedónicos
# ------------------------------------------------------------------------------------------------------------------
def _hedonicas(deck):
    rid = pd.read_csv(D / "ppt_palabras_ridge.csv")
    s = _nueva(deck, "162 variables: el precio refleja la vivienda y su entorno (precios hedónicos)")
    texto(s, 0.55, 1.3, 7.4, 0.5, "P = f(c1, c2, …, cn) + u", size=24, color="navy")
    texto(s, 0.55, 1.85, 7.4, 1.25, ["P: precio del aviso. c: sus características (área, baños, entorno). f: la regla que las "
                                     "convierte en precio; el modelo la aprende de los datos. u: lo que las características "
                                     "no explican (Rosen, 1974).",
                                     "La ubicación cuenta dos veces: como variables (esta tabla) y porque los vecinos se "
                                     "parecen; por eso el modelo se prueba en un lugar que no vio (Etapa 3)."],
          size=13, color="tinta", espacio=6)
    filas = [("Fuente", "Variables", "Qué mide"),
             ("Avisos de Properati", "21", "la vivienda (área, baños, alcobas, parqueaderos, tipo), la fecha, la ubicación y "
                                           "los avisos vecinos"),
             ("Texto del aviso", "84", "19 indicadores de sí o no, 4 cifras escritas, 10 temas (LDA), 50 componentes que "
                                       "resumen las palabras (LSA) y un puntaje de precio (ridge)"),
             ("OpenStreetMap", "39", "comercio, transporte, colegios, oficinas, embajadas, parques, vías y lugares de "
                                     "referencia"),
             ("Imágenes de satélite", "7", "terreno, vegetación, luces de noche, altura construida y población"),
             ("Catastro", "11", "valor del suelo, avalúo, pisos y estrato de la manzana y a 300 m"),
             ("Total", "162", "57 de fuentes externas (mapa, satélite y Catastro) y 84 de texto")]
    y_fin = _tabla(s, 0.55, 3.2, filas, [2.1, 1.05, 4.25], minimo=0.3)
    t = s.shapes[-1].table
    for c_ in range(3):
        t.cell(len(filas) - 1, c_).fill.solid()
        t.cell(len(filas) - 1, c_).fill.fore_color.rgb = rgb("E8EEF5")
    texto(s, 8.25, 1.35, 4.6, 1.4, "El puntaje de precio del texto: el precio que sugieren las palabras del aviso, aprendido "
          "de otros avisos. Ni su precio ni el de sus copias entran a su puntaje.", size=14, color="navy")
    sube = rid[rid["lado"] == "sube"].sort_values("coef", ascending=False).head(8)
    baja = rid[rid["lado"] == "baja"].sort_values("coef").head(8)
    texto(s, 8.25, 2.8, 4.6, 0.36, "Términos que más suben el puntaje", size=14, color="azul")
    y_ = chips(s, 8.25, 3.2, 4.6, [LEG_RIDGE.get(t_, t_) for t_ in sube["termino"]], relleno="azul_claro", size=13,
               alto=0.36, sep=0.08)
    texto(s, 8.25, y_ + 0.1, 4.6, 0.36, "Términos que más lo bajan", size=14, color="ambar")
    y2 = chips(s, 8.25, y_ + 0.5, 4.6, [LEG_RIDGE.get(t_, t_) for t_ in baja["termino"]], relleno="ambar_claro", size=13,
               alto=0.36, sep=0.08)
    texto(s, 8.25, y2 + 0.1, 4.6, 0.6, "Los mismos de la lámina del ridge: los 8 de cada lado entre los términos que "
          "aparecen en 300 avisos o más.", size=12, color="muted")
    FIN[id(s)] = max(y_fin, y2 + 0.7)
    notas = ("Cerramos los datos con la idea que los une: los precios hedónicos de Rosen. El precio de una vivienda refleja lo "
             "que tiene: su área, sus baños, su entorno. La regla que convierte esas características en precio, la f, no la "
             "imponemos: el modelo la aprende de los datos; la u es lo que las características no explican. Por eso juntamos "
             "162 variables de cinco fuentes: 21 de los avisos, 84 del texto, 39 de OpenStreetMap, 7 de satélite y 11 de "
             "Catastro. El taller pide al menos cuatro predictores externos y cuatro de texto: aquí hay 57 de fuentes "
             "externas, entre mapa, satélite y Catastro, y 84 de texto. La ubicación cuenta dos veces: como variables y "
             "porque los vecinos se parecen, y por eso en la Etapa 3 probamos el modelo en un lugar que no vio. A la derecha, "
             "el puntaje del texto, con los mismos términos de la lámina del ridge. Varias palabras son barrios, así que el "
             "puntaje también mide ubicación: es útil, pero Ash y Hansen (2023) advierten que un predictor así puede fallar "
             "en un lugar nuevo, y los barrios de Chapinero los nombran muy pocos avisos del entrenamiento.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
def etapa1(deck):
    """Crea las láminas de la Etapa 1 en el orden final y devuelve [(lámina, notas), ...]."""
    salida = [_problema(deck), _ruta(deck), _mapa_precio(deck)]
    salida += _fuente1(deck)
    salida += [_apertura_clase(deck), _pasos_modelos(deck), _aviso_real(deck), _beto(deck), _listas(deck)]
    salida += _fuente2ab(deck)
    salida += [_tfidf(deck), _lsa(deck), _lda(deck), _ridge(deck), _fuente2c(deck)]
    salida += _externas(deck)
    salida.append(_hedonicas(deck))
    for s, nt in salida:
        deck.notas(s, nt)
    return salida
