"""21 · Construye 03_slides/best_equipo_02.pptx (Best Model Deep Dive) con python-pptx.

Cifras: 00_data/processed/laminas/ (scripts 12 a 19); imágenes: 02_outputs/figures/laminas/ (script 20).
Correcciones aplicadas: revision/informes/revisor_cifras_requisitos_opus.md, economista_audiencia_opus.md (versión Beamer),
cifras_v2_opus.md y disenio_v2_opus.md (versión PowerPoint v2), y requisitos_v3_opus.md, cifras_v3_opus.md y
disenio_v3_opus.md (v3). Las láminas 6 y 7 y la tabla de familias del apéndice vienen de laminas_comunes.py.

Antes document/presentaciones/construir_best_ppt.py. Las referencias a revision/informes/ son a los informes de revisión de
las láminas, que no forman parte del repositorio. Exportar a PDF: script 23 (opcional, Windows con PowerPoint).
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import json
import os
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

from functions.laminas_comunes import (chapinero_otro_mercado, dependencia_espacial, error_por_familia, texto_a_matriz,
                             texto_apendice, texto_barrios_topicos, variables_hedonicas)
from functions.ppt_lib import (Deck, _ejes, _estilo_grafico, barras, circulo, etiquetas_puntos, forma, imagen, linea, lineas_xy,
                     nodo, ocultar_leyenda, rgb, tabla, tarjeta, texto, waffle)

from functions.laminas_entorno import DATOS as D, IMG, SALIDA_PPT as OUT  # noqa: E402  (SALIDA_PPT: otra carpeta de salida)
MENOS = '0;"−"0'


def n(x, d=1, signo=False):
    """Número con coma decimal, punto de miles y signo menos tipográfico."""
    s = f"{abs(x):,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if x < 0 and float(f"{abs(x):.{d}f}") != 0:
        return "−" + s
    return ("+" + s) if signo and float(f"{abs(x):.{d}f}") != 0 else s


def leer(nombre):
    return json.load(open(D / nombre, encoding="utf-8"))


# ------------------------------------------------------------------------------------------------------------------
# Datos
# ------------------------------------------------------------------------------------------------------------------
clas = pd.read_csv(D / "clasificacion.csv")
env = pd.read_csv(D / "envios_equipo.csv")
v2 = leer("ppt_varios2.json")
v1 = leer("ppt_varios.json")
desc = leer("ppt_descripcion.json")
bo = leer("ppt_bootstrap2.json")
tres = leer("ppt_tres_avisos.json")
curva = pd.read_csv(D / "ppt_curva_aprendizaje.csv")
hist = pd.read_csv(D / "ppt_histograma_precios.csv")
cal = pd.read_csv(D / "ppt_calibracion2.csv")
lados = pd.read_csv(D / "ppt_lados.csv")
lq = pd.read_csv(D / "ppt_lados_quintil.csv")
comp = pd.read_csv(D / "ppt_compras.csv")
gp = pd.read_csv(D / "ppt_ganancia_precision.csv")
reg = pd.read_csv(D / "ppt_regla_rango2.csv").set_index("cat")
shap_imp = pd.read_csv(D / "shap_importancia.csv", index_col=0).iloc[:, 0]
shap_blq = pd.read_csv(D / "shap_bloques.csv", index_col=0).iloc[:, 0]
falt = pd.read_csv(D / "ppt_faltantes.csv").set_index("variable")
cand = pd.read_parquet(D / "ppt_texto_candidatos.parquet").set_index("property_id")
arbol = leer("ppt_arbol2.json")["arbol"]

priv = dict(zip(env["archivo"], env["privado"]))
publ = dict(zip(env["archivo"], env["publico"]))
P01, P02, P03 = priv["envio_01_lightgbm_con_catastro"], priv["envio_02_lightgbm_con_catastro_x108"], priv["envio_03_mezcla_lightgbm_catboost"]
U01, U02, U03 = publ["envio_01_lightgbm_con_catastro"], publ["envio_02_lightgbm_con_catastro_x108"], publ["envio_03_mezcla_lightgbm_catboost"]
PREVIO_PRIV = env[env["orden"] < 25]["privado"].min()
assert abs(PREVIO_PRIV - 177.18) < 0.01
MAE_CH, IC_CH = bo["lightgbm"]["mae"], (bo["lightgbm"]["ic_bajo"], bo["lightgbm"]["ic_alto"])
lado_ch = lados[(lados["esquema"] == "Chapinero fuera") & (lados["version"] == "Con ajuste x1,08")].iloc[0]
lado_kn = lados[(lados["esquema"] == "kNNDM") & (lados["version"] == "Con ajuste x1,08")].iloc[0]
lado_sin = lados[(lados["esquema"] == "Chapinero fuera") & (lados["version"] == "Sin ajuste")].iloc[0]


def compra(m, dd):
    return comp[(comp["validacion"] == "Chapinero fuera") & np.isclose(comp["descuento"], dd) & np.isclose(comp["margen"], m)].iloc[0]


c10, c20 = compra(0.10, 0.10), compra(0.20, 0.10)
pron = v2["pronostico"]
dch, dre = v2["desc_Chapinero"], v2["desc_Resto de Bogotá"]
cch, cre = v2["catastro_Chapinero"], v2["catastro_Resto de Bogotá"]
rp = v2["regla_prueba"]
ERR_CH = lado_ch["error_abs_mediano"]


def equilibrio(val, dd):
    """Error mediano en el que la ganancia por oferta (margen de 10 %) cruza el cero, por interpolación lineal."""
    t = gp[(gp["validacion"] == val) & np.isclose(gp["descuento"], dd)].sort_values("error_mediano_pct")
    x, y = t["error_mediano_pct"].to_numpy(), t["ganancia_por_oferta"].to_numpy()
    i = int(np.where((y[:-1] > 0) & (y[1:] <= 0))[0][0])
    return x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1])


EQ10, EQ5 = equilibrio("Chapinero fuera", 0.10), equilibrio("Chapinero fuera", 0.05)
print("equilibrio con margen de 10 %:", round(EQ10, 2), round(EQ5, 2))
NOMBRE_EQUIPO = {"Grupo 6": "Grupo 06", "Grupo_02": "Grupo 02", "Grupo_01": "Grupo 01", "Pishcke Blinders": "Pishcke Blinders",
                 "Grupo_05": "Grupo 05", "GRUPO_03": "Grupo 03"}
deck = Deck("Best Model Deep Dive")
NOTAS = {}
E4 = "Etapa 4. Pronóstico y negocio"


def nota(t):
    NOTAS[len(deck.pies) - 1] = t


# ------------------------------------------------------------------------------------------------------------------
# 1. Portada
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva(oscura=True)
texto(s, 0.7, 0.9, 6.3, 0.4, "Problem Set 3. Making Money with ML?", size=16, color="cielo")
texto(s, 0.7, 1.45, 6.4, 2.4, "¿Se puede confiar en un modelo para comprar vivienda en Chapinero?", size=38, color="blanco")
texto(s, 0.7, 3.85, 6.3, 0.9, "Best Model Deep Dive: LightGBM con Catastro y texto, validado como el reto de Kaggle",
      size=18, color="cielo")
linea(s, 0.7, 4.95, 3.2, 4.95, color="naranja", ancho=2.5)
texto(s, 0.7, 5.15, 6.3, 1.0, ["Equipo 02: Jhon Sebastián Vela Salcedo, Harold Santiago Aguilera Mulfo y César Augusto "
                               "Vallejo Quiroga", "25 de noviembre de 2026"], size=15, color="blanco", espacio=8)
imagen(s, IMG / "portada_mapa.png", 7.45, 0.45, h=5.95,
       alt="Mapa de Bogotá: avisos de entrenamiento en gris y de prueba en naranja, con el borde de Chapinero")
texto(s, 7.45, 6.45, 5.4, 0.5, "Gris: avisos de entrenamiento. Naranja: avisos de prueba (94 % en la localidad de "
      "Chapinero).", size=12, color="cielo")
nota("Pregunta de negocio: una empresa que compra vivienda quiere usar un modelo para decidir cuánto ofrecer en Chapinero. "
     "Vamos a mostrar qué tan bien predice nuestro mejor modelo, cómo funciona y, sobre todo, dónde falla y qué le cuesta "
     "eso a la empresa.")

# ------------------------------------------------------------------------------------------------------------------
# 2. Resultado
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Resultado: ordena bien Chapinero, pero no basta para ofertar solo", etiqueta="Resultado", seccion="Resultado")
stats2 = [("0,87", "correlación de rangos entre predicción y precio en los 307 avisos de Chapinero, predichos sin haberlos "
                    "visto"),
          (f"{n(ERR_CH, 1)} %", f"de error absoluto mediano, en % del precio real, en esos 307 avisos; el "
                                f"{n(lado_ch['sobre_10'], 0)} % queda más de 10 % arriba"),
          (n(P02), "millones de error absoluto medio en Kaggle (tabla privada): segundo de seis equipos")]
for i_, (v_, e_) in enumerate(stats2):
    y0 = 1.45 + i_ * 1.82
    texto(s, 0.55, y0, 3.7, 0.8, v_, size=44, color="azul")
    texto(s, 0.55, y0 + 0.8, 3.7, 0.95, e_, size=14, color="muted")
cats = [NOMBRE_EQUIPO.get(e, e) for e in clas["equipo"]]
cols = ["azul" if e == "Grupo_02" else "gris_claro" for e in clas["equipo"]]
texto(s, 4.6, 1.45, 4.2, 0.4, "Kaggle, tabla privada: error absoluto medio (millones)", size=12, color="muted")
ch = barras(s, 4.5, 1.8, 4.3, 3.9, cats, [("Tabla privada", list(clas["privado"]), cols)], minimo=0, maximo=200, gap=45,
            grid=False)
etiquetas_puntos(ch, [[n(v) for v in clas["privado"]]])
texto(s, 4.6, 5.75, 4.2, 1.2, ["Pública: 20 % de la prueba, visible durante la competencia. Privada: el 80 % restante, "
                               "revelada al final.", f"El error absoluto medio es cerca del {n(100 * P02 / pron['media'], 0)} % "
                               f"del precio medio predicho ({n(pron['media'], 0)} millones)."], size=12, color="muted", espacio=4)
tarjeta(s, 9.05, 1.5, 3.75, 2.95, "El modelo", [
    "Un boosting de árboles (LightGBM) con 162 variables de vivienda, entorno, Catastro, texto y ubicación; promedio de 5 "
    "corridas con distinta semilla.",
    "Ajuste final: la distancia de cada predicción a la media del entrenamiento se multiplica por 1,08, porque el modelo se "
    f"quedaba corto en los avisos caros. Se eligió mirando estos 307 avisos, así que el {n(ERR_CH, 1)} % es algo optimista "
    f"({n(lado_sin['error_abs_mediano'], 1)} % sin el ajuste); baja el error de Kaggle, pero para ofertar no conviene "
    "(Etapa 4)."], tipo="definicion")
tarjeta(s, 9.05, 4.6, 3.75, 2.35, "Respuesta corta", [
    "Sirve para ordenar, filtrar y negociar, no para ofertar sin tasación: los dueños aceptan sobre todo cuando el modelo "
    f"sobrestima y, con un error mediano de {n(ERR_CH, 1)} % al predecir Chapinero sin haberlo visto, eso hace perder "
    "dinero."], tipo="clave")
nota("Empezamos por el resultado. El modelo ordena bien los precios de Chapinero: la correlación de rangos entre predicción y "
     f"precio es 0,87. Estas cifras salen de entrenar sin los 307 avisos de Chapinero que trae el entrenamiento y "
     f"predecirlos, porque la prueba de Kaggle no tiene precios. Pero su error mediano, {n(ERR_CH, 1)} % (la mitad de los avisos yerra menos que eso, en "
     "porcentaje de su precio), no alcanza para "
     "ofertar solo: el "
     f"{n(lado_ch['sobre_10'], 0)} % de los avisos queda más de 10 % por encima del precio. En Kaggle quedó segundo de seis "
     f"equipos, con {n(P02)} millones de error absoluto medio, {n(PREVIO_PRIV - P02)} menos que el mejor envío previo del "
     "equipo. Ese puntaje es el resultado final; lo que mostramos ahora es cómo se llegó a él y qué significa para el negocio. "
     "Si preguntan por el modelo final: en 78 avisos cuya descripción repite la de uno del entrenamiento, la predicción pesa "
     "0,7 el precio de ese aviso y 0,3 la del modelo (apéndice).")

# ------------------------------------------------------------------------------------------------------------------
# 3. Negocio y hoja de ruta
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Para la empresa, pagar de más cuesta más que dejar de comprar", etiqueta="El problema de negocio",
               seccion="Negocio")
tarjeta(s, 0.55, 1.5, 6.0, 2.05, "Predicción mayor que el precio real", [
    "La empresa ofrece de más y compra caro: pierde capital al revender y carga costos de tenencia.",
    "Zillow cerró su negocio de compra en 2021 con pérdidas de cerca de 500 millones de dólares y recortó cerca del 25 % "
    "de su personal."], tipo="cuidado", size_titulo=16)
tarjeta(s, 0.55, 3.7, 6.0, 1.45, "Predicción menor que el precio real", [
    "La oferta queda corta y el dueño no vende: se pierde un negocio, pero no capital."], tipo="definicion", size_titulo=16)
imagen(s, IMG / "f_mae.png", 7.0, 1.6, h=0.95, alt="Fórmula del error absoluto medio")
tarjeta(s, 7.0, 2.75, 5.8, 2.4, "Kaggle y el negocio miden distinto", [
    "Kaggle premia el error absoluto medio: 100 millones de más pesan lo mismo que 100 de menos.",
    "Para la empresa, el error hacia arriba es el caro. Por eso hay que ver hacia dónde se equivoca el modelo."],
    tipo="neutra", size_titulo=16)
pasos = [("Datos", "entorno, Catastro y texto"), ("Algoritmo", "boosting de árboles"),
         ("Validación", "imitar el reto: un lugar nuevo"), ("Pronóstico y negocio", "dónde falla y qué cuesta")]
for i, (t1, t2) in enumerate(pasos):
    x0 = 0.55 + i * 3.1
    circulo(s, x0, 5.55, 0.62, i + 1, relleno="navy", size=18)
    texto(s, x0 + 0.72, 5.45, 2.3, 0.6, f"Etapa {i + 1}. {t1}", size=15, color="navy")
    texto(s, x0 + 0.72, 6.05, 2.3, 0.7, t2, size=13, color="muted")
nota("El problema no es simétrico. Si el modelo predice un precio mayor que el real y la empresa compra con esa predicción, "
     "paga de más y pierde capital: eso le pasó a Zillow en 2021. Si predice un precio menor, la oferta es baja y el dueño "
     "no vende: se pierde un negocio, pero no plata. Kaggle mide el error absoluto medio, que trata igual los dos errores; "
     "nosotros los miramos por separado. Seguimos cuatro etapas: datos, algoritmo, validación, y pronóstico y negocio.")

# ------------------------------------------------------------------------------------------------------------------
# 4. Datos
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Entrenamos con Bogotá, casi sin Chapinero, para predecir Chapinero", etiqueta="Etapa 1. Datos",
               seccion="Datos")
imagen(s, IMG / "mapa_datos.png", 0.55, 1.5, w=5.0,
       alt="Mapa de Bogotá con los avisos de entrenamiento, los de prueba en Chapinero y los 307 de entrenamiento en Chapinero")
stats = [(n(desc["n_train"], 0), "avisos de entrenamiento de Properati, 2019 a 2021; solo 307 en Chapinero, que se usan "
                                 "para validar (Chapinero fuera)"),
         (n(desc["n_test"], 0), f"avisos de prueba; el {n(pron['pct_localidad_chapinero'], 0)} % en la localidad de Chapinero"),
         ("300 a 1.650", f"millones: rango del precio en el entrenamiento (mediana {n(desc['precio_mediana'], 0)})"),
         (f"{n(desc['pct_apto_test'], 0)} %", f"de la prueba son apartamentos, frente a {n(desc['pct_apto_train'], 0)} % del entrenamiento")]
for i, (v, e) in enumerate(stats):
    x0, y0 = 5.9 + (i % 2) * 3.5, 1.45 + (i // 2) * 1.35
    texto(s, x0, y0, 3.4, 0.6, v, size=30, color="azul")
    texto(s, x0, y0 + 0.58, 3.3, 0.7, e, size=13, color="muted")
cent = list(hist["desde"] + 37.5)
lineas_xy(s, 5.8, 4.2, 7.0, 2.75, [
    dict(nombre="Chapinero (307 avisos)", x=cent, y=list(hist["Chapinero"]), color="azul", ancho=2.5),
    dict(nombre="Resto de Bogotá (38.337)", x=cent, y=list(hist["Resto de Bogotá"]), color="gris", ancho=2.5)],
    minx=300, maxx=1725, miny=0, fmt_x="#,##0", fmt_y="0", titulo_x="Precio del aviso en el entrenamiento (millones)",
    titulo_y="Avisos (%)", unidad_x=300, pos_leyenda="arriba")
nota("Los datos son avisos de Properati de 2019 a 2021. Entrenamos con 38.644 avisos de Bogotá, de los cuales solo 307 "
     "están en Chapinero, y predecimos 10.286 avisos, casi todos de Chapinero. El precio del entrenamiento está acotado entre 300 y "
     "1.650 millones. Chapinero es más caro: su mediana es 750 millones frente a 550 del resto, y la prueba es casi toda de "
     "apartamentos.")

# ------------------------------------------------------------------------------------------------------------------
# 5. Chapinero es otro mercado (lámina compartida con compare)
# ------------------------------------------------------------------------------------------------------------------
s, t_ = chapinero_otro_mercado(deck, "Etapa 1. Datos", "Datos")
nota(t_)

# ------------------------------------------------------------------------------------------------------------------
# 6. Texto como datos: el aviso de ejemplo
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("El texto del aviso completa lo que falta en las columnas", etiqueta="Etapa 1. Datos", seccion="Datos")
PID = "84a9da9d84b975bcd0954c36"
av = cand.loc[PID]
descr = str(av["description"]).strip()
# Los baños no se resaltan: el aviso escribe "bano", sin ñ, y la regla no los extrajo (cifras_v3_opus.md, M2).
RESALTAR = ["86 m2", "segundo piso", "exterior", "balcon", "2 habitaciones", "chimenea", "parqueadero", "deposito", "gym",
            "salon comunal", "terraza panoramica"]
patron = re.compile("|".join(re.escape(r) for r in sorted(RESALTAR + ["informes X"], key=len, reverse=True)))
partes, pos = [], 0
for m in patron.finditer(descr):
    if m.start() > pos:
        partes.append((descr[pos:m.start()], {"color": "tinta"}))
    partes.append((m.group(0), {"color": "naranja" if m.group(0) == "informes X" else "azul"}))
    pos = m.end()
if pos < len(descr):
    partes.append((descr[pos:], {"color": "tinta"}))
texto(s, 0.55, 1.4, 6.2, 0.4, f"Un aviso real de Chapinero (precio publicado: {n(av['price'] / 1e6, 0)} millones)", size=13,
      color="muted")
forma(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.55, 1.85, 6.2, 2.45, relleno="tinte", linea="gris_claro", radio=0.04)
texto(s, 0.72, 1.97, 5.9, 2.25, [[(f"{str(av['title'])}. ", {"color": "muted"})] + partes], size=16)
INDIC = {"txt_parqueadero": "parqueadero", "txt_terraza_balcon": "terraza o balcón", "txt_zona_social": "zona social",
         "txt_chimenea": "chimenea", "txt_vista": "vista", "txt_exterior": "exterior", "txt_gimnasio": "gimnasio",
         "txt_deposito": "depósito"}
ind = [v for k, v in INDIC.items() if k in av.index and av[k] == 1]
filas = [("Variable extraída del texto", "Valor"), ("Área (falta en las columnas)", f"{n(av['area_texto'], 0)} m²"),
         ("Indicadores con negación (sin ascensor no cuenta)", ", ".join(ind)), ("Cifra tapada por los organizadores", "informes X"),
         ("Puntaje de precio del texto", f"{n(av['puntaje_millones'], 0)} millones")]
tabla(s, 7.1, 1.85, 5.7, filas, [2.65, 3.05], size=13, alto_fila=0.52)
texto(s, 7.1, 4.72, 5.7, 0.3, "Azul: lo que se extrae. Naranja: cifra tapada con X.", size=12, color="muted")
texto(s, 0.55, 4.45, 6.2, 0.35, "Avisos de entrenamiento sin el dato (%)", size=13, color="muted")
ch = barras(s, 0.45, 4.8, 6.3, 2.15, ["Área", "Baños"],
            [("En las columnas", [100 - 31.3, falt.loc["Baños", "entrenamiento"]], "gris_claro"),
             ("Tras completar con el texto", [falt.loc["Área, tras completarla con el texto", "entrenamiento"],
                                              falt.loc["Baños, tras completarlos con el texto", "entrenamiento"]], "azul")],
            minimo=0, maximo=100, leyenda=True, gap=40, overlap=-10, grid=False)
etiquetas_puntos(ch, [[f"{n(100 - 31.3, 0)} %", f"{n(falt.loc['Baños', 'entrenamiento'], 0)} %"],
                      [f"{n(falt.loc['Área, tras completarla con el texto', 'entrenamiento'], 0)} %",
                       f"{n(falt.loc['Baños, tras completarlos con el texto', 'entrenamiento'], 0)} %"]])
tarjeta(s, 7.1, 5.05, 5.7, 1.9, "Faltantes en el entrenamiento", [
    "El área falta en las columnas en el 69 % de los avisos; el texto la recupera en 22 de cada 100 (un tercio de los "
    "que no la tenían).",
    f"El texto casi nunca trae el precio: {n(v1['pct_menciona_millones_con_cifra'], 1)} % menciona una cifra en millones."],
    tipo="recordar")
nota("Las columnas de Properati vienen muy incompletas: el área falta en dos de cada tres avisos. El texto del aviso trae esa "
     "información. En este aviso real el área de 86 metros no está en las columnas y la sacamos de la descripción, junto con "
     "características que se leen con diccionarios que respetan la negación: sin ascensor no cuenta como ascensor. Un ridge "
     "sobre la frecuencia ponderada de las palabras, estimado sin usar el precio del propio aviso, da un puntaje de precio: "
     "aquí 673 millones frente a un precio "
     "publicado de 749. El texto casi nunca trae el precio, y los organizadores taparon cifras con X.")

# ------------------------------------------------------------------------------------------------------------------
# 7 y 8. Texto como datos: de texto crudo a matriz y qué dice el texto (Lecture 15)
# ------------------------------------------------------------------------------------------------------------------
s, t_ = texto_a_matriz(deck, "Etapa 1. Datos", "Datos")
nota(t_)
s, t_ = texto_barrios_topicos(deck, "Etapa 1. Datos", "Datos")
nota(t_)

# ------------------------------------------------------------------------------------------------------------------
# 7. Variables (lámina compartida con compare)
# ------------------------------------------------------------------------------------------------------------------
s, t_ = variables_hedonicas(deck, "Etapa 1. Datos", "Datos")
nota(t_)

# ------------------------------------------------------------------------------------------------------------------
# 8. Un árbol del modelo
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Cada árbol hace preguntas de sí o no sobre el aviso", etiqueta="Etapa 2. Algoritmo", seccion="Algoritmo")


def nombre_corte(nd):
    v, u = nd["variable"], nd["umbral_legible"]
    return {"banos_imp": f"Baños ≤ {n(u, 1)}", "parqueaderos_texto": f"Parqueaderos (texto) ≤ {n(u, 1)}",
            "es_apartamento": "¿Es casa?", "lsa_08": f"Componente LSA 8 del texto ≤ {n(u, 3)}",
            "ln_vref_300m": f"Suelo a 300 m ≤ {n(u / 1e6, 1)} mill./m²",
            "ln_avaluo_ph_300m": f"Avalúo de apartamentos a 300 m ≤ {n(u / 1e6, 1)} mill./m²"}.get(v, v)


AX, AW = 0.55, 8.6
Y_NIV, W_NIV, H_N = [1.5, 2.75, 4.0, 5.5], [2.7, 2.6, 2.05, 1.04], 0.82


def dibujar(nd, nivel, x_izq, x_der):
    cx = (x_izq + x_der) / 2
    w = W_NIV[nivel]
    if nivel == 3 or nd["hoja"]:
        val = nd["valor"]
        nodo(s, cx - w / 2, Y_NIV[3], w, H_N, [n(val, 1, signo=True), f"{n(nd['n'], 0)} avisos"],
             relleno="azul_claro" if val > 0 else "ambar_claro", borde="gris_claro", sizes=(15, 12),
             colores=("azul" if val > 0 else "naranja", "muted"), margen=0.0)
        return
    nodo(s, cx - w / 2, Y_NIV[nivel], w, H_N, [nombre_corte(nd), f"{n(nd['n'], 0)} avisos"], relleno="tinte", sizes=(13, 12))
    for lado, (a_, b_) in (("izq", (x_izq, cx)), ("der", (cx, x_der))):
        hx = (a_ + b_) / 2
        linea(s, cx, Y_NIV[nivel] + H_N, hx, Y_NIV[nivel + 1], color="gris", ancho=1.25)
        lx = (cx + hx) / 2
        texto(s, lx - (0.42 if lado == "izq" else -0.02), (Y_NIV[nivel] + H_N + Y_NIV[nivel + 1]) / 2 - 0.2, 0.42, 0.3,
              "sí" if lado == "izq" else "no", size=12, color="muted", align="right" if lado == "izq" else "left")
        dibujar(nd[lado], nivel + 1, a_, b_)


dibujar(arbol, 0, AX, AX + AW)
texto(s, AX, 6.38, AW, 0.6, "Es el árbol número 4 de unos 250, recortado a tres niveles: cada caja final promedia las hojas "
      "de debajo y suma o resta entre −10 y +16 millones. Suma o resta poco porque cada árbol entra con peso 0,05 (lámina "
      "siguiente); cada árbol ve el 80 % de los avisos (30.947).", size=12, color="muted")
tarjeta(s, 9.45, 1.5, 3.35, 1.6, "Cómo leerlo", [
    "Cada aviso baja respondiendo preguntas y termina en una caja que suma o resta millones a la predicción."],
    tipo="definicion")
tarjeta(s, 9.45, 3.25, 3.35, 2.1, "Qué corrige este árbol", [
    "Suma millones a los avisos con más baños y suelo caro alrededor, y resta a los de 2 baños o menos y uno o ningún "
    "parqueadero, sobre lo que ya predijeron los árboles anteriores."], tipo="clave")
tarjeta(s, 9.45, 5.5, 3.35, 1.45, "Árbol de regresión (ISLR, cap. 8)", [
    "Cada corte se elige para reducir el error cuadrático."], tipo="recordar")
nota("Este es un árbol real del modelo final, el cuarto de unos 250, cortado a tres niveles. Pregunta primero por el número de baños; si son "
     "pocos, por los parqueaderos que menciona el texto; si son muchos, por el valor del suelo y el avalúo alrededor. Cada "
     "caja final suma o resta unos pocos millones. Un árbol solo es débil; el boosting suma cientos de ellos.")

# ------------------------------------------------------------------------------------------------------------------
# 9. Boosting con una variable
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Boosting: cada árbol corrige el error de los anteriores", etiqueta="Etapa 2. Algoritmo", seccion="Algoritmo")
imagen(s, IMG / "boosting_1d.png", 0.55, 1.45, w=12.3,
       alt="Cuatro paneles: precio frente al área con la predicción del boosting tras 0, 1, 20 y 241 árboles")
imagen(s, IMG / "f_boosting.png", 0.65, 4.95, h=0.7, alt="Fórmula del boosting: promedio más la suma de árboles con contracción")
texto(s, 0.55, 5.68, 5.5, 1.27, "Se arranca en el promedio. Cada árbol nuevo se ajusta al residuo que dejaron los anteriores "
      "y se suma con un peso pequeño, η = 0,05: muchos pasos cortos evitan aprender el ruido. El error medio de cada panel es "
      "el error absoluto medio en el entrenamiento.", size=14, color="tinta")
# Morado y gris: la regla del usuario no deja juntar azul y morado (disenio_v3_opus.md, H1).
tarjeta(s, 6.4, 5.0, 3.1, 1.95, "Por qué no un solo árbol", [
    "Un árbol grande tiene mucha varianza (ISLR, cap. 8); el boosting reduce el error poco a poco y la parada temprana frena "
    "antes de sobreajustar."], tipo="aclaracion")
tarjeta(s, 9.7, 5.0, 3.1, 1.95, "Ejemplo didáctico", [
    "Una sola variable (área), árboles de 4 hojas y la misma tasa η = 0,05 del modelo final. El modelo final usa 162 "
    "variables, hasta 63 hojas por árbol y cerca de 250 árboles."], tipo="neutra")
nota("Así funciona el boosting con una sola variable, para poder verlo. Arranca prediciendo el promedio, con un error medio de "
     "247 millones. El primer árbol mueve muy poco la predicción, porque entra con un peso de 0,05. A los 20 árboles ya "
     "aparece la forma: los apartamentos grandes valen más. A los 241 la escalera sigue los datos y el error baja a 161. "
     "Nuestro modelo hace lo mismo con 162 variables, árboles de hasta 63 hojas y cerca de 250 árboles. El error medio de los "
     "paneles es el error absoluto medio en el entrenamiento.")

# ------------------------------------------------------------------------------------------------------------------
# 10. Boosting en Chapinero: tres apartamentos y curva de aprendizaje
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Tres apartamentos que no vio y cuándo dejar de sumar árboles", etiqueta="Etapa 2. Algoritmo",
               seccion="Algoritmo")
COL3 = ["azul", "naranja", "4D5656"]
series = []
k = list(range(len(tres["avisos"][0]["camino"])))
for av_, c in zip(tres["avisos"], COL3):
    series.append(dict(nombre=f"{n(av_['area'], 0)} m²", x=k, y=av_["camino"], color=c, ancho=2.5))
for av_, c in zip(tres["avisos"], COL3):
    series.append(dict(nombre=f"Real {n(av_['real'], 0)}", x=[0, k[-1]], y=[av_["real"]] * 2, color=c, ancho=1.5, guiones=True))
lineas_xy(s, 0.45, 1.45, 6.3, 4.05, series, leyenda=False, minx=0, maxx=250, miny=300, maxy=1600, fmt_x="0",
          fmt_y="#,##0", titulo_x="Número de árboles", titulo_y="Predicción (millones)", unidad_x=50, unidad_y=200)
lin3 = []
for av_, c in zip(tres["avisos"], COL3):
    lado = "arriba" if av_["err_pct"] > 0 else "abajo"
    extra = ": aquí se pagaría de más" if av_["err_pct"] > 0 else ""
    lin3.append([(f"{n(av_['area'], 0)} m²: predice {n(av_['pred'], 0)}, real {n(av_['real'], 0)} "
                  f"({n(abs(av_['err_pct']), 0)} % {lado}{extra})", {"color": c})])
texto(s, 0.55, 5.55, 6.3, 0.75, lin3, size=14, espacio=2)
texto(s, 0.55, 6.3, 6.3, 0.65, f"Línea continua: predicción árbol a árbol desde el promedio sin Chapinero ({n(tres['base'], 0)}); "
      "punteada: precio real. Elegidos cerca de los percentiles 15, 50 y 85 del precio, con un error cercano al mediano de "
      "este modelo (14 %).", size=12, color="muted")
cv = curva
i241 = int(np.where(cv["arboles"] == tres["parada"])[0][0])
imin = int(cv["mae_chapinero"].idxmin())
ch2 = lineas_xy(s, 6.9, 1.45, 5.95, 4.05, [
    dict(nombre="Chapinero fuera", x=list(cv["arboles"]), y=list(cv["mae_chapinero"]), color="navy", ancho=2.5),
    dict(nombre="Entrenamiento", x=list(cv["arboles"]), y=list(cv["mae_entrenamiento"]), color="gris", ancho=2.0),
    dict(nombre="Parada", x=[tres["parada"]] * 2, y=[0, 320], color="gris", ancho=1.5, guiones=True)],
    leyenda=False, minx=0, maxx=800, miny=0, maxy=320, fmt_x="0", fmt_y="0", titulo_x="Número de árboles",
    titulo_y="Error absoluto medio (millones)", unidad_x=200, unidad_y=50)
etiquetas_puntos(ch2, [[None] * 599 + ["Chapinero fuera (307)"] + [None] * (len(cv) - 600),
                       [None] * 599 + ["Entrenamiento"] + [None] * (len(cv) - 600),
                       [None, f"Parada: {tres['parada']}"]], size=12)
ch2.plots[0].series[0].points[599].data_label.position = XL_LABEL_POSITION.ABOVE
ch2.plots[0].series[1].points[599].data_label.position = XL_LABEL_POSITION.BELOW
ch2.plots[0].series[2].points[1].data_label.position = XL_LABEL_POSITION.RIGHT
texto(s, 6.95, 5.55, 5.9, 1.4, "Modelo de ilustración: la especificación final con una semilla, entrenada sin Chapinero. "
      f"La parada en {tres['parada']} árboles se decide con zonas de Bogotá apartadas como validación (su curva no se "
      "muestra), sin mirar Chapinero; "
      "parar temprano es una forma de regularizar (Lecture 12). Mirando Chapinero habríamos parado en "
      f"{int(cv['arboles'][imin])}: sería ajustar con la misma validación que mide el error.", size=13, color="tinta")
nota("Ahora la especificación final, con una semilla y entrenada sin Chapinero, sobre tres apartamentos que nunca vio, elegidos con un error cercano al "
     f"mediano de este modelo sin el ×1,08, que es de {n(tres['error_abs_mediano'], 0)} %. Las tres predicciones arrancan en el "
     "promedio y cada árbol las corrige un poco. El de 125 metros termina por encima de su precio real: es el error caro para "
     f"la empresa. A la derecha, el error en Chapinero baja con los árboles; paramos en {tres['parada']} con una regla que no "
     f"mira Chapinero. Para preguntas: en el árbol {tres['parada']} este modelo da {n(cv['mae_chapinero'][i241])} en Chapinero "
     f"fuera, la validación oficial da {n(MAE_CH)}, y mirando Chapinero habríamos parado en {int(cv['arboles'][imin])} con "
     f"{n(cv['mae_chapinero'][imin])}.")

# ------------------------------------------------------------------------------------------------------------------
# 11. Qué mueve la predicción: SHAP y cascada
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Qué mueve la predicción: el texto, el tamaño y la ubicación", etiqueta="Etapa 2. Algoritmo", seccion="Algoritmo")
NOMBRE = {"puntaje_texto_ridge": "Puntaje de precio del texto", "area_imp": "Área", "banos_imp": "Baños", "bedrooms": "Alcobas",
          "ln_dist_embajada_m": "Distancia a embajadas", "ln_vref_300m": "Valor del suelo a 300 m",
          "estrato_medio_300m": "Estrato medio a 300 m", "prop_apartamentos_500m": "Apartamentos a 500 m (%)",
          "elevacion_m": "Elevación", "lsa_09": "Componente LSA 9 del texto", "lsa_04": "Componente LSA 4 del texto",
          "lsa_03": "Componente LSA 3 del texto", "log_area": "Logaritmo del área", "rooms": "Ambientes", "y_km": "Coordenada norte"}
top = shap_imp.head(8)
texto(s, 0.55, 1.4, 5.6, 0.4, "Aporte absoluto medio en la prueba, valores de Shapley (millones; Lecture 13)", size=12,
      color="muted")
ch = barras(s, 0.45, 1.75, 5.3, 3.5, [NOMBRE.get(k_, k_) for k_ in top.index], [("SHAP", list(top.values), "azul")],
            minimo=0, maximo=170, gap=40, grid=False)
etiquetas_puntos(ch, [[n(v, 0) for v in top.values]])
tot = shap_blq.sum()
texto(s, 0.55, 5.3, 5.2, 0.45, f"Por bloques: texto {n(100 * shap_blq['Texto'] / tot, 0)} %, vivienda {n(100 * shap_blq['Vivienda'] / tot, 0)} %, "
      f"entorno {n(100 * shap_blq['Entorno (OSM, GEE, coordenadas)'] / tot, 0)} % y Catastro {n(100 * shap_blq['Catastro'] / tot, 0)} %.",
      size=13, color="tinta")
imagen(s, IMG / "f_shap.png", 0.6, 5.75, h=0.5, alt="Descomposición SHAP de la predicción")
texto(s, 0.55, 6.3, 5.45, 0.65, "Cada φ es lo que una variable suma o resta frente a φ₀, la predicción promedio del "
      "entrenamiento; los aportes suman la predicción. TreeSHAP los calcula exactos para árboles (Lecture 13). No son efectos "
      "causales.", size=12, color="muted")
av2 = tres["avisos"][1]
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
    return NOMBRE.get(k_, k_)


X0C, WC, LAB = 6.2, 6.6, 2.85
xmin, xmax = 550, 850
esc = lambda v: X0C + LAB + (v - xmin) / (xmax - xmin) * (WC - LAB - 0.15)  # noqa: E731
texto(s, X0C, 1.4, WC, 0.45, f"Un apartamento de {n(av2['area'], 0)} m², {n(av2['alcobas'], 0)} alcobas y {n(av2['banos'], 0)} baños "
      f"(precio real: {n(av2['real'], 0)})", size=13, color="muted")
filas_c = [("Promedio del entrenamiento sin Chapinero", tres["base"], "total")]
for k_, d_ in pasos_c:
    filas_c.append((etiqueta_var(k_, av2["valores"].get(k_, 0)), d_, "paso"))
filas_c.append(("Predicción", tres["base"] + sum(d for _, d in pasos_c), "total"))
y0c, hc = 2.25, 0.42
acum = tres["base"]
for i, (et, val, tipo) in enumerate(filas_c):
    yy = y0c + i * hc
    texto(s, X0C, yy + 0.02, LAB - 0.1, hc - 0.04, et, size=12, color="tinta", align="right", anchor="middle")
    if tipo == "total":
        forma(s, MSO_SHAPE.RECTANGLE, esc(xmin), yy + 0.07, esc(val) - esc(xmin), hc - 0.14, relleno="navy")
        if i == 0:
            texto(s, esc(val) - 0.7, yy + 0.02, 0.62, hc - 0.04, n(val, 0), size=12, color="blanco", align="right", anchor="middle")
        else:
            texto(s, esc(val) + 0.05, yy + 0.02, 0.9, hc - 0.04, n(val, 0), size=12, color="navy", anchor="middle")
    else:
        a_, b_ = sorted([acum, acum + val])
        forma(s, MSO_SHAPE.RECTANGLE, esc(a_), yy + 0.07, max(esc(b_) - esc(a_), 0.02), hc - 0.14,
              relleno="azul" if val > 0 else "naranja")
        texto(s, esc(b_) + 0.05, yy + 0.02, 0.9, hc - 0.04, n(val, 1, signo=True), size=12,
              color="azul" if val > 0 else "naranja", anchor="middle")
        acum += val
yb = y0c + len(filas_c) * hc
xr = esc(av2["real"])
linea(s, xr, y0c - 0.05, xr, yb, color="rojo", ancho=1.5)
texto(s, xr - 1.2, y0c - 0.38, 2.4, 0.3, f"precio real {n(av2['real'], 0)}", size=12, color="rojo", align="center")
for v in (600, 700, 800):
    texto(s, esc(v) - 0.4, yb + 0.06, 0.8, 0.3, n(v, 0), size=12, color="muted", align="center")
    linea(s, esc(v), yb + 0.02, esc(v), yb + 0.08, color="gris")
linea(s, esc(xmin), yb + 0.02, esc(xmax), yb + 0.02, color="gris_claro")
texto(s, X0C + LAB, yb + 0.36, WC - LAB, 0.3, "millones; el eje empieza en 550", size=12, color="muted", align="center")
texto(s, X0C, yb + 0.64, WC, 0.3, "El área resta porque el puntaje de texto ya recoge parte del tamaño.", size=12,
      color="tinta", align="right")
nota("Para ver qué usa el modelo, repartimos cada predicción entre las variables con valores de Shapley. En la prueba, el "
     "puntaje de texto es lo que más aporta, seguido del área y los baños; por bloques, el texto concentra cerca de la mitad "
     f"del aporte. A la derecha, el apartamento de 125 metros: arranca en el promedio de {n(tres['base'], 0)}, el texto le suma "
     f"{n(av2['cascada']['puntaje_texto_ridge'], 0)} millones y termina en {n(av2['pred'], 0)}, más de 100 millones sobre su "
     "precio real. Si la empresa comprara con esa predicción, pagaría de más. El área resta porque el texto ya recoge parte "
     "del tamaño: el aporte se reparte entre variables correlacionadas. Esta lámina responde también la pregunta del enunciado "
     "sobre qué variables pesan más. No son efectos causales ni precios implícitos de Rosen.")

# ------------------------------------------------------------------------------------------------------------------
# 12. Validación: cómo
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Validar como el reto: predecir un lugar que no se ha visto", etiqueta="Etapa 3. Validación", seccion="Validación")
imagen(s, IMG / "pliegues.png", 0.55, 1.4, w=12.3, alt="Cinco mapas con los avisos de validación de un pliegue en cada esquema")
imagen(s, IMG / "f_sesgo.png", 0.65, 5.1, w=5.8, alt="Error cuadrático esperado con un vecino en el entrenamiento y en un lugar nuevo")
texto(s, 0.55, 5.7, 6.1, 1.25, "El ruido es lo que las variables no explican. Si el ruido de dos vecinos tiene "
      "correlación ρ y el modelo copia al vecino, el error cuadrático esperado es 2σ²(1 − ρ); en un lugar nuevo, 2σ². La "
      "validación cruzada (CV) aleatoria lo subestima en 2σ²ρ (Lecture 14).", size=13, color="tinta")
tarjeta(s, 6.9, 5.05, 5.9, 1.9, "Por qué importa aquí", [
    "En la CV aleatoria, casi 2 de cada 3 avisos de validación (64 %) tienen un aviso de entrenamiento en la misma "
    "coordenada (hay avisos repetidos y coordenadas compartidas); para esos pares el residuo de LightGBM tiene correlación "
    "0,31, y cerca de 0 a más de 500 m (apéndice). En la prueba real, la mitad de las coordenadas están redondeadas a unos "
    "110 m y el vecino mediano está a 96 m; kNNDM (emparejamiento de distancias al vecino más cercano) elige pliegues con "
    "esa distancia (Linnenbrink y otros, 2024)."], tipo="cuidado")
nota("¿Cómo anticipar el error de Kaggle? Probamos seis esquemas; en los mapas van cinco, porque el buffer de 300 m es una "
     "variante de los bloques. La CV aleatoria mezcla vecinos: casi dos de cada tres avisos de validación tienen otro en la "
     "misma coordenada en el entrenamiento, así que el modelo aprovecha el ruido que comparten con su vecino: medimos una "
     "correlación de 0,31 entre residuos en la misma coordenada. Los bloques, los clústeres y kNNDM "
     "separan el espacio, y el último esquema imita el taller: entrenar sin Chapinero y predecir sus 307 avisos. Es el "
     "argumento de la lectura de validación cruzada espacial: como no conocemos precios de Chapinero, no podemos aprovechar la "
     "correlación con los vecinos, y una validación que sí la aprovecha promete un error que no veremos.")

# ------------------------------------------------------------------------------------------------------------------
# 13. Validación: resultado
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Solo dejar Chapinero fuera anticipa el nivel del error de Kaggle", etiqueta="Etapa 3. Validación",
               seccion="Validación")
esq = [("CV aleatoria", 78.83), ("kNNDM", 111.68), ("Bloques de 2 km", 119.08), ("Bloques con buffer de 300 m", 123.13),
       ("Clústeres k-means", 124.99), ("Chapinero fuera", MAE_CH), ("Kaggle, tabla privada", P01), ("Kaggle, tabla pública", U01)]
cols = ["gris_claro"] * 5 + ["naranja", "navy", "navy"]
ch = barras(s, 0.45, 1.45, 7.6, 4.55, [e for e, _ in esq], [("MAE", [v for _, v in esq], cols)], minimo=0, maximo=200, gap=35,
            grid=False)
etiquetas_puntos(ch, [[n(v) for _, v in esq]])
texto(s, 0.55, 6.05, 7.5, 0.9, "Error absoluto medio de LightGBM sin el ×1,08, millones. Chapinero fuera es dejar "
      "una localización por fuera (leave-location-out, Lecture 14); el buffer retira del entrenamiento los avisos a 300 m o "
      "menos de los validados. En el cuaderno de la Lecture 14 (viviendas de Ames; error absoluto medio del logaritmo del "
      "precio), la CV aleatoria da 0,11 y dejar el barrio fuera 0,16, frente a 0,15 en la prueba.", size=12, color="muted")
tarjeta(s, 8.3, 1.5, 4.5, 2.4, "Cambia el nivel del error, no el ganador", [
    f"La CV aleatoria promete un error {n(100 * (1 - 78.83 / P01), 0)} % menor que Kaggle; Chapinero fuera queda a "
    f"{n(P01 - MAE_CH, 0)} millones de la tabla privada y a {n(U01 - MAE_CH, 0)} de la pública.",
    "Las dos eligen el mismo ganador: el boosting (LightGBM) es la mejor de las seis familias con la CV aleatoria y con "
    "Chapinero fuera "
    "(apéndice). Ese nivel es el que usamos para medir el riesgo de negocio en la Etapa 4. Con el ×1,08, la privada baja "
    "a 150,8."], tipo="clave")
tarjeta(s, 8.3, 4.02, 4.5, 1.3, "kNNDM sigue siendo optimista", [
    "Imita las distancias de la prueba, no el mercado: los 307 avisos de Chapinero dentro de kNNDM dan 137,2."],
    tipo="aclaracion")
tarjeta(s, 8.3, 5.44, 4.5, 1.51, "Incertidumbre", [
    f"Con 307 avisos, el intervalo de Chapinero fuera va de {n(IC_CH[0], 0)} a {n(IC_CH[1], 0)} (bootstrap, 10.000 réplicas). "
    "Trata los avisos como independientes; como los vecinos comparten ruido, el intervalo real es algo más ancho."],
    tipo="recordar")
nota(f"Este es el hallazgo de validación. La CV aleatoria promete {n(78.83)} millones de error; Kaggle dio {n(P01)}, casi el "
     f"doble. Los esquemas espaciales se acercan, y el único que llega cerca es entrenar sin Chapinero: {n(MAE_CH)}. Ojo con "
     f"la incertidumbre: con 307 avisos el intervalo va de {n(IC_CH[0], 0)} a {n(IC_CH[1], 0)}. Comparamos con el envío 01, "
     "que es el mismo modelo sin el estiramiento ×1,08. Y las dos validaciones eligen el mismo ganador: LightGBM es el mejor "
     "de las seis familias con la aleatoria y con Chapinero fuera. Lo que la aleatoria no anticipa es el nivel del error.")

# ------------------------------------------------------------------------------------------------------------------
# 14. Pronóstico
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Pronóstico: mediana de 799 millones, cerca de los avisos de Chapinero", etiqueta=E4,
               seccion="Pronóstico y negocio")
imagen(s, IMG / "pronostico_mapa.png", 0.55, 1.5, w=4.6, alt="Mapa de Chapinero con el precio predicho por metro cuadrado de cada aviso de prueba")
lineas_xy(s, 5.35, 1.45, 7.5, 3.15, [
    dict(nombre=f"Pronóstico para la prueba ({n(pron['n'], 0)})", x=cent, y=list(hist["Pronóstico para la prueba"]),
         color="naranja", ancho=3),
    dict(nombre="Chapinero, entrenamiento (307)", x=cent, y=list(hist["Chapinero"]), color="azul", ancho=2),
    dict(nombre="Resto de Bogotá, entrenamiento (38.337)", x=cent, y=list(hist["Resto de Bogotá"]), color="gris", ancho=2)],
    minx=300, maxx=1725, miny=0, fmt_x="#,##0", fmt_y="0", titulo_x="Precio (millones)", titulo_y="Avisos (%)",
    unidad_x=300, pos_leyenda="arriba")
stats = [(n(pron["mediana"], 0), "millones: predicción mediana (media de 840)"),
         (f"{n(pron['pct_sobre_1200'], 0)} %", "de los avisos con predicción mayor a 1.200 millones"),
         (n(pron["m2_mediana"], 1), "millones por m²: predicción mediana en los avisos con área (observado: 5,1 en el "
                                    "resto y 6,8 en Chapinero)")]
for i, (v, e) in enumerate(stats):
    x0 = 5.45 + i * 2.5
    texto(s, x0, 4.8, 2.4, 0.6, v, size=30, color="naranja")
    texto(s, x0, 5.4, 2.35, 0.95, e, size=13, color="muted")
texto(s, 5.45, 6.35, 7.4, 0.65, "Parecerse a Chapinero en la distribución no es acertar aviso por aviso: en los 307 avisos "
      "con precio conocido, el modelo queda por debajo (lámina siguiente). El mapa y el precio por m² usan los 5.832 avisos "
      "de prueba con área.", size=12, color="muted")
nota(f"Así se ve el pronóstico. La predicción mediana es de {n(pron['mediana'], 0)} millones, por encima del resto de Bogotá "
     f"y parecida a los 307 avisos de Chapinero que sí teníamos. El {n(pron['pct_sobre_1200'], 0)} % de la prueba tiene "
     "predicción mayor a 1.200 millones. En el mapa, el precio por metro cuadrado predicho varía mucho entre vecinos. Esta "
     "lámina sirve para comprobar que el pronóstico es plausible antes de ver dónde falla. Si preguntan: "
     f"{pron['n_sobre_1650']} predicciones superan 1.650 millones, el máximo del entrenamiento, por el ×1,08.")

# ------------------------------------------------------------------------------------------------------------------
# 15. Dónde se equivoca
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Dónde se equivoca: no capta la prima de ubicación de Chapinero", etiqueta=E4, seccion="Pronóstico y negocio")
cc = cal[cal["esquema"] == "Chapinero fuera"]
sa, ca = cc[cc["version"] == "Sin ajuste"]["sesgo"].tolist(), cc[cc["version"] == "Con ajuste x1,08"]["sesgo"].tolist()
texto(s, 0.55, 1.4, 7.4, 0.4, "Chapinero fuera: predicción menos precio real, promedio por quintil de la predicción (millones)",
      size=13, color="muted")
ch = barras(s, 0.45, 1.75, 7.5, 3.5, ["1 (más barato)", "2", "3", "4", "5 (más caro)"],
            [("Sin ajuste", sa, "gris_claro"), ("Con ajuste ×1,08", ca, "navy")], horizontal=False, minimo=-120, maximo=0,
            unidad=40, leyenda=True, gap=60, overlap=-5, valor_visible=True, fmt=MENOS, etiquetas_abajo=True)
etiquetas_puntos(ch, [[n(v, 0) for v in sa], [n(v, 0) for v in ca]])
tarjeta(s, 0.55, 5.38, 3.65, 1.57, "Por qué agrupar por la predicción", [
    "La empresa solo ve la predicción; agrupar por el precio real haría parecer que el modelo sobrestima los baratos y "
    "subestima los caros (regresión a la media)."], tipo="aclaracion")
tarjeta(s, 4.35, 5.38, 3.6, 1.57, "El ajuste ×1,08", [
    "Aleja cada predicción de la media: sube las altas y baja las bajas. Reduce el sesgo de los quintiles caros, lo aumenta en "
    "los baratos y, en Bogotá (kNNDM), sobrestima las predicciones altas (+28 y +36; apéndice)."], tipo="neutra")
imagen(s, IMG / "errores_mapa.png", 8.25, 1.45, w=4.6,
       alt="Mapa de los 307 avisos de Chapinero con el signo del error: rojo si predice de más, azul si predice de menos")
texto(s, 8.3, 1.4, 4.5, 0.4, "Los 307 avisos de Chapinero fuera", size=13, color="muted")
nota("Para la empresa importa el error dado lo que ve, que es la predicción. En Chapinero el modelo predice por debajo en "
     "todos los niveles: es una prima de ubicación que no capta. El ajuste por 1,08 reduce el sesgo de los tres quintiles más "
     "caros, pero lo aumenta en los dos más baratos; en Bogotá, con kNNDM, el modelo sin ajuste está calibrado y el ajuste "
     "sobrestima las predicciones altas. El mapa muestra que hay errores en los dos sentidos por todo Chapinero.")

# ------------------------------------------------------------------------------------------------------------------
# 16. Fuera del rango del entrenamiento
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Dos o más variables fuera de rango: más error en millones, no en porcentaje", etiqueta=E4,
               seccion="Pronóstico y negocio")
imagen(s, IMG / "confianza_mapa.png", 0.55, 1.5, w=4.4,
       alt="Mapa de los avisos de prueba según cuántas variables quedan fuera del rango del entrenamiento")
pv = rp["por_variable_pct"]
VN = {"ln_vref_300m": "Valor del suelo a 300 m", "ln_n_embajada_1000m": "Embajadas a 1 km", "elevacion_m": "Elevación",
      "ln_dist_area_protegida_m": "Distancia a los cerros", "ln_n_oficina_1000m": "Oficinas a 1 km", "area_imp": "Área",
      "ln_n_hotel_1000m": "Hoteles a 1 km"}
orden = sorted(pv, key=pv.get, reverse=True)
texto(s, 5.2, 1.4, 7.6, 0.4, "Avisos de prueba con la variable fuera del rango del entrenamiento: bajo el percentil 1 o "
      "sobre el 99 (%)", size=13, color="muted")
ch = barras(s, 5.1, 1.85, 7.7, 2.3, [VN[k_] for k_ in orden], [("fuera", [pv[k_] for k_ in orden], "naranja")],
            minimo=0, maximo=75, gap=30, grid=False)
etiquetas_puntos(ch, [[n(pv[k_], 1) for k_ in orden]])
filas = [("Variables fuera del rango", "Avisos de prueba (Kaggle)",
          "Error absoluto medio en Chapinero fuera (avisos de los 307)",
          "Error porcentual mediano"),
         ("Ninguna", n(rp["dentro"], 0), f"{n(reg.loc['Dentro del rango', 'mae'])} ({int(reg.loc['Dentro del rango', 'n'])})",
          f"{n(reg.loc['Dentro del rango', 'error_abs_mediano'], 0)} %"),
         ("Una", n(rp["una"], 0), f"{n(reg.loc['Una fuera', 'mae'])} ({int(reg.loc['Una fuera', 'n'])})",
          f"{n(reg.loc['Una fuera', 'error_abs_mediano'], 0)} %"),
         ("Dos o más", n(rp["dos_o_mas"], 0), f"{n(reg.loc['Dos o más fuera', 'mae'])} ({int(reg.loc['Dos o más fuera', 'n'])})",
          f"{n(reg.loc['Dos o más fuera', 'error_abs_mediano'], 0)} %")]
tabla(s, 5.2, 4.02, 7.6, filas, [1.9, 1.5, 2.6, 1.6], size=13, alto_fila=0.4)
tarjeta(s, 5.2, 6.12, 7.6, 0.83, None, [
    f"El {n(100 * rp['dos_o_mas'] / pron['n'], 0)} % de la prueba tiene dos o más variables fuera del rango: son avisos más "
    "caros y yerran más en millones, no en porcentaje. El rango sirve para priorizar la revisión, no para confiar a ciegas en "
    "el resto."], tipo="neutra")
nota(f"¿Importa salir del rango del entrenamiento? En millones sí, cuando son dos o más variables; en porcentaje no. El "
     f"{n(100 * rp['dos_o_mas'] / pron['n'], 0)} % de la prueba tiene dos o más variables fuera del rango, sobre todo valor del "
     "suelo, embajadas y elevación, en el norte y junto a los cerros. En la validación, esos avisos tienen el mayor error en "
     "millones, porque son más caros, pero en porcentaje el error es parecido en los tres grupos. Los conteos son de la prueba y "
     "el error se mide en los 307 avisos de Chapinero fuera, clasificados igual. El rango cuenta los dos lados; por eso el suelo "
     "da 59,5 % y no el 57 % de la lámina de Chapinero, que solo cuenta los que pasan el percentil 99.")

# ------------------------------------------------------------------------------------------------------------------
# 17. Precio mayor o menor
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Predecir de más o de menos: dos errores con costos distintos", etiqueta=E4, seccion="Pronóstico y negocio")
cd = CategoryChartData()
cd.categories = ["Chapinero fuera (307)", "Bogotá, kNNDM (38.644)"]
cd.add_series("Menor que el real en más de 10 %", [lado_ch["bajo_10"], lado_kn["bajo_10"]])
cd.add_series("Dentro de 10 %", [lado_ch["dentro_10"], lado_kn["dentro_10"]])
cd.add_series("Mayor que el real en más de 10 %", [lado_ch["sobre_10"], lado_kn["sobre_10"]])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_STACKED_100, Inches(0.45), Inches(1.45), Inches(7.6), Inches(2.4), cd)
ch = gf.chart
_estilo_grafico(ch, 12)
ch.has_title = False
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.TOP
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(12)
ch.plots[0].gap_width = 40
for ser, c in zip(ch.plots[0].series, ["azul", "gris_claro", "rojo"]):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = rgb(c)
_ejes(ch, valor_visible=False, grid=False, invertir_cat=True)
etiquetas_puntos(ch, [[f"{n(lado_ch['bajo_10'], 0)} %", f"{n(lado_kn['bajo_10'], 0)} %"],
                      [f"{n(lado_ch['dentro_10'], 0)} %", f"{n(lado_kn['dentro_10'], 0)} %"],
                      [f"{n(lado_ch['sobre_10'], 0)} %", f"{n(lado_kn['sobre_10'], 0)} %"]], color="tinta",
                 posicion=XL_LABEL_POSITION.CENTER)
for i in range(2):
    for j in (0, 2):
        for r in ch.plots[0].series[j].points[i].data_label.text_frame.paragraphs[0].runs:
            r.font.color.rgb = rgb("blanco")
texto(s, 0.55, 3.95, 7.5, 0.4, "Chapinero fuera: % de avisos con error mayor a 10 %, por quintil de la predicción (millones)",
      size=13, color="muted")
cats = [f"{n(a_, 0)} a {n(b_, 0)}" for a_, b_ in zip(lq["pred_min"], lq["pred_max"])]
ch = barras(s, 0.45, 4.3, 7.6, 2.65, cats, [("Predice de menos", list(lq["bajo"]), "azul"), ("Predice de más", list(lq["sobre"]), "rojo")],
            horizontal=False, minimo=0, maximo=50, leyenda=True, gap=60, overlap=-5, grid=False)
etiquetas_puntos(ch, [[n(v, 0) for v in lq["bajo"]], [n(v, 0) for v in lq["sobre"]]])
lado_sin = lados[(lados["esquema"] == "Chapinero fuera") & (lados["version"] == "Sin ajuste")].iloc[0]
tarjeta(s, 8.3, 1.5, 4.5, 1.95, f"Predice de más: {n(lado_ch['sobre_10'], 0)} % de Chapinero", [
    "Si se compra con la predicción, se paga de más: pérdida de capital, costos de tenencia y de reventa.",
    f"Modelo final, con el ×1,08; sin el ajuste, {n(lado_sin['sobre_10'], 0)} %."], tipo="cuidado", size_titulo=16)
tarjeta(s, 8.3, 3.6, 4.5, 1.6, f"Predice de menos: {n(lado_ch['bajo_10'], 0)} %", [
    "La oferta queda corta y el dueño no vende: costo de oportunidad, sin pérdida de capital.",
    f"Sin el ajuste, {n(lado_sin['bajo_10'], 0)} %."], tipo="definicion", size_titulo=16)
tarjeta(s, 8.3, 5.35, 4.5, 1.6, "Lectura", [
    f"El riesgo de pagar de más no se concentra en un segmento: va de {n(lq['sobre'].min(), 0)} % a {n(lq['sobre'].max(), 0)} % "
    f"por quintil de la predicción. El error porcentual mediano es {n(ERR_CH, 1)} % ({n(lado_sin['error_abs_mediano'], 1)} % sin "
    "el ×1,08)."], tipo="neutra")
nota(f"Separemos los dos errores. En Chapinero, con el modelo final, que incluye el ×1,08, el {n(lado_ch['sobre_10'], 0)} % de los avisos queda más "
     f"de 10 % por encima del precio real: ahí la empresa pagaría de más. El {n(lado_ch['bajo_10'], 0)} % queda más de 10 % por "
     "debajo: ahí perdería el negocio, pero no plata. Y el riesgo de pagar de más aparece en todos los niveles de precio.")

# ------------------------------------------------------------------------------------------------------------------
# 18. La trampa de Zillow
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("La trampa de Zillow: el dueño acepta cuando el modelo sobrestima", etiqueta=E4, seccion="Pronóstico y negocio")
acep, pierde = int(round(c10["aceptadas_pct"])), int(round(c10["pagan_de_mas_pct_ofertas"]))
gana = acep - pierde
texto(s, 0.55, 1.4, 4.6, 0.5, "100 ofertas en Chapinero fuera, con margen de 10 %", size=13, color="muted")
waffle(s, 0.6, 1.95, 0.36, ["rojo"] * pierde + ["azul"] * gana + ["gris_claro"] * (100 - acep), gap=0.06)
ley = [("rojo", f"{pierde} compran y pagan de más (en promedio {n(c10['perdida_media_si_paga_de_mas'], 0)} millones)"),
       ("azul", f"{gana} compran y ganan (en promedio {n(c10['ganancia_media_si_gana'], 0)} millones)"),
       ("gris_claro", f"{100 - acep} no aceptan")]
for i, (c, t) in enumerate(ley):
    yl = 6.2 + i * 0.26
    forma(s, MSO_SHAPE.RECTANGLE, 0.6, yl + 0.045, 0.18, 0.18, relleno=c)
    texto(s, 0.85, yl - 0.01, 4.4, 0.28, t, size=12, color="tinta", anchor="middle")
imagen(s, IMG / "f_oferta.png", 5.35, 1.45, h=0.46, alt="Regla de oferta y aceptación")
texto(s, 5.35, 2.02, 7.5, 1.2, [[("Con margen igual al descuento que aceptan los dueños (m = d = 10 %), solo aceptan cuando la "
                                  "predicción supera el valor: ", {}), ("justo donde el modelo sobrestimó. ", {"color": "rojo"}),
                                 (f"De 100 ofertas se aceptan {acep}: la empresa gana si paga menos que el valor ({gana}) y pierde "
                                  f"si paga más ({pierde}).", {})],
                                f"Resultado: {n(c10['ganancia_por_oferta'], 1)} millones por oferta hecha, aceptada o no (−13,2 sin el ×1,08). "
                                f"Con margen "
                                f"de 20 % se compran {n(c20['aceptadas_pct'], 0)} de cada 100 y aún se pierden "
                                f"{n(-c20['ganancia_por_oferta'], 1)} millones por oferta."], size=13, color="tinta")
g10 = gp[(gp["validacion"] == "Chapinero fuera") & np.isclose(gp["descuento"], 0.10)].sort_values("error_mediano_pct")
g05 = gp[(gp["validacion"] == "Chapinero fuera") & np.isclose(gp["descuento"], 0.05)].sort_values("error_mediano_pct")
ch = lineas_xy(s, 5.25, 3.25, 4.4, 3.7, [
    dict(nombre="Dueños aceptan 10 % menos", x=list(g10["error_mediano_pct"]), y=list(g10["ganancia_por_oferta"]), color="azul", ancho=2.5),
    dict(nombre="Dueños aceptan 5 % menos", x=list(g05["error_mediano_pct"]), y=list(g05["ganancia_por_oferta"]), color="naranja", ancho=2.5),
    dict(nombre="Hoy", x=[ERR_CH, ERR_CH], y=[-30, 30], color="gris", ancho=1.25, guiones=True),
    dict(nombre="Equilibrio", x=[2, 14], y=[0, 0], color="gris_claro", ancho=1.0)],
    leyenda=True, minx=2, maxx=14, miny=-30, maxy=30, fmt_x="0", fmt_y=MENOS, titulo_x="Error mediano simulado del modelo (%)",
    titulo_y="Ganancia por oferta (millones)", unidad_x=2, unidad_y=10, pos_leyenda="arriba")
etiquetas_puntos(ch, [None, None, [None, f"hoy: {n(ERR_CH, 1)} %"], None], size=12)
ocultar_leyenda(ch, [2, 3])
ch.plots[0].series[2].points[1].data_label.position = XL_LABEL_POSITION.LEFT
tarjeta(s, 9.85, 3.3, 2.95, 1.75, "Precisión necesaria", [
    f"Con margen de 10 %, la regla deja de perder solo si el error mediano baja a cerca de {n(EQ10, 0)} % (o {n(EQ5, 0)} % si "
    "los dueños aceptan solo 5 % menos)."], tipo="neutra")
tarjeta(s, 9.85, 5.17, 2.95, 1.78, "Supuestos", [
    "El valor de cada vivienda es su precio publicado, y el dueño solo ve la oferta, como en Zillow Offers. La curva reduce "
    "todos los errores en la misma proporción."], tipo="aclaracion")
nota("Esta es la trampa de Zillow. Supongamos que la empresa ofrece 10 % menos que la predicción y que los dueños aceptan si la "
     "oferta es al menos 90 % de lo que vale su casa. Entonces solo venden cuando el modelo sobrestimó. De 100 ofertas en "
     f"Chapinero, {acep} se aceptan y {pierde} de ellas pagan de más; en total la empresa pierde {n(-c10['ganancia_por_oferta'], 1)} "
     "millones por oferta. Subir el margen reduce la pérdida pero también las compras. Para no perder con margen de 10 %, el "
     f"error mediano tendría que bajar de {n(ERR_CH, 0)} % a cerca de {n(EQ10, 0)} %, en un modelo hipotético con todos los "
     "errores reducidos en la misma proporción. Para ofertar conviene la predicción sin estirar, que pierde menos.")

# ------------------------------------------------------------------------------------------------------------------
# 19. Recomendación
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Usar el modelo para filtrar y negociar, no para ofertar sin tasación", oscura=True, etiqueta="Recomendación",
               seccion="Recomendación")
col = [("Usarlo para", "azul_claro", "azul", "0,87", "correlación de rangos (Spearman) entre predicción y precio en los 307 "
        "avisos de Chapinero fuera", [
    f"Ordenar y filtrar avisos y dar una referencia de precio (error mediano de {n(ERR_CH, 1)} %).",
    "Usar la brecha entre predicción y precio pedido como alerta, no como prueba de una ganga: con ese error, una brecha "
    "grande puede ser error del modelo."]),
       ("No usarlo para", "rojo_claro", "rojo", f"{n(ERR_CH, 1)} %", "de error mediano al predecir Chapinero sin haberlo visto", [
    "Ofertas automáticas: los dueños aceptan cuando el modelo sobrestima y la regla pierde dinero.",
    "Confiar en un aviso por estar dentro del rango del entrenamiento: el error porcentual es parecido dentro y fuera.",
    "Ofertar con el ×1,08: sube las predicciones altas y pierde más (−14,8 frente a −13,2 millones por oferta)."]),
       ("Cómo hacerlo", "tinte", "navy", f"{n(EQ10, 0)} %", "de error mediano haría falta para no perder con margen de 10 %, si "
        "los dueños aceptan 10 % menos", [
    "Toda alerta pasa por comparables y tasación antes de ofertar.",
    "La oferta se ancla en el menor valor entre la predicción con margen y el precio pedido.",
    "Siguiente paso: pérdida de cuantiles para ofertas conservadoras y variables de una red convolucional preentrenada "
    "sobre imágenes satelitales (Lecture 14), que no necesitan precios de Chapinero."])]
for i, (tit, fondo, borde, cifra_, etq_, items) in enumerate(col):
    x0 = 0.55 + i * 4.13
    forma(s, MSO_SHAPE.ROUNDED_RECTANGLE, x0, 1.5, 3.95, 5.35, relleno=fondo, radio=0.04)
    texto(s, x0 + 0.2, 1.62, 3.6, 0.5, tit, size=20, color=borde)
    texto(s, x0 + 0.2, 2.12, 3.6, 0.75, cifra_, size=40, color=borde)
    texto(s, x0 + 0.2, 2.9, 3.55, 0.75, etq_, size=13, color="muted")
    linea(s, x0 + 0.2, 3.68, x0 + 3.75, 3.68, color="gris_claro")
    texto(s, x0 + 0.2, 3.8, 3.6, 3.0, items, size=15, color="tinta", espacio=10)
nota("Nuestra recomendación. Sí usar el modelo para ordenar y filtrar avisos y como referencia de precio: ordena bien, con una "
     "correlación de rangos de 0,87. No usarlo para ofertar de forma automática: con casi 13 % de error mediano, los dueños "
     "aceptan justo cuando el modelo se equivoca hacia arriba. Toda alerta pasa por tasación, y la oferta se ancla en el "
     "menor valor entre la predicción con margen y el precio pedido. El siguiente paso es predecir cuantiles para ofertar de "
     "forma conservadora y sumar variables de imágenes satelitales, que no necesitan precios de Chapinero.")

# ------------------------------------------------------------------------------------------------------------------
# Apéndice
# ------------------------------------------------------------------------------------------------------------------
s = deck.nueva("Especificación del modelo y búsqueda de hiperparámetros", etiqueta="Apéndice", seccion="Apéndice")
filas = [("Elemento", "Valor y cómo se eligió"),
         ("Algoritmo", "LightGBM, promedio de 5 semillas, sobre el precio en millones"),
         ("Objetivo", "L2 (error cuadrático); L1 y logaritmo probados sin ganancia (L1 subestimó más: sesgo −57 frente a −30 en "
                      "Chapinero y localidades vecinas, con un conjunto anterior de variables)"),
         ("Tasa de aprendizaje η", "0,05"),
         ("Máximo de hojas; mínimo de avisos por hoja", "63; 20"),
         ("Submuestreo de filas; de columnas; λ L2", "80 %; 70 %; 1"),
         ("Número de árboles", "parada temprana (100 rondas sin mejora) en una partición espacial interna; cerca de 250 árboles"),
         ("Búsqueda", "variantes comparadas con validación espacial: objetivo L2, L1 o logaritmo, y 63 hojas con 20 por hoja "
                      "frente a 31 con 50; quedó L2 con 63 y 20. Antes, una rejilla de 16 configuraciones de LightGBM, XGBoost y "
                      "CatBoost evaluada en la tabla pública (175,6 a 187,5, otras variables)"),
         ("Ajuste final", "×1,08 alrededor de la media, elegido con Chapinero fuera y la zona cercana; gemelos de texto: en "
                          "78 avisos cuya descripción repite la de uno del entrenamiento, pesa 0,7 su precio y 0,3 el modelo")]
tabla(s, 0.55, 1.45, 12.25, filas, [3.6, 8.65], size=13, alto_fila=0.5)
tarjeta(s, 0.55, 6.35, 12.25, 0.6, None, ["El 140,8 de Chapinero fuera con el ×1,08 se mide en la validación que lo eligió; Kaggle "
                                          "lo confirmó fuera de muestra (152,4 a 150,8)."], tipo="cuidado", maximo=14)
nota("Detalle de la especificación para preguntas.")

s, t_ = error_por_familia(deck, "Apéndice", "Apéndice")
nota(t_)

s, t_ = texto_apendice(deck, "Apéndice", "Apéndice")
nota(t_)

s = deck.nueva("Qué movió el error y qué no funcionó", etiqueta="Apéndice", seccion="Apéndice")
filas = [("Idea", "Validación con que se probó", "Chapinero fuera", "Kaggle, tabla privada"),
         ("Variables de Catastro", ("clústeres: −2,4", "azul"), ("−9,0", "azul"), "en todos los envíos"),
         ("Ajuste ×1,08", ("kNNDM: +0,5 (111,7 a 112,2); clústeres: −0,9 (125,0 a 124,1)", "tinta"),
          ("−2,9 (elegido con esta validación)", "azul"),
          (f"{n(P02 - P01, 1)} ({n(P01)} a {n(P02)})", "azul")),
         ("Mezcla 40 % LightGBM y 60 % CatBoost", ("clústeres: −1,4 (pesos elegidos con esta validación)", "azul"),
          (f"{n(bo['mezcla_40_60']['dif'], 1, True)}", "rojo"), (f"{n(P03 - P01, 1, True)} ({n(P03)})", "rojo")),
         ("BETO ajustado como variable", ("zona cercana a la prueba: −1,1 (128,2 a 127,1)", "azul"), ("+2,6 (147,1 a 149,7)", "rojo"),
          "no se envió"),
         ("Kriging de los residuos", ("kNNDM: +3,0 (111,7 a 114,7)", "rojo"),
          ("no se corrió; los 307 avisos de Chapinero en kNNDM: +2,4 (137,2 a 139,6)", "rojo"), "no se envió"),
         ("Red con atención en vez de LightGBM", ("clústeres: +3,8", "rojo"), ("+32,9", "rojo"), "no se envió")]
tabla(s, 0.55, 1.45, 12.25, filas, [3.2, 3.6, 3.25, 2.2], size=13, alto_fila=0.55)
texto(s, 0.55, 5.4, 12.25, 0.75, "Cambio en el error absoluto medio (millones). Azul: mejora; rojo: empeora; negro: depende de la "
      "validación. Zona cercana: avisos de entrenamiento a 1 km o menos de la prueba. BETO: modelo de lenguaje BERT en español, "
      "probado antes de agregar Catastro. Kriging: interpolación espacial de los residuos.", size=12, color="muted")
tarjeta(s, 0.55, 6.2, 12.25, 0.75, None, [
    "En Chapinero fuera, aparte del ×1,08, que se eligió con esa misma validación, solo bajó el error cambiar los datos "
    "(Catastro, −9,0); los cambios de algoritmo lo subieron (mezcla +3,0; red +32,9)."], tipo="neutra", maximo=14)
nota("Lo que probamos y no funcionó, con la validación que usó cada idea.")

s = deck.nueva("Calibración en Bogotá con kNNDM", etiqueta="Apéndice", seccion="Apéndice")
kk = cal[cal["esquema"] == "kNNDM"]
sk, ck = kk[kk["version"] == "Sin ajuste"]["sesgo"].tolist(), kk[kk["version"] == "Con ajuste x1,08"]["sesgo"].tolist()
texto(s, 0.55, 1.4, 12.2, 0.4, "Predicción menos precio real, promedio por decil de la predicción (millones), 38.644 avisos",
      size=13, color="muted")
ch = barras(s, 0.45, 1.8, 12.3, 3.9, [str(i) for i in range(1, 11)], [("Sin ajuste", sk, "gris_claro"), ("Con ajuste ×1,08", ck, "navy")],
            horizontal=False, minimo=-40, maximo=40, unidad=20, leyenda=True, gap=50, overlap=-5, valor_visible=True,
            fmt=MENOS, etiquetas_abajo=True)
etiquetas_puntos(ch, [[n(v, 0) for v in sk], [n(v, 0) for v in ck]])
tarjeta(s, 0.55, 5.85, 12.25, 1.1, "Lectura", [
    "Sin ajuste, el modelo está calibrado en Bogotá (−9 a +9 por decil). El ×1,08 corrige parte del nivel de Chapinero, pero "
    "en Bogotá sobrestima las predicciones altas (+28 y +36)."], tipo="neutra")
nota("Calibración del modelo en toda Bogotá, para preguntas.")

s = deck.nueva("Distancia al aviso de entrenamiento más cercano", etiqueta="Apéndice", seccion="Apéndice")
imagen(s, IMG / "distancias.png", 0.55, 1.45, w=6.5, alt="Distribución acumulada de la distancia al vecino de entrenamiento por esquema")
filas = [("Esquema", "Distancia mediana"), ("CV aleatoria", "0 m"), ("kNNDM", "79 m"), ("Prueba real (Kaggle)", "96 m"),
         ("Bloques de 2 km", "375 m"), ("Chapinero fuera", "553 m"), ("Clústeres k-means", "985 m")]
tabla(s, 7.4, 1.55, 5.4, filas, [3.2, 2.2], size=13, alto_fila=0.45)
tarjeta(s, 7.4, 4.95, 5.4, 1.95, "Lectura", [
    "kNNDM imita las distancias de la prueba, pero no reproduce el 18 % de avisos de prueba con un vecino de entrenamiento "
    "en la misma coordenada (19 % a menos de 1 m, como en la figura) ni el cambio de mercado de Chapinero."], tipo="neutra")
nota("Distancias que respaldan la lámina de validación.")

s, t_ = dependencia_espacial(deck, "Apéndice", "Apéndice")
nota(t_)

s = deck.nueva("Cómo cambia el aporte del área y del valor del suelo", etiqueta="Apéndice", seccion="Apéndice")
imagen(s, IMG / "dependencia.png", 0.55, 1.45, w=12.3, alt="Aporte SHAP del área y del valor del suelo en entrenamiento y prueba")
tarjeta(s, 0.55, 5.3, 6.0, 1.65, "Área", [
    "En Chapinero el área pesa más en ambos sentidos: resta más en los pequeños y suma más en los grandes (interacción del "
    "área con la ubicación)."], tipo="neutra")
tarjeta(s, 6.8, 5.3, 6.0, 1.65, "Valor del suelo", [
    "El aporte se aplana pasado el percentil 99 del entrenamiento: es una causa probable de la subestimación en Chapinero."],
    tipo="cuidado")
nota("Dependencias SHAP para preguntas sobre extrapolación. Estas curvas no son precios implícitos en el sentido de Rosen: "
     "muestran cómo reparte el modelo, y por encima del percentil 99 del entrenamiento el árbol ya no tiene datos.")

s = deck.nueva("¿Qué validación ordena bien los tres envíos?", etiqueta="Apéndice", seccion="Apéndice")
filas = [("Envío", "Chapinero fuera", "Clústeres k-means", "Kaggle, tabla privada", "Kaggle, tabla pública"),
         ("01. LightGBM con Catastro", n(MAE_CH), "125,0", n(P01), n(U01)),
         ("02. Con ajuste ×1,08", n(bo["lightgbm_x108"]["mae"]), "124,1", n(P02), n(U02)),
         ("03. Mezcla con CatBoost", n(bo["mezcla_40_60"]["mae"]), "123,6", n(P03), n(U03))]
tabla(s, 0.55, 1.5, 12.25, filas, [3.65, 2.15, 2.15, 2.15, 2.15], size=14, alto_fila=0.55)
tarjeta(s, 0.55, 4.0, 12.25, 1.6, "Cuidado al leerlo", [
    "Los pesos de la mezcla se eligieron en clústeres y el ×1,08 con Chapinero fuera: cada validación favorece al envío que "
    "ayudó a elegir.",
    f"Con 3 envíos y 307 avisos es una señal, no una prueba: 02 menos 01 = {n(bo['lightgbm_x108']['dif'], 1)} (intervalo de "
    f"95 %: {n(bo['lightgbm_x108']['dif_bajo'], 1)} a {n(bo['lightgbm_x108']['dif_alto'], 1)}); 03 menos 01 = "
    f"{n(bo['mezcla_40_60']['dif'], 1, True)} (intervalo de 95 %: {n(bo['mezcla_40_60']['dif_bajo'], 1)} a "
    f"{n(bo['mezcla_40_60']['dif_alto'], 1)}). Además, el bootstrap supone avisos independientes."],
    tipo="cuidado")
tarjeta(s, 0.55, 5.75, 12.25, 1.2, "Lo que sí es robusto", [
    "Kaggle y Chapinero fuera coinciden en el nivel del error (152 frente a 144) y en que la CV aleatoria es demasiado "
    "optimista."], tipo="neutra")
nota("Orden de los envíos con sus salvedades.")

s = deck.nueva("Fuentes", etiqueta="Apéndice", seccion="Apéndice")
refs = ["Datos: Properati (avisos de la competencia de Kaggle del curso); Catastro de Bogotá (valor de referencia del suelo, "
        "avalúos, pisos y estratificación); OpenStreetMap; Google Earth Engine (SRTM, Sentinel-2, VIIRS y GHSL).",
        "Ash, E. y Hansen, S. (2023). Text algorithms in economics. Annual Review of Economics, 15, 659-688.",
        "Blei, D. M., Ng, A. Y. y Jordan, M. I. (2003). Latent Dirichlet allocation. Journal of Machine Learning Research, 3, "
        "993-1022.",
        "Cañete, J. y otros (2020). Spanish pre-trained BERT model and evaluation data. PML4DC, ICLR.",
        "Gentzkow, M., Kelly, B. y Taddy, M. (2019). Text as data. Journal of Economic Literature, 57(3), 535-574.",
        "Hengl, T., Heuvelink, G. y Rossiter, D. (2007). About regression-kriging. Computers & Geosciences, 33(10), 1301-1315.",
        "James, G., Witten, D., Hastie, T. y Tibshirani, R. (2021). An Introduction to Statistical Learning, segunda edición. "
        "Springer, caps. 8 y 10.",
        "Ke, G. y otros (2017). LightGBM: a highly efficient gradient boosting decision tree. NeurIPS 30.",
        "Linnenbrink, J., Milà, C., Ludwig, M. y Meyer, H. (2024). kNNDM CV. Geoscientific Model Development, 17, 5897-5912.",
        "Lundberg, S. M. y Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS 30.",
        "Milà, C., Mateu, J., Pebesma, E. y Meyer, H. (2022). Nearest neighbour distance matching LOO CV. Methods in Ecology "
        "and Evolution, 13(6), 1304-1316.",
        "Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. Biometrika, 37(1/2), 17-23.",
        "Roberts, D. R. y otros (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or "
        "phylogenetic structure. Ecography, 40(8), 913-929.",
        "Rosen, S. (1974). Hedonic prices and implicit markets. Journal of Political Economy, 82(1), 34-55.",
        "Shapley, L. S. (1953). A value for n-person games. Contributions to the Theory of Games II, 307-317.",
        "Stanford GSB Insights (2021). Flip Flop: Why Zillow's algorithmic home buying venture imploded.",
        "Tobler, W. R. (1970). A computer movie simulating urban growth in the Detroit region. Economic Geography, 46, 234-240.",
        "Material del curso: Lectures 12 a 15; Validación cruzada espacial (Rojas y Sarmiento-Barbieri, 2026); Datos espaciales "
        "y Texto como datos (material de clase); enunciado del Problem Set 3."]
texto(s, 0.55, 1.45, 12.25, 5.5, refs, size=12, color="tinta", espacio=4)
nota("Referencias citadas.")

for i, sl in enumerate(deck.prs.slides):
    if i in NOTAS:
        deck.notas(sl, NOTAS[i])
deck.guardar(OUT / "best_equipo_02.pptx")
