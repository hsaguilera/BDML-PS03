"""21 · Construye 03_slides/best_equipo_02.pptx (Best Model Deep Dive) con python-pptx, con las notas del orador.

Qué hace (antes document/presentaciones/construir_best_final.py)
    1. Crea todas las láminas disponibles: la portada; la lámina de resultados del mejor modelo; la Etapa 1, con el
       problema de negocio, la hoja de ruta y los datos (functions/laminas_etapa1.py); y las etapas 2 a 4 (algoritmo,
       evaluación con SHAP, validación, pronóstico y negocio), la recomendación y las fuentes (functions/laminas_etapas2a4.py).
    2. Deja solo las láminas de SELECCION, en ese orden. La numeración «i de N» y los pies se calculan sobre la selección.

Cómo cambiar las láminas
    Se edita la lista SELECCION: quitar una clave quita la lámina y moverla la mueve. Cada clave es el nombre de la función
    que crea la lámina (e1. = functions/laminas_etapa1.py, e2. = functions/laminas_etapas2a4.py; si una función crea varias
    láminas, llevan _1, _2, ...). Para ver todas las claves con su título:
        python 01_code/21_construir_presentacion_best.py --catalogo

Cifras: 00_data/processed/laminas/ (scripts 12 a 19 y 24 a 30; los de 24 a 30 se versionan). Imágenes:
02_outputs/figures/laminas/ (scripts 20, 32 y 33). Exportar a PDF: script 23 (opcional, Windows con PowerPoint).
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import sys  # noqa: E402

from functions import laminas_etapa1 as e1  # noqa: E402
from functions import laminas_etapas2a4 as e2  # noqa: E402
from functions.construir_final_comun import (DeckConClaves, ajustes_de_control, catalogo, portada,  # noqa: E402
                                             seleccionar)
from functions.entorno_ps3 import log  # noqa: E402
from functions.laminas_entorno import SALIDA_PPT  # noqa: E402  (03_slides/)

PPTX = SALIDA_PPT / "best_equipo_02.pptx"

# === INICIO DE LA CONFIGURACIÓN (se copia de los constructores de document/presentaciones/) ===
TITULO = "LightGBM: cómo predice los precios de vivienda en Chapinero"
SUBTITULO = ("Best Model Deep Dive: ¿podemos confiar en nuestro mejor modelo, un boosting de árboles con 162 variables de "
             "vivienda, entorno, Catastro y texto, para comprar vivienda en Chapinero?")
NOTAS_PORTADA = ("Pregunta de negocio: una empresa que compra vivienda quiere usar un modelo para decidir cuánto ofrecer en "
                 "Chapinero. Vamos a mostrar qué tan bien predice nuestro mejor modelo, cómo funciona y, sobre todo, dónde "
                 "falla y qué le cuesta eso a la empresa.")

# ------------------------------------------------------------------------------------------------------------------
# Láminas de la presentación, en orden (una clave por lámina; ver --catalogo)
# ------------------------------------------------------------------------------------------------------------------
SELECCION = [
    "portada",                         # LightGBM: cómo predice los precios de vivienda en Chapinero
    "e2.resultados_mejor_modelo",      # El modelo ordena bien Chapinero, pero no basta para ofertar sin tasación
    # Problema, hoja de ruta y Etapa 1 (datos)
    "e1.problema",                     # Para la empresa, pagar de más cuesta más que dejar de comprar
    "e1.ruta",                         # De los avisos al pronóstico: el camino completo
    "e1.mapa_precio",                  # Entrenamos con Bogotá, casi sin Chapinero, para predecir Chapinero
    "e1.hedonicas",                    # 162 variables: el precio refleja la vivienda y su entorno (precios hedónicos)
    "e1.pasos_modelos",                # Del texto crudo a 84 variables: pasos, modelos y lo que queda en los datos
    "e1.aviso_real",                   # Un aviso real: qué sacamos de su texto y con qué regla
    "e1.lsa_lda",                      # LSA y LDA: dos resúmenes de las palabras que no miran el precio
    "e1.ridge",                        # Ridge: el puntaje de precio del texto, aprendido de las palabras
    # Etapa 2. Algoritmo
    "e2.general_arbol",                # Un árbol de decisión: preguntas de sí o no que terminan en hojas
    # "e2.general_boosting",  (quitada a pedido del usuario: ejemplo de juguete confuso)             # Boosting: sumar muchos árboles pequeños que corrigen el error poco a poco
    "e2.corte",                        # En nuestros datos, la mejor primera pregunta es el puntaje de precio del texto
    "e2.hiperparametros",              # El modelo estimado: qué minimiza, sus hiperparámetros y cómo se eligieron
    "e2.parada",                       # La parada temprana eligió 252 árboles: parar cuando el error deja de bajar
    "e2.shap_importancia",             # Qué variables pesan más: la importancia según los valores SHAP
    "e2.shap_enjambre",                # Cómo pesa cada variable: el gráfico de enjambre de los valores SHAP
    "e2.shap_cascada",                 # Una predicción explicada paso a paso: la cascada de un apartamento
    "e2.shap_dependencia",             # Dependencia: cómo cambia el aporte del área y del suelo con su valor
    # Etapa 3. Validación
    "e2.esquemas",                     # Validar: esconder avisos con precio y predecirlos, de seis formas
    "e2.resultado_validacion",         # Resultado: solo dejar Chapinero fuera anticipa el error de Kaggle
    # Etapa 4. Pronóstico y negocio, recomendación y fuentes
    "e2.pronostico",                   # Del modelo al pronóstico: cinco corridas, un ajuste y los gemelos de texto
    "e2.calibracion",                  # Calibración: en Chapinero predice de menos en todos los niveles; en Bogotá, no
    "e2.lados",                        # Predecir de más o de menos: el 27 % queda más de 10 % arriba del precio
    "e2.zillow",                       # La trampa de Zillow: el dueño acepta cuando el modelo sobrestima
    "e2.recomendacion",                # Usar el modelo para filtrar y negociar, no para ofertar sin tasación
    "e2.fuentes",                      # Fuentes
]


def construir():
    deck = DeckConClaves("Best Model Deep Dive")
    portada(deck, TITULO, SUBTITULO, 18, 0.95, NOTAS_PORTADA)
    if hasattr(e2, "resultados_mejor_modelo"):
        s, notas = e2.resultados_mejor_modelo(deck)
        deck.notas(s, notas)
    else:
        print("aviso: laminas_etapas2a4.resultados_mejor_modelo no existe todavía; la lámina de resultados no se incluye")
    e1.etapa1(deck)
    e2.etapas2a4(deck)
    return deck


# === FIN DE LA CONFIGURACIÓN ===


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    deck = construir()
    if "--catalogo" in sys.argv:
        for k, (clave, tit) in enumerate(catalogo(deck), 1):
            print(f"{k:3d}  {clave:32s} {tit}")
        return
    # Si la lámina de resultados no existe en el módulo, se omite su clave en vez de fallar.
    seleccion = [k for k in SELECCION if k in deck.claves() or k != "e2.resultados_mejor_modelo"]
    ajustes_de_control(deck)
    quitadas = seleccionar(deck, seleccion)
    if quitadas:
        log(f"Láminas disponibles que no están en SELECCION (no se incluyen): {quitadas}")
    deck.guardar(PPTX)


if __name__ == "__main__":
    main()
