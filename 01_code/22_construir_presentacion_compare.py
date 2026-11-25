"""22 · Construye 03_slides/compare_equipo_02.pptx (Algorithm Comparison) con python-pptx, con las notas del orador.

Qué hace (antes document/presentaciones/construir_compare_final.py)
    1. Crea todas las láminas disponibles: la portada; la lámina del mejor modelo (functions/laminas_compare_nuevo.py); la
       Etapa 1, la misma de best, con el pie «Equipo 02. Algorithm Comparison» (functions/laminas_etapa1.py); y el cuerpo
       de la comparación de las seis familias, con sus conclusiones y fuentes (functions/laminas_compare_nuevo.py).
    2. Deja solo las láminas de SELECCION, en ese orden. La numeración «i de N» y los pies se calculan sobre la selección.

Cómo cambiar las láminas
    Se edita la lista SELECCION: quitar una clave quita la lámina y moverla la mueve. Cada clave es el nombre de la función
    que crea la lámina (e1. = functions/laminas_etapa1.py, cmp. = functions/laminas_compare_nuevo.py; si una función crea
    varias láminas, llevan _1, _2, ...). Para ver todas las claves con su título:
        python 01_code/22_construir_presentacion_compare.py --catalogo

Cifras: 00_data/processed/laminas/ (scripts 12 a 19 y 24 a 31; los de 24 a 31 se versionan) y
00_data/processed/modelos/ (script 07). Imágenes: 02_outputs/figures/laminas/ (scripts 20, 32 y 34). Exportar a PDF:
script 23 (opcional, Windows con PowerPoint).
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import sys  # noqa: E402

from functions import laminas_compare_nuevo as cn  # noqa: E402
from functions import laminas_etapa1 as e1  # noqa: E402
from functions.construir_final_comun import (DeckConClaves, ajustes_de_control, catalogo, portada,  # noqa: E402
                                             seleccionar)
from functions.entorno_ps3 import log  # noqa: E402
from functions.laminas_entorno import SALIDA_PPT  # noqa: E402  (03_slides/)

PPTX = SALIDA_PPT / "compare_equipo_02.pptx"

# === INICIO DE LA CONFIGURACIÓN (se copia de los constructores de document/presentaciones/) ===
TITULO = "Qué modelo predice mejor los precios de vivienda en Chapinero"
SUBTITULO = ("Algorithm Comparison: seis familias y siete modelos bajo las mismas reglas para predecir precios en "
             "Chapinero")
NOTAS_PORTADA = ("Comparamos seis familias de algoritmos para predecir el precio de vivienda en Chapinero, con las mismas "
                 "variables y las mismas validaciones. La pregunta es por qué gana el boosting y qué se sacrifica con cada "
                 "alternativa.")

# ------------------------------------------------------------------------------------------------------------------
# Láminas de la presentación, en orden (una clave por lámina; ver --catalogo)
# ------------------------------------------------------------------------------------------------------------------
SELECCION = [
    "portada",                         # Qué modelo predice mejor los precios de vivienda en Chapinero
    # El mejor modelo
    "cmp.mejor_modelo",                # El mejor modelo: boosting de árboles (LightGBM) con texto, entorno y Catastro
    # Problema y Etapa 1 (datos): las mismas láminas de best
    "e1.problema",                     # Para la empresa, pagar de más cuesta más que dejar de comprar
    "e1.mapa_precio",                  # Entrenamos con Bogotá, casi sin Chapinero, para predecir Chapinero
    "e1.hedonicas",                    # 162 variables: el precio refleja la vivienda y su entorno (precios hedónicos)
    "e1.pasos_modelos",                # Del texto crudo a 84 variables: pasos, modelos y lo que queda en los datos
    # Cuerpo de la comparación
    "cmp.criterios",                   # Cómo comparamos: el error en avisos que el modelo no vio
    "cmp.como_predice",                # Cómo predice cada familia y qué controla su flexibilidad
    "cmp.boosting",                    # Boosting: árboles pequeños en secuencia que corrigen el error anterior
    "cmp.red",                         # Red con atención: capas de neuronas que además miran el precio de 20 vecinos
    "cmp.hiperparametros",             # Hiperparámetros de cada familia: qué fija su flexibilidad y cómo se eligió
    "cmp.validaciones",                # Tres validaciones: del azar a predecir una zona que el modelo no vio
    "cmp.reglas",                      # Las mismas reglas para las seis familias
    "cmp.resultado",                   # Resultado: el boosting tiene el menor error en las tres validaciones
    "cmp.diferencias",                 # ¿Son reales las diferencias? Sí: ningún intervalo toca el cero
    "cmp.estandar_espacial",           # Validación estándar y espacial: el mismo ganador, distinto orden detrás
    "cmp.kaggle",                      # ¿Qué validación refleja Kaggle? En el nivel, Chapinero fuera
    "cmp.forma",                       # Por qué gana el boosting (1): el precio no crece en línea recta
    "cmp.promediar_corregir",          # Por qué gana el boosting (2): promediar baja la varianza; corregir baja el sesgo
    "cmp.subestiman",                  # Dónde fallan: todas subestiman Chapinero, y más los avisos que predicen caros
    "cmp.riesgo",                      # Para la empresa: un sesgo medio bajo no es un riesgo bajo
    "cmp.complejidad",                 # Qué movió el resultado: entre familias, el algoritmo; dentro del boosting, los datos
    "cmp.compromisos",                 # Compromisos: qué se gana y qué se pierde con cada familia
    "cmp.conclusion",                  # Gana el boosting, pero ninguna familia corrige el nivel de Chapinero
    "cmp.fuentes",                     # Fuentes
]


def construir():
    deck = DeckConClaves("Algorithm Comparison")
    portada(deck, TITULO, SUBTITULO, 19, 1.6, NOTAS_PORTADA)
    s, notas = cn._mejor_modelo(deck)
    deck.notas(s, notas)
    e1.etapa1(deck)
    cn.compare_cuerpo(deck)
    return deck


# === FIN DE LA CONFIGURACIÓN ===


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    deck = construir()
    if "--catalogo" in sys.argv:
        for k, (clave, tit) in enumerate(catalogo(deck), 1):
            print(f"{k:3d}  {clave:32s} {tit}")
        return
    ajustes_de_control(deck)
    quitadas = seleccionar(deck, SELECCION)
    if quitadas:
        log(f"Láminas disponibles que no están en SELECCION (no se incluyen): {quitadas}")
    deck.guardar(PPTX)


if __name__ == "__main__":
    main()
