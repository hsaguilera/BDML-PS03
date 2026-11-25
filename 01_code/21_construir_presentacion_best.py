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
    "e1.fuente1_1",                    # Fuente 1a. Avisos de Properati: el precio que se predice y la vivienda
    "e1.fuente1_2",                    # Fuente 1b. Avisos de Properati: texto, fecha, ubicación y avisos vecinos
    "e1.apertura_clase",               # Del texto a variables: nueve técnicas y dónde entra cada una
    "e1.pasos_modelos",                # Del texto crudo a 84 variables: pasos, modelos y lo que queda en los datos
    "e1.aviso_real",                   # Un aviso real: qué sacamos de su texto y con qué regla
    "e1.beto",                         # BETO: un modelo de lenguaje que corrige la ortografía según el contexto
    "e1.listas",                       # Listas de palabras que se quitan del texto: de dónde salen y cómo se usan
    "e1.fuente2ab_1",                  # Fuente 2a. Texto del aviso: marcas de sí o no por palabra clave, sin contar las negadas
    "e1.fuente2ab_2",                  # Fuente 2b. Texto del aviso: más marcas de sí o no y las cifras escritas
    "e1.tfidf",                        # TF-IDF: cuánto pesa cada término en cada aviso
    "e1.lsa",                          # LSA: la tabla de pesos resumida en 50 componentes
    "e1.lda",                          # LDA: cada aviso como una mezcla de 10 tópicos (temas)
    "e1.ridge",                        # Ridge: un puntaje de precio aprendido de las palabras
    "e1.fuente2c",                     # Fuente 2c. Texto del aviso: temas, resumen en 50 números y puntaje de precio
    "e1.externas_1",                   # Fuente 3a. OpenStreetMap: comercio y transporte alrededor del aviso
    "e1.externas_2",                   # Fuente 3b. OpenStreetMap: colegios, TransMilenio, oficinas y embajadas
    "e1.externas_3",                   # Fuente 3c. OpenStreetMap: distancias a parques, vías, servicios y referencias
    "e1.externas_4",                   # Fuente 4. Imágenes de satélite: 7 variables del terreno y de lo construido
    "e1.externas_5",                   # Fuente 5. Catastro: 11 variables del valor del suelo y de la manzana
    "e1.hedonicas",                    # 162 variables: el precio refleja la vivienda y su entorno (precios hedónicos)
    # Etapa 2. Algoritmo
    "e2.general_arbol",                # Un árbol de decisión: preguntas de sí o no que terminan en hojas
    "e2.general_boosting",             # Boosting: sumar muchos árboles pequeños que corrigen el error poco a poco
    "e2.boosting",                     # Datos reales y una sola variable: cada árbol acerca la predicción a los precios
    "e2.entrada",                      # Nuestro modelo recibe 162 variables por aviso y devuelve un precio en millones
    "e2.corte",                        # En nuestros datos, la mejor primera pregunta es el puntaje de precio del texto
    "e2.objetivo",                     # Lo que minimiza nuestro modelo: error cuadrático más una penalización
    "e2.hiperparametros",              # Hiperparámetros: qué controla cada uno, su valor y cómo se eligió
    "e2.busqueda",                     # Búsqueda al azar posterior de hiperparámetros (comprobación de la configuración final)
    "e2.parada",                       # La parada temprana eligió 252 árboles: parar cuando el error deja de bajar
    "e2.arbol",                        # Un aviso real recorre 241 árboles: del promedio, 653, a una predicción de 783
    "e2.shap_importancia",             # Qué variables pesan más: la importancia según los valores SHAP
    "e2.shap_enjambre",                # Cómo pesa cada variable: el gráfico de enjambre de los valores SHAP
    "e2.shap_cascada",                 # Una predicción explicada paso a paso: la cascada de un apartamento
    "e2.shap_dependencia",             # Dependencia: cómo cambia el aporte del área y del suelo con su valor
    # Etapa 3. Validación
    "e2.validar",                      # Validar: esconder avisos con precio, predecirlos y medir el error
    "e2.esquemas",                     # Seis formas de esconder avisos: del azar a dejar Chapinero fuera
    "e2.resultado_validacion",         # Resultado: solo dejar Chapinero fuera anticipa el error de Kaggle
    "e2.metricas",                     # Cuatro medidas del error en cada validación: MAE, RMSE, R² y sesgo
    "e2.predicho_real",                # Predicho frente a real: los 307 avisos de Chapinero que el modelo no vio
    "e2.vecinos",                      # Los errores se parecen entre vecinos: queda algo del lugar sin explicar
    # Etapa 4. Pronóstico y negocio, recomendación y fuentes
    "e2.pronostico",                   # Del modelo al pronóstico: cinco corridas, un ajuste y los gemelos de texto
    "e2.calibracion",                  # Calibración: en Chapinero predice de menos en todos los niveles; en Bogotá, no
    "e2.lados",                        # Predecir de más o de menos: el 27 % queda más de 10 % arriba del precio
    "e2.fuera_rango",                  # Fuera del rango del entrenamiento: más error en millones, no en porcentaje
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
