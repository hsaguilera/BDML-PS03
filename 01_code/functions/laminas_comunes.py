"""Láminas que comparten best_equipo_02 y compare_equipo_02, y láminas de análisis que se arman con datos guardados.

- chapinero_otro_mercado: en qué difiere Chapinero del resto de Bogotá (Etapa 1; best y compare).
- variables_hedonicas: las 162 variables, con los predictores externos y de texto que pide el taller (Etapa 1; best y compare).
- texto_a_matriz: del texto crudo a la matriz documento término, con frecuencias y ley de Zipf (Etapa 1; best).
- texto_barrios_topicos: qué dice el texto (barrios que distinguen Chapinero y tópicos LDA) (Etapa 1; best).
- texto_apendice: LSA, LDA, TF-IDF, puntaje ridge fuera de pliegue y BETO (apéndice de best).
- dependencia_espacial: I de Moran, correlación del residuo, buffer y kriging (apéndice de best).
- error_por_familia: error de las siete especificaciones en las tres validaciones (apéndice de las dos).

Cada función crea la lámina en el Deck y devuelve (lámina, texto de las notas del orador). Solo leen archivos de datos/,
02_outputs/figures/laminas/ y 02_outputs/tables/ (script 05), y cifras de la bitácora citadas en el código: no cambian datos ni modelos.
"""
import json
import math
from pathlib import Path

import pandas as pd
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.util import Pt

from functions.ppt_lib import barras, chips, lineas_xy, circulo, etiquetas_puntos, forma, imagen, linea, rgb, tabla, tarjeta, texto

from functions.entorno_ps3 import DIR_MODELOS as STORES, DIR_TAB  # noqa: E402  (modelos/ y tablas)
from functions.laminas_entorno import DATOS as D, IMG  # noqa: E402  (datos e imágenes de las láminas)
FAM = [("lightgbm", "LightGBM"), ("catboost", "CatBoost"), ("random_forest", "Random Forest"), ("cart", "CART"),
       ("nn_atencion", "Red con atención"), ("elastic_net", "Elastic Net"), ("ols", "Regresión lineal")]


def n(x, d=1, signo=False):
    """Número con coma decimal, punto de miles y signo menos tipográfico."""
    s = f"{abs(x):,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if x < 0 and float(f"{abs(x):.{d}f}") != 0:
        return "−" + s
    return ("+" + s) if signo and float(f"{abs(x):.{d}f}") != 0 else s


def chapinero_otro_mercado(deck, etiqueta, seccion, contexto=""):
    v2 = json.load(open(D / "ppt_varios2.json", encoding="utf-8"))
    dch, dre = v2["desc_Chapinero"], v2["desc_Resto de Bogotá"]
    cch, cre = v2["catastro_Chapinero"], v2["catastro_Resto de Bogotá"]
    s = deck.nueva("Chapinero es otro mercado: más caro y fuera del rango conocido", etiqueta=etiqueta, seccion=seccion)
    paneles = [("Precio mediano por m² de los avisos de entrenamiento (millones)", dch["m2_mediana"], dre["m2_mediana"]),
               ("Avisos de entrenamiento con precio mayor a 1.200 millones (%)", dch["pct_sobre_1200"], dre["pct_sobre_1200"]),
               ("Manzanas de estrato 6 (%, capa de estratificación vigente)", cch["pct_estrato6"], cre["pct_estrato6"]),
               ("Valor de referencia del suelo, mediana de las manzanas (millones por m², Catastro 2021)",
                cch["vref_mediana_millones"], cre["vref_mediana_millones"])]
    for i, (tit, a_, b_) in enumerate(paneles):
        x0, y0 = 0.55 + (i % 2) * 3.95, 1.45 + (i // 2) * 2.05
        texto(s, x0, y0, 3.8, 0.6, tit, size=13, color="tinta")
        ch = barras(s, x0 - 0.05, y0 + 0.55, 3.8, 1.35, ["Chapinero", "Resto de Bogotá"],
                    [("v", [a_, b_], ["azul", "gris_claro"])], minimo=0, maximo=max(a_, b_) * 1.3, gap=35, grid=False)
        etiquetas_puntos(ch, [[n(a_, 1), n(b_, 1)]])
    tarjeta(s, 0.55, 5.6, 7.75, 1.35, None, [
        [(contexto + "El 57 % de los avisos de prueba tiene un valor del suelo a 300 m mayor que el percentil 99 del "
          "entrenamiento "
          "(5,9 millones por m²). ", {}),
         ("Un árbol predice con promedios de avisos que ya vio: ahí predice como en el borde de lo que conoce. Una regresión "
          "lineal sí extrapola, en línea recta, pero al predecir Chapinero sin haberlo visto yerra más: 185,7 millones de "
          "error absoluto medio frente a 143,8 de LightGBM.", {"color": "rojo"})]], tipo="cuidado")
    imagen(s, IMG / "mapa_valor_suelo.png", 8.45, 1.5, w=4.4, alt="Mapa del valor de referencia del suelo 2021 por manzana")
    notas = ("Chapinero no es una versión más cara del resto de Bogotá: es otro mercado. El precio por metro es un tercio "
             "mayor, hay casi tres veces más avisos de más de 1.200 millones, un tercio de sus manzanas son de estrato 6 "
             "frente a 1,4 % en el resto, y el suelo vale casi tres veces más. Lo más importante para el modelo: más de la "
             "mitad de la prueba tiene un valor del suelo por encima del percentil 99 del entrenamiento, porque los avisos "
             "de prueba se concentran en la parte más cara de la localidad, y un modelo de árboles no extrapola.")
    return s, notas


def variables_hedonicas(deck, etiqueta, seccion, nota_extra="", texto_extra=""):
    s = deck.nueva("162 variables con la lógica de precios hedónicos", etiqueta=etiqueta, seccion=seccion)
    imagen(s, IMG / "f_rosen.png", 0.6, 1.55, h=0.42, alt="Ecuación de precios hedónicos de Rosen")
    texto(s, 0.55, 2.05, 7.3, 0.85, "Rosen (1974): el precio refleja las características; la forma de f se aprende de los "
          "datos. La ubicación entra de dos maneras: como variables, en esta tabla, y como dependencia entre vecinos, que "
          "obliga a validar por lugar (Lecture 14)." + (" " + texto_extra if texto_extra else ""), size=13, color="muted")
    filas = [("Bloque", "Variables", "Qué mide y de dónde sale"),
             ("Vivienda", "15", "área, baños, alcobas, parqueaderos y tipo; completados con el texto"),
             ("Entorno", "50", "OpenStreetMap: TransMilenio, parques, oficinas, hoteles, embajadas, con distancias en "
                               "metros (EPSG:9377). Imágenes de satélite (Earth Engine): elevación, luces nocturnas, vegetación"),
             ("Catastro", "11", "valor del suelo, avalúo y pisos de la manzana en el año del aviso, y estrato (capa "
                                "vigente); en la manzana y a 300 m"),
             ("Texto", "84", "19 indicadores con negación, 4 cantidades, 10 tópicos LDA (palabras que suelen ir juntas), "
                             "50 componentes LSA (el PCA de la matriz TF-IDF) y un puntaje ridge"),
             ("Ubicación", "2", "coordenadas este y norte")]
    tabla(s, 0.55, 3.0, 7.3, filas, [1.3, 1.1, 4.9], size=13, alto_fila=0.6)
    texto(s, 8.25, 1.45, 4.6, 1.5, "El puntaje de precio del texto: una regresión ridge del logaritmo del precio sobre la "
          "matriz TF-IDF, estimada fuera de pliegue: ni el precio del aviso ni el de sus copias de texto entran a su puntaje.",
          size=14, color="navy")
    texto(s, 8.25, 3.0, 4.6, 0.4, "Palabras que más suben el puntaje", size=14, color="azul")
    y_ = chips(s, 8.25, 3.42, 4.6, ["3 o 4 parqueaderos", "ascensor privado", "Chicó", "Carolina", "penthouse", "terraza",
                                    "casa", "oficina"], relleno="azul_claro", size=13, alto=0.38, sep=0.09)
    texto(s, 8.25, y_ + 0.12, 4.6, 0.4, "Palabras que más lo bajan", size=14, color="naranja")
    chips(s, 8.25, y_ + 0.54, 4.6, ["apartaestudio", "1 parqueadero", "habitación", "Cedritos", "Kennedy", "Hayuelos",
                                    "sala comedor", "dos baños"], relleno="ambar_claro", size=13, alto=0.38, sep=0.09)
    notas = ("Con la lógica de precios hedónicos de Rosen ampliamos los datos a 162 variables en cinco bloques: vivienda, "
             "entorno de OpenStreetMap y de imágenes de satélite, Catastro por manzana, texto y ubicación. El taller pide al "
             "menos cuatro predictores externos y cuatro de texto: aquí hay 61 externos, entre entorno y Catastro, y 84 de "
             "texto. El puntaje de texto es un ridge del logaritmo del precio sobre la matriz TF-IDF, estimado fuera de "
             "pliegue: el precio de un aviso y el de sus copias nunca entran a su propio puntaje. Varias palabras son "
             "barrios, así que el puntaje también mide ubicación: es útil, pero Ash y Hansen (2023) advierten que un "
             "predictor así puede fallar en un dominio nuevo, y los barrios propios de Chapinero casi no aparecen en el "
             "entrenamiento." + (" " + nota_extra if nota_extra else ""))
    return s, notas


def texto_a_matriz(deck, etiqueta, seccion):
    lim = json.load(open(D / "ppt_texto_limpieza.json", encoding="utf-8"))
    frec = pd.read_csv(D / "ppt_texto_frecuentes.csv")
    s = deck.nueva(f"Del texto crudo a una matriz de {n(lim['avisos'], 0)} avisos por {n(lim['terminos'], 0)} términos",
                   etiqueta=etiqueta, seccion=seccion)
    pasos = [("Reparar la codificación. ", f"La ñ y las tildes se perdieron en origen: se repararon {n(lim['reparaciones'], 0)} "
              f"términos; bao, que es baño, estaba en {n(lim['ejemplos_reparacion'][0][2], 0)} avisos."),
             ("Corregir la ortografía con BETO. ", "Un modelo de lenguaje en español propone la palabra más probable en su "
              f"contexto: se aceptaron {n(lim['beto_aceptadas'], 0)} de {n(lim['beto_candidatas'], 0)} candidatas, como "
              "apartmamento por apartamento: la tarea de la palabra enmascarada (Ash y Hansen, 2023)."),
             ("Quitar palabras vacías sin perder la negación. ", "Se eliminan de, la o con, pero se conservan no, sin y ni: "
              "sin ascensor no cuenta como ascensor (Texto como datos; Lecture 15)."),
             ("Reducir a la raíz. ", "Con Snowball, apartamento y apartamentos cuentan como un solo término."),
             ("Contar y ponderar. ", f"Palabras y pares de palabras presentes en 5 avisos o más ({n(lim['unigramas'], 0)} y "
              f"{n(lim['bigramas'], 0)}); cada conteo se pondera por lo rara que es la palabra entre los avisos (TF-IDF, "
              "Lecture 15).")]
    for i, (tit, desc) in enumerate(pasos):
        y0 = 1.5 + i * 1.02
        circulo(s, 0.55, y0 + 0.02, 0.48, i + 1, relleno="navy", size=16)
        texto(s, 1.15, y0, 5.45, 0.98, [[(tit, {"color": "azul"}), (desc, {})]], size=13, color="tinta")
    acentos = {"bano": "baño", "deposito": "depósito"}
    uni = frec[frec["tipo"] == "palabra"].head(10)
    cats = [acentos.get(w, w) for w in uni["legible"]]
    texto(s, 7.05, 1.45, 5.75, 0.4, "Palabras más frecuentes (% de avisos; cada raíz agrupa sus variantes)", size=13,
          color="muted")
    ch = barras(s, 6.95, 1.8, 5.85, 2.8, cats, [("Avisos (%)", list(uni["pct"]), "azul")], minimo=0, maximo=100, gap=35,
                grid=False)
    etiquetas_puntos(ch, [[n(v, 0) for v in uni["pct"]]])
    imagen(s, IMG / "texto_zipf.png", 7.0, 4.68, w=5.8, alt="Frecuencia documental de los términos por rango, en escala logarítmica")
    texto(s, 7.05, 6.45, 5.75, 0.5, "Pocas palabras aparecen en casi todos los avisos y la mayoría en muy pocos, como en la "
          "ley de Zipf (Ash y Hansen, 2023): por eso conviene ponderar por rareza.", size=12, color="muted")
    notas = ("Antes de usar el texto hubo que limpiarlo. La base perdió la ñ y las tildes, así que reparamos más de mil "
             "términos, como bao por baño. Luego corregimos la ortografía con BETO, un modelo de lenguaje en español que "
             "elige la palabra más probable en el contexto. Quitamos palabras vacías, pero conservamos no, sin y ni para "
             "leer la negación; reducimos cada palabra a su raíz y contamos palabras y pares de palabras. El resultado es la "
             "matriz documento término de la Lecture 15, con los avisos de entrenamiento y de prueba, lo que es legítimo porque "
             "no usa el precio. Como la mayoría de los términos aparecen en muy pocos avisos, cada "
             "conteo se pondera por rareza: eso es TF-IDF.")
    return s, notas


DISTINTIVOS = [("rosal", "Rosales"), ("chic nort", "Chicó Norte"), ("virrey", "Virrey"), ("parqu 93", "Parque 93"),
               ("chic reserv", "Chicó Reservado"), ("cabrer", "Cabrera"), ("usaqun", "Usaquén"), ("bell suiz", "Bella Suiza"),
               ("callej", "Calleja"), ("colin campestr", "Colina Campestre"), ("aven boyac", "Avenida Boyacá"),
               ("lisbo", "Lisboa")]
TOPICOS = {0: "Agencias en Chapinero y Carolina", 1: "Conjunto con amenidades", 2: "Acabados y dotación",
           3: "Plataforma Houm", 4: "Santa Bárbara y cifras en letras", 5: "Cercanía a comercio",
           6: "Redacción publicitaria", 7: "Casas", 8: "Apartamento y servicios del edificio", 9: "Contacto de inmobiliarias"}


def texto_barrios_topicos(deck, etiqueta, seccion):
    dist = pd.read_csv(D / "ppt_texto_chapinero.csv").set_index("termino")
    top = pd.read_csv(D / "ppt_texto_topicos.csv")
    s = deck.nueva("El texto dice dónde está el aviso y qué tipo de vivienda es", etiqueta=etiqueta, seccion=seccion)
    texto(s, 0.55, 1.4, 5.95, 0.4, "Términos que más separan la prueba del entrenamiento (% de avisos)", size=13,
          color="muted")
    cats = [nom for _, nom in DISTINTIVOS]
    v_ch = [dist.loc[t, "pct_chapinero"] for t, _ in DISTINTIVOS]
    v_re = [dist.loc[t, "pct_resto"] for t, _ in DISTINTIVOS]
    ch = barras(s, 0.45, 1.75, 6.05, 3.5, cats, [("Prueba (Chapinero)", v_ch, "naranja"),
                                                  ("Entrenamiento fuera de Chapinero", v_re, "gris_claro")],
                minimo=0, maximo=20, leyenda=True, gap=45, overlap=-5, grid=False)
    etiquetas_puntos(ch, [[n(v, 1) for v in v_ch], [n(v, 1) for v in v_re]])
    top = top.assign(prom=(top["media_chapinero"] + top["media_resto"]) / 2).sort_values("prom", ascending=False)
    texto(s, 6.85, 1.4, 5.95, 0.4, "Tópicos LDA: proporción media en los avisos (%)", size=13, color="muted")
    ch = barras(s, 6.75, 1.75, 6.05, 3.5, [TOPICOS[k] for k in top["topico"]],
                [("Prueba (Chapinero)", list(100 * top["media_chapinero"]), "naranja"),
                 ("Entrenamiento fuera de Chapinero", list(100 * top["media_resto"]), "gris_claro")],
                minimo=0, maximo=55, leyenda=True, gap=45, overlap=-5, grid=False)
    etiquetas_puntos(ch, [[n(100 * v, 0) for v in top["media_chapinero"]], [n(100 * v, 0) for v in top["media_resto"]]])
    rho5 = float(top.loc[top["topico"] == 5, "rho_precio"].iloc[0])
    texto(s, 0.55, 5.3, 12.25, 0.75, "LDA agrupa en tópicos las palabras que suelen aparecer juntas; cada aviso es una mezcla "
          "de tópicos, y el nombre de cada uno lo pone el equipo. K = 10 tuvo la menor perplejidad fuera de muestra, es decir, "
          "predijo mejor las palabras de avisos no usados (Lecture 15; Blei y otros, 2003). El de cercanía a comercio es el más "
          f"ligado al precio en el entrenamiento, con signo negativo (Spearman {n(rho5, 2)}): los avisos con más peso en ese "
          "tópico son más baratos.", size=12, color="muted")
    tarjeta(s, 0.55, 6.08, 12.25, 0.87, None, [
        "El texto codifica la ubicación (barrios) y el tipo de vivienda (casas, conjuntos, edificios con servicios). Los "
        "barrios propios de Chapinero casi no aparecen en el entrenamiento: el modelo los lee, pero no tiene precios para "
        "aprenderlos, y un predictor así puede fallar en un dominio nuevo (Ash y Hansen, 2023)."], tipo="clave", maximo=15)
    notas = ("¿Qué información trae el texto? Primero, ubicación: Rosales, Chicó, El Virrey o el Parque 93 aparecen en una "
             "parte importante de los avisos de prueba y casi nunca en el entrenamiento; Usaquén, Cedritos o Bella Suiza, al "
             "revés. Segundo, tipo de vivienda: con LDA, un modelo que agrupa palabras que aparecen juntas, encontramos 10 "
             "tópicos. Chapinero tiene más descripciones de apartamentos con servicios de edificio y casi nada de casas o de "
             "conjuntos con amenidades. La consecuencia para el modelo: lee esos barrios, pero no tiene precios de ellos "
             "para aprender cuánto valen. Si preguntan por K: la perplejidad casi no cambia entre 5 y 20 tópicos (de 327,0 a "
             "328,6); como dice la Lecture 15, no hay un criterio objetivo para el número de componentes.")
    return s, notas


def texto_apendice(deck, etiqueta, seccion):
    s = deck.nueva("Del texto a 84 variables: resumen, puntaje fuera de pliegue y BETO", etiqueta=etiqueta, seccion=seccion)
    texto(s, 0.55, 1.45, 5.95, 1.05, [[("Resumir sin usar el precio. ", {"color": "navy"}),
                                       ("LSA: 50 componentes de la matriz TF-IDF, el PCA del texto (Lecture 15). LDA: 10 "
                                        "tópicos; K = 10 minimizó la perplejidad fuera de muestra entre 5, 10, 15, 20 y 25 "
                                        "(Blei y otros, 2003).", {})]], size=13, color="tinta")
    imagen(s, IMG / "f_tfidf.png", 0.65, 2.6, h=0.62, alt="Fórmula de TF-IDF")
    texto(s, 0.55, 3.3, 5.95, 0.7, "tf: frecuencia del término en el aviso; df: avisos que lo usan; D: total de avisos. En el "
          "código, la versión suavizada de scikit-learn, con tf sublineal.", size=12, color="muted")
    imagen(s, IMG / "f_ridge.png", 0.65, 4.15, h=0.55, alt="Problema de la regresión ridge")
    texto(s, 0.55, 4.8, 5.95, 0.95, "Puntaje de precio: ridge del logaritmo del precio sobre la matriz TF-IDF, con λ = 1 "
          "fijo. Gentzkow y otros (2019, pp. 543 y 544) recomiendan elegirlo por validación cruzada; aquí no se hizo. Entra al "
          "modelo como una variable más.", size=12, color="muted")
    tarjeta(s, 6.75, 1.5, 6.05, 1.7, "Fuera de pliegue", [
        "El puntaje de cada aviso sale de un ridge estimado sin ese aviso ni sus copias de texto, que van juntas en el mismo "
        "pliegue, y se recalcula dentro de cada pliegue de validación. Si no, el modelo vería el precio a través del texto: "
        "el 32 % de las descripciones del entrenamiento se repite."], tipo="definicion")
    tarjeta(s, 6.75, 3.35, 6.05, 1.6, "Por qué ridge y luego árboles", [
        "Con miles de palabras correlacionadas, promediar predice mejor que seleccionar (Gentzkow y otros, 2019, p. 544): el "
        "ridge superó a un Elastic Net (correlación con el precio de 0,81 frente a 0,76 en un pliegue, los dos sin afinar). "
        "Los árboles van al final, sobre el texto ya resumido (p. 546): es lo que hace LightGBM."], tipo="recordar")
    tarjeta(s, 6.75, 5.1, 6.05, 1.85, "BETO: dos caminos, ninguno se usó", [
        "Congelado, como variables: mejoró Chapinero fuera (149,0 a 146,3) y empeoró la zona cercana (128,5 a 129,4), la "
        "regla fijada antes, así que quedó fuera. Con ajuste fino, al revés (zona cercana de 128,2 a 127,1; Chapinero de "
        "147,1 a 149,7): la regla lo elegía, pero sus pliegues no coincidían con los de las validaciones finales y usarlo "
        "habría filtrado precios (fuga). Medido sin Catastro."], tipo="neutra")
    notas = ("Detalle del texto para preguntas: cómo se resume (LSA y LDA), cómo se calcula el puntaje sin fuga y por qué "
             "BETO no entró al modelo final. El λ del ridge quedó fijo en 1; la clase recomienda elegirlo por validación "
             "cruzada, y lo decimos de frente. Reajustar BETO para cada esquema de validación eran unas 34 corridas en GPU, y "
             "el ajuste fino además empeoraba Chapinero, así que no lo forzamos.")
    return s, notas


def dependencia_espacial(deck, etiqueta, seccion):
    mo = pd.read_csv(DIR_TAB / "moran_global.csv", index_col=0)
    sel = [("luces_nocturnas", "Luces nocturnas", "gris"), ("elevacion_m", "Elevación", "gris"),
           ("ndvi_200m", "Vegetación (NDVI, 200 m)", "gris"), ("log_precio", "Logaritmo del precio", "azul"),
           ("precio_mill", "Precio", "azul"), ("precio_m2_mill", "Precio por m²", "azul")]
    s = deck.nueva("Cuánto y hasta dónde se parecen los vecinos", etiqueta=etiqueta, seccion=seccion)
    texto(s, 0.55, 1.4, 6.0, 0.4, "I de Moran con 10 vecinos (todas con p = 0,005 por permutación)", size=13, color="muted")
    vals = [float(mo.loc[k, "I de Moran"]) for k, _, _ in sel]
    ch = barras(s, 0.45, 1.75, 6.05, 2.5, [nom for _, nom, _ in sel], [("I", vals, [c for _, _, c in sel])], minimo=0,
                maximo=1.1, gap=35, grid=False)
    etiquetas_puntos(ch, [[n(v, 2) for v in vals]])
    tabla(s, 6.75, 1.5, 6.05, [("Correlación del residuo de LightGBM entre avisos", "Correlación"),
                               ("En la misma coordenada", "0,31"), ("De 1 a 100 m", "0,11"), ("De 100 a 300 m", "0,05"),
                               ("De 500 m a 1 km", "cerca de 0")], [4.45, 1.6], size=13, alto_fila=0.36)
    tabla(s, 6.75, 3.4, 6.05, [("Con el promedio de sus 10 vecinos", "Correlación"), ("CV aleatoria", "0,11"),
                               ("Bloques de 2 km", "0,43"), ("Clústeres k-means", "0,46")], [4.45, 1.6], size=13,
          alto_fila=0.36)
    texto(s, 0.55, 4.3, 6.0, 0.7, "Gris: capas de satélite; azul: precio. La tabla de 10 vecinos replica el diagnóstico del "
          "cuaderno de la Lecture 14 (en Ames, 0,60). Residuo: error fuera de pliegue de LightGBM con Catastro (cuaderno 01 "
          "del equipo).", size=12, color="muted")
    tarjeta(s, 0.55, 5.02, 3.95, 1.93, "Dependencia espacial", [
        "Lo cercano se parece más (Tobler, 1970; Lecture 14), y el I de Moran (1950) lo mide. El precio se parece al de sus "
        "vecinos hasta 3 a 3,5 km; el residuo, menos de 1 km: las variables ya recogen la gran escala."], tipo="definicion")
    tarjeta(s, 4.7, 5.02, 3.95, 1.93, "Ancho del buffer", [
        "El buffer retira del entrenamiento los avisos cercanos a los validados. Debería medir de 500 m a 1 km (Lecture 14); "
        "el de 300 m, que probamos con bloques de 2 km (123,1), es corto. Pero el ancho no explica la brecha con Kaggle: los "
        "clústeres, con el vecino mediano a 985 m, dan 125,0 frente a 152,4."],
        tipo="recordar")
    tarjeta(s, 8.85, 5.02, 3.95, 1.93, "Por qué no ayudó el kriging", [
        "En el variograma del residuo, la pepita (varianza sin estructura, unos 22.500) es siete veces la parte estructurada "
        "(unos 3.200), y su alcance práctico es cerca de 1 km (tres veces el parámetro de 320 m): casi no queda correlación "
        "que interpolar."], tipo="neutra")
    notas = ("Evidencia de la Etapa 3 para preguntas. El precio está correlacionado entre vecinos hasta 3 a 3,5 km, pero el "
             "error del modelo solo hasta 500 m a 1 km: las variables recogen la dependencia de gran escala, y por eso el "
             "buffer se mide con el residuo, como en el cuaderno de clase. Con pliegues espaciales, el error de un aviso se "
             "parece al de sus vecinos (0,43 a 0,46): hay estructura que el modelo no ve, y una validación aleatoria la "
             "esconde. Romper la dependencia corrige el optimismo por tener vecinos en el entrenamiento, no el de predecir "
             "otro mercado: por eso la validación que anticipa Kaggle es dejar Chapinero fuera.")
    return s, notas


def error_por_familia(deck, etiqueta, seccion):
    cvf = pd.read_csv(D / "cv_familias.csv")
    bo = json.load(open(D / "ppt_bootstrap2.json", encoding="utf-8"))
    M = {(r.familia, r.esquema): r.mae for r in cvf.itertuples()}
    S = {(r.familia, r.esquema): r.sesgo for r in cvf.itertuples()}
    AL, CL, CH = "CV aleatoria", "Clústeres k-means", "Chapinero fuera"
    s = deck.nueva("Error por familia y validación", etiqueta=etiqueta, seccion=seccion)
    fil = [("Familia", "CV aleatoria", "Clústeres k-means", "Chapinero fuera", "Intervalo de 95 %, Chapinero fuera",
            "Sesgo, Chapinero fuera")]
    for f, nom in FAM:
        b = bo[f]
        fil.append((nom, n(M[(f, AL)]) if (f, AL) in M else "no se corrió", n(M[(f, CL)]), n(M[(f, CH)]),
                    f"{n(b['ic_bajo'])} a {n(b['ic_alto'])}", n(S[(f, CH)])))
    tabla(s, 0.55, 1.45, 12.25, fil, [2.6, 1.9, 2.0, 1.9, 2.15, 1.7], size=14, alto_fila=0.52)
    texto(s, 0.55, 6.05, 12.25, 0.85, "Seis familias: LightGBM y CatBoost son dos versiones del boosting. Error absoluto medio "
          "(MAE) en millones; sesgo: predicción menos precio real. Mismas "
          "162 variables y mismos pliegues para todas. Intervalo por bootstrap de los 307 avisos (10.000 réplicas), que trata "
          "los avisos como independientes. 38.644 avisos en cada validación general y 307 en Chapinero fuera.", size=13,
          color="muted")
    notas = ("Tabla completa para preguntas. LightGBM es el mejor de las seis familias con la CV aleatoria y con Chapinero "
             "fuera; CatBoost, la otra implementación de boosting, gana en clústeres por menos de un millón.")
    return s, notas



def _caja_dag(s, x, y, w, h, titulo, detalle, relleno="tinte", borde="gris_claro", color_titulo="navy"):
    caja = forma(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, relleno=relleno, linea=borde, radio=0.1)
    texto(s, x + 0.06, y + 0.05, w - 0.12, 0.34, titulo, size=13, color=color_titulo, align="center")
    texto(s, x + 0.06, y + 0.36, w - 0.12, h - 0.4, detalle, size=12, color="tinta", align="center")
    return caja


def _arista(s, origen, destino, lado_origen=3, lado_destino=1, recta=False):
    """Conector sin punta entre dos cajas (sitios: 0 arriba, 1 izquierda, 2 abajo, 3 derecha)."""
    con = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT if recta else MSO_CONNECTOR.ELBOW, 0, 0, 0, 0)
    con.begin_connect(origen, lado_origen)
    con.end_connect(destino, lado_destino)
    con.line.color.rgb = rgb("gris")
    con.line.width = Pt(1.5)
    return con


def ruta_al_pronostico(deck, etiqueta, seccion):
    """DAG del proceso: de las fuentes de datos al pronóstico y a su uso para la empresa."""
    s = deck.nueva("De los avisos al pronóstico: qué alimenta a qué", etiqueta=etiqueta, seccion=seccion)
    cols = [(0.55, 2.1), (3.0, 2.2), (5.55, 1.85), (7.75, 2.3), (10.5, 2.3)]
    cabezas = [("Fuentes", 0, 0), ("Variables", 1, 1), ("Unión", 2, 2), ("Modelos y validación", 3, 3), ("Resultado", 4, 4)]
    for nombre, a, b in cabezas:
        x = cols[a][0]
        texto(s, x, 1.33, cols[b][0] + cols[b][1] - x, 0.36, nombre, size=13, color="muted", align="center")
    X = lambda c: cols[c][0]  # noqa: E731
    Wc = lambda c: cols[c][1]  # noqa: E731
    hA, yA = 0.86, [1.75, 2.79, 3.83, 4.87, 5.91]
    fuentes = [_caja_dag(s, X(0), y, Wc(0), hA, t, d) for y, (t, d) in zip(yA, [
        ("Avisos de Properati", "38.644 de entrenamiento y 10.286 de prueba"),
        ("Texto del aviso", "título y descripción"),
        ("OpenStreetMap", "comercio, transporte, embajadas y cerros"),
        ("Imágenes de satélite", "luces, elevación y vegetación"),
        ("Catastro de Bogotá", "valor del suelo, avalúo, pisos y estrato")])]
    hB = 0.95
    variables = [_caja_dag(s, X(1), y, Wc(1), hB, t, d) for y, (t, d) in zip([1.75, 3.0, 4.4, 5.82], [
        ("Vivienda y ubicación (17)", "área, baños, alcobas y coordenadas"),
        ("Texto (84)", "BETO, TF-IDF, LSA, LDA y puntaje ridge"),
        ("Entorno (50)", "conteos y distancias alrededor del aviso"),
        ("Catastro (11)", "manzana y radio de 300 m")])]
    union = _caja_dag(s, X(2), 3.55, Wc(2), 1.25, "162 variables", "las mismas para todos los modelos",
                      relleno="azul_claro", borde="azul", color_titulo="azul")
    familias = _caja_dag(s, X(3), 2.55, Wc(3), 1.05, "Seis familias", "lineal, Elastic Net, CART, Random Forest, "
                         "boosting y red")
    validacion = _caja_dag(s, X(3), 4.75, Wc(3), 1.05, "Seis validaciones", "de la CV aleatoria a dejar Chapinero fuera")
    yE, hE = [1.75, 3.0, 4.25, 5.5], 1.05
    gana = _caja_dag(s, X(4), yE[0], Wc(4), hE, "Gana LightGBM", "143,8 en Chapinero fuera; ajuste ×1,08")
    final = _caja_dag(s, X(4), yE[1], Wc(4), hE, "Modelo final", "38.644 avisos, 5 corridas y gemelos de texto")
    prono = _caja_dag(s, X(4), yE[2], Wc(4), hE, "Pronóstico", "10.286 avisos; Kaggle 150,8, segundo de seis",
                      relleno="verde_claro", borde="verde", color_titulo="verde")
    uso = _caja_dag(s, X(4), yE[3], Wc(4), hE + 0.25, "Uso para la empresa", "ordena bien (0,87); ofertar solo pierde "
                    "14,8 millones por oferta")
    for o, d in [(0, 0), (0, 2), (1, 0), (1, 1), (2, 2), (3, 2), (4, 3)]:
        _arista(s, fuentes[o], variables[d])
    for v in variables:
        _arista(s, v, union)
    _arista(s, union, familias)
    _arista(s, union, validacion)
    _arista(s, familias, gana)
    _arista(s, validacion, gana)
    for a, b in [(gana, final), (final, prono), (prono, uso)]:
        _arista(s, a, b, 2, 0, recta=True)
    texto(s, 0.55, 6.85, 9.6, 0.3, "Se lee de izquierda a derecha: cada conexión une una entrada con lo que produce. "
          "Entre paréntesis, el número de variables.", size=12, color="muted")
    notas = ("Este es el camino completo, de izquierda a derecha. Partimos de cinco fuentes: los avisos de Properati, su "
             "texto, OpenStreetMap, imágenes de satélite y Catastro. Con ellas armamos cuatro grupos de variables, 162 en "
             "total, que reciben por igual todos los modelos. Comparamos seis familias con seis esquemas de validación; ganó "
             "LightGBM, y dejar Chapinero fuera eligió el ajuste ×1,08. El modelo final, entrenado con los 38.644 avisos, "
             "da el pronóstico de los 10.286 avisos de prueba, con el que quedamos segundos de seis en Kaggle, y lo "
             "traducimos a lo que le sirve a la empresa. El resto de la presentación recorre este camino.")
    return s, notas


def _caja_texto(s, x, y, w, h, titulo, parrafos, relleno="tinte", borde="gris_claro", color_titulo="navy"):
    caja = forma(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, relleno=relleno, linea=borde, radio=0.06)
    texto(s, x + 0.16, y + 0.1, w - 0.32, 0.4, titulo, size=15, color=color_titulo)
    texto(s, x + 0.16, y + 0.52, w - 0.32, h - 0.6, parrafos, size=14, color="tinta", espacio=6)
    return caja


def problema_negocio(deck, etiqueta, seccion):
    """El problema de negocio con el mismo lenguaje visual del DAG de la lámina siguiente."""
    s = deck.nueva("Para la empresa, pagar de más cuesta más que dejar de comprar", etiqueta=etiqueta, seccion=seccion)
    empresa = _caja_texto(s, 0.55, 1.9, 3.35, 2.9, "La empresa", [
        "Quiere comprar la mayor cantidad de viviendas en Chapinero gastando lo menos posible.",
        "Decide cuánto ofrecer con el precio que predice el modelo."])
    mas = _caja_texto(s, 4.45, 1.6, 4.35, 2.0, "Predecir de más", [
        "Ofrece de más y compra caro: pierde capital al revender y carga costos de tenencia.",
        "Zillow cerró su negocio de compra en 2021 con pérdidas de cerca de 500 millones de dólares."],
        relleno="rojo_claro", borde="rojo", color_titulo="rojo")
    menos = _caja_texto(s, 4.45, 3.8, 4.35, 1.4, "Predecir de menos", [
        "La oferta queda corta y el dueño no vende: se pierde un negocio, pero no capital."],
        relleno="azul_claro", borde="azul", color_titulo="azul")
    meta = _caja_texto(s, 9.35, 1.9, 3.45, 2.9, "Qué debe lograr el modelo", [
        "Predecir el precio de 10.286 avisos de Chapinero aprendiendo de 38.644 avisos de Bogotá, de los que solo 307 "
        "están en Chapinero.",
        "Y mostrar hacia dónde se equivoca, porque el error hacia arriba es el caro."])
    for a, b in [(empresa, mas), (empresa, menos), (mas, meta), (menos, meta)]:
        _arista(s, a, b)
    texto(s, 0.55, 5.55, 12.25, 0.75, ["Kaggle mide el error absoluto medio, que trata igual los dos errores: es un resultado "
                                       "más, no el objetivo. La lámina siguiente muestra el camino completo, de los datos al "
                                       "pronóstico."], size=13, color="muted")
    notas = ("Empecemos por el problema. Una empresa quiere comprar muchas viviendas en Chapinero gastando lo menos posible, y "
             "decide cuánto ofrecer con el precio que predice el modelo. Los dos errores no cuestan lo mismo: si el modelo "
             "predice de más, la empresa compra caro y pierde capital, que fue lo que le pasó a Zillow en 2021; si predice de "
             "menos, el dueño no vende y se pierde un negocio, pero no plata. Por eso el modelo tiene que predecir bien un "
             "lugar que casi no vio, solo 307 de 38.644 avisos, y tenemos que saber hacia dónde se equivoca. Kaggle trata igual "
             "los dos errores; para nosotros es un resultado más.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# Las cinco fuentes de datos: qué variables salen de cada una y cuántas suman hasta 162
# ------------------------------------------------------------------------------------------------------------------
FUENTES = [("Avisos de Properati", 21), ("Texto del aviso", 84), ("OpenStreetMap", 39), ("Imágenes de satélite", 7),
           ("Catastro", 11)]
GRUPOS_F = [("Prueba (Chapinero)", "naranja"), ("Entrenamiento fuera de Chapinero", "gris_claro")]


def _contador(s, k, leyenda=True):
    """Barra de 162 variables dividida por fuente; resalta la fuente k (0 a 4)."""
    total = sum(c for _, c in FUENTES)
    llevamos = sum(c for _, c in FUENTES[:k + 1])
    if leyenda:
        texto(s, 0.55, 6.3, 12.25, 0.3, "Variables por fuente (avisos 21, texto 84, OpenStreetMap 39, satélite 7 y "
              f"Catastro 11): llevamos {llevamos} de {total}", size=12, color="muted")
    x = 0.55
    for i, (nombre, c) in enumerate(FUENTES):
        w = 12.25 * c / total
        relleno = "azul" if i == k else ("gris" if i < k else "gris_claro")
        forma(s, MSO_SHAPE.RECTANGLE, x, 6.62, w - 0.03, 0.34, relleno=relleno)
        etq = f"{nombre} {c}" if w > 2.0 else str(c)
        texto(s, x, 6.62, w - 0.03, 0.34, etq, size=12, color="blanco" if i <= k else "tinta", align="center",
              anchor="middle")
        x += w


def _tabla_fuente(s, filas, total, y=1.5, alto=0.5):
    tabla(s, 0.55, y, 5.6, [("Grupo", "Variables", "Cuántas")] + filas + [("Total", "", str(total))], [1.55, 3.3, 0.75],
          size=12, alto_fila=alto)


def _pares(s, x0, y0, metricas, R, ancho=3.0, alto=2.05, cols=2):
    """Pequeñas gráficas de dos barras: prueba (naranja) frente a entrenamiento fuera de Chapinero (gris)."""
    for i, (titulo_m, clave, d) in enumerate(metricas):
        x, y = x0 + (i % cols) * (ancho + 0.25), y0 + (i // cols) * (alto + 0.1)
        texto(s, x, y, ancho, 0.4, titulo_m, size=12, color="tinta")
        a, b = R[clave]["prueba"], R[clave]["entrenamiento_resto"]
        ch = barras(s, x - 0.05, y + 0.4, ancho, alto - 0.45, ["Prueba", "Entrenamiento"],
                    [("v", [a, b], ["naranja", "gris_claro"])], minimo=0, maximo=(max(a, b) * 1.35) or 1, gap=35,
                    grid=False)
        etiquetas_puntos(ch, [[n(a, d), n(b, d)]])


def _barras_grupos(s, R, items, titulo, maximo, d=0):
    texto(s, 6.55, 1.45, 6.25, 0.35, titulo, size=12, color="muted")
    ch = barras(s, 6.45, 1.8, 6.35, 4.3, [nom for nom, _ in items],
                [(GRUPOS_F[0][0], [R[c]["prueba"] for _, c in items], GRUPOS_F[0][1]),
                 (GRUPOS_F[1][0], [R[c]["entrenamiento_resto"] for _, c in items], GRUPOS_F[1][1])],
                minimo=0, maximo=maximo, leyenda=True, gap=45, overlap=-5, grid=False)
    etiquetas_puntos(ch, [[n(R[c]["prueba"], d) for _, c in items], [n(R[c]["entrenamiento_resto"], d) for _, c in items]])


def fuentes_datos(deck, etiqueta, seccion):
    """Cinco láminas, una por fuente. Devuelve [(lámina, notas), ...] en el orden de FUENTES."""
    R = json.load(open(D / "ppt_fuentes.json", encoding="utf-8"))
    nprueba, nresto = R["n"]["prueba"], R["n"]["entrenamiento_resto"]
    pie = (f"Medianas en los {n(nprueba, 0)} avisos de prueba (Chapinero) y en los {n(nresto, 0)} de entrenamiento fuera "
           "de Chapinero; naranja, prueba; gris, entrenamiento.")
    salida = []

    # 1. Avisos de Properati: cada variable construida, de qué columna bruta sale, cómo, cuántos no nulos y su promedio
    est, est_g, brut_g = R["avisos_variables"], R["avisos_variables_grupo"], R["avisos_brutas_no_nulos_grupo"]
    ntot = n(R["n_avisos_total"], 0)

    def _par(e_, p_):
        e_, p_ = n(e_, 0), n(p_, 0)
        return f"{e_} %" if e_ == p_ else f"{e_} % y {p_} %"

    def nb(*cols):
        pares_ = [_par(brut_g[c]["entrenamiento"], brut_g[c]["prueba"]) for c in cols]
        if len(set(pares_)) == 1:
            return f"{' y '.join(cols)} ({pares_[0]})"
        return " y ".join(f"{c} ({t_})" for c, t_ in zip(cols, pares_))

    def fila_var(clave, nombre, como, d):
        e = est[clave]
        prom = f"{e['promedio']:.1f}".replace(".", ",") if clave == "tiempo" else n(e["promedio"], d)
        eg = est_g[clave]
        return nombre, como, _pct(eg["entrenamiento"]), _pct(eg["prueba"]), prom

    partes = [
        ("Fuente 1a. Avisos de Properati: la variable que se predice y las de la vivienda", [
            (nb("price"), [
                ("Precio (millones)", "price dividido por un millón; es la variable que se predice", "100 %", "0 %",
                 n(R["precio_promedio_entrenamiento"], 1))]),
            (nb('property_type'), [fila_var("es_apartamento", "Es apartamento",
                                                                 "1 si es apartamento y 0 si es casa", 2)]),
            (nb('bedrooms'), [fila_var("bedrooms", "Alcobas", "tal como viene", 1)]),
            (nb('rooms'), [fila_var("rooms", "Ambientes", "tal como viene, con faltantes", 1)]),
            (nb('surface_covered', 'surface_total'), [
                fila_var("area_imp", "Área (m²)", "la cubierta; si falta, la total; solo entre 15 y 1.000 m²; si no hay, "
                                                   "la escrita en el texto", 1),
                fila_var("log_area", "Logaritmo del área", "logaritmo natural del área", 2),
                fila_var("area_desde_texto", "Área del texto", "1 si el área salió del texto", 2),
                fila_var("area_falta", "Área faltante", "1 si no hay área en columnas ni en texto", 2),
                fila_var("area_inconsistente", "Área inconsistente", "1 si la cubierta supera la total", 2)]),
            (nb('bathrooms'), [
                fila_var("banos_imp", "Baños", "la columna; si falta, la cifra del texto", 1),
                fila_var("banos_desde_texto", "Baños del texto", "1 si los baños salieron del texto", 2)])]),
        ("Fuente 1b. Avisos de Properati: texto, fecha, ubicación y entorno de avisos", [
            (nb('title', 'description'), [
                fila_var("parqueaderos_imp", "Parqueaderos", "la cifra escrita; mencionados sin cifra, 1; sin mención, 0", 1),
                fila_var("piso_texto", "Piso", "el piso escrito en el texto", 1),
                fila_var("estrato_texto", "Estrato", "el estrato escrito en el texto", 1),
                fila_var("antiguedad_texto", "Antigüedad", "los años de construido escritos en el texto", 1)]),
            (nb('month', 'year'), [fila_var("tiempo", "Fecha del aviso", "año más (mes − 1)/12", 1)]),
            (nb('lat', 'lon'), [
                fila_var("x_km", "Coordenada este (km)", "proyectada a EPSG:9377", 1),
                fila_var("y_km", "Coordenada norte (km)", "proyectada a EPSG:9377", 1),
                fila_var("coord_redondeada", "Coordenada redondeada", "1 si la latitud tiene 3 decimales o menos", 2)]),
            ("Todos los avisos, sin precio", [
                fila_var("n_avisos_500m", "Avisos a 500 m", "número de avisos alrededor", 1),
                fila_var("prop_apartamentos_500m", "Apartamentos a 500 m", "proporción de apartamentos alrededor", 2),
                fila_var("area_mediana_500m", "Área mediana a 500 m", "área mediana de los avisos alrededor", 1)])])]
    notas_1 = [
        "La primera fuente son los avisos de Properati. La primera fila es el precio: es la variable que se predice; está en "
        "las 38.644 filas de entrenamiento y viene vacía en las 10.286 de prueba. Después, la vivienda: cada columna bruta da "
        "una o varias variables, y al "
        "lado está cuántos avisos las tienen y su promedio. Alcobas y tipo vienen completas. El área es el gran problema: "
        "las columnas de superficie solo vienen en uno de cada cinco avisos; con la regla de la cubierta, la total y luego "
        "el texto, la tenemos en 54 %, y tres banderas dicen de dónde salió. Los baños pasan de 74 % a 85 % con el texto.",
        "Del título y la descripción salen parqueaderos, piso, estrato y antigüedad, aunque piso, estrato y antigüedad "
        "aparecen en pocos avisos. La fecha se vuelve un número continuo, las coordenadas se proyectan a kilómetros y "
        "marcamos las redondeadas, que son 43 %. Y con todos los avisos, sin su precio, describimos el entorno a 500 "
        "metros. En total, 21 variables de esta fuente. El precio también viene de esta fuente: es la variable que se "
        "predice, y las 21 son variables que lo explican."]
    acum = 0
    for (titulo, grupos), nota_, en_tabla in zip(partes, notas_1, (10, 11)):
        acum += en_tabla
        s = deck.nueva(titulo, etiqueta=etiqueta, seccion=seccion)
        filas = [("Columna bruta (no nulos: entrenamiento y prueba)", "Variable", "Cómo se construye",
                  "No nulos en entrenamiento", "No nulos en prueba", "Promedio")]
        tramos = []
        for col, vars_ in grupos:
            tramos.append((len(filas), len(filas) + len(vars_) - 1))
            for k, fv in enumerate(vars_):
                filas.append((col if k == 0 else "",) + fv)
        texto(s, 0.55, 1.28, 12.25, 0.38, f"Base de Properati: {n(R['n_filas']['entrenamiento'], 0)} filas de "
              f"entrenamiento y {n(R['n_filas']['prueba'], 0)} de prueba, con las mismas 16 columnas; en la prueba el precio "
              "viene vacío.", size=13, color="navy")
        extra = " (el precio no se cuenta: es la variable que se predice)" if acum == 10 else ""
        filas.append(_fila_total(en_tabla, acum, extra))
        gf = tabla(s, 0.55, 1.7, 12.25, filas, [2.75, 2.05, 4.15, 1.2, 1.0, 1.1], size=12, alto_fila=0.29)
        t = gf.table
        _formato_total(t, len(filas) - 1)
        for g_, (a_, b_) in enumerate(tramos):
            fondo = rgb("FFFFFF" if g_ % 2 else "F7F9FB")
            for r_ in range(a_, b_ + 1):
                for c_ in range(6):
                    t.cell(r_, c_).fill.solid()
                    t.cell(r_, c_).fill.fore_color.rgb = fondo
            if b_ > a_:
                t.cell(a_, 0).merge(t.cell(b_, 0))
        texto(s, 0.55, 6.02, 12.25, 0.42, f"No nulos: porcentaje de filas con dato en entrenamiento "
              f"({n(R['n_filas']['entrenamiento'], 0)}) y en prueba ({n(R['n_filas']['prueba'], 0)}). Promedio en los {ntot} "
              "avisos; el del precio, solo en entrenamiento. En las variables de 0 y 1, el promedio es la proporción.",
              size=12, color="muted")
        _contador(s, 0, leyenda=False)
        salida.append((s, nota_))

    # 2 a 5. Las demás fuentes, con la misma tabla de la fuente 1: dato bruto, variable, cómo se construye, no nulos en
    # entrenamiento y en prueba, y promedio (laminas/ppt_variables.json, de 25_datos_laminas_variables_por_fuente.py)
    R7 = json.load(open(D / "ppt_variables.json", encoding="utf-8"))
    V = R7["variables"]
    nent, npru = n(R7["n_filas"]["entrenamiento"], 0), n(R7["n_filas"]["prueba"], 0)
    osm_o = {**R7["osm_objetos"], **R7.get("osm_extra_objetos", {})}

    def fv(c, nombre, como, d=1, escala="original", factor=1.0):
        """Fila de una variable. escala: original (deshace log(1 + x)), modelo (tal cual) o geometrica (exp del promedio)."""
        e = V[c]
        if escala == "geometrica":
            prom = math.exp(e["promedio"]) * factor
        elif escala == "original" and e["promedio_original"] is not None:
            prom = e["promedio_original"] * factor
        else:
            prom = e["promedio"] * factor
        return nombre, como, _pct(e["entrenamiento"]), _pct(e["prueba"]), n(prom, d)

    def obj(cat_):
        return f"{n(osm_o[cat_], 0)} objetos"

    pie_osm = (f"No nulos: % de filas con dato en entrenamiento ({nent}) y en prueba ({npru}). El modelo usa log(1 + x); "
               "el promedio va en la escala original, en metros, m² o hectáreas.")
    base_osm = (f"Base: extracto de OpenStreetMap de Bogotá (BBBike): {n(sum(R7['osm_objetos'].values()), 0)} objetos en 12 "
                f"categorías y {n(sum(R7.get('osm_extra_objetos', {}).values()), 0)} en 6 más, y las áreas protegidas.")

    # 2. Texto del aviso
    base_txt = (f"Base: título y descripción de los 48.930 avisos ({nent} de entrenamiento y {npru} de prueba), en "
                "minúsculas y sin tildes.")
    pie_ind = (f"Un indicador vale 1 si el texto trae alguna de las palabras y no está negada (sin, no tiene, no cuenta con, "
               f"no hay, no incluye). No nulos: % de filas con dato en entrenamiento ({nent}) y en prueba ({npru}). En "
               "las variables de 0 y 1, el promedio es la proporción.")
    clase = "title y description: 8 indicadores de la clase"
    ext = "title y description: extensiones"
    paginas = [
        ("Fuente 2a. Texto del aviso: indicadores que leen la negación", base_txt, [
            (clase, [fv("txt_parqueadero", "Parqueadero", "parqueadero", 2),
                     fv("txt_piscina", "Piscina", "piscina o jacuzzi", 2),
                     fv("txt_ascensor", "Ascensor", "ascensor", 2),
                     fv("txt_terraza_balcon", "Terraza o balcón", "terraza o balcón", 2),
                     fv("txt_vigilancia", "Vigilancia", "vigilancia, seguridad, portería o portero", 2),
                     fv("txt_remodelado", "Remodelado", "remodelado, renovado o remodelación", 2),
                     fv("txt_cocina_integral", "Cocina integral", "cocina integral", 2),
                     fv("txt_zona_social", "Zona social", "zona social o salón social, comunal o de reuniones", 2)]),
            (ext, [fv("txt_chimenea", "Chimenea", "chimenea", 2),
                   fv("txt_vista", "Vista", "vista o panorámica", 2),
                   fv("txt_estudio", "Estudio", "estudio", 2),
                   fv("txt_apartaestudio", "Apartaestudio", "apartaestudio, aparta estudio o loft", 2)])],
         pie_ind,
         "La segunda fuente es el texto del aviso: 84 variables. Empezamos por los indicadores. Cada uno vale 1 si el texto "
         "menciona el atributo y la mención no está negada: no tiene ascensor no cuenta como ascensor. Ocho son los "
         "indicadores de la clase y el resto son extensiones nuestras. Al lado, qué tan común es cada uno: el parqueadero "
         "aparece en siete de cada diez avisos; la piscina, en menos de uno de cada diez."),
        ("Fuente 2b. Texto del aviso: más indicadores y las cifras escritas", base_txt, [
            (ext, [fv("txt_duplex_penthouse", "Dúplex o penthouse", "dúplex, penthouse o pent house", 2),
                   fv("txt_exterior", "Exterior", "exterior", 2),
                   fv("txt_gimnasio", "Gimnasio", "gimnasio o gym", 2),
                   fv("txt_deposito", "Depósito", "depósito o bodega", 2),
                   fv("txt_conjunto", "Conjunto", "conjunto o conjunto cerrado", 2),
                   fv("txt_nuevo_estrenar", "Nuevo", "para estrenar, sobre planos o proyecto nuevo", 2),
                   fv("txt_para_remodelar", "Para remodelar", "para remodelar o reformar, requiere o necesita "
                                                              "remodelación", 2)]),
            ("title y description: cifras escritas", [
                fv("alcobas_texto", "Alcobas del texto", "número antes de alcoba, entre 1 y 10", 1),
                fv("banos_texto", "Baños del texto", "número antes de baño, entre 1 y 10", 1),
                fv("parqueaderos_texto", "Parqueaderos del texto", "número antes de parqueadero, entre 1 y 6", 1),
                fv("area_texto", "Área del texto (m²)", "cifra tras área o antes de m2, entre 20 y 1.000", 1)])],
         pie_ind + " Piso, estrato y antigüedad también salen del texto, pero se cuentan en la fuente 1.",
         "Siguen los demás indicadores y las cuatro cifras que leemos en el texto: alcobas, baños, parqueaderos y área. Solo "
         "se acepta un número en un rango creíble, por eso tienen faltantes: el área escrita aparece en uno de cada tres "
         "avisos. Estas cifras completan las columnas de la fuente 1 y además entran solas al modelo."),
    ]
    top = pd.read_csv(DIR_TAB / "texto_lda_topicos.csv")
    raices = [", ".join(str(r).split(", ")[:3]) for r in top.iloc[:, 1]]
    lsa = [V[f"lsa_{k:02d}"]["promedio"] for k in range(50)]
    paginas.append(
        ("Fuente 2c. Texto del aviso: tópicos, componentes y puntaje de precio",
         "Base: matriz de 48.930 avisos por 44.888 términos (raíces de palabras y pares de palabras que aparecen en 5 "
         "avisos o más).", [
            ("Conteos de la matriz: LDA con 10 tópicos", [
                fv(f"topico_{k:02d}", f"Tópico {k}", f"raíces más probables: {raices[k]}", 2) for k in range(10)]),
            ("Matriz TF-IDF", [
                ("50 componentes LSA", "SVD truncada de la matriz TF-IDF (el PCA del texto)",
                 _pct(min(V[f"lsa_{k:02d}"]["entrenamiento"] for k in range(50))),
                 _pct(min(V[f"lsa_{k:02d}"]["prueba"] for k in range(50))), f"{n(min(lsa), 2)} a {n(max(lsa), 2)}"),
                fv("puntaje_ridge", "Puntaje de precio", "ridge de ln(precio) sobre TF-IDF, fuera de pliegue", 2,
                   escala="modelo")])],
         f"No nulos: % de filas con dato en entrenamiento ({nent}) y en prueba ({npru}). Los pesos de los 10 tópicos de un "
         "aviso suman 1. El puntaje está en logaritmo de millones (6,41 equivale a unos 610 millones).",
         "Por último, el texto como matriz. Con los conteos, LDA encuentra diez tópicos y cada aviso recibe su peso en cada "
         "uno; las raíces dicen de qué habla el tópico, por ejemplo el tópico 0 junta Chapinero y Rosales. Con TF-IDF, la "
         "SVD resume la matriz en 50 componentes, y una regresión ridge da un puntaje de precio. Ese puntaje se calcula "
         "fuera de pliegue, para que el modelo no vea el precio del mismo aviso: es la variable de texto más fuerte."))
    for k_, (titulo, base, grupos, pie_, nota_) in enumerate(paginas):
        en_tabla = (12, 11, 61)[k_]
        acum += en_tabla
        s = _lamina_variables(deck, etiqueta, seccion, titulo, base, grupos, pie_, 1, en_tabla, acum)
        salida.append((s, nota_))

    # 3. OpenStreetMap
    paginas = [
        ("Fuente 3a. OpenStreetMap: comercio y transporte alrededor del aviso", [
            (f"amenity=restaurant ({obj('restaurante')})", [
                fv("ln_n_restaurante_300m", "Restaurantes a 300 m", "número dentro de un círculo de 300 m", 1),
                fv("ln_n_restaurantes_500m", "Restaurantes a 500 m", "número dentro de un círculo de 500 m", 1),
                fv("ln_n_restaurante_1km", "Restaurantes a 1 km", "número dentro de un círculo de 1 km", 1),
                fv("ln_n_restaurantes_2km", "Restaurantes a 2 km", "número dentro de un círculo de 2 km", 1)]),
            (f"amenity=bar, pub, cafe o nightclub ({obj('bar_cafe')})", [
                fv("ln_n_bar_cafe_300m", "Bares y cafés a 300 m", "número dentro de un círculo de 300 m", 1),
                fv("ln_n_bares_cafes_500m", "Bares y cafés a 500 m", "número dentro de un círculo de 500 m", 1),
                fv("ln_n_bar_cafe_1km", "Bares y cafés a 1 km", "número dentro de un círculo de 1 km", 1)]),
            (f"shop=supermarket ({obj('supermercado')})", [
                fv("ln_n_supermercado_300m", "Supermercados a 300 m", "número dentro de un círculo de 300 m", 1),
                fv("ln_n_supermercados_1km", "Supermercados a 1 km", "número dentro de un círculo de 1 km", 1),
                fv("ln_n_supermercado_1km", "Supermercados a 1 km (repetida)", "la misma cuenta de la fila anterior", 1)]),
            (f"highway=bus_stop ({obj('paradero')})", [
                fv("ln_n_paradero_300m", "Paraderos a 300 m", "número dentro de un círculo de 300 m", 1),
                fv("ln_n_paraderos_500m", "Paraderos a 500 m", "número dentro de un círculo de 500 m", 1),
                fv("ln_n_paradero_1km", "Paraderos a 1 km", "número dentro de un círculo de 1 km", 1)])],
         "La tercera fuente es OpenStreetMap, el mapa abierto: 39 variables. Del extracto de Bogotá tomamos los objetos por "
         "su etiqueta, por ejemplo amenity igual a restaurant, y alrededor de cada aviso contamos cuántos hay en círculos "
         "de 300 metros a 2 kilómetros. Una aclaración: supermercados a 1 km quedó dos veces, con la misma cuenta; para el "
         "boosting la copia no cambia nada, pero lo decimos."),
        ("Fuente 3b. OpenStreetMap: colegios, TransMilenio, oficinas y embajadas", [
            (f"amenity=school ({obj('colegio')})", [
                fv("ln_n_colegio_300m", "Colegios a 300 m", "número dentro de un círculo de 300 m", 1),
                fv("ln_n_colegios_1km", "Colegios a 1 km", "número dentro de un círculo de 1 km", 1),
                fv("ln_n_colegio_1km", "Colegios a 1 km (repetida)", "la misma cuenta de la fila anterior", 1)]),
            (f"public_transport=station o amenity=bus_station ({obj('estacion')})", [
                fv("ln_dist_transmilenio_m", "Distancia a TransMilenio (m)", "estación más cercana, sin metro, tren ni SITP", 0),
                fv("ln_n_transmilenio_1km", "TransMilenio a 1 km", "estaciones dentro de un círculo de 1 km", 1)]),
            (f"office ({n(osm_o['oficina'], 0)}) y building=office o commercial ({obj('edificio_oficinas')})", [
                fv("ln_n_oficina_500m", "Oficinas a 500 m", "número dentro de un círculo de 500 m", 1),
                fv("ln_n_oficina_1000m", "Oficinas a 1 km", "número dentro de un círculo de 1 km", 1),
                fv("ln_n_edificio_oficinas_500m", "Edificios de oficinas a 500 m", "número dentro de un círculo de 500 m",
                   1)]),
            (f"tourism=hotel, amenity=bank y leisure=fitness_centre ({n(osm_o['hotel'], 0)}, {n(osm_o['banco'], 0)} y "
             f"{obj('gimnasio')})", [
                fv("ln_n_hotel_1000m", "Hoteles a 1 km", "número dentro de un círculo de 1 km", 1),
                fv("ln_n_banco_500m", "Bancos a 500 m", "número dentro de un círculo de 500 m", 1),
                fv("ln_n_gimnasio_1000m", "Gimnasios a 1 km", "número dentro de un círculo de 1 km", 1)]),
            (f"office=diplomatic o amenity=embassy ({obj('embajada')})", [
                fv("ln_n_embajada_1000m", "Embajadas a 1 km", "número dentro de un círculo de 1 km", 1),
                fv("ln_dist_embajada_m", "Distancia a una embajada (m)", "a la embajada más cercana", 0)])],
         "Siguen colegios, TransMilenio, oficinas, hoteles, bancos, gimnasios y embajadas. Estas últimas las agregamos "
         "pensando en Chapinero, donde se concentran: a un kilómetro de un aviso de prueba hay muchas más oficinas, hoteles "
         "y embajadas que en el resto de Bogotá. Colegios a 1 km también quedó repetida."),
        ("Fuente 3c. OpenStreetMap: distancias a parques, vías, servicios y referencias", [
            ("leisure=park, solo polígonos", [
                fv("ln_dist_parque_m", "Distancia al parque (m)", "al parque más cercano", 0),
                fv("ln_area_parque_m2", "Área de ese parque (m²)", "área del parque más cercano", 0),
                fv("ln_dist_parque_grande_m", "Distancia a un parque grande (m)", "al parque de 1 ha o más más cercano", 0),
                fv("ln_area_parques_500m_ha", "Parques a 500 m (ha)", "área de los parques con centro a 500 m o menos",
                   1)]),
            ("highway=motorway, trunk o primary", [
                fv("ln_dist_via_principal_m", "Distancia a vía principal (m)",
                   f"al tramo más cercano ({n(osm_o['via_principal'], 0)} tramos)", 0)]),
            ("amenity=university, shop=mall, amenity=hospital o clinic y amenity=police", [
                fv("ln_dist_universidad_m", "Distancia a universidad (m)", f"a la más cercana ({obj('universidad')})", 0),
                fv("ln_dist_centro_comercial_m", "Distancia a centro comercial (m)",
                   f"al más cercano ({obj('centro_comercial')})", 0),
                fv("ln_dist_hospital_m", "Distancia a hospital (m)", f"al más cercano ({obj('hospital')})", 0),
                fv("ln_dist_policia_m", "Distancia a policía (m)", f"a la más cercana ({obj('policia')})", 0)]),
            ("boundary=protected_area o leisure=nature_reserve", [
                fv("ln_dist_area_protegida_m", "Distancia a los cerros (m)", "al área protegida de 100 ha o más más "
                                                                            "cercana", 0)]),
            ("Distancia en línea recta a tres puntos fijos", [
                fv("ln_dist_p93_m", "Al Parque 93 (m)", "desde la coordenada fija del lugar", 0),
                fv("ln_dist_zonat_m", "A la Zona T (m)", "desde la coordenada fija del lugar", 0),
                fv("ln_dist_centro_internacional_m", "Al Centro Internacional (m)", "desde la coordenada fija del lugar",
                   0)])],
         "Por último, distancias: al parque más cercano y a uno grande, a la vía principal, a universidades, centros "
         "comerciales, hospitales y policía, a los cerros, y a tres lugares de referencia: el Parque 93, la Zona T y el "
         "Centro Internacional. Las distancias van en metros y el modelo usa su logaritmo, porque pasar de 100 a 200 metros "
         "pesa más que pasar de 5 a 5,1 kilómetros."),
    ]
    for titulo, grupos, nota_ in paginas:
        en_tabla = sum(len(v) for _, v in grupos)
        acum += en_tabla
        s = _lamina_variables(deck, etiqueta, seccion, titulo, base_osm, grupos, pie_osm, 2, en_tabla, acum)
        salida.append((s, nota_))

    # 4. Imágenes de satélite
    grupos = [
        ("SRTM (NASA): elevación con píxel de 30 m", [
            fv("elevacion_m", "Elevación (m)", "valor del píxel en el aviso", 0),
            fv("pendiente_grados", "Pendiente (grados)", "pendiente del terreno calculada sobre el SRTM", 1),
            fv("elevacion_relativa_1km", "Elevación relativa (m)", "elevación menos la mediana de los avisos a 1 km", 1)]),
        ("Sentinel-2: 10 m, 2020 y 2021, sin nubes", [
            fv("ndvi_200m", "Vegetación (NDVI a 200 m)", "(B8 − B4)/(B8 + B4), promedio en 200 m", 2)]),
        ("VIIRS (NOAA): luces de noche, 2020", [
            fv("luces_nocturnas", "Luces nocturnas", "radiancia promedio de los meses de 2020", 1)]),
        ("GHSL (Comisión Europea): 100 m", [
            fv("altura_construida_m", "Altura construida (m)", "altura media de lo construido en 2018", 1),
            fv("ln_poblacion_100m", "Población", "personas en la celda de 100 m, en 2020", 0)])]
    s = _lamina_variables(deck, etiqueta, seccion, "Fuente 4. Imágenes de satélite: 7 variables del terreno y de lo "
                          "construido", "Base: cinco capas de Google Earth Engine, leídas en la coordenada de cada aviso "
                          "(escala de 10 m).", grupos,
                          f"No nulos: % de filas con dato en entrenamiento ({nent}) y en prueba ({npru}). La población "
                          "entra al modelo como log(1 + x); su promedio va en personas.", 3, 7, acum + 7)
    acum += 7
    salida.append((s, "La cuarta fuente son las imágenes de satélite, que bajamos con Google Earth Engine: siete variables. "
                      "Del modelo de elevación salen la elevación, la pendiente y la elevación relativa, que dice si el aviso "
                      "está más alto que sus vecinos, como pasa al pie de los cerros. Sentinel-2 da la vegetación, VIIRS las "
                      "luces de noche y GHSL la altura de lo construido y la población. Todas tienen dato en todos los "
                      "avisos."))

    # 5. Catastro
    grupos = [
        ("Valor de referencia del suelo (V_REF), por manzana y año", [
            fv("ln_vref_manzana", "Suelo en la manzana", "mediana del valor del terreno por m², año del aviso",
               2, escala="geometrica", factor=1e-6),
            fv("ln_vref_300m", "Suelo a 300 m", "mediana de las manzanas a 300 m", 2,
               escala="geometrica", factor=1e-6)]),
        ("Avalúo catastral en propiedad horizontal (AV_CAT_PH)", [
            fv("ln_avaluo_ph_manzana", "Avalúo en la manzana", "avalúo por m² de los apartamentos de la manzana", 2,
               escala="geometrica", factor=1e-6),
            fv("ln_avaluo_ph_300m", "Avalúo a 300 m", "mediana entre las manzanas a 300 m", 2, escala="geometrica",
               factor=1e-6)]),
        ("Altura de las manzanas (pisos ponderados)", [
            fv("pisos_manzana", "Pisos en la manzana", "pisos ponderados de la manzana", 1, escala="modelo"),
            fv("pisos_300m", "Pisos a 300 m", "mediana entre las manzanas a 300 m", 1, escala="modelo")]),
        ("Estratificación por manzana", [
            fv("estrato_manzana", "Estrato de la manzana", "0 si la manzana no tiene estrato residencial", 1,
               escala="modelo"),
            fv("estrato_medio_300m", "Estrato medio a 300 m", "promedio entre las manzanas a 300 m", 1, escala="modelo"),
            fv("prop_estrato6_300m", "Estrato 6 a 300 m", "proporción de manzanas de estrato 6 a 300 m", 2,
               escala="modelo")]),
        ("Geometría de las manzanas, 2021", [
            fv("n_manzanas_300m", "Manzanas a 300 m", "número de manzanas con centro a 300 m o menos", 0, escala="modelo"),
            fv("dist_manzana_m", "Distancia a la manzana (m)", "0 si el aviso cae dentro; si cae en la vía, hasta 100 m", 1,
               escala="modelo")])]
    s = _lamina_variables(deck, etiqueta, seccion, "Fuente 5. Catastro: 11 variables del valor del suelo y de la manzana",
                          "Base: capas de Catastro Bogotá (UAECD) por manzana, de 2019 a 2021; cada aviso toma su manzana "
                          "y el año del aviso.", grupos,
                          f"No nulos: % de filas con dato en entrenamiento ({nent}) y en prueba ({npru}). Suelo y avalúo van "
                          "en millones por m²: el modelo usa su logaritmo y el promedio es el geométrico.", 4, 11, acum + 11)
    acum += 11
    assert acum == 162, acum
    salida.append((s, "La quinta fuente es Catastro, con capas por manzana de 2019 a 2021. Cada aviso toma la manzana que lo "
                      "contiene, o la más cercana si cae en la vía, y el valor del año del aviso. De ahí salen el valor de "
                      "referencia del suelo, el avalúo de los apartamentos, los pisos y el estrato, en la manzana y en un "
                      "radio de 300 metros. El avalúo de la manzana es el que más falta, porque no todas las manzanas tienen "
                      "propiedad horizontal. Con esto completamos las 162 variables."))
    return salida


def _pct(v):
    """Porcentaje sin decimales, salvo que el redondeo oculte que no es 0 ni 100."""
    t = n(v, 0)
    if t == "100" and v < 100:
        t = n(math.floor(v * 10) / 10, 1)
    elif t == "0" and v > 0:
        t = n(math.ceil(v * 10) / 10, 1)
    return f"{t} %"


def _fila_total(en_tabla, acum, extra=""):
    return ("Total", f"{en_tabla} variables en esta tabla; acumulado: {acum} de 162{extra}", "", "", "", "")


def _formato_total(t, r):
    """Fila de total: texto en azul oscuro sobre el color del encabezado, con la descripción en una celda unida."""
    for c_ in range(6):
        t.cell(r, c_).fill.solid()
        t.cell(r, c_).fill.fore_color.rgb = rgb("E8EEF5")
        for par in t.cell(r, c_).text_frame.paragraphs:
            for run in par.runs:
                run.font.color.rgb = rgb("navy")
    t.cell(r, 1).merge(t.cell(r, 5))


def _lamina_variables(deck, etiqueta, seccion, titulo, base, grupos, pie, k, en_tabla, acum):
    """Lámina de una fuente: una fila por variable, agrupadas por su dato bruto (celda unida), como la fuente 1."""
    s = deck.nueva(titulo, etiqueta=etiqueta, seccion=seccion)
    filas = [("Dato bruto", "Variable", "Cómo se construye", "No nulos en entrenamiento", "No nulos en prueba",
              "Promedio")]
    tramos = []
    for col, vars_ in grupos:
        tramos.append((len(filas), len(filas) + len(vars_) - 1))
        for j, fila in enumerate(vars_):
            filas.append((col if j == 0 else "",) + tuple(fila))
    filas.append(_fila_total(en_tabla, acum))
    texto(s, 0.55, 1.28, 12.25, 0.38, base, size=13, color="navy")
    gf = tabla(s, 0.55, 1.7, 12.25, filas, [2.75, 2.45, 3.85, 1.2, 1.0, 1.0], size=12, alto_fila=0.26)
    t = gf.table
    _formato_total(t, len(filas) - 1)
    for g_, (a_, b_) in enumerate(tramos):
        fondo = rgb("FFFFFF" if g_ % 2 else "F7F9FB")
        for r_ in range(a_, b_ + 1):
            for c_ in range(6):
                t.cell(r_, c_).fill.solid()
                t.cell(r_, c_).fill.fore_color.rgb = fondo
        if b_ > a_:
            t.cell(a_, 0).merge(t.cell(b_, 0))
    texto(s, 0.55, 6.02, 12.25, 0.42, pie, size=12, color="muted")
    _contador(s, k, leyenda=False)
    return s


# ------------------------------------------------------------------------------------------------------------------
# Entrenamiento y prueba: mapa con dos grupos e histograma del precio de entrenamiento
# ------------------------------------------------------------------------------------------------------------------
def _saltar_etiquetas(ch, cada):
    """Muestra una de cada `cada` etiquetas del eje de categorías (c:tickLblSkip y c:tickMarkSkip)."""
    from lxml import etree
    ns = "http://schemas.openxmlformats.org/drawingml/2006/chart"
    cat = ch._chartSpace.find(f".//{{{ns}}}catAx")
    for nombre in ("tickLblSkip", "tickMarkSkip"):
        el = etree.SubElement(cat, f"{{{ns}}}{nombre}")
        el.set("val", str(cada))
    sig = cat.find(f"{{{ns}}}noMultiLvlLbl")
    if sig is not None:
        for nombre in ("tickLblSkip", "tickMarkSkip"):
            el = cat.find(f"{{{ns}}}{nombre}")
            sig.addprevious(el)


def _area_manual(ch, x, y, w, h):
    """Fija el área interna de trazado del gráfico en fracciones del marco (c:manualLayout, destino inner)."""
    from lxml import etree
    ns = "http://schemas.openxmlformats.org/drawingml/2006/chart"
    pa = ch._chartSpace.find(f".//{{{ns}}}plotArea")
    lay = pa.find(f"{{{ns}}}layout")
    if lay is None:
        lay = etree.Element(f"{{{ns}}}layout")
        pa.insert(0, lay)
    for hijo in list(lay):
        lay.remove(hijo)
    ml = etree.SubElement(lay, f"{{{ns}}}manualLayout")
    for nombre, val in (("layoutTarget", "inner"), ("xMode", "edge"), ("yMode", "edge"), ("x", x), ("y", y), ("w", w),
                        ("h", h)):
        etree.SubElement(ml, f"{{{ns}}}{nombre}").set("val", str(val))


def entrenamiento_y_prueba(deck, etiqueta, seccion):
    """Lámina de datos: dónde están los avisos de entrenamiento y de prueba, y cómo se reparte el precio a predecir."""
    import numpy as np

    desc = json.loads((D / "ppt_descripcion.json").read_text(encoding="utf-8"))
    pron = json.loads((D / "ppt_varios2.json").read_text(encoding="utf-8"))["pronostico"]
    ent = pd.read_parquet(D / "entrenamiento_mapa.parquet")
    precio = ent["price"].to_numpy() / 1e6
    bordes = np.arange(300, 1651, 50)
    cuenta, _ = np.histogram(precio, bins=bordes)
    med, prom = float(np.median(precio)), float(precio.mean())

    def n(v, d=0):
        return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")

    s = deck.nueva("Entrenamos con Bogotá, casi sin Chapinero, para predecir Chapinero", etiqueta=etiqueta,
                   seccion=seccion)
    imagen(s, IMG / "mapa_entrenamiento_prueba.png", 0.55, 1.5, w=5.0,
           alt="Mapa de Bogotá con los avisos de entrenamiento en gris y los de prueba en naranja, casi todos en Chapinero")
    stats = [(n(desc["n_train"]), "avisos de entrenamiento (2019 a 2021), con precio; solo 307 en Chapinero"),
             (n(desc["n_test"]), f"avisos de prueba, sin precio; el {n(pron['pct_localidad_chapinero'])} % en la "
                                 "localidad de Chapinero"),
             ("300 a 1.650", f"millones: rango del precio en el entrenamiento"),
             (f"{n(desc['pct_apto_test'])} %", f"de la prueba son apartamentos, frente a {n(desc['pct_apto_train'])} % "
                                               "del entrenamiento")]
    for i, (v, e) in enumerate(stats):
        x0, y0 = 5.9 + (i % 2) * 3.5, 1.4 + (i // 2) * 1.3
        texto(s, x0, y0, 3.4, 0.6, v, size=30, color="azul")
        texto(s, x0, y0 + 0.58, 3.3, 0.7, e, size=13, color="muted")
    texto(s, 5.9, 4.05, 7.0, 0.4, f"Precio de los {n(len(precio))} avisos de entrenamiento, con su mediana y su promedio",
          size=13, color="navy")
    cats = [n(b) for b in bordes[:-1]]
    gx, gy, gw, gh = 5.8, 4.4, 7.05, 2.55
    ch = barras(s, gx, gy, gw, gh, cats, [("Entrenamiento", [int(c) for c in cuenta], "gris")], horizontal=False,
                fmt="#,##0", etiquetas=False, gap=8, valor_visible=True, titulo_valor="Avisos", maximo=6000, unidad=1000,
                titulo_cat="Precio del aviso (millones); cada barra va de su cifra a 50 millones más")
    _saltar_etiquetas(ch, 4)
    # Área de trazado fija (fracciones del gráfico) para ubicar con exactitud las líneas verticales
    lx, ly, lw, lh = 0.12, 0.05, 0.85, 0.66
    _area_manual(ch, lx, ly, lw, lh)
    x0, y0, ancho, alto = gx + lx * gw, gy + ly * gh, lw * gw, lh * gh
    for valor, nombre, color, guiones, lado in ((med, "Mediana", "navy", False, "izq"),
                                                (prom, "Promedio", "azul", True, "der")):
        xv = x0 + (valor - bordes[0]) / (bordes[-1] - bordes[0]) * ancho
        ln = linea(s, xv, y0, xv, y0 + alto, color=color, ancho=2.25)
        if guiones:
            from pptx.enum.dml import MSO_LINE_DASH_STYLE
            ln.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        if lado == "izq":
            texto(s, xv - 1.55, y0 - 0.02, 1.5, 0.32, f"{nombre}: {n(valor)}", size=12, color="tinta", align="right")
        else:
            texto(s, xv + 0.05, y0 - 0.02, 1.6, 0.32, f"{nombre}: {n(valor)}", size=12, color="tinta")
    notas = ("Los datos son avisos de Properati de 2019 a 2021, en dos grupos. En gris, los 38.644 de entrenamiento, que "
             "traen el precio; solo 307 están en Chapinero. En naranja, los 10.286 de prueba, sin precio y casi todos en "
             f"Chapinero: es lo que hay que predecir. El histograma muestra el precio del entrenamiento: va de 300 a 1.650 "
             f"millones, se concentra entre 300 y 600 y tiene una cola larga hacia los precios altos. Por esa cola, el "
             f"promedio, {n(prom)} millones (línea punteada), queda a la derecha de la mediana, {n(med)} millones (línea "
             "continua): la mitad de los avisos cuesta menos de 560. La prueba es casi toda de apartamentos.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# Texto como datos, paso a paso: un aviso real regla por regla, y la limpieza hasta la matriz TF-IDF
# ------------------------------------------------------------------------------------------------------------------
def aviso_regla_por_regla(deck, etiqueta, seccion):
    """Lámina: el texto de un aviso real y, fila por fila, qué dice, qué regla se aplica y qué variable resulta."""
    import re as _re
    cand = pd.read_parquet(D / "ppt_texto_candidatos.parquet").set_index("property_id")
    R = json.load(open(D / "ppt_fuentes.json", encoding="utf-8"))
    pid = "84a9da9d84b975bcd0954c36"
    av = cand.loc[pid]
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
    s = deck.nueva("Un aviso real: qué sacamos de su texto y con qué regla", etiqueta=etiqueta, seccion=seccion)
    texto(s, 0.55, 1.25, 12.25, 0.35, f"Aviso de Chapinero de julio de 2021, precio publicado de {n(av['price'] / 1e6, 0)} "
          "millones. Sus columnas traen alcobas (2) y ambientes (2), pero no área ni baños.", size=13, color="navy")
    forma(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.55, 1.62, 12.25, 0.95, relleno="tinte", linea="gris_claro", radio=0.05)
    texto(s, 0.7, 1.66, 11.95, 0.9, [[(f"{str(av['title'])}. ", {"color": "muted"})] + partes], size=14)
    ind = ["terraza o balcón", "chimenea", "vista", "exterior", "gimnasio", "depósito", "zona social", "parqueadero"]
    filas = [("En el aviso", "Regla", "Variable que resulta", "Valor"),
             ("86 m2", "cifra antes de m2, entre 20 y 1.000", "Área del texto y, como faltaba, Área", "86 m²"),
             ("2 habitaciones", "lista fija de sinónimos (paso 1): pasa a 2 alcoba; luego, la cifra antes de alcoba (1 a 10)",
              "Alcobas del texto", "2"),
             ("segundo piso", "lista fija de ordinales (paso 1): pasa a piso 2; luego, la cifra después de piso (1 a 40)", "Piso",
              "2"),
             ("parqueadero", "se menciona sin cifra", "Parqueaderos", "1"),
             ("bano privado, bano social", "no hay cifra antes de baño", "Baños del texto; Baños sigue faltando",
              "vacío"),
             ("balcon, terraza, chimenea, panoramica, exterior, gym, deposito, salon comunal, parqueadero",
              "palabra del diccionario y sin negación", "8 indicadores: " + ", ".join(ind), "1 cada uno"),
             ("no dice ascensor", "sin mención (igual si dijera sin ascensor)", "Ascensor y los otros 10 indicadores",
              "0"),
             ("informes X", "cifra tapada por los organizadores", "se borra y no se usa", "no aplica"),
             ("todo el texto", "LDA sobre los conteos de la matriz", "Tópico 8 (raíces apartament, ban, alcob)", "0,98"),
             ("todo el texto", "ridge de ln(precio) sobre TF-IDF, fuera de pliegue", "Puntaje de precio",
              f"{n(av['puntaje_millones'], 0)} millones")]
    gf = tabla(s, 0.55, 2.7, 12.25, filas, [3.45, 3.55, 4.1, 1.15], size=12, alto_fila=0.28)
    t = gf.table
    for r_ in range(1, len(filas)):
        cel = t.cell(r_, 0).text_frame.paragraphs[0]
        for run in cel.runs:
            run.font.color.rgb = rgb("naranja" if filas[r_][0] == "informes X" else "azul")
    ar = R["avisos_variables_grupo"]
    texto(s, 0.55, 6.5, 12.25, 0.5, "Las cifras y los indicadores se leen sobre el texto limpio de los pasos 1 a 4 de "
          "la lámina anterior; el tópico y el puntaje usan la matriz de los pasos 5 a 8. En todo el entrenamiento, el texto "
          f"sube el área con dato de 31 % a {n(ar['area_imp']['entrenamiento'], 0)} % de los avisos, y los baños de "
          f"{n(R['avisos_brutas_no_nulos_grupo']['bathrooms']['entrenamiento'], 0)} % a "
          f"{n(ar['banos_imp']['entrenamiento'], 0)} %. Azul: lo que se lee; naranja: cifra tapada.", size=12,
          color="muted")
    notas = ("Ahora apliquemos todo a un aviso real de Chapinero, de punta a punta. Sus columnas no traen área ni baños. En el texto, la regla de "
             "área busca una cifra antes de m2 y encuentra 86: esa es su área, y una bandera recuerda que salió del texto. "
             "Con listas fijas, sin ningún modelo, habitaciones se vuelve alcoba y segundo piso se vuelve piso 2; después se "
             "toma la cifra junto a alcoba o a piso. El parqueadero, mencionado sin "
             "cifra, cuenta como uno. Los baños no tienen cifra, así que siguen faltando. Ocho palabras del diccionario "
             "prenden sus indicadores; el ascensor no se menciona y queda en cero, igual que si dijera sin ascensor. La X es "
             "una cifra tapada por los organizadores y se borra. Con todo el texto, el aviso pesa casi por completo en el "
             f"tópico 8, y el ridge le da un puntaje de {n(av['puntaje_millones'], 0)} millones frente a un precio "
             f"publicado de {n(av['price'] / 1e6, 0)}.")
    return s, notas


def texto_paso_a_paso(deck, etiqueta, seccion):
    """Lámina: los siete pasos de la limpieza del texto hasta la matriz TF-IDF, cada uno con un ejemplo y su efecto."""
    lim = json.load(open(D / "ppt_texto_limpieza.json", encoding="utf-8"))
    s = deck.nueva(f"Del texto crudo a la matriz TF-IDF de {n(lim['avisos'], 0)} avisos por {n(lim['terminos'], 0)} términos",
                   etiqueta=etiqueta, seccion=seccion)
    rep = lim["ejemplos_reparacion"]
    filas = [("Paso", "Qué se hace y cómo", "Ejemplo: antes", "Ejemplo: después", "En toda la base"),
             ("1. Normalizar", "minúsculas y sin tildes; se borran las X, los enlaces y el HTML; ordinales y números "
                               "escritos pasan a cifra; sinónimos a una forma (habitación, cuarto o dormitorio: alcoba)",
              "segundo piso, 2 habitaciones, informes X", "piso 2, 2 alcoba, informes", "48.930 avisos"),
             ("2. Reparar letras perdidas", "la base perdió la ñ y las vocales con tilde; se pone la letra que forma una "
                                            "palabra al menos dos veces más frecuente",
              f"{rep[0][0]}, {rep[3][0]}, {rep[2][0]}", f"{rep[0][1]}, {rep[3][1]}, {rep[2][1]}",
              f"{n(lim['reparaciones'], 0)} formas; bao en {n(rep[0][2], 0)} avisos"),
             ("3. Corregir con BETO", "para cada palabra rara, BETO puntúa en su contexto las palabras a una o dos letras; "
                                      "se acepta la más probable si su probabilidad es alta",
              "apartmamento, cocona", "apartamento, cocina",
              f"{n(lim['beto_aceptadas'], 0)} de {n(lim['beto_candidatas'], 0)} candidatas"),
             ("4. Quitar firmas de agentes", "tras contáctame, asesora, informes con o llamar a, se borran hasta 4 palabras "
                                             "que estén en las listas de nombres y apellidos del DNP",
              "asesora [nombre] [apellido]", "asesora", "1.424 avisos (2,9 %)"),
             ("5. Quitar palabras vacías", "listas del DNP y de NLTK, salvo no, sin y ni, que leen la negación, y palabras de "
                                           "oficio como vender; quedan palabras de 2 letras o más y cifras de 1 o 2 dígitos",
              "chico refugio de 86 m2 en piso 2", "chico refugio 86 piso 2", "17.611 a 17.428 palabras"),
             ("6. Reducir a la raíz", "Snowball en español: singular, plural y derivadas cuentan como un término",
              "chico refugio 86 piso 2", "chic refugi 86 pis 2", "11.949 raíces"),
             ("7. Contar palabras y pares", "unigramas y bigramas que aparecen en 5 avisos o más y en menos del 90 %",
              "chic refugi 86 pis", "chic, refugi, 86, chic refugi, 86 pis",
              f"{n(lim['unigramas'], 0)} palabras y {n(lim['bigramas'], 0)} pares"),
             ("8. Ponderar por rareza (TF-IDF)", "peso = (1 + log del conteo) por (1 + log((1 + N)/(1 + avisos con el "
                                                 "término))); cada aviso queda con largo 1",
              "refugi: 2 veces, en 277 avisos; gym: 1 vez, en 1.100", "pesos 0,27 y 0,13",
              f"matriz de {n(lim['avisos'], 0)} por {n(lim['terminos'], 0)}")]
    gf = tabla(s, 0.55, 1.35, 12.25, filas, [1.75, 4.25, 2.3, 2.3, 1.65], size=12, alto_fila=0.29)
    t = gf.table
    for r_ in range(1, len(filas)):
        for c_ in (2, 3):
            for run in t.cell(r_, c_).text_frame.paragraphs[0].runs:
                run.font.color.rgb = rgb("azul")
    texto(s, 0.55, 6.3, 12.25, 0.7, "Los ejemplos son de un aviso real que se sigue completo en la lámina siguiente. "
          "Las cifras y los indicadores de la fuente 2 se leen después del paso 4. La matriz usa los avisos de "
          "entrenamiento y de prueba, lo que es legítimo porque no usa el precio (Lecture 15). Con ella salen los tópicos, "
          "los 50 componentes y el puntaje de precio de la fuente 2c.", size=12, color="muted")
    notas = ("Este es el camino del texto crudo a la matriz, con ejemplos de un aviso real que veremos completo en la "
             "siguiente lámina. Primero normalizamos: minúsculas, sin "
             "tildes, sin las X, y una sola forma para cada cosa: segundo piso se vuelve piso 2 y habitaciones, alcoba. "
             "Segundo, la base perdió la ñ y las tildes, así que bao se repara como bano; fueron 1.210 formas. Tercero, BETO "
             "corrige la ortografía según el contexto: aceptamos 732 correcciones. Cuarto, borramos las firmas de los "
             "agentes con las listas de nombres del DNP. Sobre este texto limpio se leen las cifras y los indicadores. "
             "Quinto, quitamos palabras vacías, pero dejamos no, sin y ni "
             "para leer la negación. Sexto, reducimos a la raíz. Séptimo, contamos palabras y pares de palabras: quedan "
             "44.888 términos. Y octavo, cada conteo se pondera por rareza: refugio, que el aviso repite dos veces y que solo "
             "está en 277 avisos, pesa 0,27; gym, una vez y en 1.100 avisos, pesa 0,13. Esa es la matriz TF-IDF de la "
             "Lecture 15.")
    return s, notas


# ------------------------------------------------------------------------------------------------------------------
# Los modelos del texto: resumen y una lámina por modelo (qué es, cómo se construye y cómo queda en los datos)
# ------------------------------------------------------------------------------------------------------------------
def _tres_tarjetas(s, que_es, como, datos, h=2.5):
    """Tres tarjetas en fila: qué es (definición), cómo se construye (pasos) y cómo queda en los datos (clave)."""
    for x, (titulo, cuerpo, tipo) in zip((0.55, 4.7, 8.85), (("Qué es", que_es, "definicion"),
                                                            ("Cómo se construye", como, "neutra"),
                                                            ("Cómo queda en los datos", datos, "clave"))):
        tarjeta(s, x, 1.33, 3.95, h, titulo, cuerpo, tipo=tipo, maximo=14)


def _negativo(v, d=2):
    t = n(v, d)
    return t.replace("-", "−")


def modelos_texto(deck, etiqueta, seccion):
    """Siete láminas: resumen, BETO, listas del DNP y de NLTK, TF-IDF, LSA, LDA y ridge. Devuelve [(lámina, notas)]."""
    M = json.load(open(D / "ppt_modelos_texto.json", encoding="utf-8"))
    lim = json.load(open(D / "ppt_texto_limpieza.json", encoding="utf-8"))
    top = pd.read_csv(D / "ppt_texto_topicos.csv")
    rid = pd.read_csv(D / "ppt_palabras_ridge.csv")
    dist = pd.read_csv(D / "ppt_texto_chapinero.csv").set_index("termino")
    salida = []

    # A. Resumen
    s = deck.nueva("Los cinco modelos del texto: qué hace cada uno y qué deja en los datos", etiqueta=etiqueta,
                   seccion=seccion)
    filas = [("Modelo", "Qué es", "Recibe", "Cómo se fijó", "Qué deja en los datos"),
             ("BETO", "modelo de lenguaje en español: adivina una palabra tapada por su contexto",
              f"{n(lim['beto_candidatas'], 0)} palabras raras con su contexto", "se acepta la corrección solo si es muy "
              "probable", f"{n(lim['beto_aceptadas'], 0)} palabras corregidas dentro del texto; ninguna columna"),
             ("TF-IDF", "pondera los conteos: pesa más lo que se repite en el aviso y es raro en la base",
              f"conteos de {n(lim['terminos'], 0)} términos", "log del conteo e idf suavizado; cada aviso con largo 1",
              f"matriz de {n(M['N'], 0)} × {n(M['P'], 0)}; insumo de LSA y ridge"),
             ("LSA", "SVD truncada: resume la matriz en pocos ejes, como un PCA del texto", "la matriz TF-IDF",
              "50 ejes, número fijo", "50 columnas: lsa_00 a lsa_49"),
             ("LDA", "modelo de tópicos: cada aviso es una mezcla de temas", f"conteos de {n(lim['unigramas'], 0)} "
              "palabras sueltas", "10 tópicos: la menor perplejidad fuera de muestra",
              "10 columnas: topico_00 a topico_09, que suman 1"),
             ("Ridge", "regresión lineal penalizada de ln(precio) sobre la matriz TF-IDF", "la matriz TF-IDF y el precio "
              "del entrenamiento", "α = 1; fuera de pliegue con 5 grupos", "1 columna: el puntaje de precio")]
    tabla(s, 0.55, 1.33, 12.25, filas, [1.15, 3.35, 2.45, 2.45, 2.85], size=12, alto_fila=0.46)
    texto(s, 0.55, 4.25, 12.25, 0.3, "Qué alimenta a qué: 61 columnas llegan al modelo (50 + 10 + 1); las otras 23 "
          "son los 19 indicadores y las 4 cifras. Las listas del DNP y de NLTK deciden qué palabras salen.", size=12,
          color="muted")
    a = _caja_dag(s, 0.55, 5.15, 2.3, 0.85, "Texto limpio", "pasos 1 a 4; BETO corrige")
    b1 = _caja_dag(s, 3.25, 4.6, 2.5, 0.85, "Conteos", f"{n(lim['terminos'], 0)} términos")
    b2 = _caja_dag(s, 3.25, 5.85, 2.5, 0.8, "Conteos", f"{n(lim['unigramas'], 0)} palabras sueltas")
    c1 = _caja_dag(s, 6.15, 4.6, 2.4, 0.85, "TF-IDF", "pesos por rareza")
    c2 = _caja_dag(s, 6.15, 5.55, 2.4, 0.45, "Precio del entrenamiento", "")
    d1 = _caja_dag(s, 9.0, 4.6, 3.8, 0.5, "LSA: lsa_00 a lsa_49", "", relleno="azul_claro", color_titulo="azul")
    d2 = _caja_dag(s, 9.0, 5.35, 3.8, 0.5, "Ridge: puntaje de precio", "", relleno="azul_claro", color_titulo="azul")
    d3 = _caja_dag(s, 9.0, 6.1, 3.8, 0.5, "LDA: topico_00 a topico_09", "", relleno="azul_claro",
                   color_titulo="azul")
    for o, d_ in ((a, b1), (a, b2), (b1, c1), (c1, d1), (c1, d2), (c2, d2), (b2, d3)):
        _arista(s, o, d_)
    salida.append((s, "Detrás de las variables del texto hay cinco modelos. BETO es un modelo de lenguaje que solo corrige "
                      "la ortografía: no crea columnas. TF-IDF pondera los conteos de términos y produce la matriz que usan "
                      "otros dos modelos. LSA resume esa matriz en 50 ejes, y esos 50 números por aviso son columnas del "
                      "modelo. LDA, con los conteos de las palabras sueltas, da a cada aviso su mezcla de 10 tópicos. Y el "
                      "ridge aprende de la matriz y del precio de entrenamiento un puntaje de precio. En total, 61 columnas "
                      "llegan al modelo desde estos modelos. Veamos cada uno."))

    # B. BETO
    s = deck.nueva("BETO: un modelo de lenguaje que corrige la ortografía según el contexto", etiqueta=etiqueta,
                   seccion=seccion)
    _tres_tarjetas(s, "BETO es un BERT entrenado con texto en español (Cañete y otros, 2020; versión sin mayúsculas). Si se "
                      "tapa una palabra, calcula la probabilidad de cada palabra posible en ese lugar mirando el contexto a "
                      "los dos lados.",
                   ["1. Sospechosas: palabras de 4 letras o más, en menos de 5 avisos y que BETO no conoce.",
                    "2. Candidatas: palabras frecuentes (20 avisos o más) a 1 letra, o a 2 si tiene 6 letras o más.",
                    "3. Se tapa la palabra en hasta 2 avisos, con 40 palabras a cada lado, y se elige la candidata más "
                    "probable."],
                   ["Se acepta si su log-probabilidad es al menos −3 (o −5 si difiere en una letra) y no es solo agregar "
                    "o quitar una s.",
                    f"{n(lim['beto_aceptadas'], 0)} de {n(lim['beto_candidatas'], 0)} aceptadas. La palabra se corrige "
                    "dentro del texto, antes de leer cifras e indicadores y de contar: no crea columnas."])
    filas = [("Palabra en el aviso", "Candidata", "Letras de diferencia", "Log-probabilidad en su contexto", "Decisión"),
             ("sauiza", "suiza", "1", "−0,002", "aceptada"),
             ("cocona", "cocina", "1", "−3,03", "aceptada: a una letra basta −5"),
             ("tcon", "con", "1", "−3,76", "aceptada"),
             ("apartmamento", "apartamento", "1", "−3,78", "aceptada"),
             ("intetior", "interior", "1", "−7,66", "rechazada: poco probable en su contexto"),
             ("certicado", "certificado", "2", "−9,60", "rechazada: poco probable"),
             ("juegoss", "juegos", "1", "−0,76", "rechazada: solo quita una s")]
    texto(s, 0.55, 3.98, 12.25, 0.32, "Ejemplos de la auditoría, que guarda las 3.652 candidatas con su decisión",
          size=13, color="navy")
    tabla(s, 0.55, 4.33, 12.25, filas, [2.4, 2.2, 1.75, 2.6, 3.3], size=12, alto_fila=0.29)
    salida.append((s, "El primer modelo es BETO, un modelo de lenguaje en español del tipo BERT. Su tarea es adivinar una "
                      "palabra tapada a partir del contexto. Lo usamos como corrector: tomamos las palabras raras que BETO "
                      "no conoce, buscamos palabras frecuentes que difieran en una o dos letras y le preguntamos a BETO "
                      "cuál es más probable en ese aviso. Somos conservadores: aceptamos solo si la probabilidad es alta. "
                      "Por eso cocona pasa a cocina, pero intetior no se corrige: interior es la palabra correcta, pero en "
                      "su contexto no fue lo bastante probable. BETO no crea columnas: cambia el texto antes de contar."))

    # B2. Listas del DNP y de NLTK
    s = deck.nueva("Las listas del DNP y de NLTK: qué son y cómo se usan en la limpieza", etiqueta=etiqueta,
                   seccion=seccion)
    tarjeta(s, 0.55, 1.33, 3.95, 2.15, "ConTexto, del DNP", "Librería de texto en español de la Unidad de Científicos de "
            "Datos del Departamento Nacional de Planeación (Colombia), con licencia MIT; la cita la Lecture 15. No la "
            "instalamos, porque no funciona con Python 3.13: copiamos sus listas y escribimos nuestras funciones.",
            tipo="definicion", maximo=14)
    tarjeta(s, 4.7, 1.33, 3.95, 2.15, "NLTK", "Natural Language Toolkit, la librería clásica de procesamiento de lenguaje "
            "en Python (Bird, Klein y Loper, 2009). Trae listas de palabras vacías por idioma: guardamos una copia de la "
            "lista en español.", tipo="definicion", maximo=14)
    tarjeta(s, 8.85, 1.33, 3.95, 2.15, "Cómo queda en los datos", [
        "No crean columnas: deciden qué palabras salen del texto.",
        "Las firmas de agentes se borran en 1.424 avisos (2,9 %), y las 327 palabras vacías nunca entran a la matriz."],
        tipo="clave", maximo=14)
    filas = [("Lista", "De dónde sale", "Palabras", "Ejemplos", "Cómo se usa"),
             ("Palabras vacías del DNP", "ConTexto: lista_sw_es", "329", "de, la, que, además, aunque, sino",
              "se quitan al contar (paso 5)"),
             ("Palabras vacías de NLTK", "NLTK: stopwords en español", "313; 306 sin tildes, todas ya en la del DNP",
              "de, la, que, el, en, y", "se unen con la del DNP"),
             ("Unión que usamos", "las dos, en minúsculas y sin tildes", "327",
              "se agregan ser, estar, haber y tener; se sacan no, sin, ni, estado y estados",
              "se descarta cada palabra que esté en ella; no, sin y ni se quedan para la negación"),
             ("Palabras de oficio", "lista del equipo", "18", "vender, venta, inmueble, oferta, contacto, bogota",
              "se quitan sus raíces al contar"),
             ("Nombres y apellidos del DNP", "ConTexto: 3 listas de nombres y 1 de apellidos",
              "3.572, 6.439, 6.316 y 2.388", "nombres de pila y apellidos comunes",
              "solo en las 4 palabras tras una expresión de contacto (paso 4); se protegen barrios como Chicó o "
              "Rosales")]
    tabla(s, 0.55, 3.62, 12.25, filas, [2.0, 2.25, 1.95, 2.75, 3.3], size=12, alto_fila=0.36)
    salida.append((s, "Dos objetos externos deciden qué palabras salen del texto. El primero es ConTexto, la librería de "
                      "texto del Departamento Nacional de Planeación, que cita la clase; como no funciona con nuestra "
                      "versión de Python, tomamos sus listas. El segundo es NLTK, la librería clásica de lenguaje en Python. "
                      "Las dos traen listas de palabras vacías, como de, la o que: unimos las dos, que dan 327 palabras, y "
                      "sacamos no, sin y ni para poder leer la negación. Además, el DNP tiene listas de nombres y apellidos: "
                      "con ellas borramos la firma del agente, pero solo después de expresiones como contáctame o asesora, "
                      "y protegiendo nombres que también son barrios, como Rosales o Chicó. Pasó en 1.424 avisos. Ninguna "
                      "de estas listas crea columnas: solo limpian el texto antes de contar."))

    # C. TF-IDF
    s = deck.nueva("TF-IDF: cuánto pesa cada término en cada aviso", etiqueta=etiqueta, seccion=seccion)
    _tres_tarjetas(s, "Una ponderación de la matriz de conteos (Lecture 15). Un término pesa más si se repite en el aviso "
                      "(TF) y si aparece en pocos avisos (IDF): las palabras que todos usan, como apartamento, pesan poco.",
                   ["1. Conteos de cada término en cada aviso (paso 7).",
                    "2. TF = 1 + ln(veces en el aviso).",
                    f"3. IDF = 1 + ln((1 + N)/(1 + avisos con el término)), con N = {n(M['N'], 0)}.",
                    "4. Peso = TF × IDF; cada fila se divide por su largo, para que un aviso largo no pese más."],
                   [f"Una matriz dispersa de {n(M['N'], 0)} × {n(M['P'], 0)} con {n(M['nnz'], 0)} pesos distintos de cero "
                    f"({n(M['densidad_pct'], 2)} %): un aviso típico usa {n(M['terminos_por_aviso_mediana'], 0)} términos.",
                    "No entra como columnas: es el insumo del LSA y del ridge. Usa entrenamiento y prueba, porque no usa "
                    "el precio."])
    av = M["aviso"]
    texto(s, 0.55, 3.98, 12.25, 0.32, f"Una fila de la matriz: el aviso del ejemplo ({av['terminos']} términos; largo de la "
          f"fila antes de dividir: {n(av['norma'], 2)})", size=13, color="navy")
    orden = sorted(av["tabla"], key=lambda r: -r["peso"])
    filas = [("Término (raíz)", "Veces en el aviso", "Avisos con el término", "TF", "IDF", "TF × IDF",
              "Peso final")]
    for r in orden:
        tf_ = 1 + math.log(r["conteo"])
        filas.append((r["termino"], str(r["conteo"]), n(r["df"], 0), n(tf_, 2), n(r["idf"], 2),
                      n(r["peso_sin_normalizar"], 2), n(r["peso"], 2)))
    tabla(s, 0.55, 4.33, 12.25, filas, [2.45, 1.55, 1.9, 1.3, 1.5, 1.7, 1.85], size=12, alto_fila=0.26)
    salida.append((s, "El segundo modelo no se estima: es una ponderación. Cada celda de la matriz dice cuántas veces aparece "
                      "un término en un aviso; TF-IDF la convierte en un peso. Miren la fila del aviso del ejemplo. Refugio "
                      "aparece dos veces y solo en 277 avisos: TF de 1,69, IDF de 6,17, peso 10,45, y al dividir por el "
                      "largo de la fila, 0,27. Apartamento también aparece dos veces, pero está en 40.371 avisos: su IDF es "
                      "1,19 y su peso final 0,05. Así, lo que distingue al aviso pesa más que lo que todos dicen. La matriz "
                      "es casi toda ceros: un aviso usa unos 75 de los 44.888 términos."))

    # D. LSA
    s = deck.nueva("LSA: la matriz TF-IDF resumida en 50 ejes", etiqueta=etiqueta, seccion=seccion)
    L = M["lsa"]
    _tres_tarjetas(s, "Análisis semántico latente: una SVD truncada de la matriz, X ≈ U Σ V', con 50 ejes. Cada eje es una "
                      "combinación de términos que suelen ir juntos, y cada aviso recibe su coordenada en cada eje. Es el "
                      "PCA del texto de la Lecture 15, sin centrar.",
                   ["1. Insumo: la matriz TF-IDF completa.",
                    "2. SVD truncada aleatorizada con 50 componentes y semilla 2026.",
                    "3. El número de ejes, 50, se fijó; no se afinó.",
                    "4. Coordenada del aviso en el eje k: su fila de U por el valor singular del eje."],
                   ["50 columnas, lsa_00 a lsa_49, sin faltantes.",
                    f"Proporción de varianza explicada (Lecture 15): los 50 ejes explican el "
                    f"{n(L['pve_pct_acumulada_50'], 1)} % de la varianza de la matriz. El primero, solo el "
                    f"{n(L['pve_pct_por_componente'][0], 1)} %: como LSA no resta el promedio, ese eje recoge el "
                    "vocabulario común a todos los avisos."], h=2.2)
    comp = {c["k"]: c for c in L["componentes"]}
    e = L["pve_pct_por_componente"]
    acum_e = [sum(e[:k + 1]) for k in range(len(e))]
    # Screeplot como el de la Lecture 15 (diapositiva 65): A, varianza explicada por eje; B, acumulada
    for x_, tit_, ys_, ymax_, uy_ in ((0.55, "A. Varianza explicada por eje (%)", e, 1.5, 0.5),
                                      (3.85, "B. Varianza explicada acumulada (%)", acum_e, 15, 5)):
        texto(s, x_, 3.68, 3.2, 0.3, tit_, size=12, color="navy")
        lineas_xy(s, x_ - 0.1, 3.98, 3.3, 2.75, [dict(nombre="v", x=list(range(1, 51)), y=ys_, color="azul", ancho=1.5,
                                                      marcador="circulo", tam=4)],
                  leyenda=False, minx=0, maxx=50, miny=0, maxy=ymax_, fmt_x="0", fmt_y="0.0" if ymax_ < 5 else "0",
                  titulo_x="Componente", unidad_x=10, unidad_y=uy_)
    filas = [("Eje", "Varianza", "Cómo lo leemos", "Aviso del ejemplo"),
             ("lsa_00", f"{n(e[0], 1)} %", "vocabulario común de un aviso de apartamento", _negativo(comp[0]["valor_aviso"])),
             ("lsa_01", f"{n(e[1], 1)} %", "firma de una agente frente a la descripción típica",
              _negativo(comp[1]["valor_aviso"])),
             ("lsa_02", f"{n(e[2], 1)} %", "dos plantillas de agencia, una contra otra", _negativo(comp[2]["valor_aviso"])),
             ("lsa_03", f"{n(e[3], 1)} %", "casas y comercio frente a apartamento en edificio",
              _negativo(comp[3]["valor_aviso"]))]
    texto(s, 7.3, 3.68, 5.5, 0.3, "Los cuatro primeros ejes y el aviso del ejemplo", size=12, color="navy")
    tabla(s, 7.3, 4.0, 5.5, filas, [0.8, 0.9, 2.65, 1.15], size=12, alto_fila=0.46)
    salida.append((s, "El tercer modelo, LSA, resume la matriz. Una descomposición en valores singulares encuentra los ejes "
                      "en los que más varía el texto y se queda con 50. Cada eje es una mezcla de términos: el primero junta "
                      "el vocabulario común de un aviso de apartamento; otros separan plantillas de agencias, o casas y "
                      "locales frente a apartamentos en edificio. Cada aviso recibe 50 números, sus coordenadas, y esas son "
                      "las columnas lsa_00 a lsa_49. El aviso del ejemplo tiene 0,27 en el primer eje, más que el promedio, "
                      "porque es una descripción típica, y −0,11 en el cuarto: es un apartamento en edificio. ¿Cuánto "
                      "resume? Como en el PCA de la clase, con la proporción de varianza explicada: los 50 ejes explican "
                      f"el {n(L['pve_pct_acumulada_50'], 1)} % de la varianza de la matriz. El primer eje explica casi nada, "
                      f"{n(L['pve_pct_por_componente'][0], 1)} %, porque LSA no resta el promedio y ese eje recoge el "
                      "vocabulario que comparten todos los avisos; el que más explica es el segundo, "
                      f"{n(L['pve_pct_por_componente'][1], 1)} %. Es poco porque los avisos usan vocabularios muy variados "
                      "y la información queda repartida en miles de direcciones; por eso además usamos el ridge, que "
                      "trabaja con la matriz completa. Comprobamos que nuestra SVD recalculada coincide exactamente con las "
                      "columnas guardadas."))

    # E. LDA
    s = deck.nueva("LDA: cada aviso como una mezcla de 10 tópicos", etiqueta=etiqueta, seccion=seccion)
    _tres_tarjetas(s, "Asignación latente de Dirichlet (Blei, Ng y Jordan, 2003). Cada tópico es una lista de probabilidades "
                      "de palabras y cada aviso una mezcla de tópicos: cada palabra se escribe eligiendo un tópico según la "
                      "mezcla y luego una palabra según el tópico.",
                   [f"1. Insumo: conteos de las {n(lim['unigramas'], 0)} palabras sueltas, sin pares.",
                    "2. K de 5 a 25: ajuste con el 80 % de los avisos y perplejidad en el 20 % restante. K = 10 dio la "
                    "menor (327,0), aunque entre 5 y 20 casi no cambia.",
                    "3. Ajuste final con los 48.930 avisos: Bayes variacional en línea, 15 pasadas, semilla 2026."],
                   ["10 columnas, topico_00 a topico_09: los pesos de cada aviso, que suman 1.",
                    "El aviso del ejemplo pesa 0,98 en el tópico 8. Los nombres los puso el equipo leyendo las palabras de "
                    "cada tópico."], h=2.2)
    filas = [("Tópico", "Nombre del equipo", "Palabras más probables", "Peso: prueba", "Peso: entrenamiento",
              "Con el precio")]
    for _, r in top.iterrows():
        k = int(r["topico"])
        filas.append((str(k), TOPICOS[k], ", ".join(str(r["probables"]).split(", ")[:5]),
                      f"{n(100 * r['media_chapinero'], 0)} %", f"{n(100 * r['media_resto'], 0)} %",
                      _negativo(r["rho_precio"])))
    tabla(s, 0.55, 3.66, 12.25, filas, [0.75, 3.0, 3.85, 1.4, 1.85, 1.4], size=12, alto_fila=0.24)
    texto(s, 0.55, 6.66, 12.25, 0.3, "Peso: promedio de cada tópico en los avisos de prueba y de entrenamiento fuera de "
          "Chapinero. Con el precio: correlación de Spearman en el entrenamiento.", size=12, color="muted")
    salida.append((s, "El cuarto modelo es LDA, un modelo de tópicos. Supone que cada aviso se escribe mezclando temas, y "
                      "que cada tema tiene sus palabras más probables. Con los conteos, encuentra los temas y la mezcla de "
                      "cada aviso. Probamos de 5 a 25 tópicos y nos quedamos con 10, el que mejor predijo las palabras de "
                      "avisos que el modelo no vio, aunque la diferencia es pequeña. Cada aviso recibe 10 pesos que suman 1. "
                      "La tabla muestra que la prueba habla mucho más de apartamentos en edificio y casi nada de casas, y que "
                      "el tópico de cercanía a comercio es el más ligado a precios bajos."))

    # F. Ridge
    s = deck.nueva("Ridge: un puntaje de precio aprendido de las palabras", etiqueta=etiqueta, seccion=seccion)
    Rr = M["ridge"]
    _tres_tarjetas(s, "Una regresión lineal de ln(precio) sobre los 44.888 pesos TF-IDF con penalización: minimiza la suma "
                      "de (ln p − b − x'β)² más α por la suma de β². Con tantos términos, la penalización encoge los "
                      "coeficientes para que un término raro no explique un precio por casualidad (Gentzkow y otros, 2019).",
                   ["1. α = 1, fijo; la validación cruzada confirma que es el mejor entre 0,03 y 100.",
                    "2. Fuera de pliegue: el entrenamiento se parte en 5 grupos, con los textos duplicados siempre juntos; "
                    "el puntaje de cada grupo sale de un ridge ajustado con los otros 4.",
                    "3. Prueba: un ridge con los 38.644 avisos de entrenamiento.",
                    "4. En cada validación se repite dentro del pliegue."],
                   ["1 columna: el puntaje, en ln(millones).",
                    f"Sin ver el precio de cada aviso, en el entrenamiento tiene correlación {n(Rr['corr_oof'], 2)} con "
                    f"ln(precio) y explica el {n(100 * Rr['r2_oof'], 0)} % de su varianza.",
                    f"Aviso del ejemplo: {n(Rr['puntaje_aviso'], 2)}, es decir {n(Rr['puntaje_aviso_millones'], 0)} "
                    "millones; su precio era 749."])
    LEG = {"3 parqueader": "3 parqueaderos", "cas": "casa", "ascensor priv": "ascensor privado",
           "4 parqueader": "4 parqueaderos", "carolin": "Carolina", "aprovech excelent": "aproveche excelente",
           "2 parqueader": "2 parqueaderos", "chic": "Chicó", "apartaestudi": "apartaestudio",
           "parqueader": "parqueadero", "1 parqueader": "1 parqueadero", "habit": "habitación", "cedrit": "Cedritos",
           "sal comedor": "sala comedor", "1": "1 (cifra)", "ban": "baño"}
    sube = rid[rid["lado"] == "sube"].sort_values("coef", ascending=False).head(8)
    baja = rid[rid["lado"] == "baja"].sort_values("coef").head(8)
    texto(s, 0.55, 3.98, 8.1, 0.32, "Coeficientes del ridge ajustado con todo el entrenamiento: los 8 más altos y los 8 más "
          "bajos", size=13, color="navy")
    for x0, d_, col, tit in ((0.45, sube, "azul", "Suben el puntaje"), (4.45, baja, "naranja", "Bajan el puntaje")):
        texto(s, x0 + 0.1, 4.33, 3.8, 0.3, tit, size=12, color="muted")
        ch = barras(s, x0, 4.6, 4.0, 2.4, [LEG.get(t, t) for t in d_["termino"]],
                    [("coef", [abs(v) for v in d_["coef"]], col)], minimo=0, maximo=2.3, gap=40, grid=False)
        etiquetas_puntos(ch, [[_negativo(v) for v in d_["coef"]]])
    filas = [("Término", "Prueba", "Entrenamiento")]
    for t, nombre in (("rosal", "Rosales"), ("chic nort", "Chicó Norte"), ("virrey", "Virrey"), ("parqu 93", "Parque 93"),
                      ("chic reserv", "Chicó Reservado"), ("cabrer", "Cabrera")):
        filas.append((nombre, f"{n(dist.loc[t, 'pct_chapinero'], 1)} %", f"{n(dist.loc[t, 'pct_resto'], 2)} %"))
    texto(s, 8.85, 3.98, 3.95, 0.32, "Barrios de Chapinero en el texto", size=13, color="navy")
    tabla(s, 8.85, 4.33, 3.95, filas, [1.75, 1.0, 1.2], size=12, alto_fila=0.27)
    texto(s, 8.85, 6.25, 3.95, 0.75, "Entrenamiento fuera de Chapinero. Con tan pocos avisos que los nombran, el ridge "
          "casi no aprende cuánto valen.", size=12, color="muted")
    salida.append((s, "El quinto modelo es el que más aporta: un ridge, la regresión con penalización L2 de la clase, de ln "
                      "del precio sobre los 44.888 pesos TF-IDF. Lo importante es cómo se construye: fuera de pliegue. Para "
                      "cada aviso de entrenamiento, el puntaje viene de un ridge que no vio su precio, y los textos "
                      "duplicados van siempre juntos para que un aviso no se prediga con su copia. Así, el puntaje tiene "
                      f"correlación {n(Rr['corr_oof'], 2)} con el precio sin hacer trampa. Los coeficientes se leen bien: "
                      "más parqueaderos, casa, ascensor privado o Chicó suben el precio; apartaestudio, un parqueadero, "
                      "Cedritos o Kennedy lo bajan. Pero los barrios de Chapinero casi no aparecen en el entrenamiento, así "
                      "que el ridge casi no sabe cuánto valen."))
    return salida


def texto_pasos_y_modelos(deck, etiqueta, seccion):
    """Una lámina: los 8 pasos del texto, el modelo o lista que usa cada uno y lo que queda en los datos (84 variables)."""
    lim = json.load(open(D / "ppt_texto_limpieza.json", encoding="utf-8"))
    M = json.load(open(D / "ppt_modelos_texto.json", encoding="utf-8"))
    pesos = {r["termino"]: r["peso"] for r in M["aviso"]["tabla"]}
    s = deck.nueva("Del texto crudo a 84 variables: pasos, modelos y lo que queda en los datos", etiqueta=etiqueta,
                   seccion=seccion)
    pasos = [
        ("1. Normalizar", "reglas fijas", False,
         "minúsculas, sin tildes ni X; ordinales y sinónimos: segundo piso pasa a piso 2"),
        ("2. Reparar letras perdidas", "regla de frecuencia", False,
         f"pone la ñ o la vocal que faltaba: bao pasa a bano; {n(lim['reparaciones'], 0)} formas reparadas"),
        ("3. Corregir la ortografía", "modelo: BETO", True,
         f"la palabra más probable en su contexto: apartmamento pasa a apartamento ({n(lim['beto_aceptadas'], 0)})"),
        ("4. Quitar firmas de agentes", "lista: nombres del DNP", True,
         "borra nombres después de contáctame o asesora; 1.424 avisos (2,9 %)"),
        ("5. Quitar palabras vacías", "listas: DNP y NLTK", True,
         "quita 327 palabras como de, la o que; conserva no, sin y ni para leer la negación"),
        ("6. Reducir a la raíz", "Snowball", False,
         "apartamento y apartamentos cuentan como un solo término: apartament; 11.949 raíces"),
        ("7. Contar palabras y pares", "conteos", False,
         f"en 5 avisos o más: {n(lim['unigramas'], 0)} palabras y {n(lim['bigramas'], 0)} pares, matriz de "
         f"{n(M['N'], 0)} × {n(M['P'], 0)}"),
        ("8. Ponderar por rareza", "TF-IDF", True,
         f"pesa más lo que se repite en el aviso y es raro en la base: refugi {n(pesos['refugi'], 2)}; apartament "
         f"{n(pesos['apartament'], 2)}"),
    ]
    x0, w, h, paso_y, y0 = 0.55, 7.45, 0.56, 0.62, 1.36
    cajas = []
    for i, (titulo, herramienta, es_modelo, detalle) in enumerate(pasos):
        y = y0 + i * paso_y
        caja = forma(s, MSO_SHAPE.ROUNDED_RECTANGLE, x0, y, w, h, relleno="tinte",
                     linea="azul_suave" if es_modelo else "gris_claro", radio=0.12)
        texto(s, x0 + 0.12, y + 0.03, 4.3, 0.26, titulo, size=13, color="navy")
        texto(s, x0 + w - 2.75, y + 0.03, 2.63, 0.26, herramienta, size=12, color="azul" if es_modelo else "muted",
              align="right")
        texto(s, x0 + 0.12, y + 0.28, w - 0.24, 0.26, detalle, size=12, color="tinta")
        cajas.append(caja)
    xr, wr = 8.5, 4.3
    tarjeta(s, xr, 1.36, wr, 1.62, "84 variables del texto entran al LightGBM",
            ["Son ramas paralelas: ninguna alimenta a otra, y entran juntas con las otras 78 variables.",
             "BETO, las listas y TF-IDF no crean columnas: preparan el texto y la matriz."], tipo="clave", maximo=14)

    def salida_(y, alto, titulo, detalle, modelo=True):
        caja = forma(s, MSO_SHAPE.ROUNDED_RECTANGLE, xr, y, wr, alto, relleno="azul_claro" if modelo else "tinte",
                     linea="azul_suave" if modelo else "gris_claro", radio=0.12)
        texto(s, xr + 0.12, y + 0.03, wr - 0.24, 0.26, titulo, size=13, color="azul" if modelo else "navy")
        texto(s, xr + 0.12, y + 0.28, wr - 0.24, alto - 0.3, detalle, size=12, color="tinta")
        return caja

    centro = lambda i: y0 + i * paso_y + h / 2  # noqa: E731
    a = salida_(centro(3) - 0.33, 0.66, "Indicadores y cifras: 23 columnas",
                "reglas sobre el texto limpio; no usan el precio", modelo=False)
    texto(s, xr, centro(3) + 0.45, wr, 1.0, "No supervisado: no usa el precio. Supervisado: aprende del precio del "
          "entrenamiento (Lecture 15). Cada modelo tiene su lámina más adelante.", size=12, color="muted")
    b = salida_(centro(6) - 0.3, 0.6, "LDA: 10 columnas", "no supervisado: agrupa las palabras en 10 tópicos")
    c = salida_(centro(7) - 0.3, 0.6, "LSA: 50 columnas", "no supervisado: resume la matriz en 50 ejes")
    d = salida_(centro(7) + 0.36, 0.6, "Ridge: 1 columna", "supervisado: aprende del precio del entrenamiento")
    for o, d_ in ((cajas[3], a), (cajas[6], b), (cajas[7], c), (cajas[7], d)):
        _arista(s, o, d_)
    notas = ("Este es el camino completo del texto, de punta a punta. Los cuatro primeros pasos limpian: normalizamos con "
             "reglas fijas, reparamos la ñ y las tildes perdidas, BETO corrige la ortografía según el contexto y, con las "
             "listas de nombres del DNP, borramos la firma del agente. Sobre ese texto limpio se leen los 19 indicadores y "
             "las 4 cifras: 23 columnas. Los pasos 5 a 7 cuentan: quitamos las palabras vacías de las listas del DNP y de "
             "NLTK, reducimos a la raíz y contamos palabras y pares. Con esos conteos, LDA da 10 columnas de tópicos. Y el "
             "paso 8, TF-IDF, pondera por rareza; de esa matriz salen las 50 columnas de LSA y, con el precio del "
             "entrenamiento, el puntaje del ridge. ¿Cómo se relacionan? Son ramas paralelas del mismo texto: ninguna "
             "alimenta a otra. LDA y LSA son no supervisados: resumen el texto sin mirar el precio. El ridge es "
             "supervisado: usa la misma matriz que LSA, pero aprende qué palabras van con precios altos o bajos. Las 84 "
             "variables entran juntas al LightGBM, con las otras 78, y el boosting decide cuáles usar. En las láminas "
             "siguientes vemos un aviso real y después cada modelo en detalle.")
    return s, notas


def lsa_graficas(deck, etiqueta, seccion):
    """Lámina: proporción de varianza explicada (screeplot de los 50 ejes) y el plano lsa_03 por lsa_08 de los avisos."""
    M = json.load(open(D / "ppt_modelos_texto.json", encoding="utf-8"))
    e = M["lsa"]["pve_pct_por_componente"]
    acum = M["lsa"]["pve_pct_acumulada_50"]
    s = deck.nueva("LSA en gráficas: varianza explicada y dónde cae cada aviso", etiqueta=etiqueta, seccion=seccion)
    tarjeta(s, 0.55, 1.33, 4.35, 2.55, "Proporción de varianza explicada", [
        "Como en el PCA de la Lecture 15: qué parte de la variación entre avisos captura cada eje. La varianza total es "
        "la suma de las varianzas de las columnas de la matriz TF-IDF.",
        f"Los 50 ejes explican el {n(acum, 1)} %. El que más explica es lsa_01, con {n(e[1], 1)} %; lsa_00, solo "
        f"{n(e[0], 1)} %, porque recoge el vocabulario común a todos los avisos."], tipo="definicion", maximo=14)
    texto(s, 0.55, 4.0, 4.35, 0.32, "Screeplot: varianza explicada por cada eje (%)", size=13, color="navy")
    lineas_xy(s, 0.45, 4.3, 4.5, 2.6, [dict(nombre="Varianza explicada", x=list(range(1, 51)), y=e, color="azul",
                                             ancho=1.5, marcador="circulo", tam=4)],
              leyenda=False, minx=0, maxx=50, miny=0, maxy=1.5, fmt_x="0", fmt_y="0.0", titulo_x="Eje",
              titulo_y="%", unidad_x=10, unidad_y=0.5)
    imagen(s, IMG / "lsa_plano.png", 5.15, 1.33, w=7.7,
           alt="Avisos en el plano de los ejes lsa_03 y lsa_08: la prueba en naranja y el entrenamiento en gris; a la "
               "derecha, el entrenamiento coloreado por precio")
    texto(s, 5.15, 6.33, 7.7, 0.62, "Cada punto es un aviso, ubicado en dos de los 50 ejes. La prueba se concentra en "
          "apartamentos en edificio; arriba en lsa_08 están los avisos más caros (Spearman 0,38).", size=12, color="muted")
    notas = ("Como en el PCA de la clase, medimos cuánto resume cada eje con la proporción de varianza explicada, y la "
             f"graficamos en un screeplot. Los 50 ejes explican el {n(acum, 1)} % de la varianza de la matriz. El primero "
             f"explica casi nada, {n(e[0], 1)} %, porque LSA no resta el promedio y ese eje recoge el vocabulario que "
             f"comparten todos los avisos; el que más explica es el segundo, {n(e[1], 1)} %, y de ahí la curva cae "
             "despacio: no hay un codo claro, como dice la Lecture 15 no hay un criterio objetivo para el número de ejes. "
             "A la derecha, cada aviso es un punto según sus coordenadas en dos "
             "ejes: el 3, que va de apartamento en edificio a casas y comercio, y el 8, que va de apartamento estándar a "
             "vivienda amplia con servicio y Chicó. La prueba, en naranja, se concentra a la izquierda y arriba: "
             "apartamentos en edificio de vivienda amplia. Y en el entrenamiento, coloreado por precio, los caros quedan "
             "arriba: el eje 8 tiene correlación de Spearman de 0,38 con el precio. Ese mismo eje aparece más adelante "
             "como una de las preguntas del árbol.")
    return s, notas


def lda_ridge_graficas(deck, etiqueta, seccion):
    """Lámina con las dos gráficas del cuaderno 02_LDA de la Lecture 15: peso de un tópico según el precio y error de
    validación cruzada de la regresión con tópicos frente a la regresión con palabras (datos de 27_datos_laminas_graficas_texto.py)."""
    G = json.load(open(D / "ppt_graficas_clase.json", encoding="utf-8"))
    pal = min(G["palabras"], key=lambda r: r["mse"])
    top = min(G["topicos"], key=lambda r: r["mse"])
    r2_pal, r2_top = 1 - pal["mse"] / G["var_lp"], 1 - top["mse"] / G["var_lp"]
    caja = G["caja_topico_5"]
    s = deck.nueva("LDA frente a ridge: las gráficas del cuaderno de clase", etiqueta=etiqueta, seccion=seccion)
    texto(s, 0.55, 1.3, 5.0, 0.5, "Peso del tópico 5 (cercanía a comercio) según el precio", size=13, color="navy")
    imagen(s, IMG / "lda_caja_topico5.png", 0.55, 1.8, w=5.0,
           alt="Diagrama de caja del peso del tópico 5 por quintil de precio en el entrenamiento")
    texto(s, 0.55, 5.65, 5.0, 1.1, f"A mayor precio, menos peso de este tópico: su promedio baja de {n(caja[0]['media'], 2)} "
          f"a {n(caja[-1]['media'], 2)}. Caja: percentiles 25 a 75; línea: mediana; punto: promedio; bigotes: "
          "percentiles 10 y 90.", size=12, color="muted")
    texto(s, 5.85, 1.3, 6.95, 0.5, "Error de validación cruzada de ln(precio): tópicos frente a palabras", size=13,
          color="navy")
    imagen(s, IMG / "cv_topicos_palabras.png", 5.85, 1.8, w=6.95,
           alt="Error cuadrático medio fuera de pliegue según la penalización, con los 10 tópicos y con las palabras")
    texto(s, 5.85, 5.65, 6.95, 1.1, f"Con los 10 tópicos el error casi no baja: explican el {n(100 * r2_top, 0)} % de la "
          f"varianza. Con las palabras baja a {n(pal['mse'], 3)}: explican el {n(100 * r2_pal, 0)} %, con el mínimo en "
          "α = 1, el que usa el modelo. Entrenamiento, 5 grupos por texto.", size=12, color="muted")
    notas = ("Estas son las dos gráficas del cuaderno de LDA de la clase, con nuestros datos. A la izquierda, como el "
             "diagrama de caja del cuaderno: el peso de un tópico según el resultado. El tópico de cercanía a comercio pesa "
             f"cada vez menos a medida que sube el precio: su promedio pasa de {n(caja[0]['media'], 2)} en los avisos más "
             f"baratos a {n(caja[-1]['media'], 2)} en los más caros. Los tópicos sí dicen algo del precio. A la derecha, la "
             "comparación del cuaderno entre la regresión con tópicos y la regresión con palabras, con su error de "
             f"validación cruzada. Con los 10 tópicos se explica apenas el {n(100 * r2_top, 0)} % de la varianza del "
             f"logaritmo del precio; con las 44.888 palabras, el {n(100 * r2_pal, 0)} %. Por eso el texto llega al modelo "
             "sobre todo por el puntaje del ridge. Y la curva tiene su mínimo justo en α = 1, el valor que dejamos fijo: "
             "la validación lo confirma.")
    return s, notas
