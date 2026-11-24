"""Láminas que comparten best_equipo_02 y compare_equipo_02, y láminas de análisis que se arman con datos guardados.

- chapinero_otro_mercado: en qué difiere Chapinero del resto de Bogotá (Etapa 1; best y compare).
- variables_hedonicas: las 162 variables, con los predictores externos y de texto que pide el taller (Etapa 1; best y compare).
- texto_a_matriz: del texto crudo a la matriz documento término, con frecuencias y ley de Zipf (Etapa 1; best).
- texto_barrios_topicos: qué dice el texto (barrios que distinguen Chapinero y tópicos LDA) (Etapa 1; best).
- texto_apendice: LSA, LDA, TF-IDF, puntaje ridge fuera de pliegue y BETO (apéndice de best).
- dependencia_espacial: I de Moran, correlación del residuo, buffer y kriging (apéndice de best).
- error_por_familia: error de las siete especificaciones en las tres validaciones (apéndice de las dos).

Cada función crea la lámina en el Deck y devuelve (lámina, texto de las notas del orador). Solo leen archivos de
00_data/processed/laminas/, 02_outputs/figures/laminas/ y 02_outputs/tables/moran_global.csv (script 05), y cifras de la bitácora citadas en el código: no cambian datos ni modelos.
"""
import json
from pathlib import Path

import pandas as pd
from pptx.enum.shapes import MSO_SHAPE

from functions.ppt_lib import barras, chips, circulo, etiquetas_puntos, imagen, tabla, tarjeta, texto

from functions.entorno_ps3 import DIR_TAB
from functions.laminas_entorno import DATOS as D, IMG
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
