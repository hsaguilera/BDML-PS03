"""Piezas comunes de los constructores finales (21_construir_presentacion_best.py y 22_construir_presentacion_compare.py).

- DeckConClaves: el Deck de ppt_lib que, además, le pone a cada lámina una clave estable con el nombre de la función que la
  creó (por ejemplo e1.problema, e2.zillow, cmp.criterios). Si una función crea varias láminas, llevan _1, _2, ...
- seleccionar: deja solo las láminas de la lista SELECCION y en ese orden (borra las demás antes de guardar), de modo que
  la numeración «i de N» se calcula sobre la selección.
- portada: la portada de cada presentación, tal como quedó editada en PowerPoint (sin fecha).
- El PDF lo hace el script 23 (opcional, Windows con PowerPoint).

Prefijos de las claves: portada (este módulo), e1. (laminas_etapa1), e2. (laminas_etapas2a4), cmp. (laminas_compare_nuevo).
"""
import inspect
from collections import Counter

from functions.laminas_comunes import IMG
from functions.ppt_lib import Deck, imagen, linea, texto

PREFIJO = {"laminas_etapa1": "e1.", "laminas_etapas2a4": "e2.", "laminas_compare_nuevo": "cmp.",
           "construir_final_comun": ""}
PUBLICAS = {"etapa1", "etapas2a4", "compare_cuerpo", "compare_portada_y_resumen"}
EQUIPO = ("Equipo 02: Jhon Sebastián Vela Salcedo, Harold Santiago Aguilera Mulfo y César Augusto Vallejo Quiroga")
PIE_MAPA = "Gris: avisos de entrenamiento. Naranja: avisos de prueba (94 % en la localidad de Chapinero)."


class DeckConClaves(Deck):
    """Deck que anota, para cada lámina, la función que la creó."""

    def __init__(self, titulo_corto):
        super().__init__(titulo_corto)
        self.funciones = []

    def nueva(self, *args, **kwargs):
        s = super().nueva(*args, **kwargs)
        self.funciones.append(self._funcion_que_llama())
        return s

    @staticmethod
    def _funcion_que_llama():
        """La función de lámina más externa dentro de un módulo de láminas (se salta ayudantes como _nueva)."""
        clave = None
        for fr in inspect.stack()[2:]:
            mod = fr.frame.f_globals.get("__name__", "").rsplit(".", 1)[-1]  # también functions.laminas_* en el repositorio
            if mod not in PREFIJO or fr.function in PUBLICAS or fr.function == "<module>":
                break
            clave = PREFIJO[mod] + fr.function.lstrip("_")
        return clave or "sin_clave"

    def claves(self):
        """Claves únicas: si una función creó varias láminas, cada una lleva _1, _2, ..."""
        total, vistos, out = Counter(self.funciones), Counter(), []
        for f in self.funciones:
            vistos[f] += 1
            out.append(f"{f}_{vistos[f]}" if total[f] > 1 else f)
        return out


def catalogo(deck):
    """Clave y título de cada lámina creada, en el orden de creación."""
    filas = []
    for clave, s in zip(deck.claves(), deck.prs.slides):
        tit = next((sh.text_frame.text for sh in s.shapes if sh.name == "Título"), None)
        if tit is None:
            tit = next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.name.startswith("TextBox")), "")
        filas.append((clave, tit.replace("\n", " ")))
    return filas


def seleccionar(deck, seleccion):
    """Deja en el Deck solo las láminas de `seleccion` (lista de claves), en ese orden."""
    claves = deck.claves()
    faltan = [k for k in seleccion if k not in claves]
    if faltan:
        raise SystemExit(f"Claves que no existen: {faltan}. Corra con --catalogo para ver las disponibles.")
    repetidas = [k for k, c in Counter(seleccion).items() if c > 1]
    if repetidas:
        raise SystemExit(f"Claves repetidas en la selección: {repetidas}")
    lista = deck.prs.slides._sldIdLst
    ids = list(lista)
    por_clave = dict(zip(claves, ids))
    pies = dict(zip(claves, deck.pies))
    for el in ids:
        lista.remove(el)
    for k in seleccion:
        lista.append(por_clave[k])
    for k in claves:
        if k not in seleccion:
            deck.prs.part.drop_rel(por_clave[k].rId)
    deck.pies = [pies[k] for k in seleccion]
    return [k for k in claves if k not in seleccion]


def ajustes_de_control(deck):
    """Lo que las copias de control tienen distinto de lo que generan los módulos (se aplica antes de seleccionar).

    - Fuente 4 (e1.externas_4): la tarjeta de Google Earth Engine está a 12 puntos en las presentaciones (el módulo la
      calcula a 14 con su tamaño automático). Se fija en 12 y se saca del ajuste automático de ppt_lib.
    """
    from pptx.util import Pt

    from functions import ppt_lib
    for clave, s in zip(deck.claves(), deck.prs.slides):
        if clave != "e1.externas_4":
            continue
        for sh in s.shapes:
            if sh.has_text_frame and sh.text_frame.text.startswith("Google Earth Engine es la plataforma"):
                ppt_lib._AUTO[:] = [r for r in ppt_lib._AUTO if r[1]._element is not sh._element]
                for p in sh.text_frame.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(12)


def portada(deck, titulo, subtitulo, size_subtitulo, alto_subtitulo, notas):
    """Portada oscura con el mapa de entrenamiento y prueba, como quedó editada en PowerPoint (sin fecha)."""
    s = deck.nueva(oscura=True)
    texto(s, 0.7, 0.9, 6.3, 0.4, "Problem Set 3. Making Money with ML?", size=16, color="cielo")
    texto(s, 0.7, 1.45, 6.4, 1.96, titulo, size=38, color="blanco")
    texto(s, 0.7, 3.576, 6.3, alto_subtitulo, subtitulo, size=size_subtitulo, color="cielo")
    linea(s, 0.7, 4.95, 3.2, 4.95, color="naranja", ancho=2.5)
    texto(s, 0.7, 5.15, 6.3, 0.545, [EQUIPO], size=15, color="blanco", espacio=8)
    imagen(s, IMG / "portada_mapa.png", 7.45, 0.45, h=5.95,
           alt="Mapa de Bogotá: avisos de entrenamiento en gris y de prueba en naranja, con el borde de Chapinero")
    texto(s, 7.45, 6.45, 5.4, 0.5, PIE_MAPA, size=12, color="cielo")
    deck.notas(s, notas)
    return s, notas

