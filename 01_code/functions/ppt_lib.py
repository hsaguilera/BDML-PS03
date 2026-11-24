"""Utilidades para construir las presentaciones en PowerPoint con python-pptx (scripts 21 y 22).

Estilo: lámina de 13,33 x 7,5 pulgadas; Calibri; sin negrilla (preferencia del usuario: el énfasis va con color y
tamaño); tarjetas con el mismo código de color de las presentaciones Beamer; gráficos nativos y editables; notas del
orador en cada lámina.
"""
import math
import re

from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import (XL_AXIS_CROSSES, XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE,
                             XL_TICK_LABEL_POSITION)
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

C = dict(navy="1B2A41", azul="1F5FA8", naranja="D9822B", rojo="C0392B", gris="7F8C8D", gris_claro="C8CDD0",
         verde="2E8B57", tinta="1C2833", muted="5D6D7E", tinte="F3F5F8", blanco="FFFFFF", azul_claro="E8F0FA",
         rojo_claro="FBEAE8", verde_claro="E7F5EC", ambar="B9770E", ambar_claro="FDF3E1", morado="6C3483",
         morado_claro="F3EAF7", teal="0B6E8A", teal_claro="E4F1F6", linea="E3E7EB", azul_suave="7FA7D4", cielo="CADCFC")
TIPOS = {"definicion": ("azul", "azul_claro"), "clave": ("verde", "verde_claro"), "recordar": ("ambar", "ambar_claro"),
         "cuidado": ("rojo", "rojo_claro"), "aclaracion": ("morado", "morado_claro"), "verifica": ("teal", "teal_claro"),
         "pregunta": ("muted", "tinte"), "neutra": ("navy", "tinte")}
FUENTE = "Calibri"
W, H = 13.333, 7.5


def rgb(c):
    return RGBColor.from_string(C.get(c, c))


_NBSP = re.compile(r"(\d) (?=(%|millones|millón|m²|m\b|km|avisos|árboles|puntos|semillas))")
_NBSP_ANTES = re.compile(r"\b(Lecture|Lectures|cap\.|caps\.|percentil|percentiles|Etapa|Grupo|envío|Envío|Componente) (?=\d)")


def limpia(t):
    """Espacios que no se parten: entre un número y su unidad, entre una palabra y el número que la sigue
    (Lecture 14, cap. 8, percentil 99) y alrededor de un signo igual."""
    t = _NBSP.sub("\\1\u00a0", t)
    t = _NBSP_ANTES.sub("\\1\u00a0", t)
    return t.replace(" = ", "\u00a0=\u00a0").replace(" − ", "\u00a0−\u00a0")


def _run(p, texto, size=16, color="tinta", italic=False):
    r = p.add_run()
    r.text = limpia(texto)
    f = r.font
    f.name, f.size, f.bold, f.italic = FUENTE, Pt(size), False, italic
    f.color.rgb = rgb(color)
    return r


def _vinetas(p, nivel=0, color="azul"):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(0.28 + 0.25 * nivel))))
    pPr.set("indent", str(int(-Inches(0.22))))
    for tag in ("a:buNone", "a:buChar", "a:buAutoNum"):
        for e in pPr.findall(qn(tag)):
            pPr.remove(e)
    clr = pPr.makeelement(qn("a:buClr"), {})
    s = clr.makeelement(qn("a:srgbClr"), {"val": C.get(color, color)})
    clr.append(s)
    pPr.append(clr)
    pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "–"}))


def texto(slide, x, y, w, h, contenido, size=16, color="tinta", align="left", anchor="top", espacio=6, margen=0.05,
          nombre=None):
    """contenido: str, o lista de párrafos; cada párrafo es str, o lista de (texto, opciones) para colorear partes;
    un párrafo que empieza con '- ' lleva viñeta."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if nombre:
        tb.name = nombre
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(margen)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = {"top": MSO_ANCHOR.TOP, "middle": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM}[anchor]
    parrafos = contenido if isinstance(contenido, list) else [contenido]
    for i, par in enumerate(parrafos):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[align]
        p.space_after = Pt(espacio)
        vin = False
        if isinstance(par, str) and par.startswith("- "):
            par, vin = par[2:], True
        if isinstance(par, list) and par and isinstance(par[0], tuple) and str(par[0][0]).startswith("- "):
            par = [(par[0][0][2:], par[0][1])] + par[1:]
            vin = True
        if vin:
            p.space_after = Pt(espacio + 4)
        if isinstance(par, str):
            _run(p, par, size, color)
        else:
            for t, op in par:
                _run(p, t, op.get("size", size), op.get("color", color), op.get("italic", False))
    return tb


def forma(slide, tipo, x, y, w, h, relleno=None, linea=None, ancho_linea=0.75, radio=0.08, nombre=None):
    s = slide.shapes.add_shape(tipo, Inches(x), Inches(y), Inches(w), Inches(h))
    if nombre:
        s.name = nombre
    if tipo == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radio
    if relleno:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(relleno)
    else:
        s.fill.background()
    if linea:
        s.line.color.rgb = rgb(linea)
        s.line.width = Pt(ancho_linea)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def _largo(par):
    return len(par) if isinstance(par, str) else sum(len(t) for t, _ in par)


_MEDIDOR = None


def _medidor():
    """Calibri para medir el ancho real del texto; False si la fuente no está instalada."""
    global _MEDIDOR
    if _MEDIDOR is None:
        try:
            from PIL import ImageFont
            _MEDIDOR = ImageFont.truetype("calibri.ttf", 200)   # PIL la busca en las fuentes del sistema
        except Exception:  # noqa: BLE001
            _MEDIDOR = False
    return _MEDIDOR


def _texto_plano(par):
    return par if isinstance(par, str) else "".join(t for t, _ in par)


def _lineas(t, ancho, size):
    """Renglones que ocupa t en `ancho` pulgadas, con corte por palabras como PowerPoint."""
    t = limpia(t)
    f = _medidor()
    if not f:
        return max(1, math.ceil(len(t) / (0.94 * ancho / (0.47 * size / 72))))
    esc = size / 72 / 200  # pulgadas por píxel de la fuente medida a 200 px por em
    esp = f.getlength(" ") * esc
    n_lin, actual = 1, 0.0
    for palabra in t.split(" "):
        w_ = f.getlength(palabra) * esc
        if actual == 0:
            actual = w_
        elif actual + esp + w_ <= ancho:
            actual += esp + w_
        else:
            n_lin, actual = n_lin + 1, w_
    return n_lin


def _cabe(titulo, cuerpos, w, h, size, size_titulo):
    """Mide con Calibri si el texto cabe en la tarjeta: márgenes de 0,16 pulgadas a los lados, 3 % de holgura en el
    ancho, interlineado de 1,22 y 4 puntos después de cada párrafo."""
    ancho = (w - 0.32) * 0.97
    alto = sum(_lineas(_texto_plano(c), ancho, size) for c in cuerpos) * 1.22 * size / 72 + len(cuerpos) * 4 / 72
    if titulo:
        alto += _lineas(titulo, ancho, size_titulo) * 1.22 * size_titulo / 72 + 4 / 72
    return alto <= h - 0.24


def _mayor_que_cabe(titulo, cuerpos, w, h, maximo, size_titulo):
    for t_ in range(maximo, 11, -1):
        if _cabe(titulo, cuerpos, w, h, t_, max(size_titulo, t_ + 1)):
            return t_
    return 12


# Tarjetas con tamaño automático. Al guardar, todas las de una lámina quedan con el mismo tamaño: el mayor que cabe en
# todas (revision/informes/disenio_v3_opus.md, M3).
_AUTO = []


def tarjeta(slide, x, y, w, h, titulo, cuerpo, tipo="neutra", size="auto", size_titulo=15, nombre=None, maximo=17):
    """Tarjeta redondeada con fondo suave; el título va en el color del tipo (sin negrilla).
    Con size="auto" usa la letra más grande (entre 12 y `maximo` puntos) que cabe en la caja y, al guardar, la iguala con
    las demás tarjetas automáticas de la misma lámina."""
    c_titulo, c_fondo = TIPOS[tipo]
    cuerpos_ = cuerpo if isinstance(cuerpo, list) else [cuerpo]
    auto = size == "auto"
    titulo_base = size_titulo
    if auto:
        size = _mayor_que_cabe(titulo, cuerpos_, w, h, maximo, size_titulo)
    size_titulo = max(size_titulo, size + 1)
    s = forma(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, relleno=c_fondo, radio=0.06, nombre=nombre)
    if auto:
        _AUTO.append((slide, s, titulo, cuerpos_, w, h, maximo, titulo_base))
    tf = s.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.16)
    tf.margin_top = Inches(0.1)
    tf.margin_bottom = Inches(0.08)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    p.space_after = Pt(4)
    if titulo:
        _run(p, titulo, size_titulo, c_titulo)
    cuerpos = cuerpo if isinstance(cuerpo, list) else [cuerpo]
    for i, par in enumerate(cuerpos):
        q = p if (not titulo and i == 0) else tf.add_paragraph()
        q.alignment = PP_ALIGN.LEFT
        q.space_after = Pt(4)
        vin = isinstance(par, str) and par.startswith("- ")
        if vin:
            par = par[2:]
            q.space_after = Pt(8)
        if isinstance(par, str):
            _run(q, par, size, "tinta")
        else:
            for t, op in par:
                _run(q, t, op.get("size", size), op.get("color", "tinta"))
    return s


def _igualar_tarjetas():
    """Pone un solo tamaño de letra a las tarjetas automáticas de cada lámina: el mayor que cabe en todas."""
    grupos = {}
    for reg in _AUTO:
        grupos.setdefault(id(reg[0]), []).append(reg)
    for regs in grupos.values():
        t = min(_mayor_que_cabe(tit, cu, w, h, mx, st) for _, _, tit, cu, w, h, mx, st in regs)
        for _, shp, tit, _, _, _, _, st in regs:
            for k, p in enumerate(shp.text_frame.paragraphs):
                for r in p.runs:
                    r.font.size = Pt(max(st, t + 1) if (tit and k == 0) else t)


def cifra(slide, x, y, w, valor, etiqueta, color="azul", size=40, size_etq=13, alto_etq=0.6):
    texto(slide, x, y, w, 0.75, valor, size=size, color=color)
    texto(slide, x, y + 0.72, w, alto_etq, etiqueta, size=size_etq, color="muted")


def circulo(slide, x, y, d, numero, relleno="navy", color="blanco", size=16):
    s = forma(slide, MSO_SHAPE.OVAL, x, y, d, d, relleno=relleno)
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    _run(p, str(numero), size, color)
    return s


def flecha(slide, x1, y1, x2, y2, color="gris", ancho=1.25):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(ancho)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    return c


def linea(slide, x1, y1, x2, y2, color="gris_claro", ancho=1.0):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(ancho)
    return c


def imagen(slide, ruta, x, y, w=None, h=None, nombre=None, alt=None):
    pic = slide.shapes.add_picture(str(ruta), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)
    if nombre:
        pic.name = nombre
    if alt:
        pic._element.nvPicPr.cNvPr.set("descr", alt)
    return pic


def tabla(slide, x, y, w, filas, anchos, size=13, alto_fila=0.36, encabezado=True, colores_col=None):
    nr, nc = len(filas), len(filas[0])
    gf = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(alto_fila * nr))
    t = gf.table
    t.first_row = encabezado
    t.horz_banding = False
    for j, a in enumerate(anchos):
        t.columns[j].width = Inches(a)
    for i in range(nr):
        t.rows[i].height = Inches(alto_fila)
        for j in range(nc):
            cel = t.cell(i, j)
            cel.margin_left = cel.margin_right = Inches(0.08)
            cel.margin_top = cel.margin_bottom = Inches(0.03)
            cel.vertical_anchor = MSO_ANCHOR.MIDDLE
            cel.fill.solid()
            cel.fill.fore_color.rgb = rgb("E8EEF5" if (encabezado and i == 0) else ("FFFFFF" if i % 2 else "F7F9FB"))
            tf = cel.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            for r in list(p.runs):
                r._r.getparent().remove(r._r)
            col = "navy" if (encabezado and i == 0) else "tinta"
            if colores_col and i > 0 and colores_col.get(j):
                col = colores_col[j]
            val = filas[i][j]
            if isinstance(val, tuple):
                val, col = val
            _run(p, str(val), size, col)
    return gf


# ------------------------------------------------------------------------------------------------------------------
# Gráficos nativos
# ------------------------------------------------------------------------------------------------------------------
def _estilo_grafico(ch, size=12):
    ch.font.name = FUENTE
    ch.font.size = Pt(size)
    ch.font.bold = False
    ch.font.color.rgb = rgb("muted")


def _ejes(ch, valor_visible=True, fmt=None, minimo=None, maximo=None, unidad=None, grid=True, titulo_valor=None,
          titulo_cat=None, size=12, invertir_cat=False):
    va, ca = ch.value_axis, ch.category_axis
    va.visible = valor_visible
    va.has_major_gridlines = grid
    if grid:
        va.major_gridlines.format.line.color.rgb = rgb("linea")
        va.major_gridlines.format.line.width = Pt(0.75)
    va.format.line.fill.background()
    if minimo is not None:
        va.minimum_scale = minimo
    if maximo is not None:
        va.maximum_scale = maximo
    if unidad is not None:
        va.major_unit = unidad
    if fmt:
        va.tick_labels.number_format = fmt
        va.tick_labels.number_format_is_linked = False
    va.tick_labels.font.size = Pt(size)
    ca.tick_labels.font.size = Pt(size)
    ca.format.line.color.rgb = rgb("gris_claro")
    ca.has_major_gridlines = False
    if invertir_cat:
        ca.reverse_order = True
    for eje, tit in ((va, titulo_valor), (ca, titulo_cat)):
        if tit:
            eje.has_title = True
            eje.axis_title.text_frame.text = tit
            r = eje.axis_title.text_frame.paragraphs[0].runs[0]
            r.font.size, r.font.bold, r.font.name = Pt(size), False, FUENTE
            r.font.color.rgb = rgb("muted")


def barras(slide, x, y, w, h, categorias, series, horizontal=True, fmt="0.0", etiquetas=True, fmt_etiqueta=None,
           minimo=None, maximo=None, unidad=None, leyenda=False, gap=55, overlap=None, size=12, valor_visible=False,
           titulo_valor=None, titulo_cat=None, apiladas=False, pos_etiqueta="fuera", color_etiqueta="tinta",
           etiquetas_abajo=False, grid=True):
    """series: lista de (nombre, valores, color o lista de colores por punto, opciones)."""
    cd = CategoryChartData()
    cd.categories = categorias
    for s in series:
        cd.add_series(s[0], s[1], number_format=fmt)
    if apiladas:
        tipo = XL_CHART_TYPE.BAR_STACKED if horizontal else XL_CHART_TYPE.COLUMN_STACKED
    else:
        tipo = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    gf = slide.shapes.add_chart(tipo, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    _estilo_grafico(ch, size)
    ch.has_title = False
    ch.has_legend = leyenda
    if leyenda:
        ch.legend.position = XL_LEGEND_POSITION.TOP
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(size)
    plot = ch.plots[0]
    plot.gap_width = gap
    if overlap is not None:
        plot.overlap = overlap
    for ser, s in zip(plot.series, series):
        col = s[2]
        op = s[3] if len(s) > 3 else {}
        if col is None:
            ser.format.fill.background()
        elif isinstance(col, list):
            ser.format.fill.solid()
            ser.format.fill.fore_color.rgb = rgb(col[0])
            for i, c_ in enumerate(col):
                pt = ser.points[i]
                pt.format.fill.solid()
                pt.format.fill.fore_color.rgb = rgb(c_)
        else:
            ser.format.fill.solid()
            ser.format.fill.fore_color.rgb = rgb(col)
        ser.invert_if_negative = False
        if etiquetas and op.get("etiquetas", True) and col is not None:
            ser.data_labels.show_value = True
            dl = ser.data_labels
            dl.number_format = op.get("fmt_etiqueta", fmt_etiqueta or fmt)
            dl.number_format_is_linked = False
            dl.font.size = Pt(size)
            dl.font.color.rgb = rgb(op.get("color_etiqueta", color_etiqueta))
            dl.font.bold = False
            if not apiladas:
                dl.position = {"fuera": XL_LABEL_POSITION.OUTSIDE_END, "dentro": XL_LABEL_POSITION.INSIDE_END}[pos_etiqueta]
            else:
                dl.position = XL_LABEL_POSITION.CENTER
    _ejes(ch, valor_visible=valor_visible, fmt=fmt, minimo=minimo, maximo=maximo, unidad=unidad, titulo_valor=titulo_valor,
          titulo_cat=titulo_cat, size=size, invertir_cat=horizontal, grid=grid)
    if etiquetas_abajo:
        ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    return ch


def lineas_xy(slide, x, y, w, h, series, size=12, leyenda=True, minx=None, maxx=None, miny=None, maxy=None, fmt_x="0",
              fmt_y="0", titulo_x=None, titulo_y=None, unidad_x=None, unidad_y=None, pos_leyenda="arriba", x_abajo=True):
    """series: lista de dict(nombre, x, y, color, ancho, marcador (None/'circulo'), tam, linea (True/False), guiones)."""
    cd = XyChartData()
    for s in series:
        ss = cd.add_series(" " + s["nombre"])  # el espacio separa el nombre del trazo en la leyenda
        for a, b in zip(s["x"], s["y"]):
            ss.add_data_point(a, b)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    _estilo_grafico(ch, size)
    ch.has_title = False
    ch.has_legend = leyenda
    if leyenda:
        ch.legend.position = {"arriba": XL_LEGEND_POSITION.TOP, "abajo": XL_LEGEND_POSITION.BOTTOM,
                              "derecha": XL_LEGEND_POSITION.RIGHT}[pos_leyenda]
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(size)
    for ser, s in zip(ch.plots[0].series, series):
        ser.smooth = False
        if s.get("linea", True):
            ser.format.line.color.rgb = rgb(s["color"])
            ser.format.line.width = Pt(s.get("ancho", 2.0))
            if s.get("guiones"):
                ser.format.line.dash_style = MSO_LINE.DASH
        else:
            ser.format.line.fill.background()
        if s.get("marcador"):
            ser.marker.style = XL_MARKER_STYLE.CIRCLE
            ser.marker.size = s.get("tam", 6)
            ser.marker.format.fill.solid()
            ser.marker.format.fill.fore_color.rgb = rgb(s["color"])
            ser.marker.format.line.fill.background()
        else:
            ser.marker.style = XL_MARKER_STYLE.NONE
    va, ca = ch.value_axis, ch.category_axis
    for eje, mn, mx, fmt, tit, un in ((ca, minx, maxx, fmt_x, titulo_x, unidad_x), (va, miny, maxy, fmt_y, titulo_y, unidad_y)):
        if mn is not None:
            eje.minimum_scale = mn
        if mx is not None:
            eje.maximum_scale = mx
        if un is not None:
            eje.major_unit = un
        eje.tick_labels.number_format = fmt
        eje.tick_labels.number_format_is_linked = False
        eje.tick_labels.font.size = Pt(size)
        eje.has_major_gridlines = eje is va
        if eje is va:
            eje.major_gridlines.format.line.color.rgb = rgb("linea")
            eje.major_gridlines.format.line.width = Pt(0.75)
            eje.format.line.fill.background()
        else:
            eje.format.line.color.rgb = rgb("gris_claro")
        if tit:
            eje.has_title = True
            eje.axis_title.text_frame.text = tit
            r = eje.axis_title.text_frame.paragraphs[0].runs[0]
            r.font.size, r.font.bold, r.font.name = Pt(size), False, FUENTE
            r.font.color.rgb = rgb("muted")
    if x_abajo:
        ch.value_axis.crosses = XL_AXIS_CROSSES.MINIMUM
    return ch


# ------------------------------------------------------------------------------------------------------------------
# Presentación
# ------------------------------------------------------------------------------------------------------------------
class Deck:
    def __init__(self, titulo_corto):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(W), Inches(H)
        self.blank = self.prs.slide_layouts[6]
        self.titulo_corto = titulo_corto
        self.pies = []
        _AUTO.clear()

    def nueva(self, titulo=None, oscura=False, seccion="", subtitulo=None, etiqueta=None, size_titulo=26):
        s = self.prs.slides.add_slide(self.blank)
        if oscura:
            bg = s.background.fill
            bg.solid()
            bg.fore_color.rgb = rgb("navy")
        if etiqueta:
            texto(s, 0.55, 0.22, 9.0, 0.34, etiqueta, size=13, color="cielo" if oscura else "azul", nombre="Etiqueta")
        if titulo:
            texto(s, 0.55, 0.52 if etiqueta else 0.32, 12.2, 0.62, titulo, size=size_titulo,
                  color="blanco" if oscura else "navy", anchor="top", nombre="Título")
        if subtitulo:
            texto(s, 0.55, 1.08, 12.2, 0.4, subtitulo, size=15, color="cielo" if oscura else "muted", nombre="Subtítulo")
        texto(s, 0.55, 7.05, 9.0, 0.3, f"Equipo 02. {self.titulo_corto}" + (f". {seccion}" if seccion else ""), size=12,
              color="cielo" if oscura else "muted", nombre="Pie")
        num = texto(s, 11.3, 7.05, 1.5, 0.3, "", size=12, color="cielo" if oscura else "muted", align="right", nombre="Número")
        self.pies.append((num, oscura))
        return s

    def notas(self, slide, texto_notas):
        slide.notes_slide.notes_text_frame.text = texto_notas

    def guardar(self, ruta):
        n = len(self.prs.slides)
        for i, (num, oscura) in enumerate(self.pies, 1):
            p = num.text_frame.paragraphs[0]
            _run(p, f"{i} de {n}", 12, "cielo" if oscura else "muted")
        _igualar_tarjetas()
        cp = self.prs.core_properties
        cp.title = f"{self.titulo_corto}. Equipo 02"
        cp.subject = "Problem Set 3, Big Data and Machine Learning para Economía Aplicada (MECA 4107)"
        cp.author = cp.last_modified_by = "Equipo 02"
        cp.keywords = cp.comments = ""
        self.prs.save(str(ruta))
        print("guardado", ruta, n, "láminas")


# ------------------------------------------------------------------------------------------------------------------
# Piezas gráficas con formas nativas
# ------------------------------------------------------------------------------------------------------------------
def etiquetas_puntos(ch, textos_por_serie, size=12, color="tinta", posicion=None):
    """Etiquetas de datos con texto propio (coma decimal sin depender de la configuración regional)."""
    for ser, textos in zip(ch.plots[0].series, textos_por_serie):
        if textos is None:
            continue
        for i, t in enumerate(textos):
            if t is None:
                continue
            dl = ser.points[i].data_label
            tf = dl.text_frame
            tf.text = t
            for par in tf.paragraphs:
                for r in par.runs:
                    r.font.size, r.font.bold, r.font.name = Pt(size), False, FUENTE
                    r.font.color.rgb = rgb(color)
            if posicion is not None:
                dl.position = posicion


def chips(slide, x, y, w, palabras, relleno, color="tinta", size=13, alto=0.36, sep=0.1, pad=0.14):
    """Palabras en cápsulas que fluyen en filas; devuelve la y final."""
    cx, cy = x, y
    for pal in palabras:
        ancho = pad * 2 + 0.083 * len(pal) * size / 13
        if cx + ancho > x + w:
            cx, cy = x, cy + alto + sep
        s = forma(slide, MSO_SHAPE.ROUNDED_RECTANGLE, cx, cy, ancho, alto, relleno=relleno, radio=0.5)
        tf = s.text_frame
        tf.margin_left = tf.margin_right = Inches(0.04)
        tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.word_wrap = False
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        _run(p, pal, size, color)
        cx += ancho + sep
    return cy + alto


def waffle(slide, x, y, lado, colores, gap=0.06, columnas=10):
    """Cuadrícula de 100 cuadros; colores: lista de 100 claves de color, por filas."""
    for i, c in enumerate(colores):
        f, k = divmod(i, columnas)
        forma(slide, MSO_SHAPE.RECTANGLE, x + k * (lado + gap), y + f * (lado + gap), lado, lado, relleno=c)


def nodo(slide, x, y, w, h, lineas, relleno="tinte", borde="gris_claro", sizes=(13, 11), colores=("tinta", "muted"),
         margen=0.05):
    s = forma(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, relleno=relleno, linea=borde, radio=0.12)
    tf = s.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(margen)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, ln in enumerate(lineas):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        p.space_after = Pt(0)
        _run(p, ln, sizes[min(i, len(sizes) - 1)], colores[min(i, len(colores) - 1)])
    return s



def ocultar_leyenda(ch, indices):
    """Quita de la leyenda las series indicadas (por ejemplo, líneas de referencia)."""
    leg = ch.legend._element
    pos = leg.find(qn("c:legendPos"))
    for i in sorted(indices, reverse=True):
        le = leg.makeelement(qn("c:legendEntry"), {})
        le.append(le.makeelement(qn("c:idx"), {"val": str(i)}))
        le.append(le.makeelement(qn("c:delete"), {"val": "1"}))
        pos.addnext(le)
