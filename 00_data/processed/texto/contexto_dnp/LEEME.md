# Listas de ConTexto (DNP) — datos brutos

Fuente: librería **ConTexto** de la Unidad de Científicos de Datos del Departamento Nacional de Planeación (DNP),
https://github.com/ucd-dnp/ConTexto (carpeta `contexto/data/listas_stopwords/`), licencia MIT (ver `LICENSE`).
Citada en la Lecture 15 del curso ("Ej: Contexto DNP", diapositiva de lematización).

No se instala la librería: no es compatible con Python 3.13 (`from collections import Iterable`, gensim 4.0.1 solo para
cp36–cp38, googletrans 2.4.0). Se usan sus listas con funciones propias.

| Archivo | Contenido |
|---|---|
| `lista_sw_es.txt` | 350 stopwords en español (latin-1; incluye *no*, *sin*, *ni*, que deben conservarse) |
| `lista_nombres_*.txt` | nombres propios (ambos / hombres / mujeres) |
| `lista_apellidos.txt` | apellidos |
| `lista_municipios_col.txt` | municipios de Colombia |
| `lista_departamentos_col.txt` | departamentos de Colombia |

**Advertencia:** varias entradas coinciden con barrios y vocabulario inmobiliario (chico, cabrera, rosales, retiro,
andino, castillo, prado, bosque, sala, casa, linda…). No usarlas para borrar palabras a ciegas.
