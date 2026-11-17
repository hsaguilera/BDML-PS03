# Problem Set 3: Making Money with ML?

Big Data y Machine Learning para Economía Aplicada (MECA 4107, 2026-20) · Universidad de los Andes ·
**Equipo 02**: Jhon Sebastián Vela Salcedo, Harold Santiago Aguilera Mulfo y Cesar Augusto Vallejo Quiroga.

## Objetivo

Una empresa que compra y vende vivienda quiere comprar la mayor cantidad posible de inmuebles en
Chapinero (Bogotá) gastando lo menos posible. Este repositorio construye modelos que predicen el precio de
venta de cada inmueble con datos de Properati: se entrena con anuncios del resto de Bogotá y se predice en
Chapinero, una zona que no aparece en el entrenamiento. Como sobrestimar un precio destruye capital y
subestimarlo solo hace perder un negocio, además del error medio (MAE, la métrica de Kaggle) se analiza
hacia qué lado se equivoca el modelo.

> **Pregunta.** ¿Cuál es el mejor modelo para predecir precios de vivienda en Chapinero y se puede confiar
> en él para comprar?
>
> **Respuesta.** *Pendiente: se escribe cuando existan los resultados.*

## Resultados

*Pendiente.* Esta tabla se llena con cifras tomadas de `02_outputs/tables/`.

| Familia | Mejor especificación | MAE validación cruzada estándar | MAE validación cruzada espacial | MAE Kaggle público |
|---|---|---|---|---|
| Regresión lineal | | | | |
| Elastic Net | | | | |
| CART | | | | |
| Random Forest | | | | |
| Boosting | | | | |
| Red neuronal | | | | |

## Cómo reproducir

1. Clone el repositorio y descargue los datos de Kaggle en `00_data/` (ver [`00_data/README.md`](00_data/README.md)).
2. Abra R en la raíz del repositorio y corra una sola instrucción:

```r
source("01_code/00_rundirectory.R")
```

El script maestro carga el entorno común (`00_setup.R`: semilla, rutas y paquetes), corre en orden numérico
todos los scripts `NN_*.R` y `NN_*.py` de `01_code/` y guarda las versiones de R, Python y sus paquetes en
`02_outputs/session_info.txt`. Para correr solo un tramo: `Sys.setenv(PS3_DESDE = "5", PS3_HASTA = "8")`
antes del `source`.

## Estructura

```
BDML-PS03/
├── README.md
├── requirements_R.txt        versiones de los paquetes de R
├── requirements.txt          versiones de los paquetes de Python
├── 00_data/                  datos de Kaggle (no se versionan) y processed/ (la genera el pipeline)
├── 01_code/
│   ├── 00_rundirectory.R     script maestro: reproduce todo
│   ├── 00_setup.R            entorno común de los scripts de R
│   ├── NN_*.R, NN_*.py       scripts numerados, uno por tarea
│   └── functions/            funciones reutilizables (setup_python.py: entorno común de Python)
├── 02_outputs/
│   ├── figures/              figuras (.png)
│   ├── tables/               tablas (.tex, .csv)
│   └── predictions/          envíos a Kaggle (.csv)
└── 03_slides/                best_equipo_02 y compare_equipo_02 (.tex y .pdf), preamble.tex, numbers.tex, tables/
```

## Scripts y salidas

Cada fila dice qué hace el script, qué archivos produce, **qué contiene cada archivo** y en qué diapositiva
se usa (`best` o `compare`, con número de página).

| Script | Qué hace | Salida | Qué contiene la salida | Diapositiva |
|---|---|---|---|---|
| `00_rundirectory.R` | Corre todo el pipeline en orden | `02_outputs/session_info.txt` | Versiones de R, Python y paquetes | — |
| `00_setup.R` | Semilla, rutas, paquetes y funciones | — | — | — |
| *(pendiente)* | | | | |

## Envíos a Kaggle

Mínimo 20 envíos y al menos uno por cada una de las seis familias. El nombre de cada `.csv` de
`02_outputs/predictions/` indica modelo e hiperparámetros.

| # | Fecha | Familia | Archivo de predicción | Script | MAE público |
|---|---|---|---|---|---|
| *(pendiente)* | | | | | |

## Convenciones

* **Numeración.** Dos dígitos consecutivos (`01_`, `02_`, ...), sin números repetidos ni saltos, compartida
  por R y Python. El script maestro se detiene si la numeración no cumple.
* **Nombres.** `snake_case` y que digan qué hace el script o qué contiene la salida. Los objetos y los
  argumentos de las funciones también llevan nombres descriptivos (no letras sueltas).
* **Cada script corre solo.** Su primera línea de código carga el entorno común:
  * R: `if (!exists("DIR_PROC")) source(file.path(here::here(), "01_code", "00_setup.R"), encoding = "UTF-8")`
  * Python: `from functions.setup_python import *`
* **Sin rutas absolutas.** Todo se escribe relativo a la raíz (carpeta con el archivo `.here`).
* **Cifras de las diapositivas.** Salen de `03_slides/numbers.tex`, que genera un script; no se escriben a mano.

## Software

* R 4.6.1 y los paquetes de `requirements_R.txt`.
* Python 3 y los paquetes de `requirements.txt`.
* Las versiones exactas de la última corrida quedan en `02_outputs/session_info.txt`.
