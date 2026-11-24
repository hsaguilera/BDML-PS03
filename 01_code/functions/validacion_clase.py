"""Validación cruzada espacial tal como en la lectura "Validación cruzada espacial" (Rojas y Sarmiento-Barbieri, 2026) y la
Lecture 14 del curso (lo usa flujo_final.py). Traducción a Python de `rsample::vfold_cv`, `spatialsample::spatial_block_cv`,
`spatial_block_cv(..., buffer = 300)` y `spatial_leave_location_out_cv(group = ..., v = NULL)`.

Esquemas externos (v = 5, semilla 2026):
- "aleatoria":       5 pliegues al azar.
- "bloques":         grilla cuadrada de 2.000 m sobre la caja de los datos; los bloques con datos se asignan al azar a los
                     5 pliegues (como sample(rep(1:v, length.out = n_bloques))); un pliegue puede tener bloques no contiguos.
- "bloques_buffer":  los mismos bloques; se retiran del entrenamiento los avisos a <= 300 m de algún aviso de validación
                     (el buffer no entrena ni valida). La lectura usa 300 m.
- "localidades":     dejar una localidad por fuera (equivalente a las comunas de Cali en la lectura).
- "clusters":        `spatialsample::spatial_clustering_cv` (decisión del usuario: en vez de un dato administrativo, grupos
                     construidos por un algoritmo): k-means sobre las coordenadas en metros, v = 10 (valor por defecto del
                     paquete); cada clúster es un pliegue completo.

Partición interna para el early stopping (función `fun_fit_net` de la lectura): sigue la lógica del esquema externo.
- aleatoria → 80 % ajuste / 20 % parada (initial_split, prop = 0.8).
- bloques → spatial_block_cv(v = 5) sobre el entrenamiento del pliegue y se usa solo la primera partición.
- bloques_buffer → lo mismo, con el buffer de 300 m.
- localidades → una localidad elegida al azar entre las del entrenamiento del pliegue.
- clusters → k-means (v = 10) sobre el entrenamiento del pliegue y se usa la primera partición.

Todas las funciones devuelven índices locales (posiciones dentro de los arreglos que reciben).
"""
import numpy as np
from scipy.spatial import cKDTree

TAM_BLOQUE, BUFFER, V, SEMILLA, V_CLUSTERS = 2000.0, 300.0, 5, 2026, 10


def bloques_cv(x, y, v=V, tam=TAM_BLOQUE, buffer=0.0, semilla=SEMILLA):
    x0, y0 = x.min(), y.min()
    celda = np.floor((x - x0) / tam).astype(np.int64) * 1_000_003 + np.floor((y - y0) / tam).astype(np.int64)
    unicas = np.unique(celda)
    asignacion = np.random.default_rng(semilla).permutation(np.resize(np.arange(v), len(unicas)))
    fold = asignacion[np.searchsorted(unicas, celda)]
    pliegues = []
    for k in range(v):
        va, tr = np.where(fold == k)[0], np.where(fold != k)[0]
        if buffer > 0 and len(va):
            d = cKDTree(np.c_[x[va], y[va]]).query(np.c_[x[tr], y[tr]], k=1, distance_upper_bound=buffer)[0]
            tr = tr[~(d <= buffer)]
        pliegues.append((tr, va))
    return pliegues


def aleatoria_cv(n, v=V, semilla=SEMILLA):
    fold = np.random.default_rng(semilla).permutation(np.resize(np.arange(v), n))
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(v)]


def clusters_cv(x, y, v=V_CLUSTERS, semilla=SEMILLA):
    from sklearn.cluster import KMeans
    etiqueta = KMeans(n_clusters=v, n_init=10, random_state=semilla).fit_predict(np.c_[x, y])
    return [(np.where(etiqueta != k)[0], np.where(etiqueta == k)[0]) for k in range(v)]


def localidades_cv(grupo):
    return [(np.where(grupo != g_)[0], np.where(grupo == g_)[0]) for g_ in np.unique(grupo)]


def pliegues_externos(esquema, x, y, grupo):
    if esquema == "aleatoria":
        return aleatoria_cv(len(x))
    if esquema == "bloques":
        return bloques_cv(x, y)
    if esquema == "bloques_buffer":
        return bloques_cv(x, y, buffer=BUFFER)
    if esquema == "localidades":
        return localidades_cv(grupo)
    if esquema == "clusters":
        return clusters_cv(x, y)
    raise ValueError(esquema)


def particion_interna(esquema, x, y, grupo, semilla):
    """(ajuste, parada) dentro del entrenamiento de un pliegue, con la misma lógica del esquema externo."""
    rng = np.random.default_rng(semilla)
    n = len(x)
    if esquema == "aleatoria":
        perm = rng.permutation(n)
        corte = int(round(0.8 * n))
        return np.sort(perm[:corte]), np.sort(perm[corte:])
    if esquema in ("bloques", "bloques_buffer"):
        return bloques_cv(x, y, buffer=BUFFER if esquema == "bloques_buffer" else 0.0, semilla=semilla)[0]
    if esquema == "localidades":
        elegida = rng.choice(np.unique(grupo))
        return np.where(grupo != elegida)[0], np.where(grupo == elegida)[0]
    if esquema == "clusters":
        return clusters_cv(x, y, semilla=semilla)[0]
    raise ValueError(esquema)
