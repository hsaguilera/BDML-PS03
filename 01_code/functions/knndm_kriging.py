"""Componente espacial explícito: pliegues kNNDM y regression kriging de residuos (scripts 08 y 09).

1. kNNDM (k-fold Nearest Neighbour Distance Matching; Milà et al. 2022, Methods Ecol. Evol.; Linnenbrink et al. 2024):
   se buscan pliegues cuya distribución de distancias "aviso de validación → aviso de entrenamiento más cercano" (G*_j) se
   parezca a la de la tarea real "aviso de prueba → aviso de entrenamiento más cercano" (G_ij). Para cada q de una rejilla se
   agrupan las coordenadas de entrenamiento con k-means en q grupos, los grupos se reparten al azar en k pliegues de tamaño
   parecido y se mide la distancia de Wasserstein W(G*_j, G_ij); se elige el q con menor W (q = n equivale a la CV aleatoria).

2. Regression kriging (Hengl, Heuvelink y Rossiter 2007): ŷ_RK(s) = ŷ_ML(s) + r̂(s), donde r̂ es el kriging simple de los
   residuos r = y - ŷ_ML de los avisos de entrenamiento cercanos.
   Variograma exponencial con pepita: γ(h) = c0 + c1 (1 - exp(-h/a)), ajustado por mínimos cuadrados ponderados al
   semivariograma empírico γ̂(h) = ½ E[(r_i - r_j)²] en clases de distancia.
   Kriging simple local con los K vecinos más cercanos: (C + c0 I) λ = c, con C_ij = c1 exp(-h_ij/a), c_i = c1 exp(-h_i0/a),
   r̂(s0) = m + λ'(r - m), m = media de los residuos. Si los vecinos están lejos (h >> a), r̂ → m.
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance


# ----------------------------------------------------------------------------------------------------------------------
# kNNDM
# ----------------------------------------------------------------------------------------------------------------------
def _repartir(etiqueta, k, rng):
    """Reparte grupos en k pliegues de tamaño parecido (orden aleatorio de grupos, corte por tamaño acumulado)."""
    ids, n = np.unique(etiqueta, return_counts=True)
    orden = rng.permutation(len(ids))
    acum = np.cumsum(n[orden])
    fold_grupo = np.minimum((acum - 1) * k // acum[-1], k - 1)
    mapa = dict(zip(ids[orden], fold_grupo))
    return np.array([mapa[e] for e in etiqueta])


def distancias_cv(xy, fold):
    d = np.empty(len(xy))
    for f in np.unique(fold):
        m = fold == f
        d[m] = cKDTree(xy[~m]).query(xy[m], k=1)[0]
    return d


def knndm(xy_tr, xy_pred, k=5, rejilla=(10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400), semilla=2026):
    from sklearn.cluster import MiniBatchKMeans
    rng = np.random.default_rng(semilla)
    g_ij = cKDTree(xy_tr).query(xy_pred, k=1)[0]
    candidatos = []
    for q in list(rejilla) + ["aleatoria"]:
        if q == "aleatoria":
            etiqueta = np.arange(len(xy_tr))
        else:
            etiqueta = MiniBatchKMeans(n_clusters=q, n_init=3, random_state=semilla, batch_size=4096).fit_predict(xy_tr)
        fold = _repartir(etiqueta, k, rng)
        g_j = distancias_cv(xy_tr, fold)
        candidatos.append({"q": q, "W": wasserstein_distance(g_j, g_ij), "mediana_m": float(np.median(g_j)),
                           "p90_m": float(np.percentile(g_j, 90)), "pct_0m": float(100 * np.mean(g_j < 1)),
                           "fold": fold, "grupo": etiqueta})
    mejor = min(candidatos, key=lambda c: c["W"])
    resumen = [{k_: v for k_, v in c.items() if k_ not in ("fold", "grupo")} for c in candidatos]
    objetivo = {"mediana_m": float(np.median(g_ij)), "p90_m": float(np.percentile(g_ij, 90)), "pct_0m": float(100 * np.mean(g_ij < 1))}
    return mejor["fold"], mejor["grupo"], mejor["q"], resumen, objetivo


# ----------------------------------------------------------------------------------------------------------------------
# Regression kriging
# ----------------------------------------------------------------------------------------------------------------------
CLASES = np.array([0, 1, 25, 50, 100, 200, 400, 800, 1600, 3000])


def _exp(h, c0, c1, a):
    return c0 + c1 * (1 - np.exp(-h / a))


def ajustar_variograma(xy, r, n_muestra=8000, semilla=2026):
    rng = np.random.default_rng(semilla)
    s = rng.choice(len(r), size=min(n_muestra, len(r)), replace=False)
    P, rs = xy[s], r[s]
    pares = cKDTree(P).query_pairs(CLASES[-1], output_type="ndarray")
    h = np.linalg.norm(P[pares[:, 0]] - P[pares[:, 1]], axis=1)
    sv = 0.5 * (rs[pares[:, 0]] - rs[pares[:, 1]]) ** 2
    idx = np.digitize(h, CLASES) - 1
    hc, gc, nc = [], [], []
    for b in range(len(CLASES) - 1):
        m = idx == b
        if m.sum() >= 30:
            hc.append(h[m].mean()); gc.append(sv[m].mean()); nc.append(m.sum())
    hc, gc, nc = map(np.array, (hc, gc, nc))
    var = np.var(r)
    p, _ = curve_fit(_exp, hc, gc, p0=[0.5 * var, 0.5 * var, 300.0], sigma=1 / np.sqrt(nc),
                     bounds=([1e-6 * var, 0, 10], [2 * var, 2 * var, 5000]), maxfev=20000)
    return {"c0": p[0], "c1": p[1], "a": p[2], "var": var, "empirico": list(zip(hc.round(0), gc.round(1), nc))}


def kriging_simple(xy_src, r_src, xy_dst, vg, K=32, lote=4000):
    """Kriging simple local de residuos (media m = promedio de r_src)."""
    m = float(np.mean(r_src))
    d, nb = cKDTree(xy_src).query(xy_dst, k=K)
    c0, c1, a = vg["c0"], vg["c1"], vg["a"]
    out = np.empty(len(xy_dst))
    for i in range(0, len(xy_dst), lote):
        sl = slice(i, i + lote)
        P = xy_src[nb[sl]]                                   # (B, K, 2)
        H = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1)
        C = c1 * np.exp(-H / a) + c0 * np.eye(K)[None]
        c = c1 * np.exp(-d[sl] / a)
        lam = np.linalg.solve(C, c[..., None])[..., 0]
        out[sl] = m + np.einsum("bk,bk->b", lam, r_src[nb[sl]] - m)
    return out
