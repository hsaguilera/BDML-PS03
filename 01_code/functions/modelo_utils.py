"""Ajuste y predicción de las seis familias del taller (lo usan los scripts 07, 08, 10 y 11 a través de flujo_final.py).

Es el mismo código que produjo los envíos a Kaggle (se archivó una copia con cada envío). En el flujo final se llama con
contexto=False y submercado=False: el rezago espacial y los submercados se probaron y no mejoraron la validación.

Se ubican en un módulo (y no en el script) para poder evaluar los pliegues de validación en paralelo con `joblib`:
los procesos de trabajo importan este archivo y reciben los datos como arreglos de NumPy.

Contenido
---------
- `bloques_internos`: partición espacial interna (para early stopping y elección de hiperparámetros sin fuga).
- `rezago_espacial`: mediana del log-precio/m² de los k vecinos de entrenamiento (sin el propio aviso, sin su
  coordenada y sin sus duplicados de texto).
- `codificacion_submercado`: precio esperado del submercado bayesiano, ponderado por la probabilidad de pertenencia.
- `ajustar_predecir`: ajusta una de las seis familias del taller y predice; agrega las variables de contexto dentro del pliegue.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PRECIO_MIN, PRECIO_MAX = 300.0, 1650.0     # rango observado del precio en entrenamiento (millones)


# ---------------------------------------------------------------------------------------------------------------------
# Particiones internas
# ---------------------------------------------------------------------------------------------------------------------
def bloques_internos(x, y, tam=2000, frac=0.2, semilla=2026):
    """Asigna celdas cuadradas de `tam` metros al azar: ~`frac` de las celdas forman la validación interna."""
    celdas = np.floor(x / tam).astype(np.int64) * 1_000_003 + np.floor(y / tam).astype(np.int64)
    unicas = np.unique(celdas)
    rng = np.random.default_rng(semilla)
    elegidas = rng.choice(unicas, size=max(1, int(round(frac * len(unicas)))), replace=False)
    es_val = np.isin(celdas, elegidas)
    return np.where(~es_val)[0], np.where(es_val)[0]


# ---------------------------------------------------------------------------------------------------------------------
# Variables de contexto calculadas dentro del pliegue (usan el precio: nunca con datos de validación)
# ---------------------------------------------------------------------------------------------------------------------
def rezago_espacial(x, y, lpm2, grupo, idx_fuente, idx_destino, k=15):
    """Para cada aviso de `idx_destino`: mediana del log(precio/m²) de sus k vecinos más cercanos en `idx_fuente`,
    excluyendo avisos en la misma coordenada exacta y avisos con el mismo texto (duplicados). Devuelve (mediana, distancia
    media en km a esos vecinos)."""
    fuente = idx_fuente[np.isfinite(lpm2[idx_fuente])]
    arbol = cKDTree(np.c_[x[fuente], y[fuente]])
    kk = min(k + 80, len(fuente))
    d, j = arbol.query(np.c_[x[idx_destino], y[idx_destino]], k=kk)
    vec = fuente[j]
    excluir = (np.abs(x[vec] - x[idx_destino][:, None]) < 1) & (np.abs(y[vec] - y[idx_destino][:, None]) < 1)
    excluir |= grupo[vec] == grupo[idx_destino][:, None]
    d = np.where(excluir, np.inf, d)
    orden = np.argsort(d, axis=1)[:, :k]
    v_sel = np.take_along_axis(vec, orden, axis=1)
    d_sel = np.take_along_axis(d, orden, axis=1)
    validos = np.isfinite(d_sel)
    med = np.nanmedian(np.where(validos, lpm2[v_sel], np.nan), axis=1)
    dist = np.nanmean(np.where(validos, d_sel, np.nan), axis=1) / 1000
    return med, dist


def codificacion_submercado(R, objetivo, idx_fuente, idx_destino, a=50.0):
    """Valor esperado del `objetivo` según el submercado: sum_k R_ik m_k, con
    m_k = (sum_{i en fuente} R_ik * obj_i + a * media) / (sum_{i en fuente} R_ik + a) (suavizado hacia la media global)."""
    f = idx_fuente[np.isfinite(objetivo[idx_fuente])]
    media = objetivo[f].mean()
    peso = R[f].sum(axis=0)
    m = (R[f].T @ objetivo[f] + a * media) / (peso + a)
    return R[idx_destino] @ m


# ---------------------------------------------------------------------------------------------------------------------
# Red neuronal (PyTorch): MLP con BatchNorm, dropout, AdamW y early stopping
# ---------------------------------------------------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, p, capas=(256, 128), dropout=0.2):
        super().__init__()
        bloques, d = [], p
        for h in capas:
            lin = nn.Linear(d, h)
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            nn.init.zeros_(lin.bias)
            bloques += [lin, nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        bloques.append(nn.Linear(d, 1))
        self.red = nn.Sequential(*bloques)

    def forward(self, x):
        return self.red(x).squeeze(-1)


def _entrenar_mlp(Xtr, ytr, Xva=None, yva=None, epocas=None, semilla=2026, perdida="mae", lr=1e-3, wd=1e-4,
                  batch=256, max_epocas=300, paciencia=15):
    torch.manual_seed(semilla)
    gen = torch.Generator().manual_seed(semilla)
    Xt, yt = torch.as_tensor(Xtr, dtype=torch.float32), torch.as_tensor(ytr, dtype=torch.float32)
    modelo = MLP(Xt.shape[1])
    decay, no_decay = [], []
    for nombre, p in modelo.named_parameters():
        (no_decay if p.ndim == 1 else decay).append(p)          # sin weight decay en sesgos ni BatchNorm (Lecture 12)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    f_perdida = nn.L1Loss() if perdida == "mae" else nn.MSELoss()
    n, mejor, mejor_ep, estado, sin_mejora = len(Xt), np.inf, 0, None, 0
    hay_val = Xva is not None and epocas is None
    if hay_val:
        Xv, yv = torch.as_tensor(Xva, dtype=torch.float32), torch.as_tensor(yva, dtype=torch.float32)
    for ep in range(1, (epocas or max_epocas) + 1):
        modelo.train()
        perm = torch.randperm(n, generator=gen)
        Xp, yp = Xt[perm], yt[perm]
        for i in range(0, n, batch):
            xb, yb = Xp[i:i + batch], yp[i:i + batch]
            if len(xb) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            f_perdida(modelo(xb), yb).backward()
            opt.step()
        if hay_val:
            modelo.eval()
            with torch.inference_mode():
                pv = (modelo(Xv) - yv).abs().mean().item()
            if pv < mejor - 1e-4:
                mejor, mejor_ep, sin_mejora = pv, ep, 0
                estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
            else:
                sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    if hay_val and estado is not None:
        modelo.load_state_dict(estado)
    return modelo, (mejor_ep if hay_val else epocas)


# ---------------------------------------------------------------------------------------------------------------------
# Ajuste y predicción de una familia (con variables de contexto dentro del pliegue)
# ---------------------------------------------------------------------------------------------------------------------
def ajustar_predecir(familia, X, y, x, yc, lpm2, lp, grupo, R, idx_tr, idx_va, contexto=True, submercado=True,
                     params=None, semilla=2026):
    """Ajusta `familia` con los avisos `idx_tr` y predice `idx_va` (precio en millones, recortado al rango de entrenamiento).

    X: variables base (n × p). y: precio (millones; NaN en prueba). x, yc: coordenadas en metros (EPSG:9377).
    lpm2, lp: log(precio/m²) y log(precio) (NaN donde no hay). grupo: identificador de texto duplicado.
    R: probabilidades de submercado (n × K). contexto/submercado: si se agregan esas variables (calculadas solo con idx_tr).
    """
    torch.set_num_threads(1)
    params = dict(params or {})
    particion = params.pop("particion_interna", None)   # (i_in, i_iv) relativos a idx_tr; mismo esquema que la CV externa
    bloques = [X[idx_tr]], [X[idx_va]]
    if contexto:
        # Para el entrenamiento, el rezago de cada aviso usa a los demás avisos de entrenamiento (sin sí mismo,
        # sin su coordenada y sin sus duplicados); para validación usa solo el entrenamiento.
        m_tr, d_tr = rezago_espacial(x, yc, lpm2, grupo, idx_tr, idx_tr)
        m_va, d_va = rezago_espacial(x, yc, lpm2, grupo, idx_tr, idx_va)
        bloques[0].append(np.c_[m_tr, d_tr]); bloques[1].append(np.c_[m_va, d_va])
    if submercado and R is not None:
        s_tr = np.c_[codificacion_submercado(R, lpm2, idx_tr, idx_tr), codificacion_submercado(R, lp, idx_tr, idx_tr)]
        s_va = np.c_[codificacion_submercado(R, lpm2, idx_tr, idx_va), codificacion_submercado(R, lp, idx_tr, idx_va)]
        bloques[0].append(s_tr); bloques[1].append(s_va)
    Xtr, Xva = np.hstack(bloques[0]), np.hstack(bloques[1])
    ytr = y[idx_tr]
    if particion is not None:
        i_in, i_iv = particion
    else:
        i_in, i_iv = bloques_internos(x[idx_tr], yc[idx_tr], semilla=semilla)     # partición espacial interna

    if familia in ("ols", "elastic_net", "nn", "nn_atencion"):
        pre = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler())
        Ztr = pre.fit_transform(Xtr)
        Zva = pre.transform(Xva)

    if familia == "ols":
        from sklearn.linear_model import LinearRegression
        pred = LinearRegression().fit(Ztr, ytr).predict(Zva)
    elif familia == "elastic_net":
        from sklearn.linear_model import ElasticNetCV
        cv = [bloques_internos(x[idx_tr], yc[idx_tr], semilla=semilla + s) for s in range(3)]
        m = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], n_alphas=30, cv=cv, max_iter=5000, n_jobs=1, random_state=semilla)
        pred = m.fit(Ztr, ytr).predict(Zva)
    elif familia == "cart":
        from sklearn.tree import DecisionTreeRegressor
        mejores = []
        for prof in params.get("profundidades", [6, 8, 10, 12, 15]):
            t = DecisionTreeRegressor(max_depth=prof, min_samples_leaf=20, random_state=semilla).fit(Xtr[i_in], ytr[i_in])
            mejores.append((np.mean(np.abs(t.predict(Xtr[i_iv]) - ytr[i_iv])), prof))
        prof = min(mejores)[1]
        pred = DecisionTreeRegressor(max_depth=prof, min_samples_leaf=20, random_state=semilla).fit(Xtr, ytr).predict(Xva)
    elif familia == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=params.get("n_estimators", 300), max_features=params.get("max_features", 0.3),
                                  min_samples_leaf=params.get("min_samples_leaf", 5), n_jobs=3, random_state=semilla)
        pred = m.fit(Xtr, ytr).predict(Xva)
    elif familia == "lightgbm":
        import lightgbm as lgb
        p = dict(objective="l2", learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.7, reg_lambda=1.0, n_estimators=5000, random_state=semilla, verbose=-1, n_jobs=3)
        p.update(params)
        metrica = "l2" if p["objective"] in ("l2", "regression") else "l1"
        m = lgb.LGBMRegressor(**p).fit(Xtr[i_in], ytr[i_in], eval_set=[(Xtr[i_iv], ytr[i_iv])], eval_metric=metrica,
                                       callbacks=[lgb.early_stopping(100, verbose=False, first_metric_only=True)])
        p["n_estimators"] = max(int(m.best_iteration_ or 100), 50)
        pred = lgb.LGBMRegressor(**p).fit(Xtr, ytr).predict(Xva)
    elif familia == "catboost":
        from catboost import CatBoostRegressor
        p = dict(loss_function="RMSE", depth=8, learning_rate=0.05, iterations=4000, l2_leaf_reg=3.0, random_seed=semilla,
                 verbose=0, thread_count=3, allow_writing_files=False)
        p.update(params)
        m = CatBoostRegressor(**p).fit(Xtr[i_in], ytr[i_in], eval_set=(Xtr[i_iv], ytr[i_iv]), early_stopping_rounds=150)
        p["iterations"] = max(int(m.get_best_iteration() or 100), 50)
        pred = CatBoostRegressor(**p).fit(Xtr, ytr).predict(Xva)
    elif familia == "nn":
        med, de = float(np.mean(ytr)), float(np.std(ytr))
        _, ep = _entrenar_mlp(Ztr[i_in], (ytr[i_in] - med) / de, Ztr[i_iv], (ytr[i_iv] - med) / de, semilla=semilla,
                              perdida=params.get("perdida", "mae"))
        modelo, _ = _entrenar_mlp(Ztr, (ytr - med) / de, epocas=max(ep, 5), semilla=semilla, perdida=params.get("perdida", "mae"))
        modelo.eval()
        with torch.inference_mode():
            pred = modelo(torch.as_tensor(Zva, dtype=torch.float32)).numpy() * de + med
    elif familia == "nn_atencion":
        return ajustar_predecir_atencion(Ztr, Zva, ytr, x, yc, lp, lpm2, grupo, params["H_base"], idx_tr, idx_va, semilla,
                                         particion=(i_in, i_iv))
    else:
        raise ValueError(familia)
    return pred                                  # sin recorte al rango de entrenamiento (decisión del usuario)


# ---------------------------------------------------------------------------------------------------------------------
# Red neuronal espacial con atención a los vecinos (inspirada en Viana y Barbosa, 2021, "Attention-based spatial
# interpolation for house price prediction")
# ---------------------------------------------------------------------------------------------------------------------
def vecinos_fuente(x, y, grupo, idx_fuente, idx_destino, k=20):
    """k vecinos más cercanos de cada destino dentro de `idx_fuente`, sin la misma coordenada ni duplicados de texto.
    Devuelve (índices n_dest × k, distancias en metros, máscara de vecinos válidos)."""
    arbol = cKDTree(np.c_[x[idx_fuente], y[idx_fuente]])
    kk = min(k + 80, len(idx_fuente))
    d, j = arbol.query(np.c_[x[idx_destino], y[idx_destino]], k=kk)
    vec = idx_fuente[j]
    excluir = (np.abs(x[vec] - x[idx_destino][:, None]) < 1) & (np.abs(y[vec] - y[idx_destino][:, None]) < 1)
    excluir |= grupo[vec] == grupo[idx_destino][:, None]
    d = np.where(excluir, np.inf, d)
    orden = np.argsort(d, axis=1)[:, :k]
    v = np.take_along_axis(vec, orden, axis=1)
    dd = np.take_along_axis(d, orden, axis=1)
    mascara = np.isfinite(dd)
    dd = np.where(mascara, dd, 0.0)
    return v, dd, mascara


class RedAtencion(nn.Module):
    """ŷ_i = f(x_i, Σ_j α_ij v(h_j)); α_ij = softmax_j( q(x_i)·k(h_j, d_ij) / √d_a ) sobre los k vecinos de entrenamiento."""

    def __init__(self, p, q, d=128, da=32, dropout=0.2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(p, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(256, d), nn.ReLU())
        self.query = nn.Linear(d, da)
        self.key = nn.Sequential(nn.Linear(q + 1, 64), nn.ReLU(), nn.Linear(64, da))
        self.val = nn.Sequential(nn.Linear(q + 1, 64), nn.ReLU(), nn.Linear(64, d))
        self.head = nn.Sequential(nn.Linear(2 * d, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))
        self.escala = da ** 0.5

    def forward(self, xb, hb, db, mb):
        e = self.enc(xb)                                              # B × d
        hn = torch.cat([hb, db.unsqueeze(-1)], dim=-1)               # B × k × (q+1)
        s = (self.key(hn) * self.query(e).unsqueeze(1)).sum(-1) / self.escala
        s = s.masked_fill(~mb, -1e9)
        a = torch.softmax(s, dim=1)                                    # pesos de atención α_ij
        c = (a.unsqueeze(-1) * self.val(hn)).sum(1)                    # contexto de la vecindad
        return self.head(torch.cat([e, c], dim=-1)).squeeze(-1), a


def _entrenar_atencion(Z, ystd, H, V, D, M, filas_ent, filas_val=None, epocas=None, semilla=2026, lr=1e-3, wd=1e-4,
                       batch=512, max_epocas=200, paciencia=12):
    """Z: variables propias (n_all × p), H: atributos de vecinos (n_all × q), V/D/M: vecinos, distancias y máscara por
    fila de Z. `filas_ent`/`filas_val`: posiciones (en Z) de entrenamiento y validación interna."""
    torch.manual_seed(semilla)
    gen = torch.Generator().manual_seed(semilla)
    Zt, Ht = torch.as_tensor(Z, dtype=torch.float32), torch.as_tensor(H, dtype=torch.float32)
    Vt, Mt = torch.as_tensor(V, dtype=torch.long), torch.as_tensor(M)
    Dt = torch.as_tensor(np.log1p(D / 100.0), dtype=torch.float32)
    yt = torch.as_tensor(ystd, dtype=torch.float32)
    modelo = RedAtencion(Z.shape[1], H.shape[1])
    decay = [p for n_, p in modelo.named_parameters() if p.ndim > 1]
    no_decay = [p for n_, p in modelo.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    f = nn.L1Loss()
    ent = torch.as_tensor(filas_ent, dtype=torch.long)
    mejor, mejor_ep, estado, sin_mejora = np.inf, 0, None, 0
    hay_val = filas_val is not None and epocas is None
    for ep in range(1, (epocas or max_epocas) + 1):
        modelo.train()
        perm = ent[torch.randperm(len(ent), generator=gen)]
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            if len(b) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            pred, _ = modelo(Zt[b], Ht[Vt[b]], Dt[b], Mt[b])
            f(pred, yt[b]).backward()
            opt.step()
        if hay_val:
            modelo.eval()
            with torch.inference_mode():
                bv = torch.as_tensor(filas_val, dtype=torch.long)
                pv = (modelo(Zt[bv], Ht[Vt[bv]], Dt[bv], Mt[bv])[0] - yt[bv]).abs().mean().item()
            if pv < mejor - 1e-4:
                mejor, mejor_ep, sin_mejora = pv, ep, 0
                estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
            else:
                sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    if hay_val and estado is not None:
        modelo.load_state_dict(estado)
    return modelo, (mejor_ep if hay_val else epocas)


def ajustar_predecir_atencion(Ztr, Zva, ytr, x, yc, lp, lpm2, grupo, H_base, idx_tr, idx_va, semilla=2026, k=20,
                              particion=None):
    """Red con atención: los vecinos de cada aviso salen SOLO del entrenamiento del pliegue (sin fuga)."""
    torch.set_num_threads(2)
    n_tr = len(idx_tr)
    # Atributos de los vecinos: log-precio, log-precio/m² (con bandera de faltante) y atributos estructurales
    lpm2_f = np.where(np.isfinite(lpm2), lpm2, np.nanmean(lpm2[idx_tr]))
    Hraw = np.c_[lp, lpm2_f, (~np.isfinite(lpm2)).astype(float), H_base]
    Hraw = np.where(np.isfinite(Hraw), Hraw, np.nanmedian(Hraw[idx_tr], axis=0))
    mu_h, sd_h = Hraw[idx_tr].mean(axis=0), Hraw[idx_tr].std(axis=0) + 1e-9
    H = (Hraw - mu_h) / sd_h
    H[idx_va, 0] = 0.0; H[idx_va, 1] = 0.0                         # nunca se usa el precio de validación
    # Filas: 0..n_tr-1 = entrenamiento del pliegue; n_tr.. = validación. Z_all apila ambas.
    Z_all = np.vstack([Ztr, Zva])
    pos = {int(i): r for r, i in enumerate(np.r_[idx_tr, idx_va])}
    med, de = float(np.mean(ytr)), float(np.std(ytr))
    y_all = np.r_[(ytr - med) / de, np.zeros(len(idx_va))]
    # Partición interna para el early stopping (la misma lógica de la CV externa si se entrega)
    i_in, i_iv = particion if particion is not None else bloques_internos(x[idx_tr], yc[idx_tr], semilla=semilla)
    fuente_in = idx_tr[i_in]
    V1, D1, M1 = vecinos_fuente(x, yc, grupo, fuente_in, idx_tr, k)          # vecinos solo de la parte interna de entrenamiento
    _, ep = _entrenar_atencion(Z_all[:n_tr], y_all[:n_tr], H, V1, D1, M1, i_in, i_iv, semilla=semilla)
    # Reentrenamiento con todo el entrenamiento del pliegue; vecinos de entrenamiento y de validación desde idx_tr
    V2, D2, M2 = vecinos_fuente(x, yc, grupo, idx_tr, np.r_[idx_tr, idx_va], k)
    modelo, _ = _entrenar_atencion(Z_all, y_all, H, V2, D2, M2, np.arange(n_tr), epocas=max(ep, 5), semilla=semilla)
    modelo.eval()
    with torch.inference_mode():
        filas = torch.arange(n_tr, n_tr + len(idx_va))
        Zt = torch.as_tensor(Z_all, dtype=torch.float32)
        pred, alfa = modelo(Zt[filas], torch.as_tensor(H, dtype=torch.float32)[torch.as_tensor(V2[n_tr:])],
                            torch.as_tensor(np.log1p(D2[n_tr:] / 100.0), dtype=torch.float32), torch.as_tensor(M2[n_tr:]))
    return pred.numpy() * de + med               # sin recorte al rango de entrenamiento
