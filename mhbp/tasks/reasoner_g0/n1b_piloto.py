"""
N1b — PILOTO OFFLINE de la tarea con acantilados (frontera declarada en
ROADMAP tras N3: «valor/anticipación pagarían incluso con posterior»).

Piloto a coste CERO GPU sobre los perfiles correct(n) congelados de la
sonda N3 (12 checkpoints × 1024 instancias × malla n). Un «entorno» es
una estructura de pago/presupuesto sobre esos perfiles medidos.

HISTORIA DE RONDAS (las dos primeras las bloqueó el control, como debía):
- R1: posterior round-robin ingenuo → tan débil que el stake añadía valor
  espurio; el control lineal no reprodujo el cero de VG-B0 → no concluir.
- R2: índice por hazard MIOPE (paso siguiente) → nunca invierte profundo
  y el peso del stake compensa la miopía en vez de medir valor; además el
  diseño conflaba DOS cambios respecto a N3 (pago cliff Y sesión pequeña
  con escasez acoplada). Control roto de nuevo → no concluir.
- R3 (esta): índice NO-MIOPE (tasa máxima a cualquier horizonte, tipo
  Gittins) idéntico en ambos brazos, y diseño FACTORIZADO en 3 celdas:
    lineal-LOTE  (J=64, B=192): régimen de N3 — el control ancla: aquí
                 Δ(post+valor − post) debe ≈ 0 (el cero conocido).
    lineal-SESIÓN (J=8, B=24): ¿la escasez acoplada SOLA reabre el valor?
    cliff-SESIÓN  (J=8, B=24): la predicción del paper (primaria):
                 pago = s_alto·𝟙[TODAS las altas correctas] + Σ bajas.

Políticas (todas con la misma maquinaria; ajuste en mitad fit, eval en
mitad held-out, sesiones idénticas entre políticas = pareado):
  unif   n=3 · dif  greedy Σp̂ · valor  greedy E[pago del modo] ·
  post   índice w=1 (ciega a stakes/estructura) ·
  postval índice con w = stake (lineal) / marginal miope del producto
          (cliff: altas s_alto·Π q_j de las otras altas no convergidas).
Parada por instancia al observarse correcta (oráculo del brazo posterior
de N3).

VEREDICTO pre-declarado:
  INSTRUMENTO VÁLIDO ⇔ IC95 de Δ en lineal-LOTE contiene 0.
  Si válido: INVERSIÓN CONFIRMADA (piloto) ⇔ Δ cliff-SESIÓN > 0 con IC
  excluyendo 0. Δ lineal-SESIÓN se reporta como efecto de escasez
  (secundario, informativo para el diseño del confirmatorio).
  Si el instrumento no valida: no se concluye nada.

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.n1b_piloto
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CKPTS = [f"n2_e1_blind_flat_s{s}_r{r}" for s in range(3, 9) for r in (0, 1)]
N_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]
S_ALTO = 8.0
RNG_SES = 20260810
CELDAS = [("lineal", 64, 192, 400, "lineal_lote"),
          ("lineal", 8, 24, 2000, "lineal_sesion"),
          ("cliff", 8, 24, 2000, "cliff_sesion")]


def cargar(name):
    with open(os.path.join(RES, f"n3_sonda_{name}.json"),
              encoding="utf-8") as f:
        d = json.load(f)
    perf = np.stack([np.array(d["correct"][str(n)]) for n in N_GRID], 1)
    return perf, np.array(d["K"]), np.array(d["stake"])


def fit_tablas(perf, Ks, idx_fit):
    """p̂(n|K) isotónica simple; RATE[K][j] = tasa no-miope máxima de
    conversión por tick desde el estado j (tipo Gittins); Q[K][j] =
    P(rescate: correcto en malla máxima | no correcto en j)."""
    P = perf[idx_fit]
    phat, RATE, Q = {}, {}, {}
    for k in np.unique(Ks):
        pk = P[Ks[idx_fit] == k]
        phat[int(k)] = np.maximum.accumulate(pk.mean(0))
        nJ = len(N_GRID)
        rate = np.zeros(nJ)
        q = np.zeros(nJ)
        for j in range(nJ):
            nc = pk[:, j] == 0
            if nc.sum() >= 8:
                q[j] = pk[nc, -1].mean()
                tasas = [pk[nc, j2].mean() / (N_GRID[j2] - N_GRID[j])
                         for j2 in range(j + 1, nJ)]
                rate[j] = max(tasas) if tasas else 0.0
            else:
                q[j], rate[j] = 0.1, 0.02
        RATE[int(k)], Q[int(k)] = rate, q
    return phat, RATE, Q


def e_pago_cliff(ns, phat, Ks, altas):
    p = np.array([phat[int(Ks[i])][ns[i]] for i in range(len(ns))])
    e = S_ALTO * np.prod(p[altas]) if altas.any() else 0.0
    return e + p[~altas].sum()


def e_pago_lineal(ns, phat, Ks, stakes):
    p = np.array([phat[int(Ks[i])][ns[i]] for i in range(len(ns))])
    return float((stakes * p).sum())


def greedy_exante(phat, Ks, stakes, modo, J, B):
    altas = stakes > 1
    ns = np.zeros(J, dtype=int)
    gasto = J
    e_fn = ((lambda nn: e_pago_cliff(nn, phat, Ks, altas)) if modo == "cliff"
            else (lambda nn: e_pago_lineal(nn, phat, Ks, stakes)))
    e_cur = e_fn(ns)
    while True:
        best, bi = 0.0, -1
        for i in range(J):
            if ns[i] + 1 >= len(N_GRID):
                continue
            coste = N_GRID[ns[i] + 1] - N_GRID[ns[i]]
            if gasto + coste > B:
                continue
            n2 = ns.copy()
            n2[i] += 1
            g = (e_fn(n2) - e_cur) / coste
            if g > best:
                best, bi = g, i
        if bi < 0:
            break
        gasto += N_GRID[ns[bi] + 1] - N_GRID[ns[bi]]
        ns[bi] += 1
        e_cur = e_fn(ns)
    return ns


def rollout_posterior(perf_s, Ks, stakes, modo, con_valor, RATE, Q, J, B):
    """Índice no-miope con observación: avanza la instancia no convergida
    con mayor w·RATE; parada al observarse correcta (oráculo N3)."""
    ns = np.zeros(J, dtype=int)
    gasto = J
    conv = perf_s[np.arange(J), ns] == 1
    altas = stakes > 1
    while gasto < B:
        best, bi, bcoste = -1.0, -1, 0
        for i in range(J):
            if conv[i] or ns[i] + 1 >= len(N_GRID):
                continue
            coste = N_GRID[ns[i] + 1] - N_GRID[ns[i]]
            if gasto + coste > B:
                continue
            if not con_valor:
                w = 1.0
            elif modo == "lineal":
                w = float(stakes[i])
            elif altas[i]:
                w = S_ALTO
                for j in range(J):
                    if altas[j] and j != i and not conv[j]:
                        w *= Q[int(Ks[j])][ns[j]]
            else:
                w = 1.0
            sc = w * RATE[int(Ks[i])][ns[i]]
            if sc > best:
                best, bi, bcoste = sc, i, coste
        if bi < 0:
            break
        gasto += bcoste
        ns[bi] += 1
        conv[bi] = perf_s[bi, ns[bi]] == 1
    return ns


def pago(perf_s, ns, stakes, modo):
    c = perf_s[np.arange(len(ns)), ns]
    altas = stakes > 1
    if modo == "cliff":
        base = S_ALTO * float(c[altas].all()) if altas.any() else 0.0
        return base + float(c[~altas].sum())
    return float((stakes * c).sum())


def eval_celda(perf, Ks, stakes, phat, RATE, Q, idx_ev, modo, J, B, n_ses):
    rng = np.random.default_rng(RNG_SES)
    out = {p: [] for p in ("unif", "dif", "valor", "post", "postval")}
    i_n3 = N_GRID.index(3)
    for _ in range(n_ses):
        ses = rng.choice(idx_ev, J, replace=False)
        pf, ks, st = perf[ses], Ks[ses], stakes[ses]
        ns_u = np.full(J, i_n3)
        ns_d = greedy_exante(phat, ks, np.ones(J), "lineal", J, B)
        ns_v = greedy_exante(phat, ks, st, modo, J, B)
        ns_p = rollout_posterior(pf, ks, st, modo, False, RATE, Q, J, B)
        ns_pv = rollout_posterior(pf, ks, st, modo, True, RATE, Q, J, B)
        for tag, ns in (("unif", ns_u), ("dif", ns_d), ("valor", ns_v),
                        ("post", ns_p), ("postval", ns_pv)):
            out[tag].append(pago(pf, ns, st, modo))
    # normaliza por instancia para comparar celdas de J distinto
    return {p: float(np.mean(v)) / J for p, v in out.items()}


def main():
    res = {"ckpts": {}, "agg": {}}
    for name in CKPTS:
        perf, Ks, stakes = cargar(name)
        M = perf.shape[0]
        idx_fit, idx_ev = np.arange(M // 2), np.arange(M // 2, M)
        phat, RATE, Q = fit_tablas(perf, Ks, idx_fit)
        r = {}
        for modo, J, B, n_ses, tag in CELDAS:
            r[tag] = eval_celda(perf, Ks, stakes, phat, RATE, Q, idx_ev,
                                modo, J, B, n_ses)
        res["ckpts"][name] = r
        print(f"{name} " + " | ".join(
            tag + ": " + " ".join(f"{p}:{v:.3f}" for p, v in r[tag].items())
            for tag in ("lineal_lote", "cliff_sesion")), flush=True)
    for _, _, _, _, tag in CELDAS:
        d = np.array([res["ckpts"][n][tag]["postval"]
                      - res["ckpts"][n][tag]["post"] for n in CKPTS])
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        ic = (m - 2.201 * se, m + 2.201 * se)
        res["agg"][tag] = {"delta_postval_post": float(m), "se": float(se),
                           "ic95": [float(ic[0]), float(ic[1])]}
        print(f"Δ(post+valor − post)/inst {tag}: {m:+.4f} "
              f"IC95 [{ic[0]:+.4f}, {ic[1]:+.4f}]", flush=True)
    lote = res["agg"]["lineal_lote"]
    instr_ok = lote["ic95"][0] <= 0 <= lote["ic95"][1]
    cl = res["agg"]["cliff_sesion"]
    if not instr_ok:
        v = "INSTRUMENTO NO VALIDA (lineal-lote ≠ 0) — no concluir"
    elif cl["ic95"][0] > 0:
        v = "INVERSION CONFIRMADA (piloto)"
    else:
        v = "PREDICCION REFUTADA (piloto)"
    res["veredicto"] = v
    print(f"\nVEREDICTO N1b-piloto: {v}", flush=True)
    with open(os.path.join(RES, "n1b_piloto.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("Guardado n1b_piloto.json", flush=True)


if __name__ == "__main__":
    main()
