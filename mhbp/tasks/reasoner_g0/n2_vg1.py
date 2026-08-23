"""
N2 — VG1 (PREREG_N2 v2 §6): headroom de VALOR en S₅ con asignación ÓPTIMA
EXACTA y factibilidad conjunta. CPU puro sobre el perfil real.

Por celda de diales (E[n̄] × s_alto × p_hi) computa:
  · payoff_norm de 4 políticas ÓPTIMAS (barrido de multiplicador λ — óptimo
    exacto del problema relajado, clases (K, stake) continuas):
      uniforme      n = E[n̄] constante (referencia secundaria)
      dificultad    n(K), CIEGA al stake  ← EL BASELINE (lo que blind aproxima)
      valor         n(K, s)               ← el techo
      valor_retard  n(K, s) con n ≥ 3     ← coste estructural de endogeneidad
  · headroom := valor − dificultad  (el número del gate)
  · coste_acc := acc_no_pond(valor) − acc_no_pond(dificultad)  (factibilidad
    conjunta con el guardarraíl re-anclado)
  · r_pb := corr punto-biserial(n, stake) bajo la política valor (ancla C3)
  · σ_sim del payoff_norm (bootstrap de m=4096 instancias Bernoulli, eval
    común congelada — el suelo de muestreo del §8)

Umbral provisional (pre-VG4): headroom ≥ 0.03 absoluto (suelo conservador
del panel) — el definitivo (4·σ_run^UCB90 y 2×MDD) se fija por enmienda tras
VG4. Salida: results/n2_vg1.json + tabla.

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.n2_vg1
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
K_LO, K_HI = 13, 24
N_MAX = 24


def load_acc():
    p = json.load(open(os.path.join(RES, "f3b_acc_profile.json"),
                       encoding="utf-8"))
    ks = sorted(int(k) for k in p["acc"])
    k_min = ks[0]
    acc = np.stack([np.array(p["acc"][str(K)], dtype=float) for K in ks],
                   axis=1)                        # (n=1..24, K)
    return acc, k_min


def opt_alloc(values, lam):
    """n*(clase) = argmax_n [values(n, clase) − λ·n]; values (N_MAX, C)."""
    n_grid = np.arange(1, N_MAX + 1)[:, None]
    return np.argmax(values - lam * n_grid, axis=0) + 1


def solve_budget(values, weights, target_En, lo=0.0, hi=2.0, iters=60):
    """Bisección de λ para que E_w[n*] = target (óptimo exacto relajado;
    en el λ crítico mezcla las dos soluciones adyacentes)."""
    w = weights / weights.sum()
    for _ in range(iters):
        lam = 0.5 * (lo + hi)
        n = opt_alloc(values, lam)
        En = float((w * n).sum())
        if En > target_En:
            lo = lam
        else:
            hi = lam
    n_hi = opt_alloc(values, lo)                  # gasta ≥ target
    n_lo = opt_alloc(values, hi)                  # gasta ≤ target
    En_hi, En_lo = float((w * n_hi).sum()), float((w * n_lo).sum())
    if abs(En_hi - En_lo) < 1e-9:
        return n_lo.astype(float), 1.0
    alpha = (target_En - En_lo) / (En_hi - En_lo)  # mezcla exacta
    return n_lo.astype(float), n_hi.astype(float), float(np.clip(alpha, 0, 1))


def eval_mix(alloc, acc_c, stakes_c, weights):
    """payoff_norm y acc de una asignación (posiblemente mezcla)."""
    if len(alloc) == 3:
        n_lo, n_hi, a = alloc
        pay_lo, acc_lo = eval_mix((n_lo,), acc_c, stakes_c, weights)
        pay_hi, acc_hi = eval_mix((n_hi,), acc_c, stakes_c, weights)
        return (1 - a) * pay_lo + a * pay_hi, (1 - a) * acc_lo + a * acc_hi
    n = alloc[0].astype(int) - 1
    a = acc_c[n, np.arange(len(n))]
    w = weights / weights.sum()
    pay = float((w * stakes_c * a).sum() / (w * stakes_c).sum())
    acc_u = float((w * a).sum())
    return pay, acc_u


def celda(acc, k_min, En, s_alto, p_hi, delay=False):
    Ks = np.arange(K_LO, K_HI + 1)
    C = len(Ks) * 2
    acc_c = np.zeros((N_MAX, C))
    stakes_c = np.zeros(C)
    weights = np.zeros(C)
    for i, K in enumerate(Ks):
        col = acc[:, K - k_min]
        acc_c[:, 2 * i], acc_c[:, 2 * i + 1] = col, col
        stakes_c[2 * i], stakes_c[2 * i + 1] = 1.0, float(s_alto)
        weights[2 * i] = (1 - p_hi) / len(Ks)
        weights[2 * i + 1] = p_hi / len(Ks)
    v_value = stakes_c[None, :] * acc_c           # valor por clase
    if delay:                                     # endogeneidad: n ≥ 3
        v_value = v_value.copy()
        v_value[:2, :] = -1e9
    # dificultad (stake-CIEGA): mismo n para ambos stakes de un K → colapsa
    # clases por K con valor esperado del stake
    s_bar = (1 - p_hi) + p_hi * s_alto
    v_diff = np.zeros((N_MAX, len(Ks)))
    for i in range(len(Ks)):
        v_diff[:, i] = s_bar * acc_c[:, 2 * i]
    w_K = np.full(len(Ks), 1.0 / len(Ks))

    a_val = solve_budget(v_value, weights, En)
    a_dif = solve_budget(v_diff, w_K, En)
    pay_val, acc_val = eval_mix(a_val if len(a_val) == 3 else (a_val[0],),
                                acc_c, stakes_c, weights)
    # expandir la asignación por-K de dificultad a las clases (K, s)
    def expand(nK):
        n = np.zeros(C)
        for i in range(len(Ks)):
            n[2 * i] = n[2 * i + 1] = nK[i]
        return n
    if len(a_dif) == 3:
        a_dif_c = (expand(a_dif[0]), expand(a_dif[1]), a_dif[2])
    else:
        a_dif_c = (expand(a_dif[0]),)
    pay_dif, acc_dif = eval_mix(a_dif_c, acc_c, stakes_c, weights)
    uni = (np.full(C, float(En)),)
    # uniforme entera más cercana (mezcla de floor/ceil)
    fl, ce = math.floor(En), math.ceil(En)
    if fl == ce:
        pay_uni, acc_uni = eval_mix((np.full(C, float(fl)),),
                                    acc_c, stakes_c, weights)
    else:
        a_mix = (np.full(C, float(fl)), np.full(C, float(ce)), En - fl)
        pay_uni, acc_uni = eval_mix(a_mix, acc_c, stakes_c, weights)

    # r_pb(n, stake) bajo la política de valor (mezcla → n esperado por clase)
    if len(a_val) == 3:
        n_exp = (1 - a_val[2]) * a_val[0] + a_val[2] * a_val[1]
    else:
        n_exp = a_val[0]
    w = weights / weights.sum()
    mu = (w * n_exp).sum()
    var_n = (w * (n_exp - mu) ** 2).sum()
    hi_mask = stakes_c > 1
    p = w[hi_mask].sum()
    mu1 = (w[hi_mask] * n_exp[hi_mask]).sum() / max(p, 1e-9)
    mu0 = (w[~hi_mask] * n_exp[~hi_mask]).sum() / max(1 - p, 1e-9)
    r_pb = ((mu1 - mu0) * math.sqrt(p * (1 - p))
            / max(math.sqrt(var_n), 1e-9))

    # σ_sim: bootstrap del payoff_norm con m=4096 (Bernoulli, eval congelada)
    rng = np.random.default_rng(0)
    m = 4096
    cls = rng.choice(len(w), size=m, p=w)
    if len(a_val) == 3:
        pick_hi = rng.random(m) < a_val[2]
        n_i = np.where(pick_hi, a_val[1][cls], a_val[0][cls]).astype(int)
    else:
        n_i = a_val[0][cls].astype(int)
    p_i = acc_c[n_i - 1, cls]
    s_i = stakes_c[cls]
    boots = np.empty(400)
    hits = (rng.random((400, m)) < p_i[None, :]).astype(float)
    boots = (hits * s_i).sum(1) / s_i.sum()
    sigma_sim = float(boots.std())

    return {"payoff": {"uniforme": pay_uni, "dificultad": pay_dif,
                       "valor": pay_val},
            "acc": {"uniforme": acc_uni, "dificultad": acc_dif,
                    "valor": acc_val},
            "headroom": pay_val - pay_dif,
            "coste_acc": acc_val - acc_dif,
            "r_pb_oraculo": r_pb, "sigma_sim_eval": sigma_sim}


def main():
    acc, k_min = load_acc()
    UMBRAL_PROVISIONAL = 0.03      # suelo conservador pre-VG4 (panel)
    GUARD_ACC = -0.03              # referencia del guardarraíl re-anclado
    out = {"config": {"K": [K_LO, K_HI], "umbral_provisional": UMBRAL_PROVISIONAL},
           "celdas": {}}
    print(f"{'En':>4} {'s':>3} {'p_hi':>5} | {'unif':>6} {'dif':>6} {'val':>6} "
          f"{'val_d3':>6} | {'headrm':>7} {'coste':>7} {'r_pb':>5} {'σ_sim':>6} | gate")
    aprobadas = []
    for En in (8, 10, 12):
        for s in (8, 16):
            for p_hi in (0.05, 0.10, 0.15, 0.25):
                c = celda(acc, k_min, En, s, p_hi)
                cd = celda(acc, k_min, En, s, p_hi, delay=True)
                key = f"En{En}_s{s}_p{p_hi}"
                c["payoff"]["valor_retardado"] = cd["payoff"]["valor"]
                ok = (c["headroom"] >= UMBRAL_PROVISIONAL
                      and c["coste_acc"] >= GUARD_ACC)
                c["aprueba_provisional"] = bool(ok)
                out["celdas"][key] = c
                if ok:
                    aprobadas.append((key, c["headroom"], c["coste_acc"]))
                print(f"{En:>4} {s:>3} {p_hi:>5.2f} | "
                      f"{c['payoff']['uniforme']:.4f} "
                      f"{c['payoff']['dificultad']:.4f} "
                      f"{c['payoff']['valor']:.4f} "
                      f"{cd['payoff']['valor']:.4f} | "
                      f"{c['headroom']:+.4f} {c['coste_acc']:+.4f} "
                      f"{c['r_pb_oraculo']:.2f} {c['sigma_sim_eval']:.4f} | "
                      f"{'APRUEBA*' if ok else '—'}")
    out["aprobadas_provisional"] = aprobadas
    with open(os.path.join(RES, "n2_vg1.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\n{len(aprobadas)} celdas aprueban el umbral PROVISIONAL "
          f"(headroom≥{UMBRAL_PROVISIONAL}, coste≥{GUARD_ACC}); el definitivo "
          f"se fija tras VG4. Guardado results/n2_vg1.json")


if __name__ == "__main__":
    main()
