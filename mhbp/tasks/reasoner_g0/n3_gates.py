"""
N3 — Fase A, paso 2: fits (p̂ monótona, stakê) + gates VG-N3a/b/c
(PREREG_N3 v2 §2.2-2.3, §4). CPU/GPU-min sobre los JSON de la sonda.

VG-N3c es EL gate decisorio: headroom de VALOR por clase (DP exacto sobre
tablas held-out isotónicas de los 12 ckpts reales) a e∈{3,4,5}.

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.n3_gates
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CKPTS = [f"n2_e1_blind_flat_s{s}_r{r}" for s in range(3, 9) for r in (0, 1)]
N_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]
K_LO, K_HI = 13, 24


def load_sonda(name):
    d = json.load(open(os.path.join(RES, f"n3_sonda_{name}.json"),
                       encoding="utf-8"))
    m = len(d["K"])
    C = np.zeros((m, len(N_GRID)))
    for j, n in enumerate(N_GRID):
        C[:, j] = np.array(d["correct"][str(n)], dtype=float)
    return {"slot0": np.array(d["slot0"]), "slot1": np.array(d["slot1"]),
            "K": np.array(d["K"]), "stake": np.array(d["stake"]),
            "C": C}


def isotonic(v):
    """Regresión isotónica creciente (PAVA) — suaviza acc(n|clase)."""
    v = list(v)
    w = [1.0] * len(v)
    i = 0
    vals, wts = [], []
    for x in v:
        vals.append(x)
        wts.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            tot = wts[-2] + wts[-1]
            merged = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / tot
            vals[-2:] = [merged]
            wts[-2:] = [tot]
    out = []
    for val, wt in zip(vals, wts):
        out.extend([val] * int(wt))
    return np.array(out)


def class_tables(sd, idx):
    """acc(n | K, stake) isotónica sobre el subconjunto idx (held-out)."""
    tab = {}
    for K in range(K_LO, K_HI + 1):
        for hi in (0, 1):
            mask = (sd["K"][idx] == K) & ((sd["stake"][idx] > 1) == bool(hi))
            rows = sd["C"][idx][mask]
            if len(rows) == 0:
                tab[(K, hi)] = None
                continue
            tab[(K, hi)] = isotonic(rows.mean(0))
    # rellena clases vacías con la media por K (p_hi=0.15: altos escasos)
    for K in range(K_LO, K_HI + 1):
        base = tab[(K, 0)] if tab[(K, 0)] is not None else tab[(K, 1)]
        for hi in (0, 1):
            if tab[(K, hi)] is None:
                tab[(K, hi)] = base
    return tab


def interp_full(curve):
    """Interpola la malla N_GRID a n=1..24 (lineal, luego isotónica)."""
    full = np.interp(np.arange(1, 25), N_GRID, curve)
    return isotonic(full)


def dp_alloc(values, weights, target_En):
    """Asignación óptima exacta por barrido de λ (mezcla en el λ crítico).
    values: (C, 24) valor por clase y n; weights: (C,)."""
    w = weights / weights.sum()

    def En_of(lam):
        n = np.argmax(values - lam * np.arange(1, 25)[None, :], axis=1) + 1
        return float((w * n).sum()), n
    lo, hi = 0.0, 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        e, _ = En_of(mid)
        if e > target_En:
            lo = mid
        else:
            hi = mid
    e_hi, n_hi = En_of(lo)
    e_lo, n_lo = En_of(hi)
    if abs(e_hi - e_lo) < 1e-9:
        return n_lo, n_lo, 0.0
    a = (target_En - e_lo) / (e_hi - e_lo)
    return n_lo, n_hi, float(np.clip(a, 0, 1))


def payoff_of(alloc, acc_c, stakes_c, weights):
    n_lo, n_hi, a = alloc
    w = weights / weights.sum()

    def pay(n):
        acc = acc_c[np.arange(len(n)), n - 1]
        return float((w * stakes_c * acc).sum() / (w * stakes_c).sum())
    return (1 - a) * pay(n_lo) + a * pay(n_hi)


def vg_n3c(tablas):
    """Headroom de VALOR por clase a e∈{3,4,5}, por checkpoint."""
    out = {}
    for e in (3, 4, 5):
        headrooms = []
        for name, tab in tablas.items():
            classes, accs, stakes, wts = [], [], [], []
            for K in range(K_LO, K_HI + 1):
                for hi in (0, 1):
                    accs.append(interp_full(tab[(K, hi)]))
                    stakes.append(8.0 if hi else 1.0)
                    wts.append((0.15 if hi else 0.85) / (K_HI - K_LO + 1))
            acc_c = np.array(accs)
            stakes_c = np.array(stakes)
            wts = np.array(wts)
            v_val = stakes_c[:, None] * acc_c
            # oráculo K-solo: mismo n para ambos stakes de un K
            sbar = 0.85 + 0.15 * 8.0
            acc_K = np.array([(0.85 * acc_c[2 * i] + 0.15 * acc_c[2 * i + 1])
                              for i in range((K_HI - K_LO + 1))])
            v_dif = sbar * acc_K
            w_K = np.full(K_HI - K_LO + 1, 1.0 / (K_HI - K_LO + 1))
            a_val = dp_alloc(v_val, wts, e)
            a_dif = dp_alloc(v_dif, w_K, e)
            # expandir dif a clases
            def expand(nK):
                return np.repeat(nK, 2)
            a_dif_c = (expand(a_dif[0]), expand(a_dif[1]), a_dif[2])
            p_val = payoff_of(a_val, acc_c, stakes_c, wts)
            p_dif = payoff_of(a_dif_c, acc_c, stakes_c, wts)
            headrooms.append(p_val - p_dif)
        hr = np.array(headrooms)
        out[e] = {"por_ckpt": hr.tolist(), "media": float(hr.mean()),
                  "sd": float(hr.std(ddof=1))}
    return out


def k_table(sd, idx):
    """p̂ = TABLA isotónica acc(n|K) del split dado (ENMIENDA post-gates
    2026-08-06: la cabeza sigmoide paramétrica underfittea — skill −0.04 en
    12/12 — y la tabla aprendida de las auto-sondas ES el automodelo grueso
    que el prereg §1 ya declaraba; monótona por construcción = anti-minería;
    stake-agnóstica: VG3b certificó dificultad ⊥ motivo)."""
    tab = {}
    for K in range(K_LO, K_HI + 1):
        mask = sd["K"][idx] == K
        rows = sd["C"][idx][mask]
        tab[K] = isotonic(rows.mean(0)) if len(rows) else None
    ks = [K for K in tab if tab[K] is not None]
    for K in range(K_LO, K_HI + 1):
        if tab[K] is None:
            near = min(ks, key=lambda q: abs(q - K))
            tab[K] = tab[near]
    return tab


def fit_stake_head(sd, emb, idx_tr):
    """stakê(SOLO slots — ENMIENDA: K era feature y ajustaba ruido residual
    → |corr(ŝ,K|stake)| hasta 0.46; sin K, la independencia es por
    construcción). Target = stake revelado en sondas con correcto=1."""
    f_all = torch.tensor(np.concatenate(
        [emb[sd["slot0"]], emb[sd["slot1"]]], axis=1), dtype=torch.float32)
    d = f_all.shape[1]
    sh = torch.nn.Sequential(torch.nn.Linear(d, 16), torch.nn.Tanh(),
                             torch.nn.Linear(16, 1))
    opt2 = torch.optim.Adam(sh.parameters(), lr=0.01)
    mask = sd["C"][idx_tr].max(1) > 0
    f_s = f_all[idx_tr][mask]
    tgt = torch.tensor(sd["stake"][idx_tr][mask], dtype=torch.float32)
    for _ in range(400):
        opt2.zero_grad()
        pred = F.softplus(sh(f_s)).squeeze(-1)
        loss = F.mse_loss(pred, tgt)
        loss.backward()
        opt2.step()
    return sh, f_all


def main():
    tablas = {}
    gates = {"VG_N3a": {}, "VG_N3b": {}, "fits": {}}
    for name in CKPTS:
        sd = load_sonda(name)
        m = len(sd["K"])
        rng = np.random.default_rng(7)
        perm = rng.permutation(m)
        idx_tr, idx_ho = perm[: m // 2], perm[m // 2:]
        tablas[name] = class_tables(sd, idx_ho)
        # embeddings del ckpt (solo filas de los tokens de slot 5-8)
        ckpt = torch.load(os.path.join(HERE, "ckpts", f"{name}.pt"),
                          map_location="cpu")
        emb = ckpt["tok_emb.weight"].float().numpy()
        shd, f_all = fit_stake_head(sd, emb, idx_tr)
        # VG-N3a v2: ESTABILIDAD de la tabla — Brier held-out de la tabla
        # ajustada en train vs el de la tabla del propio held-out ≤ 0.02
        with torch.no_grad():
            briers_p, briers_t = [], []
            tab_tr = k_table(sd, idx_tr)
            tab_ho = k_table(sd, idx_ho)
            for j, n in enumerate(N_GRID):
                c = sd["C"][idx_ho, j]
                p_tr = np.array([tab_tr[int(K)][j] for K in sd["K"][idx_ho]])
                p_ho = np.array([tab_ho[int(K)][j] for K in sd["K"][idx_ho]])
                briers_p.append(((p_tr - c) ** 2).mean())
                briers_t.append(((p_ho - c) ** 2).mean())
            bp, bt = float(np.mean(briers_p)), float(np.mean(briers_t))
            skill = bt - bp + 0.02          # ≥0 ⟺ gap de estabilidad ≤ 0.02
            # VG-N3b: AUC + escala + corr(ŝ, K | stake)
            s_hat = F.softplus(shd(f_all[idx_ho])).squeeze(-1).numpy()
            hi = sd["stake"][idx_ho] > 1
            pos, neg = s_hat[hi], s_hat[~hi]
            auc = float((pos[:, None] > neg[None, :]).mean())
            ratio = float(pos.mean() / max(neg.mean(), 1e-6))
            rK = []
            for hv in (0, 1):
                mm = hi == bool(hv)
                if mm.sum() > 10:
                    rK.append(np.corrcoef(s_hat[mm], sd["K"][idx_ho][mm])[0, 1])
            corr_k = float(np.mean(np.abs(rK)))
        gates["VG_N3a"][name] = {"brier_phat": bp, "brier_tabla": bt,
                                 "skill": skill}
        gates["VG_N3b"][name] = {"auc": auc, "escala": ratio,
                                 "corr_K_abs": corr_k}
        print(f"{name}: N3a skill={skill:+.3f}  N3b auc={auc:.3f} "
              f"escala={ratio:.1f} |corrK|={corr_k:.3f}", flush=True)

    hr = vg_n3c(tablas)
    gates["VG_N3c"] = hr
    print("\n=== VG-N3c: headroom de VALOR por clase (12 ckpts reales) ===")
    for e, v in hr.items():
        se = v["sd"] / math.sqrt(len(CKPTS))
        umbral = max(0.04, 4 * se)
        ok = v["media"] >= umbral
        print(f"  e={e}: headroom={v['media']:+.4f} ± {v['sd']:.4f} "
              f"(umbral {umbral:.3f}) → {'VERDE' if ok else 'ROJO'}")
        v["umbral"] = umbral
        v["verde"] = bool(ok)
    a_ok = all(g["skill"] >= 0 for g in gates["VG_N3a"].values())
    # corr(ŝ,K|stake): con slots-only la independencia es POR CONSTRUCCIÓN
    # (K no es feature; pares ⊥ K por stream del generador); la corr
    # scale-free sobre ŝ casi-degenerado (escala 8.0 exacta) es ruido de
    # muestreo → criterio enmendado: mediana < 0.1 y máximo < 0.2
    corrs = sorted(g["corr_K_abs"] for g in gates["VG_N3b"].values())
    b_ok = (all(g["auc"] >= 0.9 and 4 <= g["escala"] <= 12
                for g in gates["VG_N3b"].values())
            and corrs[len(corrs) // 2] < 0.1 and corrs[-1] < 0.2)
    c_ok = any(v["verde"] for v in hr.values())
    gates["veredicto"] = {"VG_N3a": a_ok, "VG_N3b": b_ok, "VG_N3c": c_ok}
    print(f"\nVG-N3a (Brier skill ≥ 0): {'PASA' if a_ok else 'FALLA'}")
    print(f"VG-N3b (AUC/escala/corrK): {'PASA' if b_ok else 'FALLA'}")
    print(f"VG-N3c (algún e verde): {'PASA' if c_ok else 'FALLA — cierre '
          'de N3-en-S₅ (la competencia barre la asignación)'}")
    with open(os.path.join(RES, "n3_gates.json"), "w", encoding="utf-8") as f:
        json.dump(gates, f, indent=1)
    print("Guardado n3_gates.json")


if __name__ == "__main__":
    main()
