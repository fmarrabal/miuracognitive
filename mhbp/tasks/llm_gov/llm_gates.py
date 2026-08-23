"""
Capítulo LLM — Gates VG-L1/L2/L3 sobre el CACHÉ (PREREG_LLM v2 §6). CPU.

  VG-L1  pendiente NETA del voto + fracción modal-erróneo + (p1−p2)
  VG-L2  headroom de VALOR ex-ante (DP en TOKENS, oráculo K+stake vs K-solo)
  VG-L3  el gate PRE-EVAL: techos por DP del POSTERIOR (parada óptima dada
         la fracción modal) con y sin valor → ¿subsume el posterior al
         valor (como en S₅, VG-B0) o no (la inversión predicha)?

Todo se computa como FRONTERAS payoff(coste en tokens) y se compara por
interpolación a presupuesto común — evita el problema de emparejar
políticas adaptativas con ex-ante.

  PYTHONPATH=. python -m mhbp.tasks.llm_gov.llm_gates
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CACHE = os.path.join(HERE, "cache")
N_GRID = [1, 3, 5, 7, 9, 11, 13, 15]
N_MAX = 11
R_MC = 2000         # réplicas del voto (subconjuntos SIN reemplazo)
R_PERM = 200        # permutaciones por instancia para el DP posterior
RNG = np.random.default_rng(20260808)


# ----------------------------- carga ----------------------------- #
def load_cache(split="sonda", seed=0, tag=""):
    suf = f"_{tag}" if tag else ""
    path = os.path.join(CACHE, f"cache_{split}{suf}_s{seed}.jsonl")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            resp = [m["resp"] for m in d["muestras"]]
            toks = np.array([m["tokens"] for m in d["muestras"]], dtype=float)
            out.append({"K": d["K"], "stake": float(d["stake"]),
                        "truth": d["truth"], "resp": resp,
                        "tok": toks, "t_med": float(np.mean(toks))})
    return out


# --------------------- voto i.i.d.: acc(n) exacta-MC --------------------- #
def vote_curve(inst, n_grid=N_GRID, R=R_MC, rng=RNG):
    """P(voto-de-n correcto), estimador INSESGADO (PREREG §3).

    FIX 2026-08-08 (panel de cierre, hallazgo crítico): antes se muestreaba
    CON reemplazo del pool (multinomial sobre frecuencias empíricas). Ese
    plug-in está sesgado a la BAJA y el sesgo crece con la diversidad del
    pool — ciega el estimador justo en el régimen donde la self-consistency
    funcionaría (verificado: con 8 clases de error el sesgo es −0.068).
    Correcto: subconjuntos SIN reemplazo (hipergeométrica multivariante);
    por intercambiabilidad, n muestras distintas del pool SON n draws
    i.i.d. de P ⇒ U-estadístico insesgado para n ≤ m."""
    resp = inst["resp"]
    m = len(resp)
    clases = {}
    for r in resp:
        clases[r] = clases.get(r, 0) + 1
    keys = list(clases.keys())                      # incluye None
    cuentas = np.array([clases[k] for k in keys])
    idx_none = keys.index(None) if None in keys else -1
    idx_ok = keys.index(inst["truth"]) if inst["truth"] in keys else -1
    out = []
    for n in n_grid:
        if idx_ok < 0 or n > m:
            out.append(0.0 if idx_ok < 0 else float("nan"))
            continue
        draws = rng.multivariate_hypergeometric(cuentas, n, size=R)  # (R, d)
        votos = draws.copy()
        if idx_none >= 0:
            votos[:, idx_none] = 0                   # los no-parse no votan
        mx = votos.max(axis=1)
        vivos = mx > 0
        empatados = (votos == mx[:, None]).sum(axis=1)
        gana_ok = (votos[:, idx_ok] == mx) & vivos
        pcorr = np.where(gana_ok, 1.0 / empatados, 0.0)
        out.append(float(pcorr.mean()))
    return np.array(out)


# ------------------------- tablas de clase (K, stake) ------------------- #
def class_tables(data, curves):
    ks = sorted({d["K"] for d in data})
    tab = {}
    for K in ks:
        for hi in (0, 1):
            sel = [i for i, d in enumerate(data)
                   if d["K"] == K and (d["stake"] > 1) == bool(hi)]
            if not sel:
                tab[(K, hi)] = None
                continue
            acc = np.mean([curves[i] for i in sel], axis=0)
            cost = np.array([n * np.mean([data[i]["t_med"] for i in sel])
                             for n in N_GRID])
            tab[(K, hi)] = {"acc": acc, "cost": cost, "w": len(sel) / len(data),
                            "stake": 8.0 if hi else 1.0}
    for K in ks:                                     # relleno si clase vacía
        for hi in (0, 1):
            if tab[(K, hi)] is None:
                tab[(K, hi)] = dict(tab[(K, 1 - hi)])
                tab[(K, hi)]["w"] = 0.0
                tab[(K, hi)]["stake"] = 8.0 if hi else 1.0
    return tab


def frontera_exante(tab, con_stake, lams):
    """Frontera (coste, payoff) del oráculo ex-ante por clase, con o sin
    conocimiento del stake (K siempre conocido)."""
    cls = list(tab.values())
    w = np.array([c["w"] for c in cls])
    w = w / w.sum()
    st = np.array([c["stake"] for c in cls])
    acc = np.stack([c["acc"] for c in cls])          # (C, n)
    cost = np.stack([c["cost"] for c in cls])
    val = (st[:, None] if con_stake else st.mean()) * acc
    pts = []
    for lam in lams:
        j = np.argmax(val - lam * cost, axis=1)
        c_tot = float((w * cost[np.arange(len(cls)), j]).sum())
        pay = float((w * st * acc[np.arange(len(cls)), j]).sum()
                    / (w * st).sum())
        pts.append((c_tot, pay))
    return sorted(set(pts))


# --------------------- DP del TECHO POSTERIOR (VG-L3) -------------------- #
def posterior_walks(data, curves, rng=RNG, R=R_PERM):
    """Camina permutaciones del pool: estado (k, c) = nº de muestras vistas
    y conteo modal (sobre parseadas). Devuelve, por instancia, arrays
    (R, N_MAX) de: estado c, acierto si se para ahí, coste acumulado."""
    walks = []
    for d in data:
        m = len(d["resp"])
        C = np.zeros((R, N_MAX), dtype=np.int16)
        OK = np.zeros((R, N_MAX))
        CT = np.zeros((R, N_MAX))
        for r in range(R):
            orden = rng.permutation(m)[:N_MAX]
            cuenta = {}
            coste = 0.0
            for k, i in enumerate(orden):
                coste += d["tok"][i]
                a = d["resp"][i]
                if a is not None:
                    cuenta[a] = cuenta.get(a, 0) + 1
                if cuenta:
                    mx = max(cuenta.values())
                    emp = [a for a, v in cuenta.items() if v == mx]
                    ok = (1.0 / len(emp)) if d["truth"] in emp else 0.0
                else:
                    mx, ok = 0, 0.0
                C[r, k] = mx
                OK[r, k] = ok
                CT[r, k] = coste
        walks.append({"c": C, "ok": OK, "ct": CT, "stake": d["stake"]})
    return walks


def dp_posterior(walks, idx_fit, con_stake, lams):
    """Backward induction sobre (k, conteo modal) [× stake si con_stake].
    Devuelve frontera (coste, payoff) evaluada en idx_eval."""
    grupos = [(1.0,), (8.0,)] if con_stake else [(1.0, 8.0)]
    pts = []
    for lam in lams:
        # --- política por grupo, ajustada en idx_fit --- #
        politica = {}
        for g in grupos:
            sel = [i for i in idx_fit if walks[i]["stake"] in g]
            if not sel:
                continue
            c = np.concatenate([walks[i]["c"] for i in sel])      # (N, K)
            ok = np.concatenate([walks[i]["ok"] for i in sel])
            ct = np.concatenate([walks[i]["ct"] for i in sel])
            stk = np.concatenate([np.full(walks[i]["c"].shape[0],
                                          walks[i]["stake"]) for i in sel])
            V = {}
            pol = {}
            for k in range(N_MAX - 1, -1, -1):
                for cc in range(0, k + 2):
                    mask = c[:, k] == cc
                    if mask.sum() < 5:
                        continue
                    parar = float((stk[mask] * ok[mask, k]).mean())
                    if k == N_MAX - 1:
                        V[(k, cc)] = parar
                        pol[(k, cc)] = True
                        continue
                    dc = float((ct[mask, k + 1] - ct[mask, k]).mean())
                    nxt = c[mask, k + 1]
                    seguir = -lam * dc
                    vals = [V.get((k + 1, int(x)), None) for x in nxt]
                    vals = [v for v in vals if v is not None]
                    seguir += float(np.mean(vals)) if vals else -1e9
                    V[(k, cc)] = max(parar, seguir)
                    pol[(k, cc)] = parar >= seguir
            politica[g] = pol
        # --- simular la política sobre TODAS las instancias --- #
        num, den, coste = 0.0, 0.0, 0.0
        n_w = 0
        for i, w in enumerate(walks):
            g = next(gg for gg in grupos if w["stake"] in gg)
            pol = politica.get(g, {})
            R = w["c"].shape[0]
            for r in range(R):
                k = 0
                while k < N_MAX - 1 and not pol.get((k, int(w["c"][r, k])), True):
                    k += 1
                num += w["stake"] * w["ok"][r, k]
                den += w["stake"]
                coste += w["ct"][r, k]
                n_w += 1
        pts.append((coste / n_w, num / den))
    return sorted(set(pts))


def techo_con_ic(curves, n_grid, nbar, p_hi, B=10000, rng=RNG):
    """Techo del headroom de VALOR con el n ASEQUIBLE y su IC BOOTSTRAP
    PAREADO por instancia (exigencia del panel de cierre: el gate se compara
    contra max(0.04, 4·SE) sobre la DIFERENCIA, no sobre los niveles).

      n_asequible = (n̄ − (1−p_hi)·1) / p_hi   [dar n=1 a las baratas]
      techo_i     = acc_i(n_asequible) − acc_i(n̄)
    """
    C = np.asarray(curves)                       # (I, n)
    ok = ~np.isnan(C).any(axis=0)
    xs = np.asarray(n_grid)[ok]
    n_af = min(float(xs.max()), (nbar - (1 - p_hi)) / p_hi)
    a_af = np.array([np.interp(n_af, xs, c[ok]) for c in C])
    a_nb = np.array([np.interp(nbar, xs, c[ok]) for c in C])
    d = a_af - a_nb
    I = len(d)
    idx = rng.integers(0, I, size=(B, I))
    boots = d[idx].mean(axis=1)
    se = float(boots.std(ddof=1))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    umbral = max(0.04, 4 * se)
    return {"n_asequible": n_af, "techo": float(d.mean()), "se": se,
            "ic95": [float(lo), float(hi)], "umbral": umbral,
            "pasa": bool(lo > umbral)}


def interp_frontera(pts, coste):
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    o = np.argsort(xs)
    return float(np.interp(coste, xs[o], ys[o]))


# ------------------------------- main ------------------------------- #
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--nbar", type=float, default=3.0)
    ap.add_argument("--p_hi", type=float, default=0.15)
    args = ap.parse_args()
    data = load_cache("sonda", 0, args.tag)
    print(f"caché: {len(data)} instancias, K={sorted({d['K'] for d in data})}, "
          f"p_hi={np.mean([d['stake'] > 1 for d in data]):.3f}", flush=True)
    curves = [vote_curve(d) for d in data]
    acc_pool = np.mean(curves, axis=0)
    out = {"n_grid": N_GRID, "acc_voto_global": acc_pool.tolist()}
    print("acc(voto-de-n): " +
          " ".join(f"n{n}:{a:.3f}" for n, a in zip(N_GRID, acc_pool)),
          flush=True)

    # ---- VG-L1: pendiente neta + modal-erróneo + (p1−p2) ---- #
    modal_err, gaps = [], []
    for d in data:
        cnt = {}
        for r in d["resp"]:
            if r is not None:
                cnt[r] = cnt.get(r, 0) + 1
        if not cnt:
            modal_err.append(1.0)
            gaps.append(0.0)
            continue
        srt = sorted(cnt.values(), reverse=True)
        p1 = srt[0] / len(d["resp"])
        p2 = (srt[1] / len(d["resp"])) if len(srt) > 1 else 0.0
        gaps.append(p1 - p2)
        modal = max(cnt, key=cnt.get)
        modal_err.append(float(modal != d["truth"]))
    pend = float(acc_pool[-1] - acc_pool[0])
    vg1 = {"pendiente_n1_n11": pend, "frac_modal_erroneo": float(np.mean(modal_err)),
           "mediana_p1_menos_p2": float(np.median(gaps)),
           "pasa": bool(pend > 0 and np.median(gaps) > 0)}
    out["VG_L1"] = vg1
    print(f"VG-L1: pendiente(n1→n11)={pend:+.3f} | modal-erróneo="
          f"{vg1['frac_modal_erroneo']:.3f} | mediana(p1−p2)="
          f"{vg1['mediana_p1_menos_p2']:.3f} → "
          f"{'PASA' if vg1['pasa'] else 'FALLA'}", flush=True)

    # ---- VG-L2 primario: techo con n ASEQUIBLE + IC PAREADO ---- #
    n_max_cache = min(len(d["resp"]) for d in data)
    grid_ok = [n for n in N_GRID if n <= n_max_cache]
    t = techo_con_ic([c[:len(grid_ok)] for c in curves], grid_ok,
                     args.nbar, args.p_hi)
    out["VG_L2_techo"] = t
    print(f"VG-L2 (techo con n asequible={t['n_asequible']:.1f}, "
          f"n̄={args.nbar:.0f}, p_hi={args.p_hi:.2f}): "
          f"{t['techo']:+.4f} IC95 [{t['ic95'][0]:+.4f}, {t['ic95'][1]:+.4f}] "
          f"SE={t['se']:.4f} umbral={t['umbral']:.4f} → "
          f"{'PASA' if t['pasa'] else 'NO PASA'}", flush=True)

    # ---- fronteras ex-ante (VG-L2 secundario) ---- #
    tab = class_tables(data, curves)
    lams = np.concatenate([[0.0], np.logspace(-6, -1.5, 60)])
    f_dif = frontera_exante(tab, False, lams)
    f_val = frontera_exante(tab, True, lams)
    t_med = float(np.mean([d["t_med"] for d in data]))
    presupuestos = [3 * t_med, 5 * t_med, 7 * t_med]
    vg2 = {}
    for b in presupuestos:
        hd = interp_frontera(f_val, b) - interp_frontera(f_dif, b)
        vg2[f"{b:.0f}"] = {"dificultad": interp_frontera(f_dif, b),
                           "valor": interp_frontera(f_val, b), "headroom": hd}
        print(f"VG-L2 @{b:.0f} tok: dificultad={vg2[f'{b:.0f}']['dificultad']:.3f} "
              f"valor={vg2[f'{b:.0f}']['valor']:.3f} → headroom={hd:+.4f}",
              flush=True)
    out["VG_L2"] = vg2

    # ---- VG-L3: techos posteriores por DP ---- #
    print("DP posterior (permutaciones + backward induction)...", flush=True)
    walks = posterior_walks(data, curves)
    idx = np.arange(len(data))
    fit = idx[idx % 2 == 0]
    lams_p = np.concatenate([[0.0], np.logspace(-6, -1.5, 40)])
    f_post = dp_posterior(walks, fit, False, lams_p)
    f_postval = dp_posterior(walks, fit, True, lams_p)
    vg3 = {}
    for b in presupuestos:
        p_post = interp_frontera(f_post, b)
        p_pv = interp_frontera(f_postval, b)
        p_val = interp_frontera(f_val, b)
        vg3[f"{b:.0f}"] = {"posterior": p_post, "posterior_valor": p_pv,
                           "exante_valor": p_val,
                           "headroom_valor_dado_posterior": p_pv - p_post,
                           "headroom_posterior": p_post - p_val}
        print(f"VG-L3 @{b:.0f} tok: posterior={p_post:.3f} "
              f"posterior+valor={p_pv:.3f} exante+valor={p_val:.3f} | "
              f"valor|posterior={p_pv - p_post:+.4f} "
              f"posterior−exante={p_post - p_val:+.4f}", flush=True)
    out["VG_L3"] = vg3
    out["fronteras"] = {"dificultad": f_dif, "valor": f_val,
                        "posterior": f_post, "posterior_valor": f_postval}
    with open(os.path.join(RES, "llm_gates.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("Guardado llm_gates.json", flush=True)


if __name__ == "__main__":
    main()
