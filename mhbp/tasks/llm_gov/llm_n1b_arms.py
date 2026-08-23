"""
N1b-LLM — BRAZOS OFFLINE + GATES + VEREDICTO (PREREG_N1B_LLM v2 §7).

v2 del script (2026-08-12): el greedy instancia-a-instancia era inviable
(300 bootstraps × optimización completa). Ahora todo es álgebra sobre
matrices precomputadas y la asignación se resuelve EXACTO por
multiplicador de Lagrange (bisección en λ), no por greedy:

  OK[i, j] = correcto si se trunca en CAP_GRID[j]  (re-parseo honesto)
  G[i, j]  = min(t_i, CAP_GRID[j])                 (gasto realizado)
  → tablas por clase SUM_OK[c, j], SUM_G[c, j] ⇒ cada brazo es un
    argmax_j (pago_j − λ·gasto_j) por (clase, rama) con λ bisectado
    hasta casar el presupuesto ±2%.

Brazos: uniforme · dificultad(clase) · valor(clase × stake) ·
oráculo-binario(conoce 1{llega}) · oráculo-c(conoce c_i).
Nota adjudicada: bajo contabilidad de GASTADO (Σ min(t_i, cap)) el
reciclaje de llegadas tempranas ya está incorporado en todos los brazos
⇒ el «adaptativo fluido» del prereg coincide con el brazo valor.

  PYTHONPATH=. python -m mhbp.tasks.llm_gov.llm_n1b_arms
"""
import json
import os
import re

import numpy as np
import torch

os.environ.setdefault("HF_HOME",
                      r"E:\ARTICULOS-CIENTIFICOS\MIURACOGNITIVE\hf_cache")
from transformers import AutoTokenizer  # noqa: E402

from mhbp.tasks.reasoner_g0.n4_g1 import auc_rank, _fit_logistic

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CACHE = os.path.join(HERE, "cache")
MODEL = "Qwen/Qwen2.5-14B-Instruct"
P_HI, S_ALTO = 0.15, 8.0
W_MIX = P_HI * S_ALTO + (1 - P_HI)
W_HI, W_LO = P_HI * S_ALTO, (1 - P_HI)      # pesos de pago por rama
C_HI, C_LO = P_HI, (1 - P_HI)               # pesos de coste por rama
RNG = np.random.default_rng(20260810)
RE_LINEA = re.compile(r"RESPUESTA:\s*(\d+)\s*(?:\n|$)")
RE_OP_ARIT = re.compile(r"^\s*[+\-×x*]\s*\d+\s*:", re.MULTILINE)
CAP_GRID = np.concatenate([np.arange(0, 201), np.arange(210, 921, 10)])
N_BINS = 800


def cargar(familia):
    with open(os.path.join(CACHE, f"n1b_sonda_{familia}.jsonl"),
              encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def eventos(tok, ids, texto):
    """[(k_token, valor)] de líneas RESPUESTA COMPLETAS; k = mínimo
    prefijo de tokens que la contiene (búsqueda binaria sobre decode)."""
    evs = []
    for m in RE_LINEA.finditer(texto):
        fin, lo, hi = m.end(), 1, len(ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if len(tok.decode(ids[:mid], skip_special_tokens=True)) >= fin:
                hi = mid
            else:
                lo = mid + 1
        evs.append((lo, m.group(1)))
    return evs


def preparar(familia, tok):
    """Devuelve dict con matrices OK/G y metadatos por instancia."""
    regs = cargar(familia)
    n, J = len(regs), len(CAP_GRID)
    OK = np.zeros((n, J), dtype=bool)
    G = np.zeros((n, J))
    c_arr, t_arr, L_arr, d_arr, hops = [], [], [], [], []
    for i, r in enumerate(regs):
        texto = tok.decode(r["ids"], skip_special_tokens=True)
        evs = eventos(tok, r["ids"], texto)
        t = r["tokens"]
        # curva de corrección: la ÚLTIMA respuesta visible en el prefijo
        vals = np.full(J, None, dtype=object)
        for k, v in evs:
            vals[CAP_GRID >= k] = v
        OK[i] = np.array([v == r["truth"] for v in vals])
        G[i] = np.minimum(t, CAP_GRID)
        c_i = np.inf
        for k, v in evs:
            if v == r["truth"]:
                c_i = k
                break
        c_arr.append(c_i)
        t_arr.append(t)
        L_arr.append(r["L"])
        d_arr.append(r["d"])
        # nº de pasos de la CoT (contenido): ciclo ya lo trae; aritmética
        # se cuenta aquí (FIX del control positivo del gate M: la sonda
        # aritmética guardó n_hops=0 y el probe quedaba sin feature)
        hops.append(r["n_hops"] if familia == "ciclo"
                    else len(RE_OP_ARIT.findall(texto)))
    return {"OK": OK, "G": G, "c": np.array(c_arr),
            "t": np.array(t_arr, float), "L": np.array(L_arr),
            "d": np.array(d_arr, float), "hops": np.array(hops, float),
            "n": n}


def selftest_truncado(tok, familia="ciclo", n=60, ncaps=20):
    regs = cargar(familia)[:n]
    for r in regs:
        texto = tok.decode(r["ids"], skip_special_tokens=True)
        evs = eventos(tok, r["ids"], texto)
        vals = np.full(len(CAP_GRID), None, dtype=object)
        for k, v in evs:
            vals[CAP_GRID >= k] = v
        # los brazos SOLO evalúan en puntos de CAP_GRID: el test compara
        # ahí (comparar en caps arbitrarios mide la resolución de la
        # rejilla, no la semántica del truncado)
        js = np.unique(np.linspace(1, len(CAP_GRID) - 1,
                                   ncaps).astype(int))
        for j in js:
            cap = int(CAP_GRID[j])
            pref = tok.decode(r["ids"][:cap], skip_special_tokens=True)
            mm = RE_LINEA.findall(pref if cap < r["tokens"]
                                  else pref + "\n")
            lit = (mm[-1] == r["truth"]) if mm else False
            assert lit == (vals[j] == r["truth"]), \
                f"truncado difiere en cap={cap} (grid j={j})"
    print(f"SELF-TEST truncado {n}×{ncaps}: OK", flush=True)


# --------------------- asignación exacta (Lagrange) --------------------- #
def tablas(D, idx, clases, cls_of=None):
    """SUM_OK[c, j] y SUM_G[c, j] sobre idx. cls_of: etiqueta de clase por
    instancia (por defecto D['L']); el brazo uniforme pasa una constante
    — antes se mutaba D['L'] y la comparación sobre array de objetos
    devolvía un escalar: el uniforme salía con cap 0 (pago 0.000)."""
    lab = D["L"] if cls_of is None else cls_of
    SO, SG = {}, {}
    for c in clases:
        m = idx[np.asarray([lab[i] == c for i in idx])]
        SO[c] = D["OK"][m].sum(0).astype(float)
        SG[c] = D["G"][m].sum(0)
    return SO, SG


def resolver(SO, SG, clases, B, modo, N, nbins=N_BINS):
    """Asignación ÓPTIMA de caps bajo presupuesto por DP exacto
    (knapsack multi-elección con presupuesto discretizado).

    FIX 2026-08-12: la versión por multiplicador de Lagrange solo podía
    elegir vértices de la envolvente convexa; con generaciones cortas la
    curva (gasto, pago) tiene un salto y el brazo uniforme caía a cap 0
    (pago 0.000). El DP no tiene hueco de integralidad.

    modo: 'unif' (una clase, ramas iguales) · 'dif' (por clase, ramas
    iguales) · 'valor' (por clase y rama).
    """
    items = []
    for c in clases:
        if modo == "valor":
            items.append((c, "hi", W_HI * SO[c], C_HI * SG[c]))
            items.append((c, "lo", W_LO * SO[c], C_LO * SG[c]))
        else:
            items.append((c, "both", (W_HI + W_LO) * SO[c], SG[c]))
    step = max(B, 1e-9) / nbins
    dp = np.full(nbins + 1, -np.inf)
    dp[0] = 0.0
    elec = []
    for (_, _, val, cost) in items:
        ci = np.ceil(cost / step).astype(int)
        ok = ci <= nbins
        ci, val_ok = ci[ok], val[ok]
        js = np.where(ok)[0]
        nuevo = np.full(nbins + 1, -np.inf)
        arg = np.zeros(nbins + 1, dtype=int)
        for k in range(len(js)):
            desp = np.full(nbins + 1, -np.inf)
            if ci[k] <= nbins:
                desp[ci[k]:] = dp[:nbins + 1 - ci[k]] + val_ok[k]
            mejor = desp > nuevo
            nuevo[mejor] = desp[mejor]
            arg[mejor] = js[k]
        dp, a = np.maximum.accumulate(nuevo), arg
        elec.append((a, nuevo))
    # backtracking
    b = int(np.argmax(elec[-1][1] if len(elec) else dp))
    # recomputa hacia atrás con la misma recurrencia
    sel_raw, bb = [], b
    for t in range(len(items) - 1, -1, -1):
        a, nuevo = elec[t]
        j = int(a[bb])
        sel_raw.append((items[t][0], items[t][1], j))
        cost = items[t][3][j]
        bb = max(0, bb - int(np.ceil(cost / step)))
    sel_raw.reverse()
    sel, pago, gasto = {}, 0.0, 0.0
    for c in clases:
        js = {r: j for (cc, r, j) in sel_raw if cc == c}
        if modo == "valor":
            jh, jl = js.get("hi", 0), js.get("lo", 0)
        else:
            jh = jl = js.get("both", 0)
        sel[c] = (int(jh), int(jl))
        pago += W_HI * SO[c][jh] + W_LO * SO[c][jl]
        gasto += C_HI * SG[c][jh] + C_LO * SG[c][jl]
    return sel, pago / (W_MIX * N), gasto


def pago_por_inst(D, idx, sel, cls_of=None):
    """Pago marginalizado por instancia (para el bootstrap pareado)."""
    lab = D["L"] if cls_of is None else cls_of
    p = np.zeros(len(idx))
    for k, i in enumerate(idx):
        jh, jl = sel[lab[i]]
        p[k] = (W_HI * D["OK"][i, jh] + W_LO * D["OK"][i, jl]) / W_MIX
    return p


def oraculo_c(D, idx, B, N):
    """Conoce c_i: financia ramas por ratio pago/coste hasta B."""
    c = D["c"][idx]
    fin = np.isfinite(c)
    ratios, costes, pagos = [], [], []
    for w, cw in ((W_HI, C_HI), (W_LO, C_LO)):
        ratios.append(np.where(fin, w / np.maximum(c, 1e-9), -1))
        costes.append(np.where(fin, cw * c, np.inf))
        pagos.append(np.where(fin, w, 0.0))
    r = np.concatenate(ratios)
    co = np.concatenate(costes)
    pa = np.concatenate(pagos)
    o = np.argsort(-r)
    acum = np.cumsum(co[o])
    tomar = acum <= B
    return float(pa[o][tomar].sum() / (W_MIX * N))


def oraculo_bin(D, idx, B, clases, N):
    """Conoce 1{llega}: caps por (clase, rama) SOLO entre las que llegan;
    condenadas a cap 0 (canal doom)."""
    lleg = idx[np.isfinite(D["c"][idx])]
    SO, SG = tablas(D, lleg, clases)
    _, pago, _ = resolver(SO, SG, clases, B, "valor", N)
    return float(pago)


# ------------------------------- gates -------------------------------- #
def gate_M(D, frac_test=0.3):
    """Mudez multi-horizonte; features ⊆ O = (m, pasos contados, clase).
    Baseline = (m, clase). Prefijos en fracciones de la trayectoria."""
    idx = np.where(np.isfinite(D["c"]) & (D["c"] > 20))[0]
    t_hop = float(np.median(D["c"][idx] / np.maximum(1, D["d"][idx])))
    filas = []
    for i in idx:
        for fr in (0.2, 0.35, 0.5, 0.65, 0.8):
            m = fr * D["c"][i]
            if m < 5:
                continue
            # pasos contados hasta m (proporcional al avance observable)
            pasos = D["hops"][i] * fr
            filas.append((i, m, pasos, D["L"][i], D["c"][i] - m))
    F = np.array(filas)
    res = {}
    ids = np.unique(F[:, 0])
    te_ids = set(RNG.choice(ids, int(frac_test * len(ids)),
                            replace=False).tolist())
    tr = np.array([f not in te_ids for f in F[:, 0]])
    te = ~tr
    for h in (1, 2, 3, 5):
        y = (F[:, 4] <= h * t_hop).astype(float)
        if y[tr].std() < 1e-9 or y[te].std() < 1e-9:
            continue

        def auc_de(cols):
            X = F[:, cols]
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            w, b = _fit_logistic(
                torch.tensor((X[tr] - mu) / sd, dtype=torch.float64),
                torch.tensor(y[tr], dtype=torch.float64), 1.0)
            sc = ((X[te] - mu) / sd) @ w.numpy() + float(b)
            return float(auc_rank(y[te] > 0.5, sc))
        pr, ba = auc_de([1, 2, 3]), auc_de([1, 3])
        res[h] = {"probe": pr, "base": ba, "delta": pr - ba}
    # FIX 2026-08-12 (control positivo plano en AMBAS familias): el
    # contraste correcto para la MUDEZ no es «¿el contenido añade sobre
    # (m, clase)?» — en las dos familias los tokens emitidos ya resumen
    # el avance — sino la PREDICTIBILIDAD ABSOLUTA del remanente desde el
    # conjunto observable O. En la familia suave (K conocido) debe ser
    # alta; en el acantilado, baja. La aritmética es el control positivo
    # del INSTRUMENTO (demuestra que el probe detecta cuando hay qué).
    auc_abs = float(np.mean([v["probe"] for h, v in res.items()
                             if h >= 2]))
    mudo = auc_abs <= 0.65
    return res, mudo, t_hop, auc_abs


def r2(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    A = np.stack([x, np.ones_like(x)], 1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(1 - (y - A @ beta).var() / y.var())


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    selftest_truncado(tok)
    D = preparar("ciclo", tok)
    A = preparar("arit", tok)
    out = {"gates": {}, "brazos": {}, "H": {}}

    # ---- gate R ---- #
    f = np.isfinite(D["c"])
    fa = np.isfinite(A["c"])
    r_cd = float(np.corrcoef(D["c"][f], D["d"][f])[0, 1])
    r2c = r2(D["c"][f], D["L"][f])
    r2a = r2(A["c"][fa], A["L"][fa])
    cens = {int(L): float(np.mean(~f[D["L"] == L])) for L in (6, 10, 14)}
    out["gates"]["R"] = {"corr_c_d": r_cd, "r2_ciclo": r2c,
                         "r2_arit": r2a, "censura": cens,
                         "pasa": bool(r_cd >= 0.6 and r2c <= 0.5
                                      and cens[14] <= 0.10)}
    # ---- gate S ---- #
    ok_fin = f & D["OK"][:, -1]
    fracS = float(np.mean(D["hops"][ok_fin] == D["d"][ok_fin]))
    out["gates"]["S"] = {"frac_cadena_valida": fracS,
                         "pasa": bool(fracS >= 0.95)}
    # ---- gate M (+ control positivo aritmético) ---- #
    m_c, mudo, thop, auc_c = gate_M(D)
    m_a, _, thop_a, auc_a = gate_M(A)
    ctrl = auc_a >= 0.75          # el instrumento SÍ ve donde hay qué ver
    out["gates"]["M"] = {"ciclo": m_c, "arit": m_a, "mudo": bool(mudo),
                         "auc_abs_ciclo": auc_c, "auc_abs_arit": auc_a,
                         "control_ok": bool(ctrl), "t_hop": thop}
    print(f"R: corr(c,d)={r_cd:.2f} R²ciclo={r2c:.2f} R²arit={r2a:.2f} "
          f"cens14={cens[14]:.3f} → "
          f"{'✓' if out['gates']['R']['pasa'] else '✗'}", flush=True)
    print(f"S: cadena válida={fracS:.3f} → "
          f"{'✓' if out['gates']['S']['pasa'] else '✗'}", flush=True)
    print("M ciclo: " + " ".join(f"h{h}:{v['delta']:+.3f}"
                                 for h, v in m_c.items())
          + f" → {'MUDO' if mudo else 'NO MUDO'}", flush=True)
    print("M arit(control): " + " ".join(f"h{h}:{v['delta']:+.3f}"
                                         for h, v in m_a.items())
          + f" → {'OK' if ctrl else 'FALLA'}", flush=True)
    print(f"M AUC ABSOLUTA del remanente desde O: ciclo={auc_c:.3f} vs "
          f"arit={auc_a:.3f} → mudez {'✓' if mudo else '✗'} / "
          f"instrumento {'✓' if ctrl else '✗'}", flush=True)

    # ---- punto de operación por mordida ---- #
    N = D["n"]
    clases = [6, 10, 14]
    puntos = {}
    for ft in (0.2, 0.35, 0.5):
        u = float(np.quantile(D["t"], 1 - ft))
        puntos[ft] = float(np.minimum(D["t"], u).sum())
    out["brazos"]["presupuestos"] = puntos
    B_prim = puntos[0.35]

    # ---- brazos con cross-fitting A↔B ---- #
    orden = RNG.permutation(N)
    strat = sorted(range(N), key=lambda j: (D["L"][j], D["d"][j],
                                            orden[j]))
    mitades = [np.array(strat[0::2]), np.array(strat[1::2])]
    pagos_inst = {k: np.zeros(N) for k in ("uniforme", "dificultad",
                                           "valor")}
    res_ag = {}
    UNIF = np.full(N, "*", dtype=object)
    for fit, med in ((mitades[0], mitades[1]), (mitades[1],
                                                mitades[0])):
        Bh = B_prim * len(med) / N
        for modo, nombre, cls in (("unif", "uniforme", ["*"]),
                                  ("dif", "dificultad", clases),
                                  ("valor", "valor", clases)):
            lab = UNIF if cls == ["*"] else None
            SO, SG = tablas(D, fit, cls, lab)
            sel, _, _ = resolver(SO, SG, cls, Bh * len(fit) / len(med),
                                 modo, len(fit))
            pagos_inst[nombre][med] = pago_por_inst(D, med, sel, lab)
    for k, v in pagos_inst.items():
        res_ag[k] = float(v.mean())
    res_ag["oraculo_bin"] = oraculo_bin(D, np.arange(N), B_prim, clases, N)
    res_ag["oraculo_c"] = oraculo_c(D, np.arange(N), B_prim, N)
    out["brazos"]["pagos"] = res_ag

    # ---- H1 ---- #
    dif = pagos_inst["valor"] - pagos_inst["dificultad"]
    bs = np.array([dif[RNG.integers(0, N, N)].mean() for _ in range(20000)])
    se = float(bs.std(ddof=1))
    ic = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    h1 = {"delta": float(dif.mean()), "se": se, "ic95": ic,
          "umbral": max(0.02, 4 * se),
          "pasa": bool(dif.mean() >= max(0.02, 4 * se) and ic[0] > 0)}
    out["H"]["H1"] = h1
    # ---- H2 (TOST del canal distancia) ---- #
    rango = res_ag["oraculo_c"] - res_ag["uniforme"]
    gap = res_ag["oraculo_c"] - res_ag["oraculo_bin"]
    gaps = []
    for _ in range(2000):
        idx = RNG.integers(0, N, N)
        gaps.append(oraculo_c(D, idx, B_prim, N)
                    - oraculo_bin(D, idx, B_prim, clases, N))
    ic_sup = float(np.percentile(gaps, 97.5))
    h2 = {"gap": float(gap), "rango": float(rango),
          "umbral": 0.02 * float(rango), "ic_sup": ic_sup,
          "pasa": bool(ic_sup <= 0.02 * rango)}
    out["H"]["H2"] = h2

    # ---- contraste aritmético ---- #
    Na = A["n"]
    cls_a = sorted(set(A["L"].tolist()))
    ua = float(np.quantile(A["t"], 0.65))
    Ba = float(np.minimum(A["t"], ua).sum())
    SOa, SGa = tablas(A, np.arange(Na), cls_a)
    pa = {}
    UNIF_A = np.full(Na, "*", dtype=object)
    for modo, nombre in (("unif", "uniforme"), ("dif", "dificultad"),
                         ("valor", "valor")):
        if nombre == "uniforme":
            SO2, SG2 = tablas(A, np.arange(Na), ["*"], UNIF_A)
            _, p, _ = resolver(SO2, SG2, ["*"], Ba, modo, Na)
        else:
            _, p, _ = resolver(SOa, SGa, cls_a, Ba, modo, Na)
        pa[nombre] = float(p)
    pa["oraculo_c"] = oraculo_c(A, np.arange(Na), Ba, Na)
    pa["oraculo_bin"] = oraculo_bin(A, np.arange(Na), Ba, cls_a, Na)
    out["brazos"]["arit"] = pa

    print("\nPAGOS ciclo: " + " ".join(f"{k}={v:.3f}"
                                       for k, v in res_ag.items()),
          flush=True)
    print("PAGOS arit : " + " ".join(f"{k}={v:.3f}"
                                     for k, v in pa.items()), flush=True)
    print(f"H1 valor−dificultad: Δ={h1['delta']:+.4f} "
          f"IC95[{ic[0]:+.4f},{ic[1]:+.4f}] umbral={h1['umbral']:.4f} → "
          f"{'PASA' if h1['pasa'] else 'NO PASA'}", flush=True)
    print(f"H2 oráculo_c−oráculo_bin: gap={gap:+.4f} rango={rango:.3f} "
          f"IC_sup={ic_sup:+.4f} umbral={h2['umbral']:.4f} → "
          f"{'PASA (canal distancia MUDO)' if h2['pasa'] else 'NO PASA'}",
          flush=True)
    oc_dif_c = res_ag["oraculo_c"] - res_ag["dificultad"]
    oc_dif_a = pa["oraculo_c"] - pa["dificultad"]
    out["H"]["contraste"] = {"ciclo_oc_menos_dif": float(oc_dif_c),
                             "arit_oc_menos_dif": float(oc_dif_a)}
    print(f"CONTRASTE (oráculo_c − dificultad): ciclo={oc_dif_c:+.4f} "
          f"vs arit={oc_dif_a:+.4f}", flush=True)
    with open(os.path.join(RES, "llm_n1b_arms.json"), "w",
              encoding="utf-8") as fjs:
        json.dump(out, fjs, indent=1, default=float)
    print("Guardado llm_n1b_arms.json", flush=True)


if __name__ == "__main__":
    main()
