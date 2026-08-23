"""
Revisión TMLR — ICs bootstrap para el hallazgo post-hoc del acantilado:
la fracción ex-ante del rango (0.759 vs 0.764) y la ratio de rangos (×5).

Convención de asignación CONGELADA (declarada en el prereg v2 §7.6): las
caps de cada brazo se calculan UNA vez sobre los datos completos; el
bootstrap remuestrea instancias y re-evalúa los pagos con esas caps
fijas. El IC mide la incertidumbre de muestreo del resultado dada la
política ejecutada.

  PYTHONPATH=. python -m mhpb… → python -m mhbp.tasks.llm_gov.llm_ic_fracciones
"""
import json
import os

import numpy as np

os.environ.setdefault("HF_HOME",
                      r"E:\ARTICULOS-CIENTIFICOS\MIURACOGNITIVE\hf_cache")
from transformers import AutoTokenizer  # noqa: E402

from .llm_n1b_arms import (CAP_GRID, MODEL, W_HI, W_LO, W_MIX, oraculo_c,
                           preparar, resolver, tablas)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
B = 2000
RNG = np.random.default_rng(20260817)


def pagos_con_caps(D, idx, sel, lab):
    p = 0.0
    for i in idx:
        jh, jl = sel[lab[i] if lab is not None else D["L"][i]]
        p += W_HI * D["OK"][i, jh] + W_LO * D["OK"][i, jl]
    return p / (W_MIX * len(idx))


def preparar_familia(D, clases, frac_trunc=0.35):
    """Caps congeladas de uniforme y valor + presupuesto primario."""
    N = D["n"]
    u = float(np.quantile(D["t"], 1 - frac_trunc))
    Bp = float(np.minimum(D["t"], u).sum())
    idx = np.arange(N)
    UNIF = np.full(N, "*", dtype=object)
    SOu, SGu = tablas(D, idx, ["*"], UNIF)
    sel_u, _, _ = resolver(SOu, SGu, ["*"], Bp, "unif", N)
    SO, SG = tablas(D, idx, clases)
    sel_v, _, _ = resolver(SO, SG, clases, Bp, "valor", N)
    return {"Bp": Bp, "sel_u": sel_u, "sel_v": sel_v, "UNIF": UNIF,
            "clases": clases}


def medidas(D, cfg, idx):
    p_u = pagos_con_caps(D, idx, cfg["sel_u"], cfg["UNIF"])
    p_v = pagos_con_caps(D, idx, cfg["sel_v"], None)
    # oráculo-c se re-resuelve por réplica (conoce c_i: su política ES
    # función de la muestra; congelarlo lo debilitaría artificialmente)
    B_r = cfg["Bp"] * len(idx) / D["n"]
    p_oc = oraculo_c(D, np.asarray(idx), B_r, len(idx))
    rango = p_oc - p_u
    frac = (p_v - p_u) / rango if rango > 1e-9 else np.nan
    return frac, rango


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    Dc = preparar("ciclo", tok)
    Da = preparar("arit", tok)
    cc = preparar_familia(Dc, [6, 10, 14])
    ca = preparar_familia(Da, sorted(set(Da["L"].tolist())))
    f_c, r_c = medidas(Dc, cc, np.arange(Dc["n"]))
    f_a, r_a = medidas(Da, ca, np.arange(Da["n"]))
    print(f"puntuales: frac_cliff={f_c:.3f} frac_suave={f_a:.3f} "
          f"rangos={r_c:.3f}/{r_a:.3f} ratio={r_c / r_a:.2f}", flush=True)
    fr_c, fr_a, ratios, difs = [], [], [], []
    for r in range(B):
        ic_ = RNG.integers(0, Dc["n"], Dc["n"])
        ia_ = RNG.integers(0, Da["n"], Da["n"])
        fc, rc = medidas(Dc, cc, ic_)
        fa, ra = medidas(Da, ca, ia_)
        fr_c.append(fc)
        fr_a.append(fa)
        ratios.append(rc / ra if ra > 1e-9 else np.nan)
        difs.append(fc - fa)
        if (r + 1) % 500 == 0:
            print(f"[{r + 1}/{B}]", flush=True)
    def ic(v):
        v = np.array(v)
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    res = {"frac_cliff": {"punto": float(f_c), "ic95": ic(fr_c)},
           "frac_suave": {"punto": float(f_a), "ic95": ic(fr_a)},
           "dif_fracciones": {"punto": float(f_c - f_a), "ic95": ic(difs)},
           "ratio_rangos": {"punto": float(r_c / r_a), "ic95": ic(ratios)},
           "B": B}
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"{k}: {v['punto']:+.3f} IC95 [{v['ic95'][0]:+.3f}, "
                  f"{v['ic95'][1]:+.3f}]", flush=True)
    with open(os.path.join(RES, "llm_ic_fracciones.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("Guardado llm_ic_fracciones.json", flush=True)


if __name__ == "__main__":
    main()
