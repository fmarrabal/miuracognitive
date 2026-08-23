"""
Fase 2b — Contrastes (PREREG_PHASE2B.md v2): campo-modulador vs campo-fuente.

Familia Holm-3 (con dirección explícita en tabla):
  D1  mhbp_gov vs mhbp     (conductual: ¿desaparece el fallo OOD del rol fuente?)
  D2b mhbp_gov vs gru_gov  (LA física del campo aislada de la memoria genérica)
  D3  mhbp_gov vs gru      (contexto)
Etiquetados fuera de Holm:
  D2a mhbp_gov vs react    (modulación+memoria+capacidad, confundido — panel)
  E1  react vs mlp         (gate de equivalencia del reactivo de referencia)
+ mecanismos (plan/hard con λ de EnvConfig) + intervención g≡1 (eval_mod_off).

  python -m mhbp.analysis.phase2b_report
"""
from __future__ import annotations
import glob
import json
import os

import numpy as np

from .phase2_report import RES, paired, contrast_stats, holm
from ..tasks.synthetic_multiscale.env import EnvConfig

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CFG = EnvConfig()
EXPECTED_STEPS = 500

HOLM_FAMILY = [("D1_gov_vs_mhbp", "mhbp_gov", "mhbp", "esperado gov MEJOR (mecanismo)"),
               ("D2b_gov_vs_grugov", "mhbp_gov", "gru_gov", "física del campo aislada"),
               ("D3_gov_vs_gru", "mhbp_gov", "gru", "contexto")]
LABELED = [("D2a_gov_vs_react", "mhbp_gov", "react", "CONFUNDIDO: modulación+memoria+capacidad"),
           ("E1_react_vs_mlp", "react", "mlp", "gate de equivalencia del reactivo")]

R = []
def w(s=""):
    R.append(s)


def load_2b():
    """Como phase2_report.load() pero persistiendo los avisos en el informe y
    verificando la homogeneidad del fingerprint del entorno (panel 2b)."""
    runs, skipped, fps = {}, [], {}
    for fp in glob.glob(os.path.join(RES, "*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("steps") != EXPECTED_STEPS:
                skipped.append((os.path.basename(fp), f"steps={d.get('steps')}"))
                continue
            runs.setdefault(d["controller"], {})[d["seed"]] = d
            fps.setdefault(d.get("env_fingerprint", "SIN_FP(fase2)"), 0)
            fps[d.get("env_fingerprint", "SIN_FP(fase2)")] += 1
        except Exception as e:
            skipped.append((os.path.basename(fp), repr(e)[:60]))
    return runs, skipped, fps


def main():
    runs, skipped, fps = load_2b()
    w("# Fase 2b — Campo-modulador vs campo-fuente (PREREG v2)\n")
    if skipped:
        w(f"**AVISO (persistido):** {len(skipped)} JSON saltados:")
        for name, why in skipped[:10]:
            w(f"- `{name}`: {why}")
        w("")
    w(f"Huellas de entorno presentes: {fps} — las corridas 'SIN_FP' son de la "
      "Fase 2 (anteriores al fingerprint); la identidad del entorno entre fases "
      "está garantizada por registro de sesión (sin ediciones de env.py entre "
      "el confirmatorio F2 y el F2b), no por hash.\n")

    for c in ("mhbp_gov", "gru_gov", "react", "mhbp", "gru", "mlp"):
        if c not in runs:
            w(f"_(pendiente: {c} sin corridas)_")
            continue
        vs = runs[c]
        j = [d["eval"]["ood_primary_J"] for d in vs.values()]
        ji = [d["eval"]["iid"]["J_mean"] for d in vs.values()]
        d0 = next(iter(vs.values()))
        w(f"- **{c}** (n={len(j)}): J_OOD={np.mean(j):.4f}±{np.std(j):.4f}  "
          f"J_iid={np.mean(ji):.4f}  core={d0['params_core']}  total={d0['params_total']}")
    w("")

    def table(rows_def, holm_family=True):
        stats_rows, pvals = [], []
        for name, a, b, note in rows_def:
            _, xa, xb = paired(runs, a, b)
            st = contrast_stats(xa, xb) if len(xa) >= 3 else {"n": len(xa)}
            stats_rows.append((name, st, note))
            pvals.append(st.get("p_t", 1.0))
        surv = holm(pvals) if (holm_family and pvals) else [False] * len(pvals)
        w("| contraste | n | Δ(J_OOD) | dir. | t | p | p_wx | dz | IC95 | Holm | nota |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for (name, st, note), sv in zip(stats_rows, surv):
            if st.get("n", 0) < 3:
                w(f"| {name} | {st.get('n', 0)} | — | — | — | — | — | — | — | — | {note} |")
                continue
            direc = "gov mejor" if st["mean_diff"] < 0 else "gov PEOR"
            if name.startswith("E1"):
                direc = "react mejor" if st["mean_diff"] < 0 else "react peor"
            hol = ("SÍ" if sv else "no") if holm_family else "(fuera)"
            if sv and name == "D1_gov_vs_mhbp" and st["mean_diff"] > 0:
                hol = "SÍ (dir. CONTRARIA)"
            w(f"| {name} | {st['n']} | {st['mean_diff']:+.4f} | {direc} | {st['t']:.2f} "
              f"| {st['p_t']:.4f} | {st['p_wilcoxon']:.4f} | {st['dz']:.2f} "
              f"| [{st['ci95'][0]:+.4f}, {st['ci95'][1]:+.4f}] | {hol} | {note} |")

    w("## Familia Holm-3 (pareado por semilla; dirección EXPLÍCITA)\n")
    table(HOLM_FAMILY, holm_family=True)
    w("")
    w("## Contrastes etiquetados (fuera de Holm)\n")
    table(LABELED, holm_family=False)
    w("")

    # mecanismos con λ de EnvConfig (no hardcodeados — panel)
    w("## Mecanismos (términos del fallo de la F2; umbrales pre-registrados:")
    w(f"plan·λ ≤ 1.0 y hard·λ ≤ 5.0 = 'colapsa a baseline')\n")
    w("| controlador | plan(budget_hi)·λ | hard(e4)·λ | settling(vent.) |")
    w("|---|---|---|---|")
    for c in ("mhbp", "mhbp_gov", "gru_gov", "react", "gru", "mlp"):
        if c not in runs:
            continue
        vs = list(runs[c].values())
        plan = np.mean([d["eval"]["ood_budget_hi"]["plan"] for d in vs]) * _CFG.lam_plan
        hard = np.mean([d["eval"]["e4_step"]["hard"] for d in vs]) * _CFG.lam_hard
        setl = np.mean([d["eval"]["e4_step"].get("settling_windows_mean", np.nan)
                        for d in vs])
        w(f"| {c} | {plan:.2f} | {hard:.2f} | {setl:.2f} |")
    w("")

    # intervención causal g≡1 (eval_mod_off, pareada por episodio por construcción)
    w("## Intervención g≡1 (¿la modulación es load-bearing?)\n")
    w("| controlador | J_OOD (normal) | J_OOD (g≡1) | Δ | lectura |")
    w("|---|---|---|---|---|")
    for c in ("mhbp_gov", "gru_gov"):
        if c not in runs:
            continue
        vs = [d for d in runs[c].values() if "eval_mod_off" in d]
        if not vs:
            w(f"| {c} | — | — | — | sin eval_mod_off |")
            continue
        jn = np.mean([d["eval"]["ood_primary_J"] for d in vs])
        jo = np.mean([d["eval_mod_off"]["ood_primary_J"] for d in vs])
        delta = jn - jo
        lect = ("modulación AYUDA" if delta < -1e-3 else
                ("modulación DAÑA" if delta > 1e-3 else "inerte"))
        w(f"| {c} | {jn:.4f} | {jo:.4f} | {delta:+.4f} | {lect} |")
    w("")

    out = os.path.join(HERE, "analysis", "REPORT_PHASE2B.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(R))
    print("\n".join(R))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
