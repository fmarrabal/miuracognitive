"""
G0 — Veredicto contra los umbrales pre-registrados (PREREG_G0.md).

T: gap acc-largo (recurrente − vanilla) >= 0.15, régimen indist, por genset.
M1: trend_spearman >= 0.5  Y  F_final >= 0.8 × output_acc.
M2: (follow_donor + rederive_cost) >= 0.6  Y  free_revert <= 0.3 (media de t*).
M3: corr(fidelidad, acierto OOD) >= 0.2.
I : AUC(p_stop, corrección) >= 0.65 en >= 1 genset.
P : corr_K_niter_ood >= 0.10 (hbp_full, del JSON de train).
Gate por (variant, genset): PASS si T∧M1∧M2∧M3∧I∧P; global: PASS en >=2/3 seeds.

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.g0_report
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEEDS = [0, 1, 2]
GENS = ["adjacent", "cycle_transp"]

R = []
def w(s=""):
    R.append(s)


def jload(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    w("# G0 — Veredicto (batería T-M-I-P sobre el reasoner de MiuraCognitive)\n")
    verdicts = {}
    for variant in ("gating_wm", "hbp_full"):
        for gens in GENS:
            w(f"## {variant} × {gens}\n")
            w("| seed | T gap | M1 trend | M1 F_fin/acc | M2 f+r | M2 free | M3 corr | I AUC | P corrOOD | GATE |")
            w("|---|---|---|---|---|---|---|---|---|---|")
            passes = []
            for seed in SEEDS:
                tr_i = jload(f"{variant}_{gens}_indist_s{seed}.json")
                tr_v = jload(f"vanilla_{gens}_indist_s{seed}.json")
                tr_o = jload(f"{variant}_{gens}_ood_s{seed}.json")
                mip = jload(f"mip_{variant}_{gens}_s{seed}.json")
                if not all((tr_i, tr_v, tr_o, mip)):
                    w(f"| {seed} | (celdas incompletas) |||||||||")
                    continue
                gap = tr_i["final_acc"]["largo"] - tr_v["final_acc"]["largo"]
                m1 = mip["M1"]; m2 = mip["M2"]; m3 = mip["M3"]; ii = mip["I"]
                fr = np.mean([m2[t]["follow_donor"] + m2[t]["rederive_cost"]
                              for t in m2])
                fv = np.mean([m2[t]["free_revert"] for t in m2])
                p_corr = (tr_o.get("compute_diag") or {}).get("corr_K_niter_ood")
                t_ok = gap >= 0.15
                m1_ok = (m1["trend_spearman"] >= 0.5
                         and m1["F_final_over_acc"] >= 0.8)
                m2_ok = fr >= 0.6 and fv <= 0.3
                m3_ok = m3["corr_fidelity_correct"] >= 0.2
                i_ok = ii["auc_pstop_correct"] >= 0.65
                p_ok = (p_corr is not None and p_corr >= 0.10) \
                    if variant == "hbp_full" else True
                gate = all((t_ok, m1_ok, m2_ok, m3_ok, i_ok, p_ok))
                passes.append(gate)
                fm = lambda v, ok: f"{v:+.2f}{'✓' if ok else '✗'}"
                w(f"| {seed} | {fm(gap, t_ok)} | {fm(m1['trend_spearman'], m1_ok)} "
                  f"| {m1['F_final_over_acc']:.2f} | {fm(fr, fr >= 0.6)} "
                  f"| {fm(fv, fv <= 0.3)} | {fm(m3['corr_fidelity_correct'], m3_ok)} "
                  f"| {fm(ii['auc_pstop_correct'], i_ok)} "
                  f"| {('—' if p_corr is None else fm(p_corr, p_ok))} "
                  f"| {'**PASS**' if gate else 'FAIL'} |")
            n_pass = sum(passes)
            verdict = "PASS" if n_pass >= 2 else "FAIL"
            verdicts[(variant, gens)] = (verdict, n_pass, len(passes))
            w(f"\n**Gate {variant}×{gens}: {verdict}** ({n_pass}/{len(passes)} seeds)\n")

    w("## Veredicto global\n")
    for (variant, gens), (v, n, tot) in verdicts.items():
        w(f"- {variant} × {gens}: **{v}** ({n}/{tot})")
    any_pass = any(v == "PASS" for v, _, _ in verdicts.values())
    w("")
    if any_pass:
        w("**G0 PASS en al menos un par (variante, genset): la Fase 3 puede "
          "proceder sobre esos pares (el gobernador tiene algo que gobernar).**")
    else:
        w("**G0 FAIL global: el par (reasoner, tarea) NO está en régimen de "
          "razonamiento — aplicar los remedios pre-registrados (rediseñar "
          "supervisión/tarea para hacer visible el camino) ANTES de integrar.**")
    out = os.path.join(HERE, "REPORT_G0.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(R))
    print("\n".join(R))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
