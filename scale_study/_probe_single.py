"""Probe: ¿el test de solitón individual (sech² exacto) da un solitón LIMPIO en el
régimen bien-resuelto (W>=8 nodos, ν pequeño)? Confirma antes de fijar el plan."""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scale_study.run_night import single_soliton_run

print(f"{'caso':38s} {'W_teo':>6s} {'v_teo':>7s} {'v_med':>7s} {'vErr':>6s} "
      f"{'ampR':>6s} {'wCV':>6s} {'nPk':>4s} {'life':>5s}")
for N in (1024, 4096):
    for nu in (0.2,):
        for beta in (0.1,):
            for a in (0.3, 0.6):
                for nl in ("genuine", "saturated"):
                    t = time.time()
                    r = single_soliton_run(N, a, beta, nl, "none",
                                           ticks=24000, dt=0.05, nu=nu)
                    dt = time.time() - t
                    tag = f"N{N}_nu{nu}_b{beta}_a{a}_{nl}"
                    def f(x, d=3):
                        return "None" if x is None else f"{x:.{d}f}"
                    print(f"{tag:38s} {r['soliton_width_theory']:6.1f} "
                          f"{f(r['v_lab_theory'],4):>7s} {f(r['velocity_measured'],4):>7s} "
                          f"{f(r['vel_err'],2):>6s} {f(r['amp_ratio_final_initial'],2):>6s} "
                          f"{f(r['width_cv'],2):>6s} {r['n_peaks_final']:>4d} "
                          f"{f(r['life_frac'],2):>5s}  ({dt:.0f}s)", flush=True)
