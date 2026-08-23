"""Agrega results_pde/*.json y emite el veredicto de la familia PDE:
  (Q1) ¿Diferencia la física? -> inputStd(α), tickRange(α), corr_K_α consistentes.
  (Q2) ¿Responde a la estructura de tarea? -> α medio adjacent vs cycle_transp.
  (Q3) ¿Estable entre semillas? -> media±std.
  (Q4) ¿b y D son load-bearing? -> ablaciones vs base en accuracy.
  (Q5) ¿hbp_mix mejora accuracy vs hbp_full? -> anclas.
"""
import os, json, glob, statistics

OUT = os.environ.get("PDE_OUT", "results_pde")
runs = {}
for p in sorted(glob.glob(os.path.join(OUT, "*.json"))):
    r = json.load(open(p)); runs[r["name"]] = r

def acc(n): return runs[n]["acc"] if n in runs else None
def phys(n): return runs[n].get("physics") if n in runs else None

print("=" * 100)
print(f"{'run':30s} {'accL':>6s} {'accM':>6s} {'accS':>6s} | "
      f"{'αmeanK':>7s} {'αrngK':>6s} {'corrKα':>7s} {'inStd':>6s} {'tickR':>6s} {'Drng':>5s} {'brng':>5s}")
print("-" * 100)
for n in runs:
    a = acc(n); pp = phys(n)
    line = f"{n:30s} {a['largo']:6.3f} {a['medio']:6.3f} {a['corto']:6.3f} | "
    if pp:
        amean = statistics.mean(float(pp['perK'][k]['alpha']) for k in pp['perK'])
        line += (f"{amean:7.3f} {pp['alpha_range_K']:6.3f} {pp['corr_K_alpha']:+7.2f} "
                 f"{pp['alpha_input_std_K12']:6.3f} {pp['alpha_tick_range']:6.3f} "
                 f"{pp['D_range_K']:5.3f} {pp['b_range_K']:5.3f}")
    else:
        line += " " * 8 + "(sin física: hbp_full)"
    print(line)

def grp(names):
    names = [n for n in names if n in runs]
    if not names: return None
    return {k: (statistics.mean(runs[n]['acc'][k] for n in names),
                statistics.pstdev(runs[n]['acc'][k] for n in names) if len(names) > 1 else 0.0)
            for k in ('largo', 'medio', 'corto')}

print("\n" + "=" * 100 + "\nVEREDICTOS\n" + "=" * 100)

# Q5: mix vs full por tarea (semillas 0-2 de mix vs ancla full s0)
for gens in ["adjacent", "cycle_transp"]:
    mix = grp([f"mix_s1_{gens}_s{s}" for s in (0, 1, 2)])
    full = acc(f"full_anchor_{gens}_s0")
    if mix and full:
        print(f"\n[Q5 accuracy · {gens}]  hbp_mix(3 seeds, media±std) vs hbp_full(s0):")
        for k in ('largo', 'medio', 'corto'):
            m, sd = mix[k]
            print(f"   {k:6s}: mix {m:.3f}±{sd:.3f}   full {full[k]:.3f}   Δ={m-full[k]:+.3f}")

# Q1/Q3: diferenciación de física (consistencia entre semillas)
print("\n[Q1/Q3 diferenciación de física · mix_s1]")
for gens in ["adjacent", "cycle_transp"]:
    ns = [f"mix_s1_{gens}_s{s}" for s in (0, 1, 2) if f"mix_s1_{gens}_s{s}" in runs]
    if not ns: continue
    istd = [phys(n)['alpha_input_std_K12'] for n in ns]
    tick = [phys(n)['alpha_tick_range'] for n in ns]
    corr = [phys(n)['corr_K_alpha'] for n in ns]
    print(f"   {gens:13s}: inputStd(α)={[f'{x:.3f}' for x in istd]}  "
          f"tickRange(α)={[f'{x:.3f}' for x in tick]}  corrKα={[f'{x:+.2f}' for x in corr]}")
    consistent = all(c > 0 for c in corr) or all(c < 0 for c in corr)
    print(f"                  signo corr_K_α consistente entre semillas: {consistent}  "
          f"| inputStd medio={statistics.mean(istd):.3f} (>~0.02 = conmuta por input)")

# Q2: estructura de tarea (α medio adjacent vs cycle_transp)
def amean_grp(gens):
    ns = [f"mix_s1_{gens}_s{s}" for s in (0, 1, 2) if f"mix_s1_{gens}_s{s}" in runs]
    vals = [statistics.mean(float(phys(n)['perK'][k]['alpha']) for k in phys(n)['perK']) for n in ns]
    return (statistics.mean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0) if vals else None
aa, cc = amean_grp("adjacent"), amean_grp("cycle_transp")
if aa and cc:
    print(f"\n[Q2 estructura de tarea]  α medio: adjacent={aa[0]:.3f}±{aa[1]:.3f}  "
          f"cycle_transp={cc[0]:.3f}±{cc[1]:.3f}  Δ={cc[0]-aa[0]:+.3f}")
    print("   (adjacent=transposiciones; cycle_transp incluye un 5-ciclo → más oscilatorio → esperaríamos α mayor)")

# Q4: ablaciones b y D
base = acc("mix_s1_adjacent_s0")
for ab, lab in [("mix_noadv_adjacent_s0", "sin advección b"), ("mix_nodiff_adjacent_s0", "sin difusión D")]:
    a = acc(ab)
    if base and a:
        print(f"\n[Q4 ablación · {lab}]  vs base(mix_s1_adjacent_s0):")
        for k in ('largo', 'medio', 'corto'):
            print(f"   {k:6s}: base {base[k]:.3f}  ablado {a[k]:.3f}  Δ={a[k]-base[k]:+.3f}")

# init 8.0 vs 1.0
for gens in ["adjacent", "cycle_transp"]:
    s8 = phys(f"mix_s8_{gens}_s0"); s1 = phys(f"mix_s1_{gens}_s0")
    if s8 and s1:
        print(f"\n[init prior · {gens}]  inputStd(α): s1={s1['alpha_input_std_K12']:.3f}  "
              f"s8={s8['alpha_input_std_K12']:.3f}  | tickRange: s1={s1['alpha_tick_range']:.3f} s8={s8['alpha_tick_range']:.3f}")
print("\n" + "=" * 100)
