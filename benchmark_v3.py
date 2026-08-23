"""
BENCHMARK v3 — RE-VALIDACIÓN de v2 con el FIX del bug de congelación BF16
==========================================================================
IDÉNTICO protocolo pre-registrado que benchmark_v2.py (mismas celdas, mismas
seeds frescas 10.., mismos pasos), con UNA diferencia: train_run ahora aplica
`hbp.pin_fp32()` (los parámetros físicos raw ζ/ω₀/c/f_gain estaban CONGELADOS
en su init en BF16 porque el paso de Adam ~3e-4 es menor que ULP/2 de sus
valores ~0.3-1.6; ver memoria bug-bf16-congelacion).

Pregunta: ¿el efecto v2 (ventaja de adaptividad de cómputo OOD del HBP de 2º
orden) SOBREVIVE / SE REFUERZA / CAMBIA cuando ζ, ω₀, c, f_gain aprenden de
verdad? Es la validación que el paper necesita para cualquier claim de
"parámetros físicos aprendibles".

Resultados en results_benchmark_v3/ (v2 se conserva intacto como registro del
régimen de física congelada). Análisis: benchmark_report_v2.py apuntando aquí.

Uso:  $env:PYTHONPATH="." ; python benchmark_v3.py
"""
import json, os, time
from training.config import TrainConfig
from training.trainer import train_run

OUT = "results_benchmark_v3"
os.makedirs(OUT, exist_ok=True)

MAX_STEPS = 2500
N_MAX = 24
GENS = ["adjacent", "cycle_transp"]

# Celdas idénticas a v2. Orden: primero el confirmatorio OOD (headline),
# después el in-dist (secundario) — por si hay que cortar antes.
CELLS = []
for gens in GENS:
    for v in ["gating_wm", "hbp_first", "hbp_full"]:          # OOD confirmatorio
        for s in range(10, 20):
            CELLS.append((gens, "ood", 12, v, s))
for gens in GENS:
    for v in ["vanilla", "gating", "gating_wm", "hbp_first", "hbp_full"]:   # in-dist
        for s in [10, 11, 12]:
            CELLS.append((gens, "indist", 24, v, s))

print(f"BENCHMARK v3 (fix BF16): {len(CELLS)} celdas (N_max={N_MAX}, seeds frescas >=10)")
t0 = time.time()
for i, (gens, regime, tmw, variant, seed) in enumerate(CELLS, 1):
    name = f"{regime}_{gens}_{variant}_seed{seed}.json"
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        try:
            old = json.load(open(path, encoding="utf-8"))
            if old.get("cfg", {}).get("max_steps") == MAX_STEPS and \
               old.get("cfg", {}).get("max_halt_steps") == N_MAX:
                print(f"[{i:3d}/{len(CELLS)}] skip {name}", flush=True)
                continue
        except Exception:
            pass
    cfg = TrainConfig(task="permcomp", perm_gens=gens, variant=variant, max_writes=24,
                      train_max_writes=tmw, max_steps=MAX_STEPS, seed=seed,
                      max_halt_steps=N_MAX)
    res = train_run(cfg, results_dir=None, verbose=False)
    res["_cell"] = {"regime": regime, "gens": gens, "variant": variant, "seed": seed,
                    "fix": "pin_fp32"}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=2)
    os.replace(tmp, path)
    a = res["final_acc"]
    cd = res.get("compute_diag") or {}
    corr = cd.get("corr_K_niter_ood", cd.get("corr_K_niter"))
    corr_s = f"{corr:+.2f}" if corr is not None else "  -  "
    print(f"[{i:3d}/{len(CELLS)}] {name:44s} largo={a['largo']:.3f} corrOOD={corr_s} "
          f"({(time.time()-t0)/60:.0f} min)", flush=True)
print(f"\nBENCHMARK v3 COMPLETO en {(time.time()-t0)/60:.1f} min.")
