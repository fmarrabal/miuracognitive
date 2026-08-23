"""A/B del acoplamiento ELÍPTICO NO-LOCAL (hbp_elliptic) contra hbp_full, sobre el
MISMO protocolo pre-registrado v3 (donde TODA física local resultó neutra en
accuracy y donde el 2º orden mostró la única señal: adaptividad de cómputo OOD).

Pregunta: ¿la NO-LOCALIDAD (la modulación de cada nodo lee ψ=L⁺(h−h*), función de
todo el campo) es load-bearing donde la física local no lo fue? h_full ya está en
results_benchmark_v3/; aquí generamos las celdas de hbp_elliptic con el mismo
naming (resumible, JSON atómico por celda) y comparamos pareado por seed.

Uso: $env:PYTHONPATH="." ; python benchmark_elliptic.py
"""
import json, os, time
from training.config import TrainConfig
from training.trainer import train_run

OUT = "results_benchmark_v3"
os.makedirs(OUT, exist_ok=True)
MAX_STEPS, N_MAX = 2500, 24
GENS = ["adjacent", "cycle_transp"]

CELLS = [(g, "ood", 12, "hbp_elliptic", s) for g in GENS for s in range(10, 20)]

print(f"A/B ELÍPTICO: {len(CELLS)} celdas hbp_elliptic (protocolo v3)")
t0 = time.time()
for i, (gens, regime, tmw, variant, seed) in enumerate(CELLS, 1):
    name = f"{regime}_{gens}_{variant}_seed{seed}.json"
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        try:
            if json.load(open(path))["cfg"]["max_steps"] == MAX_STEPS:
                print(f"[{i:2d}/{len(CELLS)}] skip {name}", flush=True); continue
        except Exception:
            pass
    cfg = TrainConfig(task="permcomp", perm_gens=gens, variant=variant, max_writes=24,
                      train_max_writes=tmw, max_steps=MAX_STEPS, seed=seed,
                      max_halt_steps=N_MAX)
    res = train_run(cfg, results_dir=None, verbose=False)
    res["_cell"] = {"regime": regime, "gens": gens, "variant": variant, "seed": seed}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=2)
    os.replace(tmp, path)
    a = res["final_acc"]
    cd = res.get("compute_diag") or {}
    corr = cd.get("corr_K_niter_ood", cd.get("corr_K_niter"))
    print(f"[{i:2d}/{len(CELLS)}] {name:44s} largo={a['largo']:.3f} "
          f"corrOOD={corr:+.2f} ({(time.time()-t0)/60:.0f} min)", flush=True)
print(f"\nCOMPLETO en {(time.time()-t0)/60:.1f} min.  Análisis: python _elliptic_report.py")
