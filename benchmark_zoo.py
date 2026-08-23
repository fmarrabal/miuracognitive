"""
ZOO DE HOMEOSTASIS — hbp_kdv sobre el protocolo OOD pre-registrado de v3
=========================================================================
Extiende benchmark_v3 con la variante hbp_kdv (onda 2º orden + dispersión β·A³
+ advección no lineal ν·tanh(u)⊙(A·u), operadores antisimétricos en colocación
GIROSCÓPICA — corrección de Ziegler). Mismas celdas OOD (2 gen sets × seeds
10-19), mismos pasos, pin_fp32. Escribe en results_benchmark_v3/ con el mismo
naming para el análisis pareado conjunto (misma seed = mismos datos que las
demás variantes).

Pregunta (eje de CONTROLADOR, no accuracy): ¿la dispersión/no-linealidad añade
o resta adaptividad de cómputo OOD frente a la onda pura (hbp_full)?
"""
import json, os, time
from training.config import TrainConfig
from training.trainer import train_run

OUT = "results_benchmark_v3"
os.makedirs(OUT, exist_ok=True)
MAX_STEPS = 2500
N_MAX = 24

CELLS = [(gens, "ood", 12, "hbp_kdv", s)
         for gens in ["adjacent", "cycle_transp"] for s in range(10, 20)]

print(f"ZOO hbp_kdv: {len(CELLS)} celdas (protocolo v3)")
t0 = time.time()
for i, (gens, regime, tmw, variant, seed) in enumerate(CELLS, 1):
    name = f"{regime}_{gens}_{variant}_seed{seed}.json"
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        print(f"[{i:2d}/{len(CELLS)}] skip {name}", flush=True)
        continue
    cfg = TrainConfig(task="permcomp", perm_gens=gens, variant=variant, max_writes=24,
                      train_max_writes=tmw, max_steps=MAX_STEPS, seed=seed,
                      max_halt_steps=N_MAX)
    res = train_run(cfg, results_dir=None, verbose=False)
    res["_cell"] = {"regime": regime, "gens": gens, "variant": variant, "seed": seed,
                    "fix": "pin_fp32", "zoo": "kdv_gyroscopic"}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=2)
    os.replace(tmp, path)
    a = res["final_acc"]
    cd = res.get("compute_diag") or {}
    corr = cd.get("corr_K_niter_ood", cd.get("corr_K_niter"))
    corr_s = f"{corr:+.2f}" if corr is not None else "  -  "
    print(f"[{i:2d}/{len(CELLS)}] {name:42s} largo={a['largo']:.3f} corrOOD={corr_s} "
          f"({(time.time()-t0)/60:.0f} min)", flush=True)
print(f"\nZOO COMPLETO en {(time.time()-t0)/60:.1f} min.")
