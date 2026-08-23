"""Genera los datos del demo interactivo desde un checkpoint entrenado."""
import json, sys, torch
from miura_infer import MiuraModel
from eval.diagnostics import run_full_diagnostics

ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/hbp_full_adjacent.pt"
m = MiuraModel.from_checkpoint(ckpt, device="cpu")
ds = m.ds

def perm_str(p): return "".join(map(str, p))

# Composiciones concretas de distinta profundidad, con traza del HBP
gen = torch.Generator().manual_seed(7)
sessions = []
for K in [2, 3, 5, 7, 9, 11]:
    ops = [int(torch.randint(0, ds.n_gen, (1,), generator=gen).item()) for _ in range(K)]
    r = m.compose(ops)
    sessions.append({
        "K": K, "ops": r.ops,
        "true": perm_str(r.true_perm), "pred": perm_str(r.predicted),
        "correct": r.correct, "n_iter": round(r.n_iter, 2),
        "confidence": round(r.confidence, 3),
        "trace": [{"n": s["n"], "vei_var": round(s["vei_var"], 5),
                   "deviation": round(s["deviation"], 4),
                   "halt": round(s["halt_threshold"], 4),
                   "mod": round(s["reasoner_mod_norm"], 4)} for s in r.trace],
    })

# Barrido cómputo-vs-dificultad (n_iter y accuracy por K)
sweep = []
for K in range(2, 13):
    nit, cor = [], []
    for _ in range(24):
        ops = [int(torch.randint(0, ds.n_gen, (1,), generator=gen).item()) for _ in range(K)]
        r = m.compose(ops, trace=False)
        nit.append(r.n_iter); cor.append(1.0 if r.correct else 0.0)
    sweep.append({"K": K, "niter": round(sum(nit)/len(nit), 2), "acc": round(sum(cor)/len(cor), 3)})

# Espectro del campo homeostático (oscilación en evolución libre)
rep = run_full_diagnostics(m.model.hbp, n_ticks=200, device="cpu", dtype=torch.float32)
spec = rep["spectrum"]
power = spec["power_spectrum"]; freqs = spec["freqs"]
mx = max(power) if power else 1.0
spectrum = {"freqs": [round(f, 4) for f in freqs],
            "power": [round(p / mx, 4) for p in power],
            "dominant_freq": round(spec["dominant_freq"], 4),
            "zeta": round(rep["damping"]["zeta_mean"], 3),
            "oscillates": spec["has_oscillation"]}

out = {"meta": m.meta, "n_params": m.model.num_parameters(),
       "max_halt_steps": m.model.cfg.max_halt_steps,
       "n_elems": ds.n, "n_gen": ds.n_gen,
       "sessions": sessions, "sweep": sweep, "spectrum": spectrum}
json.dump(out, open("paper/demo_data.json", "w"), indent=1)
print("demo_data.json escrito |", len(sessions), "sesiones |",
      "acc corta:", sweep[0]["acc"], "acc K12:", sweep[-1]["acc"])
