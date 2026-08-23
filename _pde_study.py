"""Estudio decisivo de la familia PDE (hbp_mix). Responde con datos:
  (Q1) ¿La física plana es artefacto del init tímido? -> init tímido (0.1) vs sensible (1.0).
  (Q2) ¿La física responde a la ESTRUCTURA de la tarea? -> adjacent vs cycle_transp.
  (Q3) ¿Es estable entre semillas? -> seeds 0,1,2.
  (Q4) ¿La advección (b) es load-bearing? -> ablación b_adv_max=0.
Para cada modelo mide: accuracy; física media vs K (corr con dificultad); dependencia
del input a K fijo (std de α entre problemas distintos); rango de α por tick (adaptación
DURANTE el pensamiento). Vuelca un JSON por run en results_pde/.
"""
import os, json, math, statistics, torch
from data.synthetic_recall import PermutationCompositionDataset
from training.trainer import build_model, evaluate

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
STEPS = int(os.environ.get("PDE_STEPS", "2500"))
OUT = os.environ.get("PDE_OUT", "results_pde"); os.makedirs(OUT, exist_ok=True)
KS = [2, 4, 6, 9, 12, 16, 20, 24]

def _pearson(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    num = sum((a-mx)*(b-my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a-mx)**2 for a in x)); dy = math.sqrt(sum((b-my)**2 for b in y))
    return num/(dx*dy) if dx > 1e-12 and dy > 1e-12 else 0.0

def _make_batch(ds, K, B, gen):
    seqs = []
    for _ in range(B):
        ops = [int(torch.randint(0, ds.n_gen, (1,), generator=gen).item()) for _ in range(K)]
        seq = [ds.gen_offset + j for j in ops] + [ds.Q, ds.ARROW]
        seq += [ds.PAD] * (ds.seq_len - len(seq))
        seqs.append(torch.tensor(seq))
    return torch.stack(seqs)

@torch.no_grad()
def physics_diag(m, ds):
    m.eval(); gen = torch.Generator().manual_seed(12345)
    perK = {}
    for K in KS:
        m(_make_batch(ds, K, 48, gen).to(dev), None)
        perK[K] = m.hbp.physics_state()
    a = [perK[K]["alpha"] for K in KS]; D = [perK[K]["D"] for K in KS]; b = [perK[K]["b"] for K in KS]
    # dependencia del input a K=12: 64 problemas distintos, uno a uno
    a_single = []
    for _ in range(64):
        m(_make_batch(ds, 12, 1, gen).to(dev), None)
        a_single.append(m.hbp.physics_state()["alpha"])
    # trayectoria por tick (adaptación durante el pensamiento) en un K=16
    m.record_trace = True
    m(_make_batch(ds, 16, 1, gen).to(dev), None)
    m.record_trace = False
    a_traj = [t["alpha"] for t in m._trace if "alpha" in t]
    return {
        "perK": {str(K): perK[K] for K in KS},
        "alpha_range_K": max(a) - min(a), "corr_K_alpha": _pearson([float(k) for k in KS], a),
        "D_range_K": max(D) - min(D), "corr_K_D": _pearson([float(k) for k in KS], D),
        "b_range_K": max(b) - min(b), "corr_K_b": _pearson([float(k) for k in KS], b),
        "alpha_input_std_K12": statistics.pstdev(a_single) if len(a_single) > 1 else 0.0,
        "alpha_tick_range": (max(a_traj) - min(a_traj)) if a_traj else 0.0,
        "alpha_traj": a_traj,
    }

def train_eval(name, variant, gens, seed, overrides):
    torch.manual_seed(seed)
    ds = PermutationCompositionDataset(min_ops=2, max_ops=16, seq_len=128, seed=seed, gen_set=gens)
    ds_eval = PermutationCompositionDataset(min_ops=2, max_ops=24, seq_len=128, seed=seed+100000, gen_set=gens)
    m = build_model(variant, ds.vocab_size, 128, max_halt_steps=24, hbp_overrides=overrides).to(dev, dt)
    act = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(act, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    m.train()
    for s in range(1, STEPS+1):
        lr = 3e-4 * min(1, s/200) * 0.5*(1+math.cos(math.pi*min(1, s/STEPS)))
        for g in opt.param_groups: g["lr"] = lr
        inp, tgt, _ = ds.batch(32)
        _, ld = m(inp.to(dev), tgt.to(dev))
        opt.zero_grad(); ld["total"].backward()
        torch.nn.utils.clip_grad_norm_(act, 1.0); opt.step()
    acc = evaluate(m, ds_eval, dev, dt, n_batches=40)
    rec = {"name": name, "variant": variant, "gens": gens, "seed": seed,
           "overrides": overrides, "acc": acc}
    if variant == "hbp_mix":
        rec["physics"] = physics_diag(m, ds)
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(rec, f, indent=2)
    tail = ""
    if "physics" in rec:
        p = rec["physics"]
        tail = (f" | αrangeK={p['alpha_range_K']:.3f} corrKα={p['corr_K_alpha']:+.2f}"
                f" inputStd={p['alpha_input_std_K12']:.3f} tickRange={p['alpha_tick_range']:.3f}")
    print(f"[done] {name:28s} acc L/M/S={acc['largo']:.3f}/{acc['medio']:.3f}/{acc['corto']:.3f}{tail}", flush=True)
    del m; torch.cuda.empty_cache()
    return rec

# ---- Matriz del estudio (con phys_norm; gradiente ya no hambriento) ----
BASE = {"gate_init_scale": 1.0, "D_max": 0.4, "b_adv_max": 0.4}
CONFIGS = []
for gens in ["adjacent", "cycle_transp"]:
    for seed in [0, 1, 2]:
        CONFIGS.append((f"mix_s1_{gens}_s{seed}", "hbp_mix", gens, seed, dict(BASE)))
    # prior de input fuerte (empieza input-sensible): ¿cambia el desenlace?
    CONFIGS.append((f"mix_s8_{gens}_s0", "hbp_mix", gens, 0, {**BASE, "gate_init_scale": 8.0}))
    CONFIGS.append((f"full_anchor_{gens}_s0", "hbp_full", gens, 0, None))
# ablaciones (sensible, adjacent, s0)
CONFIGS.append(("mix_noadv_adjacent_s0", "hbp_mix", "adjacent", 0, {**BASE, "b_adv_max": 0.0}))
CONFIGS.append(("mix_nodiff_adjacent_s0", "hbp_mix", "adjacent", 0, {**BASE, "D_max": 0.0}))

print(f"ESTUDIO PDE: {len(CONFIGS)} runs en {dev}\n" + "="*70, flush=True)
for name, variant, gens, seed, ov in CONFIGS:
    path = os.path.join(OUT, f"{name}.json")
    if os.path.exists(path):
        print(f"[skip] {name} (ya existe)", flush=True); continue
    try:
        train_eval(name, variant, gens, seed, ov)
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
print("="*70 + "\nESTUDIO COMPLETO", flush=True)
