"""SCAN DE DIFERENCIACIÓN DE FÍSICA: ¿qué tareas prefieren onda (α=1) y cuáles
difusión (α=0)? Entrena hbp_full con alpha_const forzado en {0,1} sobre las 4
tareas existentes. El par con signos OPUESTOS será la base de la tarea dual
(el oráculo tendrá hueco por construcción). JSON por run en results_alpha_scan/."""
import os, json, math, torch
from data.synthetic_recall import (SyntheticRecallDataset, RunningSumModDataset,
                                   PermutationCompositionDataset, IteratedFunctionDataset)
from training.trainer import build_model, evaluate

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
STEPS = int(os.environ.get("SCAN_STEPS", "2500"))
OUT = "results_alpha_scan"; os.makedirs(OUT, exist_ok=True)

def make_ds(task, seed, for_eval=False):
    if task == "permcomp":
        return PermutationCompositionDataset(min_ops=2, max_ops=24 if for_eval else 16,
                                             seq_len=128, seed=seed, gen_set="adjacent")
    if task == "iterfunc":
        return IteratedFunctionDataset(min_ops=2, max_ops=24 if for_eval else 16,
                                       seq_len=128, seed=seed)
    if task == "runsum":
        return RunningSumModDataset(mod=3, min_writes=2, max_writes=24 if for_eval else 16,
                                    seq_len=128, seed=seed)
    if task == "recall":
        return SyntheticRecallDataset(min_distractors=2, max_distractors=60,
                                      seq_len=128, seed=seed)
    raise ValueError(task)

def run(task, alpha, seed=0):
    name = f"{task}_a{int(alpha*100):03d}_s{seed}"
    path = os.path.join(OUT, f"{name}.json")
    if os.path.exists(path):
        print(f"[skip] {name}", flush=True); return
    torch.manual_seed(seed)
    ds = make_ds(task, seed)
    ds_eval = make_ds(task, seed + 100000, for_eval=True)
    ov = {"alpha_const": float(alpha)}
    m = build_model("hbp_full", ds.vocab_size, 128, max_halt_steps=24, hbp_overrides=ov).to(dev, dt)
    act = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(act, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    m.train()
    for s in range(1, STEPS + 1):
        lr = 3e-4 * min(1, s/200) * 0.5*(1+math.cos(math.pi*min(1, s/STEPS)))
        for g in opt.param_groups: g["lr"] = lr
        inp, tgt, _ = ds.batch(32)
        _, ld = m(inp.to(dev), tgt.to(dev))
        opt.zero_grad(); ld["total"].backward()
        torch.nn.utils.clip_grad_norm_(act, 1.0); opt.step()
    acc = evaluate(m, ds_eval, dev, dt, n_batches=40)
    stab = float(m.hbp.stability_penalty().detach())
    rec = {"task": task, "alpha": alpha, "seed": seed, "acc": acc, "stab_penalty": stab,
           "zeta_mean": float(m.hbp.zeta.detach().mean()), "omega0_mean": float(m.hbp.omega0.detach().mean())}
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[done] {name:22s} L/M/S={acc['largo']:.3f}/{acc['medio']:.3f}/{acc['corto']:.3f} stab={stab:.4f}", flush=True)
    del m; torch.cuda.empty_cache()

TASKS = ["permcomp", "iterfunc", "runsum", "recall"]
print(f"SCAN α∈{{0,1}} × {TASKS} en {dev}\n" + "="*66, flush=True)
for task in TASKS:
    for alpha in (1.0, 0.0):
        try:
            run(task, alpha)
        except Exception as e:
            print(f"[FAIL] {task} a={alpha}: {type(e).__name__}: {e}", flush=True)
print("="*66 + "\nSCAN COMPLETO", flush=True)
# resumen
print(f"\n{'task':10s} {'acc(α=1) L/M/S':>22s} {'acc(α=0) L/M/S':>22s}  Δlargo(0-1)")
for task in TASKS:
    try:
        a1 = json.load(open(os.path.join(OUT, f"{task}_a100_s0.json")))["acc"]
        a0 = json.load(open(os.path.join(OUT, f"{task}_a000_s0.json")))["acc"]
        f3 = lambda a: f"{a['largo']:.3f}/{a['medio']:.3f}/{a['corto']:.3f}"
        print(f"{task:10s} {f3(a1):>22s} {f3(a0):>22s}  {a0['largo']-a1['largo']:+.3f}")
    except FileNotFoundError:
        print(f"{task:10s} (incompleto)")
