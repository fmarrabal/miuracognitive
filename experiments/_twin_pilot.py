"""PILOTO v2: tarea TWIN/MONO (la demanda oscilatoria montada sobre S_5, que fuerza
iteración por dureza NC¹ — fix del colapso de halting del piloto PHASE/SETTLE).
Gates: (G0) ¿el reasoner ITERA? (n_iter debe crecer con K; si ~2, el diseño vuelve
a fallar el prerequisito). (G2) TWIN: ¿onda (α=1) > difusión (α=0)? (multiplexado
período-2). (G3) MONO: ¿difusión ≥ onda? JSON en results_dual/twin_*.json."""
import os, json, math, torch
from data.synthetic_recall import TwinStreamCompositionDataset
from training.trainer import build_model

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
STEPS = int(os.environ.get("PILOT_STEPS", "2500"))
OUT = "results_dual"; os.makedirs(OUT, exist_ok=True)

def eval_twin(m, ds, n_batches=40):
    """Accuracy y n_iter por bucket (última posición supervisada)."""
    m.eval()
    acc = {"corto": [0, 0], "medio": [0, 0], "largo": [0, 0]}
    nit = {"corto": [], "medio": [], "largo": []}
    a0, a1 = ds.answer_offset, ds.answer_offset + ds.answer_size
    with torch.no_grad():
        for _ in range(n_batches):
            inp, tgt, diffs, _ = ds.batch(32)
            inp, tgt = inp.to(dev), tgt.to(dev)
            if m.cfg.use_hbp:
                m.hbp.reset_state(inp.size(0), device=dev, dtype=dt)
            logits, _ = m(inp, None)
            preds = a0 + logits[..., a0:a1].argmax(dim=-1)
            ne = m._last_n_expected.float().cpu() if m._last_n_expected is not None else None
            for b in range(inp.size(0)):
                pos = (tgt[b] != -1).nonzero(as_tuple=True)[0]
                if len(pos) == 0: continue
                p = pos[-1].item()
                bk = ds.bucket(int(diffs[b]))
                acc[bk][0] += int(preds[b, p].item() == tgt[b, p].item())
                acc[bk][1] += 1
                if ne is not None: nit[bk].append(float(ne[b]))
    m.train()
    out = {k: (v[0] / v[1] if v[1] else 0.0) for k, v in acc.items()}
    out["n_iter"] = {k: (sum(v) / len(v) if v else 0.0) for k, v in nit.items()}
    return out

def run(name, task_mode, alpha_const, seed=0):
    path = os.path.join(OUT, f"twin_{name}.json")
    if os.path.exists(path):
        print(f"[skip] {name}", flush=True); return
    torch.manual_seed(seed)
    ds = TwinStreamCompositionDataset(min_ops=2, max_ops=16, seq_len=128, seed=seed, mode=task_mode)
    ds_ev = TwinStreamCompositionDataset(min_ops=2, max_ops=24, seq_len=128, seed=seed + 100000, mode=task_mode)
    ov = {"diff_solver": "implicit", "alpha_const": float(alpha_const)}
    m = build_model("hbp_full", ds.vocab_size, 128, max_halt_steps=24, hbp_overrides=ov).to(dev, dt)
    m.hbp.pin_fp32()
    act = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(act, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    m.train()
    for s in range(1, STEPS + 1):
        lr = 3e-4 * min(1, s/200) * 0.5*(1+math.cos(math.pi*min(1, s/STEPS)))
        for g in opt.param_groups: g["lr"] = lr
        inp, tgt, _, _ = ds.batch(32)
        _, ld = m(inp.to(dev), tgt.to(dev))
        opt.zero_grad(); ld["total"].backward()
        torch.nn.utils.clip_grad_norm_(act, 1.0); opt.step()
    acc = eval_twin(m, ds_ev)
    rec = {"name": name, "task_mode": task_mode, "alpha_const": alpha_const, "seed": seed,
           "acc": {k: acc[k] for k in ("corto", "medio", "largo")}, "n_iter": acc["n_iter"],
           "zeta_mean": float(m.hbp.zeta.detach().mean()),
           "omega0_mean": float(m.hbp.omega0.detach().mean())}
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    ni = acc["n_iter"]
    print(f"[done] {name:10s} L/M/S={acc['largo']:.3f}/{acc['medio']:.3f}/{acc['corto']:.3f} "
          f"n_iter(C/M/L)={ni['corto']:.1f}/{ni['medio']:.1f}/{ni['largo']:.1f} "
          f"ζ={rec['zeta_mean']:.3f} ω₀={rec['omega0_mean']:.3f}", flush=True)
    del m; torch.cuda.empty_cache()

ARMS = [
    ("twin_a1", "twin", 1.0),
    ("twin_a0", "twin", 0.0),
    ("mono_a1", "mono", 1.0),
    ("mono_a0", "mono", 0.0),
]
print(f"PILOTO TWIN/MONO: {len(ARMS)} brazos en {dev}\n" + "="*72, flush=True)
for name, tm, ac in ARMS:
    try:
        run(name, tm, ac)
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
print("="*72 + "\nPILOTO COMPLETO", flush=True)
