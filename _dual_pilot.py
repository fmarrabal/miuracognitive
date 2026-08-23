"""PILOTO de la tarea dual PHASE/SETTLE (gates de decisión antes del grid con oráculo):
  (G1) ATAJO: ¿vanilla resuelve PHASE en largo? (si sí -> rediseñar: la paridad se
       está resolviendo por atención, no por iteración).
  (G2) MECANISMO en PHASE: ¿onda (α=1) > difusión (α=0)? (la demanda oscilatoria
       debería favorecer la memoria de fase del Verlet).
  (G3) MECANISMO en SETTLE: ¿difusión ≥ onda? (demanda monótona de asentarse).
Todos los brazos HBP: hbp_full + alpha_const + solver implícito + pin_fp32 (fixes
FATAL-2/3 aplicados). 2500 pasos, seed 0. JSON en results_dual/pilot_*.json."""
import os, json, math, torch
from data.synthetic_recall import DualRegimePhaseSettleDataset
from training.trainer import build_model

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
STEPS = int(os.environ.get("PILOT_STEPS", "2500"))
OUT = "results_dual"; os.makedirs(OUT, exist_ok=True)

def eval_dual(m, ds, n_batches=40, alpha_of_modes=None):
    """Accuracy (última posición supervisada) por bucket + E[n_iter] medio.
    alpha_of_modes: callable(modes)->(B,) o None."""
    m.eval()
    buckets = {"corto": [0, 0], "medio": [0, 0], "largo": [0, 0]}
    n_iters = []
    a0, a1 = ds.answer_offset, ds.answer_offset + ds.answer_size
    with torch.no_grad():
        for _ in range(n_batches):
            inp, tgt, diffs, modes = ds.batch(32)
            inp, tgt = inp.to(dev), tgt.to(dev)
            af = alpha_of_modes(modes).to(dev) if alpha_of_modes else None
            if m.cfg.use_hbp:
                m.hbp.reset_state(inp.size(0), device=dev, dtype=dt)
            logits, _ = m(inp, None, alpha_force=af) if af is not None else m(inp, None)
            preds = a0 + logits[..., a0:a1].argmax(dim=-1)
            if m._last_n_expected is not None:
                n_iters.append(float(m._last_n_expected.float().mean()))
            for b in range(inp.size(0)):
                pos = (tgt[b] != -1).nonzero(as_tuple=True)[0]
                if len(pos) == 0: continue
                p = pos[-1].item()
                buckets[ds.bucket(int(diffs[b]))][0] += int(preds[b, p].item() == tgt[b, p].item())
                buckets[ds.bucket(int(diffs[b]))][1] += 1
    m.train()
    acc = {k: (v[0] / v[1] if v[1] else 0.0) for k, v in buckets.items()}
    acc["n_iter"] = sum(n_iters) / len(n_iters) if n_iters else 0.0
    return acc

def run(name, variant, task_mode, alpha_const=None, seed=0):
    path = os.path.join(OUT, f"pilot_{name}.json")
    if os.path.exists(path):
        print(f"[skip] {name}", flush=True); return
    torch.manual_seed(seed)
    ds = DualRegimePhaseSettleDataset(min_ops=2, max_ops=16, seq_len=128, seed=seed, mode=task_mode)
    ds_ev = DualRegimePhaseSettleDataset(min_ops=2, max_ops=24, seq_len=128, seed=seed + 100000, mode=task_mode)
    ov = None
    if variant != "vanilla":
        ov = {"diff_solver": "implicit"}
        if alpha_const is not None:
            ov["alpha_const"] = float(alpha_const)
    m = build_model(variant, ds.vocab_size, 128, max_halt_steps=24, hbp_overrides=ov).to(dev, dt)
    if m.cfg.use_hbp:
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
    acc = eval_dual(m, ds_ev)
    extra = {}
    if m.cfg.use_hbp:
        extra = {"zeta_mean": float(m.hbp.zeta.detach().mean()),
                 "omega0_mean": float(m.hbp.omega0.detach().mean()),
                 "gamma_diff_mean": float(m.hbp.gamma_diff.detach().mean()) if hasattr(m.hbp, "raw_gamma_diff") else None}
    rec = {"name": name, "variant": variant, "task_mode": task_mode,
           "alpha_const": alpha_const, "seed": seed, "acc": acc, **extra}
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[done] {name:16s} L/M/S={acc['largo']:.3f}/{acc['medio']:.3f}/{acc['corto']:.3f} "
          f"n_iter={acc['n_iter']:.1f}" + (f" ζ={extra.get('zeta_mean'):.3f} ω₀={extra.get('omega0_mean'):.3f}" if extra else ""), flush=True)
    del m; torch.cuda.empty_cache()

ARMS = [
    ("phase_vanilla", "vanilla", "phase", None),     # G1: atajo
    ("phase_a1", "hbp_full", "phase", 1.0),          # G2: onda en PHASE
    ("phase_a0", "hbp_full", "phase", 0.0),          # G2: difusión en PHASE
    ("settle_a1", "hbp_full", "settle", 1.0),        # G3
    ("settle_a0", "hbp_full", "settle", 0.0),        # G3
]
print(f"PILOTO DUAL: {len(ARMS)} brazos en {dev}\n" + "="*70, flush=True)
for name, variant, tm, ac in ARMS:
    try:
        run(name, variant, tm, ac)
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
print("="*70 + "\nPILOTO COMPLETO", flush=True)
