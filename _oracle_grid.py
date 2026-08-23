"""GRID DEL ORÁCULO sobre la tarea dual PHASE/SETTLE (pre-registrado):
Brazos (todos hbp_full + solver implícito + pin_fp32, tarea dual 50/50):
  const0   α≡0 (difusión pura)
  const1   α≡1 (onda pura)
  oracle   α por modo: PHASE→1 (onda), SETTLE→0 (difusión)   [predicción mecanicista]
  anti     α por modo INVERTIDO: PHASE→0, SETTLE→1           [control FATAL-4]
  random   α ~ Bernoulli(0.5) por instancia                   [placebo]
× SEEDS (5). Métrica primaria: accuracy total (modos y buckets agregados) pareada
por semilla; secundarias: por modo × bucket. Además, INTERVENCIÓN CAUSAL en eval:
cada modelo entrenado se evalúa también con la política de α INVERTIDA (mide el
poder causal del canal de física en el modelo ya entrenado).
Criterio de validación del formalismo: oracle > mejor constante (pareado, signo
consistente) Y oracle > anti. Si anti ≈ oracle: confound (cualquier etiqueta de
modo por canal lateral ayuda), no física. JSON en results_dual/grid_*.json."""
import os, json, math, torch
from data.synthetic_recall import DualRegimePhaseSettleDataset
from training.trainer import build_model

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
STEPS = int(os.environ.get("GRID_STEPS", "2500"))
SEEDS = [int(x) for x in os.environ.get("GRID_SEEDS", "0 1 2 3 4").split()]
OUT = "results_dual"; os.makedirs(OUT, exist_ok=True)

# políticas de α (modes: 0=PHASE, 1=SETTLE)
POLICIES = {
    "const0": lambda modes: torch.zeros(len(modes)),
    "const1": lambda modes: torch.ones(len(modes)),
    "oracle": lambda modes: (modes == 0).float(),          # PHASE→1, SETTLE→0
    "anti":   lambda modes: (modes == 1).float(),          # invertido
    "random": lambda modes: torch.randint(0, 2, (len(modes),)).float(),
}
SWAP = {"const0": "const1", "const1": "const0", "oracle": "anti", "anti": "oracle"}

def eval_dual(m, ds, policy, n_batches=40):
    """Accuracy por modo × bucket (última posición supervisada) + n_iter por modo."""
    m.eval()
    acc = {mo: {"corto": [0, 0], "medio": [0, 0], "largo": [0, 0]} for mo in ("phase", "settle")}
    nit = {"phase": [], "settle": []}
    a0, a1 = ds.answer_offset, ds.answer_offset + ds.answer_size
    with torch.no_grad():
        for _ in range(n_batches):
            inp, tgt, diffs, modes = ds.batch(32)
            inp, tgt = inp.to(dev), tgt.to(dev)
            af = policy(modes).to(dev)
            m.hbp.reset_state(inp.size(0), device=dev, dtype=dt)
            logits, _ = m(inp, None, alpha_force=af)
            preds = a0 + logits[..., a0:a1].argmax(dim=-1)
            ne = m._last_n_expected.float().cpu() if m._last_n_expected is not None else None
            for b in range(inp.size(0)):
                mo = "phase" if int(modes[b]) == 0 else "settle"
                pos = (tgt[b] != -1).nonzero(as_tuple=True)[0]
                if len(pos) == 0: continue
                p = pos[-1].item()
                bk = ds.bucket(int(diffs[b]))
                acc[mo][bk][0] += int(preds[b, p].item() == tgt[b, p].item())
                acc[mo][bk][1] += 1
                if ne is not None: nit[mo].append(float(ne[b]))
    m.train()
    out = {}
    tot_c, tot_n = 0, 0
    for mo in acc:
        out[mo] = {k: (v[0] / v[1] if v[1] else 0.0) for k, v in acc[mo].items()}
        out[mo]["n_iter"] = sum(nit[mo]) / len(nit[mo]) if nit[mo] else 0.0
        tot_c += sum(v[0] for v in acc[mo].values()); tot_n += sum(v[1] for v in acc[mo].values())
    out["total"] = tot_c / max(tot_n, 1)
    return out

def run(arm, seed):
    name = f"grid_{arm}_s{seed}"
    path = os.path.join(OUT, f"{name}.json")
    if os.path.exists(path):
        print(f"[skip] {name}", flush=True); return
    torch.manual_seed(seed)
    ds = DualRegimePhaseSettleDataset(min_ops=2, max_ops=16, seq_len=128, seed=seed, mode="dual")
    ds_ev = DualRegimePhaseSettleDataset(min_ops=2, max_ops=24, seq_len=128, seed=seed + 100000, mode="dual")
    m = build_model("hbp_full", ds.vocab_size, 128, max_halt_steps=24,
                    hbp_overrides={"diff_solver": "implicit"}).to(dev, dt)
    m.hbp.pin_fp32()
    act = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(act, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    pol = POLICIES[arm]
    m.train()
    for s in range(1, STEPS + 1):
        lr = 3e-4 * min(1, s/200) * 0.5*(1+math.cos(math.pi*min(1, s/STEPS)))
        for g in opt.param_groups: g["lr"] = lr
        inp, tgt, _, modes = ds.batch(32)
        _, ld = m(inp.to(dev), tgt.to(dev), alpha_force=pol(modes).to(dev))
        opt.zero_grad(); ld["total"].backward()
        torch.nn.utils.clip_grad_norm_(act, 1.0); opt.step()
    res = {"arm": arm, "seed": seed,
           "eval": eval_dual(m, ds_ev, pol),
           "zeta_mean": float(m.hbp.zeta.detach().mean()),
           "omega0_mean": float(m.hbp.omega0.detach().mean()),
           "gamma_diff_mean": float(m.hbp.gamma_diff.detach().mean())}
    # INTERVENCIÓN CAUSAL: evalúa con la política contraria (sin re-entrenar)
    if arm in SWAP:
        res["eval_swapped"] = eval_dual(m, ds_ev, POLICIES[SWAP[arm]])
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    e = res["eval"]
    sw = f" | swap_total={res['eval_swapped']['total']:.3f}" if "eval_swapped" in res else ""
    print(f"[done] {name:16s} total={e['total']:.3f} "
          f"phase(L)={e['phase']['largo']:.3f} settle(L)={e['settle']['largo']:.3f}{sw}", flush=True)
    del m; torch.cuda.empty_cache()

print(f"GRID ORÁCULO: {len(POLICIES)} brazos × {len(SEEDS)} seeds en {dev}\n" + "="*70, flush=True)
for seed in SEEDS:
    for arm in POLICIES:
        try:
            run(arm, seed)
        except Exception as e:
            print(f"[FAIL] {arm} s{seed}: {type(e).__name__}: {e}", flush=True)
print("="*70 + "\nGRID COMPLETO", flush=True)
