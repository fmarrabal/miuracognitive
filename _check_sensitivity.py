import statistics, torch
from data.synthetic_recall import PermutationCompositionDataset
from training.trainer import build_model
torch.manual_seed(0)
ds = PermutationCompositionDataset(seq_len=128, seed=0)

def check(scale):
    m = build_model("hbp_mix", ds.vocab_size, 128, max_halt_steps=24,
                    hbp_overrides={"gate_init_scale": scale, "D_max": 0.4, "b_adv_max": 0.4}).float().eval()
    a = []
    with torch.no_grad():
        for _ in range(6):
            inp, _, _ = ds.batch(8); m(inp, None); a.append(m.hbp.physics_state()["alpha"])
    m.record_trace = True
    with torch.no_grad():
        inp, _, _ = ds.batch(1); m(inp, None)
    traj = [t["alpha"] for t in m._trace if "alpha" in t]
    tr = (max(traj) - min(traj)) if traj else 0.0
    print(f"scale={scale}: alpha/batch={['%.4f' % x for x in a]} std={statistics.pstdev(a):.4f}")
    print(f"          n_ticks={len(traj)} traj={['%.4f' % x for x in traj]} tickRange={tr:.4f}")

check(1.0)
check(8.0)
