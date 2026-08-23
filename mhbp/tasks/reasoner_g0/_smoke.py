"""Smoke G0: parches inertes + record + intervene + cadena entera en miniatura."""
import os
import time

import torch

from training.config import TrainConfig
from training.trainer import train_run, build_model, build_dataset
from .instruments import (collect_trajectories, train_probes, probe_accuracy,
                          swap_experiment)

HERE = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

# 1) retro-compat: forward normal SIN flags == comportamiento previo (sin estados)
cfg = TrainConfig(task="permcomp", perm_gens="adjacent", variant="hbp_full",
                  max_writes=12, max_steps=1, seed=0, max_halt_steps=8)
ds = build_dataset(cfg)
m = build_model("hbp_full", ds.vocab_size, cfg.seq_len, max_halt_steps=8
                ).to(device=device, dtype=dtype)
m.hbp.pin_fp32(); m.eval()
inp, tgt, nd = ds.batch(8)
inp = inp.to(device)
with torch.no_grad():
    m.hbp.reset_state(device=device, dtype=dtype)
    m(inp, None)
assert m._last_step_states is None, "record off debe dejar states=None"
print("[OK] retro-compat (sin flags, sin estados)")

# 2) record: estados por tick presentes y completos (sin corte temprano)
m.record_step_states = True
with torch.no_grad():
    m.hbp.reset_state(device=device, dtype=dtype)
    m(inp, None)
ss = m._last_step_states
assert ss is not None and len(ss) == 8, f"esperaba 8 ticks, hay {len(ss) if ss else 0}"
# batch() recorta el último token: T = seq_len − 1 (labels alineadas, sin shift)
assert ss[0].shape == (8, cfg.seq_len - 1, 256), tuple(ss[0].shape)
m.record_step_states = False
print(f"[OK] record: {len(ss)} ticks de (8,{cfg.seq_len},256)")

# 3) intervene: un swap de mitades cambia la salida
def swapper(x, n):
    if n == 2:
        return torch.cat([x[4:], x[:4]], 0)
    return x
with torch.no_grad():
    m.hbp.reset_state(device=device, dtype=dtype)
    l0, _ = m(inp, None)
    m.tick_intervene = swapper
    m.hbp.reset_state(device=device, dtype=dtype)
    l1, _ = m(inp, None)
    m.tick_intervene = None
assert not torch.allclose(l0, l1), "el intervene no tuvo efecto"
print("[OK] intervene: el swap altera la salida")

# 4) cadena entera en miniatura: train 40 pasos + ckpt + instrumentos
t0 = time.time()
ck = os.path.join(HERE, "ckpts")
cfg2 = TrainConfig(task="permcomp", perm_gens="adjacent", variant="hbp_full",
                   max_writes=12, train_max_writes=12, max_steps=40, seed=0,
                   max_halt_steps=8, out_dir=ck)
res = train_run(cfg2, results_dir=None, save_ckpt=True, verbose=False)
p = os.path.join(ck, "hbp_full_seed0_final.pt")
assert os.path.exists(p)
m2 = build_model("hbp_full", ds.vocab_size, cfg2.seq_len, max_halt_steps=8
                 ).to(device=device, dtype=dtype)
m2.load_state_dict(torch.load(p, map_location=device, weights_only=True)["model"])
m2.hbp.pin_fp32(); m2.eval()
tr = collect_trajectories(m2, ds, 96, device, dtype)
assert tr["feat_ans"].shape[1] == 8 and tr["feat_ans"].shape[2] == 256
heads = train_probes(tr["feat_ans"], tr["ans"], 120, device, epochs=60)
acc = probe_accuracy(heads, tr["feat_ans"], tr["ans"], device)
sw = swap_experiment(m2, ds, device, dtype, t_stars=(2, 4), n_pairs=24,
                     k_min=4, k_max=10, batch_pairs=12)
os.remove(p)
print(f"[OK] cadena: probes F(t) {acc[0]:.2f}->{acc[-1]:.2f} | "
      f"swap t=4: n={sw[4]['n']} fd={sw[4]['follow_donor']:.2f} "
      f"fr={sw[4]['free_revert']:.2f} ({time.time()-t0:.0f}s)")
print("SMOKE G0 OK")
