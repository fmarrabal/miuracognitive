"""Equivalencia de urgency_signal vectorizada vs referencia en bucle."""
import torch
from data.exec_v2 import ExecV2Config, ExecV2Dataset, urgency_signal


def reference(cfg, sc):
    c = cfg
    B, T, K = sc.urgency_base.shape
    u = sc.urgency_base.clone()
    for s in range(sc.spike_t.shape[1]):
        for row in range(B):
            t0 = int(sc.spike_t[row, s])
            p = int(sc.spike_proj[row, s])
            if bool(sc.spike_is_crisis[row, s]):
                for j in range(c.precursor_len):
                    u[row, t0 - c.precursor_len + j, p] = max(
                        float(u[row, t0 - c.precursor_len + j, p]), 0.3 * (j + 1))
            for tt in range(t0, min(T, t0 + c.crisis_window)):
                u[row, tt, p] = 1.0
    return u.clamp(0.0, 1.0)


cfg = ExecV2Config()
for seed in (0, 7, 123):
    ds = ExecV2Dataset(cfg, seed=seed)
    sc = ds.batch(128)
    gap = float((urgency_signal(cfg, sc) - reference(cfg, sc)).abs().max())
    print(f"seed {seed}: gap_max={gap:.2e} {'OK' if gap < 1e-6 else 'DIFIERE'}")

import time
sc = ExecV2Dataset(cfg, seed=1).batch(256)
t0 = time.time(); [urgency_signal(cfg, sc) for _ in range(20)]
tv = (time.time() - t0) / 20
t0 = time.time(); reference(cfg, sc)
tr = time.time() - t0
print(f"velocidad: vectorizada {tv*1000:.1f} ms vs referencia {tr*1000:.0f} ms "
      f"(x{tr/tv:.0f})")
