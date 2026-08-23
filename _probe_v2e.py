"""Probe v2e — barrido de descomposición con gating_wm (detector limpio, sin
HBP) en indist adjacent seed 0, N_max=24. Aísla qué factor colapsa n_iter:
  E0 (=v2d):  beta_final=0.5, frac_valid ON,  wm loop      [referencia: niter 6.5]
  E1:         beta_final=0.0, frac_valid ON,  wm loop
  E2:         beta_final=0.5, frac_valid OFF, wm loop
  E3:         beta_final=0.0, frac_valid OFF, wm preloop   ["casi-H1" + medición limpia]
Meta: recuperar niter ~10+ (nivel H1)."""
import json, os, time
from training.config import TrainConfig
from training.trainer import train_run, build_model, build_dataset

os.makedirs("results_v2e", exist_ok=True)

EXPS = [
    ("E1_nofinal", dict(beta_final=0.0, halt_len=True, loop=True)),
    ("E2_nofrac", dict(beta_final=0.5, halt_len=False, loop=True)),
    ("E3_casiH1", dict(beta_final=0.0, halt_len=False, loop=False)),
]

# beta_final y halt_len_feature no están en TrainConfig: los inyecto vía un
# hook sobre build_model (monkeypatch mínimo del probe, documentado).
import training.trainer as T
orig_build = T.build_model
t0 = time.time()
for tag, e in EXPS:
    path = f"results_v2e/{tag}.json"
    if os.path.exists(path):
        print(f"skip {tag}", flush=True)
        continue

    def patched(variant, vocab, seqlen, **kw):
        m = orig_build(variant, vocab, seqlen, **kw)
        m.cfg.beta_final = e["beta_final"]
        m.cfg.halt_len_feature = e["halt_len"]
        return m
    T.build_model = patched
    try:
        cfg = TrainConfig(task="permcomp", perm_gens="adjacent", variant="gating_wm",
                          max_writes=24, train_max_writes=24, max_steps=2500,
                          seed=0, max_halt_steps=24, wm_in_loop=e["loop"])
        res = train_run(cfg, results_dir=None, verbose=False)
    finally:
        T.build_model = orig_build
    with open(path + ".tmp", "w") as f:
        json.dump(res, f, indent=2)
    os.replace(path + ".tmp", path)
    a = res["final_acc"]; d = res["compute_diag"]
    print(f"{tag:12s} corto={a['corto']:.3f} medio={a['medio']:.3f} largo={a['largo']:.3f} "
          f"niter={d['mean_niter']:.1f} corr={d['corr_K_niter']:+.3f} ({(time.time()-t0)/60:.0f} min)", flush=True)
print("PROBE v2e COMPLETO")
