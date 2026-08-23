"""
F3a — Diagnóstico de LESIÓN POR VÍAS (rama 3 del PREREG_F3A; FINDINGS_F3A).

Sobre los ckpts OOD de miura_mhbp: M3 on-policy con cada vía de modulación
neutralizada EN EVAL (halt / wm / gate / todas), × 6 seeds. La vía cuya
lesión RECUPERA el acople (hacia el 0.735 del incumbente) es la que
descorrelaciona fidelidad-resultado. Probes re-entrenadas POR CONDICIÓN
(la distribución de trayectorias cambia bajo lesión). Post-hoc etiquetado.

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.f3a_lesion
"""
import json
import os
import time

import torch

from .g0_run import load_model
from .f3a_run import mk_ds, onpolicy_fidelity
from .instruments import collect_trajectories, train_probes, pearson

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEEDS = [0, 1, 2, 3, 4, 5]
CONDITIONS = {"baseline": set(), "sin_halt": {"halt"}, "sin_wm": {"wm"},
              "sin_gate": {"gate"}, "sin_todas": {"halt", "wm", "gate"}}
REF_INCUMBENTE = 0.735


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    path = os.path.join(RES, "f3a_lesion.json")
    results = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    for seed in SEEDS:
        for cond, les in CONDITIONS.items():
            key = f"s{seed}_{cond}"
            if key in results:
                print(f"skip {key}", flush=True)
                continue
            t0 = time.time()
            model = load_model(f"miura_mhbp_cycle_transp_ood_s{seed}.pt",
                               "miura_mhbp", "cycle_transp", device, dtype)
            model.hbp.lesion = set(les)
            tr = collect_trajectories(model, mk_ds(seed + 500_000, mw=12),
                                      2560, device, dtype)
            heads = train_probes(tr["feat_ans"], tr["ans"], 120, device)
            ood = collect_trajectories(model, mk_ds(seed + 600_000), 768,
                                       device, dtype, k_min=14)
            fid = onpolicy_fidelity(heads, ood["feat_ans"], ood["ans"],
                                    ood["n_exp"], device)
            results[key] = {
                "m3_onpolicy": pearson(fid, ood["correct"].float()),
                "ood_acc": float(ood["correct"].float().mean()),
                "mean_nexp": float(ood["n_exp"].mean()),
                "wall_s": round(time.time() - t0, 1),
            }
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=1)
            os.replace(path + ".tmp", path)
            print(f"OK {key}: M3onp={results[key]['m3_onpolicy']:+.3f} "
                  f"acc={results[key]['ood_acc']:.3f} "
                  f"E[n]={results[key]['mean_nexp']:.1f} "
                  f"({results[key]['wall_s']}s)", flush=True)
            model.hbp.lesion = set()
            del model
            torch.cuda.empty_cache()

    # tabla resumen
    import numpy as np
    print("\n=== LESIÓN POR VÍAS — M3 on-policy (media 6 seeds; ref. incumbente "
          f"{REF_INCUMBENTE}) ===")
    for cond in CONDITIONS:
        vals = [results[f"s{s}_{cond}"]["m3_onpolicy"] for s in SEEDS
                if f"s{s}_{cond}" in results]
        if vals:
            rec = np.mean(vals) - 0.679          # vs baseline mhbp
            print(f"  {cond:10s}: {np.mean(vals):.3f} ± {np.std(vals):.3f} "
                  f"(Δ vs baseline mhbp {rec:+.3f})")
    print("LESIÓN COMPLETA")


if __name__ == "__main__":
    main()
