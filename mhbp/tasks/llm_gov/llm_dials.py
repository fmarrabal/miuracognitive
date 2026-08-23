"""
Capítulo LLM — RONDA 2 de diales VG-L0 (≤3 declaradas; §6), sobre DEV.

Motivo (registrado): con K∈{9,10,11} y T=0.7 la curva del voto es PLANA
(acc 0.132→0.140 de n=1 a n=11) porque el modal del pool es ERRÓNEO en el
86% de las instancias: los errores del actuador son SISTEMÁTICOS, no
diversos. Sin palanca no hay asignación que medir (VG-L2 +0.002).

Este diagnóstico separa las dos causas candidatas con celdas pequeñas:
  · T más alta (1.0, 1.3) a K fijo   → ¿decorrelaciona los errores?
  · K más bajo (5,6) a T fija        → ¿basta con acc(1) mayor?
Criterio de la ronda: pendiente del voto (acc n=11 − n=1) y fracción
modal-erróneo. Nada de esto toca sonda/val/eval.

  PYTHONPATH=. python -m mhbp.tasks.llm_gov.llm_dials
"""
import json
import os
import time

import numpy as np
import torch

os.environ.setdefault("HF_HOME",
                      r"E:\ARTICULOS-CIENTIFICOS\MIURACOGNITIVE\hf_cache")
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from .llm_gates import vote_curve, N_GRID

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MODEL = "Qwen/Qwen2.5-14B-Instruct"
FAMILIA = os.environ.get("LLM_FAMILIA", "s5")
if FAMILIA == "arit":
    from .llm_env2 import (LlmSpec, LlmTaskStream, build_messages,  # noqa
                           parse)
    # familia 2: barrido de dificultad (K) y de temperatura sobre dev
    # ronda 3: multiplicadores de 2 dígitos (ver LlmSpec) — K y T
    CELDAS = [((4, 5), 0.7), ((6, 7), 0.7), ((8, 9), 0.7), ((6, 7), 1.0)]
    N_INST, M, MAX_NEW = 64, 16, 700
else:
    from .llm_env import (LlmSpec, LlmTaskStream, build_messages,  # noqa
                          parse)
    CELDAS = [((9, 11), 1.0), ((9, 11), 1.3), ((5, 6), 0.7), ((5, 6), 1.0)]
    N_INST, M, MAX_NEW = 96, 16, 640


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    tok.padding_side = "left"
    out = {}
    t0 = time.time()
    for (k_lo, k_hi), T in CELDAS:
        spec = LlmSpec(k_lo=k_lo, k_hi=k_hi)
        stream = LlmTaskStream(spec, "dev")
        insts = [stream.sample() for _ in range(N_INST)]
        data = []
        per_batch = max(1, 64 // M)
        for bi in range(0, len(insts), per_batch):
            chunk = insts[bi:bi + per_batch]
            prompts = []
            for inst in chunk:
                p = tok.apply_chat_template(build_messages(inst),
                                            tokenize=False,
                                            add_generation_prompt=True)
                prompts.extend([p] * M)
            enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=MAX_NEW,
                                   do_sample=True, temperature=T, top_p=0.9,
                                   pad_token_id=tok.eos_token_id)
            gen = o[:, enc["input_ids"].shape[1]:]
            txts = tok.batch_decode(gen, skip_special_tokens=True)
            ntok = (gen != tok.eos_token_id).sum(dim=1).tolist()
            for j, inst in enumerate(chunk):
                sl = slice(j * M, (j + 1) * M)
                data.append({"K": inst["K"], "stake": inst["stake"],
                             "truth": inst["truth"],
                             "resp": [parse(t) for t in txts[sl]],
                             "tok": np.array(ntok[sl], dtype=float),
                             "t_med": float(np.mean(ntok[sl]))})
        curvas = np.mean([vote_curve(d) for d in data], axis=0)
        modal_err = []
        for d in data:
            cnt = {}
            for r in d["resp"]:
                if r is not None:
                    cnt[r] = cnt.get(r, 0) + 1
            modal_err.append(1.0 if not cnt else
                             float(max(cnt, key=cnt.get) != d["truth"]))
        key = f"K{k_lo}-{k_hi}_T{T}"
        out[key] = {"acc_voto": curvas.tolist(),
                    "pendiente": float(curvas[-1] - curvas[0]),
                    "modal_erroneo": float(np.mean(modal_err)),
                    "t_med": float(np.mean([d["t_med"] for d in data]))}
        print(f"{key:16s} acc(n): " +
              " ".join(f"{a:.3f}" for a in curvas) +
              f" | pendiente={out[key]['pendiente']:+.3f} "
              f"modal-err={out[key]['modal_erroneo']:.3f} "
              f"t̄={out[key]['t_med']:.0f} [{(time.time()-t0)/60:.0f} min]",
              flush=True)
    suf = "_arit" if FAMILIA == "arit" else "_r2"
    with open(os.path.join(RES, f"llm_dials{suf}.json"), "w",
              encoding="utf-8") as f:
        json.dump({"n_grid": N_GRID, "celdas": out}, f, indent=1)
    print("Guardado llm_dials_r2.json", flush=True)


if __name__ == "__main__":
    main()
