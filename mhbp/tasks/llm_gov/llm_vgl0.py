"""
Capítulo LLM — VG-L0 mini-piloto (PREREG_LLM v2 §6), stream DEV.

  gate de convención: acc(K∈{1,2}, greedy) ≥ 0.95
  ventana: acc(1 muestra, T) ∈ [0.1, 0.8] en algún K-subrango
  no-parseo < 5% por celda; t̄(K) para la moneda de tokens (§4/§9)

  PYTHONPATH=. python -m mhbp.tasks.llm_gov.llm_vgl0
"""
import json
import os
import time

import numpy as np
import torch

os.environ.setdefault("HF_HOME",
                      r"E:\ARTICULOS-CIENTIFICOS\MIURACOGNITIVE\hf_cache")
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from .llm_env import LlmSpec, LlmTaskStream, build_messages, parse

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
MODEL = "Qwen/Qwen2.5-14B-Instruct"
TEMP = 0.7                       # dial de VG-L0; se congela con timestamp
K_GRID = [4, 6, 8, 10, 12, 14, 16]
M_POR_K = 48
MAX_NEW = 640                    # cap duro de tokens por muestra (§ coste)


@torch.no_grad()
def gen_batch(model, tok, prompts, do_sample):
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**enc, max_new_tokens=MAX_NEW,
                         do_sample=do_sample,
                         temperature=TEMP if do_sample else None,
                         top_p=0.9 if do_sample else None,
                         pad_token_id=tok.eos_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    texts = tok.batch_decode(gen, skip_special_tokens=True)
    ntok = (gen != tok.eos_token_id).sum(dim=1).tolist()
    return texts, ntok


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    spec = LlmSpec(k_lo=1, k_hi=24)
    stream = LlmTaskStream(spec, "dev")
    out = {"temp": TEMP, "max_new": MAX_NEW, "celdas": {}}

    # --- gate de convención: K ∈ {1, 2}, GREEDY ---
    for K in (1, 2):
        insts = [stream.sample(k=K) for _ in range(24)]
        texts, _ = gen_batch(model, tok,
                             [tok.apply_chat_template(
                                 build_messages(i), tokenize=False,
                                 add_generation_prompt=True) for i in insts],
                             do_sample=False)
        acc = float(np.mean([parse(t) == i["truth"]
                             for t, i in zip(texts, insts)]))
        out["celdas"][f"conv_K{K}"] = {"acc_greedy": acc}
        print(f"convención K={K} (greedy): acc={acc:.3f}", flush=True)

    # --- ventana: 1 muestra a T por K de la rejilla ---
    for K in K_GRID:
        insts = [stream.sample(k=K) for _ in range(M_POR_K)]
        prompts = [tok.apply_chat_template(build_messages(i), tokenize=False,
                                           add_generation_prompt=True)
                   for i in insts]
        accs, parses, toks = [], [], []
        for i in range(0, M_POR_K, 48):
            texts, ntok = gen_batch(model, tok, prompts[i:i + 48], True)
            for t, inst in zip(texts, insts[i:i + 48]):
                p = parse(t)
                parses.append(p is not None)
                accs.append(p == inst["truth"])
            toks.extend(ntok)
        out["celdas"][f"K{K}"] = {
            "acc_1muestra": float(np.mean(accs)),
            "parse_rate": float(np.mean(parses)),
            "t_med": float(np.median(toks)),
            "t_p90": float(np.percentile(toks, 90))}
        c = out["celdas"][f"K{K}"]
        print(f"K={K:>2}: acc={c['acc_1muestra']:.3f} "
              f"parse={c['parse_rate']:.3f} t̄={c['t_med']:.0f} "
              f"(p90 {c['t_p90']:.0f}) [{(time.time() - t0) / 60:.0f} min]",
              flush=True)

    conv_ok = all(out["celdas"][f"conv_K{k}"]["acc_greedy"] >= 0.95
                  for k in (1, 2))
    ventana = [K for K in K_GRID
               if 0.1 <= out["celdas"][f"K{K}"]["acc_1muestra"] <= 0.8
               and out["celdas"][f"K{K}"]["parse_rate"] >= 0.95]
    out["veredicto"] = {"convencion": conv_ok, "ventana_K": ventana}
    print(f"\nVG-L0: convención {'PASA' if conv_ok else 'FALLA'}; "
          f"ventana K con pendiente y parseo: {ventana}", flush=True)
    with open(os.path.join(RES, "llm_vgl0.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("Guardado llm_vgl0.json", flush=True)


if __name__ == "__main__":
    main()
