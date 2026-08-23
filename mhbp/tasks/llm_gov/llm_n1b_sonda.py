"""
N1b-LLM — SONDA (PREREG_N1B_LLM v2 §7.6): UNA generación GREEDY por
instancia, persistiendo los TOKEN-IDS COMPLETOS (§7.1: el truncado
honesto exige re-parseo de prefijos — el texto recortado no basta).

Dos familias:
  ciclo : 768 instancias (256 por L ∈ {6,10,14}), split sonda (7001),
          dedupe contra el dev del VG0 (hash canónico).
  arit  : 384 instancias (96 por K ∈ {4,5,6,7}), llm_env2 spec ronda 3
          (mul 23-97) — el contraste suave y el control positivo de M.

  PYTHONPATH=. python -m mhbp.tasks.llm_gov.llm_n1b_sonda --familia ciclo
  PYTHONPATH=. python -m mhbp.tasks.llm_gov.llm_n1b_sonda --familia arit
"""
import argparse
import json
import os
import time

import numpy as np
import torch

os.environ.setdefault("HF_HOME",
                      r"E:\ARTICULOS-CIENTIFICOS\MIURACOGNITIVE\hf_cache")
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CACHE = os.path.join(HERE, "cache")
MODEL = "Qwen/Qwen2.5-14B-Instruct"
MAX_NEW = 900
BATCH = 24


def instancias_ciclo():
    from .llm_env3 import LlmSpec, LlmTaskStream
    spec = LlmSpec()
    # dedupe contra dev (VG0 consumió 96 del stream dev; ambas celdas
    # idénticas). Margen: baneamos 192.
    dev = LlmTaskStream(spec, "dev")
    banned = set()
    for L in spec.l_set:
        for _ in range(64):
            banned.add(dev.sample(L=L)["hash"])
    st = LlmTaskStream(spec, "sonda", banned_hashes=banned)
    insts = []
    for L in spec.l_set:
        insts += [st.sample(L=L) for _ in range(256)]
    from .llm_env3 import build_messages, parse, parse_hops
    return insts, build_messages, parse, parse_hops, "clase_L"


def instancias_arit():
    from .llm_env2 import LlmSpec, LlmTaskStream, build_messages, parse
    insts = []
    for K in (4, 5, 6, 7):
        st = LlmTaskStream(LlmSpec(k_lo=K, k_hi=K), "sonda")
        for _ in range(96):
            i = st.sample(k=K)
            i["L"] = K                       # clase visible homogénea
            i["d"] = K
            insts.append(i)
    return insts, build_messages, parse, (lambda t: []), "clase_K"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--familia", required=True, choices=("ciclo", "arit"))
    args = ap.parse_args()
    if args.familia == "ciclo":
        insts, build_messages, parse, parse_hops, _ = instancias_ciclo()
    else:
        insts, build_messages, parse, parse_hops, _ = instancias_arit()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    tok.padding_side = "left"
    path = os.path.join(CACHE, f"n1b_sonda_{args.familia}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["hash"])
    print(f"{args.familia}: {len(insts)} instancias, {len(done)} hechas "
          f"(resume)", flush=True)
    todo = [i for i in insts if i["hash"] not in done]
    fout = open(path, "a", encoding="utf-8")
    t0 = time.time()
    for bi in range(0, len(todo), BATCH):
        chunk = todo[bi:bi + BATCH]
        prompts = [tok.apply_chat_template(
            build_messages(i), tokenize=False,
            add_generation_prompt=True) for i in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=MAX_NEW,
                               do_sample=False,
                               pad_token_id=tok.eos_token_id)
        gen = o[:, enc["input_ids"].shape[1]:]
        for inst, ids_t in zip(chunk, gen):
            ids = ids_t.tolist()
            # recorta el padding EOS final (conserva un EOS si lo hay)
            while len(ids) > 1 and ids[-1] == tok.eos_token_id \
                    and ids[-2] == tok.eos_token_id:
                ids.pop()
            txt = tok.decode(ids, skip_special_tokens=True)
            rec = {"hash": inst["hash"], "L": inst.get("L"),
                   "d": inst.get("d"), "stake": inst["stake"],
                   "meta_slots": list(inst["meta_slots"]),
                   "truth": inst["truth"], "resp": parse(txt),
                   "n_hops": len(parse_hops(txt)),
                   "tokens": len(ids), "ids": ids}
            fout.write(json.dumps(rec) + "\n")
        fout.flush()
        if (bi // BATCH) % 4 == 0:
            print(f"[{min(bi + BATCH, len(todo))}/{len(todo)}] "
                  f"{(time.time() - t0) / 60:.0f} min", flush=True)
    fout.close()
    print(f"SONDA {args.familia} COMPLETA en "
          f"{(time.time() - t0) / 60:.1f} min → {path}", flush=True)


if __name__ == "__main__":
    main()
