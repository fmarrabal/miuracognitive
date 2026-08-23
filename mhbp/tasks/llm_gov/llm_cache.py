"""
Capítulo LLM — Generador del CACHÉ único (PREREG_LLM v2 §3).

Por instancia × semilla-de-muestreo: m muestras i.i.d. de CoT (T congelada
de VG-L0), guardando por muestra: respuesta parseada, tokens generados,
logprob media (por RESCORING batcheado — una pasada teacher-forcing, no
output_scores en memoria), orden de muestreo. Todos los brazos serán
funciones deterministas de este caché.

  python -m mhbp.tasks.llm_gov.llm_cache --split sonda --n_inst 1024 --m 16
  (K-rango: de results/llm_vgl0.json salvo --k_lo/--k_hi explícitos)
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

from .llm_env import LlmSpec, LlmTaskStream, build_messages, parse, word_hash

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
MODEL = "Qwen/Qwen2.5-14B-Instruct"
MAX_NEW = 640


@torch.no_grad()
def rescore_logprob(model, tok, full_ids, prompt_len):
    """logprob media por token generado, vía teacher forcing batcheado."""
    out = model(full_ids).logits[:, :-1, :].float().log_softmax(-1)
    tgt = full_ids[:, 1:]
    lp = out.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    means = []
    for b in range(full_ids.shape[0]):
        gen = lp[b, prompt_len[b] - 1:]
        mask = tgt[b, prompt_len[b] - 1:] != tok.eos_token_id
        n = int(mask.sum())
        means.append(float(gen[mask].mean()) if n else 0.0)
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True,
                    choices=("sonda", "val", "dev", "eval"))
    ap.add_argument("--n_inst", type=int, required=True)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--seed_muestreo", type=int, default=0)
    ap.add_argument("--k_lo", type=int, default=None)
    ap.add_argument("--k_hi", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--logprobs", action="store_true")
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    vg = json.load(open(os.path.join(RES, "llm_vgl0.json"), encoding="utf-8"))
    temp = args.temp if args.temp is not None else vg["temp"]
    ks = vg["veredicto"]["ventana_K"]
    k_lo = args.k_lo if args.k_lo is not None else min(ks)
    k_hi = args.k_hi if args.k_hi is not None else max(ks)
    print(f"caché {args.split}: K∈[{k_lo},{k_hi}] T={temp} m={args.m} "
          f"seed_muestreo={args.seed_muestreo}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    # instancias: dedupe contra los DEMÁS splits por hash (regenera los
    # streams ligeros y prohíbe sus palabras — §2)
    spec = LlmSpec(k_lo=k_lo, k_hi=k_hi)
    banned = set()
    for other in ("sonda", "val", "dev", "eval"):
        if other == args.split:
            continue
        s = LlmTaskStream(spec, other)
        for _ in range(args.n_inst * 2):
            banned.add(s.sample()["hash"])
    stream = LlmTaskStream(spec, args.split, banned_hashes=banned)
    insts = [stream.sample() for _ in range(args.n_inst)]

    suf = f"_{args.tag}" if args.tag else ""
    path = os.path.join(
        CACHE, f"cache_{args.split}{suf}_s{args.seed_muestreo}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["hash"])
    print(f"{len(done)} instancias ya cacheadas (resume)", flush=True)

    torch.manual_seed(args.seed_muestreo * 7919 + 13)
    t0 = time.time()
    fout = open(path, "a", encoding="utf-8")
    tok.padding_side = "left"
    todo = [i for i in insts if i["hash"] not in done]
    per_batch = max(1, args.batch // args.m)
    for bi in range(0, len(todo), per_batch):
        chunk = todo[bi:bi + per_batch]
        prompts = []
        for inst in chunk:
            p = tok.apply_chat_template(build_messages(inst), tokenize=False,
                                        add_generation_prompt=True)
            prompts.extend([p] * args.m)          # m muestras por instancia
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True,
                             temperature=temp, top_p=0.9,
                             pad_token_id=tok.eos_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        ntoks = (gen != tok.eos_token_id).sum(dim=1).tolist()
        lps = [None] * len(texts)
        if args.logprobs:
            plens = enc["attention_mask"].sum(dim=1).tolist()
            lps = rescore_logprob(model, tok, out, plens)
        for j, inst in enumerate(chunk):
            muestras = []
            for k in range(args.m):
                idx = j * args.m + k
                muestras.append({"resp": parse(texts[idx]),
                                 "tokens": int(ntoks[idx]),
                                 "logprob": lps[idx],
                                 # texto crudo (recortado): permite recomputar
                                 # canales del posterior sin re-inferencia
                                 "text": texts[idx][-600:]})
            rec = {"hash": inst["hash"], "K": inst["K"],
                   "stake": inst["stake"], "meta_slots": inst["meta_slots"],
                   "truth": inst["truth"], "muestras": muestras}
            fout.write(json.dumps(rec) + "\n")
        fout.flush()
        n_done = min(bi + per_batch, len(todo))
        if (bi // per_batch) % 8 == 0:
            tot_tok = sum(ntoks)
            print(f"[{n_done}/{len(todo)}] ~{tot_tok / max(1e-9, 1):,} tok "
                  f"último lote | {(time.time() - t0) / 60:.0f} min",
                  flush=True)
    fout.close()
    print(f"CACHÉ {args.split} COMPLETO en {(time.time() - t0) / 60:.1f} min "
          f"→ {path}", flush=True)


if __name__ == "__main__":
    main()
