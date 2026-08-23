"""
Revisión TMLR — CELDA ANCLA GSM8K (petición del revisor de arquitecturas):
reconciliar la ley empírica ε ≤ 0.08 (dos familias sintéticas) con los
resultados publicados de self-consistency en GSM8K (+10-18 pp).

Mismos instrumentos del capítulo: m=16 muestras a T=0.7/top_p=0.9,
estimador de voto INSESGADO (subconjuntos sin reemplazo), modal-erróneo,
ε = (1−acc₁) − P(modal erróneo). 256 instancias del test oficial.

  PYTHONPATH=. python -m mhbp.tasks.llm_gov.llm_gsm8k
"""
import json
import os
import re
import time
import urllib.request

import numpy as np
import torch

os.environ.setdefault("HF_HOME",
                      r"E:\ARTICULOS-CIENTIFICOS\MIURACOGNITIVE\hf_cache")
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from .llm_gates import vote_curve, N_GRID

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CACHE = os.path.join(HERE, "cache")
MODEL = "Qwen/Qwen2.5-14B-Instruct"
URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
       "master/grade_school_math/data/test.jsonl")
N_INST, M, T, MAX_NEW, BATCH_INST = 256, 16, 0.7, 448, 6
RNG = np.random.default_rng(20260817)

SYSTEM = ("You are a careful math problem solver. Reason step by step, "
          "and ALWAYS finish with a final line of the exact form: "
          "ANSWER: <integer>")
RE_ANS = re.compile(r"ANSWER:\s*\$?(-?[\d,]+)")


def parse(text):
    m = RE_ANS.findall(text or "")
    if not m:
        # tolera formato #### del dataset si el modelo lo imita
        m = re.findall(r"####\s*\$?(-?[\d,]+)", text or "")
    if not m:
        return None
    try:
        return str(int(m[-1].replace(",", "")))
    except ValueError:
        return None


def cargar_gsm():
    path = os.path.join(CACHE, "gsm8k_test.jsonl")
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL, path)
    items = [json.loads(l) for l in open(path, encoding="utf-8")]
    idx = RNG.permutation(len(items))[:N_INST]
    out = []
    for i in idx:
        it = items[int(i)]
        gold = re.findall(r"####\s*\$?(-?[\d,]+)", it["answer"])[-1]
        out.append({"q": it["question"],
                    "truth": str(int(gold.replace(",", "")))})
    return out


def main():
    insts = cargar_gsm()
    print(f"GSM8K: {len(insts)} instancias (test oficial)", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    tok.padding_side = "left"
    path = os.path.join(CACHE, "gsm8k_pool.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            done.add(json.loads(l)["q"][:80])
    fout = open(path, "a", encoding="utf-8")
    t0 = time.time()
    todo = [x for x in insts if x["q"][:80] not in done]
    for bi in range(0, len(todo), BATCH_INST):
        chunk = todo[bi:bi + BATCH_INST]
        prompts = []
        for it in chunk:
            p = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": it["q"]}],
                tokenize=False, add_generation_prompt=True)
            prompts.extend([p] * M)
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=MAX_NEW,
                               do_sample=True, temperature=T, top_p=0.9,
                               pad_token_id=tok.eos_token_id)
        gen = o[:, enc["input_ids"].shape[1]:]
        txts = tok.batch_decode(gen, skip_special_tokens=True)
        nt = (gen != tok.eos_token_id).sum(dim=1).tolist()
        for j, it in enumerate(chunk):
            sl = slice(j * M, (j + 1) * M)
            rec = {"q": it["q"][:80], "truth": it["truth"],
                   "resp": [parse(t) for t in txts[sl]],
                   "tok": [int(x) for x in nt[sl]]}
            fout.write(json.dumps(rec) + "\n")
        fout.flush()
        if (bi // BATCH_INST) % 8 == 0:
            print(f"[{min(bi + BATCH_INST, len(todo))}/{len(todo)}] "
                  f"{(time.time() - t0) / 60:.0f} min", flush=True)
    fout.close()

    # ---- análisis con los instrumentos del capítulo ---- #
    data = [json.loads(l) for l in open(path, encoding="utf-8")]
    curves = np.array([vote_curve(d) for d in data])
    acc = np.nanmean(curves, axis=0)
    acc1 = float(acc[0])
    modal_err = []
    for d in data:
        cnt = {}
        for r in d["resp"]:
            if r is not None:
                cnt[r] = cnt.get(r, 0) + 1
        modal_err.append(1.0 if not cnt
                         else float(max(cnt, key=cnt.get) != d["truth"]))
    me = float(np.mean(modal_err))
    eps = (1 - acc1) - me
    res = {"n_grid": N_GRID, "acc_voto": acc.tolist(), "acc1": acc1,
           "modal_err": me, "epsilon": eps,
           "pendiente_n1_n15": float(acc[-1] - acc[0]),
           "n": len(data), "T": T, "m": M}
    print("acc(voto): " + " ".join(f"n{n}:{a:.3f}"
                                   for n, a in zip(N_GRID, acc)), flush=True)
    print(f"acc1={acc1:.3f} modal_err={me:.3f} "
          f"pendiente(n1→n15)={acc[-1] - acc[0]:+.3f} ε={eps:+.3f} "
          f"(la ley sintética decía ε≤0.08)", flush=True)
    with open(os.path.join(RES, "llm_gsm8k.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("Guardado llm_gsm8k.json", flush=True)


if __name__ == "__main__":
    main()
