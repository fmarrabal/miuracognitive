"""
Capítulo LLM — bring-up smoke: carga Qwen2.5-14B-Instruct local, chat
template, generación BATCHEADA con muestreo, parse de una respuesta S₅
simple y THROUGHPUT real (el número para el prereg v2).

  PYTHONPATH=. HF_HOME=... python -m mhbp.tasks.llm_gov.llm_smoke
"""
import os
import re
import time

import torch

os.environ.setdefault("HF_HOME",
                      r"E:\ARTICULOS-CIENTIFICOS\MIURACOGNITIVE\hf_cache")
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL = "Qwen/Qwen2.5-14B-Instruct"

SYSTEM = ("Eres un ejecutor preciso de operaciones sobre una fila de 5 "
          "fichas. Operaciones: R = rotar la fila un puesto a la derecha "
          "(la última ficha pasa a ser la primera); S = intercambiar las "
          "dos primeras fichas. Aplica las operaciones EN ORDEN a la fila "
          "inicial A B C D E y responde con la fila final en la última "
          "línea con el formato exacto: RESPUESTA: X X X X X")

FEWSHOT = [("Operaciones: S R",
            "Empiezo con A B C D E.\nS: B A C D E.\nR: E B A C D.\n"
            "RESPUESTA: E B A C D")]


def compose(ops):
    fila = list("ABCDE")
    for o in ops:
        if o == "R":
            fila = [fila[-1]] + fila[:-1]
        else:
            fila[0], fila[1] = fila[1], fila[0]
    return " ".join(fila)


def parse(text):
    m = re.findall(r"RESPUESTA:\s*([A-E])\s+([A-E])\s+([A-E])\s+([A-E])\s+([A-E])",
                   text)
    return " ".join(m[-1]) if m else None


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"carga: {time.time() - t0:.0f}s | "
          f"VRAM {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    import random
    random.seed(0)
    prompts, truths = [], []
    for _ in range(16):
        ops = [random.choice("RS") for _ in range(8)]
        msgs = [{"role": "system", "content": SYSTEM}]
        for q, a in FEWSHOT:
            msgs += [{"role": "user", "content": q},
                     {"role": "assistant", "content": a}]
        msgs.append({"role": "user",
                     "content": "Operaciones: " + " ".join(ops)})
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))
        truths.append(compose(ops))

    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=220, do_sample=True,
                             temperature=0.7, top_p=0.9,
                             pad_token_id=tok.eos_token_id)
    dt = time.time() - t1
    gen = out[:, enc["input_ids"].shape[1]:]
    n_tok = int((gen != tok.eos_token_id).sum())
    texts = tok.batch_decode(gen, skip_special_tokens=True)
    n_parse, n_ok = 0, 0
    for txt, truth in zip(texts, truths):
        p = parse(txt)
        if p is not None:
            n_parse += 1
            n_ok += int(p == truth)
    print(f"batch 16 × ≤220 tok: {dt:.0f}s → {n_tok / dt:.0f} tok/s "
          f"(batcheado) | parseadas {n_parse}/16 | correctas {n_ok}/16 "
          f"(K=8)", flush=True)
    print("EJEMPLO:\n" + texts[0][-300:], flush=True)


if __name__ == "__main__":
    main()
