"""
N1b-LLM — VG0: bring-up + gates W (¿camina?) y R (rango de la palanca)
sobre dev (PREREG_N1B_LLM §5). Dos celdas de temperatura: greedy y 0.3.

Por instancia: UNA generación larga; se graba texto, tokens, hops
parseados, corrección — el patrón de la sonda. Reporta por celda:
  W: acc sin cap (global y por L), compliance de formato.
  R: corr(c_tok, d | correcta), CV(c|L), R²(c|L) (debe ser BAJO: el
     acantilado), linealidad hops↔tokens.

  PYTHONPATH=. python -m mhbp.tasks.llm_gov.llm_n1b_vg0
"""
import json
import os
import time

import numpy as np
import torch

os.environ.setdefault("HF_HOME",
                      r"E:\ARTICULOS-CIENTIFICOS\MIURACOGNITIVE\hf_cache")
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from .llm_env3 import LlmSpec, LlmTaskStream, build_messages, parse, \
    parse_hops

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MODEL = "Qwen/Qwen2.5-14B-Instruct"
N_INST = 96                      # 32 por clase L
MAX_NEW = 700
BATCH = 24
CELDAS = [("greedy", None), ("T03", 0.3)]


def r2(y, x):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(y) < 8 or y.std() < 1e-9:
        return 0.0
    A = np.stack([x, np.ones_like(x)], 1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return float(1 - resid.var() / y.var())


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    tok.padding_side = "left"
    spec = LlmSpec()
    out = {}
    t0 = time.time()
    for tag, temp in CELDAS:
        stream = LlmTaskStream(spec, "dev")
        insts = []
        for L in spec.l_set:
            insts += [stream.sample(L=L) for _ in range(N_INST // 3)]
        regs = []
        for bi in range(0, len(insts), BATCH):
            chunk = insts[bi:bi + BATCH]
            prompts = [tok.apply_chat_template(
                build_messages(i), tokenize=False,
                add_generation_prompt=True) for i in chunk]
            enc = tok(prompts, return_tensors="pt",
                      padding=True).to("cuda")
            gkw = dict(max_new_tokens=MAX_NEW,
                       pad_token_id=tok.eos_token_id)
            if temp is None:
                gkw["do_sample"] = False
            else:
                gkw.update(do_sample=True, temperature=temp, top_p=0.9)
            with torch.no_grad():
                o = model.generate(**enc, **gkw)
            gen = o[:, enc["input_ids"].shape[1]:]
            txts = tok.batch_decode(gen, skip_special_tokens=True)
            nt = (gen != tok.eos_token_id).sum(dim=1).tolist()
            for inst, tx, n in zip(chunk, txts, nt):
                hops = parse_hops(tx)
                regs.append({"L": inst["L"], "d": inst["d"],
                             "stake": inst["stake"],
                             "resp": parse(tx), "tokens": int(n),
                             "n_hops": len(hops),
                             "ok": parse(tx) == inst["truth"],
                             "text": tx[-400:]})
        acc = float(np.mean([r["ok"] for r in regs]))
        accL = {L: float(np.mean([r["ok"] for r in regs if r["L"] == L]))
                for L in spec.l_set}
        compl = float(np.mean([(r["resp"] is not None
                                and r["n_hops"] > 0) for r in regs]))
        okr = [r for r in regs if r["ok"]]
        cd = (float(np.corrcoef([r["tokens"] for r in okr],
                                [r["d"] for r in okr])[0, 1])
              if len(okr) > 8 else 0.0)
        cvL = {L: (lambda v: float(np.std(v) / (np.mean(v) + 1e-9)))(
            [r["tokens"] for r in okr if r["L"] == L])
            for L in spec.l_set}
        r2_cL = r2([r["tokens"] for r in okr], [r["L"] for r in okr])
        hop_lin = (float(np.corrcoef([r["tokens"] for r in okr],
                                     [r["n_hops"] for r in okr])[0, 1])
                   if len(okr) > 8 else 0.0)
        tokL = {L: float(np.mean([r["tokens"] for r in regs
                                  if r["L"] == L])) for L in spec.l_set}
        gate_W = bool(acc >= 0.55 and accL[14] >= 0.35 and compl >= 0.80)
        gate_R = bool(cd >= 0.6 and min(cvL.values()) >= 0.3
                      and r2_cL <= 0.5)
        out[tag] = {"acc": acc, "acc_por_L": accL, "compliance": compl,
                    "corr_tok_d": cd, "cv_tok_por_L": cvL,
                    "r2_tok_L": r2_cL, "corr_hops_tok": hop_lin,
                    "tokens_por_L": tokL, "gate_W": gate_W,
                    "gate_R": gate_R, "regs": regs}
        print(f"{tag}: acc={acc:.3f} " +
              " ".join(f"L{L}:{a:.2f}" for L, a in accL.items()) +
              f" compl={compl:.2f} corr(c,d)={cd:+.2f} "
              f"R²(c|L)={r2_cL:.2f} CVmin={min(cvL.values()):.2f} "
              f"hops~tok={hop_lin:+.2f} "
              f"W={'✓' if gate_W else '✗'} R={'✓' if gate_R else '✗'} "
              f"[{(time.time() - t0) / 60:.0f} min]", flush=True)
    with open(os.path.join(RES, "llm_n1b_vg0.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("Guardado llm_n1b_vg0.json", flush=True)


if __name__ == "__main__":
    main()
