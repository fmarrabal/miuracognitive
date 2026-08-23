"""
Capítulo LLM — MEDICIÓN DECISIVA DE LA PALANCA (post-panel de cierre).

El panel encontró dos sesgos que apuntaban al mismo sitio: (1) el estimador
del voto era plug-in CON reemplazo (sesgo negativo que crece con la
diversidad del pool) y (2) n_max=11 era un tope de diseño, no saturación
(con n̄=3 y p_hi=0.15 el asignador puede pagar ~14-20 muestras a las
instancias caras). Corregidos, el techo de la mejor celda superaría el
umbral pre-registrado de 0.04.

Este script mide la curva del voto SIN sesgo y HASTA donde el presupuesto
permite, en la celda decisiva (S₅, K∈{9,10,11}, T=1.3), con m=32 muestras
por instancia y IC bootstrap por instancia.

  PYTHONPATH=. python -m mhbp.tasks.llm_gov.llm_lever
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
MODEL = "Qwen/Qwen2.5-14B-Instruct"
T_DEC = 1.3
M_POOL = 32
N_INST = 96
MAX_NEW = 640
N_GRID = [1, 3, 5, 7, 9, 11, 15, 19, 23, 27, 31]
R_MC = 4000
RNG = np.random.default_rng(20260808)


def curva_exacta(resp, truth, n_grid=N_GRID, R=R_MC, rng=RNG):
    m = len(resp)
    cl = {}
    for r in resp:
        cl[r] = cl.get(r, 0) + 1
    keys = list(cl.keys())
    cuentas = np.array([cl[k] for k in keys])
    i_none = keys.index(None) if None in keys else -1
    i_ok = keys.index(truth) if truth in keys else -1
    out = []
    for n in n_grid:
        if n > m:
            out.append(np.nan)
            continue
        if i_ok < 0:
            out.append(0.0)
            continue
        d = rng.multivariate_hypergeometric(cuentas, n, size=R)
        v = d.copy()
        if i_none >= 0:
            v[:, i_none] = 0
        mx = v.max(axis=1)
        emp = (v == mx[:, None]).sum(axis=1)
        gana = (v[:, i_ok] == mx) & (mx > 0)
        out.append(float(np.where(gana, 1.0 / emp, 0.0).mean()))
    return np.array(out)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    tok.padding_side = "left"
    spec = LlmSpec(k_lo=9, k_hi=11)
    stream = LlmTaskStream(spec, "dev")
    insts = [stream.sample() for _ in range(N_INST)]
    t0 = time.time()
    pools, toks = [], []
    for bi in range(0, N_INST, 2):                 # 2 inst × 32 = 64 secuencias
        chunk = insts[bi:bi + 2]
        prompts = []
        for inst in chunk:
            p = tok.apply_chat_template(build_messages(inst), tokenize=False,
                                        add_generation_prompt=True)
            prompts.extend([p] * M_POOL)
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True,
                               temperature=T_DEC, top_p=0.9,
                               pad_token_id=tok.eos_token_id)
        gen = o[:, enc["input_ids"].shape[1]:]
        txt = tok.batch_decode(gen, skip_special_tokens=True)
        nt = (gen != tok.eos_token_id).sum(dim=1).tolist()
        for j in range(len(chunk)):
            sl = slice(j * M_POOL, (j + 1) * M_POOL)
            pools.append([parse(t) for t in txt[sl]])
            toks.append(float(np.mean(nt[sl])))
        if bi % 16 == 0:
            print(f"[{bi + len(chunk)}/{N_INST}] "
                  f"{(time.time() - t0) / 60:.0f} min", flush=True)

    curvas = np.array([curva_exacta(p, i["truth"])
                       for p, i in zip(pools, insts)])
    media = np.nanmean(curvas, axis=0)
    # IC bootstrap por instancia
    B = 5000
    idx = RNG.integers(0, N_INST, size=(B, N_INST))
    boots = np.nanmean(curvas[idx], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    print("\nacc(voto-de-n) INSESGADA (S₅ K9-11, T=1.3, m=32):")
    for n, a, l, h in zip(N_GRID, media, lo, hi):
        print(f"  n={n:>2}: {a:.4f}  IC95 [{l:.4f}, {h:.4f}]", flush=True)

    # techo con el n ASEQUIBLE (no el tope de diseño): con n̄ y p_hi, el
    # asignador puede pagar n_hi = (n̄ − (1−p)·n_lo)/p con n_lo=1
    out = {"T": T_DEC, "m": M_POOL, "n_grid": N_GRID,
           "acc": media.tolist(), "ic": [lo.tolist(), hi.tolist()],
           "t_med": float(np.mean(toks)), "techos": {}}
    f = lambda n: float(np.interp(n, [x for x, y in zip(N_GRID, media)
                                      if not np.isnan(y)],
                                  [y for y in media if not np.isnan(y)]))
    for nbar in (3.0, 5.0):
        for p in (0.10, 0.15):
            n_hi = min(31.0, (nbar - (1 - p)) / p)
            techo = f(n_hi) - f(nbar)
            out["techos"][f"nbar{nbar:.0f}_p{p:.2f}"] = {
                "n_asequible": n_hi, "acc_nbar": f(nbar),
                "acc_n_asequible": f(n_hi), "techo": techo}
            print(f"  techo n̄={nbar:.0f} p_hi={p:.2f}: n_asequible={n_hi:.1f} "
                  f"→ {techo:+.4f} {'≥0.04 ✓' if techo >= 0.04 else '<0.04'}",
                  flush=True)
    with open(os.path.join(RES, "llm_lever.json"), "w", encoding="utf-8") as f2:
        json.dump(out, f2, indent=1)
    print("Guardado llm_lever.json", flush=True)


if __name__ == "__main__":
    main()
