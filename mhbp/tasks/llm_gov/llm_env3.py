"""
N1b-LLM — Entorno CAMINO-EN-CICLO en texto (PREREG_N1B_LLM §2).

Ciclo único de longitud L sobre letras, presentado como pares barajados;
el actuador camina en CoT («k: x→y») y cierra «RESPUESTA: d». L visible
(nº de pares), d ~ U[1, L−1] invisible, stakes en metadatos (el prompt
no los contiene). Misma API/patrones que llm_env/llm_env2.

  PYTHONPATH=. python -m mhbp.tasks.llm_gov.llm_env3   (self-tests, CPU)
"""
from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass

import numpy as np

SLOT_SYMS = ("J", "T", "V", "W")
LETRAS = "abcdefghijklmnopqrst"          # 20 letras candidatas

SYSTEM = ("Te doy una función de UN paso, escrita como pares x→y (aplicar "
          "la función a x da y). Empezando en el símbolo inicial, aplica "
          "la función paso a paso hasta llegar al símbolo objetivo. "
          "Escribe cada paso en su propia línea con el formato exacto "
          "'k: x→y' (k = número de paso, empezando en 1). Cuando el "
          "resultado de un paso sea el objetivo, para y termina SIEMPRE "
          "con una última línea con el formato exacto: "
          "RESPUESTA: <número de pasos>")

FEWSHOT_PARES = [("c", "f"), ("f", "a"), ("a", "q"), ("q", "c")]
FEWSHOT_S, FEWSHOT_T, FEWSHOT_D = "a", "f", 3


def fewshot_user():
    pares = ", ".join(f"{x}→{y}" for x, y in FEWSHOT_PARES)
    return (f"Función: {pares}. Inicial: {FEWSHOT_S}. "
            f"Objetivo: {FEWSHOT_T}.")


def fewshot_asst():
    return "1: a→q\n2: q→c\n3: c→f\nRESPUESTA: 3"


def truth(inst):
    return str(inst["d"])


def walk_check(inst):
    """Caminante independiente sobre los pares (T-verificación)."""
    f = dict(inst["pares"])
    x, d = inst["s"], 0
    while d <= len(inst["pares"]) + 1:
        if x == inst["t"]:
            return str(d)
        x = f[x]
        d += 1
    return "-1"


RE_RESP = re.compile(r"RESPUESTA:\s*(\d+)")
RE_HOP = re.compile(r"^\s*(\d+)\s*:\s*([a-t])\s*(?:→|->)\s*([a-t])\s*$",
                    re.MULTILINE)


def parse(text):
    m = RE_RESP.findall(text or "")
    return m[-1] if m else None


def parse_hops(text):
    """Lista [(k, x, y)] de pasos escritos — para el probe de mudez y
    los diagnósticos de linealidad hops↔tokens."""
    return [(int(k), x, y) for k, x, y in RE_HOP.findall(text or "")]


def build_messages(inst):
    """Prompt del actuador — SIN rastro del stake."""
    pares = ", ".join(f"{x}→{y}" for x, y in inst["pares"])
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": fewshot_user()},
        {"role": "assistant", "content": fewshot_asst()},
        {"role": "user", "content": f"Función: {pares}. "
                                    f"Inicial: {inst['s']}. "
                                    f"Objetivo: {inst['t']}."},
    ]


@dataclass(frozen=True)
class LlmSpec:
    l_set: tuple = (6, 10, 14)
    p_hi: float = 0.15
    s_alto: float = 8.0


def inst_hash(inst):
    """Hash de CONTENIDO canónico (panel v2): pares ORDENADOS = la función
    en representación canónica, invariante al barajado de presentación."""
    canon = tuple(sorted(inst["pares"]))
    return hashlib.sha1(f"{canon}|{inst['s']}|{inst['t']}".encode()
                        ).hexdigest()


class LlmTaskStream:
    SEEDS = {"sonda": 7001, "val": 7002, "dev": 7003, "eval": 7999}

    def __init__(self, spec: LlmSpec, split: str, banned_hashes=None):
        self.spec = spec
        ss = np.random.SeedSequence(self.SEEDS[split] + 900)   # familia 3
        c_task, c_stake = ss.spawn(2)
        self.rng = np.random.default_rng(c_task)
        self.rng_stake = np.random.default_rng(c_stake)
        self.banned = set(banned_hashes or ())
        self.emitted = set()
        self.n_repetidas = 0

    def sample(self, L=None, max_retries=400):
        sp = self.spec
        for intento in range(max_retries + 1):
            L_ = int(L if L is not None
                     else sp.l_set[self.rng.integers(0, len(sp.l_set))])
            simbolos = [LETRAS[i] for i in
                        self.rng.permutation(len(LETRAS))[:L_]]
            # ciclo: orden aleatorio de los símbolos
            pares = [(simbolos[j], simbolos[(j + 1) % L_])
                     for j in range(L_)]
            d = int(self.rng.integers(1, L_))
            i0 = int(self.rng.integers(0, L_))
            s = simbolos[i0]
            t = simbolos[(i0 + d) % L_]
            orden = self.rng.permutation(L_)
            inst = {"pares": tuple(pares[j] for j in orden),
                    "s": s, "t": t, "L": L_, "d": d}
            h = inst_hash(inst)
            if h in self.banned:
                continue
            if h in self.emitted and intento < max_retries:
                continue
            if h in self.emitted:
                self.n_repetidas += 1
            self.emitted.add(h)
            alto = bool(self.rng_stake.random() < sp.p_hi)
            if alto:
                a = SLOT_SYMS[int(self.rng_stake.integers(0, 4))]
                s0 = s1 = a
            else:
                while True:
                    a = SLOT_SYMS[int(self.rng_stake.integers(0, 4))]
                    b = SLOT_SYMS[int(self.rng_stake.integers(0, 4))]
                    if a != b:
                        s0, s1 = a, b
                        break
            inst.update({"truth": str(d),
                         "stake": sp.s_alto if alto else 1.0,
                         "meta_slots": (s0, s1), "hash": h})
            return inst
        raise RuntimeError("sample: sin instancias nuevas")


# ------------------------------- self-tests ------------------------------- #
def _selftests():
    spec = LlmSpec()
    st = LlmTaskStream(spec, "dev")
    insts = [st.sample() for _ in range(1500)]
    # [1] truth ≡ caminante independiente; ciclo único de longitud L
    for i in insts[:500]:
        assert truth(i) == walk_check(i), f"convención: {i}"
        f = dict(i["pares"])
        x, ln = f[i["s"]], 1
        while x != i["s"]:
            x = f[x]
            ln += 1
        assert ln == i["L"], "el ciclo no es único de longitud L"
    # [2] few-shot coherente
    fs_inst = {"pares": FEWSHOT_PARES, "s": FEWSHOT_S, "t": FEWSHOT_T}
    assert walk_check(fs_inst) == str(FEWSHOT_D)
    assert parse(fewshot_asst()) == str(FEWSHOT_D)
    assert len(parse_hops(fewshot_asst())) == FEWSHOT_D
    # [3] parser: última RESPUESTA, hops con → y ->
    assert parse("RESPUESTA: 3\nRESPUESTA: 11") == "11"
    assert parse("nada") is None
    assert parse_hops("1: a->b\n2: b→c") == [(1, "a", "b"), (2, "b", "c")]
    # [4] d ~ U[1, L−1] por clase; d y stake ⊥
    D = np.array([i["d"] for i in insts])
    Ls = np.array([i["L"] for i in insts])
    Ss = np.array([i["stake"] for i in insts])
    for L in spec.l_set:
        dd = D[Ls == L]
        assert dd.min() >= 1 and dd.max() <= L - 1
        cnt = np.bincount(dd, minlength=L)[1:L]
        assert cnt.min() > 0.5 * cnt.mean(), f"d no uniforme L={L}"
    assert abs(np.corrcoef(D, Ss)[0, 1]) < 0.06
    assert abs(np.corrcoef(Ls, Ss)[0, 1]) < 0.06
    assert abs(float((Ss > 1).mean()) - spec.p_hi) < 0.03
    # [5] el prompt no contiene stakes; sí contiene L pares (L visible)
    for i in insts[:50]:
        blob = " ".join(m["content"] for m in build_messages(i))
        for sym in SLOT_SYMS:
            assert f" {sym} " not in blob and f"{sym}:" not in blob
        assert blob.count("→") >= i["L"] + len(FEWSHOT_PARES)
    # [6] espacio y dedupe
    assert st.n_repetidas == 0
    # [7] determinismo por stream
    a = LlmTaskStream(spec, "dev").sample()["hash"]
    b = LlmTaskStream(spec, "dev").sample()["hash"]
    assert a == b
    print(f"llm_env3 SELF-TESTS 7/7 OK — p_hi={float((Ss > 1).mean()):.3f}, "
          f"E[d] por L=" + str({int(L): round(float(D[Ls == L].mean()), 2)
                                for L in spec.l_set}))


if __name__ == "__main__":
    _selftests()
