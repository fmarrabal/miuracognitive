"""
Capítulo LLM — FAMILIA 2: cadena aritmética multi-paso verificable
(PREREG_LLM v2 §2, familia declarada «solo si la fase 1 no da palanca»).

Motivo (FINDINGS_LLM §5): en S₅-en-texto los errores del actuador son
SISTEMÁTICOS → el voto converge al error → sin palanca no hay gobierno.
En aritmética encadenada los errores son DESLICES DIVERSOS (una
multiplicación mal da resultados variados), que es exactamente la
condición que la self-consistency necesita. Misma API que llm_env: el
resto del pipeline (caché, estimador exacto del voto, DP del techo
posterior, gates) se reutiliza sin cambios conceptuales.

Verificación EXACTA (entero) = el acantilado. Espacio de instancias
astronómico → el requisito 20× deja de morder a K bajo.
Stakes en METADATOS: el actuador NUNCA los ve.

  PYTHONPATH=. python -m mhbp.tasks.llm_gov.llm_env2   (self-tests, CPU)
"""
from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass

import numpy as np

SLOT_SYMS = ("J", "T", "V", "W")

SYSTEM = ("Eres una calculadora precisa. Te doy un valor inicial y una "
          "secuencia de operaciones. Aplícalas EN EL ORDEN dado, una a una, "
          "mostrando el resultado tras cada operación. Trabaja siempre con "
          "enteros. Termina SIEMPRE con una última línea con el formato "
          "exacto: RESPUESTA: <número entero>")

FEWSHOT = {"v0": 7, "ops": [("+", 5), ("*", 3), ("-", 8)]}


def apply_op(v, op):
    o, a = op
    if o == "+":
        return v + a
    if o == "-":
        return v - a
    return v * a


def truth(inst):
    v = inst["v0"]
    for op in inst["ops"]:
        v = apply_op(v, op)
    return str(v)


def literal_simulator(inst):
    """Segunda implementación, deliberadamente redundante (T-LLM0)."""
    v = int(inst["v0"])
    for (o, a) in inst["ops"]:
        if o == "+":
            v = v + int(a)
        elif o == "-":
            v = v - int(a)
        elif o == "*":
            v = v * int(a)
        else:
            raise ValueError(o)
    return str(v)


RE_RESP = re.compile(r"RESPUESTA:\s*(-?\d[\d\s.,]*)")


def parse(text):
    """Última línea RESPUESTA con un entero; tolera separadores de millar
    (1.234 / 1,234 / 1 234) y los normaliza. None si no hay entero."""
    m = RE_RESP.findall(text or "")
    if not m:
        return None
    s = m[-1].strip()
    s = re.sub(r"[.,\s](?=\d{3}\b)", "", s)        # separadores de millar
    s = s.replace(" ", "").rstrip(".,")
    if not re.fullmatch(r"-?\d+", s):
        return None
    return str(int(s))


def ops_txt(ops):
    simb = {"+": "+", "-": "-", "*": "×"}
    return ", ".join(f"{simb[o]}{a}" for o, a in ops)


def _fewshot_solution():
    v = FEWSHOT["v0"]
    lines = [f"Inicial: {v}."]
    for (o, a) in FEWSHOT["ops"]:
        v = apply_op(v, (o, a))
        lines.append(f"{ {'+': '+', '-': '-', '*': '×'}[o] }{a}: {v}")
    lines.append(f"RESPUESTA: {v}")
    return "\n".join(lines)


def build_messages(inst):
    """Prompt del ACTUADOR — sin rastro del stake."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Inicial: {FEWSHOT['v0']}. "
                                    f"Operaciones: {ops_txt(FEWSHOT['ops'])}"},
        {"role": "assistant", "content": _fewshot_solution()},
        {"role": "user", "content": f"Inicial: {inst['v0']}. "
                                    f"Operaciones: {ops_txt(inst['ops'])}"},
    ]


@dataclass(frozen=True)
class LlmSpec:
    """RONDA 2 de diales de familia 2 (2026-08-08): con multiplicadores 2-4
    y sumas <40, acc(1) = 0.96-0.97 en todo K∈[4,9] (techo: sin palanca por
    saturación). La dificultad de la aritmética NO está en la longitud de
    la cadena (error por paso ~0.3%) sino en el TAMAÑO de los operandos —
    multiplicar 4 dígitos × 2 dígitos es donde el actuador desliza, y con
    errores DIVERSOS. Diales de magnitud (el análogo del K-rango de la
    familia 1): mul_hi, add_hi, cap_mul."""
    k_lo: int = 5
    k_hi: int = 8
    p_hi: float = 0.15
    s_alto: float = 8.0
    # RONDA 3 (última declarada): ronda 2 (mul 7-19) dejó acc en 0.85-0.96 —
    # errores YA diversos (modal-erróneo 0.10-0.14, contra 0.86 en S₅) pero
    # sin margen. Multiplicadores de DOS dígitos sobre operandos de 5-6
    # cifras es donde un 14B sin herramientas desliza de verdad.
    mul_lo: int = 23
    mul_hi: int = 97          # multiplicadores de 2 dígitos
    add_lo: int = 137
    add_hi: int = 8999        # sumandos de 3-4 dígitos
    cap_mul: int = 500_000    # × solo si |v| < cap (resultados ≤ ~5·10⁷)


def inst_hash(inst):
    return hashlib.sha1(
        f"{inst['v0']}|{inst['ops']}".encode()).hexdigest()


class LlmTaskStream:
    """Mismo contrato que llm_env.LlmTaskStream (streams, dedupe, stakes en
    metadatos). Control de magnitud: × solo si |v| < 2000, para que la
    dificultad venga de la CADENA y no de aritmética gigante."""

    SEEDS = {"sonda": 7001, "val": 7002, "dev": 7003, "eval": 7999}

    def __init__(self, spec: LlmSpec, split: str, banned_hashes=None):
        self.spec = spec
        ss = np.random.SeedSequence(self.SEEDS[split] + 500)   # familia 2
        c_ops, c_stake = ss.spawn(2)
        self.rng_ops = np.random.default_rng(c_ops)
        self.rng_stake = np.random.default_rng(c_stake)
        self.banned = set(banned_hashes or ())
        self.emitted = set()
        self.n_repetidas = 0

    def sample(self, k=None, max_retries=400):
        for intento in range(max_retries + 1):
            K = int(k if k is not None
                    else self.rng_ops.integers(self.spec.k_lo,
                                               self.spec.k_hi + 1))
            sp = self.spec
            v = int(self.rng_ops.integers(11, 99))
            v0, ops = v, []
            for _ in range(K):
                permite_mul = abs(v) < sp.cap_mul
                r = self.rng_ops.random()
                if permite_mul and r < 0.45:
                    a = int(self.rng_ops.integers(sp.mul_lo, sp.mul_hi + 1))
                    o = "*"
                elif r < 0.75:
                    a = int(self.rng_ops.integers(sp.add_lo, sp.add_hi + 1))
                    o = "+"
                else:
                    a = int(self.rng_ops.integers(sp.add_lo, sp.add_hi + 1))
                    o = "-"
                ops.append((o, a))
                v = apply_op(v, (o, a))
            inst = {"v0": v0, "ops": ops, "K": K}
            h = inst_hash(inst)
            if h in self.banned:
                continue
            if h in self.emitted and intento < max_retries:
                continue
            if h in self.emitted:
                self.n_repetidas += 1
            self.emitted.add(h)
            alto = bool(self.rng_stake.random() < self.spec.p_hi)
            if alto:
                s0 = s1 = SLOT_SYMS[int(self.rng_stake.integers(0, 4))]
            else:
                while True:
                    a_ = SLOT_SYMS[int(self.rng_stake.integers(0, 4))]
                    b_ = SLOT_SYMS[int(self.rng_stake.integers(0, 4))]
                    if a_ != b_:
                        s0, s1 = a_, b_
                        break
            inst.update({"truth": truth(inst),
                         "stake": self.spec.s_alto if alto else 1.0,
                         "meta_slots": (s0, s1), "hash": h})
            return inst
        raise RuntimeError("sample: sin instancias nuevas")


# ------------------------------- T-LLM0 ------------------------------- #
def _selftests():
    spec = LlmSpec()
    st = LlmTaskStream(spec, "dev")
    insts = [st.sample() for _ in range(1200)]
    # [1] generador ≡ simulador literal
    for i in insts:
        assert truth(i) == literal_simulator(i), f"convención: {i['ops']}"
    # [2] few-shot coherente y no trivial
    fs = _fewshot_solution()
    assert fs.strip().endswith("RESPUESTA: " + literal_simulator(FEWSHOT))
    assert literal_simulator(FEWSHOT) != str(FEWSHOT["v0"])
    # [3] parser: última ocurrencia, negativos, millares, rechazo de basura
    assert parse("RESPUESTA: 12\nRESPUESTA: -345") == "-345"
    assert parse("RESPUESTA: 1.234") == "1234"
    assert parse("RESPUESTA: 1,234") == "1234"
    assert parse("RESPUESTA: 12.5") is None or parse("RESPUESTA: 12.5") == "125"
    assert parse("RESPUESTA: doce") is None
    assert parse("sin nada") is None
    # [4] magnitudes acotadas (dificultad = cadena, no números gigantes)
    mags = [abs(int(i["truth"])) for i in insts]
    assert np.percentile(mags, 99) < 5e9, f"magnitud p99={np.percentile(mags,99)}"
    # [5] stake ⊥ K, marginal, regla de igualdad
    ks = np.array([i["K"] for i in insts], dtype=float)
    ss_ = np.array([i["stake"] for i in insts])
    r = float(np.corrcoef(ks, ss_)[0, 1])
    assert abs(r) < 0.09, f"stake~K r={r:.3f}"
    assert abs(float((ss_ > 1).mean()) - spec.p_hi) < 0.05
    eq = np.array([i["meta_slots"][0] == i["meta_slots"][1] for i in insts])
    assert np.array_equal(eq, ss_ > 1)
    # [6] el prompt del actuador no contiene los símbolos de stake
    blob = " ".join(m["content"] for m in build_messages(insts[0]))
    for sym in SLOT_SYMS:
        assert f" {sym} " not in blob and f"{sym}:" not in blob
    # [7] espacio: dedupe sin repeticiones en 1200 (espacio astronómico)
    assert st.n_repetidas == 0
    # [8] determinismo por stream
    a = [x["hash"] for x in [LlmTaskStream(spec, "dev").sample()
                             for _ in range(1)]]
    b = [x["hash"] for x in [LlmTaskStream(spec, "dev").sample()
                             for _ in range(1)]]
    assert a == b
    print(f"llm_env2 T-LLM0 + SELF-TESTS 8/8 OK — few-shot: 7 "
          f"{ops_txt(FEWSHOT['ops'])} → {literal_simulator(FEWSHOT)} | "
          f"|resp| mediana={np.median(mags):.0f} p99={np.percentile(mags,99):.0f} "
          f"| r(stake,K)={r:+.3f}")


if __name__ == "__main__":
    _selftests()
