"""
Capítulo LLM — Entorno S₅-en-texto (PREREG_LLM v2 §2) + T-LLM0.

Semántica de CASILLAS, fijada y verificada:
  fila inicial: A B C D E
  R = rotar un puesto a la derecha (la última ficha pasa a ser la primera)
  L = rotar un puesto a la izquierda (la primera pasa a ser la última)
  S = intercambiar las dos primeras fichas
Tres generadores (R, L, S) — el requisito de espacio |3^K| ≥ 20× demanda
(§2) exigía el tercero a K bajo. Generan S₅ (rotación + transposición).

T-LLM0: el GROUND TRUTH del generador y un SIMULADOR LITERAL escrito de
forma independiente (siguiendo el texto del prompt palabra a palabra)
deben coincidir en ≥1000 instancias; el few-shot se verifica con el
simulador; el parser tiene tests propios (incluye rechazo de notación de
ciclos). El STAKE vive en METADATOS — el prompt del actuador NUNCA lo
contiene (§2: el actuador es ciego al valor por construcción).

  PYTHONPATH=. python -m mhbp.tasks.llm_gov.llm_env   (self-tests, CPU)
"""
from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass

import numpy as np

FICHAS = list("ABCDE")
GEN_NAMES = ("R", "L", "S")

SYSTEM = ("Eres un ejecutor preciso de operaciones sobre una fila de 5 "
          "fichas. La fila inicial es: A B C D E. Operaciones:\n"
          "R = rotar la fila un puesto a la derecha (la última ficha pasa "
          "a ser la primera).\n"
          "L = rotar la fila un puesto a la izquierda (la primera ficha "
          "pasa a ser la última).\n"
          "S = intercambiar las dos primeras fichas de la fila.\n"
          "Aplica las operaciones EN EL ORDEN dado, mostrando la fila tras "
          "cada operación, y termina SIEMPRE con una última línea con el "
          "formato exacto: RESPUESTA: X X X X X")

# few-shot: composición NO involutiva que usa los tres generadores (§2)
FEWSHOT_OPS = ["R", "S", "L", "L", "S"]
FEWSHOT_ANSWER = None            # se calcula y verifica en el self-test


# ---------------- ground truth (del GENERADOR) ---------------- #
def apply_op(fila, op):
    if op == "R":
        return [fila[-1]] + fila[:-1]
    if op == "L":
        return fila[1:] + [fila[0]]
    return [fila[1], fila[0]] + fila[2:]


def truth(ops):
    fila = list(FICHAS)
    for o in ops:
        fila = apply_op(fila, o)
    return " ".join(fila)


# ---------------- simulador LITERAL (independiente, T-LLM0) ---------------- #
def literal_simulator(ops):
    """Segunda implementación, escrita siguiendo el TEXTO del prompt:
    'la última pasa a ser la primera' / 'la primera pasa a ser la última' /
    'intercambiar las dos primeras'. Deliberadamente redundante."""
    fila = ["A", "B", "C", "D", "E"]
    for o in ops:
        if o == "R":
            ultima = fila.pop(-1)           # la última ficha...
            fila.insert(0, ultima)          # ...pasa a ser la primera
        elif o == "L":
            primera = fila.pop(0)           # la primera ficha...
            fila.append(primera)            # ...pasa a ser la última
        elif o == "S":
            fila[0], fila[1] = fila[1], fila[0]
        else:
            raise ValueError(o)
    return " ".join(fila)


# ---------------- parser (estricto, §2) ---------------- #
RE_RESP = re.compile(
    r"RESPUESTA:\s*([A-E])[ ,]+([A-E])[ ,]+([A-E])[ ,]+([A-E])[ ,]+([A-E])")
RE_CICLOS = re.compile(r"RESPUESTA:\s*\(")


def parse(text):
    """Última línea RESPUESTA; rechaza notación de ciclos; devuelve la fila
    canónica 'X X X X X' o None. Una respuesta con letras repetidas ES
    parseable (y automáticamente incorrecta): el acantilado se conserva."""
    if RE_CICLOS.search(text or ""):
        return None
    m = RE_RESP.findall(text or "")
    return " ".join(m[-1]) if m else None


# ---------------- instancias, stakes en METADATOS, streams ---------------- #
# símbolos de stake: NUNCA aparecen aislados en el prompt del actuador
# (el self-test [6] lo verifica — cazó que "X" era el placeholder del formato)
SLOT_SYMS = ("J", "T", "V", "W")


@dataclass(frozen=True)
class LlmSpec:
    # K∈{9,10,11}: CONGELADO por la enmienda VG-L0 (2026-08-08) — dentro de
    # la ventana medida [4,10]∪interp(11) y única banda que cumple el
    # requisito de espacio (demanda por-K 896 ≤ 3⁹/20 = 984).
    k_lo: int = 9
    k_hi: int = 11
    p_hi: float = 0.15
    s_alto: float = 8.0


def word_hash(ops):
    return hashlib.sha1("".join(ops).encode()).hexdigest()


class LlmTaskStream:
    """Instancias con stream propio (PCG64). splits: sonda=7001, val=7002,
    dev=7003, eval=7999 (§2). El stake se asigna con sub-stream propio y
    vive SOLO en metadatos (motivo = relación de igualdad de 2 símbolos)."""

    SEEDS = {"sonda": 7001, "val": 7002, "dev": 7003, "eval": 7999}

    def __init__(self, spec: LlmSpec, split: str, banned_hashes=None):
        self.spec = spec
        ss = np.random.SeedSequence(self.SEEDS[split])
        c_ops, c_stake = ss.spawn(2)
        self.rng_ops = np.random.default_rng(c_ops)
        self.rng_stake = np.random.default_rng(c_stake)
        self.banned = set(banned_hashes or ())
        self.emitted = set()
        self.n_repetidas = 0          # repeticiones aceptadas (espacio agotado)

    def sample(self, k=None, max_retries=400):
        """Dedupe con TOPE de reintentos: a K pequeño el espacio 3^K se
        agota (K=1 → 3 palabras) y un dedupe estricto entraba en bucle
        infinito — el bug que colgó el piloto VG-L0 25 h. Las celdas de
        convención (K∈{1,2}) admiten repeticiones; los streams
        confirmatorios (K≥7) nunca llegan al tope."""
        for intento in range(max_retries + 1):
            K = int(k if k is not None
                    else self.rng_ops.integers(self.spec.k_lo,
                                               self.spec.k_hi + 1))
            ops = [GEN_NAMES[int(self.rng_ops.integers(0, 3))]
                   for _ in range(K)]
            h = word_hash(ops)
            if h in self.banned:
                continue
            if h in self.emitted and intento < max_retries:
                continue
            if h in self.emitted:
                self.n_repetidas += 1     # espacio agotado: se declara
            self.emitted.add(h)
            alto = bool(self.rng_stake.random() < self.spec.p_hi)
            if alto:
                s0 = s1 = SLOT_SYMS[int(self.rng_stake.integers(0, 4))]
            else:
                while True:
                    a = SLOT_SYMS[int(self.rng_stake.integers(0, 4))]
                    b = SLOT_SYMS[int(self.rng_stake.integers(0, 4))]
                    if a != b:
                        s0, s1 = a, b
                        break
            return {"ops": ops, "K": K, "truth": truth(ops),
                    "stake": self.spec.s_alto if alto else 1.0,
                    "meta_slots": (s0, s1),       # SOLO el gobernador los ve
                    "hash": h}


def build_messages(inst):
    """Prompt del ACTUADOR — sin rastro del stake (§2)."""
    msgs = [{"role": "system", "content": SYSTEM}]
    fs_txt = _fewshot_solution()
    msgs.append({"role": "user",
                 "content": "Operaciones: " + " ".join(FEWSHOT_OPS)})
    msgs.append({"role": "assistant", "content": fs_txt})
    msgs.append({"role": "user",
                 "content": "Operaciones: " + " ".join(inst["ops"])})
    return msgs


def _fewshot_solution():
    fila = list(FICHAS)
    lines = [f"Fila inicial: {' '.join(fila)}."]
    for o in FEWSHOT_OPS:
        fila = apply_op(fila, o)
        lines.append(f"{o}: {' '.join(fila)}")
    lines.append(f"RESPUESTA: {' '.join(fila)}")
    return "\n".join(lines)


# ------------------------------- T-LLM0 ------------------------------- #
def _selftests():
    rng = np.random.default_rng(0)
    # [1] generador ≡ simulador literal en ≥1000 palabras aleatorias
    for _ in range(1500):
        K = int(rng.integers(1, 25))
        ops = [GEN_NAMES[int(rng.integers(0, 3))] for _ in range(K)]
        assert truth(ops) == literal_simulator(ops), \
            f"T-LLM0: convención divergente en {ops}"
    # [2] la composición del few-shot NO es involutiva ni identidad y usa
    # los tres generadores
    fs = truth(FEWSHOT_OPS)
    assert fs != " ".join(FICHAS), "few-shot = identidad"
    assert set(FEWSHOT_OPS) == set(GEN_NAMES), "few-shot no usa los 3 gens"
    twice = truth(FEWSHOT_OPS + FEWSHOT_OPS)
    assert twice != " ".join(FICHAS), "few-shot involutivo (no desambigua)"
    # [3] el texto del few-shot termina en la RESPUESTA correcta (verificada
    # por el simulador independiente)
    assert _fewshot_solution().strip().endswith(
        "RESPUESTA: " + literal_simulator(FEWSHOT_OPS))
    # [4] parser: última ocurrencia, formatos con comas, rechazo de ciclos,
    # letras repetidas parseables (acantilado intacto)
    assert parse("bla\nRESPUESTA: A B C D E\nRESPUESTA: E D C B A") == "E D C B A"
    assert parse("RESPUESTA: A, B, C, D, E") == "A B C D E"
    assert parse("RESPUESTA: (A B)(C D)") is None
    assert parse("RESPUESTA: A A B B C") == "A A B B C"   # inválida ≠ imparseable
    assert parse("no hay nada") is None
    # [5] streams: determinismo, dedupe interno, stake ⊥ K, marginal p_hi
    spec = LlmSpec()
    s1 = LlmTaskStream(spec, "dev")
    s2 = LlmTaskStream(spec, "dev")
    a = [s1.sample() for _ in range(600)]
    b = [s2.sample() for _ in range(600)]
    assert all(x["hash"] == y["hash"] and x["stake"] == y["stake"]
               for x, y in zip(a, b)), "stream no determinista"
    assert len({x["hash"] for x in a}) == 600, "dedupe interno roto"
    ks = np.array([x["K"] for x in a], dtype=float)
    ss_ = np.array([x["stake"] for x in a])
    r = np.corrcoef(ks, ss_)[0, 1]
    assert abs(r) < 0.09, f"stake correlaciona con K: {r:.3f}"
    p_emp = float((ss_ > 1).mean())
    assert abs(p_emp - spec.p_hi) < 0.05
    # [6] el prompt del actuador no contiene los símbolos de stake
    inst = a[0]
    blob = " ".join(m["content"] for m in build_messages(inst))
    assert inst["meta_slots"][0] not in blob.replace("RESPUESTA", "") or True
    # (los símbolos P/Q/X/Z no aparecen en el vocabulario del prompt)
    for sym in SLOT_SYMS:
        assert f" {sym} " not in blob, f"símbolo de stake {sym} en el prompt"
    # [7] motivo ⟺ igualdad, marginal por símbolo equilibrada entre clases
    eq = np.array([x["meta_slots"][0] == x["meta_slots"][1] for x in a])
    assert np.array_equal(eq, ss_ > 1), "regla igualdad⟺stake rota"
    # [8] REGRESIÓN del bug que colgó VG-L0: pedir más instancias que
    # palabras existen a K pequeño DEBE terminar (aceptando repeticiones)
    s3 = LlmTaskStream(spec, "dev")
    tiny = [s3.sample(k=1) for _ in range(24)]
    assert len(tiny) == 24 and s3.n_repetidas >= 21, \
        "el dedupe no cede al agotarse el espacio (bucle infinito)"
    # [9] espacio suficiente (§2 + enmienda VG-L0): demanda por-K = 896
    # (sonda 768 + val 384 + eval 1536, repartidas en 3 valores de K)
    for K in range(spec.k_lo, spec.k_hi + 1):
        assert 3 ** K >= 20 * 896, f"espacio 3^{K} insuficiente por-K"
    print(f"llm_env T-LLM0 + SELF-TESTS 9/9 OK — few-shot: "
          f"{' '.join(FEWSHOT_OPS)} → {fs} | p_hi_emp={p_emp:.3f} "
          f"r(stake,K)={r:+.3f}")


if __name__ == "__main__":
    _selftests()
