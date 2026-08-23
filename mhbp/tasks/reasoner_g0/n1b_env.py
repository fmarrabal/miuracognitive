"""
N1b — Entorno CAMINO-EN-CICLO (acantilado mudo). PREREG_N1B §2.

La familia que la predicción del paper exige (FINDINGS_N1B_PILOTO §4):
  · VALOR TODO-O-NADA en cómputo: la respuesta (distancia d) solo se
    conoce al LLEGAR — no hay aproximación parcial útil.
  · PROGRESO INTERNO MUDO: caminando un ciclo de una permutación
    aleatoria, la posición actual no contiene información sobre la
    distancia restante (más allá del par visible (pasos dados, L)). La
    mudez no se decreta: se MIDE (gate M-N1b: probe de convergencia).
  · Dificultad ex-ante COARSE: la clase L (longitud del ciclo) es
    visible; d ~ U[1, L−1] es invisible. El ex-ante conoce la
    DISTRIBUCIÓN; el posterior, nada más hasta la llegada.

Instancia: permutación σ sobre N=20 elementos presentada como tabla
(posición i → σ(i)); s y t garantizados en un ciclo de longitud
EXACTA L ∈ {6, 12, 18} (token de clase visible); respuesta = distancia
d(s→t) por σ. Stakes en METADATOS con el motivo relacional de N2
(slots iguales ⟺ alto), stream propio ⊥ del de la tarea.

Secuencia (longitud constante 48, sin fuga por longitud). RONDA 1 de
diales (gate A rojo con tabla posicional: el direccionamiento indirecto
no se aprende — loss clavada en la entropía del prior): σ se presenta
como PARES (i, σ(i)) en ORDEN ALEATORIO → recall asociativo encadenado
(induction heads), el primitivo aprendible; la mudez no cambia.
  [slot0, slot1, L_class, s, t, (i₀,σ(i₀)), ..., (i₁₉,σ(i₁₉)), ARROW, PAD, PAD]
Label: -1 en todas las posiciones salvo ARROW → token de d.

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.n1b_env   (self-tests)
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch

# ------------------------------ vocabulario ------------------------------ #
PAD, ARROW = 0, 1
N_ELEM = 20
ELEM_OFF = 2                        # 2..21  : elementos / valores de σ
DIST_OFF = ELEM_OFF + N_ELEM        # 22..39 : respuestas d = 1..18
D_MAX = 18
LCLASS_OFF = DIST_OFF + D_MAX       # 40..42 : clase L ∈ {6,12,18}
L_SET = (6, 12, 18)
SLOT_OFF = LCLASS_OFF + len(L_SET)  # 43..46 : alfabeto de slots (stake)
VOCAB = SLOT_OFF + 4                # 47
SEQ_LEN = 48
ARROW_IDX = 2 + 1 + 2 + 2 * N_ELEM  # posición fija del ARROW = 45


@dataclass(frozen=True)
class N1bSpec:
    p_hi: float = 0.15
    s_alto: float = 8.0
    seq_len: int = SEQ_LEN


class N1bDataset:
    """Misma API que N2Dataset: batch(B) → (inputs, targets, Ls, stakes).
    'K' del contrato pasa a ser L (la dificultad ex-ante visible). La d
    oculta del último batch queda en .last_d (diagnóstico/sonda)."""

    def __init__(self, spec: N1bSpec, seed: int, split: str = "train"):
        self.spec = spec
        off = {"train": 0, "eval": 800_000, "val": 400_000}[split]
        self.rng_task = np.random.default_rng(seed + off)
        self.rng_stake = np.random.default_rng(seed * 1000 + 13 + off)
        self.vocab_size = VOCAB
        self.answer_offset = DIST_OFF
        self.answer_size = D_MAX
        self.last_d = None

    def _instancia(self):
        rt = self.rng_task
        L = L_SET[int(rt.integers(0, len(L_SET)))]
        d = int(rt.integers(1, L))                    # d ~ U[1, L-1]
        elems = rt.permutation(N_ELEM)
        ciclo = elems[:L]                             # ciclo de longitud L
        resto = elems[L:]
        sigma = np.empty(N_ELEM, dtype=np.int64)
        for j in range(L):
            sigma[ciclo[j]] = ciclo[(j + 1) % L]
        # resto: permutación aleatoria cerrada sobre sí misma
        if len(resto):
            perm_r = rt.permutation(len(resto))
            for a, b in zip(resto, resto[perm_r]):
                sigma[a] = b
        i0 = int(rt.integers(0, L))
        s = int(ciclo[i0])
        t = int(ciclo[(i0 + d) % L])
        return sigma, s, t, L, d

    def _slots(self, alto: bool):
        r = self.rng_stake
        if alto:
            a = SLOT_OFF + int(r.integers(0, 4))
            return a, a
        while True:
            a = SLOT_OFF + int(r.integers(0, 4))
            b = SLOT_OFF + int(r.integers(0, 4))
            if a != b:
                return a, b

    def batch(self, batch_size: int):
        xs, ys, Ls, stakes, ds = [], [], [], [], []
        for _ in range(batch_size):
            sigma, s, t, L, d = self._instancia()
            alto = bool(self.rng_stake.random() < self.spec.p_hi)
            s0, s1 = self._slots(alto)
            tok = np.full(SEQ_LEN, PAD, dtype=np.int64)
            tok[0], tok[1] = s0, s1
            tok[2] = LCLASS_OFF + L_SET.index(L)
            tok[3], tok[4] = ELEM_OFF + s, ELEM_OFF + t
            orden = self.rng_task.permutation(N_ELEM)   # pares barajados
            for j, i_el in enumerate(orden):
                tok[5 + 2 * j] = ELEM_OFF + int(i_el)
                tok[6 + 2 * j] = ELEM_OFF + int(sigma[i_el])
            tok[ARROW_IDX] = ARROW
            lab = np.full(SEQ_LEN, -1, dtype=np.int64)
            lab[ARROW_IDX] = DIST_OFF + d
            xs.append(tok)
            ys.append(lab)
            Ls.append(L)
            stakes.append(self.spec.s_alto if alto else 1.0)
            ds.append(d)
        tokens = torch.tensor(np.stack(xs))
        labels = torch.tensor(np.stack(ys))
        self.last_d = torch.tensor(ds)
        return (tokens[:, :-1].contiguous(), labels[:, :-1].contiguous(),
                torch.tensor(Ls), torch.tensor(stakes, dtype=torch.float32))


SEQ_LEN_D = 56          # secuencia extendida con 8 queries auxiliares
N_Q = 8


class N1bDatasetDenso(N1bDataset):
    """RONDA 3 (2026-08-10, con base empírica): el retrieval en contexto
    GROKKEA con supervisión densa + batch 128 (probe: silla en ln20 hasta
    ~9k pasos, acc 1.000 en 15k) y NO se forma con 1 posición supervisada
    (5 probes en azar). Curriculum denso: 8 queries auxiliares tras el
    ARROW, cada una etiquetada σ(q_j) (enseñan el circuito); la pregunta
    por d en el ARROW monta encima. Las queries van DESPUÉS del ARROW
    (causal ⇒ cero fuga hacia el readout de d). Los gates puntúan SOLO
    la posición ARROW."""

    def batch(self, batch_size):
        X0, Y0, Ls, Ss = super().batch(batch_size)
        B = X0.shape[0]
        X = torch.zeros(B, SEQ_LEN_D - 1, dtype=torch.long)
        Y = torch.full((B, SEQ_LEN_D - 1), -1, dtype=torch.long)
        X[:, :X0.shape[1]] = X0
        Y[:, :Y0.shape[1]] = Y0
        for b in range(B):
            sigma = sigma_de_pares(X[b])
            qs = self.rng_task.permutation(N_ELEM)[:N_Q]
            for j, q in enumerate(qs):
                pos = ARROW_IDX + 1 + j
                X[b, pos] = ELEM_OFF + int(q)
                Y[b, pos] = ELEM_OFF + int(sigma[q])
        return X, Y, Ls, Ss


def walk_distance(sigma, s, t, cap=N_ELEM + 1):
    """Caminante independiente (verificación T-N1b0)."""
    x, d = s, 0
    while d < cap:
        if x == t:
            return d
        x = int(sigma[x])
        d += 1
    return -1


def sigma_de_pares(row):
    """Reconstruye σ desde los pares (i, σ(i)) de la secuencia."""
    sigma = np.full(N_ELEM, -1, dtype=np.int64)
    for j in range(N_ELEM):
        i_el = int(row[5 + 2 * j]) - ELEM_OFF
        sigma[i_el] = int(row[6 + 2 * j]) - ELEM_OFF
    return sigma


# ------------------------------- self-tests ------------------------------- #
def _selftests():
    spec = N1bSpec()
    ds = N1bDataset(spec, seed=0)
    X, Y, Ls, Ss = ds.batch(6000)
    D = ds.last_d
    # [1] label ≡ caminante independiente, y ciclo de longitud EXACTA L
    for b in range(400):
        row = X[b]
        sigma = sigma_de_pares(row)
        s = int(row[3] - ELEM_OFF)
        t = int(row[4] - ELEM_OFF)
        d_lab = int(Y[b, ARROW_IDX]) - DIST_OFF
        assert walk_distance(sigma, s, t) == d_lab, "label ≠ caminante"
        assert walk_distance(sigma, s, s + 0) == 0
        # longitud del ciclo de s == L declarado
        x, ln = int(sigma[s]), 1
        while x != s:
            x = int(sigma[x])
            ln += 1
        assert ln == int(Ls[b]), f"ciclo {ln} ≠ L {int(Ls[b])}"
    # [2] d ~ U[1, L−1] dentro de clase, y d ⊥ stake
    for L in L_SET:
        dd = D[Ls == L].numpy()
        assert dd.min() >= 1 and dd.max() <= L - 1
        cnt = np.bincount(dd, minlength=L)[1:L]
        assert cnt.min() > 0.6 * cnt.mean(), f"d no uniforme en L={L}"
    r_ds = np.corrcoef(D.numpy(), Ss.numpy())[0, 1]
    assert abs(r_ds) < 0.04, f"stake~d r={r_ds:.3f}"
    # [3] motivo relacional exacto + marginal p_hi + stake ⊥ L
    hi = (Ss > 1).float()
    assert torch.equal((X[:, 0] == X[:, 1]).float(), hi)
    assert abs(hi.mean().item() - spec.p_hi) < 0.02
    r_Ls = np.corrcoef(Ls.numpy(), Ss.numpy())[0, 1]
    assert abs(r_Ls) < 0.04, f"stake~L r={r_Ls:.3f}"
    # [4] longitud constante, ARROW fijo, vocab en rango
    assert (X[:, ARROW_IDX] == ARROW).all()
    assert int(X.max()) < VOCAB and int(X.min()) >= 0
    assert (Y[:, ARROW_IDX] >= DIST_OFF).all() \
        and (Y[:, ARROW_IDX] < DIST_OFF + D_MAX).all()
    mask = Y != -1
    assert int(mask.sum()) == X.shape[0], "más de una posición supervisada"
    # [5] σ biyectiva, TODOS los índices presentes en los pares, y el
    # orden de los pares BARAJADO (no siempre ascendente)
    n_ordenados = 0
    for b in range(200):
        row = X[b]
        idxs = [int(row[5 + 2 * j]) - ELEM_OFF for j in range(N_ELEM)]
        assert sorted(idxs) == list(range(N_ELEM)), "faltan índices"
        sig = sigma_de_pares(row)
        assert sorted(sig.tolist()) == list(range(N_ELEM))
        if idxs == sorted(idxs):
            n_ordenados += 1
    assert n_ordenados <= 1, "los pares no están barajados"
    # [6] determinismo por seed / divergencia por split
    a1 = N1bDataset(spec, 7).batch(64)
    a2 = N1bDataset(spec, 7).batch(64)
    b1 = N1bDataset(spec, 7, split="eval").batch(64)
    assert torch.equal(a1[0], a2[0]) and torch.equal(a1[3], a2[3])
    assert not torch.equal(a1[0], b1[0])
    # [7] sin atajo trivial: d no correlaciona con NINGÚN token visible
    # (dentro de clase; el único canal legítimo es caminar la tabla)
    for L in L_SET:
        sel = (Ls == L).numpy()
        for pos in (3, 4):                    # s y t crudos
            r = np.corrcoef(X[sel, pos].numpy(), D[sel].numpy())[0, 1]
            assert abs(r) < 0.06, f"d~token pos{pos} r={r:.3f} (L={L})"
    print(f"n1b_env SELF-TESTS 7/7 OK — p_hi={hi.mean():.3f}, "
          f"r(stake,d)={r_ds:+.3f}, r(stake,L)={r_Ls:+.3f}, "
          f"E[d]por L=" + str({L: float(D[Ls == L].float().mean())
                               for L in L_SET}))


if __name__ == "__main__":
    _selftests()
