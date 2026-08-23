"""
N2 — Entorno (PREREG_N2 v2 §2): cycle_transp K∈[13,24] + MOTIVO de stake.

Diseño del motivo (decisiones declaradas):
  · 2 SLOTS en posiciones fijas 0-1, antepuestos a toda instancia (longitud
    CONSTANTE: stake ⊥ longitud por construcción; frac_valid idéntica).
  · Alfabeto de slots = 4 tokens de PERMUTACIÓN existentes {5,6,7,8} (los
    F3a-ckpts tienen embedding entrenado → VG2b/VG3 son medibles; los
    tokens de generador alterarían la COMPOSICIÓN y PAD fugaría longitud;
    la invariante «perm nunca como input» se relaja aquí sin atajo de copia
    — los slots son independientes de la respuesta — y la interferencia la
    mide VG3 con acc(n,K)|motivo).
  · REGLA: stake = s_alto ⟺ el par de slots es EXACTAMENTE (5, 6) (el
    motivo); stake bajo → par uniforme de los 15 restantes. Se asigna el
    stake PRIMERO (Bernoulli p_hi, stream propio ⊥ del de la tarea) y los
    slots después → stake ⊥ K exacto por construcción.
  · Labels desplazadas +2 (slots a -1); answer_pos = K+3; ARROW sigue
    siendo el primer token 2 de la secuencia (readout intacto).

Brazos (§4): los PESOS de CE los da weights_for_arm — el modelo es idéntico.

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.n2_env   (self-tests)
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch

from data.synthetic_recall import PermutationCompositionDataset

SLOT_ALPHABET = (5, 6, 7, 8)          # perm tokens con embedding entrenado
# RONDA 1 de calibración del motivo (2026-08-04; motivo: VG3b FALLÓ con el
# par concreto (5,6) — Δacc −0.025/−0.056 en n=12/24, el token 5 es la
# permutación identidad e interfiere con la supervisión; VG3a marginal con
# r_E[n] idiosincrático por token). El motivo pasa a ser una RELACIÓN:
#   stake alto ⟺ s0 == s1 (slots IGUALES)
# → la marginal de CADA token es idéntica entre clases por construcción
# (alto: s0=s1~U(4); bajo: par desigual ~U(12)) y los efectos por-token se
# cancelan; solo difiere la correlación entre slots. VG3b re-verificado
# comparando pares iguales vs desiguales.


def is_motif_pair(s0: int, s1: int) -> bool:
    return s0 == s1


@dataclass(frozen=True)
class N2Spec:
    k_lo: int = 13
    k_hi: int = 24
    p_hi: float = 0.10                # dial VG1 (celda provisional En8/s8/p.10)
    s_alto: float = 8.0
    seq_len: int = 128                # total (incluye los 2 slots)


class N2Dataset:
    """Genera batches (inputs, targets, K, stakes) con el motivo. El stream
    de stakes/slots es PROPIO (numpy PCG64), independiente del de la tarea
    (torch.Generator del dataset base) — stake ⊥ contenido por construcción."""

    def __init__(self, spec: N2Spec, seed: int, split: str = "train"):
        self.spec = spec
        off = {"train": 0, "eval": 800_000, "val": 400_000}[split]
        self.base = PermutationCompositionDataset(
            min_ops=spec.k_lo, max_ops=spec.k_hi,
            seq_len=spec.seq_len - 2, seed=seed + off,
            gen_set="cycle_transp")
        self.rng = np.random.default_rng(seed * 1000 + 13 + off)
        self.vocab_size = self.base.vocab_size
        self.answer_offset = self.base.answer_offset
        self.answer_size = self.base.answer_size

    def _slots(self, alto: bool):
        if alto:
            t = SLOT_ALPHABET[int(self.rng.integers(0, 4))]
            return t, t                            # par IGUAL (el motivo)
        while True:
            a = SLOT_ALPHABET[int(self.rng.integers(0, 4))]
            b = SLOT_ALPHABET[int(self.rng.integers(0, 4))]
            if a != b:
                return a, b                        # par desigual

    def batch(self, batch_size: int):
        toks, labs, ks, stakes = [], [], [], []
        for _ in range(batch_size):
            t, l, k = self.base.sample()
            alto = bool(self.rng.random() < self.spec.p_hi)
            s0, s1 = self._slots(alto)
            t2 = torch.cat([torch.tensor([s0, s1], dtype=torch.long), t])
            l2 = torch.cat([torch.tensor([-1, -1], dtype=torch.long), l])
            toks.append(t2)
            labs.append(l2)
            ks.append(k)
            stakes.append(self.spec.s_alto if alto else 1.0)
        tokens = torch.stack(toks)
        labels = torch.stack(labs)
        inputs = tokens[:, :-1].contiguous()
        targets = labels[:, :-1].contiguous()
        return (inputs, targets, torch.tensor(ks),
                torch.tensor(stakes, dtype=torch.float32))


def weights_for_arm(arm: str, stakes: torch.Tensor,
                    rng: np.random.Generator | None = None) -> torch.Tensor:
    """Pesos de CE por brazo (§4): la ESTRUCTURA DE PAGO del entorno.
      endo / endo_noval / oracle : stake real
      blind_flat                 : uniforme = E[stake] (control sin ruido)
      blind_shuffle              : PERMUTACIÓN del batch (marginal idéntica)
    """
    if arm in ("endo", "endo_noval", "oracle", "gating_endo"):
        return stakes.clone()
    if arm == "blind_flat":
        return torch.full_like(stakes, float(stakes.mean()))
    if arm == "blind_shuffle":
        perm = rng.permutation(len(stakes))
        return stakes[torch.as_tensor(perm.copy(), dtype=torch.long)]
    raise ValueError(f"brazo desconocido: {arm}")


# ------------------------------- self-tests ------------------------------- #
def _selftests():
    spec = N2Spec()
    ds = N2Dataset(spec, seed=0)
    xs, ts, ks, ss = ds.batch(4000)
    hi = (ss > 1).float()
    # [1] marginal de stakes
    assert abs(hi.mean().item() - spec.p_hi) < 0.02, "p_hi fuera de rango"
    # [2] regla determinista: motivo (slots IGUALES) ⟺ stake alto
    is_motif = (xs[:, 0] == xs[:, 1]).float()
    assert torch.equal(is_motif, hi), "la regla motivo⟺stake no es exacta"
    # [2b] marginal por-token idéntica entre clases (la razón de la ronda 1)
    for tok in SLOT_ALPHABET:
        f_hi = ((xs[hi > 0, :2] == tok).float().mean())
        f_lo = ((xs[hi == 0, :2] == tok).float().mean())
        assert abs(float(f_hi - f_lo)) < 0.05, \
            f"marginal del token {tok} difiere entre clases"
    # [3] stake ⊥ K
    r = np.corrcoef(ks.numpy(), ss.numpy())[0, 1]
    assert abs(r) < 0.04, f"stake correlaciona con K: r={r:.3f}"
    # [4] longitud constante y sin PAD en slots
    assert (xs[:, :2] > 0).all(), "PAD en slots (fuga de longitud)"
    valid = (xs != 0).sum(1).float()
    r2 = np.corrcoef(valid.numpy(), ss.numpy())[0, 1]
    assert abs(r2) < 0.04, f"stake correlaciona con longitud: r={r2:.3f}"
    # [5] composición intacta: la label del ARROW es la composición de los
    # gens de las posiciones 2..K+1 (desplazamiento +2 correcto)
    d0 = ds.base
    for b in range(50):
        row = xs[b]
        gens = [int(g) - d0.gen_offset for g in row[2:2 + int(ks[b])]]
        running = d0.identity
        for j in gens:
            running = d0._compose(d0.gens[j], running)
        arrow_pos = int((row == d0.ARROW).float().argmax())
        assert arrow_pos == int(ks[b]) + 3, "answer_pos desplazado mal"
        want = d0.perm_offset + d0.perm_index[running]
        assert int(ts[b, arrow_pos]) == want, "label de ARROW no es la composición"
    # [6] determinismo por seed y divergencia por split
    a1 = N2Dataset(spec, 7).batch(64)
    a2 = N2Dataset(spec, 7).batch(64)
    b1 = N2Dataset(spec, 7, split="eval").batch(64)
    assert torch.equal(a1[0], a2[0]) and torch.equal(a1[3], a2[3])
    assert not torch.equal(a1[0], b1[0]), "eval no diverge de train"
    # [7] pesos por brazo
    rng = np.random.default_rng(0)
    w_flat = weights_for_arm("blind_flat", ss)
    assert float(w_flat.std()) < 1e-5, "blind_flat no es constante"
    w_sh = weights_for_arm("blind_shuffle", ss, rng)
    assert torch.equal(w_sh.sort().values, ss.sort().values), \
        "el barajado no preserva la marginal"
    assert not torch.equal(w_sh, ss)
    print("n2_env SELF-TESTS 7/7 OK — "
          f"p_hi_emp={hi.mean():.3f}, r(stake,K)={r:+.3f}, "
          f"r(stake,len)={r2:+.3f}")


if __name__ == "__main__":
    _selftests()
