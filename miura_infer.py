"""
MiuraCognitive — API de USO (inferencia + introspección)
=========================================================
Envuelve un modelo entrenado en una interfaz limpia y reutilizable:

    from miura_infer import MiuraModel
    m = MiuraModel.from_checkpoint("checkpoints/hbp_full.pt")
    r = m.compose([0, 2, 1, 3, 1])        # aplica 5 transposiciones adyacentes de S_5
    print(r.predicted, r.correct, r.n_iter)
    for step in r.trace: print(step)      # traza del "pensamiento" del HBP por tick

Guardar un checkpoint tras entrenar:  MiuraModel.save(model, ds, path, meta=...)

El valor de esta arquitectura es la INTROSPECCIÓN: cada inferencia devuelve la
traza del campo homeostático (VEI) a lo largo de las iteraciones del reasoner —
cuántas iteró, cómo evolucionó su estado interno y su modulación.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import torch

from data.synthetic_recall import PermutationCompositionDataset
from training.trainer import build_model


@dataclass
class ComposeResult:
    ops: list[int]
    true_perm: tuple
    predicted: tuple | None
    correct: bool
    n_iter: float                 # E[nº iteraciones] del reasoner ("cuánto pensó")
    confidence: float             # prob. asignada a la respuesta predicha
    trace: list[dict] = field(default_factory=list)   # dinámica del HBP por tick
    physics: dict | None = None   # régimen PDE elegido: α (onda↔difusión), D, b

    def __repr__(self):
        pp = "".join(map(str, self.predicted)) if self.predicted else "??"
        tp = "".join(map(str, self.true_perm))
        mark = "OK" if self.correct else "X"
        return (f"ComposeResult(ops={self.ops}, true={tp}, pred={pp} [{mark}], "
                f"n_iter={self.n_iter:.2f}, conf={self.confidence:.2f})")


class MiuraModel:
    """Modelo MiuraCognitive listo para usar."""

    def __init__(self, model, ds, device=None, meta=None):
        self.model = model.eval()
        self.ds = ds
        self.device = device or next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.meta = meta or {}

    # ---------------------------------------------------------------- carga/guardado
    @staticmethod
    def save(model, ds, path: str, build_args: dict, meta: dict | None = None):
        """Guarda pesos + argumentos de construcción + metadatos (tarjeta del modelo)."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "build_args": build_args,          # variant, vocab_size, seq_len, halt_mod_gain, ...
            "task": {"kind": "permcomp", "n_elems": ds.n, "gen_set_size": ds.n_gen,
                     "vocab_size": ds.vocab_size, "seq_len": ds.seq_len},
            "meta": meta or {},
        }, path)

    @classmethod
    def from_checkpoint(cls, path: str, device: str | None = None):
        dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
        ckpt = torch.load(path, map_location=dev, weights_only=False)
        t = ckpt["task"]
        ds = PermutationCompositionDataset(n_elems=t["n_elems"], seq_len=t["seq_len"], seed=0,
                                           gen_set=ckpt["meta"].get("gen_set", "adjacent"))
        model = build_model(**ckpt["build_args"]).to(device=dev, dtype=dtype)
        # Compatibilidad con checkpoints anteriores al panel teórico: adv,
        # rho_A y mu_A son buffers geométricos DETERMINISTAS reconstruidos desde
        # n_nodes, no pesos aprendidos. Permitimos exclusivamente que falten esos
        # buffers (y los análogos KdV si el modelo los crea); cualquier otra
        # discrepancia sigue siendo un error para no ocultar pesos ausentes.
        incompatible = model.load_state_dict(ckpt["state_dict"], strict=False)
        deterministic = {
            "hbp.adv", "hbp.rho_A", "hbp.mu_A", "hbp.adv3", "hbp.rho_A3"
        }
        bad_missing = set(incompatible.missing_keys) - deterministic
        if bad_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "checkpoint incompatible; missing="
                f"{sorted(bad_missing)}, unexpected={sorted(incompatible.unexpected_keys)}")
        if model.cfg.use_hbp:
            model.hbp.pin_fp32()   # FIX BF16 (por si se sigue entrenando)
        return cls(model, ds, device=dev, meta=ckpt.get("meta", {}))

    # ---------------------------------------------------------------- inferencia
    @torch.no_grad()
    def compose(self, ops: list[int], trace: bool = True) -> ComposeResult:
        """Aplica en orden los generadores indicados al estado identidad de S_n y
        devuelve la respuesta del modelo + la traza del HBP."""
        ds = self.ds
        seq, running = [], ds.identity
        for j in ops:
            seq.append(ds.gen_offset + j)
            running = ds._compose(ds.gens[j], running)
        seq.append(ds.Q)
        arrow_pos = len(seq)
        seq.append(ds.ARROW)
        seq = seq + [ds.PAD] * (ds.seq_len - len(seq))
        tokens = torch.tensor(seq, dtype=torch.long, device=self.device).unsqueeze(0)

        want = getattr(self.model, "record_trace", False)
        self.model.record_trace = trace
        logits, _ = self.model(tokens, None)
        self.model.record_trace = want

        a0, a1 = ds.answer_offset, ds.answer_offset + ds.answer_size
        probs = torch.softmax(logits[0, arrow_pos, a0:a1].float(), dim=-1)
        conf, rel = probs.max(dim=-1)
        pred = ds.perms[int(rel)]
        n_iter = float(self.model._last_n_expected[0].item()) if self.model._last_n_expected is not None else 0.0
        phys = None
        if self.model.cfg.use_hbp and hasattr(self.model.hbp, "physics_state"):
            phys = self.model.hbp.physics_state()
        return ComposeResult(
            ops=list(ops), true_perm=running, predicted=pred, correct=(pred == running),
            n_iter=n_iter, confidence=float(conf),
            trace=list(self.model._trace) if trace else [], physics=phys)

    @torch.no_grad()
    def introspect(self) -> dict:
        """Diagnóstico del campo homeostático del modelo (evolución libre)."""
        from eval.diagnostics import run_full_diagnostics
        if not self.model.cfg.use_hbp:
            return {"use_hbp": False}
        rep = run_full_diagnostics(self.model.hbp, n_ticks=200,
                                   device=self.device, dtype=self.dtype)
        out = {"use_hbp": True,
               "vei_variance": rep["variance"]["var_mean"],
               "dominant_freq": rep["spectrum"]["dominant_freq"],
               "oscillates": rep["spectrum"]["has_oscillation"],
               "zeta_mean": rep["damping"]["zeta_mean"],
               "regimes": rep["damping"]["regime_counts"]}
        if getattr(self.model.hbp.cfg, "gate_physics", False):
            out["gate_physics"] = True
            out["physics"] = self.model.hbp.physics_state()   # α, D, b tras la evolución
        return out

    def info(self) -> dict:
        m = self.model
        return {"variant": m.cfg.__dict__.get("use_hbp") and self.meta.get("variant", "?"),
                "n_params": m.num_parameters(),
                "max_halt_steps": m.cfg.max_halt_steps,
                "use_hbp": m.cfg.use_hbp, "device": str(self.device),
                "meta": self.meta}
