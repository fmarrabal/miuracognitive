"""Fase 3 v2: descubrimiento de objetivos sobre campos DESCONOCIDOS (post-auditoría).

Regla R5: el agente NO conoce la forma funcional del campo. Cada episodio
sortea una FAMILIA {rbf, bimodal, ridge, plateau} con parámetros aleatorios
(σ incluida); las sondas devuelven la respuesta con RUIDO. El presupuesto de
sondas es limitado y el éxito es GRADUADO (regret normalizado frente al máximo
verdadero del campo; el ruido y el presupuesto impiden regret 0 sistemático).

El campo es diferenciable: el ENTRENADOR puede propagar gradiente a través de
él (meta-aprendizaje legítimo, como learned optimizers); el AGENTE solo ve
pares (posición, respuesta ruidosa). La verdad (f_max, familia) vive aquí SOLO
para la métrica.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Discovery2Config:
    n_probes: int = 12             # presupuesto de sondas por episodio
    obs_noise: float = 0.05
    grid_res: int = 64             # resolución para calcular el máximo verdadero
    sigma_min: float = 0.08
    sigma_max: float = 0.22
    families: tuple = ("rbf", "bimodal", "ridge", "plateau")


class FieldBatch:
    """Batch de campos f:[0,1]²->[0,1] con parámetros aleatorios por episodio.
    f(x) es diferenciable respecto a x (para el meta-entrenamiento)."""

    def __init__(self, cfg: Discovery2Config, batch: int, gen: torch.Generator,
                 device="cpu"):
        self.cfg = cfg
        self.B = batch
        r = lambda *s: torch.rand(*s, generator=gen).to(device)
        self.family = torch.randint(0, len(cfg.families), (batch,),
                                    generator=gen).to(device)
        # parámetros genéricos: hasta 2 centros, sigmas, orientación, pesos
        self.c1 = 0.15 + 0.7 * r(batch, 2)
        self.c2 = 0.15 + 0.7 * r(batch, 2)
        self.s1 = cfg.sigma_min + (cfg.sigma_max - cfg.sigma_min) * r(batch)
        self.s2 = cfg.sigma_min + (cfg.sigma_max - cfg.sigma_min) * r(batch)
        self.h2 = 0.4 + 0.5 * r(batch)               # altura del 2º modo
        self.theta = 3.14159 * r(batch)              # orientación del ridge
        self.width = 0.06 + 0.10 * r(batch)          # anchura ridge/plateau
        self._fmax = None

    def f(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,2) o (B,P,2) -> respuesta (B,) o (B,P) en [0,1]. Diferenciable."""
        squeeze = x.ndim == 2
        if squeeze:
            x = x.unsqueeze(1)                        # (B,1,2)
        P = x.shape[1]
        c1 = self.c1.unsqueeze(1)                     # (B,1,2)
        c2 = self.c2.unsqueeze(1)
        s1 = self.s1.unsqueeze(1)                     # (B,1)
        s2 = self.s2.unsqueeze(1)
        h2 = self.h2.unsqueeze(1)
        th = self.theta.unsqueeze(1)
        w = self.width.unsqueeze(1)
        d1 = ((x - c1) ** 2).sum(-1)                  # (B,P)
        d2 = ((x - c2) ** 2).sum(-1)
        rbf = torch.exp(-d1 / (2 * s1 ** 2))
        bimodal = torch.maximum(rbf, h2 * torch.exp(-d2 / (2 * s2 ** 2)))
        dx = x - c1
        along = dx[..., 0] * torch.cos(th) + dx[..., 1] * torch.sin(th)
        perp = -dx[..., 0] * torch.sin(th) + dx[..., 1] * torch.cos(th)
        ridge = torch.exp(-perp ** 2 / (2 * w ** 2)) \
            * torch.exp(-along ** 2 / (2 * (3 * s1) ** 2))
        plateau = torch.sigmoid((s1 * 1.5 - dx.abs().max(-1).values)
                                / (0.3 * w))
        fam = self.family.unsqueeze(1).expand(-1, P)
        out = torch.where(fam == 0, rbf,
              torch.where(fam == 1, bimodal,
              torch.where(fam == 2, ridge, plateau)))
        out = out.clamp(0.0, 1.0)
        return out.squeeze(1) if squeeze else out

    def probe(self, x: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
        """Respuesta RUIDOSA (lo único que ve el agente). Reparametrizada:
        el ruido es aditivo gaussiano -> diferenciable para el meta-train.
        Con generador explícito (eval reproducible) el ruido se muestrea en el
        device del generador y se mueve: mismos números en CPU y GPU."""
        if gen is not None:
            noise = torch.randn(x.shape[0], generator=gen).to(x.device) \
                * self.cfg.obs_noise
        else:
            noise = torch.randn(x.shape[0], device=x.device) * self.cfg.obs_noise
        return (self.f(x) + noise).clamp(-0.2, 1.2)

    @torch.no_grad()
    def f_max(self) -> torch.Tensor:
        """Máximo verdadero por rejilla densa (solo métrica)."""
        if self._fmax is not None:
            return self._fmax
        g = self.cfg.grid_res
        lin = torch.linspace(0.0, 1.0, g, device=self.c1.device)
        xx, yy = torch.meshgrid(lin, lin, indexing="ij")
        pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)   # (G²,2)
        best = torch.zeros(self.B, device=self.c1.device)
        chunk = 1024
        for i in range(0, pts.shape[0], chunk):
            p = pts[i:i + chunk].unsqueeze(0).expand(self.B, -1, 2)    # (B,C,2)
            best = torch.maximum(best, self.f(p).max(-1).values)
        self._fmax = best.clamp_min(1e-4)
        return self._fmax


def normalized_regret(fields: FieldBatch, target: torch.Tensor) -> torch.Tensor:
    """regret = 1 − f(objetivo)/f_max ∈ [0,1] (graduado; verdad solo aquí)."""
    return (1.0 - fields.f(target) / fields.f_max()).clamp(0.0, 1.0)
