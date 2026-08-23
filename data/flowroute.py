"""FlowRoute: enrutamiento simbólico con ground-truth NO-ADVECTIVA (cierra la
circularidad que el panel marcó como fatal). Sobre malla H×W:

  · K FUENTES (posiciones aleatorias) cada una con un símbolo one-hot 0..K-1.
  · K DESTINOS (posiciones aleatorias) etiquetados 0..K-1.
  · Asignación (GT): símbolo k -> destino k. Regla SIMBÓLICA (etiquetas), NUNCA
    calculada por advección: ningún flujo toca jamás la etiqueta (cierra R5).
  · El agente debe hacer que S del símbolo k LLEGUE FÍSICAMENTE a la celda del
    destino k, porque el READOUT es LOCAL (se lee S en la celda-destino). Esto
    cierra el atajo de lookup: aunque la asignación sea trivial de leer, hay que
    TRANSPORTAR el contenido para que la cabeza local lo lea (R: transporte).

Clave anti-set-point: un flujo incompresible 2D NO cruza streamlines. Las
asignaciones con CRUCES (rutas i,j cuyo orden fuente≠orden destino) son
IRREALIZABLES por un flujo ESTÁTICO. Se mide el desglose crossing/no-crossing:
si el flujo dinámico (ω(t)) resuelve cruces y el estático no, la DINÁMICA
temporal es load-bearing (escapa al set-point). Métrica graduada (fracción de
símbolos entregados), sin techo (los cruces limitan el máximo del flujo).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FlowRouteConfig:
    H: int = 12
    W: int = 12
    K: int = 3                 # símbolos / fuentes / destinos
    src_cols: tuple = (1, 4)   # banda de columnas de fuentes
    dst_cols: tuple = (7, 10)  # banda de columnas de destinos
    blob_sigma: float = 0.8    # anchura del one-hot inicial (suave, advectable)


@dataclass
class FlowRouteScenario:
    S0: torch.Tensor           # (B,K,H,W) canales one-hot de símbolo en fuentes
    src_rc: torch.Tensor       # (B,K,2) fila,col de la fuente del símbolo k
    dst_rc: torch.Tensor       # (B,K,2) fila,col del destino etiquetado k
    input_grid: torch.Tensor   # (B,C,H,W) input común a TODAS las arquitecturas
    crossings: torch.Tensor    # (B,) nº de pares de rutas que se cruzan

    def to(self, dev):
        return FlowRouteScenario(*(getattr(self, f).to(dev) for f in
                                   ("S0", "src_rc", "dst_rc", "input_grid", "crossings")))


class FlowRouteDataset:
    def __init__(self, cfg: FlowRouteConfig | None = None, seed: int = 0):
        self.cfg = cfg or FlowRouteConfig()
        self.gen = torch.Generator().manual_seed(seed)

    def _rows(self, B, K):
        # K filas distintas por instancia (permutación de filas interiores)
        out = torch.zeros(B, K, dtype=torch.long)
        for b in range(B):
            out[b] = torch.randperm(self.cfg.H - 2, generator=self.gen)[:K] + 1
        return out

    def batch(self, B: int) -> FlowRouteScenario:
        c = self.cfg
        H, W, K = c.H, c.W, c.K
        src_rows = self._rows(B, K)
        dst_rows = self._rows(B, K)
        src_col = torch.randint(c.src_cols[0], c.src_cols[1] + 1, (B, K), generator=self.gen)
        dst_col = torch.randint(c.dst_cols[0], c.dst_cols[1] + 1, (B, K), generator=self.gen)
        src_rc = torch.stack([src_rows, src_col], -1)          # (B,K,2)
        dst_rc = torch.stack([dst_rows, dst_col], -1)

        # S0: blob gaussiano del símbolo k en su fuente
        yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
        S0 = torch.zeros(B, K, H, W)
        src_mark = torch.zeros(B, 1, H, W)
        dst_lbl = torch.zeros(B, K, H, W)                       # destino etiquetado k
        for b in range(B):
            for k in range(K):
                r, col = int(src_rc[b, k, 0]), int(src_rc[b, k, 1])
                S0[b, k] = torch.exp(-((yy - r) ** 2 + (xx - col) ** 2) / (2 * c.blob_sigma ** 2))
                src_mark[b, 0, r, col] = 1.0
                dr, dc = int(dst_rc[b, k, 0]), int(dst_rc[b, k, 1])
                dst_lbl[b, k, dr, dc] = 1.0
        # input común: canales de símbolo en fuente + marca de destino etiquetado
        input_grid = torch.cat([S0, dst_lbl], dim=1)           # (B, 2K, H, W)

        # nº de cruces: pares (i,j) con orden de fila fuente != orden fila destino
        crossings = torch.zeros(B)
        for b in range(B):
            for i in range(K):
                for j in range(i + 1, K):
                    if (src_rows[b, i] - src_rows[b, j]) * (dst_rows[b, i] - dst_rows[b, j]) < 0:
                        crossings[b] += 1
        return FlowRouteScenario(S0, src_rc, dst_rc, input_grid, crossings)


@torch.no_grad()
def route_accuracy(cfg: FlowRouteConfig, sc: FlowRouteScenario, S_final: torch.Tensor,
                   local_window: int = 1) -> dict:
    """READOUT LOCAL: en cada celda-destino k, se lee S en una ventana pequeña y
    se predice el símbolo por argmax de canal. Correcto si el símbolo dominante
    en el destino k es k. Graduado: fracción de destinos correctos. Desglose por
    nº de cruces."""
    B, K = sc.S0.shape[0], cfg.K
    dev = S_final.device
    correct = torch.zeros(B, K, device=dev)
    w = local_window
    for k in range(K):
        rc = sc.dst_rc[:, k]                                   # (B,2)
        for b in range(B):
            r, col = int(rc[b, 0]), int(rc[b, 1])
            r0, r1 = max(0, r - w), min(cfg.H, r + w + 1)
            c0, c1 = max(0, col - w), min(cfg.W, col + w + 1)
            patch = S_final[b, :, r0:r1, c0:c1].sum(dim=(1, 2))  # (K,) masa por símbolo
            pred = int(patch.argmax())
            correct[b, k] = float(pred == k)
    acc = correct.mean().item()
    cr = sc.crossings.to(dev)
    out = {"acc": acc,
           "acc_nocross": correct[cr == 0].mean().item() if (cr == 0).any() else float("nan"),
           "acc_cross": correct[cr > 0].mean().item() if (cr > 0).any() else float("nan")}
    return out
