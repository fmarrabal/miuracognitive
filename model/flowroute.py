"""FlowRoute: controlador de vorticidad (ω-net) sobre el sustrato flow2d.

Modos:
  · dynamic: ω_t = ω_net(input, S_t) por tick -> flujo TIME-VARYING (dinámica real).
  · static : ω = ω_net(input) una vez, constante T ticks -> flujo estacionario
             (SKYLINE del set-point: si esto resuelve, la dinámica es inerte).
  · handset: ω artesanal (dipolo fuente->destino) SIN entrenar (GATE 1).
El READOUT es LOCAL (route_accuracy lee S en la celda-destino).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from data.flowroute import FlowRouteConfig, FlowRouteScenario, route_accuracy
from model.flow2d import Flow2DConfig, Flow2DField


class OmegaNet(nn.Module):
    """CNN pequeña input(+S) -> ω (H,W), acotada por tanh·escala (CFL)."""

    def __init__(self, in_ch: int, hidden: int = 32, scale: float = 20.0):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.scale * torch.tanh(self.net(x).squeeze(1))    # (B,H,W)


class FlowRouteModel(nn.Module):
    def __init__(self, cfg: FlowRouteConfig, flow_cfg: Flow2DConfig,
                 mode: str = "dynamic", T: int = 16, hidden: int = 32):
        super().__init__()
        self.cfg, self.T, self.mode = cfg, T, mode
        self.flow = Flow2DField(flow_cfg)
        in_ch = 2 * cfg.K + (cfg.K if mode == "dynamic" else 0)  # +S_t si dinámico
        self.omega_net = OmegaNet(in_ch, hidden)

    def rollout(self, sc: FlowRouteScenario):
        S = sc.S0                                                # (B,K,H,W)
        B, K, H, W = S.shape
        inp = sc.input_grid
        omega_static = None
        if self.mode == "static":
            omega_static = self.omega_net(inp)                   # una vez
        for t in range(self.T):
            if self.mode == "dynamic":
                omega = self.omega_net(torch.cat([inp, S], dim=1))
            else:
                omega = omega_static
            # advecta cada canal de símbolo por el MISMO flujo (una ω -> un ψ)
            psi = self.flow.poisson(omega)
            ux, uy = self.flow.velocity(psi)
            chans = []
            for k in range(K):
                sk = self.flow.advect(S[:, k], ux, uy)
                sk = self.flow.diffuse(sk)
                chans.append(sk)
            S = torch.stack(chans, dim=1)
        return S

    def forward(self, sc):
        return self.rollout(sc)


def handset_omega(cfg: FlowRouteConfig, sc: FlowRouteScenario, flow: Flow2DField):
    """ω artesanal: superposición de dipolos que empujan cada fuente hacia su
    destino (GATE 1). Un dipolo = +vorticidad arriba / −abajo del vector
    fuente->destino crea flujo a lo largo de él. SIN entrenar."""
    B, K = sc.S0.shape[0], cfg.K
    H, W = cfg.H, cfg.W
    dev = sc.S0.device
    yy, xx = torch.meshgrid(torch.arange(H, device=dev).float(),
                            torch.arange(W, device=dev).float(), indexing="ij")
    omega = torch.zeros(B, H, W, device=dev)
    for b in range(B):
        for k in range(K):
            sr, scl = float(sc.src_rc[b, k, 0]), float(sc.src_rc[b, k, 1])
            dr, dc = float(sc.dst_rc[b, k, 0]), float(sc.dst_rc[b, k, 1])
            mx, my = (sr + dr) / 2, (scl + dc) / 2               # punto medio
            # eje perpendicular al vector fuente->destino
            vx, vy = dc - scl, dr - sr
            n = (vx ** 2 + vy ** 2) ** 0.5 + 1e-6
            px, py = -vy / n, vx / n                             # perpendicular
            off = 1.5
            omega[b] += 15 * torch.exp(-(((xx - (my + px * off)) ** 2 + (yy - (mx + py * off)) ** 2)) / 2)
            omega[b] -= 15 * torch.exp(-(((xx - (my - px * off)) ** 2 + (yy - (mx - py * off)) ** 2)) / 2)
    return omega


def train_flowroute(cfg, flow_cfg, mode, seed, steps=1500, batch=128,
                    lr=2e-3, T=16, device="cpu", ds_seed=None):
    """Entrena por SOFT-accuracy: masa del símbolo k en su destino (diferenciable),
    penaliza masa en destinos equivocados. Objetivo AMBIENTAL, sin lookup."""
    import math
    from data.flowroute import FlowRouteDataset
    torch.manual_seed(seed)
    ds = FlowRouteDataset(cfg, seed=ds_seed if ds_seed is not None else seed)
    m = FlowRouteModel(cfg, flow_cfg, mode=mode, T=T).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    m.train()
    H, W = cfg.H, cfg.W
    yy, xx = torch.meshgrid(torch.arange(H, device=device).float(),
                            torch.arange(W, device=device).float(), indexing="ij")
    for s in range(1, steps + 1):
        cur = lr * min(1.0, s / 50) * 0.5 * (1 + math.cos(math.pi * s / steps))
        for g in opt.param_groups:
            g["lr"] = cur
        sc = ds.batch(batch).to(device)
        S = m(sc)
        B = sc.S0.shape[0]
        # objetivo por SOLAPAMIENTO: el canal k debe parecerse a un blob-objetivo
        # centrado en el destino k (gradiente fuerte y espacial, no una sola celda).
        loss = 0.0
        for k in range(cfg.K):
            dr = sc.dst_rc[:, k, 0].view(B, 1, 1).float()
            dc = sc.dst_rc[:, k, 1].view(B, 1, 1).float()
            tgt = torch.exp(-((yy - dr) ** 2 + (xx - dc) ** 2) / (2 * 1.0 ** 2))  # (B,H,W)
            tgt = tgt / tgt.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            deliver = (S[:, k] * tgt).sum(dim=(1, 2))           # solapamiento con SU destino
            # masa del canal k en destinos AJENOS (penaliza colisión)
            wrong = 0.0
            for j in range(cfg.K):
                if j == k:
                    continue
                dr2 = sc.dst_rc[:, j, 0].view(B, 1, 1).float()
                dc2 = sc.dst_rc[:, j, 1].view(B, 1, 1).float()
                t2 = torch.exp(-((yy - dr2) ** 2 + (xx - dc2) ** 2) / (2 * 1.0 ** 2))
                t2 = t2 / t2.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
                wrong = wrong + (S[:, k] * t2).sum(dim=(1, 2))
            loss = loss - deliver.mean() + wrong.mean()
        loss = loss / cfg.K
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return m
