"""
Controladores de la Fase 2 — API unificada: reset(B, device); step(signals)->acts.

Equiparación (PREREG): TODOS consumen las MISMAS señales (canales canónicos vía
InteroceptiveSignal + RunningNorm) y actúan por la MISMA proyección acotada
(ActuatorHead: z0, Π_A sigmoid-escalada, STE). Solo difiere el NÚCLEO:
  mhbp*      : CoupledMultiscaleHBP (variantes por config)
  hbp_single : Q=1 campo con estado equiparado (N=22, d=8 -> 176)
  gru        : GRUCell h=24        (memoria genérica; baseline fuerte)
  mlp        : MLP 26->64->32      (reactivo, sin memoria)
  ar2        : z_t = tanh(A1 z_{t-1} + A2 z_{t-2} + B s_t), z=32 (2º orden sin física)
"""
from __future__ import annotations
import torch
import torch.nn as nn

from ...coupled_fields import MHBPConfig, CoupledMultiscaleHBP, default_four_fields
from ...field import FieldConfig
from ...interoception import InteroceptiveSignal, RunningNorm, N_CHANNELS
from ...actuators import ActuatorHead


class _PseudoCfg:
    """Objeto mínimo con .d y .n_nodes para reutilizar ActuatorHead."""
    def __init__(self, d):
        self.d = d
        self.n_nodes = 1


# --------------------------------------------------------------------------- #
class MHBPController(nn.Module):
    """Envuelve CoupledMultiscaleHBP. `variant` fija la ablation."""

    def __init__(self, variant: str = "mhbp"):
        super().__init__()
        self.variant = variant
        if variant == "mhbp":
            cfg = MHBPConfig(theta=1.0, coupling_topology="chain", allostasis=True)
            fcs = None
        elif variant == "mhbp_nocpl":
            # SOLO quita el acoplamiento (la alostasis queda): atribución limpia
            cfg = MHBPConfig(theta=1.0, coupling_topology="none", allostasis=True)
            fcs = None
        elif variant == "mhbp_noallo":
            # SOLO quita la alostasis (el acoplamiento queda)
            cfg = MHBPConfig(theta=1.0, coupling_topology="chain", allostasis=False)
            fcs = None
        elif variant == "mhbp_taueq":
            cfg = MHBPConfig(theta=1.0, taus=(4.0, 4.0, 4.0, 4.0),
                             coupling_topology="chain", allostasis=True)
            fcs = None
        elif variant == "mhbp_first":
            cfg = MHBPConfig(integrator="first_order", coupling_topology="chain",
                             allostasis=True)
            fcs = None
        elif variant == "hbp_single":
            cfg = MHBPConfig(taus=(1.0,), coupling_topology="none", allostasis=False)
            fcs = [FieldConfig(name="single", n_nodes=22, d=8)]   # 176 = Σ N_q·d del mHBP
        else:
            raise ValueError(variant)
        self.model = CoupledMultiscaleHBP(cfg, field_cfgs=fcs).double()

    def reset(self, B: int, device):
        self.model.to(device)
        self.model.reset_state(B, device=device)
        self._reg_sum = torch.zeros((), device=device)
        self._reg_n = 0

    def step(self, signals: dict) -> dict:
        acts, _ = self.model.tick(InteroceptiveSignal(values=signals))
        # acumula el anti-bypass POR TICK (antes solo contaba el último tick:
        # ~T× más débil de lo declarado — hallazgo del panel)
        self._reg_sum = self._reg_sum + self.model.regularizers()["allostasis"]
        self._reg_n += 1
        return acts

    def regularizer(self) -> torch.Tensor:
        if self._reg_n == 0:
            return torch.zeros(())
        return self._reg_sum / self._reg_n

    def core_param_count(self) -> int:
        n = sum(p.numel() for f in self.model.fields for p in f.params.parameters())
        n += sum(p.numel() for f in self.model.fields for p in [f.h_star])
        if self.model.coupling is not None:
            n += sum(p.numel() for p in self.model.coupling.parameters())
        n += sum(p.numel() for p in self.model.allo.parameters()) if self.model.allo.enabled else 0
        return n


# --------------------------------------------------------------------------- #
class BaselineController(nn.Module):
    """Núcleo genérico (mlp | gru | ar2) con la misma featurización y actuadores."""

    def __init__(self, core: str, gru_h: int = 40, mlp_h: int = 96,
                 mlp_out: int = 48, ar_z: int = 44):
        # tamaños elegidos para que el TOTAL de cada baseline sea >= el total del
        # mhbp (~8.3k): equiparación conservadora A FAVOR de los baselines
        super().__init__()
        self.core_type = core
        self.norm = RunningNorm()
        if core == "gru":
            self.cell = nn.GRUCell(N_CHANNELS, gru_h)
            feat = gru_h
        elif core == "mlp":
            self.net = nn.Sequential(nn.Linear(N_CHANNELS, mlp_h), nn.SiLU(),
                                     nn.Linear(mlp_h, mlp_out), nn.Tanh())
            feat = mlp_out
        elif core == "ar2":
            self.A1 = nn.Linear(ar_z, ar_z, bias=False)
            self.A2 = nn.Linear(ar_z, ar_z, bias=False)
            self.Bin = nn.Linear(N_CHANNELS, ar_z)
            # init contractivo (espectro < 1) para arrancar estable
            with torch.no_grad():
                for M in (self.A1.weight, self.A2.weight):
                    M.mul_(0.4 / max(1.0, float(torch.linalg.svdvals(M).max())))
            feat = ar_z
        else:
            raise ValueError(core)
        self.feat_dim = feat
        self.head = ActuatorHead([_PseudoCfg(feat)])
        self._h = None
        self._z1 = None
        self._z2 = None

    def reset(self, B: int, device):
        self.to(device)
        dt = next(self.parameters()).dtype
        self._h = torch.zeros(B, self.feat_dim, device=device, dtype=dt)
        self._z1 = torch.zeros(B, self.feat_dim, device=device, dtype=dt)
        self._z2 = torch.zeros(B, self.feat_dim, device=device, dtype=dt)

    def _features(self, signals: dict, B: int, device) -> torch.Tensor:
        sig = InteroceptiveSignal(values=signals)
        s, m = sig.to_tensor(B, device, dtype=self.norm.mean.dtype)
        return self.norm(s, m)

    def step(self, signals: dict) -> dict:
        B = self._h.shape[0]
        device = self._h.device
        x = self._features(signals, B, device)
        if self.core_type == "gru":
            self._h = self.cell(x, self._h)
            feat = self._h
        elif self.core_type == "mlp":
            feat = self.net(x)
        else:                                            # ar2
            z = torch.tanh(self.A1(self._z1) + self.A2(self._z2) + self.Bin(x))
            self._z2, self._z1 = self._z1, z
            feat = z
        w = torch.zeros_like(feat)                       # pseudo-velocidad nula
        acts, _ = self.head([(feat, w)])
        return acts

    def regularizer(self) -> torch.Tensor:
        return torch.zeros((), device=self._h.device if self._h is not None else "cpu")

    def core_param_count(self) -> int:
        if self.core_type == "gru":
            return sum(p.numel() for p in self.cell.parameters())
        if self.core_type == "mlp":
            return sum(p.numel() for p in self.net.parameters())
        return (self.A1.weight.numel() + self.A2.weight.numel()
                + sum(p.numel() for p in self.Bin.parameters()))


# --------------------------------------------------------------------------- #
class GovernorController(nn.Module):
    """Fase 2b — el campo como MODULADOR, no como fuente (PREREG_PHASE2B.md v2).

    `react`   : lazo reactivo puro — señales → MLP(96) → logits SIN BIAS final
                → Π_A. El "arco reflejo". (Sin bias: cierra la vía constante
                g·b por la que la ganancia se volvía fuente — panel 2b.)
    `mhbp_gov`: el mismo lazo, con GANANCIAS multiplicativas acotadas moduladas
                por el CAMPO: z = z0 + g ⊙ z_react, g = 1 + γ·tanh(V·estado).
    `gru_gov` : idéntico, pero la ganancia la produce un GRUCell (h=40) — el
                control que AÍSLA la física del campo de un gain-scheduler
                recurrente genérico (contraste D2b, el que importa).
    Init de la vía de ganancia ≈ 0 ⇒ g≈1 ⇒ ambos ARRANCAN siendo react.
    NOTA del panel: g·z_react es bilineal — la pureza del rol es CONDUCTUAL
    (init + cota g∈[0.2,1.8] + sin bias), no un imposible arquitectural; las
    intervenciones post-hoc (g≡1, g-media, g-barajada, desde checkpoints)
    separan modulación de fuente."""

    GAMMA = 0.8

    def __init__(self, gain_core: str | None, react_h: int = 96, gru_h: int = 40):
        super().__init__()
        assert gain_core in (None, "field", "gru")
        self.gain_core = gain_core
        self.norm = RunningNorm()
        from ...actuators import MVP_ACTUATORS, _inv_sigmoid_t
        self.specs = list(MVP_ACTUATORS)
        nA = len(self.specs)
        self.react = nn.Sequential(nn.Linear(N_CHANNELS, react_h), nn.SiLU(),
                                   nn.Linear(react_h, nA, bias=False))
        nn.init.normal_(self.react[-1].weight, std=0.05)
        z0 = [float(_inv_sigmoid_t(torch.tensor((sp.a0 - sp.lo) / (sp.hi - sp.lo))))
              for sp in self.specs]
        self.register_buffer("z0", torch.tensor(z0))
        if gain_core == "field":
            cfg = MHBPConfig(theta=1.0, coupling_topology="chain", allostasis=True)
            self.field = CoupledMultiscaleHBP(cfg).double()
            self.V = nn.ParameterList([
                nn.Parameter(0.01 * torch.randn(nA, 2 * f.cfg.d))
                for f in self.field.fields])
        elif gain_core == "gru":
            self.gcell = nn.GRUCell(N_CHANNELS, gru_h)
            self.Vg = nn.Parameter(0.01 * torch.randn(nA, gru_h))
        self._mod_off = False
        self._reg_sum = None
        self._reg_n = 0
        self._last_g = None                 # introspección/diagnóstico de varianza

    def modulation_off(self, off: bool = True):
        """Intervención causal: clampa g≡1 (¿la modulación es load-bearing?)."""
        self._mod_off = off

    def reset(self, B: int, device):
        self.to(device)
        self._mod_off = False               # el flag NO sobrevive entre episodios
        if self.gain_core == "field":
            self.field.reset_state(B, device=device)
        elif self.gain_core == "gru":
            dt = self.Vg.dtype
            self._h = torch.zeros(B, self.gcell.hidden_size, device=device, dtype=dt)
        self._reg_sum = torch.zeros((), device=device)
        self._reg_n = 0
        self._B = B

    def step(self, signals: dict) -> dict:
        B = self._B
        device = self.z0.device
        sig = InteroceptiveSignal(values=signals)
        s, m = sig.to_tensor(B, device, dtype=self.norm.mean.dtype)
        x = self.norm(s, m)
        z_react = self.react(x)                                     # (B, nA)
        if self.gain_core is not None:
            if self.gain_core == "field":
                self.field.tick(sig)
                self._reg_sum = self._reg_sum + self.field.regularizers()["allostasis"]
                self._reg_n += 1
                gz = 0.0
                for f, V in zip(self.field.fields, self.V):
                    ub, wb = f.node_mean()
                    gz = gz + torch.cat([ub, wb], -1).to(V.dtype) @ V.T
            else:                                                   # gru
                self._h = self.gcell(x, self._h)
                gz = self._h @ self.Vg.T
            g = 1.0 + self.GAMMA * torch.tanh(gz)
            if self._mod_off:
                g = torch.ones_like(g)
            self._last_g = g.detach()
            z = self.z0 + g * z_react
        else:
            z = self.z0 + z_react
        acts = {}
        for j, sp in enumerate(self.specs):
            a = sp.lo + (sp.hi - sp.lo) * torch.sigmoid(z[:, j])
            if sp.discrete:
                a = a + (a.round() - a).detach()                    # STE
            acts[sp.name] = a
        return acts

    def regularizer(self) -> torch.Tensor:
        if self.gain_core != "field" or self._reg_n == 0:
            return torch.zeros(())
        return self._reg_sum / self._reg_n

    def core_param_count(self) -> int:
        """Contabilidad UNIFICADA con MHBPController (mismas categorías: física
        + h* + acoplamiento + alostasis + cabezas de ganancia + reactivo).
        h* está muerto en todos (nota del panel: nunca recibe gradiente)."""
        n = sum(p.numel() for p in self.react.parameters())
        if self.gain_core == "field":
            n += sum(p.numel() for p in self.V)
            n += sum(p.numel() for f in self.field.fields
                     for p in f.params.parameters())
            n += sum(f.h_star.numel() for f in self.field.fields)
            if self.field.coupling is not None:
                n += sum(p.numel() for p in self.field.coupling.parameters())
            if self.field.allo.enabled:
                n += sum(p.numel() for p in self.field.allo.parameters())
        elif self.gain_core == "gru":
            n += sum(p.numel() for p in self.gcell.parameters()) + self.Vg.numel()
        return n


CONTROLLERS = ["mhbp", "mhbp_nocpl", "mhbp_noallo", "mhbp_taueq", "mhbp_first",
               "hbp_single", "gru", "mlp", "ar2", "react", "mhbp_gov", "gru_gov"]


def build_controller(name: str) -> nn.Module:
    if name == "react":
        return GovernorController(gain_core=None).double()
    if name == "mhbp_gov":
        return GovernorController(gain_core="field").double()
    if name == "gru_gov":
        return GovernorController(gain_core="gru").double()
    if name.startswith("mhbp") or name == "hbp_single":
        return MHBPController(name).double()
    return BaselineController(name).double()


def param_table() -> dict:
    out = {}
    for name in CONTROLLERS:
        c = build_controller(name)
        out[name] = {"total": sum(p.numel() for p in c.parameters()),
                     "core": c.core_param_count()}
    return out
