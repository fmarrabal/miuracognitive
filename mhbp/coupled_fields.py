"""
mHBP — CoupledMultiscaleHBP: el plano autonómico completo (Fase 1).

Mantiene los Q campos, el acoplamiento por potencial de interfaz (PSD por
construcción, §3), las escalas temporales ordenadas (§5), la alostasis (§6),
la interocepción y los actuadores. Un `tick()` = un paso global del integrador
Cayley-IMEX con el acoplamiento dentro del núcleo implícito.

Jerarquía: el acoplamiento es SIMÉTRICO (certificable); la direccionalidad
lento→rápido entra por la alostasis acotada. Ver MATH_SPEC §3.

ENDURECIDO tras la verificación adversarial (2026-07-26):
 - W de la interfaz NORMALIZADAS (Frobenius=1): la escala vive solo en κ ≤ κ_max
   → ‖𝓑‖ acotada por construcción (hallazgo crítico #2 del ataque numérico).
 - Timescales con logit fantasma: el init de 'learnable' reproduce taus_init
   EXACTAMENTE y τ_Q ya no queda clavada en τ_max.
 - Modo 'context' cableado de verdad (ctx_dim en config, ctx llega a _prepare).
 - Validaciones: len(taus)==Q, dt>0, ζ_min>0, ω_min>0.
 - Caché del integrador con detección de versión de parámetros: un opt.step()
   sin reset_state ya NO ejecuta física vieja en silencio.
 - pin_fp32() recorre TODOS los parámetros (lección del bug BF16).
 - save/load_dynamic_state(): contrato de checkpoint explícito (u, w, tick,
   estado de alostasis), porque u/w son buffers no persistentes.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn

from .field import FieldConfig, HomeostaticField
from .interoception import InteroceptionEncoder, InteroceptiveSignal
from .actuators import ActuatorHead
from .allostasis import Allostasis
from .integrators.cayley_imex import CayleyIMEX, pack, unpack, field_offsets
from .integrators.verlet import VerletRef


# --------------------------------------------------------------------------- #
# Escalas temporales: fijas | aprendibles con orden garantizado | contexto
# --------------------------------------------------------------------------- #
class Timescales(nn.Module):
    """τ_q con orden estricto por construcción (§5).

    learnable/context: softmax sobre Q+1 logits (el último es un GAP FANTASMA
    que absorbe la cola hasta 1, para que τ_Q no quede pegada a τ_max), cumsum
    de los Q primeros, separación mínima y mapeo afín a [τ_min, τ_max]:
        s_q = (cumsum(softmax(a))_q + g·q) / (1 + g·(Q+1)),  τ = τ_min + Δτ·s
    Estrictamente creciente y acotado PARA TODO a ∈ R^{Q+1}. El init invierte el
    mapa completo para reproducir taus_init exactamente (si es factible con la
    separación mínima g; si no, se recorta al punto factible más cercano)."""

    def __init__(self, taus_init=(1.0, 4.0, 8.0, 32.0), mode="fixed",
                 tau_min=0.5, tau_max=64.0, min_gap=0.005, ctx_dim: int = 0):
        super().__init__()
        self.mode = mode
        self.Q = len(taus_init)
        self.tau_min, self.tau_max, self.min_gap = tau_min, tau_max, min_gap
        self.register_buffer("taus_fixed", torch.tensor(taus_init, dtype=torch.float64))
        if mode in ("learnable", "context"):
            g, Q = min_gap, self.Q
            fr = (torch.tensor(taus_init, dtype=torch.float64) - tau_min) / (tau_max - tau_min)
            # inversión exacta del mapa: t_q = fr_q·(1+g(Q+1)) − g·q  debe ser
            # creciente y en (0,1); gaps = diff([0,t]) ∪ {1 − t_Q} (fantasma)
            t = fr * (1.0 + g * (Q + 1)) - g * torch.arange(1, Q + 1, dtype=torch.float64)
            gaps = torch.diff(torch.cat([torch.zeros(1, dtype=torch.float64), t]))
            gaps = torch.cat([gaps, (1.0 - t[-1]).unsqueeze(0)])
            if (gaps <= 0).any():                      # init infactible con este g
                gaps = gaps.clamp(min=1e-4)
            self.raw_a = nn.Parameter(torch.log(gaps).float())      # (Q+1,)
            if mode == "context":
                self.ctx_head = nn.Linear(max(ctx_dim, 1), self.Q + 1)
                nn.init.zeros_(self.ctx_head.weight)
                nn.init.zeros_(self.ctx_head.bias)
                self._ctx_dim = ctx_dim

    def _ordered_from(self, a: torch.Tensor) -> torch.Tensor:
        a = torch.nan_to_num(a.double(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30, 30)
        p = torch.softmax(a, dim=-1)                                # (Q+1,)
        s = torch.cumsum(p, dim=-1)[..., :self.Q]                   # (Q,) creciente
        q = torch.arange(1, self.Q + 1, device=a.device, dtype=s.dtype)
        s = (s + self.min_gap * q) / (1.0 + self.min_gap * (self.Q + 1))
        return self.tau_min + (self.tau_max - self.tau_min) * s

    def forward(self, ctx: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "fixed":
            return self.taus_fixed
        a = self.raw_a
        if self.mode == "context" and ctx is not None:
            if self._ctx_dim <= 0:
                raise RuntimeError("Timescales: modo context requiere ctx_dim > 0")
            z = self.ctx_head(ctx.to(self.ctx_head.weight.dtype)).mean(0)  # τ nominal por lote
            a = a + z
        return self._ordered_from(a)


# --------------------------------------------------------------------------- #
# Acoplamiento por potencial de interfaz (PSD y ACOTADO por construcción)
# --------------------------------------------------------------------------- #
class InterfaceCoupling(nn.Module):
    """E_cpl = ½ Σ_e κ_e ‖y_q − y_r‖², y_q = Ŵ_qᵀ ū_q. 𝓑 = Σ κ_e M_eᵀM_e ⪰ 0
    (Gram) para CUALESQUIERA parámetros — MATH_SPEC §3.

    Las W se usan NORMALIZADAS a ‖Ŵ‖_F = 1 (la dirección es aprendible; la
    ESCALA vive únicamente en κ ∈ (0, κ_max]) ⇒ ‖𝓑‖₂ ≤ Σ_e κ_e·(1/N_q + 1/N_r)
    acotada por construcción. Sin esta caja, ‖W‖ grande arruinaba el
    condicionamiento del núcleo implícito (hallazgo crítico del ataque)."""

    def __init__(self, field_cfgs, edges: list[tuple[int, int]],
                 d_c: int = 4, kappa_max: float = 0.5, kappa_init: float = 0.1):
        super().__init__()
        self.edges = list(edges)
        self.kappa_max = float(kappa_max)
        y0 = math.log(kappa_init / (kappa_max - kappa_init)) if kappa_max > 0 else 0.0
        self.raw_kappa = nn.Parameter(torch.full((len(edges),), float(y0)))
        self.W = nn.ParameterList([
            nn.Parameter(torch.randn(c.d, d_c) / (c.d ** 0.5)) for c in field_cfgs])
        self.d_c = d_c

    @property
    def kappa(self):
        return self.kappa_max * torch.sigmoid(self.raw_kappa)

    def W_hat(self, k: int, dtype=None) -> torch.Tensor:
        """W_k normalizada a Frobenius 1 (caja de la interfaz)."""
        W = self.W[k]
        Wn = W / (W.norm() + 1e-12)
        return Wn.to(dtype) if dtype is not None else Wn

    def _Me(self, e_idx, offsets, n_tot, cfgs, dtype, device):
        """Matriz extractora M_e (d_c, n_tot) de y_q − y_r para la arista e."""
        q, r = self.edges[e_idx]
        M = torch.zeros(self.d_c, n_tot, dtype=dtype, device=device)
        for sgn, k in ((1.0, q), (-1.0, r)):
            c = cfgs[k]
            Wk = self.W_hat(k, dtype)                               # (d, d_c)
            base = offsets[k]
            for i in range(c.n_nodes):
                cols = base + i * c.d + torch.arange(c.d, device=device)
                M[:, cols] += sgn * Wk.T / c.n_nodes
        return M

    def assemble_B(self, offsets, n_tot, cfgs, dtype, device):
        B = torch.zeros(n_tot, n_tot, dtype=dtype, device=device)
        kap = self.kappa.to(dtype)
        for e in range(len(self.edges)):
            M = self._Me(e, offsets, n_tot, cfgs, dtype, device)
            B = B + kap[e] * (M.T @ M)
        return B                                                    # ⪰ 0 (Gram)

    def energy(self, means: list[torch.Tensor]) -> torch.Tensor:
        """E_cpl por muestra (B,). means = [ū_q] con (B, d)."""
        ys = [m @ self.W_hat(k, m.dtype) for k, m in enumerate(means)]
        E = 0.0
        for kap, (q, r) in zip(self.kappa, self.edges):
            E = E + 0.5 * kap.to(ys[0].dtype) * ((ys[q] - ys[r]) ** 2).sum(-1)
        return E


# --------------------------------------------------------------------------- #
# Configuración del mHBP de 4 campos (MVP)
# --------------------------------------------------------------------------- #
def default_four_fields(d: int = 8) -> list[FieldConfig]:
    return [
        FieldConfig(name="fast_executive", n_nodes=8, d=d, tau_init=1.0),
        FieldConfig(name="risk_priority", n_nodes=4, d=d, tau_init=4.0),
        FieldConfig(name="slow_deliberative", n_nodes=6, d=d, tau_init=8.0),
        FieldConfig(name="resource_metabolic", n_nodes=4, d=d, tau_init=32.0),
    ]


@dataclass
class MHBPConfig:
    d: int = 8
    dt: float = 0.5
    theta: float = 1.0                       # 1=BE (disipativo), 0.5=CN (orden 2;
                                             # exige la condición anti-resonancia, §7)
    timescale_mode: str = "fixed"            # fixed | learnable | context
    taus: tuple = (1.0, 4.0, 8.0, 32.0)
    coupling_topology: str = "chain"         # chain | full | none  (sobre orden de τ)
    kappa_max: float = 0.5
    d_c: int = 4
    f_max: float = 0.5                       # cota del forzamiento interoceptivo
    allostasis: bool = True
    eps_allostasis: float = 0.2
    ctx_dim: int = 0                         # dimensión del contexto (modo context / alostasis)
    integrator: str = "cayley_imex"          # cayley_imex | verlet_ref
    dtype: str = "float64"
    leak_mask: tuple = ()                    # canales interoceptivos vetados
    # veto POR-CABEZA {índice_de_campo: (canales,)} — N2: el canal de valor
    # entra SOLO al campo risk (frontera «valorar sin tocar contenido»)
    head_leak_masks: tuple = ()              # ((q, (canal, ...)), ...)


class CoupledMultiscaleHBP(nn.Module):
    """El plano autonómico multiescala completo (Fase 1)."""

    def __init__(self, cfg: MHBPConfig | None = None, field_cfgs=None):
        super().__init__()
        self.cfg = cfg or MHBPConfig()
        fcs = field_cfgs or default_four_fields(self.cfg.d)
        # --- validaciones duras (hallazgos major del ataque) ---
        if len(self.cfg.taus) != len(fcs):
            raise ValueError(f"len(taus)={len(self.cfg.taus)} != nº de campos={len(fcs)}")
        if self.cfg.dt <= 0:
            raise ValueError("dt debe ser > 0")
        for c in fcs:
            if c.zeta_min <= 0 or c.omega_min <= 0:
                raise ValueError(f"campo {c.name}: se exige ζ_min>0 y ω_min>0 "
                                 "(cajas seguras de MATH_SPEC §2)")
        self.fields = nn.ModuleList([HomeostaticField(c) for c in fcs])
        self.Q = len(self.fields)
        self.timescales = Timescales(self.cfg.taus, mode=self.cfg.timescale_mode,
                                     ctx_dim=self.cfg.ctx_dim)
        edges = self._edges(self.cfg.coupling_topology)
        self.coupling = (InterfaceCoupling(fcs, edges, d_c=self.cfg.d_c,
                                           kappa_max=self.cfg.kappa_max)
                         if edges else None)
        self.intero = InteroceptionEncoder(
            fcs, f_max=self.cfg.f_max, leak_mask=list(self.cfg.leak_mask),
            head_vetoes={q: names for q, names in self.cfg.head_leak_masks})
        self.actuators = ActuatorHead(fcs)
        self.allo = Allostasis(fcs[0].d, fcs[0].n_nodes,
                               slow_dims=[c.d for c in fcs[1:]],
                               eps_a=self.cfg.eps_allostasis,
                               ctx_dim=self.cfg.ctx_dim,
                               enabled=self.cfg.allostasis)
        self._integ = None
        self._param_snapshot = None
        self._tick = 0
        self._B = None

    def _edges(self, topo: str):
        if topo == "none":
            return []
        if topo == "chain":                        # cadena en orden de escala temporal
            return [(q, q + 1) for q in range(self.Q - 1)]
        if topo == "full":
            return [(q, r) for q in range(self.Q) for r in range(q + 1, self.Q)]
        raise ValueError(topo)

    # ------------------------------------------------------------------ #
    @property
    def torch_dtype(self):
        return torch.float64 if self.cfg.dtype == "float64" else torch.float32

    def reset_state(self, batch: int, device=None):
        device = device or self.fields[0].h_star.device
        for f in self.fields:
            f.reset_state(batch, device=device, dtype=self.torch_dtype)
        self.allo.reset()                           # sin fugas entre episodios
        self.invalidate()
        self._tick = 0
        self._B = batch

    def invalidate(self):
        """Invalida el integrador cacheado (siguiente tick re-ensambla)."""
        self._integ = None
        self._param_snapshot = None

    def _versions(self):
        return tuple(p._version for p in self.parameters())

    def build_integrator(self, dtype=None, ctx: torch.Tensor | None = None,
                         device=None):
        """Construye y prepara un integrador NUEVO. SIN efectos laterales sobre
        el estado del modelo (lo usan también los certificados)."""
        taus = self.timescales(ctx)
        if self.cfg.integrator == "cayley_imex":
            integ = CayleyIMEX(theta=self.cfg.theta)
        elif self.cfg.integrator == "first_order":
            from .integrators.first_order import FirstOrderIMEX
            integ = FirstOrderIMEX()
        else:
            integ = VerletRef()
        integ.prepare(list(self.fields), taus, self.coupling, self.cfg.dt,
                      dtype=dtype or self.torch_dtype,
                      device=device or self.fields[0].h_star.device)
        return integ

    def _prepare(self, ctx: torch.Tensor | None = None):
        self._integ = self.build_integrator(ctx=ctx)
        self._param_snapshot = self._versions()
        return self._integ

    # ------------------------------------------------------------------ #
    def tick(self, signal: InteroceptiveSignal | None = None,
             ctx: torch.Tensor | None = None):
        """Un paso global del plano autonómico. Devuelve (acciones, info).
        Si algún parámetro cambió desde el último prepare (p.ej. opt.step()),
        se re-ensambla automáticamente (nada de física vieja en silencio)."""
        if self._B is None:
            raise RuntimeError("mHBP: llama a reset_state(batch) antes de tick()")
        needs_ctx_prepare = (self.cfg.timescale_mode == "context" and ctx is not None)
        if (self._integ is None or needs_ctx_prepare
                or self._param_snapshot != self._versions()):
            self._prepare(ctx)
        integ = self._integ
        B = self._B
        dev = self.fields[0].h_star.device
        dt64 = self.torch_dtype

        # forzamiento interoceptivo acotado por campo
        if signal is not None:
            F_list = self.intero(signal, B, dev, dtype=dt64)
        else:
            F_list = [torch.zeros(B, f.cfg.n_nodes, f.cfg.d, device=dev, dtype=dt64)
                      for f in self.fields]
        # alostasis: campos lentos desplazan el atractor del rápido (acotado, ISS)
        if self.allo.enabled:
            slow_means = [f.node_mean()[0] for f in self.fields[1:]]
            K_fast, _, _ = self.fields[0].operators(dtype=dt64)
            F_allo = self.allo(slow_means, K_fast, ctx)
            if F_allo is not None:
                F_list[0] = F_list[0] + F_allo

        U = pack([f.u for f in self.fields])
        W = pack([f.w for f in self.fields])
        F = pack(F_list)
        if not torch.isfinite(U).all() or not torch.isfinite(W).all():
            raise FloatingPointError("mHBP: estado no finito ANTES del tick")
        Up, Wp = integ.step(U, W, F)
        if not torch.isfinite(Up).all() or not torch.isfinite(Wp).all():
            raise FloatingPointError("mHBP: estado no finito TRAS el tick")
        for f, u, w in zip(self.fields, unpack(Up, self.fields), unpack(Wp, self.fields)):
            f.u, f.w = u, w
        self._tick += 1

        pooled = [f.node_mean() for f in self.fields]
        acts, info = self.actuators(pooled)
        info["tick"] = self._tick
        info["energy"] = integ.energy(Up, Wp)
        return acts, info

    # ------------------------------------------------------------------ #
    def energy(self) -> torch.Tensor:
        integ = self._integ or self._prepare()
        U = pack([f.u for f in self.fields])
        W = pack([f.w for f in self.fields])
        return integ.energy(U, W)

    def regularizers(self) -> dict[str, torch.Tensor]:
        return {"allostasis": self.allo.regularizer()}

    def diagnostics(self) -> dict:
        d = {f.name: f.diagnostics() for f in self.fields}
        d["taus"] = [float(t) for t in self.timescales().detach()]
        if self.coupling is not None:
            d["kappa"] = [float(k) for k in self.coupling.kappa.detach()]
        d["actuator_saturation"] = dict(self.actuators.saturation_frac)
        return d

    # ------------------- checkpoint del estado dinámico ------------------- #
    # u/w son buffers NO persistentes (dependen del batch): el state_dict NO los
    # incluye. Contrato explícito: usar estos helpers junto a state_dict().
    def save_dynamic_state(self) -> dict:
        return {
            "u": [f.u.detach().clone() for f in self.fields],
            "w": [f.w.detach().clone() for f in self.fields],
            "tick": self._tick,
            "allo": self.allo.save_state(),
        }

    def load_dynamic_state(self, st: dict):
        for f, u, w in zip(self.fields, st["u"], st["w"]):
            f.u, f.w = u.clone(), w.clone()
        self._B = st["u"][0].shape[0]
        self._tick = st["tick"]
        self.allo.load_state(st["allo"])
        self.invalidate()

    def pin_fp32(self):
        """Re-ancla TODOS los parámetros a FP32 tras un cast BF16/FP16 (lección
        del bug de congelación: no solo los físicos — también κ, W, cabezas)."""
        for p in self.parameters():
            p.data = p.data.float()
        return self
