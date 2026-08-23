"""
MiuraCognitive - Homeostatic Background Processor (HBP)
=======================================================
EL MÓDULO CENTRAL Y NOVEDOSO del proyecto.

Implementa un CAMPO HOMEOSTÁTICO definido sobre el grafo de módulos del
modelo, gobernado por una ECUACIÓN DE ONDA AMORTIGUADA ESPACIO-TEMPORAL
(Klein-Gordon amortiguada con forzamiento):

    ∂²h/∂t² + 2ζω₀ ∂h/∂t - c²∇²h + ω₀²(h - h*) = f_θ(h,s) + g_φ(h,x)

donde ∇² se discretiza como el LAPLACIANO DEL GRAFO sobre los nodos
(módulos) de la arquitectura. El "tiempo" t son las ITERACIONES del reasoner
(profundidad adaptativa): el VEI da UN tick por iteración de pensamiento.

NOVEDADES DE ESTA VERSIÓN (Sprint 2 - rediseño):
  - VEI BATCHED: estado (B, N, d_h). La modulación depende del input.
  - SWITCH ARQUITECTÓNICO de orden:
        order=2 -> Verlet completo (inercia/oscilación, "hbp_full").
        order=1 -> LÍMITE SOBREAMORTIGUADO de la MISMA ecuación (Euler de
                   relajación, sin término inercial, "hbp_first").
  - ζ aprendible con SUELO ζ_min; ω₀, c en rangos seguros (squash suave).
  - g_φ ACOTADO (RMSNorm del input + tanh + ganancia) = único forzamiento
    no acotado de antes; ahora seguro (BIBS).
  - Cómputo de Verlet/Laplaciano en FP32 (la cancelación de velocidad
    h_t - h_{t-1} pierde decenas de % en BF16 cerca del equilibrio).
  - Clamp del VEI alrededor de h*; h* inicializado != 0 (rompe el punto fijo).
  - Certificado de estabilidad DISCRETO ACOPLADO (Jury/Schur-Cohn):
        0 < ζω₀Δt < 1   ∧   Δt²(ω₀² + c²ρ(L)) < 4(1 - ζω₀Δt)
    expuesto como penalización soft para el entrenamiento.

Analogía RMN: ζ ~ 1/T2, ω₀ ~ precesión, c²∇² ~ acoplamiento dipolar;
los modos normales de L son el espectro del sistema.

Autor: Francisco M. Arrabal (Curro) + asistencia Claude
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HBPConfig:
    """Configuración del HBP."""
    n_nodes: int = 6           # 4 backbone + reasoner + working memory
    d_f: int = 16              # sub-espacio fisiológico
    d_e: int = 16              # sub-espacio emocional
    d_u: int = 32              # sub-espacio de umbrales
    d_intero: int = 4          # interocepción: progreso, activación, esfuerzo, entropía_atención
    dt: float = 1.0            # paso temporal (1 tick = 1 iteración del reasoner)
    order: int = 2             # 2 = Verlet (oscilatorio); 1 = límite sobreamortiguado
    # Rango/inicialización de los parámetros físicos (aprendibles vía squash)
    omega0_init: float = 0.5
    omega0_min: float = 0.2
    omega0_max: float = 1.8
    zeta_init: float = 0.5
    zeta_min: float = 0.05     # suelo de amortiguamiento (se sube p/ hbp_first)
    c_init: float = 0.4
    c_max: float = 0.7         # tope de acoplamiento (se baja p/ hbp_first)
    f_gain_init: float = 0.05
    f_gain_max: float = 0.3    # cota dura del auto-chequeo (forzamiento BIBS)
    g_phi_gain: float = 0.3    # ganancia del forzamiento externo (acotado por tanh)
    h_clamp: float = 5.0       # clamp del VEI alrededor de h*
    alpha_router: float = 1.0
    # --- Familia de operadores PDE (difusión ↔ onda ↔ advección) ---
    # Coeficientes INDEPENDIENTES, opcionalmente elegidos por la interocepción
    # ("la homeostasis elige su física"). Con D_max=b_max=0 y gate_physics=False,
    # el HBP se reduce exactamente a la onda amortiguada (retrocompatible).
    D_max: float = 0.0         # tope de DIFUSIÓN estructural (-D·L·ḣ, amortiguamiento espacial)
    b_adv_max: float = 0.0     # tope de ADVECCIÓN (-b·A_dir·h, transporte direccional antisimétrico)
    gate_physics: bool = False # si True, α (onda↔difusión), D y b los elige la interocepción por tick
    gate_init_scale: float = 0.1  # sensibilidad inicial de las cabezas de física (0.1=tímido→casi cte;
                                  # 1.0=sensible→puede conmutar física por input desde el arranque)
    alpha_const: float | None = None  # fuerza α constante (1=onda pura, 0=difusión pura, mezcla fija);
                                      # para experimentos de oráculo/scan. Ignorado si gate_physics.
    mix_certificate: bool = False     # activa el certificado de la rama difusiva aunque no haya gating
                                      # (necesario si α se fuerza por instancia vía alpha_force)
    # --- Solver de la rama difusiva ---
    # "euler": h_t + dt·rhs/(2ζω₀) — explícito, ACOPLADO al ζ compartido (con ζ bajo
    #          para oscilar en la rama onda, el multiplicador de Euler explota).
    # "implicit": backward-Euler IMEX con tasa PROPIA γ_diff aprendible —
    #          incondicionalmente estable a cualquier ζ (desacopla α de ζ).
    diff_solver: str = "euler"
    gamma_diff_init: float = 0.5      # tasa inicial (= 2ζ₀ω₀₀ -> transición suave desde euler)
    gamma_diff_min: float = 0.1
    gamma_diff_max: float = 4.0
    # NÚCLEO del integrador (benchmark v4, baseline pedido por revisores):
    # "pde" = la familia física de siempre; "gru" = un GRUCell por nodo
    # (pesos compartidos) que sustituye SOLO al integrador — proyecciones
    # de interocepción, forzamiento externo y cabezas de modulación quedan
    # IDÉNTICAS (paridad de interfaz por construcción).
    core: str = "pde"
    # --- Operadores KdV (zoo de homeostasis; propuesta de Curro) ---
    # DISPERSIÓN β·A_dir³·u (u = h-h*): tercera derivada espacial en el grafo
    # orientado; A³ es antisimétrica ((iμ)³=-iμ³) -> conservativa EN LA RAMA DE
    # 1er ORDEN (posicional); en la rama de 2º orden entra GIROSCÓPICA (Lema 1).
    # ADVECCIÓN NO LINEAL ν·tanh(u)⊙tanh(A_dir·u): el u·u_x de KdV SATURADO —
    # jacobiano nulo en el equilibrio (no altera la linealización ni los
    # certificados) y acotado globalmente por ν (Lema 1(d)).
    # Sin ghost de Ostrogradsky (1er orden en t). Caveat honesto: a N=6 la
    # dispersión tiene solo 3 pares de modos (sin régimen solitónico); el eje de
    # test es el CONTROLADOR de cómputo, no la accuracy.
    kdv_beta_max: float = 0.0         # tope del coeficiente de dispersión β
    kdv_nl_max: float = 0.0           # tope de la advección no lineal ν
    # --- Acoplamiento ELÍPTICO NO-LOCAL (propuesta de Curro, vorticidad↔stream) ---
    # h juega el papel de vorticidad ω; la "función de corriente" ψ se esclaviza
    # por Poisson en el grafo: ∇²ψ = −ω  ⟹  ψ = L⁺(h−h*), con L⁺ la pseudo-inversa
    # del laplaciano (proyecta fuera el modo constante). Con elliptic_readout, la
    # MODULACIÓN top-down lee ψ en vez de h: cada nodo modula según TODO el campo
    # (no-localidad instantánea, tipo campo dipolar/desmagnetizante en RMN). La
    # DINÁMICA del VEI y el CERTIFICADO no cambian (solo la lectura) -> A/B de una
    # sola variable frente a hbp_full. elliptic_gain mezcla local↔no-local.
    elliptic_readout: bool = False
    elliptic_gain: float = 1.0        # 1.0 = solo ψ; <1 = blend con h local

    @property
    def d_h(self) -> int:
        return self.d_f + self.d_e + self.d_u


def build_chain_laplacian(n_nodes: int) -> torch.Tensor:
    """Laplaciano combinatorio L = D - A de un grafo en cadena 0-1-...-(n-1).
    Actúa como discretización de -∇² sobre el grafo. (n_nodes, n_nodes)."""
    A = torch.zeros(n_nodes, n_nodes)
    for i in range(n_nodes - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    D = torch.diag(A.sum(dim=1))
    return D - A


def build_chain_advection(n_nodes: int) -> torch.Tensor:
    """Matriz de advección ANTISIMÉTRICA de la cadena orientada 0->1->...->(n-1)
    (dirección semántica del grafo: backbone -> reasoner -> WM). A_dir[i,i+1]=+1,
    A_dir[i+1,i]=-1. Al ser antisimétrica, el término -b·A_dir·h TRANSPORTA el
    campo direccionalmente sin inyectar energía (autovalores imaginarios puros ->
    conservativo). Es el operador espacial de orden IMPAR bien definido en grafos."""
    A = torch.zeros(n_nodes, n_nodes)
    for i in range(n_nodes - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = -1.0
    return A


def _inv_sigmoid(y: float) -> float:
    """logit, para inicializar un parámetro squasheado por sigmoid en y."""
    y = min(max(y, 1e-4), 1 - 1e-4)
    return math.log(y / (1.0 - y))


def _inv_softplus(y: float) -> float:
    """Inversa de softplus, para inicializar zeta = zeta_min + softplus(raw)."""
    return math.log(math.expm1(max(y, 1e-4)))


class HomeostaticBackgroundProcessor(nn.Module):
    """El HBP completo. Mantiene el VEI batcheado (B, N, d_h) y lo evoluciona
    con la ecuación de onda amortiguada en cada tick (= iteración del reasoner)."""

    def __init__(self, cfg: HBPConfig):
        super().__init__()
        self.cfg = cfg
        N, d_h = cfg.n_nodes, cfg.d_h

        # --- Estado de reposo h* (aprendible), != 0 para romper el punto fijo ---
        self.h_star = nn.Parameter(0.02 * torch.randn(N, d_h))

        # --- Parámetros físicos aprendibles (raw -> squash a rango seguro) ---
        # ω₀ ∈ (ω₀_min, ω₀_max) vía sigmoid; ζ ≥ ζ_min vía softplus; c ∈ (0, c_max) vía sigmoid.
        w_span = cfg.omega0_max - cfg.omega0_min
        self.raw_omega0 = nn.Parameter(torch.full((d_h,), _inv_sigmoid((cfg.omega0_init - cfg.omega0_min) / w_span)))
        self.raw_zeta = nn.Parameter(torch.full((d_h,), _inv_softplus(cfg.zeta_init - cfg.zeta_min)))
        self.raw_c = nn.Parameter(torch.tensor(_inv_sigmoid(cfg.c_init / cfg.c_max)))
        if cfg.core == "gru":
            # núcleo GRU (v4): entrada = [interocepción proyectada, forzamiento
            # externo] por nodo; celda compartida entre nodos, estado = VEI.
            self.gru_cell = nn.GRUCell(cfg.d_intero + d_h, d_h)
        self.raw_f_gain = nn.Parameter(torch.tensor(_inv_sigmoid(cfg.f_gain_init / cfg.f_gain_max)))
        if cfg.diff_solver == "implicit":
            g_span = cfg.gamma_diff_max - cfg.gamma_diff_min
            self.raw_gamma_diff = nn.Parameter(torch.full(
                (d_h,), _inv_sigmoid((cfg.gamma_diff_init - cfg.gamma_diff_min) / g_span)))

        # --- Laplaciano del grafo y su radio espectral ρ(L) (para el certificado) ---
        L = build_chain_laplacian(N)
        self.register_buffer("laplacian", L)
        self.register_buffer("rho_L", torch.linalg.eigvalsh(L).max())
        # Pseudo-inversa del laplaciano (esclavización elíptica ψ=L⁺ω). Global y
        # simétrica; proyecta fuera el modo constante (nullspace de L). N pequeño
        # -> se precomputa una vez. Solo se usa en la lectura si elliptic_readout.
        self.register_buffer("L_pinv", torch.linalg.pinv(L))
        # --- Advección antisimétrica del grafo orientado y su radio espectral ---
        A_dir = build_chain_advection(N)
        self.register_buffer("adv", A_dir)
        self.register_buffer("rho_A", torch.linalg.svdvals(A_dir).max())
        # Espectro exacto de A (autovalores ±iμ_k): ρ(G)=max_k|b·μ_k − β·μ_k³| para
        # el certificado discreto de Schur-Cohn complejo (Prop. 2 del paper).
        self.register_buffer("mu_A", torch.linalg.eigvals(A_dir).imag.abs())
        # --- Operadores KdV: dispersión A³ (antisimétrica) y su radio ρ(A)³ ---
        self._has_kdv = cfg.kdv_beta_max > 0.0 or cfg.kdv_nl_max > 0.0
        if self._has_kdv:
            A3 = A_dir @ A_dir @ A_dir
            self.register_buffer("adv3", A3)
            self.register_buffer("rho_A3", torch.linalg.svdvals(A3).max())
            if cfg.kdv_beta_max > 0.0:
                self.raw_kdv_beta = nn.Parameter(torch.full((d_h,), _inv_sigmoid(0.5)))
            if cfg.kdv_nl_max > 0.0:
                self.raw_kdv_nu = nn.Parameter(torch.full((d_h,), _inv_sigmoid(0.5)))

        # --- Coeficientes de la familia PDE (difusión D, advección b, mezcla α) ---
        # Si gate_physics: los elige la interocepción por tick (cabezas con squash).
        # Si no: son estáticos aprendibles (si sus topes > 0) o nulos (retrocompat).
        self._has_D = cfg.D_max > 0.0
        self._has_b = cfg.b_adv_max > 0.0
        if cfg.gate_physics:
            # La interocepción s_n llega con magnitud diminuta (~std_init·√d): sin
            # normalizar, w·s_n ≪ bias y las cabezas quedan clavadas en su sesgo
            # (física inerte, indep. del input). Igual que g_φ, se normaliza a O(1).
            self.phys_norm = nn.LayerNorm(cfg.d_intero)
            self.alpha_head = nn.Linear(cfg.d_intero, d_h)   # mezcla onda(1)↔difusión(0)
            self.D_head = nn.Linear(cfg.d_intero, d_h)       # difusión estructural
            self.b_head = nn.Linear(cfg.d_intero, d_h)       # advección
        else:
            if self._has_D:
                self.raw_D = nn.Parameter(torch.full((d_h,), _inv_sigmoid(0.2)))
            if self._has_b:
                self.raw_b = nn.Parameter(torch.full((d_h,), _inv_sigmoid(0.2)))

        # --- Auto-chequeo f_θ([h ; s]) -> corrección acotada por tanh ---
        self.f_theta = nn.Sequential(
            nn.Linear(d_h + cfg.d_intero, 2 * d_h),
            nn.SiLU(),
            nn.Linear(2 * d_h, d_h),
            nn.Tanh(),
        )
        # --- Forzamiento externo g_φ (input -> VEI), acotado: RMSNorm + tanh + ganancia ---
        self.g_phi_norm = nn.LayerNorm(d_h)
        self.g_phi_proj = nn.Linear(d_h, d_h)

        # --- Cabezas de MODULACIÓN top-down ---
        self.W_tau = nn.Linear(cfg.d_u, 1)          # umbral de halting <- h^umb
        self.W_wm_write = nn.Linear(cfg.d_e, d_h)   # gate escritura WM <- h^emo
        self.W_wm_forget = nn.Linear(cfg.d_e, d_h)  # gate olvido WM <- h^emo
        self.w_router = nn.Linear(cfg.d_e, 1)       # bias router S1/S2 <- h^emo
        self.W_intero_pred = nn.Linear(cfg.d_f, cfg.d_intero)  # predictor interoceptivo (L_interoc)

        # Estado dinámico batcheado (no parámetros). reset_state lo inicializa.
        self.register_buffer("h_t", torch.zeros(1, N, d_h), persistent=False)
        self.register_buffer("h_tm1", torch.zeros(1, N, d_h), persistent=False)
        self._initialized = False
        self.init_physics()

    def init_physics(self):
        """Init de f_θ casi inactivo. Se vuelve a llamar tras el _init_weights
        global de MiuraCognitiveFull (que si no sobrescribiría f_θ)."""
        with torch.no_grad():
            self.f_theta[-2].weight.mul_(0.05)
            self.f_theta[-2].bias.zero_()
            # g_φ arranca pequeño para no patear de más en el sembrado.
            self.g_phi_proj.weight.mul_(0.1)
            self.g_phi_proj.bias.zero_()
            # Cabezas de física gateada: arrancan MOSTLY-ONDA con difusión/advección
            # pequeñas (cerca de la onda amortiguada base), y aprenden a mezclar.
            if self.cfg.gate_physics:
                for head, y0 in ((self.alpha_head, 0.8), (self.D_head, 0.15), (self.b_head, 0.15)):
                    head.weight.mul_(self.cfg.gate_init_scale)
                    head.bias.fill_(_inv_sigmoid(y0))

    # --------------------------- parámetros físicos --------------------------- #
    @property
    def omega0(self) -> torch.Tensor:
        c = self.cfg
        return c.omega0_min + (c.omega0_max - c.omega0_min) * torch.sigmoid(self.raw_omega0)

    @property
    def zeta(self) -> torch.Tensor:
        return self.cfg.zeta_min + F.softplus(self.raw_zeta)

    @property
    def c(self) -> torch.Tensor:
        return self.cfg.c_max * torch.sigmoid(self.raw_c)

    @property
    def gamma_diff(self) -> torch.Tensor:
        cf = self.cfg
        return cf.gamma_diff_min + (cf.gamma_diff_max - cf.gamma_diff_min) * torch.sigmoid(self.raw_gamma_diff)

    # FIX BF16 (bug de congelación): los raw físicos tienen |valor| ~0.3-1.6; en
    # BF16 su ULP/2 (~5e-4 a 4e-3) SUPERA el paso típico de Adam (lr~3e-4), así
    # que param - step redondea a param EXACTO: quedaban congelados en su init en
    # todos los runs GPU. El cómputo del campo ya va en FP32; los raw deben
    # almacenarse también en FP32. Llamar SIEMPRE tras model.to(device, bf16).
    _FP32_PARAMS = ("raw_omega0", "raw_zeta", "raw_c", "raw_f_gain",
                    "raw_D", "raw_b", "raw_gamma_diff", "h_star",
                    "raw_kdv_beta", "raw_kdv_nu")

    def pin_fp32(self):
        """Re-ancla los parámetros físicos escalares a FP32 (tras un cast BF16)."""
        for name in self._FP32_PARAMS:
            p = getattr(self, name, None)
            if p is not None:
                p.data = p.data.float()
        return self

    @property
    def f_gain(self) -> torch.Tensor:
        return self.cfg.f_gain_max * torch.sigmoid(self.raw_f_gain)

    def _physics_coeffs(self, intero: torch.Tensor, alpha_force: torch.Tensor | None = None):
        """Coeficientes de la familia PDE. Gateados por la interocepción si
        gate_physics (la homeostasis elige su física por tick); si no, α lo fija
        el orden (o alpha_const) y D/b son estáticos aprendibles (o nulos).

        alpha_force: (B,) opcional — α impuesto POR INSTANCIA (oráculo: el
        experimentador dicta la física según el modo de la tarea)."""
        cfg = self.cfg
        if cfg.gate_physics:
            s = self.phys_norm(intero)                                      # O(1), input-dependiente
            alpha = torch.sigmoid(self.alpha_head(s)).float()               # (B,N,d_h) onda↔difusión
            Dcoef = (cfg.D_max * torch.sigmoid(self.D_head(s))).float()
            bcoef = (cfg.b_adv_max * torch.sigmoid(self.b_head(s))).float()
        else:
            if cfg.alpha_const is not None:
                alpha = float(cfg.alpha_const)
            else:
                alpha = 1.0 if cfg.order == 2 else 0.0
            Dcoef = (cfg.D_max * torch.sigmoid(self.raw_D)).float() if self._has_D else 0.0
            bcoef = (cfg.b_adv_max * torch.sigmoid(self.raw_b)).float() if self._has_b else 0.0
        if alpha_force is not None:
            alpha = alpha_force.view(-1, 1, 1).float()                      # (B,1,1), pisa lo anterior
        return alpha, Dcoef, bcoef

    def physics_state(self) -> dict[str, float]:
        """Régimen físico actual (medias): α (1=onda, 0=difusión), D (difusión),
        b (advección). Para introspección/figuras."""
        return {"alpha": getattr(self, "_last_alpha", 1.0 if self.cfg.order == 2 else 0.0),
                "D": getattr(self, "_last_D", 0.0), "b": getattr(self, "_last_b", 0.0)}

    def reset_state(self, batch_size: int = 1, device=None, dtype=None):
        """Reinicia el VEI al reposo h* con velocidad nula. El sembrado con
        g_φ(input) (en miura.py) introduce luego v != 0 en el primer tick."""
        device = device or self.h_star.device
        dtype = dtype or self.h_star.dtype
        h0 = self.h_star.detach().to(device=device, dtype=dtype)
        h0 = h0.unsqueeze(0).expand(batch_size, -1, -1).clone()
        self.h_t = h0
        self.h_tm1 = h0.clone()
        self._initialized = True

    # --------------------------- dinámica: un tick --------------------------- #
    def step(self, intero: torch.Tensor, ext_force: torch.Tensor | None = None,
             alpha_force: torch.Tensor | None = None,
             active_idx: torch.Tensor | None = None) -> torch.Tensor:
        """Avanza el VEI un tick. intero: (B,N,d_intero). ext_force: (B,N,d_h)|None.
        alpha_force: (B,)|None — α impuesto por instancia (experimentos de oráculo).
        active_idx: índices opcionales dentro del batch de estado completo. Si se
        proporciona, solo esas muestras avanzan y las restantes conservan estado.
        El cómputo del campo va en FP32 (estabilidad numérica de la velocidad)."""
        if active_idx is not None:
            if active_idx.ndim != 1:
                raise ValueError("active_idx debe tener forma (B_activo,)")
            active_idx = active_idx.to(device=self.h_t.device, dtype=torch.long)
            if intero.shape[0] != active_idx.numel():
                raise ValueError("intero y active_idx no tienen el mismo batch")

            # Ejecuta la dinámica exacta sobre el sub-batch y dispersa el nuevo
            # estado. Las filas que agotaron su cuota no se computan ni cambian.
            h_t_full, h_tm1_full = self.h_t, self.h_tm1
            self.h_t = h_t_full.index_select(0, active_idx)
            self.h_tm1 = h_tm1_full.index_select(0, active_idx)
            try:
                h_active = self.step(intero, ext_force=ext_force,
                                     alpha_force=alpha_force, active_idx=None)
                h_tm1_active = self.h_tm1
            except Exception:
                self.h_t, self.h_tm1 = h_t_full, h_tm1_full
                raise
            self.h_t = h_t_full.index_copy(0, active_idx, h_active)
            self.h_tm1 = h_tm1_full.index_copy(0, active_idx, h_tm1_active)
            return h_active

        if not self._initialized:
            self.reset_state(batch_size=intero.shape[0], device=intero.device, dtype=intero.dtype)

        cfg = self.cfg
        if cfg.core == "gru":
            # Núcleo GRU (v4): mismo contrato que la física — recibe la MISMA
            # interocepción y forzamiento, escribe el MISMO estado (B,N,d_h);
            # el resto del módulo (modulación, L_interoc) no distingue núcleos.
            if ext_force is None:
                ext_force = torch.zeros(*intero.shape[:2], self.cfg.d_h,
                                        device=intero.device,
                                        dtype=intero.dtype)
            inp = torch.cat([intero, ext_force], dim=-1)
            Bg, Ng, Din = inp.shape
            wdt = self.gru_cell.weight_ih.dtype
            h_new = self.gru_cell(
                inp.reshape(Bg * Ng, Din).to(wdt),
                self.h_t.reshape(Bg * Ng, -1).to(wdt)).reshape(Bg, Ng, -1)
            hs_g = self.h_star.to(wdt).unsqueeze(0)
            h_new = hs_g + torch.clamp(h_new - hs_g, -cfg.h_clamp,
                                       cfg.h_clamp)
            h_new = h_new.to(self.h_t.dtype)
            self.h_tm1 = self.h_t
            self.h_t = h_new
            return h_new
        dt = cfg.dt
        # FP32 para el campo (cancelación de velocidad en BF16)
        ht = self.h_t.float()
        htm1 = self.h_tm1.float()
        hs = self.h_star.float().unsqueeze(0)                 # (1,N,d_h)
        om = self.omega0.float()                              # (d_h,)
        ze = self.zeta.float()
        cc = self.c.float()
        Lap = self.laplacian.float()

        # --- Coeficientes de la familia PDE (α onda↔difusión, D difusión, b advección) ---
        alpha, Dcoef, bcoef = self._physics_coeffs(intero, alpha_force)

        # --- Términos del lado derecho (RHS) ---
        restit = -(om ** 2) * (ht - hs)                                  # reacción/restitución
        spatial = -(cc ** 2) * torch.einsum("ij,bjd->bid", Lap, ht)      # onda/elástico: c²∇²h
        f_in = torch.cat([self.h_t, intero], dim=-1)
        autocheck = (self.f_gain * self.f_theta(f_in)).float()
        forcing = ((cfg.g_phi_gain * torch.tanh(self.g_phi_proj(self.g_phi_norm(ext_force)))).float()
                   if ext_force is not None else torch.zeros_like(ht))
        # --- Operadores ANTISIMÉTRICOS (advección b·A, dispersión β·A³) ---
        # COLOCACIÓN CORRECTA POR ORDEN (mecánica clásica, flutter de Ziegler):
        #  · rama de 1er ORDEN (difusión): actúan POSICIONALMENTE (-G·u), como en
        #    KdV/advección estándar; ahí son conservativos (autovalores ±iμ).
        #  · rama de 2º ORDEN (onda): deben actuar GIROSCÓPICAMENTE (-G·ḣ): no
        #    hacen trabajo (vᵀGv=0 por antisimetría) y la energía sigue siendo
        #    Lyapunov. En posición serían fuerzas CIRCULATORIAS -> flutter
        #    (λ²=±iμ da Re(λ)>0 incluso con amortiguamiento moderado).
        v = ht - htm1
        u = ht - hs
        anti_pos, gyro, nladv = 0.0, 0.0, 0.0
        if self._has_b or cfg.gate_physics:                             # advección b·A_dir
            # Posicional SOBRE u (no h): h* es el equilibrio exacto de la rama difusiva.
            anti_pos = anti_pos - bcoef * torch.einsum("ij,bjd->bid", self.adv.float(), u)
            gyro = gyro - (bcoef / dt) * torch.einsum("ij,bjd->bid", self.adv.float(), v)
        if self._has_kdv:
            if cfg.kdv_beta_max > 0.0:                                  # dispersión β·A³
                bet = (cfg.kdv_beta_max * torch.sigmoid(self.raw_kdv_beta)).float()
                anti_pos = anti_pos - bet * torch.einsum("ij,bjd->bid", self.adv3.float(), u)
                gyro = gyro - (bet / dt) * torch.einsum("ij,bjd->bid", self.adv3.float(), v)
            if cfg.kdv_nl_max > 0.0:
                # u·u_x SATURADA: -ν·tanh(u)⊙tanh(A·u). Jacobiano nulo en u=0 (mismo
                # orden cuadrático de KdV cerca del equilibrio) y acotada GLOBALMENTE
                # por ν (Lema 1(d); la versión sin saturar crece linealmente y su
                # acotación era solo condicional al clamp).
                nu = (cfg.kdv_nl_max * torch.sigmoid(self.raw_kdv_nu)).float()
                nladv = -nu * torch.tanh(u) * torch.tanh(
                    torch.einsum("ij,bjd->bid", self.adv.float(), u))
        # Términos comunes (simétricos/acotados); los antisimétricos se añaden
        # POR RAMA con su colocación correcta.
        rhs = restit + spatial + nladv + autocheck + forcing

        # Amortiguamiento: uniforme (ζ) + DIFUSIÓN estructural (-D·L·ḣ, independiente de c)
        damp = -(2.0 * ze * om / dt) * v
        if self._has_D or cfg.gate_physics:
            damp = damp - (Dcoef / dt) * torch.einsum("ij,bjd->bid", Lap, v)

        # Rama ONDA (Verlet, 2º orden): antisimétricos GIROSCÓPICOS (en ḣ)
        h_wave = 2.0 * ht - htm1 + (dt ** 2) * (rhs + damp + gyro)
        # Rama DIFUSIÓN (1er orden): antisimétricos POSICIONALES (en u)
        if cfg.diff_solver == "implicit":
            # IMEX backward-Euler con tasa PROPIA γ_diff, formulado sobre u = h-h*:
            #   (I + λ·(K + G))·u' = u + λ·F_acotado,   λ = dt/γ,  K = ω₀²I + c²L.
            # La parte antisimétrica ESTÁTICA G va DENTRO del núcleo (Prop. 3):
            # Re<x, M_G x> >= (1+λω₀²)||x||²  =>  ||M_G⁻¹|| <= (1+λω₀²)⁻¹ < 1
            # INCONDICIONALMENTE (sin hipótesis de conmutación [L,A]). Con G explícita
            # esto sería FALSO (contraejemplos del panel); si los coeficientes son
            # GATEADOS (por instancia) G no puede plegarse al núcleo por-dim y queda
            # explícita: la cubre el multiplicador de Euler complejo del certificado.
            gam = self.gamma_diff.float()                              # (d_h,)
            lam = dt / gam                                             # (d_h,)
            Fexp = ((nladv if torch.is_tensor(nladv) else 0.0)
                    + autocheck + forcing)
            I6 = torch.eye(Lap.shape[0], device=ht.device, dtype=torch.float32)
            Kop = ((om ** 2).view(-1, 1, 1) * I6.unsqueeze(0)
                   + (cc ** 2).view(-1, 1, 1) * Lap.unsqueeze(0))      # (d_h,N,N)
            if cfg.gate_physics:
                Fexp = Fexp + (anti_pos if torch.is_tensor(anti_pos) else 0.0)
            else:
                if self._has_b:
                    b_st = (cfg.b_adv_max * torch.sigmoid(self.raw_b)).float()
                    Kop = Kop + b_st.view(-1, 1, 1) * self.adv.float().unsqueeze(0)
                if self._has_kdv and cfg.kdv_beta_max > 0.0:
                    be_st = (cfg.kdv_beta_max * torch.sigmoid(self.raw_kdv_beta)).float()
                    Kop = Kop + be_st.view(-1, 1, 1) * self.adv3.float().unsqueeze(0)
            M = I6.unsqueeze(0) + lam.view(-1, 1, 1) * Kop             # (d_h,N,N)
            V = ((ht - hs) + lam * Fexp).permute(2, 1, 0)              # (d_h,N,B)
            h_diff = hs + torch.linalg.solve(M, V).permute(2, 1, 0)    # (B,N,d_h)
        else:
            h_diff = ht + dt * (rhs + anti_pos) / (2.0 * ze * om)
        # Mezcla: la homeostasis elige cuánto pesa cada régimen (α=1 onda, α=0 difusión).
        # Extremos exactos por rama (retrocompat bitwise con order=2/order=1 puros).
        if isinstance(alpha, float):
            if alpha >= 1.0:
                h_next = h_wave
            elif alpha <= 0.0:
                h_next = h_diff
            else:
                h_next = alpha * h_wave + (1.0 - alpha) * h_diff
        else:
            h_next = alpha * h_wave + (1.0 - alpha) * h_diff

        # Registro para introspección (medias escalares de los coeficientes)
        with torch.no_grad():
            self._last_alpha = float(alpha) if isinstance(alpha, float) else float(alpha.mean())
            self._last_D = float(Dcoef.mean()) if torch.is_tensor(Dcoef) else float(Dcoef)
            self._last_b = float(bcoef.mean()) if torch.is_tensor(bcoef) else float(bcoef)
            # Tensor α completo (B,N,d_h) del último tick, para diagnóstico de dispersión
            # por-dim (¿la media plana esconde variación por dimensión/nodo?).
            self._last_alpha_full = alpha.detach() if torch.is_tensor(alpha) else None

        # Clamp del VEI alrededor de h* (anti-saturación / anti-excursión BF16)
        h_next = hs + torch.clamp(h_next - hs, -cfg.h_clamp, cfg.h_clamp)
        h_next = h_next.to(self.h_t.dtype)

        # Rota la ventana temporal (ATTACHED dentro del rollout; el detach del
        # carry entre batches lo hace miura.py / el reset por batch).
        self.h_tm1 = self.h_t
        self.h_t = h_next
        return h_next

    # ------------------------- certificado de estabilidad ------------------- #
    def _anti_coeffs_cert(self):
        """(b_d, β_d, D_w) por dimensión para el certificado: valores APRENDIDOS si
        son estáticos (penalización diferenciable), topes de config si gateados
        (peor caso por instancia)."""
        cfg = self.cfg
        dev = self.raw_omega0.device
        zeros = torch.zeros(cfg.d_h, device=dev)
        if cfg.gate_physics:
            b_d = torch.full((cfg.d_h,), cfg.b_adv_max, device=dev)
            D_w = torch.full((cfg.d_h,), cfg.D_max, device=dev)
        else:
            b_d = cfg.b_adv_max * torch.sigmoid(self.raw_b) if self._has_b else zeros
            D_w = cfg.D_max * torch.sigmoid(self.raw_D) if self._has_D else zeros
        beta_d = (cfg.kdv_beta_max * torch.sigmoid(self.raw_kdv_beta)
                  if (self._has_kdv and cfg.kdv_beta_max > 0.0) else zeros)
        return b_d, beta_d, D_w

    def _euler_complex_margin(self, rho_G: torch.Tensor) -> torch.Tensor:
        """Rama difusiva con Euler EXPLÍCITO y parte antisimétrica: el multiplicador
        es COMPLEJO, |1 - a(k ± i·ρG)|² = (1-ak)² + (a·ρG)² con a = Δt/(2ζω₀).
        Condición real (peor apareamiento): Δt·(k² + ρG²)/(2ζω₀·k) < 2 (holgura 1.9),
        sobre k ∈ {ω₀², ω₀² + c²ρ(L)}. Con ρG=0 se reduce al multiplicador clásico."""
        dt = self.cfg.dt
        om, ze = self.omega0, self.zeta
        r = None
        for k in (om ** 2, om ** 2 + (self.c ** 2) * self.rho_L):
            rk = dt * (k ** 2 + rho_G ** 2) / (2.0 * ze * om * k)
            r = rk if r is None else torch.maximum(r, rk)
        return F.relu(r - 1.9)

    def stability_penalty(self) -> torch.Tensor:
        """Margen de violación del certificado discreto (Prop. 2 y 3 del paper),
        consciente del orden y de los operadores antisimétricos G = b·A + β·A³.
        Escalar >= 0 (0 si certificado). Se añade a la pérdida (soft).

        Rama ONDA (Verlet; amortiguamiento y giroscópicos por diferencia atrasada):
        Schur-Cohn COMPLEJO por modo. Con q = Δt²·k (k hasta ω₀²+c²ρL),
        g = Δt·(2ζω₀ [+ D·ρL]) y μ̃ = Δt·ρ(G), ρ(G) = max_k |b·μ_k - β·μ_k³|:
            (i)  μ̃² < g(2-g)                       [genuinamente discreta]
            (ii) q(g² + μ̃²) < 2g(g(2-g) - μ̃²)     [en μ̃=0: q < 4(1-ζω₀Δt) clásico]
        OJO (Prop. 2): el giroscópico discretizado con diferencia ATRASADA no hereda
        la neutralidad continua (v'Gv=0): inyecta energía a razón √(1+μ̃²) por paso,
        que la disipación debe absorber -> (i). En Δt→0, (i) se vacía (μ̃²=O(Δt²)).

        Rama DIFUSIVA: euler explícito -> multiplicador complejo (_euler_complex_margin);
        IMEX con G estática en el núcleo -> incondicional (Prop. 3, sin penalización);
        IMEX con G gateada (explícita) -> multiplicador complejo conservador con topes.
        """
        cfg = self.cfg
        dt = cfg.dt
        om, ze = self.omega0, self.zeta
        b_d, beta_d, D_w = self._anti_coeffs_cert()
        mu = self.mu_A                                                    # (N,)
        rho_G = (b_d.unsqueeze(-1) * mu - beta_d.unsqueeze(-1) * mu ** 3).abs().amax(-1)  # (d_h,)
        mu_t = dt * rho_G
        q_hi = (dt ** 2) * (om ** 2 + (self.c ** 2) * self.rho_L)
        g_lo = 2.0 * ze * om * dt
        g_hi = g_lo + D_w * dt * self.rho_L

        diff_active = (cfg.gate_physics or cfg.mix_certificate
                       or (cfg.alpha_const is not None and cfg.alpha_const < 1.0))
        # ρG para la rama difusiva: si G va plegada al núcleo IMEX, no penaliza;
        # si es euler o G gateada-explícita, usa rho_G (aprendida o tope).
        if diff_active:
            if cfg.diff_solver == "implicit" and not cfg.gate_physics:
                m_euler = 0.0                                             # Prop. 3
            else:
                m_euler = self._euler_complex_margin(rho_G).mean()
        else:
            m_euler = 0.0

        if cfg.order == 2:
            pen = torch.zeros((), device=om.device)
            for g in (g_lo, g_hi):                                        # extremos de amortiguamiento
                pen = pen + F.relu(g - 1.9).mean()                        # g < 2 (holgura)
                pen = pen + F.relu(mu_t ** 2 - 0.95 * g * (2.0 - g)).mean()               # (i)
                pen = pen + F.relu(q_hi * (g ** 2 + mu_t ** 2)
                                   - 0.95 * 2.0 * g * (g * (2.0 - g) - mu_t ** 2)).mean()  # (ii)
            return pen + m_euler
        else:
            return self._euler_complex_margin(rho_G).mean()

    @torch.no_grad()
    def certificate_spectral_radius(self) -> float:
        """Radio espectral EXACTO del mapa discreto de la rama onda, por dimensión
        (companion 2N×2N con K, C, G matriciales: cubre el caso no-normal [L,A]≠0).
        Para eval/registro; la penalización soft es una cota suficiente conservadora."""
        cfg = self.cfg
        dt = cfg.dt
        N = self.laplacian.shape[0]
        om, ze = self.omega0.float(), self.zeta.float()
        cc = self.c.float()
        b_d, beta_d, D_w = self._anti_coeffs_cert()
        I = torch.eye(N, device=om.device)
        L = self.laplacian.float()
        A = self.adv.float()
        A3 = self.adv3.float() if hasattr(self, "adv3") else torch.zeros_like(A)
        Z = torch.zeros(N, N, device=om.device)
        rho_max = 0.0
        for d in range(cfg.d_h):
            Kh = (dt ** 2) * (om[d] ** 2 * I + cc ** 2 * L)
            Ch = dt * (2.0 * ze[d] * om[d] * I + D_w[d] * L)
            Gh = dt * (b_d[d] * A + beta_d[d] * A3)
            comp = torch.cat([torch.cat([2 * I - Kh - Ch - Gh, -(I - Ch - Gh)], dim=1),
                              torch.cat([I, Z], dim=1)], dim=0)
            rho_max = max(rho_max, float(torch.linalg.eigvals(comp).abs().max()))
        return rho_max

    # --------------------------- sub-espacios -------------------------------- #
    def split(self, h: torch.Tensor):
        d_f, d_e = self.cfg.d_f, self.cfg.d_e
        return h[..., :d_f], h[..., d_f:d_f + d_e], h[..., d_f + d_e:]

    # --------------------------- modulación top-down ------------------------- #
    def _readout_source(self) -> torch.Tensor:
        """Campo desde el que se leen las señales de modulación. Con
        elliptic_readout, la MODULACIÓN es NO-LOCAL: ψ = L⁺(h−h*) mezcla todos
        los nodos (la modulación de cada nodo depende de todo el campo). Con
        gain<1 se mezcla con el h local. Sin el flag, es el h_t local (retrocompat
        bitwise)."""
        if not self.cfg.elliptic_readout:
            return self.h_t
        dt = self.h_t.dtype                    # h_star está en FP32 (pin_fp32); castea
        hs = self.h_star.to(dt).unsqueeze(0)
        psi = torch.einsum("ij,bjd->bid", self.L_pinv.to(dt),
                           self.h_t - hs) + hs                              # (B,N,d_h)
        g = self.cfg.elliptic_gain
        return psi if g >= 1.0 else g * psi + (1.0 - g) * self.h_t

    def modulation(self) -> dict[str, torch.Tensor]:
        """Señales de modulación a partir del campo de lectura (local o elíptico)."""
        src = self._readout_source()
        h_fis, h_emo, h_umb = self.split(src)
        return {
            "halt_threshold": torch.sigmoid(self.W_tau(h_umb)).squeeze(-1),   # (B,N)
            "wm_write": torch.sigmoid(self.W_wm_write(h_emo)),                # (B,N,d_h)
            "wm_forget": torch.sigmoid(self.W_wm_forget(h_emo)),             # (B,N,d_h)
            "router_bias": (self.cfg.alpha_router * self.w_router(h_emo)).squeeze(-1),  # (B,N)
            "block_gate": torch.tanh(src),                                   # (B,N,d_h)
        }

    # --------------------------- pérdidas auxiliares ------------------------- #
    def interoception_loss(self, intero_target: torch.Tensor) -> torch.Tensor:
        """L_interoc = ||ŝ - s||². h^fis debe predecir la interocepción real."""
        h_fis, _, _ = self.split(self.h_t)
        return F.mse_loss(self.W_intero_pred(h_fis), intero_target)

    def homeostatic_loss(self) -> torch.Tensor:
        """L_homeo = ||h - h*||². Penaliza alejarse del equilibrio sin motivo."""
        return (self.h_t - self.h_star.unsqueeze(0)).pow(2).mean()

    def state_summary(self) -> dict[str, float]:
        with torch.no_grad():
            dev = (self.h_t - self.h_star.unsqueeze(0)).norm(dim=-1).mean().item()
            var = self.h_t.var().item()
        return {"deviation_from_rest": dev, "vei_variance": var}
