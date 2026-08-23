"""
MiuraCognitive - Integración completa
======================================
Une los cuatro componentes en una sola arquitectura end-to-end:

  1. Transformer base (backbone)                  -> transformer.py
  2. Recurrencia adaptativa (adaptive depth)      -> adaptive_depth.py
  3. Memoria de trabajo explícita                 -> working_memory.py
  4. Homeostatic Background Processor (HBP)        -> hbp.py

DISEÑO (v2, post-auditoría): el HBP evoluciona su VEI UN tick por ITERACIÓN del
reasoner (el "tiempo" de la ecuación de onda = tiempo de pensamiento), y la
WORKING MEMORY vive DENTRO de ese bucle (un write+read por iteración, con gates
modulados por el VEI del tick actual): es memoria real del proceso de
pensamiento, no una proyección estática del input.

Correcciones de la auditoría integradas aquí:
  - Pooling con MÁSCARA de PAD en todos los canales globales (kick, WM, halting,
    g_φ): sin ella, el 60-80% de posiciones PAD domina el mean y las señales
    codifican longitud en vez de estado.
  - Interocepción POR NODO del grafo: cada bloque del backbone alimenta a su
    nodo con SUS señales (norma, activación, entropía de atención); el reasoner
    y la WM con las suyas. Así el término c²∇² tiene estructura espacial real
    que propagar.
  - L_interoc acumulada sobre TODOS los ticks del rollout (no solo el último).
  - Peso extra (beta_final) a la respuesta final: con supervisión densa la
    posición evaluada pesa 1/(K+1) en la CE — menos peso justo en lo difícil.

Nodos del grafo del HBP: N = L backbone + 1 reasoner + 1 WM.

Autor: Francisco M. Arrabal (Curro) + asistencia Claude
"""

from __future__ import annotations
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import MiuraConfig, RMSNorm, TransformerBlock, build_rope_cache
from .hbp import HBPConfig, HomeostaticBackgroundProcessor
from .adaptive_depth import AdaptiveHalting, halting_regularization
from .working_memory import WorkingMemory


@dataclass
class MiuraFullConfig:
    """Configuración del modelo completo. Compone las sub-configuraciones."""
    transformer: MiuraConfig = field(default_factory=MiuraConfig)
    hbp: HBPConfig = field(default_factory=HBPConfig)
    use_hbp: bool = True
    use_adaptive_depth: bool = True
    use_working_memory: bool = True
    max_halt_steps: int = 10           # presupuesto de pensamiento (iteraciones)
    halt_mod_gain: float = 2.0         # fuerza con que el VEI sesga el halting (0 = no sesga)
    wm_slots: int = 8
    wm_in_loop: bool = True            # WM dentro del bucle del reasoner (False = write/read único pre-bucle)
    halt_len_feature: bool = True      # longitud válida explícita como feature del halting
    # PonderNet correcto: E_{p(n|x)}[CE(logits_n,y)] en vez de aplicar CE a la
    # mezcla de estados. Da crédito directo al controlador por la utilidad de
    # cada profundidad. Opt-in para no alterar los experimentos históricos.
    ponder_expected_loss: bool = False
    # Pesos de la pérdida compuesta
    beta_halt: float = 0.01
    beta_homeo: float = 0.001
    beta_intero: float = 0.1
    beta_stab: float = 0.1             # penalización del certificado de estabilidad
    # beta_final=0: ponderar extra la respuesta final PARECE razonable, pero con
    # el objetivo de mezcla PonderNet (CE sobre Σ p_n·x_n) premia que los estados
    # TEMPRANOS ya decodifiquen la respuesta -> colapsa la iteración (E[n_iter]
    # 11.5 -> 6.5 y accuracy largo 0.20 -> 0.10; barrido v2e). Negativo empírico.
    beta_final: float = 0.0
    # --- F3b (PREREG_F3B v2) ---
    # session_mode activa: (a) el conjunto de OBSERVACIÓN DEL ENTORNO idéntico
    # para todos los brazos (§3: fracción de presupuesto gastada/restante,
    # stake, instancias restantes — como features del halting Y como cola de la
    # interocepción por nodo); (b) la interocepción de completitud
    # (margen/entropía del decode) computada en la rama COMPARTIDA (equidad);
    # (c) presupuesto por-fila (row_caps → λ-clamp) y evento de techo que
    # incluye el agotamiento de presupuesto. session_mode=False reproduce
    # exactamente el comportamiento F3a/G0.
    session_mode: bool = False
    SESSION_OBS: int = 4               # spent_frac, remaining_frac, stake, inst_left
    # --- N2 (PREREG_N2 v2 §3): módulo de valor ---
    # off            : sin valor (todas las fases previas)
    # endo           : V̂ (p̂, stakê) sobre estado tick-2 DETACHED; stakê.detach()
    #                  → value_signal del plano desde el tick 3
    # endo_uncoupled : V̂ entrenado (solo log/atribución); canal AUSENTE
    # oracle         : stake VERDADERO (value_ctx) → canal desde el tick 1
    # gating_endo    : vía espejo al halting sin plano (bias tanh, init 0)
    value_mode: str = "off"
    beta_val: float = 1.0              # L_val solo toca las cabezas V̂ (detach)

    def __post_init__(self):
        # Nodos del HBP = L backbone + reasoner + working memory
        self.hbp.n_nodes = self.transformer.n_layers + 2


class MiuraCognitiveFull(nn.Module):
    """Modelo MiuraCognitive completo con HBP, recurrencia adaptativa y WM."""

    def __init__(self, cfg: MiuraFullConfig):
        super().__init__()
        self.cfg = cfg
        tcfg = cfg.transformer

        # --- Backbone Transformer ---
        self.tok_emb = nn.Embedding(tcfg.vocab_size, tcfg.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(tcfg, layer_idx=i) for i in range(tcfg.n_layers)
        ])
        self.norm_f = RMSNorm(tcfg.d_model)
        self.lm_head = nn.Linear(tcfg.d_model, tcfg.vocab_size, bias=False)
        if tcfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(tcfg.max_seq_len, tcfg.d_head,
                                    tcfg.rope_theta, torch.device("cpu"), torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # --- HBP ---
        if cfg.use_hbp:
            self.hbp = HomeostaticBackgroundProcessor(cfg.hbp)
            # Señales crudas de interocepción por nodo (5: 3 propias del nodo +
            # esfuerzo n/N_max + longitud válida normalizada) -> d_intero
            self.intero_proj = nn.Linear(5, cfg.hbp.d_intero)
            # Modulación: block_gate del VEI (d_h) -> d_model del bloque/reasoner
            self.mod_proj = nn.Linear(cfg.hbp.d_h, tcfg.d_model)
            # Drive del input al VEI (sembrado + forzamiento g_φ): d_model -> d_h
            self.input_to_hbp = nn.Linear(tcfg.d_model, cfg.hbp.d_h)

        # --- Recurrencia adaptativa (reasoner) ---
        if cfg.use_adaptive_depth:
            self.halting = AdaptiveHalting(
                tcfg.d_model, max_steps=cfg.max_halt_steps,
                n_obs=cfg.SESSION_OBS if cfg.session_mode else 0)
            self.reasoner_block = TransformerBlock(tcfg, layer_idx=tcfg.n_layers)

        # --- Working memory ---
        if cfg.use_working_memory:
            self.wm = WorkingMemory(tcfg.d_model, n_slots=cfg.wm_slots)
            # Integración RESIDUAL del read: h <- h + wm_out(read). Dentro del
            # bucle del reasoner, una transformación que REEMPLACE el estado
            # (p.ej. Linear(cat(h, read)) con init pequeña) se aplica
            # iterativamente y aniquila la señal -> el reasoner colapsa a ~1-2
            # iteraciones (mismo mecanismo que el gate sigmoid pre-fix).
            self.wm_out = nn.Linear(tcfg.d_model, tcfg.d_model)

        self._last_n_expected = None       # E[n_iter] del reasoner (diagnóstico)
        self._last_halt_dist = None        # p(parar en n) del rollout soft, (B,N)
        self._last_effective_steps = None  # cuota entera ejecutada en modo forzado
        self._last_reasoner_step_units = None  # suma de pasos muestra-reasoner
        self.record_trace = False          # si True, forward registra la traza del HBP por tick
        self._trace = []                   # [{n, vei_var, deviation, halt_threshold, reasoner_mod_norm}]
        # --- Fase 3a: token de lectura para la interocepción de completitud
        # (posición ARROW en permcomp; solo lo usa la variante mHBP) ---
        self.readout_token_id = 2
        # --- Instrumentación G0 (inerte por defecto) ---
        self.record_step_states = False    # si True: guarda x_t de CADA tick (detached)
        self.tick_intervene = None         # callback (x, n)->x tras cada paso (swaps/lesiones)
        self._last_step_states = None      # [x_1..x_N] (B,T,d) detached
        self._last_halt_probs_live = None  # [p_1..p_N] (B,) detached
        self._wm_box = None                # referencia al wm_box del forward en curso
        # --- F3b: contexto de sesión (lo fija el runner ANTES de cada forward
        # de instancia; None = neutro, reproduce F3a). Claves: spent_frac,
        # remaining_frac, stake, inst_left (todas (B,) en [0,1]) y row_caps
        # ((B,) long o None). persist_plane=True salta el reset del controlador
        # (el runner resetea en la frontera de SESIÓN, no de instancia). ---
        self.budget_ctx = None
        self.persist_plane = False
        # --- N2: módulo de valor (PREREG_N2 v2 §3) ---
        if cfg.value_mode != "off":
            d = tcfg.d_model
            # RONDA 2 del motivo (2026-08-04): V̂ lee [pooled ⊕ estado@slot0 ⊕
            # estado@slot1] (las posiciones de anuncio son parte del SPEC del
            # entorno — auto-observación legítima; el pooled solo diluye los
            # slots 1/T) con MLP de 1 capa oculta (la igualdad es XOR-like:
            # inaccesible a un lector lineal). Misma familia que el probe VG2b.
            def _head():
                h = nn.Sequential(nn.Linear(3 * d, 32), nn.Tanh(),
                                  nn.Linear(32, 1))
                for mmod in h:
                    if isinstance(mmod, nn.Linear):
                        nn.init.normal_(mmod.weight, std=0.05)
                        nn.init.zeros_(mmod.bias)
                return h
            self.value_p = _head()               # p̂: automodelo online (BCE)
            self.value_s = _head()               # stakê (softplus, MSE payoff/p̂)
            # normalización propia (vía espejo gating_endo); EMA solo en train
            self.register_buffer("val_mean", torch.tensor(1.0))
            self.register_buffer("val_var", torch.tensor(1.0))
            if cfg.value_mode == "gating_endo":
                self.value_halt_a = nn.Parameter(torch.zeros(1))  # init NEUTRO
        # value_ctx lo fija el runner por batch: {"stake_true": (B,)|None}.
        # stake_true se usa SOLO en L_val (consecuencia realizada) salvo en
        # oracle, donde alimenta el canal desde el tick 1.
        self.value_ctx = None
        self.value_lesion = False          # endo_cut (eval): canal := media
        self._val_cache = None
        self.apply(self._init_weights)
        if cfg.use_hbp:
            # _init_weights sobrescribe la init especial de f_θ/g_φ; la restauramos.
            self.hbp.init_physics()
            # La entropía de atención (O(T²), no_grad) solo se computa donde se
            # consume: las variantes con HBP la usan como interocepción por nodo.
            for block in self.blocks:
                block.attn.compute_entropy = True
            if cfg.use_adaptive_depth:
                self.reasoner_block.attn.compute_entropy = True

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                alpha_force: torch.Tensor | None = None,
                forced_steps: torch.Tensor | None = None,
                sample_weights: torch.Tensor | None = None):
        """idx: (B, T). targets: (B, T) opcional. Devuelve (logits, loss_dict).
        alpha_force: (B,)|None — α de la física del HBP impuesto por instancia
        (experimentos de oráculo: el modo de la tarea dicta onda↔difusión).
        forced_steps: (B,)|None — intervención de evaluación que ejecuta una
        cuota entera por muestra con batch activo disperso. No forma la mezcla
        PonderNet y su suma es el presupuesto efectivo del reasoner.
        sample_weights: (B,)|None — pesos de CE por instancia (N2 §2: la
        estructura de pago del ENTORNO; payoff=stake·correcto)."""
        B, T = idx.shape
        assert T <= self.cfg.transformer.max_seq_len, \
            f"Secuencia ({T}) excede max_seq_len ({self.cfg.transformer.max_seq_len})"
        device, dtype = idx.device, self.tok_emb.weight.dtype
        tcfg = self.cfg.transformer
        n_layers = tcfg.n_layers
        reasoner_idx, wm_idx = n_layers, n_layers + 1
        N = self.cfg.hbp.n_nodes if self.cfg.use_hbp else 0
        if forced_steps is not None and not self.cfg.use_adaptive_depth:
            raise ValueError("forced_steps requiere use_adaptive_depth=True")

        # Máscara de posiciones válidas (PAD=0 en todos los datasets). Se usa
        # para la SEÑAL DE LONGITUD explícita (frac_valid), NO para enmascarar
        # el pooling: las posiciones PAD funcionan como scratchpad computacional
        # del reasoner (atienden causalmente al contenido y acumulan cómputo);
        # excluirlas de los canales globales desconecta ese cómputo y colapsa
        # la profundidad de iteración (verificado: E[n_iter] 12 -> 4.5).
        mask = (idx != 0)                                          # (B, T) bool
        pm_count = mask.to(dtype).sum(dim=1, keepdim=True).clamp_min(1.0)   # (B, 1)

        def mmean(t: torch.Tensor) -> torch.Tensor:
            """Pooling global sobre la secuencia completa (incluye scratchpad)."""
            return t.mean(dim=1)

        if self.record_trace:
            self._trace = []
        self._val_cache = None

        # --- N2: valor por tick (helper compartido por las dos vías) ---
        def value_tick(x_n, n, active_idx=None):
            """Cachea V̂ en el tick 2 (estado DETACHED) y devuelve el
            value_signal del tick (None = canal ausente)."""
            vm = self.cfg.value_mode
            if vm == "off":
                return None
            if vm == "oracle":
                st = (self.value_ctx or {}).get("stake_true")
                if st is None:
                    return None
                st = st.to(device=device, dtype=torch.float32)
                return (st if active_idx is None
                        else st.index_select(0, active_idx))
            # T-N3a: en la ruta FORCED active_idx nunca es None (nonzero
            # devuelve tensor aunque el batch entero esté activo) — sin este
            # OR, el canal de valor quedaba mudo en ejecución forzada y la
            # sonda N3 mediría OTRO sistema que el nativo (panel N3, crítico).
            full_batch = (active_idx is None
                          or active_idx.numel() == idx.shape[0])
            if n == 2 and self._val_cache is None and full_batch:
                # el valor LEE, no esculpe. VG4-fix 2 (2026-08-04): el sensor
                # de los slots son los EMBEDDINGS CRUDOS (tok_emb), no el
                # estado contextualizado — el entrenamiento de la tarea
                # SUPRIME la info de slots del estado (probe en tronco
                # N2-entrenado: 0.79 vs 1.00 en embeddings; la advertencia
                # del panel «el encoder descarta lo tarea-irrelevante»,
                # confirmada). El pooled da a p̂ el estado de la tarea.
                emb = self.tok_emb(idx[:, :2])
                feat = torch.cat([mmean(x_n), emb[:, 0, :], emb[:, 1, :]],
                                 dim=-1).detach()
                self._val_feat = feat            # para el replay del runner
                wdt_v = self.value_p[0].weight.dtype
                p_hat = torch.sigmoid(
                    self.value_p(feat.to(wdt_v))).squeeze(-1).float()
                # VG4-fix (2026-08-04): value_s regresa el PAYOFF observable
                # crudo (E[payoff|estado] = stake·p — target suave); stakê =
                # payoff̂/p̂ se computa al LEER (el target-cociente era
                # inaprendible online: AUC 0.5 en 6/6 runs del piloto).
                pay_hat = F.softplus(
                    self.value_s(feat.to(wdt_v))).squeeze(-1).float()
                s_hat = pay_hat / p_hat.detach().clamp(min=0.05)
                self._val_cache = {"p": p_hat, "pay": pay_hat, "s": s_hat}
            if (vm in ("endo_uncoupled", "gating_endo")
                    or self._val_cache is None or n < 3):
                return None
            v = self._val_cache["s"].detach()
            if self.value_lesion:                          # endo_cut (árbitro)
                v = torch.full_like(v, float(v.mean()))
            return v if active_idx is None else v.index_select(0, active_idx)

        x = self.tok_emb(idx)

        # --- F3b: observaciones del entorno (§3) — idénticas para todos los
        # brazos; neutras si el runner no ha fijado budget_ctx ---
        row_caps = None
        obs_feats = None
        if self.cfg.session_mode:
            ctx = self.budget_ctx or {}
            def _obs(key, default):
                v = ctx.get(key)
                if v is None:
                    return torch.full((B,), default, device=device,
                                      dtype=torch.float32)
                return v.to(device=device, dtype=torch.float32)
            obs_spent = _obs("spent_frac", 0.0)
            obs_remain = _obs("remaining_frac", 1.0)
            obs_stake = _obs("stake", 0.0)          # normalizado: (stake-1)/3 ∈ {0,1}
            obs_left = _obs("inst_left", 1.0)
            obs_feats = torch.stack(
                [obs_spent, obs_remain, obs_stake, obs_left], dim=-1).to(dtype)
            row_caps = ctx.get("row_caps")
            if forced_steps is not None and row_caps is not None:
                raise ValueError("forced_steps (batería) exige presupuesto ∞ "
                                 "(row_caps=None) — §8 del prereg")

        # --- Sembrado del HBP: rompe el punto fijo y activa g_φ (dep. del input) ---
        modulation = None
        if self.cfg.use_hbp:
            if self.persist_plane:
                # Persistencia de sesión (§6): el runner resetea en la frontera
                # de sesión; aquí solo se verifica la coherencia del batch.
                if self.hbp.h_t is None or self.hbp.h_t.shape[0] != B:
                    raise RuntimeError("persist_plane=True exige reset_state "
                                       "del runner en la frontera de sesión")
            else:
                self.hbp.reset_state(B, device=device, dtype=dtype)
            kick = self.input_to_hbp(mmean(x)).unsqueeze(1).expand(B, N, -1)  # (B,N,d_h)
            if hasattr(self.hbp, "on_instance_kick"):
                # Adaptadores de sesión: semántica declarada del drive en la
                # frontera de instancia (mhbp: h_t := kick, sobrescribe; §6).
                self.hbp.on_instance_kick(kick)
            else:
                self.hbp.h_t = self.hbp.h_t + kick                # v != 0 en el 1er tick
            modulation = self.hbp.modulation()

        # --- Backbone con modulación por bloque (nodos 0..L-1) ---
        for i, block in enumerate(self.blocks):
            block_mod = None
            if modulation is not None:
                block_mod = self.mod_proj(modulation["block_gate"][:, i, :])   # (B,d_model)
            x = block(x, self.rope_cos, self.rope_sin, modulation=block_mod)

        # Señales de interocepción por nodo del BACKBONE (estáticas durante el
        # pensamiento: son "el cuerpo" sobre el que se piensa).
        backbone_sigs = None
        if self.cfg.use_hbp:
            backbone_sigs = []
            for block in self.blocks:
                io = block._interoception
                backbone_sigs.append(torch.stack(
                    [io["output_norm"].to(dtype), io["activation_abs"].to(dtype),
                     io["attn_entropy"].to(dtype)], dim=-1))       # (B,3) por bloque

        # --- Working memory: memoria del PROCESO de pensamiento ---
        # El write/read vive dentro del bucle del reasoner (un write+read por
        # iteración). Con un único write sobre memoria cero, el read sería
        # colineal con write_value(state): una proyección, no una memoria.
        wm_box = {"mem": None, "w": None, "f": None}
        if self.cfg.use_working_memory:
            wm_box["mem"] = self.wm.init_memory(B, device, dtype)
            if modulation is not None:
                wm_box["w"] = modulation["wm_write"][:, wm_idx, :]
                wm_box["f"] = modulation["wm_forget"][:, wm_idx, :]
        self._wm_box = wm_box              # instrumentación G0: swaps tocan la WM

        # --- Reasoner recurrente = eje temporal del HBP ---
        n_expected = None
        step_states = None
        halt_probs_live = None
        intero_loss_acc = [torch.zeros((), device=device, dtype=torch.float32), 0]
        if self.cfg.use_adaptive_depth:
            # WM pre-bucle (wm_in_loop=False): write/read único aquí, residual.
            if self.cfg.use_working_memory and not self.cfg.wm_in_loop:
                state0 = mmean(x)
                wm_box["mem"] = self.wm.write(wm_box["mem"], state0, wm_box["w"], wm_box["f"])
                read0 = self.wm.read(wm_box["mem"], state0)
                x = x + self.wm_out(read0).unsqueeze(1)

            def step_fn(h, mod, active_idx=None):
                h = self.reasoner_block(h, self.rope_cos, self.rope_sin, modulation=mod)
                if self.cfg.use_working_memory and self.cfg.wm_in_loop:
                    state = mmean(h)                                            # (B,d)
                    if active_idx is None:
                        mem = wm_box["mem"]
                        w, f = wm_box["w"], wm_box["f"]
                    else:
                        mem = wm_box["mem"].index_select(0, active_idx)
                        w = (None if wm_box["w"] is None
                             else wm_box["w"].index_select(0, active_idx))
                        f = (None if wm_box["f"] is None
                             else wm_box["f"].index_select(0, active_idx))
                    mem_next = self.wm.write(mem, state, w, f)
                    if active_idx is None:
                        wm_box["mem"] = mem_next
                    else:
                        wm_box["mem"] = wm_box["mem"].index_copy(
                            0, active_idx, mem_next)
                    read = self.wm.read(mem_next, state)                         # (B,d)
                    # RESIDUAL (no reemplaza h): perturbación aditiva estable
                    # bajo aplicación iterada.
                    h = h + self.wm_out(read).unsqueeze(1)
                return h

            hbp_tick = None
            if self.cfg.use_hbp:
                def hbp_tick(x_n, x_prev, n, active_idx=None):
                    B_tick = x_n.shape[0]
                    effort = torch.full((B_tick,), float(n) / self.cfg.max_halt_steps,
                                        device=device, dtype=dtype)             # (B_tick,)
                    # Longitud válida normalizada: señal explícita de tamaño de
                    # la entrada (interocepción legítima del "cuerpo de trabajo")
                    pm_tick = (pm_count if active_idx is None
                               else pm_count.index_select(0, active_idx))
                    fv = (pm_tick.squeeze(-1) / T).to(dtype)                    # (B_tick,)

                    # --- Interocepción de COMPLETITUD (G0.1) y evento de techo.
                    # En session_mode se computa AQUÍ, en la rama COMPARTIDA:
                    # todos los controladores la ven (equidad §3 del prereg;
                    # en F3a era exclusiva del mhbp — crítico #2 del panel). ---
                    dmargin = dentropy = cap = None
                    is_sess = self.cfg.session_mode
                    is_mhbp_ctrl = getattr(self.hbp, "is_mhbp", False)
                    if is_sess or is_mhbp_ctrl:
                        with torch.no_grad():
                            arrow = (idx == self.readout_token_id).float().argmax(dim=1)
                            arrow = arrow.clamp(max=x_n.shape[1] - 1)
                            if active_idx is not None:
                                arrow = arrow.index_select(0, active_idx)
                            rws = torch.arange(x_n.shape[0], device=device)
                            lg = self.lm_head(self.norm_f(x_n[rws, arrow])).float()
                            pr = torch.softmax(lg, dim=-1)
                            top2 = pr.topk(2, dim=-1).values
                            dmargin = (top2[:, 0] - top2[:, 1])
                            dentropy = -(pr * (pr + 1e-9).log()).sum(-1) \
                                / torch.log(torch.tensor(float(pr.shape[-1])))
                            # Evento de techo (§6.3): N_max O agotamiento del
                            # presupuesto por-fila (incluye forzados a n=1).
                            cap = torch.full_like(
                                dmargin,
                                1.0 if n == self.cfg.max_halt_steps else 0.0)
                            if row_caps is not None:
                                rc = (row_caps if active_idx is None
                                      else row_caps.index_select(0, active_idx))
                                cap = torch.maximum(cap, (n >= rc).float())

                    if is_sess:
                        # Cola extendida (§3): [esfuerzo, long, gastado,
                        # restante, stake, inst_left, margen, entropía] — la
                        # MISMA para todos los brazos con controlador.
                        def _sel(t):
                            return (t if active_idx is None
                                    else t.index_select(0, active_idx))
                        tail = torch.stack(
                            [effort, fv, _sel(obs_spent).to(dtype),
                             _sel(obs_remain).to(dtype), _sel(obs_stake).to(dtype),
                             _sel(obs_left).to(dtype), dmargin.to(dtype),
                             dentropy.to(dtype)], dim=-1)            # (B_tick,8)
                    else:
                        tail = torch.stack([effort, fv], dim=-1)     # (B_tick,2)
                    rows = []
                    # Nodos 0..L-1: señales del PROPIO bloque del backbone
                    for sig in backbone_sigs:
                        sig_tick = (sig if active_idx is None
                                    else sig.index_select(0, active_idx))
                        rows.append(torch.cat([sig_tick, tail], dim=-1))         # (B_tick,5)
                    # Nodo reasoner: señales del tick actual (dependen del input)
                    progress = (x_n - x_prev).norm(dim=-1).mean(dim=1)          # (B,)
                    act_r = x_n.abs().mean(dim=(1, 2))                          # (B,)
                    ent_r = self.reasoner_block.attn._last_attn_entropy.to(dtype)
                    rows.append(torch.cat([torch.stack([progress, act_r, ent_r], dim=-1), tail], dim=-1))
                    # Nodo WM: ocupación de la memoria (cuán "llena" está)
                    if wm_box["mem"] is not None:
                        mem_tick = (wm_box["mem"] if active_idx is None else
                                    wm_box["mem"].index_select(0, active_idx))
                        occ = mem_tick.norm(dim=-1).mean(dim=-1)                # (B_tick,)
                    else:
                        occ = torch.zeros(B_tick, device=device, dtype=dtype)
                    rows.append(torch.cat([torch.stack([occ, occ, torch.zeros_like(occ)], dim=-1), tail], dim=-1))

                    s_raw = torch.stack(rows, dim=1)             # (B,N,3+len(tail))
                    if is_sess:
                        # --- F3b: TODOS los controladores de sesión pasan por
                        # la MISMA interfaz (paridad de observación y de
                        # actuadores por construcción, §3). ---
                        if not hasattr(self.hbp, "step_session"):
                            raise RuntimeError(
                                "session_mode exige un controlador con "
                                "step_session (adaptadores F3b)")
                        ext_m = self.input_to_hbp(mmean(x_n))               # (B, d_h)
                        mod_full = self.hbp.step_session(
                            s_raw, ext_m, effort, dmargin, dentropy, cap,
                            active_idx=active_idx)
                    elif is_mhbp_ctrl:
                        # --- Fase 3a: plano mHBP (PREREG_F3A); completitud ya
                        # computada arriba en la rama compartida. N2: el valor
                        # (stakê endógeno o stake del oráculo) entra por
                        # value_signal — DETACHED, veto por-cabeza en el plano.
                        ext_m = self.input_to_hbp(mmean(x_n))                   # (B, d_h)
                        vsig = value_tick(x_n, n, active_idx)
                        mod_full = self.hbp.step_mhbp(
                            s_raw, ext_m, effort, dmargin, dentropy, cap,
                            active_idx=active_idx, value_signal=vsig)
                    else:
                        s_n = self.intero_proj(s_raw)                           # (B,N,d_intero)
                        ext = self.input_to_hbp(mmean(x_n)).unsqueeze(1).expand(B_tick, N, -1)
                        alpha_tick = alpha_force
                        if alpha_force is not None and active_idx is not None:
                            alpha_tick = alpha_force.index_select(0, active_idx)
                        self.hbp.step(s_n, ext_force=ext, alpha_force=alpha_tick,
                                      active_idx=active_idx)
                        mod_full = self.hbp.modulation()
                    mod = (mod_full if active_idx is None else
                           {k: v.index_select(0, active_idx)
                            for k, v in mod_full.items()})
                    # La dinámica modula la WM POR TICK (gates del VEI actual)
                    if self.cfg.use_working_memory:
                        w_tick = mod["wm_write"][:, wm_idx, :]
                        f_tick = mod["wm_forget"][:, wm_idx, :]
                        if active_idx is None:
                            wm_box["w"], wm_box["f"] = w_tick, f_tick
                        else:
                            wm_box["w"] = wm_box["w"].index_copy(0, active_idx, w_tick)
                            wm_box["f"] = wm_box["f"].index_copy(0, active_idx, f_tick)
                    halt_bias = (mod["halt_threshold"][:, reasoner_idx] - 0.5) \
                        * self.cfg.halt_mod_gain                                # (B,)
                    reasoner_mod = self.mod_proj(mod["block_gate"][:, reasoner_idx, :])
                    # L_interoc acumulada sobre los ticks (el h^fis de CADA tick
                    # debe predecir la interocepción real de ese tick)
                    if forced_steps is None:
                        s_for_loss = (s_raw if (is_sess or is_mhbp_ctrl)
                                      else s_n)
                        intero_loss_acc[0] = intero_loss_acc[0] + \
                            self.hbp.interoception_loss(s_for_loss.detach()).float()
                        intero_loss_acc[1] += 1
                    if self.record_trace:
                        with torch.no_grad():
                            ss = self.hbp.state_summary()
                            rec = {
                                "n": n,
                                "vei_var": ss["vei_variance"],
                                "deviation": ss["deviation_from_rest"],
                                "halt_threshold": float(mod["halt_threshold"][:, reasoner_idx].mean()),
                                "reasoner_mod_norm": float(reasoner_mod.norm(dim=-1).mean()),
                            }
                            # Física elegida en este tick (si el HBP la gatea): permite
                            # ver cómo el modelo cambia su PDE DURANTE el pensamiento.
                            if getattr(self.hbp.cfg, "gate_physics", False):
                                rec.update(self.hbp.physics_state())
                            self._trace.append(rec)
                    return halt_bias, reasoner_mod

            if hbp_tick is None and self.cfg.value_mode == "gating_endo":
                # N2 §4: vía ESPEJO — el valor modula el mismo punto (λ
                # pre-techo) con la misma forma y cota que los brazos con
                # plano (bias tanh, init a=0 neutro), SIN plano de por medio.
                def hbp_tick(x_n, x_prev, n, active_idx=None):
                    value_tick(x_n, n, active_idx)         # cachea en tick 2
                    B_t = x_n.shape[0]
                    if self._val_cache is None or n < 3:
                        return torch.zeros(B_t, device=device), None
                    s = self._val_cache["s"].detach()
                    if self.training:
                        with torch.no_grad():
                            self.val_mean.mul_(0.99).add_(0.01 * s.mean())
                            self.val_var.mul_(0.99).add_(
                                0.01 * s.var().clamp(min=1e-3))
                    s_n = (s - self.val_mean) / torch.sqrt(self.val_var + 1e-5)
                    bias = torch.tanh(self.value_halt_a
                                      * s_n.to(self.value_halt_a.dtype)).float()
                    hb = 0.5 * bias * self.cfg.halt_mod_gain
                    if active_idx is not None:
                        hb = hb.index_select(0, active_idx)
                    return hb, None

            # pool_mask=None desactiva la feature de longitud (frac_valid=cte).
            want_states = ((self.cfg.ponder_expected_loss and targets is not None)
                           or self.record_step_states)
            if forced_steps is None:
                if want_states:
                    (x, n_expected, halt_dist, step_states,
                     halt_probs_live) = self.halting(
                        step_fn, x, hbp_tick=hbp_tick,
                        pool_mask=mask if self.cfg.halt_len_feature else None,
                        return_step_states=True,
                        intervene_fn=self.tick_intervene,
                        row_caps=row_caps, obs_feats=obs_feats)
                    if self.record_step_states:
                        self._last_step_states = [s.detach() for s in step_states]
                        self._last_halt_probs_live = [p.detach() for p in halt_probs_live]
                else:
                    x, n_expected, halt_dist = self.halting(
                        step_fn, x, hbp_tick=hbp_tick,
                        pool_mask=mask if self.cfg.halt_len_feature else None,
                        intervene_fn=self.tick_intervene,
                        row_caps=row_caps, obs_feats=obs_feats)
            else:
                x, n_expected, halt_dist = self.halting.forward_forced(
                    step_fn, x, forced_steps=forced_steps, hbp_tick=hbp_tick)
        elif self.cfg.use_working_memory:
            # Fallback sin reasoner (ninguna variante actual): write/read único
            state = mmean(x)
            wm_box["mem"] = self.wm.write(wm_box["mem"], state, wm_box["w"], wm_box["f"])
            read = self.wm.read(wm_box["mem"], state)
            x = x + self.wm_out(read).unsqueeze(1)

        self._last_n_expected = n_expected
        if self.cfg.use_adaptive_depth:
            self._last_halt_dist = torch.stack(halt_dist, dim=1)
        else:
            self._last_halt_dist = None
        if forced_steps is None:
            self._last_effective_steps = None
            self._last_reasoner_step_units = None
        else:
            self._last_effective_steps = n_expected.detach()
            self._last_reasoner_step_units = int(n_expected.sum().item())

        # --- Cabeza de lenguaje ---
        x = self.norm_f(x)
        logits = self.lm_head(x)

        # --- Pérdidas ---
        loss_dict = {}
        if targets is not None:
            if self.cfg.ponder_expected_loss and forced_steps is None:
                if step_states is None or halt_probs_live is None:
                    raise RuntimeError("faltan estados por paso para la pérdida PonderNet")
                valid = (targets != -1)
                denom = valid.sum(dim=1).clamp_min(1)
                losses_by_step = []
                for h_n in step_states:
                    logits_n = self.lm_head(self.norm_f(h_n))
                    token_loss = F.cross_entropy(
                        logits_n.view(-1, logits_n.size(-1)), targets.view(-1),
                        ignore_index=-1, reduction="none").view(B, T)
                    losses_by_step.append(token_loss.sum(dim=1) / denom)
                per_step = torch.stack(losses_by_step, dim=1)          # (B,N)
                p_live = torch.stack(halt_probs_live, dim=1)           # (B,N)
                row_loss = (p_live * per_step).sum(dim=1)              # (B,)
                if sample_weights is not None:
                    w = sample_weights.to(row_loss.dtype)
                    loss_lm = (w * row_loss).sum() / w.sum().clamp_min(1e-6)
                else:
                    loss_lm = row_loss.mean()
                loss_dict["ponder_expected_task"] = loss_lm
            elif sample_weights is not None:
                # N2: CE ponderada por instancia (la estructura de pago del
                # entorno) — media por fila, media ponderada sobre el batch
                valid = (targets != -1)
                denom_w = valid.sum(dim=1).clamp_min(1)
                token_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1),
                    ignore_index=-1, reduction="none").view(B, T)
                row_loss = token_loss.sum(dim=1) / denom_w
                w = sample_weights.to(row_loss.dtype)
                loss_lm = (w * row_loss).sum() / w.sum().clamp_min(1e-6)
            else:
                loss_lm = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss_dict["lm"] = loss_lm
            total = loss_lm

            # Peso extra a la RESPUESTA FINAL (la última posición supervisada):
            # con supervisión densa pesa 1/(K+1) en la CE media — justo menos
            # peso en las muestras difíciles, que es donde se evalúa.
            if self.cfg.beta_final > 0:
                valid = (targets != -1)
                has = valid.any(dim=1)
                if has.any():
                    idx_last = valid.float().cumsum(dim=1).argmax(dim=1)        # (B,)
                    rows = torch.arange(B, device=device)[has]
                    cols = idx_last[has]
                    loss_final = F.cross_entropy(logits[rows, cols], targets[rows, cols])
                    loss_dict["final"] = loss_final
                    total = total + self.cfg.beta_final * loss_final

            if self.cfg.use_adaptive_depth and n_expected is not None:
                loss_halt = halting_regularization(n_expected)
                loss_dict["halt"] = loss_halt
                total = total + self.cfg.beta_halt * loss_halt

            if self.cfg.use_hbp:
                if intero_loss_acc[1] > 0:
                    loss_intero = (intero_loss_acc[0] / intero_loss_acc[1]).to(loss_lm.dtype)
                    loss_dict["intero"] = loss_intero
                    total = total + self.cfg.beta_intero * loss_intero
                loss_homeo = self.hbp.homeostatic_loss()
                loss_dict["homeo"] = loss_homeo
                total = total + self.cfg.beta_homeo * loss_homeo
                loss_stab = self.hbp.stability_penalty()
                loss_dict["stab"] = loss_stab
                total = total + self.cfg.beta_stab * loss_stab

            # --- N2: L_val — aprender a predecir las CONSECUENCIAS propias
            # (payoff realizado). Solo toca las cabezas V̂ (input detached;
            # test de cableado: ∂L_val/∂backbone = 0 y ∂CE/∂ψ = 0). ---
            if (self.cfg.value_mode not in ("off", "oracle")
                    and self._val_cache is not None):
                st = (self.value_ctx or {}).get("stake_true")
                if st is not None:
                    valid = (targets != -1)
                    has = valid.any(dim=1)
                    idx_last = valid.float().cumsum(dim=1).argmax(dim=1)
                    rows = torch.arange(B, device=device)
                    with torch.no_grad():
                        pred = logits[rows, idx_last].argmax(-1)
                        correct_b = (pred == targets[rows, idx_last]).float()
                    st_f = st.to(device=device, dtype=torch.float32)
                    payoff = (st_f * correct_b)[has]
                    p_hat = self._val_cache["p"][has].clamp(1e-4, 1 - 1e-4)
                    pay_hat = self._val_cache["pay"][has]
                    l_p = F.binary_cross_entropy(p_hat, correct_b[has])
                    # regresión del payoff CRUDO (VG4-fix): target observable
                    loss_val = l_p + F.mse_loss(pay_hat, payoff)
                    loss_dict["val"] = loss_val
                    total = total + self.cfg.beta_val * loss_val.to(total.dtype)

            loss_dict["total"] = total

        return logits, loss_dict

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
