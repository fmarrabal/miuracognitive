"""
MiuraCognitive - Trainer (Sprint 1-4)
======================================
Bucle de entrenamiento para la Tarea-B (synthetic recall). Soporta las 4
variantes de ablation y registra métricas estratificadas por dificultad.

Uso:
    python -m training.trainer --variant hbp_full --max_steps 3000

Variantes:
  vanilla   : Transformer base sin HBP, sin recurrencia, sin WM.
  gating    : + gating simple por capa (sin dinámica temporal).
  hbp_first : + HBP de PRIMER orden (disipativo, ζ→∞ efectivo).
  hbp_full  : + HBP de SEGUNDO orden (onda amortiguada sobre grafo) + WM.

Autor: Francisco M. Arrabal (Curro) + asistencia Claude
"""

from __future__ import annotations
import argparse
import json
import math
import os
import time
import torch
import torch.nn.functional as F

from model.transformer import MiuraConfig
from model.miura import MiuraCognitiveFull, MiuraFullConfig
from model.hbp import HBPConfig
from data.synthetic_recall import (SyntheticRecallDataset, RunningSumModDataset,
                                    PermutationCompositionDataset, IteratedFunctionDataset)
from training.config import TrainConfig


def build_dataset(cfg: TrainConfig, max_writes_override: int | None = None):
    """Factory de la tarea según cfg.task. max_writes_override permite usar un
    rango de K distinto para train (cfg.train_max_writes) vs eval (cfg.max_writes)."""
    mw = max_writes_override if max_writes_override is not None else cfg.max_writes
    if cfg.task == "runsum":
        return RunningSumModDataset(mod=cfg.mod, n_distractors=cfg.n_distractors,
                                    min_writes=cfg.min_writes, max_writes=mw,
                                    seq_len=cfg.seq_len, seed=cfg.seed)
    elif cfg.task == "permcomp":
        return PermutationCompositionDataset(min_ops=cfg.min_writes, max_ops=mw,
                                             seq_len=cfg.seq_len, seed=cfg.seed,
                                             gen_set=cfg.perm_gens)
    elif cfg.task == "iterfunc":
        return IteratedFunctionDataset(min_ops=cfg.min_writes, max_ops=mw,
                                       seq_len=cfg.seq_len, seed=cfg.seed)
    elif cfg.task == "recall":
        return SyntheticRecallDataset(n_values=cfg.n_values, n_distractors=cfg.n_distractors,
                                      max_distractors=cfg.max_distractors,
                                      seq_len=cfg.seq_len, seed=cfg.seed)
    raise ValueError(f"Tarea desconocida: {cfg.task}")


def build_model(variant: str, vocab_size: int, seq_len: int,
                halt_mod_gain: float = 2.0, max_halt_steps: int = 10,
                wm_in_loop: bool = True,
                hbp_overrides: dict | None = None,
                ponder_expected_loss: bool = False) -> MiuraCognitiveFull:
    """Construye el modelo según la variante de ablation.

    El contraste 1er/2º orden es ARQUITECTÓNICO (HBPConfig.order), no un ζ
    congelado: hbp_first usa el límite sobreamortiguado (order=1, ζ≥2, c≤0.4);
    hbp_full usa el Verlet de 2º orden (order=2, ζ≥0.05, c≤0.7).

    hbp_overrides: dict opcional de campos de HBPConfig a sobrescribir DESPUÉS de
    fijar la variante (para barridos: gate_init_scale, D_max, b_adv_max, ...).
    """
    tcfg = MiuraConfig(vocab_size=vocab_size, d_model=256, n_layers=4,
                       n_heads=4, d_ff=1024, max_seq_len=seq_len)
    hcfg = HBPConfig()   # n_nodes lo fija MiuraFullConfig.__post_init__ (= n_layers+2)

    if variant == "vanilla":
        cfg = MiuraFullConfig(transformer=tcfg, hbp=hcfg,
                              use_hbp=False, use_adaptive_depth=False,
                              use_working_memory=False)
    elif variant == "gating":
        # Gating = recurrencia adaptativa sin HBP.
        cfg = MiuraFullConfig(transformer=tcfg, hbp=hcfg,
                              use_hbp=False, use_adaptive_depth=True,
                              use_working_memory=False)
    elif variant == "gating_wm":
        # CONTROL: recurrencia adaptativa + working memory, SIN HBP.
        # Aísla la contribución de la WM frente a la del campo homeostático.
        cfg = MiuraFullConfig(transformer=tcfg, hbp=hcfg,
                              use_hbp=False, use_adaptive_depth=True,
                              use_working_memory=True)
    elif variant == "hbp_first":
        # 1er orden: límite sobreamortiguado de la misma ecuación.
        hcfg.order = 1
        hcfg.zeta_init = 2.5
        hcfg.zeta_min = 2.0     # ζ alto -> inercia despreciable (sin oscilación)
        hcfg.c_init = 0.3
        hcfg.c_max = 0.4        # acoplamiento limitado (suave)
        # ω₀ debe respetar el certificado con ζ alto (ζω₀Δt<1 -> ω₀<1/ζ≈0.4).
        hcfg.omega0_init = 0.3
        hcfg.omega0_max = 0.45
        cfg = MiuraFullConfig(transformer=tcfg, hbp=hcfg,
                              use_hbp=True, use_adaptive_depth=True,
                              use_working_memory=True)
    elif variant == "hbp_first_eq":
        # v4 (des-confusión del ORDEN, pedida por revisores): 1er orden con
        # TOPES EQUIPARADOS a hbp_full (c_max=0.7, ω₀ por defecto hasta 1.8).
        # El hbp_first original recortaba topes por la restricción del Euler
        # explícito acoplado a ζ (ζω₀Δt<1); el solver IMPLÍCITO con tasa
        # propia γ_diff es incondicionalmente estable y los libera. Solo el
        # ORDEN (y su ζ de régimen sobreamortiguado) difiere de hbp_full.
        hcfg.order = 1
        hcfg.zeta_init = 2.5
        hcfg.zeta_min = 2.0
        hcfg.c_init = 0.4
        hcfg.c_max = 0.7
        hcfg.diff_solver = "implicit"
        cfg = MiuraFullConfig(transformer=tcfg, hbp=hcfg,
                              use_hbp=True, use_adaptive_depth=True,
                              use_working_memory=True)
    elif variant == "hbp_gru":
        # v4 (baseline aprendido de interfaz equiparada, pedido por
        # revisores): GRUCell por nodo sustituye SOLO el integrador físico;
        # interocepción, forzamiento y cabezas de modulación idénticos.
        hcfg.core = "gru"
        cfg = MiuraFullConfig(transformer=tcfg, hbp=hcfg,
                              use_hbp=True, use_adaptive_depth=True,
                              use_working_memory=True)
    elif variant in ("hbp_full", "hbp_fullR"):
        # 2º orden: Verlet completo (inercia/oscilación). _R = incumbente
        # FRESCO (re-estimación del toque, PREREG_F3A_R enmienda 1).
        hcfg.order = 2
        hcfg.zeta_init = 0.5
        hcfg.zeta_min = 0.05
        hcfg.c_init = 0.4
        hcfg.c_max = 0.7
        cfg = MiuraFullConfig(transformer=tcfg, hbp=hcfg,
                              use_hbp=True, use_adaptive_depth=True,
                              use_working_memory=True)
    elif variant in ("n2_endo", "n2_endo_noval", "n2_oracle"):
        # N2 (PREREG_N2 v2): plano mHBP oficial + módulo de valor; los brazos
        # blind_* reutilizan n2_endo (solo difieren los PESOS de CE en el
        # runner). El adaptador lleva el canal de valor con veto por-cabeza.
        cfg = MiuraFullConfig(transformer=tcfg, hbp=hcfg,
                              use_hbp=True, use_adaptive_depth=True,
                              use_working_memory=True)
        cfg.value_mode = {"n2_endo": "endo", "n2_endo_noval": "endo_uncoupled",
                          "n2_oracle": "oracle"}[variant]
    elif variant == "n2_gating_endo":
        # N2: vía espejo del valor al halting, SIN plano (¿hace falta el plano?)
        cfg = MiuraFullConfig(transformer=tcfg, hbp=hcfg,
                              use_hbp=False, use_adaptive_depth=True,
                              use_working_memory=True)
        cfg.value_mode = "gating_endo"
    elif variant in ("miura_mhbp", "miura_mhbp_pi1", "miura_mhbp_noc",
                     "miura_mhbp_ptR"):
        # Fase 3a (PREREG_F3A): el plano mHBP de 4 campos certificado sustituye
        # al campo único. Config estructural idéntica a hbp_full; el módulo hbp
        # se reemplaza por el adaptador tras construir el modelo (abajo).
        # _pi1/_noc = brazos F3a-R (PREREG_F3A_R): mismo plano, vías de
        # contenido congeladas tras tick 1 / lesionadas en train.
        cfg = MiuraFullConfig(transformer=tcfg, hbp=hcfg,
                              use_hbp=True, use_adaptive_depth=True,
                              use_working_memory=True)
    elif variant.startswith("f3b_"):
        # Fase 3b (PREREG_F3B v2 §4): brazos de SESIÓN. Config estructural
        # idéntica; session_mode activa las observaciones §3 y el presupuesto
        # por-fila; el controlador se instala tras construir (abajo).
        use_ctrl = variant != "f3b_gating_wm"
        cfg = MiuraFullConfig(transformer=tcfg, hbp=hcfg,
                              use_hbp=use_ctrl, use_adaptive_depth=True,
                              use_working_memory=True)
        cfg.session_mode = True
        if variant in ("f3b_mhbp_sess", "f3b_mhbp_sess_inst", "f3b_mhbp_noper"):
            # §6.5: β_homeo=0 en los brazos mhbp (mean u² castiga la memoria
            # inter-instancia que D2 contrasta; estabilidad por construcción)
            cfg.beta_homeo = 0.0
    elif variant == "hbp_elliptic":
        # Acoplamiento ELÍPTICO NO-LOCAL (propuesta de Curro): onda de 2º orden
        # idéntica a hbp_full, pero la MODULACIÓN lee ψ=L⁺(h−h*) -> cada nodo
        # modula según TODO el campo. Aísla la NO-LOCALIDAD como única variable
        # (dinámica y certificado sin cambios). Test: ¿la no-localidad es
        # load-bearing donde la física LOCAL fue neutra (mecanismo-null)?
        hcfg.order = 2
        hcfg.zeta_init = 0.5
        hcfg.zeta_min = 0.05
        hcfg.c_init = 0.4
        hcfg.c_max = 0.7
        hcfg.elliptic_readout = True
        cfg = MiuraFullConfig(transformer=tcfg, hbp=hcfg,
                              use_hbp=True, use_adaptive_depth=True,
                              use_working_memory=True)
    elif variant == "hbp_kdv":
        # Homeostasis KdV (zoo de físicas, propuesta de Curro): onda de 2º orden
        # + DISPERSIÓN β·A³·u (tercera derivada espacial orientada, conservativa)
        # + ADVECCIÓN NO LINEAL ν·tanh(u)⊙(A·u) (el u·u_x de KdV, acotado BIBS).
        # Eje de test: métricas de CONTROLADOR de cómputo (no accuracy; el
        # mecanismo-null muestra que el régimen no mueve accuracy).
        hcfg.order = 2
        hcfg.zeta_init = 0.5
        hcfg.zeta_min = 0.05
        hcfg.c_init = 0.4
        hcfg.c_max = 0.7
        hcfg.kdv_beta_max = 0.1   # (β·ρ(A)³)²≈0.34 en Ω²: dentro del CFL con holgura al init
        hcfg.kdv_nl_max = 0.3
        cfg = MiuraFullConfig(transformer=tcfg, hbp=hcfg,
                              use_hbp=True, use_adaptive_depth=True,
                              use_working_memory=True)
    elif variant == "hbp_mix":
        # Familia PDE con física GATEADA por la interocepción: onda base (2º orden)
        # + difusión estructural D + advección b, coeficientes elegidos por tick.
        hcfg.order = 2
        hcfg.zeta_init = 0.5
        hcfg.zeta_min = 0.05
        hcfg.c_init = 0.4
        hcfg.c_max = 0.7
        hcfg.D_max = 0.4          # difusión (D·Δt·ρ(L)=1.49 < 2, estable)
        hcfg.b_adv_max = 0.4      # advección direccional (antisimétrica)
        hcfg.gate_physics = True
        hcfg.gate_init_scale = 1.0  # con phys_norm, gradiente no-hambriento a las cabezas
        cfg = MiuraFullConfig(transformer=tcfg, hbp=hcfg,
                              use_hbp=True, use_adaptive_depth=True,
                              use_working_memory=True)
    else:
        raise ValueError(f"Variante desconocida: {variant}")

    if hbp_overrides:
        for k, v in hbp_overrides.items():
            if not hasattr(hcfg, k):
                raise ValueError(f"HBPConfig no tiene el campo '{k}'")
            setattr(hcfg, k, v)

    cfg.halt_mod_gain = halt_mod_gain
    cfg.max_halt_steps = max_halt_steps
    cfg.wm_in_loop = wm_in_loop
    cfg.ponder_expected_loss = ponder_expected_loss
    model = MiuraCognitiveFull(cfg)
    if variant in ("n2_endo", "n2_endo_noval", "n2_oracle"):
        from mhbp.adapters.reasoner_adapter import MhbpReasonerAdapter
        model.hbp = MhbpReasonerAdapter(n_nodes=cfg.hbp.n_nodes,
                                        d_h=cfg.hbp.d_h,
                                        d_intero=cfg.hbp.d_intero,
                                        value_channel=True)
    elif variant in ("miura_mhbp", "miura_mhbp_pi1", "miura_mhbp_noc",
                     "miura_mhbp_ptR"):
        # _ptR = control por-tick RE-ENTRENADO con el código actual (robustez
        # pre-declarada de PREREG_F3A_R: comparador fresco, cero flags)
        from mhbp.adapters.reasoner_adapter import MhbpReasonerAdapter
        model.hbp = MhbpReasonerAdapter(n_nodes=cfg.hbp.n_nodes,
                                        d_h=cfg.hbp.d_h,
                                        d_intero=cfg.hbp.d_intero)
        if variant == "miura_mhbp_pi1":
            # F3a-R: contenido congelado tras el tick 1 (PREREG_F3A_R)
            model.hbp.freeze_content_after_tick1 = True
        elif variant == "miura_mhbp_noc":
            # F3a-R: entrenar SIN vías de contenido (lesión en TRAIN)
            model.hbp.lesion = {"wm", "gate"}
    elif variant.startswith("f3b_") and variant != "f3b_gating_wm":
        from mhbp.adapters.reasoner_adapter import MhbpReasonerAdapter
        from mhbp.adapters.session_controllers import (
            SingleFieldSessionAdapter, GruSessionAdapter, ReactSessionAdapter)
        kw = dict(n_nodes=cfg.hbp.n_nodes, d_h=cfg.hbp.d_h,
                  d_intero=cfg.hbp.d_intero)
        if variant in ("f3b_mhbp_sess", "f3b_mhbp_noper"):
            model.hbp = MhbpReasonerAdapter(taus=(1.0, 3.0, 10.0, 32.0), **kw)
        elif variant == "f3b_mhbp_sess_inst":
            model.hbp = MhbpReasonerAdapter(taus=(1.0, 3.0, 10.0, 32.0),
                                            per_instance_gates=True, **kw)
        elif variant == "f3b_hbp_sess":
            model.hbp = SingleFieldSessionAdapter(**kw)
        elif variant == "f3b_gru_sess":
            # H igualado a la dimensión de ESTADO persistente del plano (§6.5)
            tmp = MhbpReasonerAdapter(taus=(1.0, 3.0, 10.0, 32.0), **kw)
            tmp.plane.reset_state(1)
            hidden = sum(f.u.numel() + f.w.numel() for f in tmp.plane.fields)
            del tmp
            model.hbp = GruSessionAdapter(hidden=hidden, **kw)
        elif variant == "f3b_react_sess":
            model.hbp = ReactSessionAdapter(**kw)
        else:
            raise ValueError(f"Variante F3b desconocida: {variant}")
    return model


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Schedule: warmup lineal + decaimiento cosine."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return 0.5 * cfg.lr * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model, ds, device, dtype, n_batches: int = 20):
    """Evalúa accuracy estratificada por dificultad. El argmax se RESTRINGE al
    sub-vocabulario de respuesta del dataset (mide seguimiento de estado, no
    robustez de formato: sin la máscara, masa en tokens de operador puntuaría 0
    aunque el modelo 'supiera' la respuesta dentro del rango válido)."""
    model.eval()
    buckets = {"corto": [0, 0], "medio": [0, 0], "largo": [0, 0]}
    a0, a1 = ds.answer_offset, ds.answer_offset + ds.answer_size
    for _ in range(n_batches):
        inp, tgt, nd = ds.batch(32)
        inp, tgt = inp.to(device), tgt.to(device)
        if model.cfg.use_hbp:
            model.hbp.reset_state(device=device, dtype=dtype)
        logits, _ = model(inp, None)
        preds = a0 + logits[..., a0:a1].argmax(dim=-1)
        for b in range(inp.size(0)):
            pos = (tgt[b] != -1).nonzero(as_tuple=True)[0]
            if len(pos) == 0:
                continue
            p = pos[-1].item()    # ÚLTIMA posición supervisada = respuesta final
            correct = (preds[b, p].item() == tgt[b, p].item())
            bucket = ds.bucket(nd[b].item())
            buckets[bucket][0] += int(correct)
            buckets[bucket][1] += 1
    model.train()
    return {k: (v[0] / v[1] if v[1] > 0 else 0.0) for k, v in buckets.items()}


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean()
                 / (a.std(unbiased=False) * b.std(unbiased=False)))


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    return _pearson(ra, rb)


@torch.no_grad()
def compute_diagnostics(model, ds, device, dtype, n_batches: int = 40, ood_k_min: int = 14):
    """Diagnóstico de CÓMPUTO. Además de la corr global (mezcla in-dist+OOD),
    reporta la adaptividad RESTRINGIDA al rango OOD puro (K>=ood_k_min), la
    Spearman (robusta a saturación del techo N_max) y la curva E[n_iter|K]."""
    model.eval()
    a0, a1 = ds.answer_offset, ds.answer_offset + ds.answer_size
    Ks, NIs, Cs = [], [], []
    for _ in range(n_batches):
        inp, tgt, nd = ds.batch(32)
        inp, tgt = inp.to(device), tgt.to(device)
        logits, _ = model(inp, None)               # el forward resetea el HBP
        ne = model._last_n_expected
        if ne is None:
            model.train()
            return None
        preds = a0 + logits[..., a0:a1].argmax(dim=-1)
        for b in range(inp.size(0)):
            pos = (tgt[b] != -1).nonzero(as_tuple=True)[0]
            if len(pos) == 0:
                continue
            p = pos[-1].item()
            Ks.append(float(nd[b].item()))
            NIs.append(float(ne[b].item()))
            Cs.append(float(preds[b, p].item() == tgt[b, p].item()))
    model.train()
    K, NI, C = torch.tensor(Ks), torch.tensor(NIs), torch.tensor(Cs)
    out = {"mean_niter": float(NI.mean()), "n": int(len(Ks))}
    for name in ("corto", "medio", "largo"):
        mask = torch.tensor([ds.bucket(int(k)) == name for k in Ks])
        if mask.sum() == 0:
            continue
        a, ni = float(C[mask].mean()), float(NI[mask].mean())
        out[name] = {"acc": a, "mean_niter": ni,
                     "acc_per_niter": (a / ni if ni > 0 else 0.0),
                     "n": int(mask.sum())}
    out["corr_K_niter"] = _pearson(K, NI)
    out["spearman_K_niter"] = _spearman(K, NI)
    # Adaptividad OOD PURA: solo K>=ood_k_min (evita que la corr se venda como
    # 'extrapolación de cómputo' cuando la impulsa el tramo in-dist).
    m = K >= ood_k_min
    if m.sum() >= 10:
        out["corr_K_niter_ood"] = _pearson(K[m], NI[m])
        out["spearman_K_niter_ood"] = _spearman(K[m], NI[m])
    # Curva E[n_iter | K] (para la figura y para ver saturación del techo)
    curve = {}
    for k in sorted(set(int(x) for x in Ks)):
        mk = K == k
        curve[str(k)] = {"niter": float(NI[mk].mean()), "acc": float(C[mk].mean()),
                         "n": int(mk.sum())}
    out["niter_by_K"] = curve
    return out


def select_device_dtype():
    """CUDA->BF16, CPU->FP32 con aviso."""
    if torch.cuda.is_available():
        return torch.device("cuda:0"), torch.bfloat16
    print("[INFO] Sin GPU: entrenando en CPU + FP32 (lento, solo validación).")
    return torch.device("cpu"), torch.float32


def train_run(cfg: TrainConfig, results_dir: str | None = None,
              save_ckpt: bool = False, verbose: bool = True) -> dict:
    """Ejecuta UN entrenamiento completo y devuelve un dict de resultados.

    Reutilizable desde el orquestador de ablations. Si results_dir se da,
    guarda un JSON con la traza completa (curvas de loss/acc + diagnóstico HBP).
    """
    torch.manual_seed(cfg.seed)
    device, dtype = select_device_dtype()

    ds = build_dataset(cfg, max_writes_override=cfg.train_max_writes)   # rango de ENTRENAMIENTO
    # EVAL con SEED DISJUNTA: con la misma seed, en régimen in-dist los streams
    # de train y eval son idénticos (contaminación literal). El pareado entre
    # variantes se conserva (misma seed de eval para todas).
    eval_cfg = TrainConfig(**{**cfg.__dict__, "seed": cfg.seed + 100_000})
    ds_eval = build_dataset(eval_cfg)                                   # rango de EVAL (extrapolación)
    model = build_model(cfg.variant, ds.vocab_size, cfg.seq_len,
                        halt_mod_gain=cfg.halt_mod_gain,
                        max_halt_steps=cfg.max_halt_steps,
                        wm_in_loop=cfg.wm_in_loop,
                        ponder_expected_loss=cfg.ponder_expected_loss
                        ).to(device=device, dtype=dtype)
    if model.cfg.use_hbp:
        # FIX BF16: los raw físicos (|v|~0.3-1.6) tienen ULP/2 > paso de Adam en
        # BF16 -> quedaban CONGELADOS en su init en GPU. Se re-anclan a FP32.
        model.hbp.pin_fp32()
    if verbose:
        print(f"Variante: {cfg.variant} (seed {cfg.seed}) | Parámetros: {model.num_parameters():,}")

    # Probe para excluir parámetros sin gradiente (ramas inactivas en este Sprint)
    probe_in, probe_tgt, _ = ds.batch(2)
    probe_in, probe_tgt = probe_in.to(device), probe_tgt.to(device)
    if model.cfg.use_hbp:
        model.hbp.reset_state(device=device, dtype=dtype)
    _, probe_losses = model(probe_in, probe_tgt)
    probe_losses["total"].backward()
    active = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    model.zero_grad()
    opt = torch.optim.AdamW(active, lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    # Trazas para el JSON / figuras
    trace = {"step": [], "loss": [], "vei_var": []}
    eval_curve = {"step": [], "corto": [], "medio": [], "largo": []}

    model.train()
    t0 = time.time()
    for step in range(1, cfg.max_steps + 1):
        lr = get_lr(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        inp, tgt, _ = ds.batch(cfg.batch_size)
        inp, tgt = inp.to(device), tgt.to(device)
        if model.cfg.use_hbp:
            model.hbp.reset_state(device=device, dtype=dtype)

        logits, loss_dict = model(inp, tgt)
        loss = loss_dict["total"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(active, cfg.grad_clip)
        opt.step()

        if step % cfg.log_every == 0:
            vei_var = model.hbp.state_summary()["vei_variance"] if model.cfg.use_hbp else 0.0
            trace["step"].append(step)
            trace["loss"].append(float(loss.item()))
            trace["vei_var"].append(float(vei_var))
            if verbose:
                msg = f"step {step:5d} | lr {lr:.2e} | loss {loss.item():.4f}"
                if model.cfg.use_hbp:
                    msg += f" | VEI_var {vei_var:.4f}"
                print(msg)

        if step % cfg.eval_every == 0:
            acc = evaluate(model, ds_eval, device, dtype)
            eval_curve["step"].append(step)
            for k in ("corto", "medio", "largo"):
                eval_curve[k].append(acc[k])
            if verbose:
                print(f"  [eval] corto={acc['corto']:.3f} medio={acc['medio']:.3f} "
                      f"largo={acc['largo']:.3f}")

    # Evaluación final robusta (rango de extrapolación si train_max_writes < max_writes)
    acc_final = evaluate(model, ds_eval, device, dtype, n_batches=50)
    elapsed = time.time() - t0
    if verbose:
        print(f"[FINAL {cfg.variant} s{cfg.seed}] corto={acc_final['corto']:.3f} "
              f"medio={acc_final['medio']:.3f} largo={acc_final['largo']:.3f} "
              f"({elapsed:.0f}s)")

    # Diagnóstico del HBP (si aplica). El adaptador mHBP no habla la interfaz
    # de diagnostics (crash LATENTE post-entrenamiento — panel F3a): usa el
    # diagnóstico propio del plano.
    hbp_diag = None
    if model.cfg.use_hbp:
        if getattr(model.hbp, "is_mhbp", False):
            hbp_diag = model.hbp.plane.diagnostics()
        else:
            from eval.diagnostics import run_full_diagnostics
            hbp_diag = run_full_diagnostics(model.hbp, n_ticks=200, device=device, dtype=dtype)

    # Diagnóstico de cómputo (variantes con reasoner adaptativo), en rango de eval
    compute_diag = compute_diagnostics(model, ds_eval, device, dtype) if model.cfg.use_adaptive_depth else None

    results = {
        "variant": cfg.variant,
        "seed": cfg.seed,
        "max_steps": cfg.max_steps,
        "cfg": dict(cfg.__dict__),                     # procedencia: config completa
        "env": {"torch": torch.__version__,
                "device_name": (torch.cuda.get_device_name(0)
                                 if torch.cuda.is_available() else "cpu")},
        "n_params": model.num_parameters(),
        "final_acc": acc_final,
        "trace": trace,
        "eval_curve": eval_curve,
        "hbp_diagnostics": hbp_diag,
        "compute_diag": compute_diag,
        "elapsed_s": elapsed,
        "device": device.type,
    }

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, f"{cfg.variant}_seed{cfg.seed}.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"  [results] guardado en {path}")

    if save_ckpt:
        os.makedirs(cfg.out_dir, exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__},
                   os.path.join(cfg.out_dir, f"{cfg.variant}_seed{cfg.seed}_final.pt"))

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="hbp_full",
                    choices=["vanilla", "gating", "hbp_first", "hbp_full"])
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--save_ckpt", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig(variant=args.variant, max_steps=args.max_steps, seed=args.seed)
    train_run(cfg, results_dir=args.results_dir, save_ckpt=args.save_ckpt)


if __name__ == "__main__":
    main()
