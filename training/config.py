"""
MiuraCognitive - Configuración de entrenamiento
================================================
Configuración centralizada de los experimentos del Sprint, incluyendo las
4 configuraciones de ablation (vanilla, gating simple, HBP 1er orden, HBP 2o orden).

Autor: Francisco M. Arrabal (Curro) + asistencia Claude
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Datos
    task: str = "runsum"           # "runsum" (RUNSUM-MOD) | "recall" (Tarea-B original)
    batch_size: int = 32
    seq_len: int = 128
    max_distractors: int = 60
    n_values: int = 50
    n_distractors: int = 100
    # RUNSUM-MOD / permcomp (eje de dificultad = K)
    mod: int = 3                   # módulo de la suma corriente (chance = 1/mod)
    min_writes: int = 2
    max_writes: int = 16           # rango de EVALUACIÓN (incluye extrapolación)
    train_max_writes: int = 16     # rango de ENTRENAMIENTO (si < max_writes -> test de extrapolación)
    perm_gens: str = "adjacent"    # permcomp: set de generadores (adjacent | cycle_transp)
    # Optimización
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 5000
    grad_clip: float = 1.0
    # Logging / checkpoints
    log_every: int = 50
    eval_every: int = 500
    ckpt_every: int = 1000
    out_dir: str = "checkpoints"
    use_wandb: bool = False
    wandb_project: str = "miuracognitive"
    # Modelo (qué variante)
    variant: str = "hbp_full"  # uno de: vanilla, gating, gating_wm, hbp_first, hbp_full
    halt_mod_gain: float = 2.0  # fuerza del sesgo del VEI sobre el halting
    max_halt_steps: int = 10   # iteraciones máx. del reasoner (presupuesto de pensamiento)
    wm_in_loop: bool = True    # WM dentro del bucle del reasoner (False = pre-bucle)
    ponder_expected_loss: bool = False  # E_p[CE_n] (opt-in; histórico=False)
    seed: int = 0
