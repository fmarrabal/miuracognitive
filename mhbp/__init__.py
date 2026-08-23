"""
mHBP — Multiscale Homeostatic Background Processor.

Plano de control autonómico multiescala: Q campos homeostáticos acoplados con
escalas temporales ordenadas, integrador Cayley-IMEX certificado y actuadores
acotados. Especificación matemática: MATH_SPEC.md. Fase 1 del plan maestro.
"""
from .field import FieldConfig, HomeostaticField
from .coupled_fields import (MHBPConfig, CoupledMultiscaleHBP, Timescales,
                             InterfaceCoupling, default_four_fields)
from .interoception import InteroceptiveSignal, InteroceptionEncoder, CHANNELS
from .actuators import ActuatorHead, ActuatorSpec, MVP_ACTUATORS
from .allostasis import Allostasis
from .integrators.cayley_imex import CayleyIMEX
from .integrators.verlet import VerletRef

__all__ = [
    "FieldConfig", "HomeostaticField", "MHBPConfig", "CoupledMultiscaleHBP",
    "Timescales", "InterfaceCoupling", "default_four_fields",
    "InteroceptiveSignal", "InteroceptionEncoder", "CHANNELS",
    "ActuatorHead", "ActuatorSpec", "MVP_ACTUATORS", "Allostasis",
    "CayleyIMEX", "VerletRef",
]
