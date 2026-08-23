# mHBP — Multiscale Homeostatic Background Processor

> **Estado: Fase 1 completada; Fase 2 ejecutada y REFUTADA.**
> Ver `analysis/FINDINGS_PHASE2.md`.

Plano de control **autonómico multiescala** para modelos recurrentes/LLM/agentes:
Q campos homeostáticos acoplados, con escalas temporales ordenadas por
construcción, dinámica de segundo orden certificada y actuadores acotados.
Extensión del HBP de MiuraCognitive (un campo → cuatro campos especializados).
Los términos biológicos son analogías funcionales, no atribuciones mentales.

**Especificación matemática**: [MATH_SPEC.md](MATH_SPEC.md) — el código la
implementa literalmente y `tests/` la verifica (criterios §9).

## Estado: Fase 1 completada

- 4 campos (`fast_executive` τ=1, `risk_priority` τ=4, `slow_deliberative` τ=8,
  `resource_metabolic` τ=32), grafos propios, d=8.
- Acoplamiento por **potencial de interfaz** (PSD por construcción para
  cualesquiera pesos aprendidos) — jerarquía direccional vía **alostasis acotada**.
- Integrador **Cayley-IMEX**: rotación giroscópica exactamente ortogonal
  (elimina la inyección de energía discreta del Verlet del HBP original) +
  θ-implícito global SPD con el acoplamiento dentro del núcleo.
- Certificados: estructural (K≻0, C≻0, G antisim, 𝓑⪰0, τ ordenadas) +
  **ρ(Φ) exacto por sondeo** del tick completo + disipación de energía.
- Interocepción (26 canales, normalización running, máscara, anti-leakage) y
  4 actuadores MVP acotados con intervención causal (freeze/override).

## Uso rápido

```bash
# tests (90)          — desde la raíz del repo, entorno `implanto`
PYTHONPATH=. python -m pytest mhbp/tests -q
# demo con figura
PYTHONPATH=. python -m mhbp.examples.minimal_demo
```

```python
from mhbp import MHBPConfig, CoupledMultiscaleHBP, InteroceptiveSignal
m = CoupledMultiscaleHBP(MHBPConfig()).double()
m.reset_state(batch=2)
acts, info = m.tick(InteroceptiveSignal(values={"entropy": 1.2, "token_cost": 0.5}))
# acts: halt_bias, depth_max, tool_gate, budget_scale  (acotados por construcción)
from mhbp.stability.certificates import stability_report
print(stability_report(m))   # ρ(Φ), energía, estructura
```

## Estructura

```
mhbp/
├── MATH_SPEC.md            especificación cerrada + proposiciones 1-3
├── operators.py            grafos (chain/ring/star/complete), K/C/G, cajas seguras
├── field.py                HomeostaticField (estado u,w; energía; diagnóstico)
├── coupled_fields.py       CoupledMultiscaleHBP + Timescales + InterfaceCoupling
├── interoception.py        InteroceptiveSignal (26 canales) + encoder acotado
├── actuators.py            ActuatorHead (Π_A, STE, saturación, intervenciones)
├── allostasis.py           Ψ acotada: campos lentos → setpoint del rápido
├── integrators/
│   ├── cayley_imex.py      principal (Strang: Cayley ∘ θ-implícito ∘ Cayley)
│   └── verlet.py           referencia (compatibilidad HBP; condicional)
├── stability/certificates.py  estructural + ρ(Φ) exacto + disipación
├── tests/                  90 tests (criterios de aceptación de MATH_SPEC §9)
├── examples/minimal_demo.py   impulso multiescala + certificados + figura
└── configs/mhbp_four_fields.yaml
```

## Fases siguientes (plan maestro §31)

2. ~~Tareas sintéticas multiescala + baselines MLP/GRU/AR(2)/HBP-único.~~
   **EJECUTADA Y REFUTADA** (124/124 pre-registrado, ver
   [analysis/FINDINGS_PHASE2.md](analysis/FINDINGS_PHASE2.md)): el mHBP sale
   PEOR que el GRU (dz=4.06) y el 2º orden peor que el 1º. Falla solo en
   cambio de presupuesto, por dos vías separadas — el readout de plan vía
   campo no extrapola nivel, y la inercia re-adapta lento tras el escalón.
   El acoplamiento y la alostasis resultan irrelevantes. Es un negativo
   informativo **con mecanismo**: «modula, no piensa» aplica también al
   plano de control.
3. Integración con el reasoner de MiuraCognitive (halting, WM, routing).
4. Recursos (tokens/FLOPs/latencia/precisión) + multiplicadores duales (§12).
5-9. RAG/herramientas, alostasis plena, benchmarks, estadística, paper.
