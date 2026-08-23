# AHA-1 — resultado confirmatorio

Estado: **PASS**.

Se ejecutaron 20 seeds nuevas y pareadas (`100..119`), 960 episodios por
condición y seed, con `gating_wm` y `reactive` compartiendo el mismo predictor
congelado. El piloto no se incorporó al análisis.

| criterio primario | efecto medio | dirección | p Holm |
|---|---:|---:|---:|
| supervivencia `gating_wm − reactive` | +0,72396 | 20/20 | 7,63e-6 |
| iniciativa vacía `gating_wm − reactive` | +0,73055 | 20/20 | 7,63e-6 |
| lesión predictiva sobre supervivencia | +0,95479 | 20/20 | 7,63e-6 |
| cue válido sobre supervivencia | +0,84677 | 20/20 | 7,63e-6 |

La regla congelada exigía que los cuatro efectos fueran positivos y que todos
los tests exactos bilaterales por seed pasaran Holm a `alpha=0,05`. Se cumplen
las cuatro condiciones.

| media | `gating_wm` | `reactive` |
|---|---:|---:|
| supervivencia | 0,95495 | 0,23099 |
| violación | 0,00060 | 0,02427 |
| anticipación correcta | 0,96541 | 0,17551 |
| iniciativa vacía | 0,89301 | 0,16246 |
| tasa de acción | 0,20114 | 0,22079 |
| acciones falsas | 0,41645 | 0,91722 |

Controles causales de `gating_wm`:

- supervivencia con lesión predictiva: `0,00016`;
- supervivencia con cues barajados: `0,10818`;
- predictor compartido: correlación con daño futuro `0,94505`;
- acción en mundo estacionario: `0,02375`.

El trasplante aumenta en `0,20295` la probabilidad de actuar sobre la necesidad
deficitaria, aunque no cambia el `argmax` exactamente en el tick 0. Una auditoría
post hoc de latencia encontró respuesta correcta antes del tick 6 en 56/60
combinaciones seed-necesidad y 0/60 bajo saciedad. No forma parte de la decisión
confirmatoria.

Interpretación: queda confirmada una forma operacional de **autorregulación
anticipatoria causal**. El sistema predice un déficit y actúa antes de que se
materialice; el comportamiento desaparece al cortar la predicción o invalidar
su cue. Esto no demuestra metas abiertas, consciencia ni «alma».

Archivos:

- `PROTOCOL.md`: hipótesis y regla congeladas antes de la ejecución;
- `summary.json`: agregado completo y tests exactos;
- `transplant_latency_audit.json`: auditoría post hoc reproducible;
- `{variant}_seed{100..119}.json`: evaluaciones por seed;
- `checkpoints/`: estados entrenados reproducibles;
- `shard_*.log`: trazabilidad de ejecución.
