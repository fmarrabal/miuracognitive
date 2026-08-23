# Fase 4 — resultado confirmatorio

Estado: **PASS**.

| familia primaria | efecto medio | semillas a favor | p Holm |
|---|---:|---:|---:|
| automodelo − modelo congelado | +0,40563 | 20/20 | 7,63e-6 |
| normal − lesión de actualización | +0,40563 | 20/20 | 7,63e-6 |
| normal − copia eferente barajada | +0,40563 | 20/20 | 7,63e-6 |
| normal − contenido rotado | +0,36461 | 20/20 | 7,63e-6 |

| media | `self_model` | `stale_model` |
|---|---:|---:|
| recuperación predaño | 1,00000 | 1,00000 |
| recuperación postdaño | 1,00000 | 0,59438 |
| latencia de recuperación | 2,00000 | 4,40551 |
| cambio al respaldo | 1,00000 | 0,59438 |
| repetición del actuador roto | 0,00000 | 0,85138 |
| error corporal postdaño | 0,09207 | 0,22942 |
| MSE predictivo adaptado | 1,10e-5 | 1,06e-2 |
| coincidencia con oráculo | 0,97919 | 0,59548 |
| MSE final del actuador dañado | 0,00000 | 0,02904 |

El radio espectral pasivo es `0,95079`. Tras mover shock y daño a otro nodo, el
automodelo conserva recuperación y cambio correctos en `1,000`. La lesión de
actualización coincide exactamente con el control congelado en cada semilla.

Todos los contrastes y los ocho guardrails fijados en `PROTOCOL.md` pasan.

La afirmación válida es automodelo causal de consecuencias motoras y adaptación
a daño dentro del cuerpo de onda ensayado. No equivale a un concepto abstracto
de sí mismo, memoria autobiográfica, consciencia o yo humano.
