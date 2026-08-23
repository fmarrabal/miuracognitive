# F3a — Veredicto (integración mHBP↔reasoner, cycle_transp, n=6)

## Métricas por variante (media±std entre seeds)

| variante | P (corrOOD) | E (acc/n_iter largo) | acc largo | I' |
|---|---|---|---|---|
| miura_mhbp | 0.209±0.056 | 0.058±0.004 | 0.732±0.030 | 0.855±0.029 |
| hbp_full | 0.221±0.049 | 0.057±0.003 | 0.750±0.027 | 0.853±0.042 |
| gating_wm | 0.214±0.021 | 0.060±0.002 | 0.735±0.015 | 0.851±0.039 |

## Contrastes primarios (Δ = mhbp − hbp_full; pareado por seed)

| contraste | Δ medio | IC95 | t | p | dz | no-inferioridad |
|---|---|---|---|---|---|---|
| C1-P | -0.0113 | [-0.0552, +0.0440] | -0.40 | 0.707 | -0.16 | Δ≥−0.05: ✓ |
| C1-E | +0.0010 | [+0.0002, +0.0019] | 2.06 | 0.094 | 0.84 | mhbp≥0.9·inc: ✓ |

## Guardarraíles

| seed | M1 trend(onp) | M2 follow/acc | M3 corr(onp) | M ok |
|---|---|---|---|---|
| 0 | +1.00 | 0.97 | +0.69 | ✓ |
| 1 | +1.00 | 1.05 | +0.70 | ✓ |
| 2 | +1.00 | 1.02 | +0.59 | ✗ |
| 3 | +1.00 | 1.02 | +0.73 | ✓ |
| 4 | +1.00 | 1.02 | +0.72 | ✓ |
| 5 | +1.00 | 1.02 | +0.66 | ✓ |

- Accuracy largo: Δ medio = -0.0179 (umbral ≥−0.03): ✓
- I' pareado: Δ medio = +0.0026 (umbral ≥−0.05): ✓
- Mecanismo M (on-policy, 6 seeds): ✗ TOCADO
- Anti-overclaim (P vs gating_wm): Δ = -0.0052

## Veredicto (ramas pre-declaradas)

**FAIL de guardarraíl** — la integración toca mecanismo/interfaz/accuracy: diagnóstico antes de nada (rama 3/4).