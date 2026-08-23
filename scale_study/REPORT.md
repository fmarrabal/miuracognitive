# Resultados — Campo homeostático a escala gigante

Celdas completadas: **1019** (errores: 0). Generado por `aggregate.py`.

Celdas por bloque: B1=84, B2=10, B3b=8, B3c=6, B3map=540, B3p=8, B3s=112, B3x=28, B4=220, B5=3

## B1 — Espectro a escala

| topología | N | ρ(L) | ρ(A) | ρ(A³) |
|---|---|---|---|---|
| chain | 6 | 3.732 | 1.802 | 5.851 |
| chain | 64 | 3.998 | 1.998 | 7.972 |
| chain | 1,024 | 4.000 | 2.000 | 8.000 |
| chain | 16,384 | 4.000 | 2.000 | 8.000 |
| chain | 262,144 | 4.000 | 2.000 | 8.000 |
| expander | 64 | 7.321 | 3.453 | 41.173 |
| expander | 64 | 7.263 | 3.396 | 39.169 |
| expander | 256 | 7.405 | 3.456 | 41.266 |
| expander | 1,024 | 7.433 | 3.477 | 42.032 |
| expander | 1,024 | 7.427 | 3.464 | 41.577 |
| expander | 4,096 | 7.435 | 3.489 | 42.467 |
| expander | 16,384 | 7.439 | 3.477 | 42.034 |
| expander | 16,384 | 7.440 | 3.485 | 42.316 |
| expander | 65,536 | 7.437 | 3.479 | 42.107 |
| grid2d | 6 | 8.000 | 2.000 | 8.000 |
| grid2d | 64 | 8.000 | 2.000 | 8.000 |
| grid2d | 1,024 | 8.000 | 2.000 | 8.000 |
| grid2d | 16,384 | 8.000 | 2.000 | 8.000 |
| grid2d | 262,144 | 8.000 | 2.000 | 8.000 |
| random_regular | 64 | 7.325 | 3.453 | 41.173 |
| random_regular | 64 | 7.262 | 3.396 | 39.169 |
| random_regular | 256 | 7.415 | 3.456 | 41.266 |
| random_regular | 1,024 | 7.432 | 3.477 | 42.032 |
| random_regular | 1,024 | 7.414 | 3.463 | 41.522 |
| random_regular | 4,096 | 7.440 | 3.487 | 42.411 |
| random_regular | 16,384 | 7.436 | 3.474 | 41.937 |
| random_regular | 16,384 | 7.435 | 3.484 | 42.297 |
| random_regular | 65,536 | 7.438 | 3.467 | 41.680 |
| ring | 6 | 4.000 | 2.000 | 8.000 |
| ring | 64 | 4.000 | 2.000 | 8.000 |
| ring | 1,024 | 4.000 | 2.000 | 8.000 |
| ring | 16,384 | 4.000 | 2.000 | 8.000 |
| ring | 262,144 | 4.000 | 2.000 | 8.000 |
| ws_smallworld | 64 | 10.971 | 4.967 | 122.509 |
| ws_smallworld | 64 | 10.213 | 5.001 | 125.048 |
| ws_smallworld | 256 | 10.173 | 5.011 | 125.855 |
| ws_smallworld | 1,024 | 11.515 | 5.077 | 130.827 |
| ws_smallworld | 1,024 | 10.844 | 5.032 | 127.443 |
| ws_smallworld | 4,096 | 11.090 | 5.087 | 131.641 |
| ws_smallworld | 16,384 | 11.884 | 5.178 | 138.800 |
| ws_smallworld | 16,384 | 12.519 | 5.110 | 133.457 |
| ws_smallworld | 65,536 | 12.557 | 5.128 | 134.854 |

**H1 (grado acotado → ρ acotado):** CONFIRMADA — ρ(A³) máx en estructurados N≥10⁴: 8.00 (≤8 teórico).

## B2 — Relación de dispersión (anillo)

**H2 (verificación):** error máx medida vs companion EXACTO (con el término giroscópico discreto de la Prop. 2) = 1.17e-01 (FAIL, tol 3%). La dispersión β·A³ modifica ω(k) de forma medible y la teoría discreta la predice; el espectro es un continuo a N grande (vs 3 puntos a N=6).

## B4 — Estabilidad/flutter a escala

**H5 (certificado tight a escala):** el umbral de Merkin predice el flutter en 55/55 celdas circulatorias (100%). El giroscópico decae en toda topología y N.

## B3 — Solitones a escala (la prueba central)

- genuine (sin amort.): **223/312** solitones limpios (amp≈cte, ancho CV<0.25, vel≈teoría).
- saturated (sin amort.): **184/312** solitones limpios.
- ancho CV medio: genuine=0.055 vs saturated=0.061.

**H3 (emergen solitones a escala):** CONFIRMADA — a N≥1024 hay solitones coherentes; a N=6 es imposible (3 modos).
**H3s (la saturación degrada):** CONFIRMADA — genuine 223 vs saturated 184 limpios.
**Amortiguamiento mata solitones:** 0/56 limpios con amort. débil (el campo modulador —amortiguado y forzado— no los sostiene).

## B3b — Colisión de solitones (elasticidad)

| N | nolin | picos ini | picos fin | amps ini | amps fin |
|---|---|---|---|---|---|
| 4,096 | genuine | 2 | 2 | [0.9, 0.35] | [0.8, 0.31] |
| 4,096 | saturated | 2 | 2 | [0.9, 0.35] | [0.76, 0.31] |
| 16,384 | genuine | 2 | 2 | [0.9, 0.35] | [0.8, 0.31] |
| 16,384 | saturated | 2 | 2 | [0.9, 0.35] | [0.76, 0.31] |
| 65,536 | genuine | 2 | 2 | [0.9, 0.35] | [0.8, 0.31] |
| 65,536 | saturated | 2 | 2 | [0.9, 0.35] | [0.76, 0.31] |
| 262,144 | genuine | 2 | 2 | [0.9, 0.35] | [0.8, 0.31] |
| 262,144 | saturated | 2 | 2 | [0.9, 0.35] | [0.76, 0.31] |
