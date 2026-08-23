# G0 — Veredicto (batería T-M-I-P sobre el reasoner de MiuraCognitive)

## gating_wm × adjacent

| seed | T gap | M1 trend | M1 F_fin/acc | M2 f+r | M2 free | M3 corr | I AUC | P corrOOD | GATE |
|---|---|---|---|---|---|---|---|---|---|
| 0 | +0.05✗ | +0.17✗ | 0.91 | +0.45✗ | +0.00✓ | +0.64✓ | +0.71✓ | +0.05✓ | FAIL |
| 1 | +0.12✗ | +0.45✗ | 0.94 | +0.51✗ | +0.00✓ | +0.51✓ | +0.53✗ | -0.06✓ | FAIL |
| 2 | +0.01✗ | +0.50✗ | 0.94 | +0.35✗ | +0.01✓ | +0.44✓ | +0.66✓ | -0.01✓ | FAIL |

**Gate gating_wm×adjacent: FAIL** (0/3 seeds)

## gating_wm × cycle_transp

| seed | T gap | M1 trend | M1 F_fin/acc | M2 f+r | M2 free | M3 corr | I AUC | P corrOOD | GATE |
|---|---|---|---|---|---|---|---|---|---|
| 0 | +0.12✗ | +0.99✓ | 0.97 | +0.87✓ | +0.00✓ | +0.84✓ | +0.13✗ | +0.21✓ | FAIL |
| 1 | +0.16✓ | +0.98✓ | 0.98 | +0.88✓ | +0.00✓ | +0.80✓ | +0.20✗ | +0.21✓ | FAIL |
| 2 | +0.14✗ | +1.00✓ | 0.98 | +0.86✓ | +0.00✓ | +0.85✓ | +0.23✗ | +0.22✓ | FAIL |

**Gate gating_wm×cycle_transp: FAIL** (0/3 seeds)

## hbp_full × adjacent

| seed | T gap | M1 trend | M1 F_fin/acc | M2 f+r | M2 free | M3 corr | I AUC | P corrOOD | GATE |
|---|---|---|---|---|---|---|---|---|---|
| 0 | +0.06✗ | -0.21✗ | 0.10 | +0.35✗ | +0.00✓ | +0.63✓ | +0.67✓ | +0.15✓ | FAIL |
| 1 | +0.08✗ | -0.03✗ | 0.05 | +0.47✗ | +0.00✓ | +0.66✓ | +0.69✓ | +0.34✓ | FAIL |
| 2 | +0.11✗ | +0.18✗ | 0.34 | +0.47✗ | +0.01✓ | +0.53✓ | +0.66✓ | -0.01✗ | FAIL |

**Gate hbp_full×adjacent: FAIL** (0/3 seeds)

## hbp_full × cycle_transp

| seed | T gap | M1 trend | M1 F_fin/acc | M2 f+r | M2 free | M3 corr | I AUC | P corrOOD | GATE |
|---|---|---|---|---|---|---|---|---|---|
| 0 | +0.15✗ | +1.00✓ | 1.00 | +0.90✓ | +0.00✓ | +0.85✓ | +0.15✗ | +0.13✓ | FAIL |
| 1 | +0.18✓ | +1.00✓ | 1.00 | +0.89✓ | +0.00✓ | +0.80✓ | +0.27✗ | +0.23✓ | FAIL |
| 2 | +0.18✓ | +1.00✓ | 1.00 | +0.91✓ | +0.00✓ | +0.78✓ | +0.18✗ | +0.22✓ | FAIL |

**Gate hbp_full×cycle_transp: FAIL** (0/3 seeds)

## Veredicto global

- gating_wm × adjacent: **FAIL** (0/3)
- gating_wm × cycle_transp: **FAIL** (0/3)
- hbp_full × adjacent: **FAIL** (0/3)
- hbp_full × cycle_transp: **FAIL** (0/3)

**G0 FAIL global: el par (reasoner, tarea) NO está en régimen de razonamiento — aplicar los remedios pre-registrados (rediseñar supervisión/tarea para hacer visible el camino) ANTES de integrar.**