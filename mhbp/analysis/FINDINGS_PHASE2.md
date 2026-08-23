# Fase 2 — Registro del resultado: H-F2 REFUTADA (negativo informativo con mecanismo)

> Confirmatorio pre-registrado (PREREG_PHASE2.md v1→v3c), 124/124 celdas,
> 0 errores, 500 pasos, eval fija compartida (300 episodios × 7 protocolos),
> oráculo por-instancia convergido (residuo <1%). Datos: `results_phase2/`;
> análisis: `REPORT_PHASE2.md`. Fecha: 2026-07-28.

## Enunciado del resultado

**Un plano de control de campos homeostáticos multiescala (mHBP) que GENERA
las acciones de asignación desde su estado interno es SIGNIFICATIVAMENTE PEOR
fuera de distribución que baselines reactivos o de memoria genérica
equiparados — y el mecanismo del fallo está localizado.**

Los tres contrastes primarios salieron significativos en dirección contraria
a H-F2 (Holm-3 irrelevante: la dirección ya la descarta):

| contraste | Δ(J_OOD) | t | dz | lectura |
|---|---|---|---|---|
| C1 mhbp vs gru | +7.68 | 18.2 | 4.06 | la memoria genérica gana al campo |
| C2 mhbp vs taueq | +3.93 | 8.95 | 2.24 | separar escalas EMPEORA aquí |
| C3 mhbp vs first | +2.13 | 4.62 | 0.94 | el 2º orden EMPEORA aquí |

El mlp reactivo sin memoria es el mejor OOD (J=1.91 vs mhbp 10.11).

## Localización: el fallo es EXCLUSIVO del cambio de presupuesto

En iid, riskfreq y long el mhbp rinde igual que todos (J≈1.87; error de tarea
IDÉNTICO a los baselines en todos los protocolos). Explota solo en:
budget_lo (16.6), budget_hi (35.9), e4_step (23.2) — vs 2-3 de gru/mlp.

## Dos mecanismos separados (descomposición post-hoc de J, etiquetada como tal)

1. **El campo-como-planificador no extrapola NIVEL** (budget_hi): err=1.15
   (idéntico a gru/mlp), hard=0 (no viola el presupuesto real); el 96% del
   daño es el término de PLAN (36.3/37.7): la cabeza de reserva lee el estado
   del campo — un filtro temporal con dinámica aprendida, calibrado a las
   estadísticas de entrenamiento — y no escala la reserva cuando el nivel del
   presupuesto sale del rango visto. mhbp_first falla IGUAL (33.5): no es la
   inercia, es el rol de fuente. El mlp lee la señal y ajusta (plan=0.22).
2. **La inercia re-adapta lento tras un shift real** (e4_step): violación del
   presupuesto REAL hard=17.3 (mhbp) vs 5.5 (first) vs 2.5 (gru) vs 0.9 (mlp);
   settling 2.5 ventanas vs 0.4. El orden mhbp > first aquí es la firma de la
   inercia de 2º orden. Es la MISMA propiedad del positivo v3 (estabilidad de
   una política buena bajo perturbación) actuando como pasivo (rancidez tras
   un cambio genuino): **la inercia es un prior bidireccional**.

Las ablations X5/X6 (acoplamiento, alostasis: sin efecto, |dz|≤0.21) localizan
el daño en la dinámica por-campo como sustrato de decisión, no en el cableado
multiescala.

## Por qué el negativo es sólido

Entorno con kill-gates en verde (exige adaptación: hueco 133% del oráculo;
resoluble: privilegiado 0.61; sin leakage: permutadas−enmascaradas=+0.01;
oráculo cota inferior verificada en todas las corridas), pre-registro con
enmiendas fechadas pre-datos, eval fija compartida (pareado por semilla y por
episodio), baselines con núcleo ≥ mhbp, y dos paneles adversariales previos
(32 hallazgos incorporados, incluido el crítico del oráculo no convergido).

## Lectura unificada con el programa MiuraCognitive

"**Modula, no piensa**" (paper governor) aplica también al plano de control:
- El campo como FUENTE de accuracy → null (programa HBP).
- El campo como FUENTE de planificación → negativo activo (esta fase).
- El campo como ESTABILIZADOR de la asignación de cómputo → único positivo
  robusto (benchmark v3).

Predicción que esto genera (falsable, Fase 2b): re-cablear el campo como
MODULADOR DE GANANCIAS de un lazo reactivo (que lee las señales directamente)
debe (D1) eliminar el fallo OOD de nivel, y (D2) la pregunta abierta es si la
modulación añade valor sobre el lazo reactivo puro. Ver PREREG_PHASE2B.md.
