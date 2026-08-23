# Pre-registro — G0: ¿opera el reasoner de MiuraCognitive en régimen de razonamiento?

> Kill-gate PREVIO a la Fase 3 (integración mHBP↔reasoner). Motivación
> (ARCO_MHBP.md §5): los nulos del programa son consistentes con que el par
> (reasoner, tareas) operaba en régimen de ASENTAMIENTO — se puso gobernador a
> un sistema posiblemente sin transitorio que gobernar. G0 certifica el régimen
> con la batería T-M-I-P antes de invertir en la integración.
> Declarado 2026-07-30, antes de correr ningún instrumento.

## Objeto

El reasoner original de MiuraCognitive (`model/miura.py`: bloque recurrente +
halting PonderNet + working memory), en la tarea de composición en S₅ con
supervisión densa y los dos conjuntos de generadores del protocolo v3 (la
tarea donde la iteración se forzó de verdad: E[n_iter] hasta 11, creciente
con K). Variantes: `hbp_full`, `gating_wm`, `vanilla` (control T), 3 semillas
(fijas: 0,1,2), protocolo de entrenamiento v3 (mismos hiperparámetros;
`pin_fp32`). Los modelos se entrenan de nuevo instrumentados (hooks que
vuelcan el estado del reasoner por tick); ~2-3 h GPU en total.

## La batería (los 4 tests, con umbrales)

### G0-T — la tarea es serialmente irreducible
`vanilla` (profundidad fija, params equiparados) vs recurrentes en el estrato
K largo. **Umbral**: gap de accuracy ≥ 0.15 en K≥18 (in-dist largo). Si T
falla, ninguna medición posterior significa nada (la tarea admite atajo
paralelo) → rediseño de tarea antes de seguir.

### G0-M — la trayectoria porta contenido (el corazón del gate)
- **M1 (decodabilidad)**: probe lineal desde el estado del reasoner en el tick
  t → permutación corriente. Dos curvas: F_full(t) (decodifica la respuesta
  final: ¿cristaliza gradualmente o está desde t=1?) y F_pref(t, j)
  (decodifica prefijos σ_{1:j}: ¿qué está representado cuándo?). **Umbral**:
  F_full creciente en t (pendiente > 0, test de tendencia) y
  F_full(t_final) ≥ 0.8 × accuracy de salida. Un F_full plano-alto desde t=1
  = el cómputo no vive en los ticks (asentamiento/shortcut).
- **M2 (causalidad de contenido — swap)**: en el tick t (t ∈ {25%, 50%, 75%}
  del cómputo), trasplantar el estado completo del reasoner entre dos inputs
  A→B con el mismo K. Tres desenlaces por swap, todos informativos:
  (a) la salida SIGUE AL DONANTE (responde lo de A) → el estado porta la
  computación ✓; (b) revierte al huésped CON ticks extra medibles (re-deriva
  del input) → re-derivación con coste ✓; (c) revierte al huésped SIN coste →
  el estado no es load-bearing ✗. **Umbral**: (a)+(b) ≥ 0.6 de los swaps y
  (c) ≤ 0.3, en el estrato K medio-largo.
- **M3 (ligadura)**: correlación por instancia entre la fidelidad de
  trayectoria (AUC de F_full(t)) y el acierto en K OOD (>12, protocolo v3).
  **Umbral**: corr ≥ 0.2 (por conjunto de generadores).

### G0-I — parada informada
AUC(p_halt en el tick de parada, corrección de la instancia). **Umbral**:
AUC ≥ 0.65 en al menos un conjunto de generadores. (NO se exige curva anytime
monótona: esa es la firma del asentador.)

### G0-P — la política de profundidad extrapola
corr(K, E[n_iter]) restringida a K≥14 (réplica del protocolo v3 sobre estos
mismos entrenamientos). **Umbral**: ≥ 0.10 en `hbp_full` (v3 dio 0.143/0.268);
se reporta el contraste con `gating_wm` como contexto, sin Holm (G0 es gate
diagnóstico, no confirmatorio de hipótesis).

## Reglas de decisión (pre-declaradas)

- **G0 PASS** (T ∧ M ∧ I ∧ P): el par (tarea, reasoner) está en régimen de
  razonamiento → la Fase 3 procede: el gobernador tiene algo que gobernar, y
  los contrastes de integración se harán sobre un M certificado.
- **Fallos parciales, con remedio asignado**:
  - T falla → rediseñar la TAREA (regla de oro: forzar por el entorno).
  - M falla (el caso que la historia del programa hace probable) → NO integrar:
    primero rediseñar la supervisión/estructura para hacer visible el camino
    (candidatos, en orden: supervisión por-operación alineada al tick;
    curriculum de alineación tick↔op; tick-por-op arquitectónico). Repetir G0.
  - I falla → re-entrenar/re-diseñar la cabeza de halting (es la interfaz del
    gobernador; sin ella la Fase 3 no tiene enchufe).
  - P falla → el positivo v3 no replica en este montaje: investigar antes de
    nada (sería un resultado en sí mismo).
- G0 es descriptivo-diagnóstico: umbrales pre-registrados, sin corrección
  múltiple (no hay familia de hipótesis confirmatorias), 3 semillas, se
  reportan las 3 por separado (el gate exige PASS en ≥2/3).

## Entregables

`mhbp/tasks/reasoner_g0/`: instrumentación de hooks (volcado de estados por
tick), `g0_probes.py` (M1, M3), `g0_swap.py` (M2), `g0_halt.py` (I),
`g0_policy.py` (P), `g0_report.py` (tabla única PASS/FAIL con los umbrales).
Todo resumible y con semillas fijas, según convenciones del proyecto.
