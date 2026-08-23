# Pre-registro — Fase 2b: campo-MODULADOR vs campo-FUENTE

> Declarado 2026-07-28, ANTES de entrenar ninguna celda nueva. Consecuencia
> directa del negativo de la Fase 2 (FINDINGS_PHASE2.md): el campo como FUENTE
> de acciones falla OOD por dos mecanismos (readout de plan no extrapola nivel;
> inercia = re-adaptación lenta). La Fase 2b aplica la tesis del programa
> ("modula, no piensa") al propio plano de control.

## Hipótesis

**H-F2b**: si el campo homeostático se re-cablea como MODULADOR DE GANANCIAS
de un lazo reactivo (que lee las señales directamente y produce las acciones
base), entonces:
- **D1 (recuperación; predicción fuerte ex ante)**: el fallo OOD de nivel
  desaparece — J_OOD(mhbp_gov) ≪ J_OOD(mhbp). El nivel del presupuesto fluye
  por la vía reactiva, que sí extrapola (el mlp lo demostró); el campo ya no
  puede bloquearlo.
- **D2 (valor añadido; PREGUNTA ABIERTA, sin predicción direccional fuerte)**:
  ¿la modulación de ganancias mejora al lazo reactivo puro? mhbp_gov vs react,
  pareado. Resultado nulo = "gobernador inerte" (no aporta, no daña) — también
  informativo. Positivo esperado, si lo hay, en protocolos de conmutación de
  riesgo (e2, riskfreq), donde ajustar ganancias por estado interno tiene
  sentido funcional.
- **D3 (contexto)**: mhbp_gov vs gru — ¿el lazo reactivo modulado alcanza a la
  memoria genérica?

## Arquitectura del campo-modulador (`mhbp_gov`)

- **Lazo reactivo base** (idéntico en `react` y `mhbp_gov`): la misma
  featurización que los baselines (26 canales, RunningNorm) → MLP 26→32→nA →
  logits reactivos z_react. Es el "arco reflejo": las acciones SIEMPRE se
  originan en las señales.
- **Campo**: el mismo CoupledMultiscaleHBP de la Fase 2 (4 campos, cadena,
  alostasis, θ=1), tickeado con la misma interocepción.
- **Modulación (solo en mhbp_gov)**: ganancias multiplicativas acotadas por
  actuador: g = 1 + γ·tanh(Σ_q V_q[ū_q; w̄_q]), γ=0.8 ⇒ g ∈ [0.2, 1.8]
  (atenúa/amplifica; NUNCA invierte ni sustituye — sin vía aditiva del campo
  a los logits: la pureza del rol es el punto del A/B).
  z = z0 + g ⊙ z_react → misma proyección Π_A (rangos MVP, STE en depth).
  Init: V≈0 ⇒ g≈1 ⇒ mhbp_gov ARRANCA siendo react (la modulación solo puede
  añadir sobre la solución reactiva).
- **Intervención causal**: flag `modulation_off()` (clampa g≡1 en eval) para
  el análisis de si la modulación es load-bearing.

## Protocolo

- Mismo entorno SMA (sin cambios: los gates de la Fase 2 siguen siendo
  válidos — son propiedades del entorno, no del controlador), mismos 500
  pasos, misma eval fija (300 episodios × 7 protocolos), mismo oráculo v3.
- Celdas nuevas: mhbp_gov × semillas 0..19, react × semillas 0..19 (40).
- **Reutilización declarada**: los contrastes D1/D3 usan las corridas YA
  EXISTENTES de mhbp (semillas 0..23) y gru (0..19) del confirmatorio de la
  Fase 2 — legítimo porque la eval es determinista y compartida; el pareado
  por semilla usa la intersección (0..19).
- Métrica primaria: la misma J_OOD de 3 componentes. Contrastes D1-D3
  pareados por semilla, Holm-3 sobre p_t, con dirección (D1: gov mejor que
  mhbp; D2: sin dirección impuesta — se reporta el signo; D3: sin dirección).
- Secundarias: los dos términos del mecanismo (plan en budget_hi; hard y
  settling en e4_step) — la predicción D1 es que AMBOS colapsan a niveles de
  baseline. Intervención g≡1 sobre mhbp_gov entrenado (post-hoc etiquetado).

## Criterios de éxito/fallo

- D1 confirmada + D2 positiva: demostración CONSTRUCTIVA de la tesis governor
  (el campo aporta como modulador donde destruía como fuente).
- D1 confirmada + D2 nula: el campo es un gobernador INERTE en este entorno —
  la tesis se sostiene en su mitad negativa; el valor de la modulación queda
  sin demostrar aquí (se dice tal cual).
- D1 refutada (gov sigue fallando OOD): el fallo NO era el rol de fuente —
  la hipótesis de mecanismo de FINDINGS_PHASE2 queda tocada y se re-examina.
  También informativo.

---

## ENMIENDA v2 (2026-07-28, tras panel adversarial 2b — 18 hallazgos, 3
## críticos — y ANTES de entrenar ninguna celda 2b)

1. **D1 es CONDUCTUAL, no arquitectural** (crítico del panel): la vía bilineal
   g·z_react permite en principio al campo actuar de fuente-acotada (constructivamente
   demostrado) o bloquear la vía reactiva (g=0.2). La "pureza del rol" la dan el
   init (g≈1), la cota g∈[0.2,1.8] y la ELIMINACIÓN DEL BIAS de la última capa
   reactiva (cierra la vía constante g·b). Lo que D1 contrasta es CONDUCTA
   (¿desaparece el fallo OOD?); el ROL efectivo lo dictaminan las
   intervenciones y la descomposición (puntos 3-4).
2. **D2 se parte en dos** (crítico: capacidad+memoria confundidas): D2a gov vs
   react (etiquetado CONFUNDIDO, fuera de Holm) y **D2b gov vs gru_gov** — el
   mismo cableado con un GRUCell (h=40) produciendo la ganancia: aísla la
   FÍSICA del campo de un gain-scheduler recurrente genérico. Familia Holm-3
   final: {D1, D2b, D3}. Un D2b nulo con D2a positivo = "la memoria en la
   ganancia aporta; la física del campo, no" (se dice tal cual).
3. **Intervenciones ejecutables** (crítico: la g≡1 era inejecutable): cada
   celda gov/gru_gov/react guarda CHECKPOINT (.ckpt.pt) y una segunda eval con
   g≡1 (eval_mod_off, pareada por episodio por determinismo). Post-hoc desde
   checkpoints (etiquetado): g congelada a su media temporal por episodio
   (¿basta el NIVEL de g?) y g barajada en el tiempo (¿importa el CUÁNDO?).
4. **Gates de mecanismo pre-registrados**: (a) descomposición de varianza del
   logit de budget_scale — si Var_t(z_react) ≈ 0 y Var_t(g) domina ⇒ MODO
   FUENTE detectado (el A/B se declara degenerado); (b) umbral cuantitativo de
   "colapso a baseline": plan(budget_hi)·λ ≤ 1.0 y hard(e4)·λ ≤ 5.0.
5. **Gate de equivalencia del reactivo** (major): react sube a 26→96→nA (sin
   bias final) y E1 (react vs mlp, etiquetado) debe dar |Δ J_OOD| pequeño; si
   react ≫ mlp J_OOD, la premisa de D1 ("la vía reactiva extrapola") se
   re-examina antes de interpretar.
6. **Ramas de resultado completadas** (major): D2b NEGATIVA significativa
   (gov peor que gru_gov) = la física DAÑA incluso como moduladora → negativo
   activo de la tesis constructiva, se reporta tal cual. "Nula" se
   operacionaliza como: no supervivencia de Holm E IC95 pareado conteniendo 0.
7. **Encuadre temporal declarado** (nota del panel): D1 es post-hoc respecto a
   los datos de la Fase 2 (el brazo mhbp ya está observado) y ex ante respecto
   a toda celda nueva; dado el init g≈1, D1 opera como MANIPULACIÓN DE
   MECANISMO (quitar el rol de fuente y ver si desaparece el fallo), no como
   descubrimiento. Los JSON llevan fingerprint del entorno; la identidad del
   entorno F2↔F2b la garantiza el registro de sesión (env.py sin cambios).
8. Celdas: mhbp_gov 20 + react 20 + gru_gov 20 (60). h* documentado como
   parámetro muerto en TODOS los controladores de campo (nota retroactiva
   también para FINDINGS_PHASE2); contabilidad de core unificada.
