# Pre-registro — Fase 3a: el plano mHBP integrado en el reasoner (intra-instancia)

> Declarado 2026-07-31, antes de entrenar ninguna celda nueva. Gateado por
> G0/G0.1 (PASS en cycle_transp). La Fase 3b (escala de sesión, persistencia
> entre instancias) se diseñará SOLO si F3a pasa sus guardarraíles.

## Hipótesis

**H-F3a**: el plano autonómico multiescala certificado (mHBP, 4 campos)
integrado en el reasoner de MiuraCognitive con la disciplina aprendida en el
arco — modulador de vías guiadas por señal, NUNCA fuente; enchufado a λ
pre-techo; techo como evento de presupuesto — iguala o mejora al campo único
incumbente (hbp_full) en la CALIDAD DE GOBIERNO del cómputo, sin tocar el
mecanismo de razonamiento (M) ni la accuracy.

## Diseño de la integración (MhbpReasonerAdapter; disciplina F2b en cada vía)

- **Un tick del plano por iteración del reasoner** (como el HBP incumbente).
  τ = (1, 3, 6, 12) — escalas DENTRO del horizonte de pensamiento (N_max=24):
  el resource integra ~2 unidades de su tiempo propio por rollout. (Lección
  SMA: las escalas deben vivir dentro de la estructura que gobiernan; el
  (1,4,8,32) del MVP dejaría al resource casi estático.)
- **Interocepción** (entra al plano por su encoder acotado): las señales por
  nodo del sistema actual (progreso, activación, entropía de atención,
  esfuerzo, longitud) mapeadas a canales canónicos + DOS señales nuevas
  motivadas por G0.1: margen y entropía del decode en la posición de
  respuesta por tick (la "completitud-en-curso" — existe y es abundante;
  probe 0.90-0.95). Sin acceso a K ni al target (anti-leakage: el margen del
  decode es auto-observación, no verdad externa).
- **Actuadores (todos modulación acotada de vías existentes; init ≈ neutro)**:
  1. *halt*: sesgo aditivo acotado γ_h·tanh(V_h·[fast; risk]) sobre el logit
     de λ — el enchufe es λ PRE-TECHO (nota G0.1 #2).
  2. *presupuesto*: presión de parada dependiente del ESFUERZO (señal) con
     ganancia modulada por el campo resource: p = softplus(a)·esfuerzo·
     (1+γ_r·tanh(V_r·resource)) — la señal manda, el campo escala (F2b).
     El evento de techo (llegar a N_max) entra como interocepción al resource.
  3. *WM*: gates write/forget = σ(base + V_w·deliberative) — retención como
     decisión de escala deliberativa.
  4. *bloques/reasoner*: gate suave 1+ε·tanh(·) desde fast (como el
     incumbente).
  Sin vía del campo a los LOGITS de la tarea (la lección F2 en su forma dura).
- SIN L_halt_aux en F3a (los incumbentes no la tienen; añadirla solo al mhbp
  confundiría el contraste; queda para F3b).

## Celdas

Tarea: **cycle_transp** (T certificada). Variantes: miura_mhbp (nueva),
hbp_full, gating_wm (incumbentes) × regímenes {indist, ood} × seeds 0..5.
Se REUTILIZAN las 12 celdas de G0 (hbp_full/gating_wm, seeds 0-2, ambos
regímenes, protocolo idéntico — declarado); nuevas: 12 de miura_mhbp +
12 de incumbentes seeds 3-5 = 24 entrenamientos (protocolo v3: 2500 pasos,
N_max=24, pin_fp32 en TODOS los parámetros del plano — lección BF16).

## Métricas y contrastes (pareados por semilla, n=6)

**Primarios (Holm-2, dirección declarada)**:
- **C1-P** mhbp vs hbp_full en P (corr_K_niter_ood): ≥, con mejora esperada.
- **C1-E** mhbp vs hbp_full en eficiencia (accuracy-largo / E[n_iter] medio,
  indist): ≥.
**Guardarraíles (cualquier violación = FAIL de la integración)**:
- Accuracy largo (indist): mhbp ≥ hbp_full − 0.03.
- **M intacto** (el test de aceptación nuevo): M1-trend ≥ 0.9 y M2
  follow/acc ≥ 0.95 y M3 ≥ 0.6 sobre miura_mhbp — el gobernador NO puede
  tocar el mecanismo (la frontera del arco, ahora medible).
- **I' ≥ I'(hbp_full) − 0.05**: el gobernador no degrada la interfaz que
  modula.
**Secundarios**: I' mejora (esperable si el plano usa la señal de completitud);
contraste vs gating_wm; curva E[n_iter|K]; diagnóstico de los 4 campos
(¿especialización? — trazas de ū_q por tick).

## Criterios de decisión

- C1-P o C1-E significativo a favor + guardarraíles ✓ → el plano multiescala
  aporta gobierno: F3b (escala de sesión, el test multiescala genuino).
- Empate (no significativo) + guardarraíles ✓ → el plano certificado NO daña
  (no-inferioridad): F3b decide (la escala intra-instancia puede ser
  demasiado corta para 4 campos — declarado como posible techo del diseño).
- Guardarraíl M o I' violado → la integración toca el mecanismo: FAIL
  informativo mayor (la disciplina F2b no bastó); diagnóstico antes de nada.
- Accuracy violada → FAIL clásico; diagnóstico.

---

## ENMIENDA v2 (2026-07-31, tras panel de cableado — 24 hallazgos, 2 críticos
## — y ANTES de entrenar celda alguna)

1. *Crítico corregido: train_run crasheaba con el adaptador en el diagnóstico
   post-entrenamiento (perdiendo la celda tras 2500 pasos); branch añadido.*
2. *Envolvente de halting SIMETRIZADA y ADITIVA (forma declarada):
   bias = tanh(V_halt·[fast;risk]) + presión, clamp ±1 — autoridad equiparada
   al incumbente ((σ−0.5)·2). Presión init softplus(−5)≈0.007 (arranque neutro;
   con −2 la rampa sesgaba +0.13·esfuerzo desde el paso 0).*
3. *C1 se declara contraste de SISTEMAS (paquete mhbp completo vs incumbente
   como está, cada uno con sus pérdidas auxiliares nativas — como en F2/F2b).
   Asimetrías DECLARADAS: interocepción de completitud (solo mhbp),
   RunningNorm EMA (solo mhbp), L_interoc≡0 / L_stab≡0 / L_homeo=mean u² en
   mhbp (vs β_intero=0.1, β_stab=0.1, homeo propio en hbp_full), regularizador
   de alostasis NO aplicado (ε_a acota igualmente). Brazos de DESCOMPOSICIÓN
   pre-declarados y CONDICIONALES a C1 favorable: (i) mhbp con
   mask_completeness=True (flag ya implementado); (ii) hbp_full+completitud.*
4. *C1-P: celdas OOD, métrica compute_diag.corr_K_niter_ood (idéntica a G0-P).
   C1-E: largo.acc_per_niter del compute_diag, celdas INDIST.*
5. *Estadística: n=6 pareado ⇒ MDE dz≈1.05 (Holm-2) — F3a se declara
   ESTIMACIÓN (IC95 pareado + márgenes) con el test como confirmación solo si
   el efecto es enorme. No-inferioridad operacionalizada: P ≥ incumbente−0.05;
   E ≥ 0.9·incumbente. Rama añadida: primario significativo EN CONTRA con
   guardarraíles ✓ = FAIL del aporte (se reporta tal cual). Condición
   anti-overclaim: "el plano aporta gobierno" exige además mhbp ≥ gating_wm
   en P (el incumbente FUERTE en P: 5/6 comparaciones sobre hbp_full).*
6. *Guardarraíl M para miura_mhbp: M1-trend y M3 se computan ON-POLICY
   ([1, n_parada]; el rango completo se reporta también) — el artefacto de
   deriva post-parada lo documentó G0 y castigaría justo a quien pare antes.*
7. *I' operacionalizado: AUC(max_{n<N_max} λ_n, corrección), stream
   seed+800000, ≥512 muestras, 3 variantes × 6 seeds; guardarraíl sobre la
   media pareada.*
8. *Config del plano fijada: θ=1 (BE — el régimen con margen certificado
   1−ρ≥2e−4 del sistema de 2º ORDEN; θ no cambia el orden de la dinámica),
   dt=1, cadena, d=8, alostasis on. Los actuadores nativos del plano
   (ActuatorHead) no se usan en F3a. token_cost/hit_cap: vía causalmente
   INERTE en F3a (solo el último tick, sin persistencia) — cobra sentido en
   F3b; declarada como limitación.*
9. *Anti-leakage re-redactado: sin acceso al TARGET; la LONGITUD (∝K) entra
   como señal en todas las variantes por igual (frac_valid). Control
   secundario declarado: enmascarar queue_load post-hoc.*
10. *readout_token_id=2 válido SOLO para permcomp (asertado en el runner).
    Repo git inicializado para procedencia (código+preregs; sin ckpts).*
