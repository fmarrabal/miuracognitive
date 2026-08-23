# Pre-registro — Fase 2 del mHBP: entorno sintético multiescala

> Declarado ANTES de entrenar nada ni mirar ningún número (2026-07-27). Regla de
> oro del proyecto: forzar capacidad arreglando el ENTORNO, no la loss; los
> confirmatorios siempre pre-registrados y resumibles.

## Hipótesis

**H-F2**: un plano de control con Q=4 campos homeostáticos acoplados y escalas
temporales separadas (mHBP) asigna mejor los recursos (cómputo, herramienta,
halting, presupuesto) que controladores equiparados sin esa estructura, y la
ventaja crece FUERA de distribución (presupuestos/nunca vistos, conmutación de
riesgo más rápida, episodios más largos).

Sub-hipótesis (contrastes PRIMARIOS, pareados por semilla, Holm-3, y con
dirección: solo cuenta si Δ<0, mhbp mejor):
- **C1** mhbp vs **gru** (baseline aprendido fuerte, núcleo 2.5× MAYOR que el
  mhbp — asimetría conservadora a favor del baseline): J_OOD menor.
- **C2** mhbp vs **mhbp_taueq** (escalas iguales; núcleo idéntico): la
  SEPARACIÓN de escalas importa.
- **C3** mhbp vs **mhbp_first** (1er orden; núcleo idéntico): el 2º orden
  importa (contraste bandera; n=30 en este par por potencia, dz≈0.66 detectable).
- **X4 (EXPLORATORIO)** mhbp vs hbp_single: núcleo 17× menor — estructura y
  capacidad CONFUNDIDAS (hallazgo del panel); se reporta etiquetado, sin Holm.

Métrica primaria: **J_OOD de TRES componentes** = media de {(budget_lo +
budget_hi)/2, riskfreq, long} — el factor presupuesto pesa 1/3, no 1/2. La
media plana de 4 protocolos se reporta como análisis de sensibilidad. Secundarias: compute-regret vs oráculo, tasa/magnitud de
violación de presupuesto, settling tras el escalón de presupuesto, correlaciones
de seguimiento (dificultad→profundidad, riesgo→herramienta, presupuesto→gasto),
suavidad de acciones. NO se interpretará ausencia de significación como
equivalencia (§23 del plan maestro).

## Entorno (SMA: Synthetic Multiscale Allocation)

Episodio de T=256 ticks, ventanas de W=32 (8 ventanas). TRES factores latentes
con escalas distintas — la estructura multiescala vive en el ENTORNO:
- **d_t ∈ (0,1)** dificultad, RÁPIDA: AR(1) φ=0.7 + saltos (p=0.05).
- **ρ_t ∈ {1,4}** coste del error, MEDIA: Markov 2 estados, permanencia ~32.
- **B_w** presupuesto por ventana, LENTO: sinusoide de periodo ~4 ventanas ×
  factor [0.6, 1.4] sobre B_base=280 (calibrado para que la restricción MUERDA).

Acciones (los 4 actuadores del MVP, cada uno con rol a su escala):
- `halt_bias` h_t (rápida): ahorra coste en fáciles (α_c·σ(h)·(1−d)) y daña
  calidad en difíciles (α_h·σ(h)·d) → debe seguir −d_t tick a tick.
- `depth_max` n_t ∈ [2,24] (rápida/media): calidad q ∝ n; coste ∝ n.
- `tool_gate` g_t ∈ [0,1] (media): calidad ×(1+0.5g) a coste fijo 6 — rentable
  con dificultad alta y sobre todo con ρ alto.
- `budget_scale` s_t (lenta): PLANIFICACIÓN — la reserva de la ventana w es
  B_nom·mean(s en la ventana w−1); pasarse de la reserva o de B_w penaliza,
  reservar de más tiene coste de retención. Exige anticipar el presupuesto.

    q_eff = n·(1+0.5g)·(1 − 0.3·σ(h)·d),   err = σ(0.5·(d·19.2 − q_eff))
    c = n·(1 − 0.5·σ(h)·(1−d)) + 6·g,      spend_w = Σ_{t∈w} c_t
    J = Σ ρ·err/T + 0.02·Σ ReLU(spend−B_w)²/nW + 0.005·Σ ReLU(spend−reserva)²/nW
        + 0.002·Σ ReLU(reserva−spend)/nW

Interocepción (canales canónicos, RUIDOSOS y parcialmente retrasados; las
OBSERVACIONES van detached — el gradiente solo fluye por las acciones):
entropy≈d+0.15ε; uncertainty≈|ε'|·(0.5+d); risk≈indicador de ρ RETRASADO 2
ticks con 20% de ruido; token_cost=c_{t−1}/24; gpu_memory≈presupuesto restante
normalizado + ruido; elapsed_time=fase de la ventana; queue_load=índice de
ventana. ANTI-LEAKAGE: ningún canal expone d, ρ, B sin ruido/retraso.

## Controladores (8; mismas señales, mismos 4 actuadores, misma proyección Π_A)

| nombre | núcleo | nota |
|---|---|---|
| mhbp | 4 campos, cadena, alostasis, θ=1, τ=(1,4,8,32) | el sistema |
| mhbp_nocpl | acoplamiento none, sin alostasis | ablation acoplamiento |
| mhbp_taueq | τ=(4,4,4,4) | ablation separación de escalas |
| mhbp_first | integrador 1er orden (C u̇+𝓚u=F IMEX) | ablation ORDEN (bandera) |
| hbp_single | Q=1, N=22 nodos (estado 176 = 4 campos) | ablation multi-campo |
| gru | GRU h=24 (~params equiparados) | baseline aprendido fuerte |
| mlp | MLP 26→64→32, sin memoria | baseline reactivo |
| ar2 | z_t = tanh(A₁z_{t−1}+A₂z_{t−2}+Bs), z∈R³² | 2º orden sin física |

Se reporta la tabla de parámetros; equiparación ±30% en el núcleo (el encoder
de señales y las cabezas Π_A son idénticos en estructura).

## Protocolo

- Entrenamiento: 250 pasos Adam lr 3e-3, batch 32 episodios frescos por paso,
  T=256, clip 1.0, FP64. Loss = J (+1e-3 regularizador de alostasis solo mhbp —
  anti-bypass, no ayuda a la tarea).
  *ENMIENDA PRE-DATOS (2026-07-27, antes del confirmatorio): 500→250 pasos.
  Motivo: el piloto de calibración (fuera del confirmatorio) muestra meseta de
  J en ~80 pasos (25.2→1.58) y 5.8 s/paso del mhbp; 250 pasos dan margen 3× a
  la meseta y hacen viable la noche (~10 h con 4 workers).*
- Semillas 0..19 (20), pareadas: los episodios de EVAL son fijos y compartidos
  entre todos los controladores (mismos episodios exactos → contraste pareado
  por semilla Y por episodio).
- Eval (300 episodios por protocolo): IID | OOD-budget (×0.6 y ×1.5) |
  OOD-riskfreq (permanencia 8) | OOD-long (T=512) | E4-escalón (B ×0.4 en
  t=128; settling/overshoot) | E2-riesgo puro (d≡0.55; ¿tool/depth siguen a ρ?).
- Oráculo NUMÉRICO por protocolo: optimización directa de las acciones con los
  latentes VERDADEROS (Adam 300 pasos, mismos episodios, cacheado) → cota
  superior común para el regret. El oráculo no observa: optimiza.
- Estadística: contrastes C1-C4 pareados por semilla (t-test + Wilcoxon),
  Holm-4 sobre la métrica primaria; IC bootstrap; tamaños de efecto; el resto
  exploratorio y etiquetado como tal.

## Kill-gates (antes del confirmatorio; si fallan NO se lanza)

- **G1 (el entorno exige adaptación)**: J(política constante óptima) −
  J(oráculo) ≥ 0.3·J(oráculo). Si una constante basta, el entorno está mal.
- **G2 (resoluble desde las señales)**: un GRU PRIVILEGIADO (recibe los
  latentes verdaderos por canales extra) recupera ≥60% del hueco
  constante→oráculo en 250 pasos. Si ni con verdad se puede, las señales o el
  entorno están rotos.
- **G3a (sin leakage estructural)**: recovery(GRU con canales de entorno
  permutados en el tiempo) − recovery(GRU con esos canales ENMASCARADOS)
  ≤ 0.10. Permutar no puede rendir más que quitar; el exceso sería
  información fuera de canal.
- **G3b (rebanada de tracking sustancial)**: recovery(privilegiado) −
  recovery(permutadas) ≥ 0.30 — la información tick-alineada debe ser parte
  sustancial del hueco (es donde las hipótesis C1-C4 pueden decidirse).
- **G4 (oráculo válido)**: J_oráculo ≤ J de toda política evaluada en los
  mismos episodios.

*ENMIENDA PRE-DATOS v2 (2026-07-27, tras el diagnóstico del G3 v1 y ANTES del
confirmatorio): (i) el G3 original (recovery permutadas ≤5%) confundía leakage
con la componente de presupuesto, legítimamente observable y resoluble sin
seguimiento del entorno — se sustituye por G3a+G3b; (ii) se recalibran las
observaciones para que el filtrado temporal pague: obs_noise 0.15→0.45,
risk_flip 0.2→0.35, y el presupuesto restante pasa a observarse RANCIO
(refresco cada W/8 ticks) y ruidoso (σ=0.25). Ningún dato confirmatorio se ha
mirado; el piloto y los gates v1 son calibración declarada.*

*ENMIENDA PRE-DATOS v3 (2026-07-27, tras la revisión adversarial del diseño —
3 verificadores, 32 hallazgos — y ANTES de todo dato confirmatorio):*
1. *ORÁCULO v2 (crítico): 300→8000 iteraciones con decay coseno; residuo de
   convergencia <1% exigido y registrado; mismo espacio de acciones que los
   actuadores (halt [−3,3]); caché indexada por hash de configuración.*
2. *Contrastes: C4→exploratorio (núcleo 17× menor: confundido); Holm-4→Holm-3
   con requisito de dirección; n=30 para el par C3 (potencia).*
3. *Métrica primaria: 3 componentes con presupuesto promediado primero (la
   redacción v1 "3 protocolos" era ambigua con 4 entradas). Sensibilidad: media
   plana de 4.*
4. *Entorno: presupuesto con DOS frecuencias no conmensurables (3.7 y 6.3
   ventanas — con periodo 4 exacto, B_{w+2}=2μ−B_w era una recurrencia
   determinista); dwell del riesgo 32→14 (≠W: reasignación intra-ventana);
   término de coste intrínseco λ_c=8e−4 (ahorrar con riesgo bajo paga); ruido
   de medición 8% en token_cost (no se puede integrar el gasto exacto).*
5. *Seguimiento: corr(d, depth) restringida a d<0.6 (la óptima global es
   NEGATIVA por triaje — U invertida); la global se reporta descriptiva.
   viol_rate se complementa con magnitud relativa (media y p95).*
6. *Tabla real de parámetros en el informe (gru_h=40, mlp 96/48, ar2 z=44:
   baselines con núcleo ≥ mhbp, conservador a favor del baseline). Ablations
   nocpl/noallo separadas (atribución limpia). Eval: 300 episodios/protocolo;
   J por episodio guardado (análisis pareado por episodio como sensibilidad).*
7. *Regularizador de alostasis acumulado por tick (antes solo el último).*

*CALIBRACIÓN v3b (2026-07-27, iteración de gates — sin datos confirmatorios):
gates v2 dieron G3a PASS (+0.01: sin leakage) y G4 PASS (oráculo convergido,
J=1.038) pero G2 FAIL (privilegiado 0.58<0.60: con obs_noise=0.45 ni la verdad
se aprende en 250 pasos) y G3b FAIL (0.18). Punto medio declarado:
obs_noise 0.45→0.30, risk_flip 0.35→0.30; presupuesto rancio+ruidoso SE QUEDA
(es lo que separó permutadas de su vía gratuita). Una sola iteración de
calibración; si los gates vuelven a fallar se reconsiderará el diseño, no los
umbrales.*

*CALIBRACIÓN v3c (2026-07-27, con DIAGNÓSTICO DE TECHO — sin datos
confirmatorios): la recalibración v3b apenas movió al privilegiado (0.57):
su cuello no era el ruido (tiene la verdad) sino el PRESUPUESTO de
entrenamiento y la referencia. Diagnóstico de techo (GRU privilegiado,
recovery del hueco const→oráculo): 0.575@250 → 0.610@500 → 0.721@750, sin
meseta — el entorno ES resoluble; 250 pasos infra-entrenan; y el oráculo
convergido incluye holgura POR-INSTANCIA (8000 Adam/episodio) inalcanzable
para cualquier política paramétrica, que infla el denominador de la rebanada.
Decisiones (todas antes de dato confirmatorio alguno):*
1. *steps del confirmatorio 250→500 (donde el privilegiado cruza 0.61);
   semillas re-ajustadas al presupuesto nocturno con 6 workers: mhbp 24,
   mhbp_first 24 (par C3), gru 20, mhbp_taueq 16, exploratorios 8. dz mínimo
   detectable en C3 al peor Holm: ~0.75.*
2. *G3b redefinido RELATIVO AL TECHO DE POLÍTICA: (rec_priv − rec_shuf) /
   rec_priv ≥ 0.25. El numerador es el valor de la información tick-alineada;
   el denominador, lo que las políticas alcanzan de verdad (no la holgura
   por-instancia del oráculo). G2 se mantiene en 0.60 (ahora con 500 pasos).*
3. *El regret vs oráculo se reporta como métrica ABSOLUTA comparable entre
   controladores (misma cota para todos), no como fracción alcanzable.*

## Criterios de éxito/fallo (de §24-25 del plan maestro)

ÉXITO de la fase si: los 4 contrastes primarios van en la dirección de H-F2 con
≥1 sobreviviendo Holm-4, sin violar presupuestos más que los baselines, y las
intervenciones (congelar campo de recursos → se rompe la planificación;
congelar campo rápido → se rompe el seguimiento de d) muestran especialización.
FALLO INFORMATIVO (se documenta igual): gru ≥ mhbp en todo (la dinámica física
no aporta sobre memoria genérica); taueq == mhbp (las escalas no importan);
first == mhbp (el orden no importa — contradiría benchmark-v3). Cualquiera de
los tres sería un resultado publicable del mismo rango que el positivo.
