# Pre-registro — N2: stakes ENDÓGENOS (decidir QUÉ IMPORTA)

> **v2 — 2026-08-03.** La v1 (git 67fd188) fue atacada por un panel de 7
> lentes: **55 hallazgos, 12 críticos**, adjudicados en §12. Tres lentes
> corrieron independientemente el cálculo de headroom con el perfil real y
> convergen. GATE DURO: nada se entrena sin esta v2 + gates VG en verde +
> go explícito de Curro con el coste final.

## 1. Hipótesis (re-atribuida tras el panel)

**HN2**: acoplar el VALOR ENDÓGENO (predicho de las consecuencias propias)
al gobernador mejora el payoff frente al MISMO sistema entrenado con las
mismas consecuencias pero con el acoplamiento cortado. La CE ponderada por
stake YA enseña al backbone/halting qué importa (canal de consecuencias
directo, declarado); HN2 afirma que la vía explícita valor̂→gobernador añade
sobre eso. Nota de honestidad (panel): V̂ es mecánicamente un clasificador
perceptivo entrenado por consecuencias — la endogeneidad está en la SEÑAL
(payoff realizado, sin etiquetas), no en la vía sensorial.

## 2. Entorno (esquina viable; formal en VG1)

- cycle_transp v3, N_MAX=24, **K∈[13,24]** (el suelo F3b + donde vive el
  headroom). β_halt calibrado para E[n̄]∈[8,12] (VG4 verifica alcanzable).
- **Stake latente** ∈ {1, **8**} con P(alto)=p_hi∈{0.10, 0.25} (dial VG1):
  motivo de 2 tokens del VOCAB EXISTENTE, POSICIÓN FIJA sustituyendo tokens
  neutros (longitud constante), presente en TODOS los brazos; stake ⊥
  {K, longitud, posición ARROW, frecuencias, señales interoceptivas tick-1}
  (VG3 ampliado con umbrales). NINGÚN brazo lleva token de stake (la
  paridad oracle se logra por canal, §4).
- Pago: payoff = stake·correcto; train = CE ponderada por el stake asignado
  al brazo; **eval idéntica para todos**: set COMÚN CONGELADO de m≥4096
  instancias, stakes de la regla VERDADERA en la métrica de todos los
  brazos, contraste pareado POR INSTANCIA (bootstrap) además de por seed.
- Métrica primaria: payoff_norm; co-descriptivas estratificadas obligadas:
  acc_alto, acc_bajo, acc no ponderada, E[n̄] por brazo×seed×run.
- **Emparejamiento de E[n̄]** (crítico ×4 lentes): mismo β_halt en train;
  en EVAL, bisección de un offset del umbral de halting por brazo hasta
  |ΔE[n̄]| ≤ 0.2 ticks vs blind; primaria en el punto emparejado, punto
  nativo como descriptivo; sensibilidad ANCOVA payoff~E[n̄] pre-declarada
  (si el signo de C1′ cambia con el ajuste → «no interpretable», rama).

## 3. El módulo de valor (especificación completa)

Sobre el estado del reasoner tras el tick 2, **DETACHED** (el valor se LEE
de la cognición, no la esculpe; test de cableado: ∂CE/∂ψ = 0 y
∂L_val/∂backbone = 0 exactos): dos cabezas — p̂ (corrección propia, BCE con
el acierto realizado; el automodelo online, familia I′) y **stakê**
(softplus, MSE contra payoff/max(p̂.detach(), 0.05), clamp): la cantidad
homóloga al stake del oracle (no el payoff mezclado — crítico #cableado).
stakê se CACHEA por instancia (escalón constante, paridad con el canal del
oracle; añadido a swap_halves y reset_state con test bit-igual) y entra
**detached** al plano.

**Enchufe (vía NUEVA, con disciplina completa — el «mismo enchufe» de v1
no existía en step_mhbp)**: argumento `value_signal (B,)` de step_mhbp →
canales task_criticality/cost_monetary del InteroceptiveSignal, con **veto
por-cabeza**: la columna solo FUERZA al campo risk_priority (test T9:
∂contenido/∂canal = 0 exacto con acoplamiento apagado). **Declaración
(hallazgo del test, 2026-08-03)**: con el acoplamiento de cadena activo, la
propagación INDIRECTA risk→(κ, PSD)→otros campos existe — es la física
certificada del plano, no una fuga; medida al init: 2º orden (9.1e-07 vs
3.6e-05 del efecto sobre halting, 40×). Queda vigilada por el guardarraíl M
y descomponible con la lesión por vías. Ticks 1-2: canal ausente (máscara →
0 exacto; bit-igual con la ruta F3a, test T9b). Ganancia del encoder desde
std=0.05 (virgen, declarado).

## 4. Brazos (5 confirmatorios + 1 solo-piloto; ×6 seeds ×2 runs)

| brazo | pesos CE (train) | V̂ | value_signal | responde |
|---|---|---|---|---|
| **endo** | stake real | entrenado | stakê (tick≥3) | HN2 |
| **endo_noval** | stake real | entrenado, solo log | AUSENTE | atribución (C1′) |
| blind_flat | uniforme = E[stake] | entrenado (→p̂·s̄) | stakê | suelo sin ruido de pesos |
| oracle | stake real | no | stake VERDADERO (tick 1) | techo entrenado |
| gating_endo | stake real | entrenado | vía espejo al halting: bias=tanh(a·stakê_norm), init a=0, misma cota y punto (λ pre-techo), SIN plano | ¿hace falta el plano? |
| (piloto) blind_shuffle | stake PERMUTADO intra-época | entrenado | stakê | control de fuga de V̂ (VG4, front-loaded) |

Inputs idénticos entre brazos (motivo presente en todos); oracle SIN token
(su stake entra por el mismo canal — paridad de formato exacta; la latencia
tick-1 vs tick-3 es el coste estructural de la endogeneidad, cuantificado
en VG1 con el oráculo-retardado).

## 5. Contrastes (jerarquía gatekeeping cerrada)

- **C1′ (primario, α=0.05 unilateral, pareado por seed sobre medias de 2
  runs; Wilcoxon como sensibilidad)**: payoff_norm(endo) > endo_noval.
  [¿el ACOPLAMIENTO añade sobre las consecuencias en la pérdida?]
- **C3 (secundario, condicionado a C1′; conjunción declarada)**: corr
  PARCIAL(E[n]_posterior, stake | K) en endo > endo_noval (Fisher-z
  pareada) Y ≥ ½·r_oráculo(diales VG1). σ_run(corr) medida en VG4.
- **C2 → DESCRIPTIVO** (crítico #estadística: con n=6 era vacuo o
  imposible): fracción capturada = (endo − blind_flat)/(oracle −
  blind_flat) con IC bootstrap.
- **Manipulation-checks de interpretabilidad** (si fallan → rama «el techo
  no se realizó al nivel entrenado», sin veredicto HN2): (a)
  oracle − blind_flat ≥ ⅔·headroom_VG1 (IC-inf); (b) corr(E[n], stake) en
  oracle ≥ ½·r_oráculo_VG1.
- Sanity floor: endo > blind_flat (descriptivo).
- **Árbitro de mecanismo (coste ~0)**: lesión EN EVAL endo_cut
  (value_signal fijado a su media): endo > endo_cut aísla el canal en
  inferencia; pre-declarado para la rama C1′✓∧C3✗.

**Guardarraíles** (todos con la regla de réplicas; comparador contemporáneo
= blind_flat): batería M/I′ on-policy con V̂ ACTIVO sobre instancias sin
motivo (stake efectivo 1; distribución de V̂ fuera de soporte reportada);
M3 dispara si Δ̄ < −max(0.03, 2σ_run^UCB90/√6) Y p<0.05 pareada; acc no
ponderada: endo ≥ acc_oráculo_valor(diales) − 2σ_run^UCB90 (re-anclado al
coste PREDICHO del óptimo — el −0.03 absoluto mataba a C1′ en la esquina
viable, crítico ×3 lentes); fuga: AUC media de V̂ vs stake en las celdas
blind_shuffle del PILOTO > 0.5 + 3·SE → no se lanza el confirmatorio
(front-loaded; en confirmatorio, AUC parcial controlando p̂ con permutación
p<0.001 → cuarentena + auditoría VG3 fechada).

## 6. Kill-gates (orden estricto; cortafuegos: diales cerrados
## {p_hi, β_halt objetivo, (s_alto solo hacia ARRIBA)}, ≤3 rondas commiteadas)

- **VG1 (CPU, formal)**: con el perfil real y asignación ÓPTIMA EXACTA
  (DP por clase K×stake, no greedy), computar por celda de diales el TRÍO
  {headroom := payoff(oráculo valor+K) − payoff(**oráculo K-solo
  stake-ciego**) a E[n̄] igual; coste_acc del óptimo de valor;
  σ_sim(payoff_norm) por bootstrap de oráculos} + oráculo-RETARDADO
  (información desde tick 3, lag τ_risk) para el coste de endogeneidad.
  Aprueba solo celdas con headroom ≥ 4·σ_sim Y ≥ 2×MDD(σ de F3a-R como
  prior, fórmula escrita) Y coste_acc compatible con el guardarraíl
  re-anclado. Candidatas del panel: (E[n̄]=8, s=8, p=0.10) h=+0.070
  c=−0.019; (12, 8, 0.25) h=+0.052 c=−0.025. El pase en esquina degenerada
  (w_alto>0.6 del peso) se declara como tal y activa las co-primarias
  estratificadas como criterio. Umbrales definitivos re-anclados a
  σ_run(payoff) de VG4 por enmienda fechada (orden: VG1 provisional → VG2b/
  VG3 → VG4 → enmienda → VG1 final; el re-chequeo consume ronda).
- **VG2b (percepción desde el ESTADO, no desde el input)**: probe lineal
  del motivo desde estados tick-1/2 de ckpts F3a (CPU/min) y de un run
  corto con CE ponderada; AUC ≥ 0.9. Si falla: «motivo indiscoverable
  desde el estado» → N2 no se lanza (el fallo sería de percepción y el
  negativo nacería mal atribuido — crítico #ramas).
- **VG3 (independencias, ampliado)**: stake ⊥ {K, longitud, ARROW,
  frecuencias, interocepción tick-1 en ckpt virgen} con umbrales; acc(n,K)
  condicionada al motivo igual (exige la ruta forced en el adaptador —
  instrumento ANTES del gate, lección G0); re-chequeo post-entrenamiento
  pre-declarado como diagnóstico.
- **VG4 (σ_run, ampliado a 12 runs)**: 3 seeds × {endo, blind_shuffle} ×
  2 runs, longitud confirmatoria + batería; estima σ_run de payoff_norm,
  M3 Y corr; run ≡ {semilla de orden de datos + muestreo de halting +
  kernels no deterministas} (misma init por seed; verificación: σ_run>0 y
  del orden F3a-R); σ_int desde el archivo F3a-R (48 celdas, prior
  declarado); umbrales usan UCB90; seeds piloto EXCLUIDAS del
  confirmatorio; el chequeo de fuga de V̂ corre AQUÍ.

## 7. Coste honesto (dos etapas pre-declaradas)

Etapa 1 = {endo, endo_noval, blind_flat} ×6×2 = 36 + ood-endo (batería M3)
12 = 48 celdas ≈ 25 h serial ≈ **2 noches** (2 workers). Decide C1′.
Etapa 2 = {oracle, gating_endo} ×6×2 = 24 ≈ 12.4 h ≈ **1 noche**, SOLO si
C1′ pasa (mismas seeds; el test no cambia por la secuenciación). Gates:
~5-6 h GPU (VG4 4.1 + VG2b/VG3 ~1) + CPU. Sweep de emparejado ≈ +2 h.
Smoke de 1 celda N2 real antes de fijar el min/celda. **Total: 2-3 noches
+ gates. El confirmatorio no se lanza sin el go de Curro.**

## 8. Ramas de veredicto (columnas: veredicto · arquitectura oficial · N3)

| resultado | veredicto | adopción | ¿N3? |
|---|---|---|---|
| C1′✓ C3✓ guardarraíles✓ | HN2: el acoplamiento del valor al gobierno paga | V̂+canal risk entra | SÍ |
| C1′✓ C3✗ | paga sin mover cómputo → árbitro endo_cut: si endo≈endo_cut, era la CE (HN2 ✗); si endo>endo_cut, canal real por vía no-cómputo (se caracteriza) | según árbitro | según árbitro |
| C1′✗, endo_noval>blind_flat | las CONSECUENCIAS en la pérdida bastan; el acoplamiento explícito no añade (negativo informativo, sin re-tune — anti-F3c portado) | endo_noval (más simple) | SÍ, sobre consecuencias-en-pérdida |
| C1′✗ ∧ AUC(V̂,stake) alta | fallo de ACOPLAMIENTO (sabe qué importa, no lo usa) → diagnóstico declarado: lesión de vía risk | — | bloqueado hasta diagnóstico |
| C1′✗ ∧ AUC baja | fallo de PERCEPCIÓN — VG2b debió verlo: se audita el gate antes de declarar | — | bloqueado |
| gating_endo ≈ endo | el plano sobra como lector de valor (parsimonia; |Δ|<2σ_run/√12) | V̂→halting sin plano | SÍ |
| gating_endo > endo | INTERFERENCIA del plano (el riesgo F2 del ROADMAP, escrito) | V̂→halting; el acoplamiento V̂→risk NO se hereda | SÍ, sin plano-valor |
| endo > gating_endo | el plano como lector de valor: primer positivo de esta familia en el arco → exige RÉPLICA propia antes de adoptarse | provisional | tras réplica |
| contrastes✓ guardarraíl✗ | adopción BLOQUEADA; el toque va al procedimiento F3a-R de 3 capas (control fresco → incumbente fresco, ~12 celdas presupuestadas como contingencia) ANTES de veredicto | — | no hasta resolver |
| manipulation-checks ✗ | el techo no se realizó al nivel entrenado: sin veredicto HN2; se audita el gap oráculo/entrenado | — | no |
| VG fallan tras ≤3 rondas | S₅ no ofrece headroom de valor factible → N2 no se lanza aquí; se diseña la familia de tarea con acantilados (hipótesis nueva, prereg nuevo) | — | — |

## 8b. ENMIENDAS de calibración del entorno (≤3 rondas; 2 consumidas)

**Ronda 1 (2026-08-04)**: el motivo par-concreto (5,6) FALLÓ VG3b (Δacc
−0.025/−0.056 en n=12/24 — el token 5 es la permutación identidad e
interfiere con la supervisión) y VG3a marginal (r_E[n] idiosincrático por
token, hasta 0.19). → motivo re-diseñado como RELACIÓN: stake alto ⟺ slots
IGUALES (alfabeto {5,6,7,8}; marginal por-token idéntica entre clases POR
CONSTRUCCIÓN → los efectos por-token se cancelan).

**Ronda 2 (2026-08-04)**: la relación de igualdad es XOR-like sobre el
estado pooled — ilegible para lector lineal (AUC 0.75-0.93) y el pooled
diluye los slots 1/T. → V̂ (y el probe VG2b, misma familia) = MLP de 1 capa
oculta (32) sobre [pooled ⊕ estado@slot0 ⊕ estado@slot1] (las posiciones de
anuncio son parte del SPEC del entorno: auto-observación legítima). Probe
certificado con clases balanceadas (la percepción no depende del prior).

**Resultado post-rondas**: VG2b AUC = 1.000 (6/6 ckpts), VG3a |r| < 0.023
(todos los confusores), VG3b |Δacc| ≤ 0.009 pareada (6/6 checks) —
n2_vg23.json. Fix adicional del runner (pre-VG4, detectado en smoke): las
CONSECUENCIAS de cada brazo son SUS pagos — en blind_shuffle, L_val ve los
stakes BARAJADOS (si V̂ aprendiera el stake real ahí, sería la fuga que el
control existe para detectar; con el bug veía el real y daba AUC 0.62).

## 8c. ENMIENDA VG4 (2026-08-04, pre-confirmatorio): números medidos,
## diseño final de V̂ y umbrales definitivos

**Piloto VG4** (12 runs endo/blind_shuffle ×2500 pasos) + 3 celdas de
verificación (vg4b/c/d):
- **σ_run(payoff_norm) = 0.012** (6 pares de réplicas; UCB90 ≈ 0.018) —
  4× más fina que la σ_run de M3 (F3a-R). σ_int ≈ 0.012 → σ_d ≈ 0.021 →
  **MDD(n=6, unilateral, 80%) ≈ 0.025**.
- Fuga: blind_shuffle AUC_V = 0.486-0.500 en 6/6 ✓ (el control controla).
- **β_halt = 0.02 → E[n̄] = 7.9** ✓ (0.01→11.5, 0.03→6.6, 0.05→5.7).
- **Tres fixes del módulo V̂, cada uno con diagnóstico** (commits c0dc58b,
  dacca4d, 2b21309): (1) regresión del payoff CRUDO (el target-cociente era
  inaprendible: AUC 0.5 en 6/6); (2) **sensor de slots = embeddings crudos**
  — el entrenamiento de la tarea SUPRIME la info de slots del estado
  contextualizado (probe en tronco N2: 0.79 vs 1.00 en embeddings; la
  advertencia del panel confirmada por el re-chequeo post-entrenamiento);
  (3) **replay estratificado** («rumiar las consecuencias»: buffer
  episódico, 50/50 por clase de payoff, k=4, opt propio) — el single-pass
  con 10% de positivos no engancha la XOR. Resultado: **AUC_V = 0.869**
  (umbral de enganche declarado: ≥ 0.8; el arranque tardío — payoff no
  informativo hasta que acc despega — es coste estructural de la
  endogeneidad, cuantificado por C2 contra oracle).

**Celda confirmatoria elegida** (de las 14 aprobadas por VG1, con los
umbrales medidos): **K∈[13,24], s_alto=8, p_hi=0.15, β_halt=0.02**
(E[n̄]≈8): headroom = 0.087 ≥ max(4·σ_run^UCB = 0.070, 2·MDD = 0.050) ✓;
coste_acc oráculo = −0.030 (guardarraíl re-anclado a ese valor − 2σ_run);
r_pb oráculo = 0.70 → **umbral C3 = 0.35**; manipulation-check:
oracle − blind_flat ≥ ⅔·headroom = 0.058 (IC-inf).

## 9-11. Instrumentación exigida antes de gates

value_signal en step_mhbp + veto por-cabeza en InteroceptionEncoder + ruta
forced en el adaptador (el NotImplementedError, pendiente desde F3a) +
caché de stakê en swap/reset + tests de cableado (∂CE/∂ψ=0,
∂wm/∂canal=0, bit-igualdad con canal ausente vs step_mhbp F3a, swap
bit-igual) + runner N2 (CE ponderada, eval congelada, offset-bisección,
payoff/estratificadas) + escaneo VG1 exacto (DP).

## 12. Registro de decisiones sobre los 12 críticos del panel

baseline VG1 → oráculo-dificultad (×4 lentes); brazo endo_noval y C1′
(×4); factibilidad conjunta headroom×coste_acc y guardarraíl re-anclado al
óptimo (×3); emparejado E[n̄] operacionalizado (×4); canal value_signal
como vía nueva con veto por-campo (#cableado); detach total de V̂ + VG2b
desde el estado (#fuga/#ramas); stakê factorizado del payoff (#cableado);
C2 degradado + manipulation-checks (#estadística/#ramas); C3 re-anclado al
oráculo con corr parcial (#estadística/#políticas); jerarquía gatekeeping
(#estadística); coste con dimensión de régimen y etapas (#coste); tabla de
ramas completa con adopción/N3 y contingencia F3a-R (#ramas).
