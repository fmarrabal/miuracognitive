# PREREG N1B — la tarea de acantilado mudo y la INVERSIÓN de la jerarquía
### v2, 2026-08-09. Pre-datos (campaña no lanzada). El panel de 4 lentes
### (~25 hallazgos, panel_n1b en journal wf_69025d21) DEMOSTRÓ que la v1
### estaba condenada en ambas direcciones; §9 adjudica TODO. Las
### definiciones de §9 SUSTITUYEN a las de §1/§3/§5 donde choquen.

## 1. Hipótesis (la predicción que cierra el paper, ahora en confirmatorio)

En S₅ (familia suave, posterior transparente) la jerarquía medida fue
uniforme 0.467 < dificultad 0.546 < valor ex-ante 0.698 < posterior
0.921 ≈ posterior+valor (VG-B0 = +0.0002). La predicción N1b, con tres
capítulos de evidencia apuntándole (F3b, N2/N3, VG-B0) y requisitos de
diseño medidos en el piloto (FINDINGS_N1B_PILOTO §4):

- **H-N1b-1 (el canal posterior colapsa)**: en una familia donde el
  progreso interno es MUDO, la ventaja del posterior sobre el mejor
  ex-ante se reduce a la detección de llegada:
  Δ(posterior − ex-ante óptimo) « +0.223 (el valor de S₅).
  Umbral: Δ_N1b ≤ 0.08 (un tercio del de S₅) → CONFIRMADA.
- **H-N1b-2 (el valor renace sobre el posterior — la INVERSIÓN)**: el
  sesgo de valor sobre la parada nativa a E[n̄] emparejado (el
  instrumento VG-B0, idéntico) deja de ser cero:
  headroom ≥ max(0.01, 4·SE) y ≥10/12 ckpts → CONFIRMADA
  (en S₅ fue +0.0002±0.0004 con el MISMO instrumento).

Ramas completas:
- H1 ✓ y H2 ✓ → inversión completa; el paper cierra con el resultado.
- H1 ✓ y H2 ✗ → el posterior colapsa pero el valor no lo aprovecha:
  la detección de llegada basta para saturar también aquí (informativo:
  el dominio del valor es más estrecho de lo predicho). Se publica así.
- H1 ✗ (el posterior conserva ≥0.08 de ventaja) → la mudez no se logró
  (ver gate M) o la predicción es FALSA; si el gate M pasó, la
  predicción queda REFUTADA y el paper lo dice.
- Cualquier gate rojo → no se corre el confirmatorio; se reporta el gate.

## 2. Entorno (n1b_env.py, self-tests 7/7)

Camino-en-ciclo: σ sobre 20 elementos como tabla visible; s, t en un
ciclo de longitud EXACTA L ∈ {6,12,18} (clase visible = dificultad
ex-ante coarse); respuesta = d(s→t), d ~ U[1, L−1] invisible.
Todo-o-nada: d solo se conoce al llegar. Stakes en metadatos (motivo
relacional de N2, p_hi=0.15, s_alto=8, ⊥ L y d por stream propio).
Secuencia constante 28; una sola posición supervisada (ARROW).

## 3. Gates (baratos primero; cada uno con umbral y rama)

- **T-N1b0 (cableado)**: self-tests del entorno — HECHO 7/7.
- **A-N1b (aprendibilidad, smoke)**: 1 seed, config n2_endo, pasos como
  n2_e1. PASA si acc(n=24) ≥ 0.60 global y ≥ 0.40 en L=18. Si falla:
  una ronda de diales declarada (≤3 por firewall: N_ELEM, L_SET, pasos)
  y re-smoke; si vuelve a fallar → capítulo se cierra «familia no
  aprendible a esta escala» sin campaña.
- **M-N1b (MUDEZ — el gate que define la familia)**: sobre el ckpt del
  smoke, probe congelado P(converge en el próximo tick | estado, no
  convergido) comparado con el baseline (m, L) [pasos dados + clase]:
  MUDA si AUC(estado) − AUC(m, L) ≤ 0.05. Contraste obligatorio: el
  MISMO probe sobre un ckpt S₅ debe dar > 0.05 (sanidad del
  instrumento: en la familia suave el estado SÍ adelanta — control
  positivo del probe, regla de la casa). Si N1b no es muda → una ronda
  de diales (subir N_ELEM/L; el doubling de punteros es la amenaza);
  si persiste → refutación DE DISEÑO (documentar, no confirmar H1 por
  vía rota).
- **D-N1b (rango dinámico)**: en el smoke, acc(n) debe CRECER con n
  (pendiente n=2→24 ≥ 0.15) y n*_i debe dispersar entre clases
  (E[n*|L=18] − E[n*|L=6] ≥ 2 ticks). Sin rango → sin pregunta.

## 3b. RONDA 1 DE DIALES (2026-08-09, gate A ROJO — pre-datos de la ronda)

Gate A falló de la forma más informativa: acc = prior informado EXACTO
(L6 0.198≈1/5, L18 0.055≈1/17), loss clavada en la entropía del prior
(~2.28) desde el paso 500, halting colapsado a 1 tick. El modelo no
aprendió NI d=1. Diagnóstico mecanicista: la tabla POSICIONAL (posición
i → σ(i)) exige direccionamiento indirecto (valor→posición), un
primitivo que los transformers pequeños no aprenden con una sola
posición supervisada. Dial (ronda 1 de ≤3): σ como PARES (i, σ(i)) en
ORDEN ALEATORIO → recall asociativo encadenado (induction heads), el
primitivo que SÍ se aprende. Secuencia pasa a 48 tokens; pasos 2500→4000.
La mudez no se toca (misma estructura de paseo). Umbrales de A/D/M sin
cambio.

## 4. Campaña (solo si gates verdes y panel adjudicado)

12 checkpoints blind_flat (6 seeds × 2 runs, regla de réplicas), config
y pasos de n2_e1, stakes solo en metadatos (CE plana). Sonda forced_steps
malla {1..6,8,10,12,16,20,24} × 1024 inst/ckpt (n3_sonda parametrizada).

## 5. Brazos de la eval (maquinaria N3 reutilizada, forward-only)

uniforme n̄ · dificultad (p̂|L) · valor ex-ante (argmax ŝ·p̂−λn, fit
isotónico por L en split de fit) · posterior (halting nativo) ·
posterior+valor (VG-B0: logit_offset por clase de ŝ a E[n̄] emparejado
±0.05, δ∈{0.5,1,2,3.5}). Payoff = Σ stake·correct normalizado (idéntico
a N3). Eval 16k común, E[n̄] emparejado entre brazos.

## 6. Estadística

12 ckpts como réplicas; IC-t (n=12) sobre la media pareada por ckpt;
bootstrap por instancia dentro de ckpt para SE de brazos; Holm si se
comparan >2 brazos en la misma familia de hipótesis. NINGÚN nulo sin
control positivo del instrumento (regla 2026-08-09): el probe de mudez
lleva su contraste S₅; el instrumento VG-B0 ya demostró sensibilidad
cero-vs-positivo en S₅ (ahí su cero es el control).

## 7. Presupuesto

Smoke ~15 min GPU · campaña 12 runs ~2-4 h · sonda ~1 h · eval ~30 min.
Total ≤ 6 h GPU. Calibración: máx. 3 rondas de diales (firewall).

## 9. ADJUDICACIÓN DEL PANEL (v2, 2026-08-09, pre-datos) — SUSTITUYE §1/§3/§5

### 9.1 H1 re-operacionalizada (crítico ×3 lentes: la v1 refutaba en falso)

La sola detección de llegada vale +0.24-0.29 en esta familia (cálculo
cerrado de dos lentes: d~U[1,L−1] tiene varianza enorme y el commit paga
la cola completa) — el umbral 0.08 contra el ex-ante puro era refutación
garantizada CON la familia perfectamente muda. Nueva definición:

  Brazo nuevo LLEGADA-ORÁCULO+CAPS: parada en min(c_i, cap_{L,ŝ}) con
  c_i = primer tick con readout correcto y ESTABLE (correcto ∀ t≥c_i, en
  rollout denso), caps de dp_alloc a E[n̄] emparejado.
  **H1 := payoff(posterior nativo) − payoff(llegada-oráculo+caps) ≤ 0.08**
  (el posterior no añade nada MÁS ALLÁ de detectar la llegada = mudez).
  Δ(posterior − ex-ante puro) se reporta como CURVA sobre presupuestos
  E[n̄] ∈ {4..12}, descriptiva (es función del punto de operación).
  Control en S₅: la misma política sintética debe quedar POR DEBAJO del
  nativo allí (la anticipación real de S₅ añade sobre la llegada).

### 9.2 H2 blindada (crítico: winner's curse + palanca posiblemente muerta)

- DOS instrumentos pre-declarados con Holm-2: (a) logit_offset por clase
  de ŝ (VG-B0 idéntico); (b) row_caps por clase de ŝ (dp_alloc, E[n̄]
  emparejado) — la palanca natural en pago todo-o-nada.
- Selección out-of-sample: δ*/caps se eligen en una mitad del eval y el
  headroom se mide SOLO en la otra mitad.
- Nulo por permutación: barajar stake dentro de celdas (L,d), pipeline
  completo incluida la selección; exigir headroom_real ≥ headroom_perm +
  max(0.01, 4·SE) y |media permutada| < 0.003.
- Gate V-N1b (palanca viva, smoke): el barrido δ∈{±0.5,±1,±2,±3.5,±6,±8}
  debe mover E[n̄] ≥ 0.5 ticks; si el offset está muerto (halting
  bimodal), H2 se adjudica SOLO por caps y así se reporta.

### 9.3 Brazos sobre TRAYECTORIA GRABADA (mayor: confound forced-vs-nativo −0.34)

TODOS los brazos = reglas de parada sobre UNA trayectoria grabada por
instancia (record_step_states, 24 ticks, decodificando cada tick en la
posición ARROW + logits de parada grabados por tick). Posterior = parada
nativa point-state; posterior+valor = re-pesado offline de la
distribución de parada con el offset sobre logits grabados (exacto, sin
re-forward); ex-ante/uniforme/dificultad/llegada-oráculo = cortes sobre
la misma trayectoria. Verificado por la lente: λ no realimenta x, las
trayectorias son idénticas bajo intervenciones del halting — el confound
es eliminable por construcción. Bonus: malla densa n=1..24.

### 9.4 Gates v2 (DAG: T → A → (C ∥ D ∥ LL) → M → campaña; contador
### GLOBAL único ≤3 rondas para A/C/D/LL/M; cualquier dial reabre TODOS)

- **A (capacidad)**: acc FORZADA(24) ≥ 0.60 global y ≥ 0.40 en L=18
  (la nativa se reporta al lado, no adjudica).
- **C (el acantilado muerde — nuevo)**: acc_forzada(24) − acc_nativa ≥
  0.05 y E[n̄] ≤ 20; en la fase densa: P(parada antes de c_i) ∈
  [0.10, 0.60]. Diales: β_halt/λ_p arriba o max_steps=16 (d>16
  insoluble → triage real). Tras CUALQUIER cambio de coste: re-M.
- **D (rango, 3 patas)**: pendiente forzada acc(24)−acc(2) ≥ 0.15
  global Y en L=18; dispersión NATIVA E[n̄|L=18] − E[n̄|L=6] ≥ 2;
  región profunda viva: acc(24)−acc(8) ≥ 0.05 en L=18. Salida honesta
  si el horizonte se queda corto: L_SET→(4,8,12) (una ronda).
- **LL (llegada legible — nuevo)**: Spearman(n̄_nativo, d | L) ≥ 0.3 en
  instancias correctas Y AUC(«ya convergido» | estado) ≥ 0.90. Si falla:
  hallazgo de SUSTRATO (el halting global no lee la llegada) — dial de
  halting (pooling en ARROW / head 2 capas) antes de tocar la familia.
- **M (mudez, multi-horizonte — la v1 daba rojo falso)**: etiquetas
  y_h = 1{c_i − n ≤ h}, h∈{1,2,3,4,6}. MUDA ⟺ ΔAUC(h) ≤ 0.05 ∀ h ≥ 2
  (h=1 = horizonte de detección, PERMITIDO: lookahead del backbone).
  Probe: MLP-32 sobre pooled x_n ⊕ one-hot(n) ⊕ one-hot(L), 5 re-inits
  (mediana), split 70/30 POR instancia, SOLO instancias resueltas
  (co-requisito M⇐A, cobertura ≥200/clase), reporte por clase L.
  Baseline: tabla hazard empírica ĥ(n,L) del split de fit. Regresión
  ΔR²(d−m) ≤ 0.05 como segunda pata. Control positivo obligatorio:
  MISMO pipeline sobre ckpt S₅ (gating_wm_cycle_transp_indist_s0, split
  val) — allí debe dar > 0.05 en h≥2. Re-verificación sobre 3 ckpts de
  campaña pre-listados (s3_r0, s5_r1, s8_r0) antes de computar H1;
  mudez por-ckpt: ≥10/12 mudos para interpretar H1 (no-mudos se
  excluyen y reportan como estrato).
- **Diagnóstico de firma de doubling (gratis, con d persistida)**: gap
  acc(d∈pow2) − acc(d∉pow2) > 0.15, o ajuste c_i~log₂(d) mejor que
  c_i~d → saltar diales y decidir rediseño directamente.

### 9.5 Entrenamiento (mecanismo de fallo atacado de frente)

Init del bias de halt_proj = −3 (masa PonderNet en ticks profundos al
arrancar; con init 0 el tick 17 recibe 8·10⁻⁶ de masa y L=18 no tiene
gradiente); β_halt = 0 durante el primer 25% de pasos; pasos = 10 000
con coseno anelado al total (FIX del bug de LR: la v1 no pasaba
max_steps al TrainConfig y el coseno nunca anelaba).

### 9.6 Congelado operativo

Semillas s3..s8 × r0/r1 en ese orden; reserva (s9, s10); exclusión:
NaN o acc_forzada(24) < 0.30 global. Seeds sonda/eval/val = 1999/999/
2999 con offsets de n1b_env. Sonda N1b PROPIA (n1b_sonda): hash de
contenido sobre tokens 2..44 (clase, s, t, pares σ; ARROW/PAD fuera —
el hash de n3_sonda es N2-específico y rompería el dedupe EN SILENCIO),
persistiendo d y L por fila (alineadas tras dedupe). IC primario sobre
medias POR SEED (n=6, t con 5 df); n=12 como sensibilidad; H2 usa el SE
de n=6. Presupuesto: smoke = cronómetro; si t_est > 6 h → 8 ckpts
(4×2, criterio ≥7/8). Eval con p_hi=0.5 sobremuestreado y reponderado
por importancia a la mezcla 0.15/0.85 (reduce ~1.8× el SE del
componente dominante del payoff).

## 10. RONDA 3 DE DIALES (2026-08-10, la última del firewall — con base
## empírica de 6 probes de diagnóstico, pre-datos de la ronda)

Cadena de diagnóstico (no consumió firewall: caza de mecanismo con
pipeline demostradamente sano): copy 1.000/1.000; 1-hop indirecto,
adyacente y denso-6k TODOS en azar exacto (silla: uniforme sobre el
soporte legal, CE=ln20 clavada); denso 30k×batch128 → **GROKKING**
(silla hasta ~9k, acc 1.000 en 15k). Mecanismo: el retrieval en
contexto SÍ se forma, pero necesita supervisión densa Y batch 128 Y
~15k pasos — señal que el régimen 1-posición/batch-32 no da jamás.

Diales de la ronda 3 (todo lo demás idéntico a v2 §9.5):
- Dataset DENSO (N1bDatasetDenso): 8 queries auxiliares tras el ARROW,
  etiquetadas σ(q_j) — enseñan el circuito; van DESPUÉS del readout de
  d (causal ⇒ cero fuga). Los gates puntúan SOLO la posición ARROW.
- batch 128; pasos 30 000 (coseno anelado al total).
- El cronómetro del smoke (§9.6) decide el re-scope de la campaña.
