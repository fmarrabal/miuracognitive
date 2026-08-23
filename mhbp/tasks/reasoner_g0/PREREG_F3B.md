# Pre-registro — Fase 3b: gobierno multiescala a escala de SESIÓN (doble objetivo)

> **v2 — 2026-08-02.** La v1 (2026-08-01, git d417462) fue atacada por un panel
> adversarial de 7 lentes: **58 hallazgos, 11 críticos**, todos adjudicados en
> §14. Esta v2 los incorpora ANTES de implementar o entrenar nada.
> **GATE DURO**: ninguna celda confirmatoria se lanza sin esta v2 commiteada
> en git con la tabla de ramas completa (§12) y el entorno congelado por hash
> tras los kill-gates (§10). Doble objetivo: (A) test multiescala GENUINO del
> plano; (B) brazo de re-cableado que responde al hallazgo M3 de F3a.

## 1. Preguntas e hipótesis

- **HA (multiescala)**: cuando el entorno construye tres escalas reales
  (tick / instancia / sesión-presupuesto-régimen), el plano de 4 campos con
  escalas ordenadas gobierna mejor que (i) un campo único persistente e
  (ii) un estado recurrente genérico (GRU) con las MISMAS observaciones y
  actuadores. Afirmar HA exige superar a AMBOS (unión-intersección).
- **HB (re-cableado)**: el toque a M3 on-policy de F3a (−0.055, horneado en
  pesos por modulación por-tick) desaparece si los gates de WM y el
  block_gate se congelan POR INSTANCIA, sin coste de gobierno.

## 2. Entorno de sesión

- **Sesión** = E=6 instancias de composición S₅ (cycle_transp) secuenciales.
  Estados del CONTROLADOR persisten entre instancias según §6; WM y estado
  del reasoner se resetean POR INSTANCIA; reset total entre sesiones.
- **Presupuesto duro** B_total de ticks por sesión (operacionalización §6.1),
  hecho del entorno en TRAIN y EVAL. Al agotarse una fila su instancia en
  curso se corta (λ:=1) y las restantes corren con n=1. Calibración en GS1
  con referencia INDEPENDIENTE de brazo: ticks-hasta-primer-decode-correcto
  de un modelo de referencia convergido (sondas G0.1), por K.
- **Regímenes de dificultad**: 2 regímenes como DISTRIBUCIONES de K con
  solape deliberado (provisional: fácil K∈[8,16], difícil K∈[12,22]; suelo
  ajustado en gates por acc(n=1|K)≤0.2, medible con ckpts F3a) — una sola
  instancia NO identifica el régimen; filtrar la historia sí (GS2).
  Permanencia ESTOCÁSTICA geométrica (media 3) y fase inicial aleatoria:
  MI(régimen; posición) baja por construcción (verificada en GS3).
- **Stakes**: observables EX ANTE (token en el input, visible para TODOS los
  brazos), composición FIJA por sesión (exactamente 2 de 6 a ×4, posiciones
  permutadas), independientes de régimen y K por construcción.
- **Métrica de sesión**: score = Σ stake_i·correcto_i, **normalizado** por el
  máximo de la sesión (∈[0,1]).
- **Streams**: RNG del entorno PROPIO e independiente del modelo; seed ⇒
  (init modelo, stream_train, stream_eval) con sesiones IDÉNTICAS entre los
  7 brazos (pareado a nivel de sesión); eval con offset +800000 y schedules
  frescos; assert por hash de las primeras 64 sesiones entre brazos antes de
  lanzar la parrilla.

## 3. Equidad informacional y de actuación (respuesta a los críticos #1-2)

**Conjunto de observación IDÉNTICO para los 7 brazos** (difieren en CÓMO lo
procesan, jamás en si lo ven): fracción gastada spent/B_total, fracción
restante, evento de techo (incluye forzados), instancias restantes /E, stake
de la instancia, esfuerzo, y la interocepción de completitud (margen +
entropía del decode en ARROW, por tick — se computa en la rama COMPARTIDA de
miura, no solo en is_mhbp). **Paridad de actuadores**: todos los brazos con
controlador (mhbp*, hbp_sess, gru_sess, react_sess) emiten las MISMAS vías
acotadas del adaptador F3a — sesgo de halting tanh ±1, presión aditiva de
presupuesto softplus·señal·(1+0.8·tanh(V·estado)), gates WM, block_gate —
cambiando solo la FUENTE del estado (plano / campo único / GRU / features
reactivas). Tabla señal×brazo y actuador×brazo publicada con el panel de
cableado (GS3).

## 4. Brazos (7)

| brazo | controlador | persistencia inter-instancia | gates WM/block | seeds |
|---|---|---|---|---|
| mhbp_sess | plano 4 campos | ✓ (campos+allo) | por-tick | 8 |
| **mhbp_sess_inst** | plano 4 campos | ✓ | **por-instancia (WM Y block_gate)** | 8 |
| mhbp_noper | plano 4 campos | ✗ (reset/instancia; MISMA interocepción) | por-tick | 6 |
| hbp_sess | campo único (VEI) | ✓ (h_t) | por-tick | 8 |
| gru_sess | GRUCell (estado igualado en dim al plano) | ✓ (hidden) | por-tick | 8 |
| gating_wm | ninguno (observaciones §3 como features al halting) | — | base | 6 |
| react_sess | política reactiva sin estado (features §3 → actuadores) | — | por-tick | 3 (etiquetado) |

- mhbp_noper recibe interocepción IDÉNTICA (incl. fracción gastada): D2 aísla
  «persistencia del ESTADO más allá de la señal». Nulo declarado: «la señal
  basta; el campo no integra nada que la señal no dé».
- gru_sess: mismas interocepciones y actuadores, hidden persistente — sin él,
  D1+ no distingue plano de «cualquier estado recurrente» (regresión vs F2b).
- react_sess = suelo de pura observabilidad (cuánto pacing da solo mirar).

## 5. mhbp_sess_inst — especificación exacta (crítico #cableado-7)

Gates WM y block_gate computados del estado PERSISTIDO en la frontera de
instancia, ANTES de tick alguno de la instancia (hipótesis de mecanismo
limpia: sin fluctuación por-tick NI feedback intra-instancia), cacheados en
el adaptador (`per_instance_gates=True`), SIN detach (coherente con BPTT §7).
Instancia 1 post-reset: estado cero → gates neutros (0.5 / 0) — documentado;
afecta por igual a la referencia por posición. La vía de halting y la presión
siguen por-tick (son el gobierno del cómputo; la hipótesis M3 es sobre las
vías de CONTENIDO). Cubre AMBAS candidatas de F3a (WM y block_gate); un D3
nulo falsa la clase «modulación por-tick de vías de contenido», no una vía.

## 6. Cableado declarado (críticos #cableado-1..6)

**Tabla de persistencia por tensor** (mhbp_sess/inst): u,w de los 4 campos ✓;
estado de alostasis ✓; h_t → SOBRESCRITO por instancia (h_t := kick, no
acumula — el acumulador monótono era un contador de posición que contaminaba
GS3); `_last_mod` → LIMPIADO en frontera (pre-bucle neutro); wm_box ✗
(recreada por instancia); RunningNorm: EMA solo en train, congelada en eval
(check del panel). hbp_sess: h_t (=su campo) persiste + kick por instancia.
gru_sess: hidden persiste. Reset entre sesiones verificado bit-cero (test).

**6.1 Presupuesto operacionalizado**: gasto por instancia = ticks EJECUTADOS
por fila hasta su corte POR-FILA (máscara activa por fila; no el
`remainders.max()` por batch). Fila con presupuesto agotado a mitad de
instancia: λ:=1 en ese tick (semántica del override N_max existente);
instancias posteriores de esa fila: n=1. Requiere ticks enmascarados en el
plano (index_copy sobre u/w/h_t/allo; RunningNorm congelada para filas
inactivas) — test unitario: estado de fila inactiva bit-idéntico. GS1/GS2
usan EXACTAMENTE esta contabilidad.

**6.2 Canal de gradiente para racionar**: (i) la fracción gastada entra como
SEÑAL a la vía de presión (diferenciable dentro del tick) en todos los
brazos con actuadores; (ii) BPTT de sesión COMPLETA — sin detach en
fronteras de instancia (9.3 GB @ batch 32 medidos; clip de gradiente y
monitor de norma a través de ≥60 ticks encadenados); (iii) GS5 verifica
aprendibilidad empíricamente antes del confirmatorio.

**6.3 Interocepción de gasto**: fracciones adimensionales (spent/B y
remaining/B ∈[0,1]) por el canal energy_cost (token_cost = evento de techo,
incluye forzados) — auto-normalizadas: el OOD de presupuesto cambia la
PENDIENTE (el test legítimo), no la escala; vía en fp32 hasta el encoder.

**6.4 τ re-derivadas** para el horizonte de sesión (regla F3a «~2 unidades
del tiempo propio por horizonte gobernado»): **τ=(1, 3, 10, 32)** — fast=tick,
risk≈intra-instancia, deliberative≈instancia (~10 ticks), resource≈régimen/
presupuesto (~30-60 ticks). Las (1,3,6,12) de F3a dejaban la escala lenta
huérfana (crítico #cableado-5).

**6.5 Pérdidas auxiliares**: **β_homeo=0 en F3b** para todos los brazos mhbp
(mean u² castigaba exactamente la memoria inter-instancia que D2 contrasta;
la estabilidad es por construcción — F1); asimetría con el homeo propio de
hbp_sess documentada. **NADIE lleva L_halt_aux** (el presupuesto duro YA es
el arreglo de entorno; regla de oro). Tabla brazo × {params del controlador,
dim de estado persistente, pérdidas+β} publicada antes de entrenar; dim de
estado igualada entre mhbp/hbp_sess/gru_sess (o desviación declarada).

## 7. Entrenamiento

- **Desde cero los 7 brazos** (warm-start F3a PROHIBIDO en confirmatorias:
  contaminaría D3/GS4 con el desacople ya horneado; los ckpts F3a solo
  calibran gates con forward_forced).
- **Currículum de 2 fases declarado** (crítico #coste-1): fase A pasos
  0-1000 con presupuesto no vinculante (≥1.5× demanda) para aprender la
  tarea; fase B pasos 1000-2400 con B_total duro. Piloto GS4 verifica
  acc base ≥0.65 al final de fase A.
- Batch **32 fijo e idéntico** (16 dominado: mismo t/paso, mitad de datos);
  sonda de batch 64 pre-lanzamiento — si t/paso ≤1.1×, se adopta 64 por
  enmienda fechada ANTES de entrenar.

## 8. Métricas y evaluación

- **Primaria**: score de sesión normalizado, in-dist.
- **Co-primaria OOD**: compuesto declarado = media de z-scores de
  {presupuesto ×0.6, ×1.5, permanencia 1} con pesos iguales. E=12 (B_total
  escala ∝E) = secundaria etiquetada, solo brazos de D1 si el calendario
  aprieta.
- Secundarias: pacing (corr ticks-dificultad SOLO sobre instancias no
  forzadas; versión con todas como robustez), tasa de mordida y forzados por
  brazo (obligatoria), I', M3-on-policy por instancia, trazas de campos
  (descriptivo exploratorio).
- **N_eval**: ≥256 sesiones/condición/celda, streams idénticos entre brazos;
  N final fijado en gates: SE_intra ≤ ⅓·sd entre-seeds (medible en GS1).
- Batería M/I': SIEMPRE sobre streams-sonda con presupuesto ∞ (sin
  forzados), estratificada por posición-en-sesión, pareada por posición y
  seed; M2-swap en condición E=1; instancias forzadas excluidas (fracción
  reportada). Batería por-instancia completa en el trío {mhbp_sess,
  mhbp_sess_inst, hbp_sess}; mhbp_noper: batería E=1 (guardarraíl mínimo).

## 9. Contrastes y estadística

**Familia Holm-3** (superioridad, unilateral, pareado por seed, n=8):
- **D1-in**: mhbp_sess > hbp_sess **Y** mhbp_sess > gru_sess (unión-
  intersección: ambos a la α del slot) en la primaria in-dist.
- **D1-ood**: ídem en el compuesto OOD.
- **D2**: mhbp_sess > mhbp_noper (primaria in-dist; n=6 pareado).

**Guardarraíles fuera de familia** (no-inferioridad, α=0.05 unilateral):
- **D3(i)**: M3-on-policy de mhbp_sess_inst ≥ hbp_sess_F3b − **0.04**
  (margen = <75% del daño F3a 0.055; potencia ~90% a n=8 con sd 0.034;
  la referencia es hbp_sess EN F3b — el 0.735 de F3a es solo contexto).
  Agregación: media pareada primaria, estricta como sensibilidad.
- **D3(ii)**: score(mhbp_sess_inst) ≥ 0.95·score(mhbp_sess).
- Guardarraíl M de F3a (M1≥0.9, M2, M3 abs ≥0.6) para todo brazo mhbp.
- **Cláusula anti-overclaim** (portada de F3a v2.5): «la multiescala paga»
  exige ADEMÁS mhbp_sess > gating_wm en la primaria.

**D1′ condicional pre-declarado** (crítico #ramas-2): si D3(i) recupera y
D3(ii) no cuesta → se ejecuta D1′ = mhbp_sess_inst vs {hbp_sess, gru_sess}
(secundario, α=0.05) y la pregunta del programa se responde con D1′;
mhbp_sess sigue de brazo primario SOLO por comparabilidad con F3a (escrito).

**MDE honesto**: n=8, Holm-3, t pareada df=7 → dz≈1.2 (n=6: ≈1.55). Un D1
nulo NO licencia «no paga»: el cierre negativo exige que el IC95 superior de
Δ_D1 quede por debajo de ⅓ del headroom perceptivo medido en GS1 (margen de
equivalencia anclado al entorno); si el IC cabalga, la rama es «no
concluyente» (§12). Direcciones: D1/D2 unilaterales >; D3 no-inferioridades
con margen. Rama pre-declarada para «IC cabalga el margen» en D3(i): se
adopta el cableado mecánicamente más limpio (sess_inst) si D3(ii) se cumple.

## 10. Kill-gates (todos ANTES del confirmatorio, con políticas NO aprendidas)

**Cortafuegos de calibración**: los gates se calibran EXCLUSIVAMENTE con
políticas de oráculo/heurísticas (ningún brazo aprendido corre antes de
congelar el entorno); diales móviles cerrados {factor de B_total, permanencia,
ratio de stakes, suelo de K}; ≤3 rondas de calibración, cada una commiteada
con fecha y motivo; al pasar, hash de la config del entorno escrito aquí por
enmienda. Cambio posterior = enmienda nueva.

- **GS1 (el presupuesto muerde — escalera de nulos)**: uniforme <
  mejor-taper-posicional < stake-greedy < reactivo-última-K <
  filtro-bayesiano-de-historial < oráculo-K. Exige: (a) mordida ≥60% de
  sesiones bajo uniforme; (b) headroom PERCEPTIVO (oráculo − mejor política
  ciega-a-K) con IC-inferior ≥ +8 puntos de score normalizado; (c) headroom
  aprendible ≥ 2× la mínima diferencia detectable implicada por la varianza
  medida. ≥1000 sesiones por política, IC bootstrap.
- **GS2 (la escala lenta paga con información APRENDIBLE)**: filtro bayesiano
  sobre la historia de K experimentada > política última-instancia (IC-inf
  ≥ +5%) **Y** > contador-posicional. Si la persistencia-de-1 ya capta todo,
  el entorno se re-calibra (más solape de K) o F3b no se lanza.
- **GS3 (sin fugas — test de proxies)**: MI(régimen; posición/stake/
  observables) < umbral declarado en ≥1000 sesiones; tabla de paridad
  señal×brazo; check h_t: con instancias i.i.d., |corr(señal, índice)| <
  0.1 en rollout sin entrenar; verificación model.eval() (EMA congelada) en
  TODOS los bucles de eval; test de independencia de instancias
  intra-sesión; controles de no-contenido: barajar orden en eval no cambia
  acc por instancia más allá del efecto presupuesto/régimen, y reset del
  plano a mitad de sesión no cambia la respuesta de la instancia en curso
  (solo el pacing).
- **GS4 (la persistencia no rompe M)**: piloto k=4 seeds × entrenamiento
  acortado declarado (1200 pasos: 800A+400B) sobre mhbp_sess Y hbp_sess;
  umbrales: M1≥0.9, M3 abs ≥0.6, Δ(M3 on-policy vs hbp_sess piloto) ≥ −0.06;
  acc base ≥0.65 tras fase A. Rama si falla en mhbp_sess: mhbp_sess_inst
  hereda el rol primario y GS4 se repite; si también falla → F3b se aborta y
  «la persistencia del campo rompe M» se publica como resultado del
  programa. Limitación declarada: GS4 verde NO garantiza M intacto a 2400
  pasos; el guardarraíl M del confirmatorio sigue vigente.
- **GS5 (aprendibilidad del racionamiento)**: brazo-trampa con K observable
  + presupuesto, ~500 pasos: debe emerger pacing no uniforme
  (corr(ticks, K) > 0.2). Si ni el tramposo raciona, la señal de
  entrenamiento está rota y NO se lanza nada.

## 11. Implementación exigida (con tests, antes de gates)

Ticks enmascarados por fila en el plano + ruta forced/active_idx del
adaptador (hoy NotImplementedError) + flag persist_plane en miura (reset
bit-cero entre sesiones) + per_instance_gates + h_t sobrescrito + canal
energy_cost + observaciones §3 en los 7 brazos + harness de sesión
(entrenamiento, eval, oráculos de GS1/GS2 con forward_forced sobre ckpts
F3a). Smoke-run de 50 pasos × 7 brazos (~30 min) la tarde antes de la
noche 1.

## 12. Ramas de veredicto (completas; adopción para Nivel 2 en cada terminal)

Regla de desempate pre-declarada: a igualdad de score dentro de margen, gana
el diseño con M más limpio; a igualdad de M, el más simple (menos estado).
Combinaciones no probadas (p.ej. hbp_sess+gates por-instancia) = «requiere
celda puente», jamás adopción sin dato.

| resultado | veredicto | adopción N2 |
|---|---|---|
| D1+ (ambos comparadores) & anti-overclaim ✓ & M limpio | la multiescala paga con escalas reales | mhbp_sess (o sess_inst vía D1′) |
| D1+ pero gating_wm ≥ mhbp_sess | overclaim vetado: nadie supera a no-tener-campo | gating_wm+observaciones; el plano no avanza |
| D1+ solo vs hbp_sess (gru_sess empata) | «cualquier recurrencia basta»: el plano no aporta sobre memoria genérica | gru_sess (más simple) |
| D1 nulo, IC < ⅓ headroom GS1, guardarraíles limpios | **CIERRE NEGATIVO de la hipótesis multiescala** (§13) | hbp_sess o gru_sess por score; según desempate |
| D1 nulo, IC cabalga | no concluyente (se declara tal cual; sin re-test — §13) | el más simple no-inferior |
| D1− | FAIL del plano a escala de sesión (modo F2 replicado a sesión) | hbp_sess/gru_sess; el plano queda para el paper como negativo |
| D2+ | la persistencia es load-bearing | (modula la lectura de D1) |
| D2 nulo | la señal basta; el estado no integra nada extra | preferir noper (más simple) si D1 vino por observación |
| D2− | la persistencia DAÑA (estado rancio) | mhbp_noper es el default del plano |
| Empate universal con GS1/GS2 verdes | «la estructura existe pero ninguna política entrenada la explota» — hallazgo de optimización, NO cierra ni apoya HA | UN re-intento declarado (2× pasos, una vez); si persiste, se reporta así |
| D3: recupera & no cuesta | HB confirmada: el toque M3 era la modulación por-tick de vías de contenido | **sess_inst = variante oficial del plano**; D1′ responde el programa |
| D3: recupera & cuesta >5% | trade-off inverso: mecanismo limpio o gobierno | se adopta sess_inst (prioridad M, regla del arco) y se declara el coste |
| D3: no recupera & no cuesta | HB falsada: el desacople NO era (solo) la modulación por-tick de WM/block | trade-off F3a se declara permanente; sin barrido de re-entrenos (cerrado) |
| D3: no recupera & cuesta | dominada | se descarta sess_inst |
| GS1/GS2/GS3/GS5 no pasan tras ≤3 rondas | el entorno no certifica la estructura → F3b no se lanza (sin GPU quemada) | se reporta el diseño fallido |

## 13. Compromiso anti-regresión («el test genuino es el siguiente»)

Si GS1 y GS2 pasan (estructura lenta demostrada POR CONSTRUCCIÓN) y D1 es
nulo-concluyente con guardarraíles limpios, la hipótesis «el plano
multiescala paga» se CIERRA en negativo: no habrá F3c con sesiones más
largas ni re-tune del plano (ω, τ, ganancias) seguido de re-test. Post-hoc
permitidos (etiquetados exploratorios): trazas de campos, especialización de
escalas, descriptivos de pacing. Cualquier re-test posterior exige hipótesis
nueva y prereg nuevo.

## 14. Presupuesto honesto y decisiones del panel

**Coste medido** (sonda Blackwell): mhbp 4.05 s/paso (batch 32, BPTT
completo, 9.3 GB); hbp/gru ≈1.7 s; gating ≈0.72 s. Entrenamiento (2400
pasos): 24 mhbp-celdas ≈ 65 h + 16 hbp/gru ≈ 18 h + 9 ligeras ≈ 5 h ≈ **88 h
secuenciales** + instrumentos/batería/OOD ≈ 10 h. Con 4 workers Blackwell
(sonda de contención de 15 min ANTES de declarar el calendario; Windows/WDDM)
≈ **3-4 noches**. Si entra la RTX 4000 Ada: celdas asignadas por SEED
completo, nunca por brazo (confound de dispositivo). Recortes pre-declarados
si aprieta: E=12 solo brazos D1; batería por-instancia solo el trío.

**Decisiones sobre los 11 críticos**: equidad informacional total (§3) en
lugar de brazos de descomposición; completitud para todos (§3); permanencia
estocástica + solape de K + GS2 aprendible (§2, §10); stakes ex ante fijos
(§2); margen D3 0.04 + n=8 en el cuarteto decisorio (§9); MDE honesto +
margen de equivalencia anclado a GS1 (§9); gru_sess añadido (§4); presupuesto
por-fila con λ-clamp + ticks enmascarados (§6.1); gradiente: señal de gasto +
BPTT completo + GS5 (§6.2); currículum 2400 pasos (§7); tabla de ramas
completa en esta v2 con gate duro (§12, cabecera).
