# PREREG N1B-LLM — el acantilado mudo en su hogar natural
### v1, 2026-08-10. Pre-datos. Mandato de Curro. Panel antes de la sonda.

## 1. Por qué aquí

FINDINGS_N1B: la familia de acantilado mudo exige retrieval en contexto,
que en el sustrato pequeño solo grokkea sin recurrencia — el test
pertenece a un LM preentrenado (induction heads gratis) con la palanca
de LONGITUD de generación (no votos: la cota del cap. LLM no aplica).

## 2. La familia: CAMINO-EN-CICLO en texto

Instancia: un ciclo único de longitud L ∈ {6, 10, 14} sobre letras
(subconjunto aleatorio de a..t), presentado como pares barajados
«x→y». Pregunta: empezando en s, nº de pasos hasta t. El actuador debe
CAMINAR en CoT (formato «k: x→y» por paso) y terminar «RESPUESTA: d».
- Todo-o-nada: d solo se conoce al llegar.
- Mudo: posición actual ⇏ distancia restante (permutación aleatoria).
- Ex-ante coarse: L VISIBLE (nº de pares listados); d ~ U[1, L−1]
  invisible. Coste c_i (tokens hasta llegar) ∝ d ⇒ invisible ex-ante.
- Stakes en METADATOS (motivo de igualdad J/T/V/W, p_hi=0.15,
  s_alto=8): el actuador nunca los ve; las políticas de caps sí.

**Contraste suave EN EL MISMO SUSTRATO**: la familia aritmética del
cap. LLM (llm_env2), donde el trabajo total K es VISIBLE ex-ante —
c_i predecible. Cliff vs suave = R²(c_i | clase) bajo vs alto.

## 3. Diseño de UNA GENERACIÓN + brazos offline (lección del panel N1b)

Por instancia: UNA generación larga (max_new suficiente para L=14),
grabando texto, tokens totales, posición de llegada (token donde
aparece RESPUESTA), hops parseados del CoT, corrección. TODOS los
brazos = políticas de CAP evaluadas por TRUNCADO offline de la
grabación: correct(cap) = (tokens_usados ≤ cap) ∧ correcto. Cero
re-inferencia; contabilidad de presupuesto idéntica entre brazos.

Brazos (presupuesto TOTAL de tokens emparejado):
- uniforme: cap común.
- dificultad: cap por L (dp sobre tabla ĉ(L) del split de fit).
- valor: cap por (L, stake) maximizando pago esperado.
- adaptativo-llegada: secuencial; los tokens liberados por llegadas
  tempranas (cap − usados) vuelven al pool (la detección de llegada es
  GRATIS en generación: el modelo para solo).
- oráculo-c: conoce c_i (techo de cualquier información mid-flight).

## 4. Hipótesis (la inversión, ramas completas)

- **H1-LLM (el valor paga bajo compromiso en el acantilado)**:
  pago(valor) − pago(dificultad) ≥ max(0.02, 4·SE) a presupuesto
  emparejado. [En S₅-neuronal el análogo dio +0.15; aquí la decisión de
  cap ES compromiso puro.]
- **H2-LLM (nada más allá de la llegada — la mudez del posterior)**:
  pago(oráculo-c) − pago(adaptativo-llegada) ≤ 0.02·rango — el techo de
  TODA información mid-flight es la detección de llegada, que ya es
  gratis. [El espejo exacto del +0.22 de S₅: allí mirar-dentro pagaba;
  aquí, por mudez, no puede pagar.]
- Ramas: H1✓H2✓ → inversión completa (el cierre del paper pasa de
  predicción a resultado). H1✓H2✗ → hay información mid-flight
  explotable (¡la mudez falló! → gate M lo explica o refuta el diseño).
  H1✗ → o la palanca de longitud no tiene rango (gate R) o el valor no
  paga ni bajo compromiso (refutación parcial de la predicción — se
  publica). Gate rojo → no se corre; se reporta.

## 5. Gates (baratos primero; sobre el dev, antes de la sonda)

- **T (cableado)**: self-tests del entorno (caminante independiente,
  parser de RESPUESTA y de hops, L visible/d invisible, stakes ⊥).
- **W (¿camina?)**: acc sin cap ≥ 0.55 global y ≥ 0.35 en L=14, con
  compliance de formato ≥ 0.80. Diales (≤3 rondas): L_SET, few-shot,
  temperatura ∈ {greedy, 0.3, 0.7}.
- **R (rango de la palanca)**: corr(c_i, d | llegó) ≥ 0.6 y
  CV(c_i | L) ≥ 0.3 (el coste debe VARIAR dentro de clase — el
  acantilado) y R²(c_i | L) ≤ 0.5 vs R²(c_i | K) ≥ 0.7 en la familia
  aritmética (el contraste cliff/suave en el mismo sustrato).
- **M (mudez mid-flight)**: probe multi-horizonte sobre prefijos:
  P(llega en ≤ h tokens | m, L, hops_parseados) − P(· | m, L) ≤ 0.05
  de AUC para h ≥ 2·t̄_hop (permitido el horizonte de 1 hop). Control
  positivo: el MISMO probe en la familia aritmética debe superar +0.05
  (allí ops-hechas vs K predice el remanente). Regla de la casa:
  ningún nulo sin control positivo del instrumento.
- Diagnóstico de firma: gap de acc en d∈pow2 (¿salta en vez de
  caminar?); linealidad hops↔tokens.

## 6. Estadística y presupuesto

Sonda: 768 inst (256/L), 1 generación, max_new 900 → ~3-4 h GPU.
IC bootstrap PAREADO por instancia para todos los Δ de brazos;
selección de caps en mitad fit / medición en mitad held-out; nulo por
permutación de stakes para H1. Presupuesto emparejado verificado ±2%.
Seeds/splits/dedupe: patrón de llm_env (sonda 7001, dev 7003,
eval 7999; hash de contenido = (ciclo, s, t)).

## 7. ADJUDICACIÓN DEL PANEL (v2, 2026-08-10, pre-sonda) — SUSTITUYE
## §3/§4/§6 donde choquen. VG0: W✓ R✓ en ambas celdas → GREEDY congelado
## (regla: la celda que pase W con mayor acc; empate → greedy).

### 7.1 Truncado honesto (crítico ×4 lentes)
La sonda persiste los TOKEN-IDS COMPLETOS por generación. c_i := índice
del token que CIERRA la línea RESPUESTA final (vía offsets del
tokenizer); correct(cap) := parse(decode(ids[:cap])) == truth — re-parseo
literal del prefijo (resuelve multi-RESPUESTA y corte entre dígitos).
Self-test pre-brazos: 100 instancias × 20 caps, la regla debe coincidir
con el re-parseo literal. Censura (sin RESPUESTA en 900): c_i=∞,
correct=0 ∀cap, CON masa en las tablas ĉ; nadie se excluye jamás;
gate: censura ≤ 0.10 en L=14.

### 7.2 Potencia de H1 (crítico): marginalización analítica del stake
Pago por instancia = 0.15·8·correct(cap_hi(L)) + 0.85·1·correct(cap_lo(L))
(mezcla EXACTA: el actuador nunca ve stakes y los brazos son offline);
presupuesto en esperanza: 0.15·cap_hi + 0.85·cap_lo. CROSS-FITTING:
caps ajustadas en mitad A → medidas en B y viceversa, promedio pareado
(N_ef=768); split estratificado por (L, décil de d), seed 20260810.
Sensibilidad: ambas mitades por separado (signo discordante → no se
adjudica). MDE(4·SE) esperado ≈ 0.02-0.04.

### 7.3 Brazos v2 (todos offline sobre las grabaciones; presupuesto =
### tokens GASTADOS Σ min(c_i, cap_i), emparejado ±2% por bisección
### escalar sobre el vector de caps)
- uniforme · dificultad: cap(L) · valor: cap(L, clase de stake).
- adaptativo-llegada (FLUIDO, sin orden): cap efectivo k_{L,ŝ} tal que
  Σ min(c_i, k) = B, con la MISMA estructura de clases que valor.
- oráculo-binario: conoce 1{llega_i} (canal doom), caps dp entre las
  que llegan.
- oráculo-c: conoce c_i (y stake) — knapsack stake/c_i.
Punto de operación por MORDIDA: B tal que P(truncada|uniforme) ∈
{0.2, 0.35, 0.5}; PRIMARIO 0.35; curva completa descriptiva (gratis).
Gate C-LLM: en el punto primario, el truncado debe morder de verdad.

### 7.4 Hipótesis v2
- **H1** := pago(valor) − pago(dificultad) ≥ max(0.02, 4·SE) en el punto
  primario, estimador marginalizado + cross-fitting; nulo por
  permutación de la ETIQUETA de clase de stake en las caps (1000 perms,
  seed 20260810).
- **H2 (el canal DISTANCIA es mudo)** := pago(oráculo-c) −
  pago(oráculo-binario) ≤ 0.02·rango, con rango := pago(oráculo-c) −
  pago(uniforme) en el punto primario; adjudicación TOST (límite
  superior del IC95 pareado ≤ umbral). El gap (oráculo-binario −
  adaptativo) se reporta como canal doom (descriptivo; esperado ≈ 0 con
  censura ~2%).
- **Conjunto observable del gobernador O** (declarado): {L, clase de
  stake, m tokens emitidos, n_hops CONTADOS, señal de llegada}. El
  prompt (pares, s, t) y la identidad de símbolos quedan EXCLUIDOS: el
  gobernador modela un meta-controlador que no puede re-resolver la
  tarea (si pudiera, no necesitaría al actuador). La mudez se enuncia
  como propiedad RELATIVA a ese observador.
- **Contraste aritmético confirmatorio-descriptivo**: los mismos brazos
  sobre 384 instancias de llm_env2 (K visible): predicción OPUESTA
  pre-registrada: (oráculo-c − dificultad)_arit ≤ 0.02 mientras
  (oráculo-c − dificultad)_ciclo ≥ 5× eso — la inversión interpretable
  dentro del mismo sustrato.

### 7.5 Gates añadidos (sobre grabaciones, antes de H1/H2)
- **S (no-salta)**: entre las correctas, fracción con cadena VÁLIDA
  completa (cada hop y=f(x), primera x=s, última y=t, n_hops=d) ≥ 0.95;
  si >5% saltan → «el LLM salta»: hallazgo de sustrato, sin confirmatorio.
- **M (mudez, features ⊆ O)**: prefijos al final de cada hop; y_h =
  1{c_i − m ≤ h·t̄_hop}, h ∈ {1,2,3,5}; probe logístico (m, n_hops, L)
  vs tabla hazard (m̃, L); MUDA ⟺ ΔAUC ≤ 0.05 ∀ h ≥ 2. Control
  positivo: mismo pipeline sobre las grabaciones aritméticas (allí
  (ops-hechas, K) predice el remanente — debe superar +0.05).
- Hash canónico del dedupe: (pares ORDENADOS, s, t); dev baneado en la
  sonda.

### 7.6 Congelado operativo
Greedy · MAX_NEW=900 · batch 24 · sonda: 768 ciclo (256/L) + 384 arit
(96 por K∈{4,5,6,7}, spec ronda-3 de llm_env2) · bootstrap pareado por
instancia B=20000 percentil seed 20260810 · réplica de pipeline: re-run
de 96 instancias con assert de identidad (greedy) · los brazos se
computan UNA vez con asignación congelada; el IC no incluye la varianza
del procedimiento de asignación (declarado como limitación).
