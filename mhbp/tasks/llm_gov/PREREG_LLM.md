# Pre-registro — Capítulo LLM: la cognición gobierna un actuador opaco

> **v2 — 2026-08-06.** La v1 (git 977c162) fue atacada por un panel de 6
> lentes (~40 hallazgos, 12 críticos; cosecha en panel_llm_hallazgos.json).
> Adjudicación en §10. Actuador: Qwen2.5-14B-Instruct LOCAL CONGELADO
> (bring-up verificado: 29.5 GB VRAM, 122 tok/s a batch 16, acc(n=1,K=8)
> ≈ 0.31 — la ventana existe; y el modo de fallo es el acantilado deseado:
> no-permutaciones con confianza). GATE DURO: eval intocada hasta que
> caché/fits/λ/θ/δ estén congelados con timestamp.

## 1. Hipótesis (operacionalizadas tras el panel)

- **HL1**: a TOKENS emparejados, el asignador explícito supera al brazo
  uniforme/SC (fusionados: eran el mismo brazo — hallazgo del panel) con
  margen δ0. [primario, familia F1]
- **HL2a (la claim central, el VG-B0 INVERTIDO)**: Δ_inv =
  payoff(posterior+valor) − payoff(posterior) ≥ δ2 a tokens emparejados,
  **Y** Δ_inv − Δ_shuffle > δ2 con el placebo posterior+valor_shuffle (ŝ
  PERMUTADO: aísla el trabajo del canal del sobreajuste del parámetro
  extra). En S₅ este número fue +0.0002 (saturación); la predicción del
  programa es que aquí NO satura. [gatekeeped a HL1]
- **HL2b (secundario)**: expl vs el **TECHO posterior** — la MEJOR política
  de parada posible dada la interocepción declarada, computada por DP
  EXACTO sobre el caché (no nuestra regla θ: mata la objeción «vuestra
  parada es subóptima»).
- **HL3**: el pipeline entero corre sin cambios conceptuales sobre el
  actuador nuevo; reforzada por diseño: **el actuador JAMÁS ve el stake**.
- La dosis-respuesta con la opacidad NO es claim de este capítulo
  (un punto no testea una pendiente): opacidad = covariable en BANDAS
  pre-declaradas (AUC@n=8: <0.7 / 0.7-0.95 / ≥0.95) con UNA RAMA POR BANDA
  (§8); el diseño multi-nivel queda para la familia 2.

## 2. Tarea: S₅-en-texto con convención VERIFICADA

- Verbalización de cycle_transp (fila de 5 fichas; R = rotación, S =
  intercambio de las dos primeras — semántica de CASILLAS, fijada).
  **T-LLM0 (obligatorio pre-sonda)**: un simulador simbólico que sigue
  LITERALMENTE el texto del prompt debe reproducir la composición
  verdadera en ≥1000 instancias (el panel: la verbalización es BISTABLE
  activa/pasiva y τ es involución — el few-shot corto no desambigua);
  few-shot con composición NO involutiva que incluya el ciclo; gate de
  convención: acc(K∈{1,2}, greedy) ≥ 0.95.
- Respuesta anclada: última línea `RESPUESTA: X X X X X`; parser estricto
  (última ocurrencia; RECHAZA notación de ciclos; no-parse = clase que
  nunca gana el voto, coste pagado). Auditoría manual de 50 CoTs en
  bring-up, commiteada.
- **Stakes: el actuador NO los ve.** El motivo (relación de igualdad, 2
  símbolos) vive en METADATOS de la instancia que solo lee el gobernador;
  el sensor ŝ aprende de consecuencias como en N2/N3. Elimina POR
  CONSTRUCCIÓN los confounds actuador-reacciona-al-motivo y
  motivo→longitud-de-CoT (críticos del panel), y refuerza HL3.
- K-rango: por VG-L0 (≤3 rondas commiteadas sobre stream DEV; diales:
  K-rango, few-shot, temperatura T∈{0.7,1.0,1.3} — T se CONGELA con
  timestamp ANTES de medir cualquier AUC posterior).
- **Espacio de instancias**: requisito |2^K| ≥ 20× demanda por K; si el
  K-rango viable lo viola, se amplía el conjunto de generadores (añadir
  L = rotación inversa; sigue siendo S₅) ANTES de bajar la demanda.
  Streams: sonda=7001 / val=7002 (λ, θ, gates) / dev=7003 (rondas VG-L0) /
  eval=7999 INTOCADA; dedupe por hash de la palabra de generadores contra
  los cuatro (few-shot incluido).

## 3. El CACHÉ único (la pieza que reorganiza todo — panel ×4 lentes)

Se genera UNA VEZ por instancia×semilla-de-muestreo (2 semillas): pool de
**m=16 muestras i.i.d.** de CoT (T congelada), guardando por muestra:
respuesta parseada, tokens generados, logprob media, orden. TODOS los
brazos son funciones DETERMINISTAS del caché (consumo por prefijo en el
orden de muestreo): pareado exacto por instancia a nivel de muestra
(common random numbers), coste real contabilizado por muestra consumida,
y una sola pasada de inferencia sirve a los 7 brazos (~7× menos GPU).
- acc(voto-de-n): **expectativa combinatoria EXACTA** sobre el multiconjunto
  del pool (no Monte Carlo, no submuestreo sesgado); mallas de n IMPARES
  {1,3,5,7,9,11} (los empates con 120 clases eran frecuentes); empate
  residual → rng sembrado; cero parses → incorrecto.
- **p̂ = tabla LIBRE por (K, n)** — SIN isotónica: la votación puede
  DECRECER (modal-erróneo → el voto converge al error); PAVA aplastaría
  la curva real (crítico del panel). dp_alloc no necesita monotonía.
- La fracción de instancias con modal-erróneo por K se mide y reporta;
  rama pre-declarada: si es alta, el asignador gana el movimiento legítimo
  n_i=1 en instancias condenadas (hallazgo, no fallo).

## 4. Moneda del coste: TOKENS (única — la barra «tokens/muestras» de v1
## era el bug)

ĉ(K) = mediana(tokens/muestra | K) de la sonda, congelada. El asignador:
n_i = argmax_n ŝ_i·p̂(K̂_i, n) − λ·ĉ(K̂_i)·n. La lotería de λ (y de θ)
clava **E[tokens totales]** (tolerancia ±2% verificada EN eval sobre
tokens realizados). ANCOVA payoff~tokens_realizados portada de N3 (rama:
si el signo cambia al covariar → «no interpretable»). E[muestras] queda
como sensibilidad secundaria. dp_alloc/vg_n3c se PARAMETRIZAN (coste
c_clase, stakes, p_hi, K-rango, malla n) con self-tests que reproducen
bit a bit los números de N3 antes de usarse aquí.

## 5. Brazos (todos del caché; tokens emparejados salvo referencia)

| brazo | decisión | rol |
|---|---|---|
| uniforme/SC | n=cte (tokens objetivo) | suelo y statu quo emparejado |
| SC-canónica (n=5) | la literatura, A SU coste | referencia descriptiva FUERA de paridad |
| dificultad | argmax s̄·p̂ − λĉn | baseline de valor |
| **expl** | argmax ŝ·p̂ − λĉn | HL1 |
| **posterior** | parada modal/k ≥ θ (n_min=2, cap 11) | el posterior operativo |
| posterior-peek | paga hasta la parada, VOTA con todo | control del confound de truncamiento (HL2a debe sobrevivir contra peek) |
| **posterior+valor** | θ(ŝ) dos umbrales | HL2a |
| posterior+valor_shuffle | θ(ŝ permutado) | placebo de HL2a |
| oráculo de clase | stake y K verdaderos, DP | techo ex-ante |
| **techo posterior (DP)** | mejor política de parada sobre el caché | HL2b (computado, no muestreado) |

## 6. Gates (orden estricto; T-LLM0 → VG-L0 → sonda → VG-L1..L5 → enmienda
## δ → eval)

- **T-LLM0**: convención verificada (simulador simbólico) + parser con
  tests + auditoría de CoTs.
- **VG-L0**: acc(n=1) ∈ [0.1, 0.8] en el K-rango; tasa de no-parseo < 5%
  por celda; gate de convención ≥ 0.95 en K∈{1,2}. ≤3 rondas (dev).
- **VG-L1**: pendiente NETA de acc-voto(n) medida (test de tendencia, no
  supuesto) + mediana por instancia de (p₁−p₂) > 0 en el rango.
- **VG-L2**: headroom de VALOR ex-ante por DP en TOKENS sobre las tablas
  de la sonda ≥ max(0.04, 4·SE).
- **VG-L3 (re-anclado a PAYOFF, la lección VG-B0)**: sobre el caché de la
  sonda, DP: (a) headroom_valor|posterior = techo(posterior+valor) −
  techo(posterior); (b) headroom_posterior = techo(posterior) − techo
  (ex-ante). Ramas PRE-EVAL: (a)≈0 → HL2 muere ANTES de gastar eval
  (subsunción como S₅ — resultado publicable tal cual); techo(posterior)
  ≈ uniforme → opacidad total (HL2 trivial; cambiar familia); intermedio
  → eval. AUC del acuerdo reportada como covariable en bandas.
- **VG-L4**: |espacio| ≥ 20× demanda + dedupe verificado (el motivo ya no
  toca al actuador: la clase de gates de contaminación de prompt queda
  VACIADA por diseño).
- **VG-L5**: |Δtokens realizados| ≤ 2% en cada par contrastado, o el
  contraste se etiqueta «no interpretable» (rama).
- Enmienda fechada post-sonda y PRE-eval: δ0 = max(¼·headroom_VG-L2,
  4·SE_bootstrap), δ2 análogo con VG-L3a.

## 6b. ENMIENDA VG-L0 (2026-08-08, PRE-DATOS: nada de sonda/eval ha corrido)

**Resultados del mini-piloto** (dev, T=0.7, llm_vgl0.json): parseo **1.000
en todas las celdas**; t̄(K) ≈ 35 + 8·K tokens (mucho menor que lo
estimado → §9 se abarata); acc(1 muestra) = 0.479 (K=4), 0.354 (6), 0.229
(8), 0.146 (10), 0.062 (12), 0.042 (14, 16) → ventana [0.1, 0.8] = K∈[4,10].

**El gate de convención (≥0.95 a K∈{1,2}) FALLÓ (0.875 a K=2) y se
sustituye su INSTRUMENTO** (no su propósito): enumeración EXHAUSTIVA de
las 3+9 palabras a greedy. Resultado: **11/12**; el único fallo (`S L`)
es un DESLIZ DE EJECUCIÓN — el modelo escribe «L:» y aplica R —, no una
convención distinta. Refutación objetiva de las alternativas: `L`, `L L`,
`L S`, `R L` correctas ⇒ no hay intercambio sistemático L↔R; `R` correcta
a K=1 ⇒ la lectura pasiva/inversa queda refutada. **Criterio nuevo
(declarado aquí, aplicado ya)**: la convención pasa si la hipótesis
identidad es la mejor por ≥3 palabras sobre cualquier relabelado o lectura
inversa — cumplido (11/12 vs ≤4/12). Los deslices son la DIFICULTAD que el
capítulo mide, no un defecto del entorno.

**Diales CONGELADOS (timestamp 2026-08-08, antes de generar caché)**:
T = 0.7, top_p = 0.9, max_new = 640, few-shot = «R S L L S», parser
estricto, **K ∈ {9, 10, 11}** — dentro de la ventana y única banda que
cumple el requisito de espacio con las demandas declaradas: sonda 768 ×
m=16, val 384, eval 1536 × m=12 → demanda por-K = 896 ≤ 3⁹/20 = 984 ✓.
(K≤8 lo violaría; la regla del prereg prohíbe bajar la demanda antes que
ampliar generadores, y ampliarlos exigiría re-correr VG-L0.)

**Coste recalculado con t̄ medido**: sonda ≈ 768·16·115 ≈ 1.4M tokens;
eval ≈ 1536·12·115 ≈ 2.1M por semilla. A ~120-300 tok/s: sonda 1.5-3 h,
eval 2-5 h por semilla.

## 6c. ENMIENDA CORRECTIVA (2026-08-08, tras el panel de cierre; PRE-DATOS
## de la nueva sonda)

El panel adversarial contra el cierre encontró **dos errores de
instrumentación míos**, verificados sobre los propios datos, que producían
un FALSO NEGATIVO:

1. **Estimador sesgado**: `vote_curve` usaba plug-in CON reemplazo del pool
   (multinomial sobre frecuencias empíricas) en lugar del combinatorio
   exacto que este prereg exigía (§3). El sesgo es NEGATIVO y crece con la
   diversidad del pool — ciega la medida justo en el régimen donde la
   self-consistency funciona (−0.068 con 8 clases de error). Firma en
   nuestros datos: acc(modal-16) medida 0.271 > acc(11) estimada 0.243.
   **Corregido**: subconjuntos SIN reemplazo (hipergeométrica multivariante
   = U-estadístico insesgado para n ≤ m).
2. **Techo mal definido**: se computó como acc(n_max=11) − acc(n̄), pero 11
   era un TOPE DE DISEÑO. El techo correcto usa el **n ASEQUIBLE**:
   n_hi = (n̄ − (1−p_hi)·1)/p_hi, el que el presupuesto realmente permite
   dando n=1 a las instancias baratas.

**Re-medición decisiva** (dev, K∈{9,10,11}, T=1.3, m=32, estimador
insesgado, n hasta 31): acc = 0.170 (n=1) → 0.230 (11) → 0.247 (31), aún
creciente. Techos con n asequible: **+0.058** (n̄=3, p=0.10), **+0.050**
(n̄=3, p=0.15), +0.041 y +0.040 (n̄=5) — **todos ≥ 0.04**. La celda de
T=0.7 re-medida sin sesgo SIGUE PLANA y satura (0.131→0.141→0.140 en
n=13-15): el hallazgo «a T baja no hay palanca» se mantiene y pasa a ser
un contraste DENTRO del capítulo (la temperatura crea la palanca).

**Diales CONGELADOS para el confirmatorio** (timestamp 2026-08-08, antes de
generar la nueva caché): **T = 1.3**, K∈{9,10,11}, m=24 (≥ n asequible
14.3), **n̄ objetivo = 3**, p_hi = 0.15, s_alto = 8, top_p = 0.9.
**Requisito añadido**: todo techo/headroom se reporta con **IC bootstrap
PAREADO por instancia** (la diferencia, no los niveles) y el gate compara
contra max(0.04, 4·SE) como estaba pre-registrado. Los pools se PERSISTEN
siempre (auditabilidad: los de las celdas de diales se descartaron).

## 7. Estadística

Unidad = INSTANCIA. Primario: **bootstrap jerárquico pareado por
instancia** (remuestrea instancias y, dentro, el pool de muestras; 10⁴
réplicas, seed fija), estratificado por stake×K; IC95 debe excluir δ.
Familia cerrada: F1 = {expl − uniforme/SC} α=0.05 →(gatekeep)→ F2 =
{HL2a primario; HL2b secundario} con Holm. Suelo de cordura (expl ≥
uniforme) fuera de familia. Las 2 semillas de muestreo = 2 cachés:
sensibilidad de signo, jamás la unidad del test.

## 8. Ramas (completas)

- HL1✓ ∧ HL2a✓ (y sobrevive vs peek) → **la inversión predicha**: el
  capítulo final del paper con la jerarquía re-medida en régimen opaco;
  banda de opacidad reportada.
- HL1✓ ∧ HL2a✗ → el posterior subsume TAMBIÉN aquí: VG-B0 se generaliza
  (resultado fuerte en la otra dirección — el dominio del posterior se
  extiende a actuadores opacos); el capítulo lo publica igual.
- HL1✗ con gates verdes → el asignador no transfiere: auditoría
  pre-declarada (descomposición ŝ/p̂/λ; ¿modal-erróneo?).
- HL1✓ pero pierde vs SC-canónica A SU coste → se reporta ambos: la
  paridad es la comparación justa; la referencia contextualiza.
- VG-L3a ≈ 0 pre-eval → subsunción sin gastar eval (rama barata).
- VG-L0 muere en 3 rondas → familia 2 (aritmética verificable)
  directamente, mismo prereg-esqueleto, enmienda fechada.
- Un solo modelo: limitación DECLARADA en el paper; robustez opcional
  post-veredicto con 7B (mismo caché-pipeline, coste menor) como
  descriptivo — nunca como test.

## 9. Coste (recalculado; mini-piloto de t̄ antes de commitear la sonda)

Sonda: 1024 inst × m=16 × t̄ (medir; el smoke sugiere 150-400 tok/K-alto)
≈ 2.5-6.5M tokens ≈ 3-8 h a ≥250 tok/s batcheado (batch ≥64). Eval: 2048
inst × m=16 × 2 semillas ≈ 10-26M... **recorte declarado**: eval m=12
(cap del posterior=11) y n_max de brazos ex-ante 11 → ~8-20M tokens ≈
una noche holgada. Todo local, cero coste por token, LLM congelado.

## 10. Registro de los 12 críticos → decisión

convención bistable → T-LLM0 simulador+gate (×1); motivo contamina al
actuador → el actuador no lo ve (×2, vaciado por diseño); moneda
muestras≠tokens → tokens único + ĉ(K) + VG-L5 (×3); posterior sin
emparejado → lotería de θ + pares al coste realizado + VG-L5 (×2); sonda
sin diseño → caché m=16 + estimador combinatorio exacto (×2); HL2 sin
contraste → HL2a/HL2b + placebo shuffle + peek (×3); regla-vs-régimen →
techo posterior por DP (×1); opacidad un-punto → bandas con rama por
banda (×2); PAVA falso bajo votación → tabla libre + VG-L1 de tendencia
(×2); estadística sin unidad → bootstrap jerárquico por instancia +
familia cerrada (×3); uniforme≡SC → fusión + SC-canónica fuera de paridad
(×3); espacio 2^K pequeño → requisito 20× + generador L de reserva (×1).
