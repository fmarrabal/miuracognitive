# Capítulo LLM — Hallazgos de la fase S₅-en-texto: sin PALANCA no hay
# gobierno (y la palanca estándar casi no existe aquí)

> 2026-08-08. Actuador: Qwen2.5-14B-Instruct local congelado. Todo
> pre-registrado (PREREG_LLM v2, panel de 6 lentes); gates VG-L1/L2/L3
> corridos sobre el caché de sonda (768 instancias × 16 muestras, 116 min).
> **La eval confirmatoria NO se lanzó**: los gates la mataron con
> mecanismo y con un techo calculado. Cero horas de GPU desperdiciadas en
> un confirmatorio inejecutable.

## 1. El actuador es competente y el entorno es limpio

Parseo **1.000** en todas las celdas; convención certificada por
enumeración exhaustiva (11/12 palabras a greedy; el único fallo es un
desliz de ejecución —escribe «L:» y aplica R—, con las hipótesis
alternativas refutadas: no hay swap L↔R ni lectura pasiva);
t̄(K) = 35 + 8·K tokens; stake ⊥ K y **el actuador nunca ve el stake**
(vive en metadatos que solo lee el gobernador).

## 2. El hallazgo: la self-consistency casi no tiene rango dinámico aquí

| configuración | acc(n=1) | acc(n=11) | pendiente | modal-erróneo |
|---|---|---|---|---|
| K∈{9,10,11}, T=0.7 | 0.132 | 0.140 | **+0.008** | **0.863** |
| K∈{9,10,11}, T=1.0 | 0.198 | 0.231 | +0.033 | 0.760 |
| K∈{9,10,11}, T=1.3 | 0.188 | 0.243 | +0.055 | 0.729 |
| K∈{5,6}, T=0.7 | 0.415 | 0.430 | +0.015 | 0.573 |
| K∈{5,6}, T=1.0 | 0.421 | 0.463 | +0.042 | 0.510 |

**Mecanismo**: a T baja los errores del modelo son SISTEMÁTICOS — repite el
mismo desliz — así que el voto mayoritario converge al error (modal
erróneo en el 86% de las instancias) y muestrear más no compra nada. La
temperatura es el dial que DESCORRELACIONA los errores: 0.7→1.3 multiplica
la pendiente por 7 (+0.008→+0.055) y, contra la intuición, **sube también
la accuracy de una sola muestra** (0.132→0.188): el muestreo casi-greedy
se encasilla en la misma trayectoria equivocada. La dificultad (K) mueve el
nivel, pero es la temperatura la que crea la palanca.

## 3-bis. CIERRE DEFINITIVO (2026-08-09, sonda completa a T=1.3)

> **Este apartado sustituye al §3 y al §4b**, escritos con dos errores de
> instrumentación que el panel de cierre encontró (estimador plug-in CON
> reemplazo; techo evaluado en el tope de diseño n=11 en vez del n
> ASEQUIBLE). Ambos están corregidos; además yo cometí un tercer error al
> re-medir con 96 instancias de dev SIN IC pareado, lo que produjo una
> sobre-estimación transitoria (+0.050). Los números que siguen son los
> únicos válidos: sonda de 768 instancias, m=24, estimador insesgado
> (subconjuntos sin reemplazo), IC bootstrap PAREADO por instancia.

**Celda congelada**: S₅-en-texto, K∈{9,10,11}, T=1.3, n̄=3, p_hi=0.15,
s_alto=8, top_p=0.9.

| medida | valor |
|---|---|
| acc(voto): n=1 → 11 → 15 | 0.128 → 0.159 → 0.162 |
| pendiente (n1→n11) | **+0.034** (a T=0.7 era +0.008) |
| **techo de valor (n asequible = 14.3)** | **+0.0236, IC95 [+0.0150, +0.0326]** |
| umbral pre-registrado max(0.04, 4·SE) | 0.0400 → **NO PASA** |
| headroom valor \| posterior (VG-L3) | +0.003 / +0.002 / +0.001 |
| posterior − ex-ante (VG-L3) | **−0.005 / −0.002 / −0.003** |

**Tres lecturas, todas cuantitativas:**

1. **La palanca EXISTE y es pequeña.** El techo es significativamente > 0
   (IC excluye el 0) y significativamente < 0.04 (el IC excluye el umbral
   por arriba): no es un nulo por falta de potencia, es una **cota**. Con
   este actuador, la asignación de muestras puede mover como mucho ~2.4
   puntos de payoff — un orden de magnitud por debajo de lo que el gobierno
   necesita para ser relevante.
2. **La temperatura crea la palanca, no la competencia.** Misma tarea,
   mismo modelo: T=0.7 → techo +0.002 (plano, satura en n=13); T=1.3 →
   +0.024. El dial que importa no es cuánto sabe el modelo sino cuánto
   DECORRELACIONA sus errores.
3. **El posterior del LLM es peor que inútil aquí: es engañoso.** El
   acuerdo entre muestras no informa (modal-erróneo 0.833: el modelo está
   de acuerdo consigo mismo mientras se equivoca), y la política de parada
   por acuerdo rinde POR DEBAJO de la asignación ex-ante (−0.005/−0.002/
   −0.003). Es la premisa de opacidad de HL2 medida directamente — y el
   contraste exacto con S₅-neuronal, donde el posterior valía +0.22 sobre
   todo lo ex-ante.

**Veredicto de gates**: VG-L1 pasa (palanca real), **VG-L2 no pasa** (cota
por debajo del umbral), VG-L3 sin headroom. Por la regla pre-registrada,
**el confirmatorio no corre**. Cero horas de GPU gastadas en él.

## 3. El techo estructural: por qué el confirmatorio no puede correr
### (APARTADO SUPERADO — ver §3-bis; se conserva por trazabilidad)

El headroom de asignación por VALOR está acotado por la propia curva:
ninguna política puede superar `acc(n_max) − acc(n̄)`. Con la mejor celda
(T=1.3), calculado exactamente sobre la curva medida:

| n̄ | uniforme | **techo absoluto** (s→∞) | s=8,p=0.15 | s=32,p=0.15 |
|---|---|---|---|---|
| 3 | 0.206 | **+0.0375** | +0.017 | +0.030 |
| 5 | 0.221 | +0.0217 | +0.009 | +0.017 |
| 7 | 0.233 | +0.0099 | +0.004 | +0.008 |

El umbral pre-registrado de VG-L2 (≥ max(0.04, 4·SE)) es **estructuralmente
inalcanzable**: ni con ratio de stake infinito. No es que el asignador
fracase — es que no hay nada que asignar. Gates: **VG-L1 pasa (la palanca
existe, es pequeña), VG-L2/VG-L3 FALLAN** (+0.002 y +0.005 en la
configuración sondeada).

## 4. Lectura para el paper

Este resultado no es un fracaso del gobernador: es la **precondición del
gobierno**, medida. Toda la serie del programa mostró *dónde* vive cada
función; aquí se establece el requisito previo a todas ellas:

> **Sin palanca de cómputo con rango dinámico, la cuestión del gobierno ni
> siquiera se plantea.** Y la palanca estándar de la práctica (muestreo +
> votación) tiene rango casi nulo en seguimiento de estado, porque los
> errores del LLM son sistemáticos, no diversos.

Corolario práctico (útil y contrastable): **la self-consistency no es una
palanca de cómputo universal**; su rango depende críticamente de la
temperatura, que es lo que decorrelaciona los errores.

## 4b. FAMILIA 2 (aritmética verificable): las 3 rondas declaradas, y la
## LEY empírica que cierra el capítulo

Se ejecutó la recomendación B (familia 2 con el pipeline reutilizado;
T-LLM0 8/8). Las tres rondas de diales declaradas barrieron la dificultad
de punta a punta:

| celda | acc(1) | pendiente(n1→n11) | modal-erróneo | 1 − acc |
|---|---|---|---|---|
| arit. cadena larga (mul 2-4) | 0.960 | +0.001 | 0.031 | 0.040 |
| arit. mul 7-19, K5-6 | 0.865 | −0.002 | 0.141 | 0.135 |
| arit. mul 23-97, K4-5 | 0.352 | +0.006 | 0.625 | 0.648 |
| arit. mul 23-97, K6-7 | 0.290 | **+0.023** | 0.688 | 0.710 |
| arit. mul 23-97, K8-9 | 0.129 | +0.009 | 0.859 | 0.871 |
| S₅ K9-11, T=1.3 (mejor global) | 0.188 | **+0.055** | 0.729 | 0.812 |

**La ley empírica (17 celdas, 2 familias, acc de 0.13 a 0.97):**

> P(la respuesta modal de 16 muestras sea errónea) ≈ (1 − acc₁) − ε,
> con **ε ≤ 0.08** en TODAS las celdas medidas.

Es decir: **la respuesta modal es esencialmente la respuesta de una sola
muestra**. Muestrear 16 veces recupera como mucho 8 puntos de la masa de
error, y sólo en la celda más favorable (S₅ a T alta). El techo de valor
correspondiente en la mejor celda aritmética (K6-7) es +0.015 a n̄=3 —
igual de inalcanzable que en S₅.

**Conclusión del capítulo (dos familias, barrido completo de dificultad,
tres diales por familia, techos calculados)**: con este actuador, la
palanca de cómputo estándar no tiene rango dinámico ni en seguimiento de
estado ni en aritmética; ni el régimen fácil (saturación) ni el difícil
(errores sistemáticos) la ofrecen. La temperatura la agranda algo (×7 en
S₅) pero no la crea. **El confirmatorio no corre en ninguna de las dos
familias** — y esta vez el negativo no es de una tarea, es del PAR
(actuador, palanca).

## 5. Estado y fork (decisión de Curro)

La familia 2 estaba pre-declarada en el prereg (§2: «aritmética multi-paso
verificable; solo si la fase 1 confirma») precisamente para este caso: es
la familia donde la literatura documenta que el voto SÍ tiene rango
(acc sube fuerte con n). Todo el pipeline (entorno con stakes en
metadatos, caché único, estimador exacto del voto, DP del techo posterior,
gates) es reutilizable sin cambios conceptuales — cambiaría el
verbalizador/verificador y habría que re-correr VG-L0/L1 (~1 h), sonda
(~2 h) y, si el gate pasa, la eval (~3-5 h).

- **(A) Cerrar aquí**: el capítulo entra en el paper como «la precondición
  del gobierno», con el mecanismo y el techo. Coste 0.
- **(B) Familia 2**: mismo pipeline sobre aritmética verificable. ~1 día.

Recomendación ejecutada: **B, conservando A como sección**. Resultado: la
familia 2 tampoco ofrece palanca (§4b) y el capítulo cierra con un
enunciado MÁS FUERTE que el previsto, porque ya no depende de una tarea.

## 5-bis. Lo que este capítulo enseñó sobre el MÉTODO (entra en el paper)

La secuencia de este capítulo es el caso de estudio más completo del
programa sobre cómo se falla y cómo se corrige:

1. Cierre en NEGATIVO con dos errores de instrumentación míos (estimador
   sesgado + techo en el n equivocado) → **falso negativo**.
2. Panel adversarial contra mi propio cierre: encuentra ambos leyendo el
   código y verificándolos contra MIS datos (la firma estaba a la vista:
   acc(modal-16) medida > acc(11) estimada — imposible con un estimador
   correcto).
3. Re-medición apresurada en 96 instancias sin IC pareado → **falso
   positivo** transitorio (+0.050).
4. Sonda completa (768) con IC pareado → el número real: **+0.0236
   [0.0150, 0.0326]**, cota decisiva bajo el umbral.

Moraleja operativa, ya incorporada a las reglas: *un estimador plug-in
sobre un pool pequeño está sesgado en la dirección de tu hipótesis nula, y
un techo evaluado en un tope de diseño no es un techo*. Y la regla que
cierra las tres: **ningún veredicto sin intervalo sobre la DIFERENCIA**.

## 6. Qué entra en el paper (capítulo final)

**Título del capítulo: la precondición del gobierno.** El programa midió
dónde vive cada función cognitiva; este capítulo establece qué debe existir
ANTES de que la pregunta del gobierno tenga sentido:

1. **La palanca no es un supuesto: es una propiedad medible del par
   (actuador, mecanismo de gasto).** Instrumento: la curva acc(voto-de-n)
   y su pendiente, más el techo `acc(n_max) − acc(n̄)` que acota a
   CUALQUIER política de asignación.
2. **La palanca estándar de la práctica (self-consistency) tiene rango
   casi nulo** para un 14B en tareas verificables: ley empírica
   modal-erróneo ≈ (1 − acc₁) − ε, ε ≤ 0.08, sobre 17 celdas y dos
   familias con acc de 0.13 a 0.97.
3. **Mecanismo doble**: en el régimen fácil no hay margen (saturación); en
   el difícil las muestras son redundantes (el modelo repite su desliz).
   La temperatura descorrelaciona parcialmente (×7 la pendiente en S₅) pero
   no crea la palanca.
4. **Consecuencia para el arco**: la jerarquía de la información medida en
   S₅-neuronal (0.467 < 0.546 < 0.698 < 0.921) describe un sistema donde la
   palanca SÍ existía (el reasoner mejora de verdad con más ticks). Un LLM
   muestreado no la tiene por esta vía — lo que explica, y predice, que el
   gobierno de cómputo en LLMs deba buscarse en palancas con rango real
   (longitud de razonamiento, escalada de modelo, herramientas), no en el
   número de muestras.
5. **Predicción falsable heredada**: el test de la inversión (HL2a) queda
   pendiente de un actuador con palanca — es el experimento que este
   capítulo deja armado, con todo el pipeline construido y certificado.

## Addendum (2026-08-17, revisión TMLR simulada): corrección de signo en
## los deltas de VG-L3

El bootstrap pareado por instancia (B=200, llm_ic_acuerdo.json), pedido
por el revisor estadístico para el claim de signo del paper, CORRIGE los
puntuales del §3-bis: Δ(posterior−exante) no es −0.005/−0.002/−0.003
sino +0.007 [−0.008,+0.015] / +0.015 [+0.001,+0.021] / +0.017
[+0.005,+0.025] a n̄≈3/5/7 — los puntuales originales eran frágiles a la
semilla de análisis y a la rejilla de λ. Δ(valor|posterior) queda nulo
con intervalos (+0.003/+0.002/+0.001, todos cruzando cero). LECTURA: la
parada-por-acuerdo no penaliza; compra casi nada (≤+0.017, un orden por
debajo del posterior nativo del sustrato). «Mide convicción, no
corrección» (modal-erróneo 0.833) se mantiene como mecanismo; cae la
afirmación de pasivo. Tercera aplicación de la regla del intervalo, esta
vez contra nuestro propio texto ya escrito.
