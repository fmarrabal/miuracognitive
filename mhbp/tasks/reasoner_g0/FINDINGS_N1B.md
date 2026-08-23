# FINDINGS N1B — la familia de acantilado mudo NO es construible en este
# sustrato: cierre por rama declarada, con el mecanismo completo
### 2026-08-10. Firewall de 3 rondas agotado. La predicción N1b queda
### ABIERTA (ni confirmada ni refutada) y con requisitos afilados.

## 1. Qué se intentó y qué protegió el proceso

Mandato: correr el N1b real (camino-en-ciclo: valor todo-o-nada,
progreso interno mudo medible, dificultad ex-ante coarse). El panel de
diseño (~25 hallazgos) corrigió ANTES de gastar nada dos errores míos
que condenaban el confirmatorio en ambas direcciones (H1 refutaba en
falso: la sola detección de llegada vale +0.24-0.29; H2 se anulaba por
saturación: coste de tick 85× menor que la ganancia de caminar). El
prereg v2 (§9) quedó blindado. Lo que lo mató no fue el diseño del
test: fue el gate de APRENDIBILIDAD, tres rondas, cada una con su
mecanismo.

## 2. Las tres rondas + la cadena de diagnóstico (todo medido)

| ronda | dial | resultado | mecanismo |
|---|---|---|---|
| 1 | tabla posicional | prior exacto | el direccionamiento indirecto no se aprende con 1 posición supervisada |
| 2 | pares (i,σ(i)) + bias −3 + 10k | prior exacto | el retrieval por CONTENIDO tampoco (vs copy posicional: 1.000 en 1 min) |
| diag | 5 probes (indirecto/adyacente/denso-6k) | azar exacto, CE=ln20 clavada | la silla: aprende el SOPORTE y nunca la dirección del matching |
| diag | denso 30k × batch 128, VANILLA | **GROKKING: acc 1.000 en 15k** | el circuito SÍ se forma: densidad + batch + paciencia |
| 3 | denso + batch 128 + 30k, n2_endo | **prior otra vez** (aux en ln20) | **el bucle del reasoner bloquea el grokking** que el backbone plano logra con el MISMO régimen de datos |

Caveat declarado: la ronda 3 añade la pregunta-d junto a las aux (dos
diferencias con el probe vanilla), pero las AUX solas quedaron en ln20
— misma tarea, mismos datos, misma densidad: la atribución al bucle es
razonable aunque no quirúrgica.

Además, economía: el run de la ronda 3 costó 416 min (E[n̄]=24 × batch
128 × 30k) → campaña ≈ 84 h ≫ techo de 6 h. La regla de re-scope del
prereg (§9.6) la habría bloqueado incluso con gate A verde.

## 3. El hallazgo (tres piezas, ninguna prevista)

1. **La familia de acantilado mudo exige retrieval EN CONTEXTO.** Su
   tabla no puede vivir en los pesos (con σ fija, el atajo espectral
   tipo Fourier resuelve d en una pasada y el acantilado muere — el
   cuerno verificado en la literatura de grokking modular; no hizo
   falta medirlo al revivir el cuerno 2), y una tabla por-instancia
   solo se consulta con el circuito de inducción.
2. **El circuito de inducción tiene su propio régimen de señal**: se
   forma por grokking (silla en ln20 → transición ~9-15k) SOLO con
   supervisión densa + batch 128 + paciencia. S₅ nunca lo formó porque
   nunca lo necesitó (su tabla de Cayley vive en los pesos).
3. **El bucle recurrente del sustrato bloquea esa formación** (≤30k
   pasos): vanilla grokkea, n2_endo no, con el mismo régimen de datos.
   El componente que la familia necesita para el CÓMPUTO (ticks
   variables) impide aprender el que necesita para la TAREA (retrieval).

## 4. Dónde vive el test de N1b (el camino adelante, concreto)

La predicción queda abierta y el paper la mantiene como cierre
falsable, ahora con requisitos MEDIDOS: el hogar natural de la familia
es un sustrato con retrieval pre-formado — **un LM preentrenado** (las
induction heads vienen gratis del pre-entrenamiento LM denso), donde el
camino-en-ciclo en contexto es trivial de plantear, la llegada es
detectable, d es invisible ex-ante, y la palanca de cómputo es la
LONGITUD de generación (no el voto — la palanca que el capítulo LLM
midió como cota). Media maquinaria (entorno de texto, caché, gates,
estimadores) ya existe en mhbp/tasks/llm_gov.

## 5. Contabilidad

3 rondas de firewall (agotado, cada una fechada pre-datos) · panel de
diseño adjudicado completo · 6 probes de diagnóstico (sin consumir
firewall: mecanismo con pipeline sano) · ~10 h GPU total · 0 h gastadas
en un confirmatorio incapaz de responder — que es exactamente lo que
los gates prometen.
