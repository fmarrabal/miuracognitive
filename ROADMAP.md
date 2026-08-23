> ## ⚠️ ADDENDUM 2026-08-19 — la jerarquía de este documento está SUPERADA
>
> Este fichero es el registro de la hoja de ruta tal como se siguió, y se
> conserva por eso. Pero el peldaño superior que declara «jerarquía FINAL»
> —`0.921` para la parada posterior, «observarse pensar vale +0.22 sobre
> todo lo decidible a priori»— **no sobrevivió a la auditoría**.
>
> Ese salto era un **artefacto de lectura**: el halting tipo PonderNet
> devuelve una mezcla de estados ocultos, los baselines de profundidad
> forzada devuelven un estado único, y la cabeza de lenguaje solo se
> entrenó sobre la mezcla. Igualando la lectura, el régimen residual es
> **exactamente `+0.000 [0.000, 0.000]`**.
>
> Los peldaños bajos (`0.467 < 0.546 < 0.698`) sí sobreviven, y también el
> `+0.151` del allocator explícito.
>
> Ver [FINDINGS_READOUT.md](FINDINGS_READOUT.md). La decisión de «un solo
> paper» del 2026-08-06 también quedó superada: son **dos** papers,
> `paper/main.tex` y `paper/governor_en.tex`.

# MiuraCognitive — Hoja de ruta hacia la arquitectura cognitiva completa

> Objetivo final (fijado por Curro, 2026-08-01): una arquitectura cognitiva
> completa capaz de TOMAR DECISIONES POR SÍ SOLA. Este documento descompone
> "decidir por sí sola" en capacidades certificables y ordena el programa.
> Regla invariante: cada nivel se fuerza desde el ENTORNO, se certifica con
> una batería pre-registrada, y no se avanza sobre un nivel no certificado.

## Qué es "decidir" (descomposición operativa)

Una decisión autónoma exige cuatro ingredientes, y cada uno tiene ya un
candidato certificado o en curso en el programa:

| ingrediente | pregunta | pieza | estado |
|---|---|---|---|
| deliberación | ¿cómo resuelvo ESTO? | reasoner (régimen M) | ✅ certificado (G0/G0.1: estado portador, cristalización) |
| gobierno | ¿cuánto cómputo le doy? | mHBP (plano 4 campos) | ✅ no-inferior intra-instancia (F3a); multiescala real = F3b |
| valoración | ¿qué me importa? | metas/drives (línea agencia v2) | ✅ confirmada en su entorno (20 seeds); SIN integrar |
| predicción | ¿qué pasará si...? | modelo del mundo / automodelo | parcial (automodelo F3 agencia); sin modelo de tarea |

La tesis arquitectónica que el arco ya destiló: **el que piensa (reasoner),
el que gobierna (mHBP) y el que quiere (metas) son regímenes distintos con
interfaces declaradas** — el gobernador se acopla por I y P sin tocar M; la
valoración fija stakes sin tocar ni M ni el gobierno. Integrar sin respetar
esas fronteras fue siempre la catástrofe (F2); respetarlas, el rescate (F2b)
y la no-inferioridad (F3a).

## Los niveles

### N0 — Sustrato certificado [HECHO]
Reasoner con régimen M certificado (G0/G0.1), plano mHBP con certificados
matemáticos (F1: PSD por construcción, Cayley-IMEX, ρ(Φ) exacto), gobierno
no-inferior con trade-off M3 declarado y localizado en los pesos (F3a+lesión).
Batería T-M-I-P operativa como test de aceptación (detectó −0.055 invisible
para toda métrica convencional).

### N1 — Decidir CUÁNTO: gobierno multiescala bajo escasez [F3b — EN DISEÑO]
La primera decisión autónoma genuina: racionar el propio cómputo a lo largo
de una SESIÓN con presupuesto duro, dificultad con estructura lenta y stakes.
No es accuracy: es política de asignación intertemporal. Doble objetivo:
(A) ¿paga el plano multiescala cuando las escalas existen de verdad?
(B) brazo de re-cableado (gates WM por-instancia) → ¿se elimina el toque M3?
Pre-registro: `mhbp/tasks/reasoner_g0/PREREG_F3B.md` (panel adversarial en
curso antes de implementar). Salida N1: una política de cómputo que anticipa
(paga en OOD de presupuesto y régimen) sin tocar M.

### N2 — Decidir QUÉ IMPORTA: stakes endógenos (integrar la línea de agencia)
En F3b los stakes los da el entorno. En N2 el sistema los DERIVA de metas
propias: se re-conecta la línea de agencia v2 (metas F1, descubrimiento F2,
automodelo F3, online F4 — confirmadas aisladas) sobre el sustrato N1. El
campo risk_priority deja de leer stakes del entorno y los lee del módulo de
metas; el entorno solo da consecuencias. Test: ¿la asignación de cómputo
sigue a las metas cuando el entorno no las señala? Riesgo declarado: la
línea de agencia se confirmó SIN el HBP — integrarla puede repetir F2; la
disciplina es idéntica (modulador acotado, init neutro, batería como gate).

### N3 — Decidir QUÉ HACER: selección de acción con modelo del mundo
Hasta N2 el sistema decide cuánto y qué le importa, pero la agenda se la da
el entorno (instancias en orden). N3 abre el espacio de ACCIÓN: elegir entre
instancias, saltar, volver, pedir más contexto, rendirse. Exige predicción
("¿qué ganaré si insisto?") — un modelo del mundo mínimo: predictor de
(probabilidad de acierto | estado, cómputo restante), que es la extensión
natural de I' (que ya predice corrección desde el estado). La decisión se
vuelve contrafactual, no solo reactiva. Entorno: sesión con menú de
instancias visible y coste por cambio.

### N4 — Cierre: el bucle completo en entorno persistente
percibe → valora (campos + metas) → decide (qué, cuánto) → actúa → aprende,
en un entorno persistente donde ninguna fase es trivial y el desempeño se
mide a horizonte largo. Certificación final: batería T-M-I-P extendida con
los tests de agencia (dosis-respuesta de metas, transferencia OOD) — TODOS
los niveles intactos simultáneamente, no cada uno por separado. Esto es "la
arquitectura completa que decide por sí sola" en versión falsable.

## Reglas del programa (heredadas del arco, no negociables)

1. Forzar capacidad desde el ENTORNO, nunca desde la pérdida.
2. Confirmatorio = pre-registro con ramas de veredicto completas + panel
   adversarial ANTES de gastar GPU + kill-gates baratos primero.
3. La batería es test de aceptación en CADA integración (lo que F3a enseñó:
   el daño real es invisible para accuracy).
4. Fronteras de régimen: gobernar sin tocar M; valorar sin tocar gobierno.
   Toda violación se declara como trade-off cuantificado o se re-cablea.
5. Modelos pequeños, formalismo primero: la contribución es la arquitectura
   certificable, no la escala. Un nulo bien cerrado también es resultado
   (F2b, PDE, sustrato: el programa ya publicó tres cierres honestos).

## Estado (2026-08-01)

- N0 cerrado; F3a con trade-off declarado (−0.055 M3 on-policy, en los pesos).
- N1 (decidir CUÁNTO): la mitad INTRA-INSTANCIA está certificada (P, v3→F3a).
  La mitad SESIÓN se CIERRA en negativo-de-diseño ROBUSTO
  (FINDINGS_F3B_GATES.md): PREREG_F3B v2 tras panel de 58 hallazgos, cirugía
  del modelo completa y verificada 7/7 (queda como infraestructura), pero los
  kill-gates + 31 configuraciones escaneadas (3 diseños de entorno, incl.
  enmienda stake×régimen, E=12, pre-compromiso) demuestran que en S₅ la
  anticipación vale ≤+0.001: el perfil de valor es suave y sin acantilados.
  **N1b (futuro, hipótesis nueva)**: re-testar la escala lenta exige una
  familia de tarea con valor todo-o-nada. 0 h GPU de sesión quemadas.
- **F3a-R CERRADA** (FINDINGS_F3A_R.md): **el toque M3 de F3a NO REPLICA**
  (48 celdas, 3 capas de robustez pre-declaradas; toque contemporáneo −0.011,
  IC95 [−0.068, +0.045] — era ruido de run por celda). Trade-off RETIRADO;
  F3a queda como no-inferioridad de gobierno LIMPIA; por-tick sigue oficial.
  **Regla nueva del programa**: contrastes ≲0.05 en métricas de batería
  exigen réplicas por celda o σ_run estimado; el pareado por seed no parea
  ruido de run.
- **N2 Etapa 1 CERRADA (2026-08-05, FINDINGS_N2.md): doble negativo con
  mecanismo.** C1′ nulo exacto (el acoplamiento valor→gobernador no añade
  nada; sensor AUC 0.95, gobernador sordo corr≈0) + rama no declarada: la
  CE ponderada por stakes SE AUTO-DERROTA (blind_flat +0.147, mejor incluso
  en acc_alto: la respuesta racional al pago 8× es competencia general).
  Cero routing de cómputo por valor en todos los brazos; mecanismo medido:
  el entrenamiento suprime del estado lo tarea-irrelevante + Δacc/Δn suave.
  **Conclusión de la serie de integración (F2b, F3b-gates, N2): las
  decisiones de segundo orden no emergen por gradiente — exigen mecanismo
  EXPLÍCITO de decisión.**
- **N3 fase A CERRADA (2026-08-06, FINDINGS_N3.md): HN3 CONFIRMADA** — la
  decisión computada (n* = argmax ŝ·p̂ − λn sobre automodelos aprendidos)
  captura el 100% del techo ex-ante (+0.151 sobre dificultad, p=8.7e-07,
  routing corr +0.79 donde el gradiente dio 0.00) en 30 min de GPU
  forward-only. Y la JERARQUÍA DE LA INFORMACIÓN cuantificada: uniforme
  0.47 < dificultad 0.55 < +valor ex-ante 0.70 < parada POSTERIOR nativa
  0.92 — observarse pensar vale +0.22 sobre todo lo decidible a priori.
  **VG-B0 ROJO (mismo día): el posterior SATURA el valor** — headroom
  +0.0002±0.0004 sobre el halting nativo; la fase B no se lanza. Jerarquía
  FINAL: 0.467 < 0.546 < 0.698 (ex-ante) < 0.921 ≈ 0.921 (posterior±valor).
  **N3 CERRADO**: la decisión explícita validada como mecanismo y acotada a
  regímenes ex-ante/commitment; donde el sistema puede observarse pensar,
  la parada posterior subsume el valor. La frontera que queda: **N1b — la
  tarea con acantilados** (valor/anticipación pagarían incluso con
  posterior), ahora con tres capítulos de evidencia apuntándole.
- **DECISIÓN (Curro, 2026-08-06): UN SOLO PAPER de cognición**, no varios.
  Estructura de 8 secciones fijada en el mapa conceptual (artifact
  'mapa_miuracognitive'): I-Pensar (régimen certificado) · II-Gobernar
  (plano y sus límites) · III-Decidir (serie de integración + N3) · la
  JERARQUÍA DE LA INFORMACIÓN como figura central (0.467<0.546<0.698<
  0.921≈0.921) · el ESPACIO NEGATIVO como mapa (7 hipótesis muertas con
  mecanismo) · el MÉTODO como contribución (prereg+paneles+gates+réplicas)
  · la predicción de ACANTILADOS como cierre falsable. Siguiente:
  esqueleto LaTeX del paper.
