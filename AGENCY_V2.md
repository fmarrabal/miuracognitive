# Línea de agencia v2 — rediseño completo tras la auditoría

> Rediseño de las fases 1-4 (AHA → metas endógenas → descubrimiento → automodelo)
> corrigiendo los hallazgos FATALES/MAYORES de la auditoría adversarial
> (memoria `auditoria-linea-agencia`). Los originales quedan intactos como
> registro; todo lo nuevo vive en `*_v2`. Escala: la capacidad debe EMERGER
> de presión de entrenamiento; el mecanismo puede ser sesgo inductivo, nunca
> la respuesta.

## Reglas anti-circularidad (vinculantes para las 4 fases)

R1. **Baselines externos fuertes, afinados.** Cada contraste primario es
    `aprendido − mejor rival scripted AFINADO` (grid-search en seeds de tuning
    disjuntas de las de evaluación). Las ablaciones de input de la propia red
    se reportan como ablaciones, nunca como "baseline".

R2. **Entornos donde el rival PUEDE ganar con habilidad.** Ninguna constante
    calibrada para matar aritméticamente al rival. Donde aplique, la ventaja
    se mide como DOSE-RESPONSE (p.ej. sobre el retardo de acción), no en un
    único punto amañado.

R3. **Señales predictivas ruidosas y parciales.** Nada de oráculos (magnitud
    exacta, canal exacto, adelanto fijo). Fiabilidad <1, adelanto con jitter,
    magnitud con ruido, falsas alarmas, canal a veces ambiguo. Anticipar =
    INFERIR bajo incertidumbre, no copiar con retardo.

R4. **Prohibido supervisar un mecanismo hacia su propia regla programada.**
    Los gates/memorias/propuestas se entrenan SOLO desde el objetivo
    ambiental (recompensa/viabilidad), jamás con BCE/CE hacia un comparador
    escrito a mano.

R5. **Prohibido inyectar la ley generativa del entorno en el agente.** Sin
    inversiones analíticas con parámetros verdaderos (σ, matrices de
    actuadores, constantes de planta). El modelo del mundo/del cuerpo se
    APRENDE (de forma genérica) desde observaciones.

R6. **Aprendizaje real y verificado en cada fase.** Test automático: el
    agente SIN ENTRENAR debe rendir sustancialmente peor que entrenado
    (guardrail G-untrained). Si un resultado se reproduce sin gradientes, la
    fase está mal diseñada.

R7. **Sin techos.** Si algún brazo supera 0.98 en la métrica primaria, la
    dificultad escala (y se reporta). Los efectos deben ser graduados; un
    p=suelo-del-sign-test con efectos deterministas no es evidencia, es
    diseño.

R8. **HBP en condiciones limpias o fuera.** Brazos HBP: mismo device, mismo
    presupuesto hasta criterio de convergencia, contraste de integrador puro
    vía `alpha_const∈{1,0}` con rangos IDÉNTICOS (solver implícito), ganancia
    de acoplamiento APRENDIBLE con init pequeña (diagnóstico de interferencia
    de la auditoría), `pin_fp32`. Exploratorios hasta que ganen algo.

## Protocolo estadístico común (pre-registrado por fase antes de mirar números)

- 20 seeds frescas pareadas por fase confirmatoria; seeds de tuning y de
  piloto disjuntas de las confirmatorias.
- Primarios: 1-2 contrastes por fase (aprendido − mejor scripted). Test de
  signo exacto + t pareada; Holm sobre la familia primaria. Efectos con IC95.
- Guardrails obligatorios: G-untrained (R6), G-ceiling (R7), G-tuning (el
  scripted se afinó de verdad: reportar el grid y el óptimo), G-LOSO
  (dejar fuera la seed más influyente), G-dose (pendiente dosis-respuesta
  donde aplique).
- Piloto (3 seeds) DECIDE la config; el confirmatorio se congela después y
  no se toca. Los pilotos se reportan como pilotos.

## Fase 1 v2 — AHA-2: regulación anticipatoria emergente

**Entorno** (`data/aha_v2.py`): necesidades continuas con drenaje basal,
eventos de hazard con precursores IMPERFECTOS: fiabilidad 0.75, adelanto
U{4..9}, magnitud ×U(0.7,1.3), 15% de cues en canal equivocado, falsas
alarmas (sin evento) a tasa 0.35/trayectoria·necesidad. Coste de acción
AMBIENTAL (drena las otras necesidades). `action_delay` es parámetro barrido
∈{0,1,2,3}: en 0-1 un reactivo hábil sobrevive bien; la ventaja anticipatoria
debe CRECER con el retardo (dose-response).

**Agentes** (`model/aha_v2.py`): política GRU aprendida (observa niveles,
error, velocidad, canal de cues actual, acción previa; la memoria debe
integrar cues con adelanto incierto). Pérdida = objetivo homeostático
ambiental idéntico para todos los brazos aprendidos (penalización por déficit
bajo setpoint + castigo por inviabilidad; el coste de actuar ya lo cobra el
entorno). Scripted afinados: `threshold` (actúa si nivel<θ), `cue_follower`
(programa acción tras cue con offset τ y umbral c), `combo` (ambas reglas,
afinado conjunto). Ablación: misma red con cues a cero (etiquetada ablación).
HBP: ejecutivo + campo con ganancia aprendible, `alpha_const` 1 vs 0.

**Primarios**: (P1) supervivencia aprendido − combo afinado en delay=2;
(P2) pendiente dosis-respuesta de esa diferencia sobre delay∈{0..3}.

**Claim si PASS**: "una política recurrente aprendida integra señales
predictivas ruidosas y parciales mejor que la mejor regla scripted afinada,
y su ventaja crece con la latencia de acción" — anticipación estadística
aprendida, sin oráculos.

## Fase 2 v2 — compromiso endógeno emergente

**Entorno** (`data/goals_v2.py`): K proyectos con progreso; completar uno
rinde su valor; el progreso DECAE al abandonar; interrupciones: crisis reales
(atenderlas rinde/salva) y distractores calibrados para aliasear el estado
instantáneo (se conserva la idea buena del original). Episodio valorado por
retorno total, no supervivencia (evita techo).

**Agentes**: política GRU con memoria (el compromiso debe emerger del coste
de cambiar), SIN gate programado ni BCE (R4). Scripted afinados: greedy
urgencia; histéresis con margen afinado; histéresis+override de crisis
(afinado conjunto). Ablación: red sin recurrencia.

**Primarios**: (P1) retorno aprendido − mejor scripted afinado; (P2) retorno
OOD con estadísticas de interrupción nuevas (generalización del compromiso).

## Fase 3 v2 — descubrimiento de objetivos sin ley conocida

**Entorno** (`data/discovery_v2.py`): campo de respuesta DESCONOCIDO por
episodio: familia sorteada entre {RBF con σ aleatoria, meseta, cresta,
bimodal}, con ruido de observación en cada sonda. Presupuesto de sondas
limitado. El agente NO recibe la forma funcional ni sus parámetros (R5).

**Agente**: política de sondeo aprendida IN-CONTEXT (GRU que consume el
historial (posición, respuesta) y propone la siguiente sonda y el objetivo
final). Meta-entrenada entre episodios para maximizar la respuesta del
objetivo elegido con el presupuesto dado. Scripted afinados: sondeo aleatorio
+ argmax observado; rejilla + argmax; ascenso por diferencias finitas con
paso afinado. Skyline (no baseline): oráculo con la forma verdadera.

**Métrica**: regret normalizado = 1 − f(objetivo_elegido)/f(máximo_real),
GRADUADO (el ruido y el presupuesto impiden 0). Primario: regret aprendido −
mejor scripted, y curva de eficiencia (regret vs nº de sondas).

## Fase 4 v2 — automodelo aprendido con daño parcial

**Entorno** (`data/self_model_v2.py`): la planta puede seguir siendo el
cuerpo de onda (buena conexión con el formalismo), pero el agente NO recibe
sus constantes ni la matriz de actuadores (R5). Daño PARCIAL a mitad de
episodio (efectividad ×U(0.2,0.6) de un actuador aleatorio, sin flag) y
ruido de observación.

**Agente**: automodelo = red genérica (lineal aprendida o MLP pequeño)
entrenada ONLINE dentro del episodio desde transiciones (acción→Δestado),
inicialización aleatoria; el controlador planifica con el modelo APRENDIDO.
Baselines: modelo congelado (sin adaptación online); re-init tras el daño;
skyline con modelo verdadero. Dose-response sobre severidad del daño.

**Métrica**: regret de control vs skyline, tiempo de recuperación. Graduado.

## Gates de decisión

- G0 (por fase): smoke CPU + G-untrained + piloto 3 seeds sin techo →
  congelar protocolo → confirmatorio 20 seeds.
- G1: si el aprendido NO bate al mejor scripted afinado en el piloto, se
  reporta tal cual y se decide si la fase tiene sentido (no se "ajusta el
  entorno hasta que gane").
- G2 (HBP): exploratorio; pasa a primario solo si gana en ≥2 pilotos
  independientes.

---

# RESULTADOS CONFIRMATORIOS (20 seeds frescas 300-319, protocolos congelados)

## Fase 1 — AHA-2 (regulación anticipatoria emergente) ✅ PASS
- **P1 learned − combo afinado @d2: +0.514** (20/20, t=19.7, p_Holm=3.8e-6).
- **P2 pendiente dosis-respuesta: +0.081/delay** (17/20, t=5.9, p_Holm=2.6e-3):
  la ventaja del aprendizaje CRECE con la latencia de acción.
- G-LOSO robusto (+0.528 sin la seed más influyente).
- Secundario pre-registrado como null esperado que resultó POSITIVO:
  learned − cueblind = **+0.116** (p=4e-4): los cues exógenos ruidosos SÍ
  aportan encima de la señal endógena (onset + batería). Se reporta tal cual.
- HBP exploratorio: hbp_diff −0.016 (p=0.82), hbp_wave −0.069 (p=0.12) vs
  learned: con ganancia de acoplamiento aprendible NO hay interferencia
  (la de v1 era artefacto de la ganancia fija) y tampoco ventaja.
- Historia del piloto: 5 iteraciones documentadas, todas con fixes
  AMBIENTALES (nunca de pérdida): ST-Gumbel→REINFORCE; actuar-siempre→tope+
  batería; profilaxis barata→tormentas letales.
- Claim defendible: "una política recurrente aprendida bajo economía de
  energía finita desarrolla regulación anticipatoria (onset interoceptivo +
  cues exógenos ruidosos + planificación de presupuesto) que la mejor
  familia scripted afinada no alcanza (+0.51 supervivencia), con ventaja
  creciente en la latencia".

## Fase 2 — compromiso endógeno ✅ PASS
- **P1 in-dist: +0.599** (19/20, p=4e-5) sobre `smart` (scripted que CONOCE
  la regla generativa del precursor).
- **P2 OOD: +0.697** (19/20): la ventaja CRECE al cambiar las estadísticas
  de interrupción (la regla fija degrada; la política aprendida aguanta).
- Recurrencia load-bearing: +0.238 vs memoryless (p=4e-4).
- Nota de mecanismo (honesta): el agente atiende crisis Y distractores;
  el mecanismo es PRIORIZACIÓN DE VALOR aprendida, no discriminación
  crisis/distractor.

## Fase 3 — descubrimiento sin ley conocida ✅ PASS
- **P1 regret mejor_scripted − learned: +0.071** (20/20, p=1.9e-6; grid
  afinado 0.211 → learned ≈0.140): el prober in-context meta-aprendido
  localiza mejor los máximos de campos DESCONOCIDOS (4 familias, σ
  aleatoria, ruido de observación, 12 sondas) que el mejor sondeo scripted
  afinado, sin recibir jamás la forma funcional (R5).
- G-untrained: 0.629 sin entrenar (el resultado se GANA con gradientes;
  contraste directo con la v1, donde el discoverer sin entrenar ya daba 1.0).

## Fase 4 — automodelo aprendido con daño parcial ✅ PASS
- **P1 frozen − adaptive: +0.062/+0.056/+0.057** en severidades {0.2,0.4,0.6}
  (20/20 en todas): la adaptación online continua paga tras el daño.
- P2 (pre-registrado tras el piloto): reinit-afinado − adaptive = 0.0000
  exacto: **el mejor detector de cambio afinado degenera en adaptación
  continua** — hallazgo, no fracaso.
- Regret vs oracle-con-verdad: 0.099 (graduado; sin techo).

## Fase 2-ESCALADA — F2X campo ejecutivo ⚠️ NEGATIVO-CON-ESTRUCTURA (honesto)
Escala F2 al conflicto ejecutivo completo: 6 proyectos con plazos (capacidad
física ≈3 → triaje forzado), ENERGÍA finita con colapso (automantenimiento;
acoplamiento con F1), crisis/distractores aliasados. Rival: `designer` afinado
= scheduling por densidad de valor + histéresis + detector del precursor
(CONOCE la regla generativa) + gestión de energía. Receta del aprendido tras 3
iteraciones documentadas (por-trayectoria→reward-to-go→A2C+curriculum;
invariante Σr_t≡retorno verificado a 0; entorno NUNCA tocado para ganar, G1).

**Resultado (20 seeds, congelado): el designer GANA en las 3 condiciones, pero
el gap se ESTRECHA monótonamente con el desplazamiento de distribución:**
- P1 in-dist:  learned 3.256±0.053 vs designer 3.727 → **gap −0.471** (0/20, LOSO −0.483)
- P2 OOD-interr.: learned 3.620 vs 3.896 → **gap −0.276** (0/20)
- P3 OOD-energía: learned 2.813 vs 2.861 → **gap −0.049** (5/20, p=0.04, LOSO −0.058)

Lectura honesta: a la escala ejecutiva, un scheduler experto que conoce la
regla generativa sigue siendo superior al ejecutivo aprendido in-distribución.
PERO la desventaja relativa del aprendido cae un 90% bajo el mayor cambio de
distribución (metabolismo +50%) — misma señal cualitativa de robustez OOD que
F2-base (donde el aprendido GANABA y ganaba más OOD), aquí partiendo de un
déficit. El aprendido SÍ desarrolla un ejecutivo competente (triaje, ~1.9
proyectos a tiempo, colapsos ≈0.5, gestión de energía), solo que no bate al
experto diseñado. Es el redseño funcionando: cuando el aprendido no gana, se
reporta; no se ajusta el entorno hasta que gane.
Secundarios (8 seeds): recurrencia load-bearing (learned 3.254 vs memoryless
3.003, **+0.251, p=0.008** → el ejecutivo SÍ necesita estado y lo usa).
**HBP NEUTRO en su prueba más natural** (conflicto multi-impulso): hbp_wave
3.286 (learned−hbp_wave=−0.032, p=0.29, no significativo), hbp_diff 3.254
(−0.0006, p=1.0, idéntico). Disociación limpia: la recurrencia importa, el
campo homeostático no. Cierra el hilo del HBP-como-sustrato en toda la línea de
agencia (F1 neutro + F2X neutro), coherente con el mecanismo-null de la línea
principal: el campo no confiere ventaja de conducta/retorno; su único valor
demostrado es como controlador de cómputo (adaptividad OOD del 2º orden).

## Lo que las 4 fases NO afirman (alcance honesto)
Capacidades de regulación/adaptación aprendidas en entornos diseñados, con
rivales afinados y controles causales. NO: metas abiertas, autoconciencia,
"alma", ni evidencia a favor del HBP como sustrato (exploratorio: neutro en
F1 con integración limpia). El marco Ello/Yo/Superyó es metáfora interna de
organización, no lenguaje científico.
