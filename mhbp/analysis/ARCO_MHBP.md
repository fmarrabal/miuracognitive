# El arco mHBP: asentar, razonar, gobernar
## Registro científico consolidado del programa (Fases 1–2b) y marco conceptual

> Documento maestro del punto (1). Consolida F1/F2/F2b con la tesis que el
> programa demostró sin haberla formulado al empezar. Fuentes primarias:
> MATH_SPEC.md, PREREG_PHASE2(B).md, FINDINGS_PHASE2(B).md, REPORT_PHASE2(B).md.
> Fecha: 2026-07-30.

---

## 1. Tesis

**Asentar, razonar y gobernar son tres regímenes dinámicos distintos, y el
programa mHBP los demarcó experimentalmente:**

- **Asentar** (settling): dinámica cuyo transitorio es coste y cuyo destino
  (punto de consigna) es la computación. Es lo que un campo homeostático hace
  por construcción — y lo que el descenso por gradiente *encuentra* siempre
  que la pérdida no haga visible el camino (nulo de coarse-graining del
  programa HBP original).
- **Razonar**: régimen en el que la trayectoria porta contenido causalmente
  usado (ver §5). No es una propiedad del módulo sino del par (sistema, tarea).
- **Gobernar**: meta-control separable — cuánto, cuándo, dónde — que se acopla
  al razonamiento por sus interfaces (halting, asignación) sin tocar su
  mecanismo.

El resultado central del arco: **un campo homeostático certificado es un
asentador excelente; usado para decidir contenidos, rompe; usado para
gobernar, estabiliza sin decidir.** Cada fase aportó una cara:

| fase | rol del campo | veredicto | evidencia clave |
|---|---|---|---|
| F1 | sustrato matemático | ✓ certificado por construcción | Cayley-IMEX ortogonal exacto; ρ(Φ) exacto por sondeo; 114/114 tests; resonancia giroscópica θ=½ descubierta y cerrada |
| F2 | FUENTE de decisiones | ✗ catástrofe OOD | dz=4.06 vs GRU; fallo exclusivo en cambio de presupuesto; dos mecanismos localizados |
| F2b | MODULADOR de ganancias | ✓ rescate / ∅ inerte | dz=−4.33 vs F2 (manipulación de mecanismo); g≡1 ⇒ Δ=0.0000; mejor settling de la tabla |
| (v3, programa HBP) | estabilizador de halting | ✓ único positivo replicado | corr(K,E[n_iter]) OOD, p=0.025/0.028; estructural, no de constantes |

## 2. Fase 1 — El asentador certificado

Cuatro campos acoplados (τ = 1/4/8/32) con estabilidad POR CONSTRUCCIÓN:
acoplamiento como potencial de interfaz (Gram PSD para cualesquiera parámetros
aprendidos), escalas ordenadas por construcción (cumsum-softmax con logit
fantasma), integrador Cayley-IMEX (rotación giroscópica exactamente ortogonal —
elimina la inyección de energía discreta del Verlet del HBP original) y
certificado ρ(Φ) EXACTO por sondeo del tick completo. Hallazgo matemático
propio: la resonancia giroscópica del esquema θ=½ (media rotación de Cayley
= π/2 compuesta con el modo flip de Crank–Nicolson ⇒ ρ=1 exacto dentro de las
cajas), cerrada con condición anti-resonancia verificada en prepare().
Verificado adversarialmente (45 hallazgos, 2 críticos, todos cerrados).

## 3. Fase 2 — El campo como fuente: catástrofe con mecanismo

Entorno SMA (tres escalas latentes genuinas; kill-gates de adaptación,
resolubilidad, no-leakage y oráculo-convergido en verde; pre-registro v1→v3c
con enmiendas fechadas). Resultado: los tres contrastes primarios
significativos EN CONTRA (mhbp peor que gru dz=4.06, que escalas-iguales
dz=2.24, y 2º orden peor que 1º dz=0.94), con el fallo EXCLUSIVO de los
protocolos de cambio de presupuesto y el error de tarea idéntico a baselines.

Dos mecanismos separados por protocolo (descomposición de J):
1. **El campo-planificador no extrapola nivel** (budget_hi: 96% del daño en el
   término de plan; TODAS las variantes de campo fallan igual → no es la
   inercia, es el rol). El estado del campo es un filtro con dinámica aprendida
   calibrado a las estadísticas de entrenamiento; el readout de reserva hereda
   esa calibración. El reactivo lee la señal y ajusta.
2. **La inercia re-adapta lento** (e4: hard 17.3 vs 5.5 del 1er orden;
   settling 2.5 ventanas vs 0.4 del mlp). La MISMA inercia del positivo v3:
   estabilidad de una política buena ↔ rancidez tras un shift real. **La
   inercia es un prior bidireccional.**

Las ablations (acoplamiento, alostasis: sin efecto) localizan el daño en la
dinámica por-campo como sustrato de decisión, no en el cableado multiescala.

## 4. Fase 2b — El campo como modulador: rescate por manipulación, gobernador inerte

Manipulación del mecanismo: mismo lazo reactivo (bit-idéntico en init), el
campo pasa de originar acciones a modular ganancias acotadas (g ∈ [0.2, 1.8],
init g≈1). Controles: react (lazo puro) y gru_gov (misma modulación, GRU en
vez de campo — aísla la física). Panel adversarial previo (18 hallazgos, 3
críticos: la vía bilineal documentada como conductual-no-arquitectural, la
intervención g≡1 hecha ejecutable con checkpoints, D2 partido).

- **D1 (rescate)**: J_OOD 10.11→2.32 (dz=−4.33, Holm SÍ). plan 34.5→2.3 (15×),
  hard 21.1→0.68 (el MEJOR de la tabla), settling 2.52→0.28 (el MÁS RÁPIDO,
  mlp incluido). El fallo de F2 era el ROL — confirmado por manipulación.
- **D2b (física aislada)**: gov peor que gru_gov (+0.27, Holm SÍ). PERO la
  intervención g≡1 da Δ=+0.0000 EXACTO en ambos: las ganancias aprendidas son
  INERTES — ni el campo ni el GRU explotan modulación variable; el óptimo del
  entorno no la requiere (confound de entorno declarado: no se puede
  distinguir "el campo no sabe modular" de "no había nada que modular").
  El +0.27 es interferencia de ENTRENAMIENTO, no modulación dañina.
- Nota descriptiva a favor: mhbp_gov logra el mejor settling y el mejor hard
  del escalón — consistente con v3: **el campo estabiliza transitorios**.

## 5. La contribución conceptual: el régimen de razonamiento (batería T-M-I-P)

El arco fuerza una definición operativa que el programa necesitaba. "Reasoner"
no es un tipo de módulo: es un RÉGIMEN de la tripleta (tarea, sistema,
entrenamiento), certificable por cuatro tests independientes:

- **T (tarea): irreducibilidad serial.** Existe una dimensión del input a lo
  largo de la cual la computación no se paraleliza (certificados: NC¹-dureza;
  instrumento: composición no conmutativa). Sin T, la iteración es una
  comodidad y el régimen no puede forzarse. *(La no conmutatividad es
  certificado suficiente, no esencia: la agregación bayesiana conmuta y
  razona; el conteo está en TC⁰.)*
- **M (mecanismo): semántica causal de la trayectoria.** Tres sub-tests:
  **M1** decodabilidad — los estados intermedios decodifican resultados
  parciales, con fidelidad creciente a lo largo del cómputo; **M2** causalidad
  de contenido — inyectar el estado intermedio de OTRO input produce errores
  específicos del contenido inyectado (la conclusión sigue al swap) o coste de
  re-derivación medible (ticks extra); **M3** ligadura — la fidelidad de
  trayectoria por instancia predice la extrapolación composicional (longitud).
  *(Diseñado contra tres contraejemplos: el caos —degrada genérico, falla M2—,
  la auto-reparación —paga ticks, pasa M2—, y el scratchpad de CoT —cómputo
  sin contenido, falla M2. La relajación tipo Adam queda fuera por T.)*
- **I (interfaz): parada informada.** La señal de halting está calibrada
  respecto a la completitud (AUC(p_halt, corrección) — NO se exige curva
  anytime monótona: esa es la firma del asentador; el insight salta).
- **P (política): la profundidad extrapola.** corr(dificultad, E[n_iter]) se
  mantiene fuera de la distribución de entrenamiento (el protocolo v3 era una
  medición de P avant la lettre).

**Frontera reasoner/gobernador, ahora exacta**: el gobernador se acopla por
I y P sin tocar M. F2 = tocar M (catástrofe). F2b = soltar M (rescate). v3 =
estabilizar P (el positivo). La tesis del paper governor ("modula, no piensa")
es el enunciado macro de esta frontera.

**Autocrítica retrospectiva que la batería habilita**: los nulos del programa
HBP original son consistentes con que el par (reasoner, tareas) operaba en
régimen de ASENTAMIENTO (agrupamiento ~5 ticks/19 ops; transitorio invisible a
la pérdida): se estuvo poniendo gobernador a un sistema sin M. De ahí G0
(PREREG_G0.md): certificar el régimen ANTES de la integración de Fase 3.

## 6. Qué queda sin demostrar (honestidad de cierre)

1. La mitad CONSTRUCTIVA de la tesis governor (la modulación del campo añade
   valor) no tiene entorno donde demostrarse: en SMA nadie necesitó ganancias
   variables. Requeriría demanda genuina de conmutación tick-alineada — y
   diseñarla ex profeso rozaría el ajuste del entorno a la hipótesis.
2. El positivo v3 sigue sin sobrevivir Holm individualmente (replicado y
   direccional, no definitivo).
3. h* (setpoint aprendible) nunca recibió gradiente en ningún experimento del
   arco: el formalismo alostático está implementado pero no ejercitado.
4. Todo a escala pequeña (≤11k parámetros de controlador; entornos sintéticos).

## 7. Piezas del arco y dónde están

- Formalismo y certificados: `mhbp/MATH_SPEC.md`, `mhbp/stability/`, 114 tests.
- Confirmatorios: `mhbp/tasks/synthetic_multiscale/` (PREREG v1→v3c, 2B v1→v2),
  184 celdas, checkpoints de la familia gov.
- Registros: `FINDINGS_PHASE2.md`, `FINDINGS_PHASE2B.md`, reports.
- Este documento: marco para la escritura formal (candidato: sección "The
  control plane is not the reasoner" del paper governor, o companion corto).
