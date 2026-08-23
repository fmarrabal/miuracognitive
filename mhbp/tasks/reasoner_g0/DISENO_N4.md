# N4 — El campo como ACUMULADOR DE EVIDENCIA INTEROCEPTIVA
### Diseño pre-panel (2026-08-09). No es prereg todavía.

## 0. La pregunta de Curro

«El campo homeostático funciona un poquito — ¿qué le falta para que
funcione bien?»

## 1. Diagnóstico desde el espacio negativo (todo ya medido)

Cada fracaso del HBP tiene la MISMA forma: le pedimos al campo que
**llevara contenido** o que **decidiera por gradiente**:

| intento | resultado | mecanismo medido |
|---|---|---|
| campo como sustrato de cómputo (familia PDE) | nulo | el gradiente coarse-grains el tiempo → toda modulación colapsa a set-point |
| campo porta el plan (mHBP F2) | peor que GRU | el readout vía campo no extrapola nivel; la inercia re-adapta lento |
| campo ejecutivo (F2X) | negativo-con-estructura | designer-con-regla gana |
| física rica (solitones, escala) | irrelevante | vive donde el modulador no la usa |
| flujo 2D (NS) | imposible | ∇·u=0: mezcla, nunca concentra/entrega |

Y cada éxito del programa tiene TAMBIÉN la misma forma:

1. **El posterior es el canal más valioso que existe**: observarse a uno
   mismo mientras piensa vale +0.22 sobre todo lo ex-ante (jerarquía N3:
   0.698 → 0.921) y satura al valor (VG-B0).
2. **Las decisiones de segundo orden deben COMPUTARSE, no emerger**
   (N2/N3: el allocator explícito n* = argmax ŝ·p̂ − λc captura el 100%
   del techo ex-ante; PonderNet+β no).
3. **Los sensores deben ser CRUDOS**: el entrenamiento suprime del estado
   la información irrelevante para la tarea (N2: probe 0.79 desde estado
   vs 1.00 desde embeddings).
4. Lo único del campo que sobrevive es **control de cómputo** y la
   **recurrencia** (+0.25, p=0.008).

## 2. La tesis de N4

Al campo no le falta un término en la EDP. Le faltan:

- **OÍDOS (interocepción del pensamiento)**: forzamiento g_φ alimentado
  con el *stream posterior crudo por tick* del reasoner — posterior de
  halting p_t y su entropía, ‖Δh_t‖ (velocidad de la trayectoria),
  actividad de gates de la WM, consistencia entre ticks. Crudo, sin pasar
  por estado entrenado (lección 3). El VEI del diseño original prometía
  interocepción del "cuerpo computacional"; nunca se conectó al único
  canal que los datos dicen que vale: el posterior del propio razonar.
- **UN LECTOR EXPLÍCITO**: la decisión (¿paro? ¿cuántos ticks más?) no la
  toma el gradiente modulando umbrales — la computa un allocator/stopper
  explícito estilo N3 que LEE el campo: parar cuando h cruza umbral,
  n* = argmax sobre p̂(éxito | estado del campo) calibrada (lección 2).

Con esas dos piezas, la dinámica del campo tiene POR FIN un trabajo que
un set-point no puede hacer: **acumular evidencia ruidosa en el tiempo**.
La ecuación de onda amortiguada ES un acumulador con fugas e inercia — el
análisis secuencial (SPRT / drift-diffusion) hecho física. Y aquí el
contraste 2º-vs-1º orden tiene por primera vez una predicción mecanicista
limpia, en lenguaje de RMN: un filtro de 2º orden subamortiguado es un
**detector resonante** (ganancia selectiva en frecuencia, matched filter);
uno de 1º orden es un paso-bajo. Si la evidencia sobre «esta instancia
está condenada / resuelta» vive en la ESTRUCTURA TEMPORAL del stream
posterior (transitorios, oscilaciones de la trayectoria), el 2º orden
puede extraerla y el 1º no. Si vive solo en el nivel medio, empatarán —
y eso también es un resultado.

## 3. Escalera de gates (baratos primero; kill-gate antes de GPU)

- **G1 (kill-gate, CPU, sobre checkpoints/trazas existentes de N3)**:
  ¿el stream posterior por tick contiene información MÁS ALLÁ del último
  tick? Probe: predecir éxito con (a) posterior del último tick vs
  (b) stream completo (features temporales). Si el último tick es
  estadístico suficiente (tarea ≈ markoviana en el posterior), N4 muere
  aquí por ~0 GPU. Umbral: ΔAUC ≥ 0.03 con IC pareado.
- **G2 (integración física)**: campo-acumulador (Verlet, ζ/ω₀ aprendibles
  con pin_fp32) alimentado por el stream vs baselines DUROS: último tick,
  EMA (1º orden puro), GRU pequeña con mismos inputs. ¿El acumulador
  cierra parte del hueco ex-ante→posterior (0.698→0.921) a cómputo
  emparejado? Gate: superar a último-tick Y no perder contra GRU.
- **G3 (el contraste 2º-vs-1º con entorno que lo fuerza)**: regla de oro
  — capacidad forzada por el ENTORNO: tarea donde la evidencia posterior
  tenga estructura temporal genuina (p.ej. dificultad que se revela en
  ráfagas/transitorios, no en nivel). hbp_wave vs hbp_first vs EMA.
  Riesgo declarado: circularidad (construir el entorno para que gane la
  onda) — el panel debe atacar esto.

## 4. Qué NO es N4

- No es re-litigar el sustrato: el campo no lleva contenido (cerrado).
- No es tocar la loss: nada de L_homeo empujando la decisión — la
  decisión es un cómputo explícito aguas abajo.
- No aplica al actuador LLM tal cual: ahí el sensor posterior está roto
  (acuerdo engañoso, modal-erróneo 0.833; parar-por-acuerdo < ex-ante).
  N4 vive donde el posterior informa: el sustrato neuronal.
