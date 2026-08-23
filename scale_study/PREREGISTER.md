# Pre-registro — Estudio de campo homeostático a ESCALA GIGANTE

> Declarado ANTES de mirar ningún número (regla del proyecto: distinguir
> validación de código de resultado científico). Fecha: 2026-07-26.
> Responde al *future work (i)* del paper: "grafos de módulos mayores donde la
> dispersión y la advección tengan espectro suficiente".

## Pregunta

A `N=6` la familia PDE del HBP (onda / difusión / advección `b·A` / dispersión
`β·A³` / no linealidad KdV `ν`) tiene sólo **3 pares de modos**: sin régimen
solitónico, sin dispersión rica, sin transporte coherente de largo alcance. El
paper lo declara como límite explícito del banco de pruebas. La pregunta:

**Cuando el grafo de módulos es GIGANTE (N de 6 → 10⁶), y la dispersión y la
advección tienen por fin espectro suficiente, ¿emerge física cualitativamente
nueva (solitones, transporte coherente, dispersión no trivial), o el campo sigue
siendo un sustrato estructuralmente sencillo?**

Esto es dinámica del campo PURA (propiedad de los operadores y la ecuación), no
entrenamiento: el "espectro" es del grafo, no de los pesos. Es el escenario donde
la palabra "espectro suficiente" cobra sentido literal.

## Hipótesis pre-registradas

- **H1 (espectro).** Para grafos de grado acotado (cadena, anillo, malla 2D) el
  *radio* espectral de L, A, A³ está ACOTADO en N (ρ(L)→4, ρ(A)→2, ρ(A³)→8); lo
  que crece con N es la DENSIDAD de modos. "Espectro suficiente" = densidad, no
  rango. Para grafos de grado creciente (regular-aleatorio, expander) ρ crece
  como √d o d.
- **H2 (dispersión).** En el anillo, la relación de dispersión de la rama de onda
  con `β·A³` es analítica y NO trivial (velocidad de grupo dependiente de k). A
  N grande se resuelve un continuo; a N=6 son 3 puntos. Predicción cuantitativa:
  ω(k) medida = ω(k) analítica a <1% (verificación, no descubrimiento).
- **H3 (SOLITONES — la prueba central).** La rama de primer orden conservativa
  (advección + dispersión + no linealidad) es una discretización de KdV sobre el
  círculo. **Predicción fuerte:** a N≥1024 en anillo, un pulso localizado se
  auto-organiza en solitones que (a) viajan a velocidad ~constante, (b) no se
  dispersan (ancho estable), (c) colisionan cuasi-elásticamente (preservan
  identidad). A N=6 esto es IMPOSIBLE (no caben). Este sería el primer caso en
  que la *sustancia* (tipo de física: dispersión KdV) produce estructura
  cualitativamente nueva a escala.
  - **Sub-hipótesis H3s (saturación).** La no linealidad del modelo es SATURADA
    (`ν·tanh(u)⊙tanh(Au)`, elegida por estabilidad BIBS). Predecimos que la
    saturación DEGRADA o DESTRUYE los solitones frente a la no linealidad genuina
    de KdV (`u⊙(Au)`): la saturación es un precio de la estabilidad. Contraste
    directo saturada vs genuina.
- **H4 (transporte).** Advección `b·A` pura: paquete de onda transportado a
  velocidad de grupo. En anillo (periódico) transporte coherente indefinido; en
  cadena (bordes) reflexión/dispersión en los extremos.
- **H5 (estabilidad a escala).** El umbral de flutter de Merkin
  `β·ρ(A³) < 2ζω₀²` y el certificado de Schur–Cohn siguen siendo TIGHT a N grande
  (se predicen y verifican divergencias reales al cruzar el umbral), y como
  ρ(A³)≤8 en grado acotado, el umbral NO se endurece con N para esas topologías.

## Diseño del barrido (lo que corre toda la noche)

Todo con la MISMA física de `model/hbp.py` (verificada por reproducción a N=6).
FP64 para física fina (solitones, conservación); FP32 permitido en barridos
masivos. GPU Blackwell. Operadores por `roll`/shift (chain/ring/grid2d, O(N)) y
sparse CSR (grafos aleatorios). Resumible: 1 JSON atómico por celda.

- **B1 Escaneo espectral.** topología ∈ {chain, ring, grid2d, ws_smallworld,
  random_regular, expander} × N ∈ {6, 16, 64, 256, 1024, 4096, 16384, 65536,
  262144, 1048576} → ρ(L), gap, densidad de modos, ρ(A), ρ(A³), umbral flutter.
- **B2 Relación de dispersión (anillo).** N ∈ {64…65536} × barrido de modo k ×
  {sin dispersión, β activo} → ω(k) medida vs analítica; velocidad de grupo.
- **B3 SOLITONES (anillo, pseudo-espectral FFT).** N ∈ {256,1024,4096,16384,
  65536} × amplitud × ancho de pulso × β × ν × {saturada, genuina} × réplicas →
  integración larga (10³–10⁵ ticks); nº de solitones, velocidad, persistencia de
  ancho, elasticidad de colisión, recurrencia FPUT. **El bloque caro.**
- **B4 Estabilidad/flutter a escala.** N grande × barrido de β cerca del umbral ×
  {giroscópico, circulatorio} → rollout Verlet real; ¿decae o diverge?; A/B
  contra el certificado.
- **B5 Onda amortiguada gigante.** La figura del paper pero a N=10⁶: impulso
  local → propagación-oscilación sobre un millón de nodos (mapa nodo×tick).

## Criterios de veredicto (pre-especificados)

- **H3 CONFIRMADA** si a N≥1024 se detectan ≥1 solitones con velocidad estable
  (CV<10% sobre la trayectoria) y ancho estable (CV<20%) durante ≥10³ ticks, y la
  colisión preserva velocidad a <15%. **REFUTADA** si el pulso se dispersa
  monótonamente (ancho crece ~√t) sin estructura coherente.
- **H3s CONFIRMADA** si los solitones de la versión saturada tienen vida media
  <50% de la genuina (o desaparecen).
- **H2/H5** son VERIFICACIONES (analítico conocido): PASS/FAIL a tolerancia.
- Nada de esto mide accuracy de tarea: es dinámica de campo. El vínculo con el
  nulo de mecanismo (la sustancia no mueve accuracy) se discute, no se re-testea.

## Lo que este estudio NO es

No es entrenamiento (el HBP acopla N a los módulos del transformer; escalar N en
la tarea exige desacoplar el campo de los módulos — trabajo arquitectónico aparte
que se menciona como siguiente paso, no se ejecuta aquí). Es la caracterización
dinámica del campo a escala, que es donde "espectro suficiente" es literal.
