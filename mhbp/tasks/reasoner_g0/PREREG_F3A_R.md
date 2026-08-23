# Pre-registro — F3a-R: ¿el re-cableado elimina el toque M3? (rama C del fork)

> Declarado 2026-08-02, ANTES de entrenar. Rama C de FINDINGS_F3B_GATES.md,
> ratificada por Curro: la hipótesis del re-cableado (HB) no depende del
> entorno de sesión — se contesta en el entorno F3a EXISTENTE, esta noche,
> mientras el entorno F3b se rediseña (rama A). Contexto: F3a encontró
> M3 on-policy reducido en miura_mhbp vs hbp_full (0.679 vs 0.735, Δ=−0.055,
> p=0.011, dz=−1.62) y la lesión en eval demostró que el desacople está EN
> LOS PESOS (entrenar bajo modulación por-tick lo hornea; FINDINGS_F3A.md).

## Hipótesis y brazos

**HB**: el desacople se hornea por la FLUCTUACIÓN por-tick de las vías de
CONTENIDO (gates WM + block_gate). Si esas vías se congelan dentro de cada
instancia durante el ENTRENAMIENTO, M3 on-policy recupera el nivel del
incumbente sin coste de gobierno.

| brazo | vías de contenido en TRAIN | vías de gobierno (halt+presión) | celdas |
|---|---|---|---|
| **mhbp_pi1** (primario) | congeladas tras el tick 1 (valor dependiente del estado/input, constante el resto de la instancia) | por-tick (intactas) | 6 seeds × {indist, ood} |
| **mhbp_noc** (atribución) | LESIONADAS en train (wm neutral 0.5, gate 0) — la lesión de F3a llevada a entrenamiento | por-tick (intactas) | 6 seeds × {indist, ood} |

Nota: «por-instancia desde el estado persistido» (la variante F3b §5) no es
instanciable en F3a (sin persistencia el estado de frontera es cero →
colapsa a mhbp_noc). mhbp_pi1 es la operacionalización F3a fiel de la
hipótesis de MECANISMO (quitar la fluctuación temporal conservando la
modulación); mhbp_noc separa fluctuación de PRESENCIA.

## Comparadores (sin re-entrenar; limitación declarada)

Las 12 celdas miura_mhbp (por-tick) y las referencias hbp_full/gating_wm de
F3a se REUTILIZAN: el test de regresión T1 (tests_f3b_wiring.py) verifica que
la rama F3a del código actual es idéntica a la que las entrenó. Riesgo
residual declarado: no-determinismo CUDA entre versiones — no re-entrenamos
el control por coste; si el resultado queda al filo del margen, se re-entrena
el control como robustez (pre-declarado, 12 celdas extra).

## Métricas y protocolo (idénticos a F3a — instruments/f3a_run sin tocar)

Primaria: **M3 on-policy** (ligadura fidelidad-de-trayectoria ↔ acierto OOD
en la ventana [1, n_parada]), batería F3a, 6 seeds. Guardarraíles (márgenes
F3a): accuracy largo ≥ incumbente−0.03; P no-inferior (Δ ≥ −0.05); I'
intacta; M1 ≥ 0.9; E reportada. Agregación: media pareada primaria, estricta
6/6 como sensibilidad (la ambigüedad que mordió en F3a, fijada aquí).

## Contrastes (declarados, unilaterales, pareados por seed, n=6)

- **R1 (recuperación, primario)**: M3onp(mhbp_pi1) > M3onp(miura_mhbp
  por-tick) — t pareada unilateral α=0.05. MDE dz≈1.2 (el efecto original
  fue dz=1.62: detectable).
- **R2 (no-inferioridad vs incumbente)**: media M3onp(mhbp_pi1) ≥ 0.735 −
  0.04 = 0.695 (margen del análisis de potencia F3b-v2: <75% del daño).
- **R3 (coste de gobierno, guardarraíl)**: los guardarraíles de arriba sobre
  mhbp_pi1.
- mhbp_noc: SOLO atribución (sin test formal; media + IC y lectura por rama).

## Ramas de veredicto

| pi1 recupera (R1∧R2) | noc recupera | lectura | adopción |
|---|---|---|---|
| ✓ | (cualquiera) | HB CONFIRMADA: la fluctuación por-tick de contenido era el mecanismo | freeze-tras-tick-1 = variante oficial del plano (N2 y brazos F3b) |
| ✗ | ✓ | la PRESENCIA de modulación de contenido (a cualquier ritmo) hornea el toque | si noc no cuesta gobierno (R3 sobre noc): plano sin vías de contenido = candidato oficial; si cuesta: trade-off permanente declarado |
| ✗ | ✗ | el toque vive en las vías de GOBIERNO por-tick o en el lazo mismo | trade-off F3a declarado PERMANENTE (−0.055); sin más barridos (cerrado) |
| ✓ pero R3 falla | — | recupera M pero cuesta gobierno | prioridad M (regla del arco): se adopta y se declara el coste |

Si el IC de R2 cabalga el margen: rama pre-declarada — se adopta el cableado
más limpio si R1 pasa y R3 no falla; si no, «no concluyente» tal cual.

## ENMIENDA 1 (2026-08-03, pre-data para la comparación que introduce)

La robustez pre-declarada (control ptR fresco) reveló una ANOMALÍA DE
REPLICACIÓN: ptR = 0.717 vs las celdas F3a por-tick = 0.679 — mismas seeds,
misma config nominal. Consecuencias: (a) R1' (pi1/noc vs ptR) es NULO — el
re-cableado ni daña ni recupera; (b) el toque F3a re-estimado contra ptR es
−0.018 (vs −0.055 original) con diffs por seed inconsistentes. Causas
indistinguibles desde aquí: ruido de lote-de-runs compartido o efecto sutil
de versión de código (las celdas F3a son pre-cirugía). AMBAS invalidan el
contraste original como medida del toque. Se PRE-DECLARA antes de correr:

- **Re-estimación contemporánea del toque**: hbp_fullR (incumbente fresco,
  12 celdas, código actual) vs ptR — mismo protocolo M3, mismas seeds,
  pareado. El toque re-estimado = media pareada ptR − hbp_fullR con IC95.
- Ramas: |toque| con IC excluyendo 0 y ≥0.03 → el toque replica (se declara
  con su tamaño re-estimado). IC conteniendo 0 → **el guardarraíl M de F3a
  se reclasifica como NO REPLICADO** (artefacto de runs únicos por celda) y
  el trade-off declarado se RETIRA del registro del arco. IC ambiguo → se
  reporta la estimación tal cual, sin veredicto binario.
- El veredicto de adopción de F3a-R pasa a: sin toque replicado no hay nada
  que re-cablear — por-tick sigue oficial por continuidad/simplicidad; la
  lección metodológica (¡una run por celda no estima el ruido de run!) se
  incorpora a las reglas del programa.

## Coste y ejecución

24 celdas × ~31 min (medido F3a) ≈ 12.4 h ≈ 1 noche con 2 workers en la
Blackwell (protocolo v3, MAX_STEPS=2500, N_MAX=24, desde cero, mismos seeds y
streams que F3a). Batería después (~30 min). Runner: f3ar_train.py
(resumible, sharding por brazo).
