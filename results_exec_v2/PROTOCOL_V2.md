# Protocolo confirmatorio F2X — campo ejecutivo (congelado tras piloto)

## Config congelada
- Entorno: ExecV2Config (6 proyectos, plazos U[0.45,1.0]·T, energía con colapso,
  capacidad ≈3/6 proyectos -> triaje forzado; crisis con precursor + distractores
  aliasados). SIN CAMBIOS desde el tune.
- Receta (historia del piloto, transparencia): REINFORCE por-trayectoria falló
  la asignación de crédito (2.50); reward-to-go incremental (invariante
  Σr_t==retorno verificado a 0.00) -> 2.60; +A2C crítico + curriculum de
  ENTRENAMIENTO sobre work_rate (0.12→0.09→0.06; eval siempre en el entorno
  real) -> 2.95@1500, 3.30@2500, 3.41@4000. CONGELADO: 4000 pasos.
- Mejor scripted (tune): designer (e_min=0.1, margin=0.02) = 3.70-3.72.

## Expectativa pre-registrada (honestidad)
El designer (conoce la regla generativa y el scheduling por densidad de valor)
es FAVORITO in-dist: se espera P1 NEGATIVO (gap ≈ −0.3). Las hipótesis vivas
son P2/P3: en F2-base la ventaja aprendida CRECÍA en OOD; si la regla del
designer es frágil ante interrupciones ×2 (P2) o metabolismo +50% (P3), el
aprendido puede ganar fuera de distribución aunque pierda dentro. Cualquier
patrón se reporta tal cual (G1).

## Primarios (seeds 300-319, pareados, Holm sobre {P1,P2,P3})
- P1: retorno learned − designer afinado, in-dist.
- P2: ídem, OOD-interrupciones (distractores 3→6, crisis 2→3).
- P3: ídem, OOD-energía (drena +50%).

## Secundarios
- learned − memoryless; brazos HBP (alpha_const 1/0, ganancia aprendible);
  regret vs skyline; colapsos; completados a tiempo.

## Guardrails
- G-untrained, G-ceiling, G-tuning (grids persistidos), G-LOSO sobre P1-P3.
