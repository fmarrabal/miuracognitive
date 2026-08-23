# Protocolo confirmatorio AHA-2 (congelado tras el piloto, antes del confirmatorio)

## Config congelada (iteración final del piloto: tormentas letales)
- Entorno: `AHA2Config` con tormentas letales (storm_len=3, hazards 0.30-0.42:
  una tormenta sin defender mata; defendida a tope sobrevive), batería
  `action_budget=14` (mantener las 3 necesidades llenas costaría ~18 → la
  profilaxis indiscriminada está infrafinanciada), tope `level_cap=1.0`, coste
  metabólico 0.02, cues imperfectos (fiabilidad 0.75, adelanto U{4..9}, ruido
  ±30%, canal equivocado 15%, falsas alarmas 0.35). Sin cambios tras este commit.
- Entrenamiento: REINFORCE, 1000 pasos, batch 256, lr 1e-3 warmup+cosine,
  entropía 0.01.
- Baselines scripted afinados en seeds 100-102 (grids en tuned_scripted.json):
  combo@d2=0.078, threshold@d1=0.036 — los scripted NO resuelven este entorno;
  el aprendizaje sí (sanity: 0.74-0.78).

## Hallazgo de diseño del piloto (pre-registrado ANTES del confirmatorio)
Cinco iteraciones de entorno documentadas (ST-Gumbel colapsante → REINFORCE;
"actuar siempre" → tope+batería; profilaxis barata → tormentas letales). La
tensión final es IRREDUCIBLE en esta familia de entornos: para que un reactivo
pueda competir (R2) la tormenta debe extenderse en ticks, y entonces su INICIO
telegrafi­a el canal (señal interoceptiva 100% fiable) mejor que el cue externo
(75%, canal 85%). El piloto muestra cueblind ≈ learned: los agentes aprenden
anticipación desde la señal ENDÓGENA (onset + planificación de batería), no
desde el cue. El confirmatorio mide ESO; el null de cues queda pre-registrado
como resultado esperado, no como fracaso.

## Primarios pre-registrados (seeds 300-319, n=20, pareados)
- P1: supervivencia learned − combo afinado en delay=2. Sign test exacto + t pareada.
- P2: pendiente dosis-respuesta de (learned − combo) sobre delay∈{0,1,2,3} por seed.
- Corrección de Holm sobre {P1, P2}.
- Claim si PASS: "una política aprendida bajo economía de energía finita
  desarrolla regulación anticipatoria ENDÓGENA (onset interoceptivo +
  planificación de presupuesto) que los mejores controladores scripted
  afinados no alcanzan, con ventaja creciente en la latencia de acción".

## Secundarios (sin corrección; exploratorios)
- learned − learned_cueblind en delay=2 (valor del cue exógeno; esperado ≈0).
- Brazos HBP (hbp_wave/hbp_diff, alpha_const 1/0, ganancia aprendible) en delay=2.

## Guardrails
- G-untrained, G-ceiling (<0.98 en primarios), G-tuning (grid persistido),
  G-LOSO sobre P1.
