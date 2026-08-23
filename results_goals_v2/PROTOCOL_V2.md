# Protocolo confirmatorio Fase 2 v2 (congelado tras el piloto)

- Entorno: Goals2Config (3 proyectos, decay 0.015, 2 crisis con precursor de 2 ticks,
  3 distractores con pico ALIASADO idéntico, ventana 4, bono/castigo 0.8).
- Entrenamiento: REINFORCE 400 pasos batch 256 lr 1e-3, entropía 0.01. Sin ninguna
  supervisión hacia reglas (R4).
- Scripted afinados (seeds 100-102): greedy(w), hysteresis(w,m), smart(w,m)=histéresis
  + detector del precursor (conoce la regla generativa; listón de diseñador).
- Primarios (seeds 300-319, pareados, Holm sobre {P1,P2}):
  P1 = retorno learned − mejor scripted (in-dist).
  P2 = ídem en OOD (distractores 3→6, crisis 2→3; misma regla de desambiguación).
- Secundario: learned − learned_memoryless (ablación de recurrencia).
- Nota de mecanismo (piloto, honestidad): el learned atiende crisis Y distractores
  ≈100% — el mecanismo es PRIORIZACIÓN DE VALOR aprendida, no discriminación
  crisis/distractor. Cualquier claim se enmarcará así.
