# Protocolo confirmatorio Fase 3 v2 (congelado tras el piloto)

- Entorno: Discovery2Config (familias {rbf,bimodal,ridge,plateau} con parámetros
  aleatorios POR EPISODIO, σ desconocida, ruido de observación 0.05, presupuesto
  12 sondas). El agente NO recibe la forma funcional (R5).
- Agente: LearnedProber GRU in-context, meta-entrenado 800 pasos por BPTT a través
  del campo (el agente en eval solo ve pares (x,y)).
- Scripted afinados: grid(jitter), fd(k_init,step), random.
- Primario (seeds 300-319): P1 = regret mejor_scripted − learned (pareado, sign test).
- Secundarios: regret por familia de campo.
- Guardrails: G-untrained (0.63 sin entrenar vs ~0.14 entrenado en piloto), sin techo
  (regret>0 estructural por ruido+presupuesto).
