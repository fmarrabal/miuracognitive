# Protocolo confirmatorio Fase 4 v2 (congelado tras el piloto)

- Entorno: WavePlant con matriz de actuadores B ALEATORIA POR EPISODIO (nada que
  memorizar), daño parcial sin flag a mitad de episodio (severidad = efectividad
  residual), ruido de observación. El agente no recibe B ni las constantes (R5).
- Automodelo: regresión lineal genérica online (NLMS), INIT ALEATORIA; controlador
  por muestreo sobre el modelo APRENDIDO (idéntico en todos los brazos).
- Brazos: adaptive (aprende siempre), frozen (identifica y congela), reinit (detector
  afinado thr=0.01), oracle (skyline con verdad), random (suelo).
- Primarios (seeds 300-319, severidades {0.2,0.4,0.6}, Holm):
  P1 = coste post-daño frozen − adaptive por severidad.
  P2 = reinit − adaptive (NOTA del piloto: el detector afinado DEGENERA en adaptive
  → se espera P2≈0; se reporta como hallazgo, no como fracaso).
- Métrica adicional: regret vs oracle (graduado; sin techo).
