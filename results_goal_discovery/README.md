# Fase 3 — descubrimiento de metas continuas

Esta fase pregunta si una prioridad endógena puede convertirse en una meta
factible cuyo contenido no figure en un catálogo del diseñador.

Cada episodio contiene una región factible bidimensional nueva y tres paisajes
de affordances con centros continuos ocultos. El agente observa su estado
homeostático, elige cuatro coordenadas de sondeo, recibe sus consecuencias y
propone una coordenada final. No recibe `goal_id`, clase de objetivo, centro ni
supervisión sobre el centro.

El mecanismo final usa identificación de sistema: convierte las respuestas RBF
en distancias y trilatera el centro que restaura la necesidad más urgente. Una
corrección neuronal pequeña y diferenciable puede adaptar sondeos y objetivo.
El control `no_feedback` tiene los mismos módulos, pasos y presupuesto, pero la
consecuencia de cada sondeo se anula antes de entrar al estado recurrente.

## Desarrollo

- `pilot_v1`: la red recurrente pura usa causalmente el feedback, pero sólo
  alcanza 0.244 de éxito frente a 0.142 del control.
- `pilot_v2`: ponderar la pérdida por urgencia observable mejora el contraste,
  pero no la precisión (0.251 frente a 0.138).
- smoke geométrico: muestra que el cuello era la decodificación. La
  trilateración logra 1.000 frente a 0.094; estos datos no pertenecen al
  confirmatorio.

No se rebajó el umbral de éxito tras los pilotos. El confirmatorio exige al
menos 0.70 en cada semilla y una familia causal de cuatro contrastes.

## Resultado confirmatorio

**PASS** en 20 semillas nuevas (`200..219`). El descubridor alcanza `1,00000`
de éxito frente a `0,12918` del control. Arquitectura, lesión, barajado de
consecuencias y reflexión de contenido producen efectos de `+0,87082`,
`+0,86996`, `+0,90688` y `+0,96754`; los cuatro pasan Holm con
`p=7,63e-6`. Pasan además los seis guardrails y la auditoría de sensibilidad a
un error de ±15% en σ.

Informe y datos: `confirmatory_v1/README.md`.

## Alcance de una señal positiva

Una señal positiva respalda descubrimiento causal de coordenadas-objetivo
continuas y composicionales dentro de esta gramática de affordances. No prueba
metas abiertas, semántica humana, consciencia ni «alma».

Ejecutar:

```powershell
python goal_discovery_benchmark.py --seeds 200,201,202 --steps 400 `
  --out-dir results_goal_discovery/confirmatory_v1
```

El protocolo congelado se encuentra en `confirmatory_v1/PROTOCOL.md`.
