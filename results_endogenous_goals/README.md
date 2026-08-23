# Fase 2 — metas endógenas sostenidas

Pregunta causal: ¿puede un agente sin `goal_id` elegir una prioridad desde su
estado corporal, mantenerla durante un proyecto y cambiarla sólo cuando una
crisis lo justifica?

## Banco de compromiso

- Tres necesidades y un vocabulario fijo de proyectos restauradores.
- Cada proyecto necesita tres acciones consistentes.
- Las acciones sólo están disponibles en ticks alternos; los gaps fuerzan
  `no-op` y borran la última acción observable.
- El progreso físico existe en el mundo, pero no entra en la política.
- Un distractor leve deja dos necesidades exactamente empatadas después del
  primer paso. El estado actual no identifica qué proyecto tiene progreso.
- Una crisis vuelve inequívocamente más urgente la necesidad rival y exige
  cambiar.

`goal_memory` mantiene una distribución latente de prioridad. Una puerta de
histéresis compara la urgencia de la meta vigente con la mejor urgencia actual.
`reactive` ejecuta las mismas redes, pero sustituye la meta por la propuesta
instantánea cada tick.

## Piloto calibrado (`pilot_v4`, seed 0)

| métrica | `goal_memory` | `reactive` |
|---|---:|---:|
| selección inicial corporal | 1,000 | 1,000 |
| compromiso leve | **1,000** | 0,450 |
| finalización leve | **1,000** | 0,450 |
| cambio ante crisis | 1,000 | 1,000 |
| rescate crítico | 1,000 | 1,000 |

Intervenciones en `goal_memory`:

- lesionar la recurrencia reduce finalización leve en `0,533`;
- rotar el contenido de la meta la reduce en `0,506`;
- trasplantar qué necesidad tiene déficit cambia meta y acción en `100%`;
- puerta media en conflicto leve `0,020`; en crisis `0,821`.

Seeds piloto 1 y 2 replicaron finalización `1,000`, rescate `1,000` y efectos de
lesión `+0,496` y `+0,492`. Estos datos son de calibración, no confirmatorios.

El protocolo congelado para 20 seeds nuevas está en
[`confirmatory_v1/PROTOCOL.md`](confirmatory_v1/PROTOCOL.md).

## Confirmación (`confirmatory_v1`, 20 seeds nuevas)

Estado: **PASS**. Los tres criterios primarios fueron positivos en 20/20 seeds:

| criterio | efecto medio | p exacta | p Holm |
|---|---:|---:|---:|
| memoria − reactivo en finalización leve | +0,51031 | 1,91e-6 | 5,72e-6 |
| estado normal − lesión de memoria | +0,50698 | 1,91e-6 | 5,72e-6 |
| meta normal − contenido rotado | +0,49849 | 1,91e-6 | 5,72e-6 |

Medias conductuales:

| métrica | `goal_memory` | `reactive` |
|---|---:|---:|
| selección inicial desde el cuerpo | 1,000 | 1,000 |
| compromiso leve | **1,000** | 0,4897 |
| finalización leve | **1,000** | 0,4897 |
| cambio ante crisis | 1,000 | 1,000 |
| rescate crítico | 1,000 | 1,000 |
| proyectos completados | 1,500 | 1,2448 |
| error homeostático absoluto | **0,1465** | 0,1543 |

Los cinco guardrails pasaron en todas las seeds: rescate crítico, selección de
meta y acción tras trasplante corporal, selección inicial y aliasado exacto del
estado (`gap máximo=5,96e-8`). La puerta abre `0,021` ante el empate leve y
`0,825` ante crisis. Por tanto la memoria no es simple perseveración.

Conclusión permitida: el agente forma y mantiene una prioridad endógena causal
dentro de un conjunto fijo de necesidades/proyectos. La conclusión no se basa
en supervivencia —ambos brazos obtienen 1,0— sino en completar el proyecto bajo
un estado que ha perdido informacionalmente su historia.

Resultados completos: [`confirmatory_v1/`](confirmatory_v1/).

No se afirma descubrimiento abierto de metas. El vocabulario de necesidades y
proyectos sigue fijado por el entorno; ésa será la pregunta separada de fase 3.
