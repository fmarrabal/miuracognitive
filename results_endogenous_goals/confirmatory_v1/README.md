# Fase 2 — resultado confirmatorio

Estado: **PASS**.

Veinte seeds nuevas y pareadas (`100..119`) evaluaron un controlador con meta
persistente frente al mismo controlador reactivo. Ningún brazo recibe
`goal_id`, metadatos del episodio o progreso del proyecto.

| familia primaria | efecto medio | seeds a favor | p Holm |
|---|---:|---:|---:|
| memoria − reactivo en finalización leve | +0,51031 | 20/20 | 5,72e-6 |
| normal − lesión de recurrencia | +0,50698 | 20/20 | 5,72e-6 |
| normal − rotación de contenido | +0,49849 | 20/20 | 5,72e-6 |

La regla congelada exigía los tres efectos positivos y significativos tras
Holm, más cinco guardrails. Todo se cumple.

| media | `goal_memory` | `reactive` |
|---|---:|---:|
| compromiso leve | 1,000 | 0,48969 |
| finalización leve | 1,000 | 0,48969 |
| cambio ante crisis | 1,000 | 1,000 |
| rescate crítico | 1,000 | 1,000 |
| proyectos completados | 1,500 | 1,24484 |
| error homeostático absoluto | 0,14647 | 0,15430 |

Guardrails de `goal_memory`, todos verdaderos en cada seed:

- rescate crítico `>=0,95`;
- meta y acción siguen al cuerpo trasplantado `>=0,95`;
- selección inicial corporal `>=0,95`;
- estado leve aliasado con diferencia `<1e-6`.

La puerta de histéresis permanece casi cerrada ante el conflicto leve
(`0,02099`) y se abre ante crisis (`0,82452`). La lesión conserva la propuesta
reactiva y todo el cómputo, pero elimina la recurrencia de la meta; la
finalización cae aproximadamente a la mitad. Rotar sólo el contenido latente
produce una caída semejante.

Interpretación: se confirma selección y mantenimiento causal de **metas
endógenas dentro de un vocabulario fijo**. No se confirma descubrimiento de
metas no enumeradas, consciencia ni voluntad humana.

Archivos:

- `PROTOCOL.md`: hipótesis, familia y guardrails congelados;
- `summary.json`: agregado completo y tests exactos;
- `{variant}_seed{100..119}.json`: condiciones e intervenciones por seed;
- `checkpoints/`: 40 estados entrenados;
- `shard_*.log`: trazabilidad de ejecución.
