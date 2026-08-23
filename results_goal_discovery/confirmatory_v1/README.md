# Fase 3 — resultado confirmatorio

Estado: **PASS**.

Veinte semillas nuevas y pareadas (`200..219`) evaluaron el descubridor
continuo frente al control `no_feedback`. Ambos ejecutan cuatro sondeos, la
misma dinámica, módulos y presupuesto. Ninguno recibe `goal_id`, un catálogo,
el centro de la affordance ni supervisión de ese centro.

| familia primaria | efecto medio | semillas a favor | p Holm |
|---|---:|---:|---:|
| descubridor − control en éxito | +0,87082 | 20/20 | 7,63e-6 |
| normal − lesión de feedback | +0,86996 | 20/20 | 7,63e-6 |
| normal − consecuencias barajadas | +0,90688 | 20/20 | 7,63e-6 |
| meta normal − contenido reflejado | +0,96754 | 20/20 | 7,63e-6 |

La regla congelada exigía los cuatro efectos positivos y significativos tras
Holm, más seis guardrails. Todo se cumple.

| media | `discoverer` | `no_feedback` |
|---|---:|---:|
| éxito de meta | 1,00000 | 0,12918 |
| respuesta de la necesidad dominante | 0,98332 | 0,67295 |
| distancia al centro objetivo | 0,11344 | 0,54000 |
| restauración dominante | 1,00000 | 0,27355 |
| metas factibles | 1,00000 | 1,00000 |
| metas continuas únicas | 1,00000 | 0,98406 |

El trasplante corporal conserva el mismo mundo pero cambia la necesidad
dominante. El descubridor se orienta hacia el centro nuevo en `0,98199` de los
episodios y tiene éxito sobre él en `1,00000`; el control queda en `0,49961` y
`0,12957`, respectivamente.

## Sensibilidad post-confirmatoria

Sin reentrenar, se introdujo un error deliberado en el ancho RBF que supone el
decodificador:

| escala de σ interno | éxito medio | mínimo entre semillas |
|---:|---:|---:|
| 0,85 | 0,84617 | 0,77656 |
| 1,00 | 1,00000 | 1,00000 |
| 1,15 | 0,94102 | 0,87188 |

La auditoría pasa el umbral de `0,70` en todas las semillas y condiciones.

## Interpretación limitada

Se confirma descubrimiento causal de una coordenada-objetivo continua no
enumerada **dentro de la gramática RBF ensayada**. El resultado es fuerte sobre
uso de consecuencias, contenido de meta y seguimiento corporal, pero la ley de
distancias es una inductive bias del agente. No demuestra propósitos abiertos,
transferencia a otras leyes de affordances, consciencia ni «alma».

Archivos:

- `PROTOCOL.md`: hipótesis, familia y umbrales congelados;
- `summary.json`: agregado completo y pruebas exactas;
- `sensitivity_audit.json`: robustez a error de σ;
- `{variant}_seed{200..219}.json`: condiciones por semilla;
- `checkpoints/`: 40 estados;
- `shard_*.log`: trazabilidad de ejecución.
