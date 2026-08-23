# Fase 2 — protocolo confirmatorio de metas endógenas

Este protocolo se congela después de calibrar la tarea con seeds `0..2` y antes
de ejecutar las seeds confirmatorias `100..119`. Los pilotos no se incorporan
al análisis.

## Hipótesis operacional

Sin recibir `goal_id`, un agente puede seleccionar desde su cuerpo una
prioridad, conservarla cuando el estado presente ya no contiene la historia del
proyecto y abandonarla si una crisis hace más urgente otra necesidad.

El entorno exige tres acciones de trabajo separadas por gaps obligatorios. Tras
el primer paso, el conflicto leve deja exactamente iguales los niveles del
objetivo inicial y su rival; el progreso físico está oculto. Por tanto, una
política reactiva al estado presente no puede reconstruir qué proyecto empezó.

## Familia primaria

Tres diferencias pareadas, orientadas a favor de la hipótesis:

1. `goal_memory − reactive` en finalización bajo conflicto leve;
2. condición normal menos lesión de la recurrencia de meta;
3. condición normal menos rotación del contenido de la meta.

Cada diferencia usa test exacto bilateral de cambio de signo por seed. Las tres
se corrigen con Holm a `alpha=0,05`. Todas deben tener media positiva y
`p_Holm<0,05`.

Guardrails obligatorios, evaluados por seed en `goal_memory`:

- rescate crítico `>=0,95` — descarta perseveración ciega;
- selección de meta tras trasplante corporal `>=0,95`;
- acción correcta tras trasplante corporal `>=0,95`;
- selección inicial desde el cuerpo `>=0,95`;
- gap máximo del estado aliasado `<1e-6`.

## Diseño congelado

- Variantes: `goal_memory,reactive`; mismas redes y presupuesto.
- Seeds confirmatorias nuevas: `100..119`.
- Entrenamiento: 800 actualizaciones, batch 128, AdamW `lr=2e-3`.
- Evaluación: 10 batches de 96 episodios por condición y seed.
- Tres necesidades, horizonte 12, proyecto de tres pasos, seis oportunidades.
- Puerta con histéresis homeostática: margen de urgencia `0,05`, ganancia `40`.
- Dispositivo: `cuda:0` para ambos brazos.
- Ninguna entrada contiene target inicial, rival, flag de crisis o progreso.

Una conclusión positiva significa selección y mantenimiento causal de metas
endógenas dentro de este vocabulario fijo. No significa descubrir metas nuevas,
consciencia ni equivalencia con voluntad humana.

## Comando lógico

```powershell
python -u endogenous_goals_benchmark.py `
  --variants goal_memory,reactive `
  --seeds 100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119 `
  --steps 800 --batch-size 128 `
  --eval-batches 10 --eval-batch-size 96 --device cuda:0 `
  --out-dir results_endogenous_goals/confirmatory_v1
```

## Registro de ejecución

Ejecutado el 15 de julio de 2026 en cuatro shards sobre la misma GPU. Cada seed
reinicia su RNG y cada shard contiene ambos brazos emparejados. No se cambiaron
datos, arquitectura, hiperparámetros, métricas ni umbrales tras congelar este
documento. Las 40 celdas y 40 checkpoints terminaron sin stderr. El agregado
final se reconstruyó en modo `--resume-existing` sin reentrenamiento.
