# AHA-1 — protocolo confirmatorio v1

Este protocolo se congela antes de ejecutar las semillas confirmatorias. El
piloto `pilot_v4` se utilizó únicamente para depurar la tarea y no se incorpora
al análisis confirmatorio.

## Hipótesis y familia primaria

Hipótesis: un ejecutivo con acceso a una predicción aprendida de necesidades
futuras muestra autorregulación anticipatoria causal frente a un ejecutivo
reactivo con el mismo presupuesto de cómputo.

Se usarán cuatro contrastes pareados, todos orientados de modo que positivo
favorece AHA:

1. `gating_wm − reactive` en supervivencia normal;
2. `gating_wm − reactive` en iniciativa correcta durante el intervalo vacío;
3. supervivencia de `gating_wm` normal menos su lesión predictiva;
4. supervivencia de `gating_wm` con cue válido menos cues barajados.

Cada contraste usa un test exacto bilateral de cambio de signo por seed. La
familia de cuatro se corrige con Holm a `alpha=0,05`. La fase sólo pasa si los
cuatro efectos medios son positivos y los cuatro valores ajustados son menores
que `0,05`. No se sustituirán métricas, seeds ni umbrales después de observar
los resultados.

## Diseño congelado

- Variantes primarias: `gating_wm,reactive`.
- Seeds nuevas: `100..119`; la seed 0 del piloto queda excluida.
- Predictor: 600 actualizaciones, compartido de forma bit-idéntica dentro de
  cada pareja y congelado durante el control.
- Controlador: 1.600 actualizaciones.
- Batch de entrenamiento: 64.
- Evaluación: 10 batches de 96 episodios por condición y seed.
- Entorno: horizonte 48, cinco perturbaciones, cue a seis ticks, efecto de
  acción a tres ticks.
- Dispositivo de ambos brazos: `cuda:0` en la misma máquina.
- Sin `goal_id`, tiempo de evento ni perturbación futura como entrada.

HBP queda fuera de la familia primaria. Sus comparaciones son secundarias y no
se recalibrarán con estas semillas.

## Controles y criterios descriptivos

Se informarán además predictor, mundo estacionario, tasa de acciones falsas y
trasplante de necesidad. No pueden rescatar una familia primaria fallida. Una
conclusión positiva significa «autorregulación anticipatoria causal en este
benchmark», no metas abiertas, consciencia ni equivalencia con biología humana.

## Comando

```powershell
python -u aha_benchmark.py `
  --variants gating_wm,reactive `
  --seeds 100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119 `
  --predictor-steps 600 --steps 1600 --batch-size 64 `
  --eval-batches 10 --eval-batch-size 96 --device cuda:0 `
  --out-dir results_aha/confirmatory_v1
```

## Registro de ejecución

Ejecutado el 15 de julio de 2026. Para aprovechar la GPU, las seeds se
distribuyeron entre cuatro procesos con RNG reiniciado por seed. No se cambió
ningún dato, modelo, hiperparámetro ni criterio. La seed 100 había terminado en
el proceso serial inicial y fue reutilizada bit a bit; el intento parcial de la
seed 101 se descartó antes de guardar checkpoint y se ejecutó de nuevo desde
cero. Los cuatro shards terminaron sin stderr. El `summary.json` final se
reconstruyó en modo `--resume-existing` sobre las 40 celdas guardadas.
