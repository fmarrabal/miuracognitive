# Cuello de botella causal con cómputo útil

Estado: **positivo causal confirmado en 3 semillas y 3 arquitecturas**.

Este benchmark se diseñó después del resultado nulo de `permcomp`. Allí, iterar
más no mejoraba de forma monótona y una asignación de cómputo podía parecer útil
por evitar *overthinking*. Aquí cada tick ejecuta una transición nueva que antes
de ese tick no está disponible para el modelo. Por construcción, el cómputo es
útil hasta terminar el trabajo y deja de serlo después.

## Pregunta causal

> Con el mismo número total de ticks, ¿asignarlos a las muestras que todavía
> tienen trabajo pendiente causa más accuracy que repartirlos uniformemente?

Sí. La política aprendida gana **+25,74 puntos porcentuales de accuracy de media**
frente al reparto uniforme, con el mismo presupuesto exacto. El efecto se
reproduce en las tres semillas. HBP de primer orden, HBP de segundo orden y el
control sin HBP obtienen exactamente el mismo resultado cuando comparten
ejecutor e inicialización.

## Diseño

- Cada muestra contiene un estado inicial y una cadena de `K=1..9` operaciones
  biyectivas y no conmutativas sobre 12 estados.
- En el tick `t` sólo se revela y ejecuta la operación `t`. Antes de `K` faltan
  operaciones necesarias para conocer el resultado; después de `K` el estado se
  congela, aunque los ticks sobrantes siguen cargándose al presupuesto.
- Cada batch tiene 36 muestras balanceadas, cuatro de cada longitud. Por tanto,
  `sum(K)=36×5=180` ticks exactamente.
- El ejecutor se preentrena 2.000 actualizaciones con supervisión del estado
  corriente y después se congela. El scheduler se entrena 1.000 actualizaciones
  para detenerse al completar `K`.
- Evaluación por checkpoint: 20 batches, 720 muestras y 3.600 ticks por política.
- Contraste principal: asignación aprendida frente a cuota uniforme `q=5`.
  También se evalúan oráculo `q=K`, anti-oráculo y cuatro barajados con el mismo
  multiconjunto de cuotas.
- Test principal: randomización exacta de signo sobre los 20 batches, bilateral.

La evaluación dura vuelve a ejecutar únicamente el sub-batch activo. Si una
muestra agota su cuota, deja de pasar por el executor. El evaluador aborta si una
política consume una suma de ticks diferente de la prescrita.

## Resultado confirmatorio

Las tres arquitecturas producen esta misma tabla, exactamente:

| seed | accuracy aprendida | accuracy uniforme | diferencia | p exacta por batch |
|---:|---:|---:|---:|---:|
| 0 | 0,8722 | 0,6222 | +0,2500 | 1,91×10⁻⁶ |
| 1 | 0,8792 | 0,6111 | +0,2681 | 1,91×10⁻⁶ |
| 2 | 0,8528 | 0,5986 | +0,2542 | 1,91×10⁻⁶ |
| **media** | **0,8681** | **0,6106** | **+0,2574** | — |

Desviación estándar de la diferencia entre semillas: `0,00945`.

Auditoría de asignación:

- correlación `corr(K,q)=1`;
- error absoluto medio `|q-K|=0`;
- coincidencia exacta `q=K`: 100% de 6.480 decisiones (720 × 9 corridas);
- política aprendida = oráculo exacto en todas las muestras;
- 3.600 unidades de executor para **cada** política y checkpoint;
- accuracy media de cuotas barajadas: 0,5795; anti-oráculo: 0,6324.

Después de detectar un confusor de inicialización, el constructor se corrigió y
se repitieron las seis corridas HBP. Los checkpoints finales comparten embeddings,
GRU, normalización y readout idénticos bit a bit con el control de su misma seed.
El test de regresión correspondiente está en `tests/test_budgeted_stream.py`.

## Qué demuestra y qué no

Demuestra un resultado positivo preciso:

> Cuando los ticks adicionales realizan trabajo secuencial necesario, aprender
> a distribuir un presupuesto fijo causa una mejora grande y reproducible de
> accuracy frente a distribuirlo uniformemente.

No demuestra todavía que el HBP sea la causa. Las tres variantes empatan porque
la señal suficiente es explícita: el scheduler observa `K` y recibe supervisión
de finalización `-log p(n=K)`. Tampoco demuestra iniciativa autónoma, generación
de metas, consciencia, “alma” ni ahorro de cómputo end-to-end: el rollout de
planificación y la integración HBP no se incluyen en la unidad presupuestaria.

La siguiente prueba científicamente informativa debe ocultar `K`, retirar la
supervisión directa de parada y obligar al controlador a inferir internamente si
queda trabajo a partir de progreso, incertidumbre y coste. Sólo en ese escenario
puede distinguirse una regulación homeostática útil de una regla explícita
`q=K`.

## Continuación realizada

Ese experimento ya está implementado y documentado en
`results_hidden_need/README.md`. Sin acceso a `K`, sin `completion loss` y con
asignación online basada sólo en el prefijo ejecutado, la mejora media frente a
uniforme es de aproximadamente `+0,177` en las tres arquitecturas. El efecto se
replica en 3/3 semillas, pero HBP vuelve a empatar con el control.

## Reproducción

Desde la raíz de `miuracognitive`:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
& 'C:\Users\fmarr\miniconda3\envs\implanto\python.exe' `
  -u budgeted_stream_benchmark.py `
  --variants gating_wm,hbp_first,hbp_full --seeds 0,1,2 `
  --pretrain-steps 2000 --steps 1000 --policy-objective completion `
  --eval-batches 20 --out-dir results_budgeted_stream
```

Archivos principales:

- `summary.json`: agregado confirmatorio.
- `{variant}_seed{0,1,2}.json`: cuotas, predicciones, tests y trazas por muestra.
- `checkpoints/`: pesos y configuración de las nueve corridas.
- `../budgeted_stream_benchmark.py`: entrenamiento y CLI.
- `../data/budgeted_stream.py`: generador causal.
- `../model/budgeted_stream.py`: executor y schedulers.
- `../eval/budgeted_stream.py`: intervención dura y controles yoked.
