# Experimento causal del cuello de botella de cómputo

Estado: **infraestructura terminada y validada; hipótesis no confirmada con el
entrenamiento actual**.

## Qué se interviene

La unidad presupuestaria es un paso de reasoner aplicado a una muestra. El
backbone es fijo e idéntico en todos los brazos. Para cada batch:

1. Un rollout PonderNet sin etiquetas produce `p(parar en n)` por muestra.
2. Programación dinámica asigna cuotas enteras maximizando la probabilidad
   conjunta bajo una suma de pasos exacta.
3. El modelo se reejecuta con halting duro y sub-batches activos; una muestra
   que agota su cuota deja de pasar por reasoner, WM y HBP.
4. Se comparan `learned`, `uniform`, cuatro permutaciones de las mismas cuotas,
   `oracle` por dificultad y `anti_oracle`.

`learned` frente a `shuffle_0` conserva incluso el multiconjunto de cuotas;
`learned` frente a `uniform` conserva el cómputo total y es el baseline práctico.
El test principal es una randomización exacta de signo por batch, evitando tratar
como independientes muestras cuya asignación está acoplada por el presupuesto.

El rollout de planificación es común a todos los brazos y queda fuera del
presupuesto. El experimento identifica el valor causal contrafactual de la
asignación, pero no demuestra todavía ahorro end-to-end de FLOPs o latencia.

## Resultado reproducible con el código actual

Protocolo bloqueado: `hbp_full`, generadores adyacentes, 1.500 pasos de training,
`K_train≤12`, máximo 12 ticks; evaluación en 1.280 ejemplos por checkpoint,
presupuesto medio `B=10`, contraste aprendido–uniforme.

| seed de training | acc aprendida | acc uniforme | Δ accuracy | p exacta por batch | corr(K, cuota) |
|---:|---:|---:|---:|---:|---:|
| 0, reentrenada | 0.3539 | 0.3617 | −0.0078 | 0.157 | −0.165 |
| 1 | 0.3766 | 0.3711 | +0.0055 | 0.274 | −0.204 |
| 2 | 0.3680 | 0.3695 | −0.0016 | 0.857 | −0.298 |

Media entre semillas: `Δacc = −0.0013`; t pareada sobre semillas `p=0.767`.
El controlador actual no aprende de forma reproducible una asignación útil y,
de hecho, ordena la dificultad en la dirección contraria en las tres corridas.

## Por qué apareció inicialmente un positivo

El checkpoint histórico `checkpoints/hbp_full_adjacent.pt` sí produce:

- `Δacc=+0.0625` frente a uniforme en 1.280 ejemplos nuevos;
- `p_cluster=3.20e-10`;
- `corr(K,cuota)=+0.603`.

Pero el efecto no reaparece al reentrenar la misma seed 0 con el código actual,
ni en seeds 1–2. Por tanto pertenece a ese checkpoint histórico —entrenado antes
de los últimos fixes— y no puede presentarse como propiedad general del método.

## Prueba de pérdida PonderNet esperada

Se añadió una opción correcta y retrocompatible:

`E_{p(n|x)}[CE(logits_n,y)]`

en vez de `CE` sobre la mezcla de estados. En una corrida exploratoria seed 0
produce ventajas frente a uniforme en B=4/6/8/10. Sin embargo, el diagnóstico
revela `E[n]=1.009`: el halting colapsa al primer paso. El asignador satisface el
presupuesto dejando muchas muestras en paso 1 y enviando una minoría al techo 12.
La ganancia aparece porque iterar de más perjudica ejemplos resolubles, mientras
el tramo largo sigue prácticamente en chance.

Esto demuestra que la intervención detecta correctamente cuándo **la asignación
importa**, pero no el mecanismo buscado de “dar más pensamiento útil a problemas
difíciles”. No debe usarse como resultado positivo del paper.

## Conclusión de esta fase

La infraestructura causal ya responde la pregunta con precisión:

> En la tarea y entrenamiento actuales, la política de cómputo no mejora la
> accuracy de manera reproducible bajo un presupuesto igualado.

El siguiente experimento no debe seguir acumulando semillas de `permcomp`.
Necesita una tarea en la que:

1. cada tick adicional revele o ejecute una transición secuencial real;
2. el cómputo hasta cierta profundidad mejore —no destruya— la predicción;
3. los batches mezclen demandas distintas bajo una suma de ticks fija;
4. uniforme desperdicie ticks en ejemplos terminados y deje otros incompletos;
5. la política se entrene con pérdida por profundidad alineada al presupuesto.

Solo después se compararán `gating_wm`, `hbp_first` y `hbp_full`, y se medirá si
la homeostasis de segundo orden aprende esa asignación mejor que los controles.

## Continuación realizada

Ese siguiente experimento ya está implementado en `budgeted_stream_benchmark.py`
y documentado en `results_budgeted_stream/README.md`. En una cadena donde cada
tick revela y ejecuta una transición necesaria, la política aprendida mejora la
accuracy en `+0,2574` frente al reparto uniforme bajo el mismo presupuesto
exacto (`p_cluster=1,91×10⁻⁶` en cada una de tres semillas). La asignación
recupera `q=K` sin error.

El positivo confirma el valor causal de **asignar bien cómputo útil**, pero las
tres arquitecturas empatan exactamente: todavía no hay evidencia de que el HBP
sea mejor que el control para decidir esa asignación.

## Archivos principales

- `pilot_hbp_full_adjacent_seed0.json`: curva inicial del checkpoint histórico.
- `confirm_hbp_full_adjacent_seed0_B10_eval20260715.json`: réplica de datos del
  checkpoint histórico (positiva, no replicada por modelo).
- `confirm_hbp_full_adjacent_seed0_current_B10_eval20260715.json`: misma seed y
  mismos ejemplos, checkpoint reentrenado con código actual (nulo).
- `confirm_hbp_full_adjacent_seed1_B10_eval20260716.json`: seed 1 (nulo).
- `confirm_hbp_full_adjacent_seed2_B10_eval20260717.json`: seed 2 (nulo).
- `exploratory_hbp_full_adjacent_seed0_ponder_expected.json`: objetivo esperado;
  positivo aparente con colapso a un tick.
- `smoke.json`: prueba mínima de integración; no usar como evidencia.

Los JSON contienen probabilidades de halting, cuotas, resultados binarios y
presupuesto ejecutado para cada batch, además de estadísticas agregadas.

## Reproducción de una celda

```powershell
$env:PYTHONPATH='.'
python compute_bottleneck.py `
  --checkpoint checkpoints/hbp_full_adjacent_seed1_bottleneck.pt `
  --n-batches 40 --batch-size 32 --n-shuffles 4 `
  --budgets 10 --primary-budget 10 --primary-control uniform `
  --seed 20260716 `
  --out results_compute_bottleneck/reproduction.json
```
