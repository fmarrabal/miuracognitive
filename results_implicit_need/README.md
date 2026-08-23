# Terminación implícita sin `K` ni `LAST`

Estado: **asignación causal confirmada en n=10; primer positivo nominal de
`hbp_full` frente al control, todavía frágil y no atribuible al segundo orden**.

Este benchmark elimina el marcador terminal del experimento anterior. El
controller no conoce `K`, no recibe una etiqueta de finalización y tampoco ve
un token `LAST`. En el verdadero tick final no existe ninguna señal que lo
distinga de otro prefijo. La terminación sólo se vuelve observable después de
financiar un tick adicional y detectar que el estado no progresa.

## Protocolo

- Cadenas de `K=1..9` transiciones no conmutativas sobre 12 estados.
- `K` uniforme y oculto; ningún token terminal en entrenamiento o evaluación.
- Scheduler basado en estado del executor, cambio, entropía y esfuerzo.
- `observe_length=false`, `completion_weight=0`.
- Executor: 2.000 actualizaciones auxiliares; después queda congelado.
- Controller: 2.000 actualizaciones con pérdida de tarea esperada y restricción
  de presupuesto medio 5.
- Asignación online unidad a unidad, consultando sólo el prefijo financiado.
- 20 batches × 36 muestras = 720 ejemplos y 3.600 ticks exactos por brazo.
- 10 semillas y tres arquitecturas: 30 celdas.

Como no hay señal en el propio tick `K`, recuperar `q=K` no es el objetivo ni es
informacionalmente posible para todas las muestras bajo `sum(q)=sum(K)`. La
política aprende a sobrepasar algunos finales, detectar progreso cero y
reasignar el presupuesto restante.

Para ahorrar cómputo sin cambiar el contraste, cada variante HBP carga el
executor congelado de `gating_wm` de la misma seed y avanza el generador hasta el
mismo batch de política. Los checkpoints finales comparten embeddings, GRU,
normalización y readout idénticos bit a bit.

## Resultado principal: asignación frente a uniforme

| arquitectura | acc. online | acc. uniforme | diferencia | SD de Δ | corr(K,q) |
|---|---:|---:|---:|---:|---:|
| `gating_wm` | 0,7275 | 0,6114 | **+0,1161** | 0,0337 | 0,708 |
| `hbp_first` | 0,7386 | 0,6114 | **+0,1272** | 0,0116 | 0,743 |
| `hbp_full` | 0,7417 | 0,6114 | **+0,1303** | 0,0114 | 0,737 |

Las 30 celdas son positivas frente a uniforme con test exacto por batch
`p<0,05`. El control es positivo en 10/10 seeds; seed 9 es un outlier de
entrenamiento (`Δ=+0,0236`) y las otras nueve quedan entre `+0,1139` y `+0,1431`.

Controles adicionales:

- accuracy oráculo `q=K`: 0,8528;
- cuotas barajadas: ≈0,5825;
- error `|q−K|`: 1,41 control; 1,30 primer orden; 1,31 segundo orden;
- `q=K` exacto: sólo ≈11%, como corresponde al régimen sin señal terminal;
- todas las políticas consumen exactamente 3.600 ticks por checkpoint.

## Contraste HBP–control

| contraste | diferencia media | aciertos por seed | p exacta por seed |
|---|---:|---|---:|
| `hbp_first − gating_wm` | +0,0111 | +6,+3,−2,0,−3,0,+5,+8,+1,+62 | 0,0859 |
| `hbp_full − gating_wm` | **+0,0142** | +10,+11,+4,+5,−5,+3,−2,+9,+5,+62 | **0,0254** |
| `hbp_full − hbp_first` | +0,0031 | +4,+8,+6,+5,−2,+3,−7,+1,+4,0 | 0,1641 |

`hbp_full` supera nominalmente al control en 8/10 seeds. Es el primer contraste
directo favorable al HBP completo bajo asignación causal y demanda no observable
por anticipado.

### Robustez de la afirmación

La conclusión debe seguir siendo prudente:

- seed 9 aporta +62 de los +102 aciertos netos de `hbp_full`;
- excluyéndola, el efecto baja a `+0,00617` y `p_exacta=0,0508`;
- el contraste t pareado no es significativo (`p≈0,122`; IC95% de la media
  aproximadamente `[-0,0046,+0,0329]`);
- corrigiendo las tres comparaciones arquitectónicas como una familia, el
  `p=0,0254` tampoco supera Holm (`≈0,076`);
- `hbp_full` no supera significativamente a `hbp_first`, por lo que no se puede
  atribuir el efecto a la inercia de segundo orden.

Veredicto preciso:

> Hay evidencia nominal de que el sistema HBP completo mejora robustez y
> asignación frente al control sin HBP, pero todavía no evidencia confirmatoria
> de que la dinámica de segundo orden sea la causa.

## Relación con la hipótesis de proactividad

Éste es el escenario más cercano logrado hasta ahora a regulación interna:

- la demanda no está dada de antemano;
- la política debe gastar recursos para descubrir si un proceso terminó;
- una experiencia dinámica interna cambia cómo redistribuye cómputo limitado;
- el efecto de asignar mejor causa más accuracy.

Sigue sin demostrar generación autónoma de metas, consciencia o “alma”. La tarea,
la recompensa y el presupuesto vienen del exterior, y la política continúa
reaccionando a estabilidad/progreso cero. El resultado respalda un mecanismo de
autorregulación computacional, no una equivalencia con la mente humana.

## Reproducción

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
& 'C:\Users\fmarr\miniconda3\envs\implanto\python.exe' `
  -u hidden_need_benchmark.py --no-terminal-marker `
  --reuse-paired-executor `
  --variants gating_wm,hbp_first,hbp_full `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --pretrain-steps 2000 --steps 2000 --eval-batches 20 `
  --out-dir results_implicit_need
```

`--resume-existing` permite continuar una matriz interrumpida sin reentrenar
celdas completas.

Archivos principales:

- `summary.json`: medias, deltas y contrastes arquitectónicos pareados.
- `{variant}_seed{0..9}.json`: cuotas, decisiones online y resultados por muestra.
- `checkpoints/`: 30 checkpoints.
- `../hidden_need_benchmark.py`: entrenamiento y agregación.
- `../eval/hidden_need.py`: asignación online sin acceso al futuro.
- `../tests/test_budgeted_stream.py`: tests de no fuga e imposibilidad causal.

