# Asignación endógena con longitud oculta

Estado: **positivo causal confirmado; ventaja específica HBP no confirmada**.

Este experimento elimina las dos ayudas principales del benchmark anterior:

1. el scheduler ya no observa la longitud `K`;
2. no se entrena con `-log p(n=K)` ni con otra etiqueta de finalización.

La política aprende únicamente de la pérdida de tarea esperada y de una
restricción de presupuesto medio. En evaluación, las unidades se asignan online
consultando sólo el score del prefijo actualmente ejecutado de cada muestra.

## Diseño causal

- Cada muestra contiene `K=1..9` transiciones no conmutativas sobre 12 estados.
- Una operación sólo se revela al ejecutar su tick.
- La operación `K` lleva una variante `LAST`; por tanto el final se vuelve
  observable únicamente al realizar la última transición, nunca por anticipado.
- El scheduler recibe estado normalizado del executor, cambio de estado,
  entropía y esfuerzo consumido. Con `observe_length=false`, ni su cabeza ni el
  HBP reciben `K`.
- El executor se preentrena 2.000 actualizaciones y se congela. El controller se
  entrena otras 2.000 con:

  `L_policy = E_p[CE_tarea(t)] + 2·(media(E[n])−5)²`

- No se usa `completion loss`; sólo se conserva como diagnóstico no optimizado.
- Cada batch balanceado tiene 36 muestras y presupuesto exacto de 180 ticks.
- Evaluación por checkpoint: 20 batches, 720 muestras y 3.600 ticks por brazo.

### Asignación online

Todas las muestras reciben un tick. Para cada unidad restante, el asignador lee
exclusivamente `halt_logit[i,q_i−1]`, el score producido después de los `q_i`
ticks ya ejecutados, elige la mayor necesidad y aumenta esa cuota en uno. Sólo
entonces puede consultar el siguiente prefijo de esa muestra. Las columnas
futuras nunca intervienen antes de ser financiadas.

Se compara con cuota uniforme `q=5`, oráculo `q=K`, anti-oráculo y cuatro
barajados del mismo multiconjunto de cuotas. El evaluador aborta si cualquier
brazo consume un presupuesto diferente.

## Resultado confirmatorio

| arquitectura | acc. online | acc. uniforme | diferencia | corr(K,q) | q=K exacto |
|---|---:|---:|---:|---:|---:|
| `gating_wm` | 0,7847 | 0,6093 | **+0,1755** | 0,898 | 54,4% |
| `hbp_first` | 0,7880 | 0,6093 | **+0,1787** | 0,903 | 55,2% |
| `hbp_full` | 0,7861 | 0,6093 | **+0,1769** | 0,907 | 55,9% |

Cada arquitectura es positiva en 3/3 semillas. En las nueve corridas, el test
exacto bilateral por clúster de batch da `p=1,91×10⁻⁶` frente a uniforme.

Por seed, la mejora del control es `+0,1889`, `+0,1806` y `+0,1569`. El oráculo
alcanza 0,8394 de media y las cuotas barajadas 0,5792: queda margen real y la
identidad de la asignación importa, no sólo su histograma.

## ¿Aporta el HBP?

No hay evidencia confirmatoria:

| contraste pareado | diferencia media de accuracy | aciertos por seed | p exacta sobre seeds |
|---|---:|---:|---:|
| `hbp_first − gating_wm` | +0,00324 | 0, −2, +9 | 1,00 |
| `hbp_full − gating_wm` | +0,00139 | −5, +1, +7 | 0,75 |
| `hbp_full − hbp_first` | −0,00185 | −5, +3, −2 | 0,75 |

Las políticas HBP cambian entre 7,5% y 16,4% de las cuotas respecto al control,
pero esos cambios mejoran unas muestras y empeoran otras. El segundo orden no
produce una ventaja robusta en el contraste primario.

## Control negativo: borrar `LAST`

Se reevaluaron los mismos checkpoints reemplazando el token terminal por su
operación ordinaria equivalente. Así se conserva la transición de tarea pero se
elimina la señal de final en el tick `K`.

| arquitectura | Δ online−uniforme con `LAST` | Δ sin `LAST` | corr(K,q) sin `LAST` |
|---|---:|---:|---:|
| `gating_wm` | +0,1755 | +0,0968 | 0,618 |
| `hbp_first` | +0,1787 | +0,0963 | 0,682 |
| `hbp_full` | +0,1769 | +0,1028 | 0,680 |

El deterioro confirma que la política usa información causal del flujo. El
efecto no desaparece porque, después de gastar un tick adicional, el controller
puede detectar progreso nulo y reasignar el resto del presupuesto.

En esta ablación `hbp_full` conserva unos `+0,0060` puntos sobre el control y lo
supera por 5, 4 y 4 aciertos en las tres seeds. Con `n=3`, el mejor p exacto
posible es 0,25: es una **señal exploratoria**, no evidencia suficiente.

## Interpretación

El resultado avanza respecto al benchmark `q=K` supervisado:

> Un controlador sin acceso anticipado a la demanda puede aprender, mediante
> pérdida de tarea y presión presupuestaria, a reconocer dentro del flujo qué
> procesos necesitan más cómputo y asignarlo online para aumentar la accuracy.

Esto se acerca a una regulación interna de recursos, pero sigue sin demostrar
metas auto-generadas, iniciativa abierta, consciencia o “alma”. El objetivo de
tarea y el presupuesto siguen viniendo del exterior, y `LAST` es una señal
ambiental explícita. Tampoco se igualan FLOPs o latencia end-to-end del HBP: la
unidad causal es un tick del executor.

El siguiente contraste informativo es el régimen sin `LAST`, entrenado así desde
el principio y con más semillas. Ahí la terminación debe inferirse de estabilidad
o progreso nulo; la pequeña ventaja exploratoria de `hbp_full` puede confirmarse
o desaparecer.

## Continuación realizada

El régimen sin `LAST` ya está entrenado desde cero con 10 seeds y documentado en
`results_implicit_need/README.md`. La asignación sigue mejorando fuertemente la
accuracy. `hbp_full` supera nominalmente al control en `+0,0142`
(`p_exacta=0,0254`), pero el efecto es sensible a una seed y no es significativo
frente a `hbp_first`; se reporta como positivo provisional, no como prueba del
segundo orden.

## Reproducción

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
& 'C:\Users\fmarr\miniconda3\envs\implanto\python.exe' `
  -u hidden_need_benchmark.py `
  --variants gating_wm,hbp_first,hbp_full --seeds 0,1,2 `
  --pretrain-steps 2000 --steps 2000 --eval-batches 20 `
  --out-dir results_hidden_need

& 'C:\Users\fmarr\miniconda3\envs\implanto\python.exe' `
  hidden_need_terminal_ablation.py --results-dir results_hidden_need
```

Archivos principales:

- `summary.json`: agregado y contrastes arquitectónicos pareados.
- `{variant}_seed{0,1,2}.json`: cuotas online, predicciones y tests por muestra.
- `terminal_ablation/`: control negativo sin marcador.
- `checkpoints/`: nueve checkpoints pareados.
- `../hidden_need_benchmark.py`: entrenamiento sin `K` ni completion loss.
- `../eval/hidden_need.py`: asignador online causal.
- `../hidden_need_terminal_ablation.py`: control negativo.
