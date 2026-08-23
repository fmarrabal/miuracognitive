# AHA-1 — Anticipatory Homeostatic Agency

Primer peldaño nuevo tras la autorregulación de cómputo:

`autorregulación → **anticipación** → metas endógenas → metas descubiertas → automodelo → metacognición`

## Pregunta causal

¿Puede un agente sin `goal_id` actuar antes de que aparezca un déficit, usando
una predicción aprendida de sus futuras necesidades, y desaparece esa conducta
al intervenir específicamente el mecanismo predictivo?

## Diseño

- Tres variables internas continuas: energía, temperatura e integridad.
- Cinco perturbaciones por trayectoria; tipo, tiempo y magnitud aleatorios.
- Un cue puntual aparece seis ticks antes del daño.
- El cue desaparece; la acción correcta debe iniciarse después, durante un
  intervalo externamente vacío.
- Las acciones restauradoras tardan tres ticks en surtir efecto.
- No se entrega tarea, meta, tiempo de evento ni perturbación futura.
- Un GRU aprende de forma auto-supervisada la perturbación acumulada futura.
  Después se congela antes de entrenar el controlador.
- La política minimiza desviación bilateral del punto de consigna, salida de la
  región viable y coste de acción.

Variantes:

- `reactive`: ejecuta el predictor para igualar cómputo, pero su salida está
  causalmente desconectada del ejecutivo;
- `gating_wm`: ejecutivo recurrente que recibe la predicción;
- `hbp_first`: mismo ejecutivo + HBP de primer orden;
- `hbp_full`: mismo ejecutivo + HBP de segundo orden.

El HBP empieza como identidad exacta sobre `gating_wm`: su cabeza de prioridad
se inicializa en cero. Los cuatro brazos reutilizan el mismo predictor congelado
por seed.

## Intervenciones

1. `prediction_lesion`: sustituye por cero sólo la salida del predictor.
2. `cue_shuffle`: conserva el multiconjunto de cues pero lo asigna a otros
   mundos, rompiendo su validez causal.
3. `stationary`: elimina perturbaciones; controla actividad preventiva espuria.
4. `need_transplant`: cambia sólo una necesidad bajo un exterior idéntico y
   mide si la prioridad conductual sigue al cuerpo trasplantado.

Métricas primarias: supervivencia, tasa de violación, anticipación correcta por
evento e iniciativa correcta en intervalo vacío. Se registran además esfuerzo,
acciones falsas, error homeostático y calibración del predictor.

## Piloto exploratorio calibrado (`pilot_v4`, seed 0)

Los números siguientes no son confirmatorios. Se usan para verificar que la
tarea separa anticipación de reacción antes de gastar seeds preregistradas.

| variante | supervivencia | violación | anticipación | iniciativa vacía | tasa de acción |
|---|---:|---:|---:|---:|---:|
| `reactive` | 0,290 | 0,0209 | 0,186 | 0,171 | 0,221 |
| `gating_wm` | **0,902** | **0,00124** | **0,900** | **0,842** | **0,205** |
| `hbp_first` | 0,792 | 0,00333 | 0,204 | 0,188 | 0,263 |
| `hbp_full` | 0,427 | 0,0155 | 0,205 | 0,193 | 0,234 |

Controles causales de `gating_wm`:

- lesión predictiva: supervivencia `0,902 → 0,002`;
- cues barajados: supervivencia `0,902 → 0,106`;
- trasplante de necesidad: la meta cambia en `100%` de las sondas;
- predictor: `corr(predicción, daño futuro)=0,927`;
- mundo estacionario: supervivencia `1,0`, tasa de acción `0,030`.

Contrastes arquitectónicos exploratorios:

- `gating_wm − reactive`: `+0,6125` de supervivencia;
- `hbp_first − gating_wm`: `−0,1104`;
- `hbp_full − gating_wm`: `−0,4750`;
- `hbp_full − hbp_first`: `−0,3646`.

La lesión directa `VEI → prioridad` muestra coadaptación distinta:

- en `hbp_first`, quitar la modulación eleva supervivencia
  `0,792 → 0,998`, pero a costa de actuar mucho más (`0,263 → 0,569`) y
  sobrecompensar el punto de consigna;
- en `hbp_full`, quitarla reduce supervivencia `0,427 → 0,023`: el campo sí es
  causal dentro de ese agente, pero no alcanza al control sin HBP.

Esto es una primera señal de **autorregulación anticipatoria causal** en
`gating_wm`: la conducta correcta ocurre antes del déficit y depende del
contenido predictivo. No hay señal de ventaja HBP en esta seed; en particular,
el segundo orden es peor que el primero. Tampoco demuestra generación abierta
de metas, consciencia o «alma».

## Confirmación preregistrada (`confirmatory_v1`, 20 seeds nuevas)

El protocolo se congeló antes de ejecutar las seeds `100..119`; la seed del
piloto no se incluyó. La familia primaria exigía que cuatro efectos fueran
positivos y que los cuatro tests exactos bilaterales por seed sobrevivieran a
Holm con `alpha=0,05`. El resultado fue **PASS en los cuatro criterios**:

| criterio primario | efecto medio | seeds a favor | p exacta | p Holm |
|---|---:|---:|---:|---:|
| supervivencia `gating_wm − reactive` | +0,7240 | 20/20 | 1,91e-6 | 7,63e-6 |
| iniciativa vacía `gating_wm − reactive` | +0,7306 | 20/20 | 1,91e-6 | 7,63e-6 |
| predicción sobre supervivencia de `gating_wm` | +0,9548 | 20/20 | 1,91e-6 | 7,63e-6 |
| cue válido sobre supervivencia de `gating_wm` | +0,8468 | 20/20 | 1,91e-6 | 7,63e-6 |

Medias normales confirmatorias:

| variante | supervivencia | violación | anticipación | iniciativa vacía | acciones |
|---|---:|---:|---:|---:|---:|
| `gating_wm` | **0,9549** | **0,00060** | **0,9654** | **0,8930** | 0,2011 |
| `reactive` | 0,2310 | 0,02427 | 0,1755 | 0,1625 | 0,2208 |

El resultado no se explica por actuar más: `gating_wm` actúa algo menos y su
fracción de acciones falsas cae de `0,917` a `0,416`. El mismo predictor se
ejecuta en ambos brazos (`corr=0,945`), pero sólo el brazo predictivo puede usar
su salida. Lesionarlo deja su supervivencia media en `0,00016`; barajar los cues
la deja en `0,1082`. En el mundo estacionario, su tasa de acción es `0,0237`.

El trasplante produjo `+0,203` de probabilidad para la necesidad intervenida,
pero la métrica preregistrada de cambio de `argmax` en el tick 0 fue `0`. Una
auditoría post hoc mostró que la respuesta aparece con latencia de 1–4 ticks:
en los primeros seis ticks hubo acción correcta en 56/60 combinaciones
seed-necesidad (`93,3%`) bajo déficit y en 0/60 bajo saciedad. Este análisis
explica la métrica puntual, pero no se usa para rescatar ni reforzar el PASS.

Conclusión permitida: **AHA-1 confirma autorregulación anticipatoria causal en
este benchmark**. No confirma generación abierta de metas, consciencia ni una
equivalencia con la homeostasis humana.

Datos, protocolo y trazabilidad: [`confirmatory_v1/`](confirmatory_v1/).

## Confusores detectados y corregidos durante el piloto

1. Penalizar sólo déficit permitía actuar siempre y saturar niveles. Se cambió
   a error homeostático bilateral y se elevó el coste de acción.
2. Un coseno corto congelaba el predictor cuando empezaba a aprender la
   latencia. Su fase usa ahora warm-up y tasa constante.
3. La modulación HBP inicialmente aleatoria rompía el emparejamiento y la
   auxiliar interoceptiva dominaba el objetivo. La rama empieza ahora como
   identidad y las auxiliares son secundarias.

Ningún resultado de las versiones anteriores a estas correcciones se considera
evidencia.

## Reproducción del piloto

Desde la raíz `miuracognitive`:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python -u aha_benchmark.py `
  --variants gating_wm,reactive,hbp_first,hbp_full `
  --seeds 0 --predictor-steps 600 --steps 600 `
  --batch-size 64 --eval-batches 5 --eval-batch-size 96 `
  --out-dir results_aha/pilot_v4
```

La réplica confirmatoria de 20 seeds ya fue ejecutada con 1.600 actualizaciones
de controlador y análisis exacto por seed con corrección de Holm. HBP quedó
fuera de la familia primaria y no se recalibró mirando estas seeds.

La arquitectura posterior está descrita en [`AGENCY_ROADMAP.md`](../AGENCY_ROADMAP.md).
