# FINDINGS READOUT — el «régimen» del control yoked es artefacto de lectura
### 2026-08-19. 12 checkpoints congelados, eval común (4096 inst), 4 min GPU.
### Script: `mhbp/tasks/reasoner_g0/n3_readout.py` → `results/n3_readout.json`

## El confound

`AdaptiveHalting.forward` devuelve `weighted_state = Σₙ pₙ·xₙ` —una MEZCLA
PonderNet de estados ocultos—, mientras `forward_forced` devuelve el estado
único tras `forced_steps` (su propio docstring: *«esta ruta no forma una
mezcla PonderNet»*). El brazo nativo lee de la mezcla; todos los brazos
forzados leen de un estado único. Y `ponder_expected_loss=False` en estos
checkpoints (default de `build_model`, que `n3_sonda.load_ckpt` no cambia):
la CE se aplicó a los logits DE LA MEZCLA, así que la `lm_head` nunca vio
un estado único durante el entrenamiento.

## Medida (post-hoc sobre la MISMA trayectoria nativa, sin reentrenar)

Controles del instrumento, ambos exactos:
grabar estados no altera el payoff nativo (+0.0000) y la reconstrucción
Σₙ pₙ·xₙ reproduce el brazo nativo (+0.0000). Además `nativo_nhat`
(selección post-hoc del estado a n̂ sobre el rollout nativo) coincide con
`yoked` (re-ejecución forzada independiente) en los 12 checkpoints — dos
cálculos independientes del mismo número.

| término | contraste | Δ | IC95 |
|---|---|---|---|
| total | mix − exante5 | **+0.4301** | [+0.3968, +0.4634] |
| A lectura | mix − nhat | **+0.3829** | [+0.3414, +0.4244] |
| B régimen residual | nhat − yoked | **+0.0000** | [+0.0000, +0.0000] |
| C información | yoked − exante5 | **+0.0472** | [+0.0292, +0.0653] |

A+B+C = +0.4301 = total (residuo 0.0e+00). Leer el estado de la moda de
pₙ cuesta −0.867: los estados individuales son indescifrables para la
cabeza.

## Veredicto

El **\Yregimen publicado (+0.383 [+0.341,+0.424]) es, cifra a cifra y en
ambos extremos del intervalo, exactamente el término de LECTURA**. La
componente de «ejecución elástica on-policy» es **exactamente cero**:
forzar a las profundidades que el halting eligió reproduce la trayectoria
nativa punto por punto. La frase del paper «its value is realized in the
native act of stopping itself» no está soportada. El control
pre-declarado VG-N3d (−0.34, «forced-depth execution is costly in
itself») es el mismo artefacto con otra etiqueta.

## Qué sobrevive y qué no

**Sobrevive** (comparaciones internamente justas, misma lectura en ambos
brazos): C = +0.047 — las profundidades posteriores trasplantadas compran
menos que la dificultad sola; y los peldaños bajos de la jerarquía
(uniforme 0.467 < dificultad 0.546 < ex-ante 0.698), todos vía
`run_forced`.

**No sobrevive**: el salto al peldaño posterior (0.698→0.921, +0.223). En
`n3_eval.py` todos los brazos de asignación pasan por `run_forced`
(estado único) y solo `nativo` por `run_native` (mezcla): el peldaño alto
se mide con otra lectura que los demás.

## Lectura para el paper

El titular pasa de cognitivo a metodológico, y gana: *toda evaluación de
profundidad adaptativa que compare halting nativo contra baselines de
profundidad forzada mide la LECTURA, no la asignación*, siempre que el
modelo se entrene con la pérdida de la mezcla. Aquí el artefacto vale
+0.383 de +0.430 con residuo exactamente cero. Diagnóstico de línea
clara: comprobar si la ruta forzada devuelve un estado único mientras la
nativa devuelve Σₙ pₙ·xₙ. Converge con Popescu et al. (arXiv 2607.20519,
julio 2026), que encuentran que las lecturas dominan sobre la
expresividad de los gates; aquí la dominación es total y cuantificada.

---

# ADENDA — el instrumento corregido, ejecutado
### `n3_readout_fair.py` → `results/n3_readout_fair.json`. 12 ckpts, 2 min GPU.

La auditoría invalida la comparación nativo-vs-forzado pero no la pregunta
de fondo. Versión limpia: **todos los brazos dentro de la familia de
mezclas**, sobre los mismos estados grabados, variando solo los pesos qₙ.
El contraste decisivo da a cada instancia la distribución de halting de
**otra** instancia (derangement, que conserva el presupuesto medio por
construcción).

| contraste | Δ | IC95 |
|---|---|---|
| asignación por-instancia (propio − permutado) | **+0.0011** | [+0.0003, +0.0019] |
| ídem, 2ª permutación independiente | +0.0006 | [−0.0004, +0.0016] n.s. |
| forma poblacional (permutado − uniforme U{1..9}) | +0.2668 | [+0.2458, +0.2878] |
| total dentro de la familia de mezclas | +0.2679 | [+0.2473, +0.2885] |
| colapsar a masa puntual (referencia) | +0.3829 | [+0.3414, +0.4244] |

**Control de presupuesto**: n̄ propio − n̄ permutado = +0.0000 ticks.
**Control positivo del nulo** (la permutación TIENE que mover algo): std
de E[n] entre instancias = 1.264; desplazamiento medio |E[n]ᵢ −
E[n]_σ(ᵢ)| = **1.443 ticks** sobre un presupuesto medio de 5.4 (≈27% de
desplazamiento relativo); distancia TV media entre pᵢ y p_σ(ᵢ) = 0.068.
La permutación reasigna de verdad — y aun así no cambia nada.

## Veredicto del instrumento corregido

Con la lectura fija y el presupuesto casado, **saber qué instancia
necesita cuánto cómputo vale +0.001**: una parte en mil. Lo que sí vale
(+0.267) es la FORMA poblacional de la mezcla aprendida frente a una
uniforme plana — que no es conocimiento sobre la instancia que tienes
delante. «El posterior actúa, no sabe» se corrige a: **el posterior ni
sabe ni actúa; la ventaja era de la lectura y de la forma de la mezcla.**

Efecto colateral que MEJORA el paper: el capítulo del LLM medía que la
parada por acuerdo compra ≤+0.017 sobre ex-ante, y se contaba como
contraste («decisivo en un régimen, casi inútil en el otro»). Con la
auditoría, **ambos regímenes convergen**: mirarse pensar es casi inútil en
los dos. La aparente diferencia existía solo mientras uno de los dos se
medía a través de una lectura que el otro no compartía.
