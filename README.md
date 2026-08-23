# MiuraCognitive

Programa de investigación sobre **dónde vive la cognición** en una arquitectura
mínima pero completa: un reasoner recurrente con parada adaptativa, un campo de
control homeostático y un módulo de valor. A cada parte le hicimos la misma
pregunta: ¿esta función **emerge** del descenso de gradiente, o hay que
**computarla** con maquinaria explícita?

El objeto central es el **Homeostatic Background Processor (HBP)**: un campo de
estado interno de baja dimensión definido sobre el grafo de módulos de un
transformer, gobernado por una **familia** de ecuaciones en derivadas parciales
sobre el laplaciano del grafo —onda amortiguada (Klein–Gordon forzada), su
límite difusivo y una dinámica tipo KdV—. Evoluciona **un tick por iteración del
reasoner** y modula su cómputo (umbral de halting, ganancia de bloque, gates de
memoria).

    ∂²h/∂t² + 2ζω₀ ∂h/∂t − c²∇²h + ω₀²(h − h*) = f_θ(h,s) + g_φ(h,x)

## Los dos papers

| | |
|---|---|
| [`paper/main.tex`](paper/main.tex) | **Where Cognition Lives** — la jerarquía de información, qué emerge y qué hay que computar, y el artefacto de lectura |
| [`paper/governor_en.tex`](paper/governor_en.tex) | **El campo como gobernador de cómputo** — la familia de PDE, los certificados de estabilidad y la campaña de des-confusión |

Son *companion* mutuos. `paper/build_all.ps1` construye las cuatro salidas
(versión con autores y versión doble ciego, para cada uno) desde el mismo
fuente, mediante el interruptor `\anonfalse` / `\anontrue`.

## Qué concluye el programa

Esta sección es el resumen honesto y **sustituye a versiones anteriores de este
README** que anunciaban como confirmados resultados que el propio repositorio
acabó refutando. Cada afirmación remite al documento de hallazgos que la
sostiene.

### La competencia emerge; la parada, no como parecía

A cómputo medio equiparado, el rendimiento sube de `0.467` (asignación
uniforme) a `0.546` (dificultad) y a `0.698` (valor ex-ante). El siguiente
peldaño aparente —`0.921`, la auto-observación posterior— **no sobrevive a la
auditoría**: el halting tipo PonderNet devuelve una **mezcla** de estados
ocultos ponderada por la distribución de parada, mientras que todos los
baselines de profundidad forzada devuelven un estado único, y la cabeza de
lenguaje se entrena solo sobre la mezcla.

Igualando la lectura, la ventaja de la ejecución nativa **se anula por
completo**: régimen residual `+0.000 [0.000, 0.000]`. Con la lectura fija y el
presupuesto igualado, saber qué instancia necesita cuánto cómputo vale
`+0.0011 [+0.0003, +0.0019]`. Añadir valor encima del posterior no compra nada
(`+0.0002 ± 0.0004`).

→ [`FINDINGS_READOUT.md`](FINDINGS_READOUT.md)

### El valor no emerge: hay que computarlo

Los acoplamientos entrenados capturan **cero** de un payoff disponible que un
allocator explícito captura por completo (`+0.151`, correlación de enrutado
`+0.79`). La anticipación, en cambio, no tiene payoff que capturar en estas
familias (`≤+0.001` en 31 configuraciones). La salvedad importa: aquí el stake
es ⊥ al contenido **por construcción**.

### Sobre un LLM congelado, los mismos instrumentos

El techo del voto por self-consistency es una cota **medida**:
`+0.0236 [+0.0150, +0.0326]`. El acuerdo entre muestras es casi inútil como
señal de parada —su masa se concentra en respuestas erróneas—.

→ [`mhbp/tasks/llm_gov/FINDINGS_LLM.md`](mhbp/tasks/llm_gov/FINDINGS_LLM.md)

### La predicción propia, ejecutada

En una familia de coste tipo acantilado, el valor bajo compromiso paga
`+0.1312 [+0.1124, +0.1502]`, unas **siete veces** la estimación puntual de la
familia suave (`+0.019`). Pero **no** porque el acantilado desplace la
información hacia lo ex-ante: porque multiplica el rango alcanzable por
`5.1× [3.4, 8.2]`. Magnitud del problema y estructura de su información son
ejes separados.

### El campo: sustancia no, estructura solo en parte, certificabilidad sí

- **Sustancia (no).** El *tipo* de física del campo es irrelevante para la
  accuracy: onda, difusión, mezclas gateadas, acoplamiento no local tipo
  Poisson e incluso un sustrato de flujo 2D con Navier–Stokes dan lo mismo.
- **Estructura (solo en parte).** El segundo orden dota a la asignación de
  cómputo OOD de una robustez que el halting aprendido end-to-end no tiene —
  pero una campaña pre-registrada de des-confusión con 20 seeds frescas acota
  el claim. Con topes equiparados, el efecto de orden es fuerte en una familia
  de generadores (`+0.087 [+0.042, +0.132]`, t=4.0) y **no se detecta** en la
  otra (`+0.014 [−0.013, +0.040]`, n.s.): parte del contraste original era
  **capacidad, no orden**. Un GRU de interfaz equiparada es indistinguible en
  la primera familia (`+0.006`, n.s.) y **gana nominalmente** en la segunda
  (`−0.035 [−0.067, −0.002]`).
- **Certificabilidad (sí).** Lo que distingue al campo no es la capacidad sino
  que su estabilidad es **demostrable**, y sobre todo que el operador de un
  paso admite una **comprobación exacta en runtime**.

→ [`FINDINGS_V4.md`](FINDINGS_V4.md) · [`FINDINGS_TEORIA.md`](FINDINGS_TEORIA.md)

### Qué está certificado, y qué no

Existe **P común sobre la envolvente de los checkpoints entrenados**
(ω₀∈[0.488,0.529], ζ∈[0.475,0.538], c≤0.371): la LMI cierra con `ρ=0.9517`,
κ=2.51, y el peor radio espectral individual es `0.7322`. Es una cota en
**norma**, no en radio espectral, así que cubre transitorios no normales y, por
convexidad y submultiplicatividad, las mezclas y el caso gateado (LTV).

Lo que **no** cierra, dicho sin adornos:

1. **La caja declarada en `HBPConfig` no es certificable, porque no es
   estable**: de sus 64 vértices, **40 divergen** (peor `ρ=5.24`). Lo que
   mantiene al modelo lejos de la divergencia es el entrenamiento, no los
   topes del código.
2. La mezcla con la rama difusiva sigue sin certificado (`max‖Φ‖_P = 1.57`).
3. El small-gain no cierra, por un factor ~48.

Es un **certificado del integrador**, no de lazo cerrado, y los papers lo dicen
explícitamente.

→ [`FINDINGS_LMI.md`](FINDINGS_LMI.md) · `certify_lmi.py`, `certify_lmi_ckpt.py`,
`certify_lmi_iss.py`, `verify_verlet_schurcohn.py`

### El campo como acumulador de evidencia: nulo con control positivo

Un kill-gate con sonda de control positivo no encuentra evidencia de la ruta
restante sobre doce solvers congelados: `ΔAUC = +0.0007 [−0.0065, +0.0079]`
frente a un umbral de paso de `0.03`. Lectura propuesta (no demostrada): el
estado recurrente ya integra su propia historia.

→ [`mhbp/tasks/reasoner_g0/FINDINGS_N4.md`](mhbp/tasks/reasoner_g0/FINDINGS_N4.md)

### Línea de agencia: v1 retirada, v2 confirmada

> **Las cuatro fases originales (v1) quedaron RETIRADAS** tras una auditoría
> adversarial que encontró circularidad: en la fase 3 el descubridor reproducía
> el `1.0000` **sin entrenar**, y la ley generativa del entorno estaba
> inyectada en el agente. Los ficheros v1 se conservan como registro histórico;
> sus números **no deben citarse**.

El rediseño v2 impone ocho reglas anti-circularidad (baselines externos
afinados, prohibido inyectar la ley del entorno, guardrail G-untrained
obligatorio) y **las cuatro fases pasan** con 20 seeds frescas:

| Fase | Contraste primario | Efecto |
|---|---|---|
| F1 anticipación | aprendido − combo scripted afinado | `+0.514` (20/20) |
| F2 compromiso | aprendido − `smart` con la regla | `+0.599`; OOD `+0.697` |
| F3 descubrimiento | regret mejor scripted − aprendido | `+0.071` (20/20) |
| F4 automodelo | frozen − adaptive, 3 severidades | `+0.062 / +0.056 / +0.057` |

La escalada F2X es un **negativo con estructura**: el designer afinado gana en
las tres condiciones, pero el gap se estrecha un 90 % con el desplazamiento de
distribución (`−0.471 → −0.276 → −0.049`).

Y el dato que cierra el hilo: **el HBP es neutro en su prueba más natural**
(conflicto multi-impulso): `hbp_wave −0.032` (p=0.29), `hbp_diff −0.0006`
(p=1.0). La recurrencia, en cambio, sí importa (`+0.251`, p=0.008). Disociación
limpia: la recurrencia hace falta, el campo homeostático no.

→ [`AGENCY_V2.md`](AGENCY_V2.md)

## Entorno

```powershell
$env:PYTHONPATH="."; $env:PYTHONIOENCODING="utf-8"   # utf-8 evita crashes cp1252 (ζ, Δ, ✓)
```

Conda con PyTorch cu128 (RTX PRO 5000 Blackwell, sm_120). Dependencias en
`requirements.txt` y `environment.yml`. El capítulo de LLM descarga
Qwen2.5-14B-Instruct (Apache-2.0, ~28 GB) la primera vez.

## Reproducción

Los JSON de resumen de todos los experimentos **están en el repositorio** (1 688
ficheros, 31 MB), de modo que las tablas y figuras de los dos papers se
regeneran sin re-entrenar nada. Los registros pesados por instancia (187 MB) y
los checkpoints (1,8 GB) viven en el depósito Zenodo cuyo DOI citan los papers.

```powershell
python verify_setup.py        # entorno: GPU / BF16 / SDPA, con fallback CPU
python _audit_hbp.py          # regresión del HBP: grad(ζ,ω₀)≠0, modulación dep. del input
python eval/diagnostics.py    # VEI vivo, FFT/oscilación, régimen de amortiguamiento
python example_routine.py     # ejemplo: hbp_full resolviendo una composición de S₅
```

### Mapa: resultado del paper → script → fichero de resultados

| Resultado | Script | Resultados |
|---|---|---|
| Jerarquía de información | `mhbp/tasks/reasoner_g0/n3_*.py` | `mhbp/tasks/reasoner_g0/results/` |
| **Artefacto de lectura** | `n3_readout.py`, `n3_readout_fair.py` | `n3_readout*.json` |
| Allocator explícito vs acoplado | `n3_allocator.py` | `FINDINGS_N3.md` |
| Acantilado (N1b) | `mhbp/tasks/llm_gov/llm_n1b*.py` | `results/llm_n1b_*.json` |
| Techo de self-consistency | `mhbp/tasks/llm_gov/llm_gsm8k.py` | `results/llm_gsm8k.json` |
| Des-confusión v4 | `benchmark_v4.py` | `results_benchmark_v4/` |
| Certificados LMI | `certify_lmi*.py` | `FINDINGS_LMI.md` |
| Criterio de Verlet | `verify_verlet_schurcohn.py` | `FINDINGS_TEORIA.md` |
| Kill-gate del acumulador | `mhbp/tasks/reasoner_g0/n4_*.py` | `FINDINGS_N4.md` |
| Agencia v2 (4 fases) | `*_v2_benchmark.py` | `results_*_v2/` |

### Benchmark completo

```powershell
python benchmark.py                 # 5 variantes × 2 generadores × 2 regímenes × 3 semillas
python extend_benchmark_ood.py      # extiende las celdas OOD a n=10
python benchmark_report.py          # tabla agregada
```

### Variantes

| Variante | HBP | Reasoner | WM | Qué aísla |
|---|---|---|---|---|
| `vanilla` | – | – | – | baseline transformer de profundidad fija |
| `gating` | – | ✓ | – | recurrencia adaptativa sola |
| `gating_wm` | – | ✓ | ✓ | **control**: recurrencia + working memory, sin HBP |
| `hbp_first` | 1º | ✓ | ✓ | límite sobreamortiguado (relajación) |
| `hbp_full` | 2º | ✓ | ✓ | Verlet completo (inercia/oscilación) |
| `hbp_gru` | – | ✓ | ✓ | GRU de **interfaz equiparada** (el rival honesto del v4) |

### Tareas

- `permcomp` — composición de generadores de S₅ (NC¹-dura; la principal).
  Generadores: `adjacent` o `cycle_transp`.
- `runsum` — suma corriente mod n; se resuelve en una pasada, útil de contraste.
- `recall` — recall con distractores (saturado; histórico).

Extrapolación: `--train_max_writes 12 --max_writes 24` entrena en K≤12 y evalúa
hasta K≤24.

## Estructura

```
model/        transformer.py · hbp.py ★ · adaptive_depth.py · working_memory.py · miura.py
data/         synthetic_recall.py · budgeted_stream.py
training/     config.py · trainer.py
eval/         diagnostics.py · aggregate.py · compute_bottleneck.py
experiments/  run_ablations.py

mhbp/                        el código de los DOS papers
  tasks/reasoner_g0/         jerarquía de información, readout, allocator, kill-gate N4
  tasks/llm_gov/             actuador LLM congelado: GSM8K, acantilado N1b, palancas
  analysis/                  FINDINGS_PHASE2.md y siguientes

paper/        main.tex · governor_en.tex · refs.bib · figures/ · build_all.ps1
              check_anon.py (verifica el doble ciego) · pack_arxiv.py

certify_lmi.py · certify_lmi_ckpt.py · certify_lmi_iss.py   certificados de estabilidad
verify_verlet_schurcohn.py                                  criterio de Verlet/Schur–Cohn
benchmark_v4.py                                             campaña de des-confusión
_audit_hbp.py                                               test de regresión del HBP
```

## Notas de implementación

- El HBP se acopla al reasoner con **modulación suave** `gate = 1 + s·tanh(·)`
  alrededor de la identidad. Un gate `sigmoid` (centrado en 0.5) decae la señal
  y rompe la iteración.
- El Verlet va en **FP32**: la cancelación de velocidad en BF16 corrompe el
  amortiguamiento cerca del equilibrio.
- **Bug de congelación en BF16 (crítico, retroactivo).** `raw_ζ`, `ω₀`, `c` y
  `f_gain` quedaban congelados en su init en todos los runs GPU/BF16, porque el
  paso de Adam caía por debajo de ULP/2. El fix `pin_fp32()` es obligatorio tras
  `.to(bf16)`. Los resultados anteriores al fix tienen la física **fija, no
  aprendida**.
- Los topes de ω₀ de `hbp_first` eran un **artefacto del integrador explícito**:
  el kernel implícito incondicionalmente contractivo los hace innecesarios, y
  por eso el brazo justo del v4 es `hbp_first_eq` con topes equiparados.

## Licencia

Código, documentos y resultados propios bajo **MIT** (`LICENSE`). El material de
terceros —el test de GSM8K, las generaciones del actuador y los ficheros de
estilo de LaTeX— conserva su propia licencia y aviso: ver [`NOTICE`](NOTICE).
