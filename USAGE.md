# Usar MiuraCognitive

MiuraCognitive es utilizable como herramienta: se entrena un modelo una vez, se
guarda un checkpoint, y luego se usa desde el CLI o desde Python. Cada inferencia
expone la **traza del campo homeostático (HBP)** — el "pensamiento" del modelo.

> Ejecutar desde la raíz del paquete con `$env:PYTHONPATH="."` (PowerShell) o
> `PYTHONPATH=.` (bash). Intérprete con torch+CUDA: el env `implanto`.

## 1. Entrenar y guardar un modelo

```powershell
python miura.py train --variant hbp_full --gens adjacent --steps 2500 --out checkpoints/hbp_full.pt
```

Opciones: `--variant {vanilla,gating,gating_wm,hbp_first,hbp_full}`,
`--gens {adjacent,cycle_transp}`, `--max_ops N` (dificultad máx. de entrenamiento),
`--max_halt_steps N` (presupuesto de pensamiento), `--device cpu|cuda:0`.

## 2. Usar el modelo: componer permutaciones de S₅

Los argumentos son índices de generador (transposiciones adyacentes: `g0=(0 1)`,
`g1=(1 2)`, ...). El modelo compone en orden y predice la permutación resultante.

```powershell
python miura.py compose 0 2 1 3 1 --ckpt checkpoints/hbp_full.pt
```
```
Estado inicial (identidad): 01234
Operaciones (K=5): g0, g2, g1, g3, g1
Resultado VERDADERO:  10423
Predicción del modelo: 10423   CORRECTO  (confianza 0.98)
El reasoner pensó E[n_iter] = 7.66 iteraciones.

Traza del HBP (campo homeostático por tick de pensamiento):
  tick   VEI_var  desviación  umbral_halt  mod_norm
     1    0.012       0.35        0.500       0.16
     2    0.089       0.73        0.504       0.30
     ...
```

La **traza** es lo distintivo: muestra cómo evoluciona el estado interno (VEI) a lo
largo de las iteraciones de pensamiento, cuánto se desvía del reposo homeostático,
y cómo cambian el umbral de parada y la modulación del reasoner.

## 3. Introspección del campo homeostático

```powershell
python miura.py introspect --ckpt checkpoints/hbp_full.pt
```
Reporta la varianza del VEI, la frecuencia dominante (¿oscila?), el ζ medio y el
régimen de amortiguamiento — el "estado físico" del campo interno del modelo.

## 4. Uso desde Python (API)

```python
from miura_infer import MiuraModel

m = MiuraModel.from_checkpoint("checkpoints/hbp_full.pt")

r = m.compose([0, 2, 1, 3, 1])           # ComposeResult
print(r.predicted, r.correct, r.n_iter, r.confidence)
for tick in r.trace:                     # dinámica del HBP por iteración
    print(tick["n"], tick["vei_var"], tick["halt_threshold"])

print(m.introspect())                    # diagnóstico del HBP
print(m.info())                          # ficha del modelo
```

`MiuraModel.compose(ops)` devuelve un `ComposeResult` con: `predicted`, `true_perm`,
`correct`, `n_iter` (cuánto pensó), `confidence`, y `trace` (la dinámica del HBP).

## Qué es cada variante

| Variante | Qué es |
|---|---|
| `vanilla` | transformer de profundidad fija (sin recurrencia) |
| `gating` | + reasoner recurrente de profundidad adaptativa |
| `gating_wm` | + memoria de trabajo (control sin HBP) |
| `hbp_first` | + HBP de 1er orden (relajación) |
| `hbp_full` | + HBP de 2º orden (onda amortiguada) — la arquitectura completa |

## Nota sobre calidad del checkpoint

Un modelo entrenado ~2500 pasos en GPU resuelve composiciones cortas (K≤6) casi
perfectamente y se degrada con la profundidad K (es la tarea difícil por diseño).
Para uso serio, entrena en GPU; en CPU sirve para probar la herramienta.
