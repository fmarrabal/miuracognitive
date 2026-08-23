> ## ⚠️ Este documento describe un manuscrito SUPERADO
>
> Se escribió para `paper/paper.pdf`, que era el manuscrito único anterior a
> la división en dos papers (2026-08-16). Ese fichero ya no forma parte del
> árbol público.
>
> El mapa vigente **resultado → script → fichero de resultados** está en
> [README.md](README.md), y cubre los dos manuscritos actuales:
> `paper/main.tex` (cognición) y `paper/governor_en.tex` (gobernador).
>
> Lo que sigue se conserva porque el detalle de comandos y de entorno sigue
> siendo válido.

# Reproducibilidad — "Una familia de campos PDE homeostáticos sobre el grafo de módulos"

Paquete de reproducción del artículo (`paper/paper.pdf`). Código PyTorch puro,
sin dependencias de *frameworks* de terceros. Semillas fijas; protocolos
congelados antes de observar los datos.

## Entorno

```
python >= 3.11
pip install -r requirements.txt          # torch (cu128 para Blackwell), numpy, scipy, matplotlib
```
Ejecutar SIEMPRE desde la raíz del paquete con `PYTHONPATH=.` (en Windows:
`$env:PYTHONPATH="."`). BF16 en GPU, FP32 en CPU (fallback automático).

## Mapa código → resultado del paper

| Sección / resultado | Script |
|---|---|
| Campo PDE, familia, colocación giroscópica, solver IMEX, certificados | `model/hbp.py` (`stability_penalty`, `certificate_spectral_radius`) |
| Sustrato de flujo 2D (Navier–Stokes) + verificación física | `model/flow2d.py`, `_check_flow2d.py` |
| §5 Benchmark v3 (efecto de 2º orden OOD, física aprendible; fix BF16) | `benchmark_v3.py` → `BENCH_DIR=results_benchmark_v3 python benchmark_report_v2.py` |
| §5 Zoo KdV (dispersión ni aporta ni daña) | `benchmark_zoo.py` |
| §6 Null (1) físicas forzadas / (2) gateada | `_alpha_scan.py`, `_pde_study.py` (+ `_pde_aggregate.py`) |
| §6 Null (3) tareas duales de régimen | `_dual_pilot.py`, `_twin_pilot.py` |
| §6 Null (4a) no-localidad elíptica (A/B) | `benchmark_elliptic.py` → `_elliptic_report.py` |
| §6 Null (4b) sustrato de flujo — kill-gates | `_gates_flowroute.py` |
| Test de regresión: el campo es load-bearing | `_audit_hbp.py` |
| Figuras del paper | `paper/make_figures.py` |

Verificaciones de sanidad rápidas (CPU, segundos):
`_check_elliptic.py`, `_check_kdv.py`, `_check_flow2d.py`, `_check_fixes.py`.

## Reproducir de cero (orden sugerido)

```bash
# 1) sanity: el HBP es load-bearing y los certificados son consistentes
PYTHONPATH=. python _audit_hbp.py
PYTHONPATH=. python _check_flow2d.py

# 2) headline: benchmark v3 pre-registrado (90 celdas, ~5 h en 1 GPU)
PYTHONPATH=. python benchmark_v3.py
PYTHONPATH=. BENCH_DIR=results_benchmark_v3 python benchmark_report_v2.py

# 3) el null exhaustivo
PYTHONPATH=. python _alpha_scan.py           # físicas forzadas (4 tareas)
PYTHONPATH=. python benchmark_elliptic.py    # no-localidad (A/B, ~4 h)
PYTHONPATH=. python _elliptic_report.py
PYTHONPATH=. python _gates_flowroute.py      # sustrato de flujo (kill-gates)

# 4) figuras
cd paper && BENCH_DIR=results_benchmark_v3 python make_figures.py
```

Notas: `benchmark_*.py` escriben un JSON atómico por celda en `results_*` y
**saltan las celdas ya hechas** (resumibles). Los parámetros físicos se anclan a
FP32 (`pin_fp32`) tras el cast a BF16; sin ese anclaje quedan congelados en su
init (fallo BF16 documentado en el paper, §5.1).

## Compilar el paper

```bash
cd paper && pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```
Para el envío a TMLR, sustituir en `paper.tex` el bloque `\documentclass[11pt]{article}`
por `\documentclass{article}\usepackage{tmlr}` (ver nota en la cabecera del .tex).
