# PREREG V4 — des-confusión del ORDEN y baseline aprendido de interfaz
# equiparada (las dos condiciones de aceptación del revisor empírico)
### v1, 2026-08-17. Pre-datos. Seeds frescas 20-39, jamás usadas en v2/v3.

## 1. Qué ataca

El efecto v3 («el 2º orden estabiliza la asignación de cómputo OOD») tiene
dos confounds señalados por la revisión: (a) hbp_first corría con TOPES
RECORTADOS (c_max 0.4 vs 0.7; ω₀_max 0.45 vs 1.8) impuestos por el Euler
explícito acoplado a ζ — el contraste orden-vs-orden estaba contaminado
por capacidad; (b) no existía un baseline APRENDIDO de interfaz
equiparada — «la política aprendida» era gating_wm, que no recibe la
interocepción del campo.

## 2. Brazos (3) × generadores (2) × seeds 20-39 (n=20/celda-familia)

- **hbp_full**: incumbente (2º orden, topes estándar).
- **hbp_first_eq** (NUEVO): 1er orden con topes EQUIPARADOS (c_max=0.7,
  ω₀ hasta 1.8) vía solver IMEX implícito con tasa propia γ_diff
  (incondicionalmente estable → los topes ya no dependen de ζ). Única
  diferencia con hbp_full: el orden y su ζ de régimen.
- **hbp_gru** (NUEVO): GRUCell por nodo (pesos compartidos) que sustituye
  SOLO al integrador físico; interocepción, forzamiento externo y cabezas
  de modulación idénticas (paridad de interfaz por construcción; conteo
  de parámetros por variante se reporta).

Protocolo idéntico a v3: OOD (train_max_writes=12, eval hasta 24),
MAX_STEPS=2500, N_MAX=24, pin_fp32, generadores {adjacent, cycle_transp}.
Métrica primaria: corr_OOD(K, E[n_iter]) del diagnóstico de cómputo;
secundaria: final_acc largo (no-inferioridad, margen 0.02).

## 3. Hipótesis y ramas (pareado por seed; IC-t n=20; Holm-2 entre
## generadores dentro de cada hipótesis; α=0.05)

- **H-ORDEN**: corr_OOD(hbp_full) − corr_OOD(hbp_first_eq) > 0 en ambos
  generadores tras Holm-2. Potencia: con la d≈0.85 de v3, n=20 da >90% a
  α=0.025. Ramas: ✓✓ = el orden es causal con topes iguales (el claim del
  paper se ASCIENDE); ✓✗ = direccional, se reporta acotado por generador;
  ✗✗ = el efecto v3 ERA capacidad, no orden → corrección mayor del
  companion (el abstract cambia).
- **H-GRU**: corr_OOD(hbp_full) − corr_OOD(hbp_gru) > 0 ídem. Ramas:
  ✓✓ = la física de 2º orden supera al aprendido equiparado (claim
  nuevo); ✗ en cualquiera = el GRU iguala → «estructura sí» se REBAJA a
  «estructura o cualquier recurrencia con esta interfaz» (se publica
  así; es el resultado honesto que el baseline existe para encontrar).
- Accuracy: no-inferioridad de hbp_full vs cada brazo (margen 0.02);
  si un brazo NUEVO gana en accuracy además de en corr, se reporta como
  hallazgo.

## 4. Operativa

Runner benchmark_v4.py REANUDABLE (skip por fichero con verificación de
cfg, idéntico patrón v3); bucle seed-EXTERNO (cortes parciales dejan
n balanceado entre brazos); resultados en results_benchmark_v4/.
Presupuesto: 120 runs × ~t_smoke; cronómetro en el smoke de cableado
(2 celdas cortas) ANTES de lanzar; si la proyección excede 5 días se
re-scopea a seeds 20-31 (n=12, potencia ~80%) DECLARÁNDOLO aquí antes
de mirar ningún resultado.

## 5. Análisis

benchmark_report_v4 (adaptación del v2/v3): medias pareadas por seed,
IC-t, Holm-2 por hipótesis; tabla de parámetros por variante; los JSONs
de v3 (seeds 10-19) NO entran en el confirmatorio (protocolo distinto en
hbp_first) — solo como contexto descriptivo.
