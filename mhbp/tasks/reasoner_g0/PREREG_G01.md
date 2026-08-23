# Pre-registro — G0.1: reparar la interfaz de parada (I) en cycle_transp

> Enmienda de G0 (remedio pre-asignado al I-fail en PREREG_G0). Declarado
> 2026-07-31 ANTES de correr nada. Alcance: SOLO la interfaz I sobre
> cycle_transp (T/M/P ya certificados ahí); adjacent queda fuera (T-fail).

## Diagnóstico que motiva la reparación (de FINDINGS_G0)

p_halt es un proxy de agotamiento de presupuesto: (i) su única señal explícita
de entrenamiento es la regularización "para pronto" + la tarea a través de la
mezcla; (ii) las muestras difíciles llegan al techo N_max donde λ:=1 concentra
la masa restante → AUC(p_stop, corrección) INVERTIDO (0.13–0.27).

## Corrección de métrica (I'), declarada ANTES de medir

El artefacto del techo contamina p_stop=max masa incluso con un halting
calibrado (las muestras sin solución también acumulan masa en el cap forzado).
La métrica de interfaz correcta es la mejor AFIRMACIÓN DE COMPLETITUD del
halting antes de que el presupuesto fuerce la parada:

    I' = AUC( max_{n < N_max} λ_n , corrección final )

Se reporta TAMBIÉN la I original (continuidad con G0); el criterio de éxito
usa I'. I' se computa retroactivamente sobre los checkpoints de G0 como
baseline (λ_n se reconstruye de p_n y los remainders).

## Etapa a — ¿la información de completitud YA está en las features? (barato)

Sobre los checkpoints INDIST de G0 (cycle_transp, 2 variantes × 3 seeds),
probes post-hoc de completitud por tick: predecir 1[decode_n correcto] (el
argmax del lm_head en la posición de respuesta en el tick n, contra el target)
desde DOS conjuntos de features del tick:
  (F1) las del halting actual: pooled global (+ frac_valid)
  (F2) pooled + estado en la POSICIÓN ARROW (la respuesta-en-curso, donde
       M1 mostró F_full≈0.89 de decodabilidad)
Métrica: AUC de la probe (max sobre n<N_max de su score) vs corrección final.
Decisión pre-declarada:
- AUC(F1) ≥ 0.65 → la señal existe en las features actuales: G0.1b solo
  necesita el TARGET auxiliar (sin cambio arquitectónico).
- AUC(F1) < 0.65 ≤ AUC(F2) → el pooling diluye la completitud: G0.1b usa
  además la feature de posición ARROW en el halting.
- ambas < 0.65 → la completitud no está representada: problema más profundo,
  re-diseño mayor (se documenta y se replantea).

## Etapa b — el fix: entrenar el halting PARA completitud

Retrain de cycle_transp (hbp_full, gating_wm × seeds 0,1,2 × {indist, ood};
12 modelos, protocolo v3 idéntico salvo):
1. **Pérdida auxiliar de completitud** (flag nuevo, inerte por defecto):
   L_halt_aux = mean_n BCE( λ_n , 1[decode_n correcto] ), β_aux = 0.05.
   El target se computa con no_grad del propio lm_head por tick (supervisión
   del INTERFAZ; sin fuga en test: en inferencia todo es aprendido).
2. **Features del halting según el resultado de la etapa a** (F1 o F2).
3. N_max=24 SIN cambios (comparabilidad de P con v3/G0).

## Criterios de éxito (todos pre-declarados)

- **I' ≥ 0.65 en ≥2/3 seeds** para al menos una variante (cycle_transp).
- **Guardarraíl de tarea**: accuracy largo (indist) ≥ baseline G0 − 0.05.
- **Guardarraíl de política**: P (corr_K_niter_ood, ood ckpt) ≥ 0.10 en
  hbp_full ≥2/3 seeds (no comprar interfaz vendiendo la política).
- PASS ⇒ la Fase 3 procede sobre (cycle_transp, variante que pase) con la
  interfaz reparada. FAIL ⇒ documentar y replantear la interfaz (no integrar).

## Sensibilidad (etiquetada, sin tocar veredictos de G0)

Con los datos de G0 ya recogidos se reportan M1-on-policy (trend en
[1, n_parada]) y M2 normalizado por accuracy — correcciones de
operacionalización identificadas post-hoc en FINDINGS_G0. Los umbrales y
veredictos formales de G0 NO se recalculan.
