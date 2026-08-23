# G0 — Hallazgos: el reasoner SÍ porta la computación; lo roto es la interfaz de parada

> Veredicto formal contra umbrales pre-registrados: **FAIL 0/12** (REPORT_G0.md).
> Pero la estructura del fallo refuta mi hipótesis previa y localiza el
> problema en otro sitio. 30 modelos (protocolo v3) + 12 celdas de
> instrumentos. Fecha: 2026-07-31. Umbrales NO movidos; los diagnósticos de
> operacionalización van etiquetados como post-hoc.

## 1. El resultado que me equivocó (M2, swap de estado completo)

Predicción previa (ARCO_MHBP §5): el reasoner "asienta" — re-lee el input cada
tick y el estado no porta la composición; esperaba dominancia de
"revierte-gratis". **Refutada limpiamente**:

- **cycle_transp**: sigue-al-donante 0.83–0.92, revierte-gratis 0.00, en los
  3 seeds y ambas variantes. Trasplantas el estado en t*∈{3,6,9} y la salida
  sigue al donante: **el estado ES el portador de la computación**.
- **adjacent**: sigue-al-donante 0.33–0.53 con "ninguna" 0.47–0.66 — pero
  normalizando por la accuracy de salida (0.47–0.59), el ratio
  sigue/acc ≈ 0.77–0.90: condicionado a que el modelo acierte, sigue al
  donante. El "ninguna" es error de tarea, no no-portabilidad. *(Fallo de
  operacionalización #1, post-hoc: el umbral M2 (0.6) no normalizaba por la
  accuracy base — en tareas con error alto castiga el error, no el mecanismo.)*
- revierte-gratis ≈ 0 EN TODO: el reasoner NO re-deriva del input. La
  trayectoria es causalmente usada. dn<0 sistemático: los estados
  trasplantados aceleran la parada (curioso, no crítico).

## 2. Cristalización (M1) — de libro en cycle, con artefacto en adjacent

- **cycle_transp**: F_full 0.5→0.85-0.89, trend Spearman +0.98..+1.00,
  F_final/acc ≈ 1.0. Cristalización gradual textbook. M3 = +0.78..+0.85: la
  fidelidad de trayectoria predice fuertemente el acierto OOD.
- **adjacent**: hbp_full COLAPSA tarde (0.47→0.03-0.20). Diagnóstico post-hoc
  (#2): la mezcla PonderNet concentra masa temprano; los ticks POSTERIORES a
  la parada efectiva son estados off-policy que ninguna pérdida moldeó — el
  registro completo a 24 ticks mide la deriva post-parada. El trend M1 debería
  computarse en el rango on-policy [1, n_parada]. (Con acc 0.57, la
  cristalización en adjacent es además genuinamente más débil.)

## 3. El hallazgo central: la interfaz de parada está rota (I)

- **cycle**: AUC(p_stop, corrección) = 0.13–0.27 — **INVERTIDO**. Mecanismo
  (post-hoc, #3): las muestras difíciles llegan al techo N_max, donde λ:=1
  fuerza la parada con toda la masa restante → p_stop alto exactamente en las
  que fallan. La señal de parada es un **proxy de agotamiento de presupuesto,
  no un estimador de completitud**.
- **adjacent**: AUC 0.53–0.71 (mediocre pero no invertido; el halting se
  reparte sin saturar el techo).

**Consecuencia para la Fase 3**: la interfaz donde el mHBP se enchufa (el
halting — su único positivo replicado, v3) no está calibrada para lo que el
gobernador necesita gobernar. Integrar ahora sería estabilizar una señal que
mide presupuesto, no progreso.

## 4. T y P

- T (irreducibilidad serial): adjacent FALLA (gaps 0.06–0.11 < 0.15: las
  transposiciones locales admiten atajo parcial de profundidad fija); cycle
  la cruza en hbp_full (0.149/0.176/0.183, 2/3) y roza en gating_wm.
- P (política de profundidad extrapola): hbp_full×cycle 3/3 (+0.13/+0.23/+0.22)
  — v3 replica una vez más; adjacent 2/3.

## 5. Lectura y camino (remedios pre-asignados en PREREG_G0)

El par (reasoner, **cycle_transp**) está en régimen de razonamiento en su
MECANISMO (M sólido: cristalización + estado portador + ligadura OOD) y en su
POLÍTICA (P), pero su INTERFAZ (I) es inválida — y adjacent queda descartada
por T. El remedio pre-registrado para I-fail aplica:

1. **Re-diseñar/calibrar la cabeza de halting para completitud** (candidatos:
   target auxiliar de corrección del decode del tick; calibración excluyendo
   el techo; presupuesto N_max mayor para descomprimir el ceiling) → G0.1
   re-testea SOLO I sobre cycle_transp (barato).
2. **G0.1 además corrige las operacionalizaciones** (enmienda pre-declarada):
   M1 on-policy, M2 normalizado por accuracy. Con ellas, M en cycle pasa
   holgado con los datos ya recogidos (se reporta como sensibilidad, no como
   veredicto: los umbrales originales no se mueven retroactivamente).
3. La Fase 3 (integración mHBP↔reasoner) queda gateada a G0.1-I PASS en
   cycle_transp.

**El fruto de G0**: tres meses de nulos NO se explicaban por "asentamiento"
(hipótesis refutada) — el reasoner razona; se explicaban, al menos en parte,
por gobernar una interfaz que mide presupuesto en vez de progreso.
