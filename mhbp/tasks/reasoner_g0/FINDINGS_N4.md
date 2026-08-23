# FINDINGS N4 — el campo como acumulador de evidencia: CERRADO EN G1
### 2026-08-09. Coste total: ~30 min de GPU trivial + CPU. Cero entrenamiento.

## 1. Qué se preguntó

DISENO_N4: la última hipótesis viva para hacer el campo load-bearing —
darle (a) oídos (el stream posterior por tick del reasoner) y (b) un
lector explícito, con el trabajo genuino de ACUMULAR EVIDENCIA temporal
(2º orden = detector resonante). Kill-gate G1: ¿el stream contiene
información sobre el éxito más allá del último tick?

## 2. Instrumento CERTIFICADO (la lección del capítulo LLM, aplicada antes)

El primer G1 dio nulo (+0.0008) con un instrumento sesgado: el control
positivo (señal temporal conocida inyectada en la misma geometría) reveló
que el probe NO la veía. Dos artefactos, ambos empujando hacia el nulo:
l2 fijo → impuesto de sobreajuste ~0.018 AUC contra el probe de más
dimensiones; y AUC agrupada sobre scores de folds con l2 distintos
(escalas incomparables). Fix: CV anidada por probe + AUC por fold.
Certificación final: carga 0 → −0.001±0.001 (insesgado); carga 0.20 →
+0.026 [+0.020, +0.031] (potencia a la escala del umbral 0.03).

## 3. Veredicto (12 checkpoints congelados, 1024 inst/ckpt, pareado)

| celda | ΔAUC (stream − último tick) |
|---|---|
| **PRIMARIA t=8, ¿resoluble?** | **+0.0007, IC95 [−0.0065, +0.0079]**, 6/12 >0 → **NO PASA** |
| todas las secundarias (t∈{2,4,16,24} × {A,B}) | entre −0.001 y +0.007, ninguna cerca de 0.03 |
| G0 estructura temporal (detrendado) | autocorr lag2-6 ≈ +0.05; picos no-DC: 0% (speed), 17% (margen) |

## 4. Lectura mecanicista (y por qué esto CIERRA la línea, no la aplaza)

**El último tick es estadístico suficiente del stream porque la
recurrencia YA ES el acumulador.** El estado h_t del reasoner integra su
propia historia; los observables del tick t ya resumen la trayectoria
entera. El nicho «integrador temporal de evidencia» está ocupado por
construcción — un segundo acumulador externo no tiene nada que añadir.
Y el G0 lo remata por la vía independiente: quitada la rampa, el stream
es esencialmente ruido blanco — no hay estructura oscilatoria que un
filtro resonante de 2º orden pudiera explotar y uno de 1º no (G3 se
queda sin materia prima además de sin motivo).

## 5. El estado final del campo homeostático (respuesta a «¿qué le falta?»)

Con esto, TODOS los empleos imaginados para el campo están medidos:

| empleo propuesto | veredicto | quién hace ese trabajo mejor |
|---|---|---|
| llevar contenido | nulo (M, mHBP F2) | la trayectoria del reasoner |
| decidir vía gradiente | nulo (N2/N3) | el allocator explícito |
| plano ejecutivo | negativo (F2X) | designer-con-regla |
| acumular evidencia posterior | **nulo (N4, este doc)** | la propia recurrencia |
| controlar cómputo | **POSITIVO pequeño y replicado** (v3) | — es su empleo real |

La respuesta a la pregunta no es una pieza que añadir: es que **el
sistema ya contiene, en componentes dedicados, todo lo que el campo
podría aportar — salvo lo que ya aporta** (modulación de cómputo, con
efecto de estructura de 2º orden replicado). El campo «funciona un
poquito» porque ese poquito ES su nicho.

## 6. Notas de proceso

- El panel de diseño N4 nunca llegó a adjudicarse (límite de sesión,
  4/4 lentes caídas); el kill-gate pre-declarado resolvió primero y a
  coste ~0 — que es el orden correcto cuando el gate es más barato que
  el panel. El papel de auditoría del instrumento lo cubrió el control
  positivo (§2), que cazó dos artefactos.
- G1 usó SOLO artefactos congelados (checkpoints + etiquetas de la sonda
  N3, con wiring-test de identidad de instancias): cero entrenamiento,
  réplica en 12 checkpoints.
