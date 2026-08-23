# F3b — Gates GS1-GS3: el entorno v1 NO certifica (y eso es el sistema funcionando)

> 2026-08-02. Estado: cableado del modelo COMPLETO y verificado (7/7,
> tests_f3b_wiring.py, git 3478fbc); entorno + perfil real + escalera de
> oráculos construidos (f3b_env / f3b_probe_acc / f3b_gates); ronda 1 de
> calibración consumida. **Veredicto: GS1b FALLA con techo ESTRUCTURAL —
> el confirmatorio NO se lanza sobre este entorno.** Cero horas de GPU de
> entrenamiento gastadas.

## Lo que pasó, en orden

1. **Perfil real** (6 ckpts F3a, forward_forced, rejilla K×n, 256/K):
   d_ref(K) 2→17 en K∈[6..24], concordante entre variantes (±1-2 ticks →
   referencia independiente de brazo ✓). acc(n=1|K) obliga a K_min≥13
   (el suelo del prereg §2: K=8 daba 0.564 de acierto con UN tick).
2. **Ronda 1 de calibración** (cortafuegos §10, diales cerrados, motivo
   commiteado en SessionSpec): k_easy (8,16)→(13,18), k_hard (12,22)→(16,22),
   B_total 60→56 (=0.75× demanda d_ref 74.1). Resultado: mordida 38.8%→99.8%
   ✓, GS2/GS3 ✓ formales… pero headroom perceptivo +0.017 (<+0.08) → GS1b ✗.
3. **Escaneo del techo** (f3b_scan_diales.py, solo políticas de oráculo,
   1500 sesiones/celda, 14 configuraciones: B∈[35..56] × stake∈{1,2,4} ×
   p_switch∈{0.20,0.33,0.45}):
   - headroom perceptivo máximo del ESPACIO ENTERO: **+0.036** (B=40, ×2).
     El umbral +0.08 es INALCANZABLE en esta familia de entornos.
   - **bayes−reactivo ≤ +0.001 en TODAS las celdas**; la permanencia no
     mueve nada (p0.20 ≡ p0.45 al cuarto decimal).

## El hallazgo (negativo-de-diseño, con mecanismo)

**En este entorno la escala lenta NO es load-bearing por construcción**, y el
mecanismo es identificable: (a) la asignación de cómputo es REVERSIBLE — el
gobernador puede reaccionar a la dificultad de la instancia corriente (que
observa vía longitud/interocepción) y re-planificar cada instancia; (b) las
curvas acc(n,K) reales son cóncavas y suaves — re-asignar en el margen cuesta
poco; (c) la diferencia de demanda entre regímenes (10.8 vs 13.9 ticks) es
pequeña frente a la granularidad del presupuesto. Consecuencia: el filtro
bayesiano perfecto sobre TODA la historia gana ≤0.001 sobre una política
reactiva-estacionaria. Anticipar no paga porque nada es irreversible.

Esto conecta con la tesis del arco: el campo/plano solo puede aportar donde
hay estructura que su física pueda gobernar. Aquí los gates certificaron la
AUSENCIA de esa estructura ANTES del confirmatorio — un D1 nulo aquí no
habría cerrado la hipótesis multiescala: habría medido el techo del entorno.

## Qué NO se hizo (disciplina)

- No se re-declaró el umbral +0.08 tras verlo fallar (sería post-hoc).
- No se tocaron diales fuera del conjunto cerrado (E, correlación
  stake×régimen: NO son diales — serían enmienda de diseño).
- Quedan 2 rondas de calibración, pero el escaneo demuestra que ninguna
  combinación del dial-set alcanza el umbral: gastarlas sería teatro.

## El fork (decisión de Curro)

- **A. Enmienda de diseño del entorno** (fechada, pre-data): crear VALOR DE
  ANTICIPACIÓN real vía irreversibilidad. Candidatos, de menos a más
  invasivo: (1) correlación stake×régimen (el régimen difícil trae stakes
  altos con prob q: anticipar la tormenta obliga a ahorrar YA; el propio
  panel dejó apuntada esta opción); (2) E=12 (horizonte y transiciones);
  (3) compromiso irreversible explícito (presupuesto por bloques sin
  reembolso). Tras la enmienda: re-scan del techo, GS1b re-anclado al techo
  medido (p.ej. IC-inf ≥ ⅔·techo y ≥4×MDD), y solo entonces gates →
  confirmatorio.
- **B. Registrar el negativo-de-diseño y parar F3b-sesión**: publicable
  dentro del arco («la anticipación multiescala exige irreversibilidad»),
  sin más GPU.
- **C. Desacoplar el doble objetivo**: la mitad B (re-cableado / hipótesis
  M3) NO depende de la sesión — puede correr como F3a-R (re-entreno F3a con
  gates por-instancia, ~12 celdas ≈ 1 noche, prereg propio corto) mientras
  el entorno de sesión se rediseña con A.

**Recomendación**: B (registrar) + A con (1)+(2) como única enmienda, y C en
paralelo esta noche si se quiere HB respondida ya. A y C no compiten por
GPU: C es 1 noche; A necesita re-scan (CPU) antes de tocar GPU.

---

## RESOLUCIÓN DEL FORK (2026-08-02, ratificado por Curro: B + A + C)

### Rama A — EJECUTADA Y CERRADA EN NEGATIVO ROBUSTO (0 GPU)

Enmienda implementada (stake_mode="regime_corr" en f3b_env, filtro bayes con
emisión de stake y previsión dependiente del régimen en f3b_gates) y re-scan
del techo con 17 configuraciones nuevas: E∈{6,8,12} × q_hard∈{⅓,½,⅔} ×
factor-B∈{0.65,0.75,0.85} + régimen difícil ANCHO (k_hard=(13,24), máxima
separación de demanda con solape identificativo). Resultado:

- **bayes − reactivo-estacionario ≤ +0.001 EN TODAS las celdas** (a menudo
  ligeramente negativo). bayes − última-K crece con E (hasta +0.012) pero el
  null estacionario lo captura todo.
- Nota estructural clave: la escalera de oráculos evalúa asignaciones
  PRE-COMPROMETIDAS (asignar = ejecutar, sin reactividad intra-instancia) —
  es decir, el escaneo YA cubre el candidato (3) (irreversibilidad máxima).
  Ni siquiera bajo pre-compromiso total la anticipación paga.

**Mecanismo del techo, identificado**: (i) el perfil de valor acc(n,K) real
es cóncavo y SUAVE (sin acantilados: pagar 2-3 ticks de más o de menos cuesta
~0.01-0.03 de acc); (ii) la separación de demanda entre regímenes es pequeña
(10.8 vs 13.9-13.3 ticks/instancia) porque d_ref(K) es somero (8→17 en todo
K∈[13..24]); (iii) los regímenes mezclan rápido (permanencia ≪ E) → el prior
estacionario es casi óptimo. Conclusión: **en la familia S₅-composición con
estos perfiles, NINGUNA estructura de sesión construible con los ingredientes
declarados hace la escala lenta load-bearing**. El hallazgo se endurece: «la
anticipación multiescala exige un perfil de valor con acantilados
(all-or-nothing) o separación fuerte de demanda — esta tarea no los ofrece».

**F3b-sesión se CIERRA sin confirmatorio** (rama declarada del prereg §12:
gates no certifican tras calibración+enmienda; 0 h de GPU de entrenamiento
quemadas en sesión). Un test futuro de la escala lenta (N1b del roadmap)
exigirá una familia de tarea nueva con valor tipo-acantilado (p.ej.
verificación todo-o-nada) — hipótesis nueva, prereg nuevo, NO enmienda.

### Rama C — EN CURSO (esta noche)

PREREG_F3A_R.md declarado y 24 celdas lanzadas (miura_mhbp_pi1 = contenido
congelado tras tick 1; miura_mhbp_noc = sin vías de contenido en train;
6 seeds × {indist, ood}; 2 workers). Espía de cableado verificado: por-tick
fluctúa (10 valores/10 ticks), pi1 congela (1 valor, dependiente del estado),
noc neutro exacto. Batería M3 on-policy al terminar.
