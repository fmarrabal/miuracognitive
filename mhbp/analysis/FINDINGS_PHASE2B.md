# Fase 2b — Registro del resultado: rescate por manipulación + gobernador inerte

> Confirmatorio pre-registrado (PREREG_PHASE2B.md v1-v2), 60 celdas nuevas
> (mhbp_gov/react/gru_gov × 20, steps 500) + reutilización declarada de las
> corridas F2 (mhbp/gru/mlp; eval determinista compartida). 184/184 en disco,
> 0 errores. Fecha: 2026-07-29. Datos: results_phase2/; REPORT_PHASE2B.md.

## Resultado en una frase

**Quitar al campo el rol de FUENTE de acciones elimina por completo la
catástrofe OOD de la Fase 2 (confirmación por manipulación del mecanismo);
como MODULADOR de ganancias, el campo resulta INERTE: nadie —ni el campo ni
un GRU equiparado— explota ganancias variables en este entorno, y la presencia
del campo en el lazo cuesta una interferencia de entrenamiento pequeña.**

## Números clave

- **D1 (gov vs mhbp)**: Δ=−7.57, t=−19.4, dz=−4.33, Holm SÍ — J_OOD 10.11→2.32.
  Mecanismos: plan(budget_hi) 34.5→2.30 (15×; umbral ≤1.0 solo PARCIAL),
  hard(e4) 21.1→0.68 (✓, el mejor de la tabla), settling 2.52→0.28 ventanas
  (el más rápido de TODOS los controladores, mlp incluido).
- **D2b (gov vs gru_gov, física aislada)**: Δ=+0.27, p=0.003, dz=0.75, Holm SÍ,
  dirección gov PEOR. **D2a** (vs react): +0.27 (p=0.013, fuera de Holm).
- **Intervención g≡1** (pareada por episodio): Δ=+0.0000 EXACTO en mhbp_gov y
  gru_gov — las ganancias aprendidas son INERTES (g≈1). El daño de D2a/D2b no
  es "modulación mala": es interferencia del campo durante el entrenamiento
  conjunto (gru_gov inerte no molesta: 2.050 ≈ react 2.052).
- **D3 (vs gru F2)**: +0.11, n.s. — gov queda en la liga de la memoria genérica.
- **E1 (gate del reactivo)**: react +0.15 peor que mlp (dz=1.52, significativo)
  — la referencia reactiva es levemente débil; no afecta al orden de magnitud
  de D1 pero se registra (arquitectura sin bias y cabeza directa).

## Interpretación (las ramas pre-declaradas que se cumplieron)

- Rama "D1 confirmada + D2 nula/negativa" del prereg: **gobernador inerte con
  coste**. La mitad NEGATIVA de la tesis governor queda demostrada por
  manipulación (el rol de fuente era el fallo). La mitad CONSTRUCTIVA (la
  física del campo aporta como moduladora) NO encuentra soporte en el SMA:
  el óptimo del entorno no requiere ganancias variables (hasta el GRU converge
  a g≈1), así que el experimento no puede distinguir "el campo no sabe" de
  "aquí no hay nada que modular". Ese confound de entorno se declara.
- Detalle no trivial a favor del campo (descriptivo): mhbp_gov logra el MEJOR
  settling tras el escalón (0.28 ventanas) y el mejor hard(e4) — compatible
  con el positivo v3 (el campo estabiliza transitorios) aunque no mueva J.

## Conexión con el programa

Fuente → catastrófico (F2). Modulador → rescata pero inerte (F2b). El único
positivo persistente del campo en todo el programa sigue siendo la
ESTABILIZACIÓN (v3: adaptividad de halting; aquí: settling/hard del escalón).
"Modula, no piensa" queda así: cuando modula, como mínimo no rompe — y
estabiliza transitorios; cuando piensa (decide niveles), rompe OOD.

## Qué NO se corrió (pendiente posible, post-hoc desde checkpoints)

Los .ckpt.pt de las 60 celdas permiten: descomposición de varianza del logit
(gate de mecanismo pre-registrado — con g inerte el resultado está anticipado),
g-congelada-a-media y g-barajada (con Δ=0.0000 de g≡1, ambas son moot). Un
entorno donde las ganancias variables SÍ paguen (demanda de conmutación de
régimen tick-alineada) queda como diseño futuro si se quiere testear la mitad
constructiva en condiciones no degeneradas.
