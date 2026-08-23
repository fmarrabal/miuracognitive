# N2 — Hallazgos de la Etapa 1: doble negativo con mecanismo (y una rama
# que el prereg no vio venir)

> 2026-08-04/05. 36 celdas confirmatorias (3 brazos × seeds 3-8 × 2 runs,
> celda certificada s_alto=8, p_hi=0.15, β=0.02, E[n̄]≈8). Análisis
> pre-declarado en n2_e1_veredicto.json. Etapa 2 NO corre (gatekeeping).

## Resultados (medias de 6 seeds × 2 runs; eval común congelada m=4096)

| brazo | payoff_norm | acc | acc_ALTO | E[n] alto/bajo | corr(E[n],stake\|K) | AUC_V |
|---|---|---|---|---|---|---|
| endo (V̂→risk) | 0.781 | 0.777 | 0.786 | 7.97 / 8.03 | **−0.010** | **0.951** |
| endo_noval (V̂ suelto) | 0.781 | 0.777 | 0.786 | 8.04 / 8.09 | −0.007 | 0.911 |
| blind_flat (pesos planos) | **0.926** | **0.921** | **0.929** | 5.42 / 5.48 | −0.009 | 0.581 |

- **C1′ (primario): NULO EXACTO** — Δ=+0.0000, sd=0.011, t=0.00, p=0.498
  (Wilcoxon 0.578). El acoplamiento valor→gobernador no añade NADA sobre
  las mismas consecuencias en la pérdida. endo ≡ endo_noval en TODAS las
  métricas a 3 decimales.
- **C3: no testeado** (gatekeeping — condicionado a C1′).
- **RAMA NO DECLARADA (el hallazgo grande)**: blind_flat domina a los dos
  brazos ponderados por **+0.147** de payoff (sd 0.020) — y lo hace
  **incluso en las instancias de stake alto** (0.929 vs 0.786) usando un
  33% menos de cómputo (5.4 vs 8.0 ticks).
- **Cero routing en todos los brazos**: corr(E[n], stake|K) ≈ −0.01;
  E[n]_alto = E[n]_bajo al segundo decimal.

## Los tres mecanismos (medidos, no conjeturados)

1. **La CE ponderada se auto-derrota**: pesos {1,8} con p_hi=0.15 → ESS
   efectivo = 0.40. La ponderación destruye al solver (−0.14 de acc global)
   más de lo que el énfasis aporta (+0.01 de sesgo alto-vs-bajo). La
   respuesta racional al pago 8× resultó ser COMPETENCIA GENERAL: como la
   accuracy transfiere a todas las instancias, aprenderlo todo bien domina
   a enfatizar lo caro. blind_flat gana en lo que endo "cuida" (0.929 vs
   0.786 en acc_alto).
2. **El canal de valor estuvo vivo y el gobernador quedó sordo**: AUC_V
   0.951 (el sensor identifica el stake casi perfectamente), el canal entra
   detached al campo risk (cableado verificado T8-T12)… y la política de
   cómputo no lo usa (corr −0.01; endo ≡ endo_noval). El plano integró la
   señal en su estado; ninguna vía la convirtió en ticks.
3. **Por qué tampoco emergió del incentivo directo**: el halting lee el
   estado pooled — y el entrenamiento SUPRIME del estado la información
   tarea-irrelevante (medido en VG4: probe 0.79 desde el estado vs 1.00
   desde embeddings). Aunque la pérdida ponderada incentivaba 8× pensar más
   en los altos, el halting no puede condicionar en un stake que su feature
   no contiene. El único módulo con acceso (V̂, vía embeddings) alimentaba
   una vía de autoridad acotada cuyo gradiente nunca encontró el headroom
   (+0.087 del oráculo VG1, que sigue AHÍ, inexplotado). Consistente con
   F3b-gates: Δacc/Δn es suave — la señal de gradiente para el routing es
   débil incluso cuando existe.

## Veredicto por ramas

- Rama aplicable más cercana: «C1′✗ → negativo informativo, sin re-tune»
  — pero con la corrección de la rama no declarada: no es que «las
  consecuencias en la pérdida basten»; es que **DAÑAN**. HN2 no soportada;
  nada se adopta; N3 NO hereda el acoplamiento V̂→risk.
- Desviaciones declaradas: (a) celdas ood-endo (batería M) OMITIDAS — el
  guardarraíl M protege adopciones y aquí no se adopta nada (ahorro de ~6 h
  sobre presupuesto); (b) nota descriptiva: AUC_V de blind_flat = 0.58
  (V̂ aprende p(correcto)·s̄; la elevación leve sobre 0.5 queda registrada;
  el chequeo formal de fuga fue blind_shuffle en el piloto: 0.49-0.50 ✓).
- Etapa 2 no corre por gatekeeping. **Candidato mecanicista pendiente**
  (opcional, exploratorio, ~3-6 h): gating_endo — stakê → bias DIRECTO del
  halting (esquiva tanto la supresión del tronco como el plano); es la
  única vía no probada con acceso limpio señal→decisión. Decisión de Curro.

## Lectura para el programa (la serie completa de integración)

Tres negativos limpios con el mismo patrón: F2b (gobernador inerte con
manipulación rescatada), F3b-gates (anticipación sin valor explotable),
N2 (valor percibido pero no convertido en política). **El sustrato
certificado modula estabilidad y completitud, pero las decisiones de
SEGUNDO ORDEN (qué importa, qué viene, cuánto asignar) no emergen por
gradiente sobre PonderNet+β**: exigen mecanismo EXPLÍCITO de decisión
(política de asignación amortizada, bias directo tipo gating_endo, o
asignación como acción con crédito propio). Ese es el insumo de diseño
para N3 — y el capítulo N2 queda como el estudio que lo estableció con
sensor perfecto, canal verificado y headroom medido: todo estaba menos el
mecanismo.
