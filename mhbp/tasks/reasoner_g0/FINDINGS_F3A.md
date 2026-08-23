# F3a — Hallazgos: no-inferioridad en gobierno, con UN toque medible al mecanismo

> **⚠️ RECLASIFICACIÓN (2026-08-03, FINDINGS_F3A_R.md)**: el hallazgo M3
> on-policy de este documento (−0.055, p=0.011) **NO REPLICA** con
> comparadores contemporáneos (toque re-estimado −0.011, IC95 [−0.068,
> +0.045]): era compatible con el ruido de run por celda (σ≈0.04-0.05) que
> este diseño (una run por celda) no estimaba. El trade-off declarado queda
> RETIRADO; la interpretación de la lesión («en los pesos») queda retirada
> con él. Lo que sigue vale como registro histórico y como caso de estudio
> metodológico. Sobreviven: no-inferioridad de gobierno, M1/M2 intactos.

> Veredicto formal (REPORT_F3A.md, ramas pre-declaradas): **FAIL de
> guardarraíl M** — pero de una forma que solo la batería T-M-I-P podía ver, y
> que queda localizada con precisión. 36 celdas (24 nuevas + 12 de G0
> verificadas comparables), batería × 6 seeds, referencia del incumbente
> computada (post-hoc etiquetado). Fecha: 2026-08-01.

## Lo que salió limpio (todo lo demás)

| métrica | miura_mhbp | hbp_full | lectura |
|---|---|---|---|
| P (corr OOD) | 0.209±0.056 | 0.221±0.049 | no-inferior (Δ=−0.011, IC [−0.055, +0.044]) |
| E (acc/n_iter largo) | 0.058 | 0.057 | favorable no-sig. (Δ=+0.001, dz=0.84, p=0.094) |
| accuracy largo | 0.732 | 0.750 | ✓ dentro de margen (−0.018 ≥ −0.03) |
| I' | 0.855 | 0.853 | ✓ intacta (+0.003) |
| M1 trend on-policy | **+1.00 × 6/6** | — | cristalización perfecta |
| M2 follow/acc | 0.97–1.05 × 6/6 | — | estado portador intacto |

## El toque al mecanismo (el hallazgo)

**M3 (ligadura fidelidad-de-trayectoria ↔ acierto OOD):**

- Full-range: mhbp 0.804 vs hbp_full 0.826 — Δ=−0.022, p=0.156, **n.s.**
- **On-policy** ([1, n_parada]): mhbp 0.679 vs hbp_full 0.735 —
  **Δ=−0.055, t=−3.98, p=0.011, dz=−1.62**, negativo en 5/6 seeds.

El efecto es pequeño, SISTEMÁTICO y está LOCALIZADO: la modulación del plano
debilita el acople entre la fidelidad de la trayectoria y el resultado
exclusivamente en la ventana on-policy (los ticks hasta la parada), sin tocar
la cristalización (M1), el rol portador del estado (M2), el destino (accuracy)
ni la interfaz (I'). El disparo formal del guardarraíl (seed 2: 0.59 < 0.60)
era la punta visible de este efecto, no ruido.

*Nota de agregación (declarada): el prereg no fijó cómo agregar M entre seeds;
el report usó la lectura estricta (6/6). Con media o mayoría, M3 on-policy
también queda por debajo del incumbente con p=0.011 — el veredicto FAIL del
guardarraíl es robusto a la ambigüedad.*

## Lectura

1. **La batería T-M-I-P funcionó como test de aceptación con precisión
   quirúrgica**: detectó un toque al mecanismo (−0.055 en una correlación de
   acople) invisible para accuracy, P, E e I'. "Acoplarse por I y P sin tocar
   M" es exigible y medible — y el plano, tal como está cableado, no lo cumple
   del todo.
2. La disciplina F2b (modulador acotado, init neutro) NO bastó para dejar M
   intacto: alguna vía de modulación por-tick (candidatas: gates de WM desde
   deliberative; block_gate desde fast) inyecta variabilidad temprana que
   descorrelaciona fidelidad y resultado sin dañar el destino.
3. En términos de GOBIERNO, el plano es no-inferior al campo único (P, E, I',
   accuracy) a escala intra-instancia — consistente con el techo declarado en
   el prereg (τ comprimidas en 24 ticks; el test multiescala genuino es F3b).

## Diagnóstico de lesión por vías — EJECUTADO (30 celdas, resultado concluyente)

M3 on-policy bajo lesión en eval (media 6 seeds; incumbente = 0.735):
baseline 0.682 · sin_halt 0.660 · sin_wm 0.681 · sin_gate 0.671 ·
sin_todas 0.656. **Ninguna lesión recupera el acople; quitar modulación
incluso lo empeora levemente** (shift off-policy del modelo entrenado con ella).

**Conclusión: el desacople está EN LOS PESOS ENTRENADOS** — es un efecto del
entrenamiento bajo modulación por-tick, no de la actividad moduladora en
inferencia. Consecuencias:
1. El re-cableado NO puede validarse con parches en eval: exige RE-ENTRENAR.
2. La atribución a una vía concreta queda indeterminada por este instrumento
   (discriminarla exigiría un barrido de re-entrenos por vía, ~20h).
3. El "toque a M" queda caracterizado como: pequeño (−0.055), localizado
   (solo ventana on-policy), horneado en entrenamiento, sin efecto en
   destino/gobierno/interfaz. Un TRADE-OFF declarable, no un fallo activo.

## Decisión recomendada

**Declarar el trade-off y pasar a F3b**, plegando el re-cableado como BRAZO de
F3b: la escala de sesión exige re-entrenar de todos modos, así que la variante
"gates de WM por-instancia" se prueba allí sin coste de programa adicional
(un brazo más del diseño F3b), convirtiendo la hipótesis de mecanismo en
contraste pre-registrado en el entorno donde la multiescala por fin tiene
espacio real.
