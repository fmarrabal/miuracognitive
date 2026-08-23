# F3a-R — Hallazgos FINALES: el toque M3 de F3a NO REPLICA

> 2026-08-03. 48 celdas (pi1, noc, ptR, hbp_fullR × 6 seeds × 2 regímenes) +
> batería completa. Veredicto por la rama pre-declarada de la Enmienda 1.
> Números: f3ar_veredicto.json. Este documento REEMPLAZA la versión
> provisional y reclasifica un hallazgo de F3a.

## La cadena completa (tres capas de robustez, cada una pre-declarada)

| medición | M3 on-policy | lectura en su momento |
|---|---|---|
| F3a: mhbp por-tick (celdas originales) | 0.679 | «toque −0.055, p=0.011, dz=−1.62» |
| F3a: incumbente (celdas originales) | 0.735 | referencia |
| F3a-R: pi1 (contenido congelado) | 0.712 | «recupera ~60%» (p=0.0567, filo) |
| F3a-R: noc (sin contenido) | 0.707 | «también recupera» |
| **Robustez: ptR (por-tick FRESCO, sin re-cableado)** | **0.717** | la «recuperación» era del comparador |
| **Enmienda 1: hbp_fullR (incumbente FRESCO)** | **0.728** | — |

**Toque contemporáneo (ptR − hbp_fullR, pareado, mismas seeds, mismo
código): Δ=−0.011, IC95=[−0.068, +0.045] → NO_REPLICADO.**
Diffs por seed: [−0.03, −0.04, +0.03, +0.03, +0.03, −0.10] — sin patrón.

## Veredicto (ramas pre-declaradas)

1. **El guardarraíl-M-FAIL de F3a se reclasifica como NO REPLICADO**: el
   efecto −0.055 era compatible con el ruido de run por celda (σ_run del
   M3 on-policy por celda ≈ 0.04-0.05, medido ahora con las réplicas), que
   el diseño F3a (una run por celda, pareado solo por seed) no estimaba.
   **El trade-off declarado se RETIRA del registro del arco.**
2. **Cascada de reclasificación**: la lesión de F3a («el desacople está en
   los pesos») perseguía un gap que no replica — su conclusión técnica sigue
   siendo correcta (ninguna vía de modulación en eval causaba el gap:
   consistente, porque el gap era varianza de run), pero su interpretación
   («efecto de entrenamiento bajo modulación por-tick») queda RETIRADA.
3. **Re-cableado: no hay nada que re-cablear.** R1' nulo (pi1−ptR=−0.004).
   Resultado útil colateral: congelar las vías de contenido por instancia es
   INDISTINGUIBLE del por-tick en TODO (M3, P, I', acc, M1, M2) — la
   fluctuación de contenido ni daña ni aporta. **Por-tick sigue como variante
   oficial** (continuidad/simplicidad, rama de la enmienda).
4. Lo que SÍ queda en pie de F3a (efectos grandes o nulos con margen,
   robustos a ±0.05): no-inferioridad de gobierno (P/E/acc/I'), M1
   cristalización +1.00, M2 estado portador ~0.88-1.0, y toda la línea
   G0/G0.1 (efectos de orden 0.3-0.8).

## La lección metodológica (entra en las reglas del programa)

**Una run por celda no estima el ruido de run.** Todo contraste entre
arquitecturas cuyo tamaño esperado sea ≲ 0.05 en métricas de batería exige
réplicas por celda (≥2 runs por seed×brazo) o una estimación previa de
σ_run, y ningún guardarraíl con umbral cerca de ese ruido puede disparar
con una sola run. El pareado por seed NO parea el ruido de run. (Las tres
capas de robustez de F3a-R — control fresco → anomalía → incumbente fresco —
son el procedimiento que lo detectó: pre-declarar la robustez «al filo» fue
lo que impidió publicar una recuperación fantasma de un daño fantasma.)

## Estado del programa tras F3a-R

- F3a queda como: **gobierno no-inferior del plano, batería M intacta, sin
  trade-off** — mejor resultado del que creíamos tener, con una medición
  menos de la que presumíamos.
- El «doble objetivo» de F3b queda cerrado por sus dos mitades: (A)
  multiescala-sesión = negativo-de-diseño robusto (FINDINGS_F3B_GATES);
  (B) re-cableado = sin objeto (el toque no replica).
- Siguiente frente natural: **N2 del roadmap** (stakes endógenos / agencia
  sobre el sustrato certificado), con la regla de réplicas incorporada.
