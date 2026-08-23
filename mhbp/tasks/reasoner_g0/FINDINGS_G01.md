# G0.1 — Hallazgos: la interfaz nunca estuvo rota; estaba mal medida

> Etapa a de PREREG_G01 (probes de suficiencia + baseline I' retroactivo,
> ambos pre-declarados). Resultado: el criterio de éxito de G0.1 se cumple
> con los checkpoints EXISTENTES — la etapa b (retrain con pérdida auxiliar)
> NO se ejecuta por innecesaria para el gate (decisión documentada aquí).
> Fecha: 2026-07-31. Datos: results/g01a_probe.json.

## Resultados (cycle_transp, 6/6 celdas)

| celda | probe F1 (pooled) | probe F2 (+arrow) | **I' del halting actual** | acc |
|---|---|---|---|---|
| gating_wm s0/s1/s2 | 0.952 / 0.907 / 0.929 | 0.929 / 0.899 / 0.933 | **0.850 / 0.788 / 0.879** | 0.86/0.83/0.86 |
| hbp_full s0/s1/s2 | 0.944 / 0.901 / 0.927 | 0.943 / 0.912 / 0.884 | **0.854 / 0.791 / 0.903** | 0.90/0.84/0.88 |

1. **La información de completitud abunda en las features actuales** (probe
   F1 = 0.90–0.95; añadir la posición ARROW no mejora — el pooling NO diluye).
2. **I' del halting existente = 0.79–0.90 en 6/6** — muy por encima del
   criterio pre-registrado (≥0.65 en ≥2/3 seeds). La λ del halting, leída
   ANTES del techo, ya es un estimador de completitud decente.

## La conclusión que corrige a G0

El I-fail de G0 (AUC invertido 0.13–0.27) era ÍNTEGRAMENTE el artefacto del
techo en MI métrica (p_stop = masa máxima, que en el cap forzado λ:=1 recoge
el remanente de las muestras difíciles). El instrumento confundía el evento de
presupuesto con la señal de la interfaz. Medida correctamente (I', declarada
en PREREG_G01 ANTES de computarla), la interfaz funciona.

Los guardarraíles del criterio se cumplen trivialmente (mismos modelos que G0:
accuracy y P intactos). **G0.1: PASS. La Fase 3 queda desbloqueada sobre
(cycle_transp, {hbp_full, gating_wm}).**

## Sensibilidades pre-declaradas (no alteran veredictos formales de G0)

- **M2 normalizado por accuracy**: cycle follow/acc = 0.99–1.03 (≈1.00: el
  estado es EL portador, sin residuo); adjacent 0.61–0.88 (parcial).
- M1 on-policy: moot en cycle (pasa con el rango completo); adjacent fuera de
  alcance (T-fail).

## Notas de diseño para la Fase 3 (acumuladas de G0/G0.1)

1. **La tarea**: cycle_transp (T certificada; adjacent descartada por atajo
   parcial).
2. **El enchufe del gobernador es λ pre-techo**: el mHBP debe leer/modular la
   señal de completitud (λ_n, n<N_max) y tratar el techo N_max como EVENTO DE
   PRESUPUESTO separado (dominio del campo resource, no del fast) — la
   confusión de ambos fue exactamente el artefacto de G0.
3. **Opcional durante el entrenamiento de Fase 3**: L_halt_aux =
   BCE(λ_n, 1[decode_n correcto]) — la probe indica techo ~0.93-0.95
   alcanzable desde 0.79-0.90; mejora disponible, no bloqueante.
4. El mecanismo M certificado (cristalización F 0.5→0.89; estado portador
   follow/acc=1.0; ligadura M3≈0.8) da al fin al gobernador un sustrato en
   régimen de razonamiento genuino — lo que el arco mHBP nunca tuvo.
