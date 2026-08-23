# Fase 4 — automodelo de consecuencias corporales

Esta fase pregunta si el sistema puede predecir qué cambia en su propio cuerpo
cuando actúa y revisar esa predicción después de sufrir daño.

El cuerpo es una cadena de cinco nodos gobernada por la ecuación de onda
amortiguada auditada. Dos crisis equivalentes requieren restaurar un nodo. En
la segunda, el actuador primario se rompe sin flag externo. El agente sólo ve
estado, velocidad, comando emitido y consecuencia; la matriz real y el nodo de
evaluación permanecen ocultos.

`self_model` resta la predicción pasiva de onda al estado observado, atribuye el
residuo a la copia eferente y actualiza el efecto del comando. Después compara
contrafactuales de todos los comandos. `stale_model` conserva exactamente el
mismo prior, observaciones, candidatos y cómputo, pero congela esa actualización
tras el daño.

## Resultado confirmatorio

**PASS** en 20 semillas nuevas (`400..419`).

- recuperación predaño: `1,000` en ambos brazos;
- recuperación postdaño: `1,000` frente a `0,59438`;
- cambio al respaldo: `1,000` frente a `0,59438`;
- repetición del actuador roto: `0,000` frente a `0,85138`;
- MSE predictivo tras adaptación: `1,10e-5` frente a `1,06e-2`;
- coincidencia con acción oráculo: `0,97919` frente a `0,59548`.

Los cuatro efectos causales —arquitectura, lesión de actualización, copia
eferente barajada y rotación del contenido— son positivos en 20/20 semillas y
pasan Holm (`p=7,63e-6`). Los ocho guardrails son verdaderos. La lesión produce
exactamente el mismo resultado que el control congelado (`gap=0`).

Interpretación: se confirma un automodelo causal y online de consecuencias
motoras, capaz de detectar daño y replantear acciones en esta planta conocida.
No se confirma un yo narrativo, autorreconocimiento, consciencia ni identidad
humana. La identificación es especialmente limpia porque entre shocks la
dinámica pasiva es conocida y el residuo de acción es observable; generalizar a
dinámicas desconocidas y ruido será una prueba posterior más fuerte.

Archivos principales:

- `confirmatory_v1/PROTOCOL.md`: protocolo congelado;
- `confirmatory_v1/README.md`: informe detallado;
- `confirmatory_v1/summary.json`: agregado y pruebas exactas;
- `{variant}_seed{400..419}.json`: intervenciones por semilla;
- `checkpoints/`: estados reproducibles del controlador.
