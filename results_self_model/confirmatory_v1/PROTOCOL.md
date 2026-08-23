# Protocolo congelado — fase 4, `confirmatory_v1`

Fecha de congelación: 2026-07-16. Escrito después del piloto de desarrollo con
seed 0 y antes de ejecutar las semillas confirmatorias.

## Hipótesis operacional

Hay automodelo causal cuando el sistema predice las consecuencias de sus
propios comandos, actualiza esa predicción al observar un daño corporal no
señalizado y usa contrafactuales internos para escoger una acción distinta que
restaure el mismo estado homeostático.

## Planta y controles fijados

- cuerpo: cadena de cinco nodos con Klein–Gordon amortiguada discreta;
- radio espectral pasivo exacto esperado: `<1`;
- crisis en `t=0`; segunda crisis y rotura del actuador primario en `t=11`;
- 21 comandos: no-op y actuadores `+/-`, primarios/respaldo, por nodo;
- variación continua de eficacia y shock en cada episodio;
- la política no recibe flag de daño, matriz real de actuadores ni `target_id`;
- `self_model`: actualiza la columna asociada a su copia eferente;
- `stale_model`: mismo prior, observaciones, acciones candidatas y cómputo,
  pero congela el modelo tras el daño;
- semillas emparejadas nuevas: `400..419`;
- evaluación: 10 batches de 128 vidas por semilla y variante.

## Familia primaria

Prueba exacta de cambio de signo sobre 20 diferencias por semilla y corrección
Holm familiar a α=0.05. Los cuatro efectos deben ser positivos:

1. `self_model − stale_model` sobre recuperación postdaño;
2. normal − lesión de actualización postdaño;
3. normal − copia eferente barajada;
4. normal − rotación del contenido del automodelo.

## Guardrails obligatorios

En cada semilla del automodelo:

- recuperación predaño ≥0.95;
- recuperación postdaño ≥0.95;
- cambio al respaldo correcto ≥0.95;
- MSE predictivo tras adaptación `<1e-3`;
- radio espectral pasivo `<1`;
- recuperación y cambio correctos tras trasplantar el daño ≥0.95;
- la lesión de actualización debe igualar exactamente al control congelado.

El resultado sólo será `pass` si pasan los cuatro contrastes y los ocho
guardrails. La afirmación queda limitada a un automodelo de consecuencias
motoras en esta planta, no a identidad narrativa, consciencia o yo humano.
