# Protocolo congelado — fase 3, `confirmatory_v1`

Fecha de congelación: 2026-07-16. Este archivo se escribió antes de ejecutar
las semillas confirmatorias.

## Hipótesis operacional

Un sistema descubre una meta en este banco si, sin catálogo ni coordenada
objetivo externa, usa consecuencias de sondeos elegidos por él para sintetizar
una coordenada factible, nueva y eficaz sobre la necesidad endógenamente
dominante en mundos no vistos.

## Diseño fijado

- semillas emparejadas: 200–219;
- variantes: `discoverer` y `no_feedback`;
- 400 pasos de ajuste, batch 128, AdamW, `lr=2e-3`;
- evaluación: 10 batches nuevos de 128 episodios por semilla;
- cuatro sondeos y el mismo presupuesto en ambas variantes;
- región factible y centros nuevos en cada episodio;
- centros ocultos a la política y sin loss de supervisión de centros;
- decodificador: trilateración RBF diferenciable más residual aprendido;
- éxito: respuesta de la necesidad dominante ≥0.90 y nivel final ≥0.70.

La ley RBF es una inductive bias conocida del controlador; el centro concreto
no lo es. Por ello la afirmación se limita expresamente a la gramática de
affordances ensayada.

## Familia primaria

Se aplicará una prueba exacta de cambio de signo sobre las 20 diferencias por
semilla y corrección Holm familiar a α=0.05. Deben ser positivos y
significativos los cuatro efectos:

1. `discoverer − no_feedback` sobre éxito;
2. normal − lesión de feedback sobre éxito del `discoverer`;
3. normal − consecuencias barajadas entre mundos;
4. meta normal − reflexión factible de su contenido.

## Guardrails, todos obligatorios

En cada semilla del `discoverer`:

- éxito normal ≥0.70;
- factibilidad =1.00;
- fracción de metas continuas únicas ≥0.95;
- distancia normalizada media entre sondas ≥0.05;
- trasplante corporal cambia hacia la necesidad nueva ≥0.70;
- éxito sobre la necesidad nueva tras trasplante ≥0.50.

El resultado sólo será `pass` si pasan la familia completa y los seis
guardrails. No se reinterpretará un fallo como confirmación parcial.
