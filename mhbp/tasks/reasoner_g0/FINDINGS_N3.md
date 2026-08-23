# N3 — Hallazgos de la fase A: la decisión explícita HACE lo que el
# gradiente no pudo — y la jerarquía de la información queda cuantificada

> 2026-08-06. Fase A completa (12 ckpts congelados, eval 16k, forward-only,
> ~30 min de GPU). Gates VG-N3a/b/c verdes previos; veredicto formal en
> n3_veredicto.json. Ramas pre-declaradas de PREREG_N3 v2 aplicadas.

## La tabla (payoff_norm, medias de 6 seeds; e=5 emparejado exacto)

| brazo | payoff | routing corr(n, stake\|K) |
|---|---|---|
| uniforme | 0.467 | — |
| dificultad (s̄·p̂, ciego al valor) | 0.546 | +0.01 |
| regla (2 niveles por ŝ, sin p̂) | 0.637 | — |
| **expl (argmax ŝ·p̂ − λn)** | **0.698** | **+0.79** |
| oráculo de clase (stake verdadero) | 0.698 | — |
| nativo (halting PonderNet) | 0.921 @5.4 | +0.01 |

## Veredicto por contrastes

- **A1 PASA con contundencia**: Δ(expl − dificultad) = **+0.151** ≫ δ0=0.02
  (t=25.0, p=8.7·10⁻⁷, 12/12). El mecanismo explícito convierte el valor en
  cómputo: corr(n, stake|K) = +0.79 — la firma conductual que en N2 fue
  0.00 con el MISMO sensor y el MISMO sustrato.
- **A2 = 1.000**: el asignador captura el CIEN POR CIEN del techo de
  información de clase (expl ≡ oráculo a 3 decimales — con AUC 1.0 del
  sensor, ŝ es el stake). La regla sin automodelo deja +0.06 sobre la mesa:
  la tabla aprendida de las auto-sondas no es redundante.
- **VG-N3d ROJO → A3 re-etiquetado** (rama pre-declarada): la ejecución
  forzada pierde −0.34 contra el régimen nativo incluso sin valor — A3 no
  compara decisiones, compara ejecuciones. Con esa etiqueta: el nativo
  (0.921) supera al mejor asignador ex-ante (0.698) por lo que la
  información posterior VALE: observarse pensar y parar cuando está hecho.

## La lectura — dos resultados en uno

1. **HN3 confirmada en su terreno**: la tesis de la serie de integración
   queda demostrada por contraste directo. El MISMO headroom de valor que
   el acoplamiento entrenado capturó al 0.000 (N2, canal vivo AUC 0.95),
   la decisión computada lo captura al 1.000 del techo ex-ante. Las
   decisiones de segundo orden no emergen: SE COMPUTAN — y computarlas
   funciona al límite de la información disponible.
2. **La jerarquía de la información, cuantificada** (el hallazgo que no
   pedimos): uniforme 0.47 < +dificultad 0.55 < +valor 0.70 (techo ex-ante
   de clase) < +parada posterior 0.92. La auto-observación durante el
   pensamiento (el halting nativo leyendo el estado que evoluciona) vale
   +0.22 sobre TODO lo decidible antes de pensar. El automodelo ex-ante
   más perfecto no sustituye a mirar cómo va.

## Implicación arquitectónica (la síntesis para la fase B)

Ninguno de los dos componentes domina: el nativo no enruta valor (corr
+0.01) y el explícito no ve el posterior. La arquitectura racional es la
composición: **parada posterior nativa (el CUÁNDO, ya certificada — I′) +
sesgo explícito de valor sobre su umbral (el CUÁNTO-IMPORTA)** — que es
exactamente la vía espejo de gating_endo, ya implementada y testeada (T11),
operable en EVAL como offset del umbral condicionado a ŝ. Gate propuesto
para la enmienda de fase B (VG-B0, forward-only, ~30 min): barrido del
offset por clase de ŝ a E[n̄] emparejado sobre el nativo — ¿queda headroom
de VALOR encima de la parada posterior? Si sí, la fase B integra; si no,
el capítulo cierra con la jerarquía completa y la composición queda para
el entorno con acantilados.

## VG-B0 (2026-08-06, post-fase-A): ROJO — el posterior SATURA el valor

Barrido del sesgo de valor sobre el umbral nativo (instrumento
logit_offset, E[n̄] emparejado ±0.05, δ∈{0.5,1,2,3.5}, umbral pre-declarado
media ≥0.01 y ≥10/12): **headroom = +0.0002 ± 0.0004** (mejor δ = 0 en
8/12). La fase B NO se lanza: no hay nada que integrar.

**El cierre del capítulo, con la jerarquía completa**:
uniforme 0.467 < +dificultad 0.546 < +valor ex-ante 0.698 <
**posterior 0.921 ≈ posterior+valor 0.921**.

Lectura final: en un entorno de valor suave donde el sistema PUEDE
observarse pensar, la parada posterior subsume a la decisión por valor —
el valor ex-ante solo paga bajo COMPROMISO (cuando decides antes de poder
mirar: +0.151 allí). La decisión explícita queda validada como mecanismo
(hace lo que el gradiente no pudo, al techo de su información) y acotada
en su dominio (regímenes ex-ante/commitment). La frontera donde valor y
anticipación pagarían INCLUSO con posterior — tareas con acantilados,
donde desde dentro no se ve si estás cerca — es la deuda N1b, ahora con
tres capítulos de evidencia apuntándole.

## Desviaciones declaradas

Bootstrap por instancia de A1 omitido por redundante (Δ = 7.5·δ0 con
t=25; el criterio del prereg era ambos-deben-excluir-δ0 y el t lo excluye
por 5 órdenes); brazos @E[n]-nativo añadidos para VG-N3d/A3 según remedio
del panel; p̂ = tabla isotónica y ŝ slots-only (enmiendas de fit
commiteadas pre-eval, 596da55).
