# FINDINGS V4 — la des-confusión pedida por los revisores, adjudicada
### 2026-08-18. 120 celdas, seeds frescas 20-39, 19.7 h GPU. PREREG_V4.

## Veredictos (pareado por seed, n=20, una cola, Holm-2 por hipótesis)

**H-ORDEN — NO confirmada como claim general; direccional acotada.**
hbp_full − hbp_first_eq (topes equiparados vía IMEX implícito):
adjacent **+0.087 [+0.042, +0.132], t=4.0, p=3·10⁻⁴** ✓;
cycle_transp +0.014 [−0.013, +0.040], n.s. ✗. Con topes iguales, el
efecto de orden de cycle_transp del v3 desaparece casi por completo
(first_eq 0.229 vs full 0.242): **parte del contraste original era
capacidad, no orden** — la sospecha exacta del revisor. En adjacent el
orden es causal y fuerte.

**H-GRU — NO confirmada; se ejecuta la rama honesta.**
hbp_full − hbp_gru: adjacent +0.006 n.s.; cycle_transp **−0.035
[−0.067, −0.002]** (el GRU gana nominalmente; es el mejor brazo del
estudio, 0.277). Accuracy: no-inferioridad mutua en las 4 comparaciones
(margen 0.02). Rama pre-registrada, ejecutada literal: **«estructura sí»
se rebaja a «un gobernador recurrente con estado y esta interfaz
basta»**. El valor distintivo del campo pasa a ser la
CERTIFICABILIDAD (apéndices A-B del paper: umbral de flutter exacto,
cota IMEX incondicional — nada de eso existe para el GRU), no la
capacidad.

## Lectura para los papers

El capítulo del gobernador gana su forma final y más honesta: un plano
modulador recurrente con la interfaz interoceptiva propuesta es viable y
útil para la asignación de cómputo OOD; la física de campo es UNA
implementación — la única con estabilidad demostrable — y el orden de la
dinámica importa en una familia de generadores y no en la otra. El
programa vuelve a comprarse honestidad con GPU: las dos condiciones del
revisor produjeron una corrección real del claim central.
