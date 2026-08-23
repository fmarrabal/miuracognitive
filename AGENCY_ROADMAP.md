> ## ⚠️ SUPERADO — ver [AGENCY_V2.md](AGENCY_V2.md)
>
> Esta es la hoja de ruta de la línea de agencia **v1**, cuyas cuatro fases
> quedaron RETIRADAS tras una auditoría adversarial que encontró
> circularidad: en la fase 3 el descubridor reproducía el `1.0000` **sin
> entrenar**, y la ley generativa del entorno estaba inyectada en el agente.
>
> El rediseño v2 impone ocho reglas anti-circularidad y vuelve a ejecutar
> las cuatro fases con 20 seeds frescas. **Los números de v1 no deben
> citarse.** Este fichero se conserva como registro histórico.

# Hoja de ruta de agencia homeostática

Secuencia experimental acordada:

`autorregulación → anticipación → metas endógenas → metas descubiertas → automodelo → metacognición`

Cada flecha exige un experimento causal nuevo. Superar una fase no autoriza a
atribuir las propiedades de las fases posteriores.

## Estado

| fase | propiedad operacional | estado |
|---|---|---|
| 0 | redistribuir cómputo limitado desde señales internas | completada; `results_implicit_need` |
| 1 | actuar antes de un déficit a partir de una predicción causal | **confirmada**; 20 seeds nuevas, cuatro criterios causales pasan Holm |
| 2 | seleccionar y sostener prioridades sin `goal_id` externo | **confirmada**; 20 seeds, lesión y rotación causales, cinco guardrails |
| 3 | proponer metas factibles no enumeradas por el diseñador | **confirmada**; 20 seeds, cuatro efectos causales, seis guardrails y sensibilidad |
| 4 | predecir las consecuencias de las propias acciones y adaptarse a daño | **confirmada**; 20 seeds, cuatro efectos causales y ocho guardrails |
| 5 | monitorizar, comunicar y regular los propios procesos cognitivos | siguiente protocolo: metacognición |

## Criterio para avanzar

Una fase sólo se considera superada cuando:

1. el comportamiento aparece en mundos no vistos;
2. una intervención sobre el mecanismo propuesto cambia específicamente el
   comportamiento;
3. controles emparejados descartan señal privilegiada, fuerza bruta y mayor
   presupuesto;
4. el resultado se replica en semillas preregistradas y sobrevive al análisis
   de sensibilidad.

## Arquitectura multihomeostática posterior

Tras completar las seis fases se acoplarán tres campos homeostáticos. Los
nombres psicoanalíticos se conservarán como metáfora de diseño; en código y en
el paper se usarán nombres funcionales para no presentar equivalencias
neuroanatómicas inexistentes.

### Campo de impulsos primarios — «Ello»

- Variables: energía, carga térmica, integridad, fatiga y seguridad.
- Escala temporal rápida y persistente.
- Produce urgencias y candidatos de acción, no decisiones finales.
- Su primer banco experimental es AHA.

### Campo ejecutivo — «Yo»

- Variables: incertidumbre, carga de memoria de trabajo, coste de cómputo,
  conflicto entre metas y horizonte de planificación.
- Mantiene el modelo del mundo y el automodelo.
- Arbitra entre urgencias, metas latentes y recursos finitos.
- Puede inhibir, posponer o secuenciar propuestas del campo primario.
- La fase 2 aporta su primer mecanismo confirmado: prioridad latente con
  histéresis, persistencia bajo ambigüedad y ruptura adaptativa ante crisis.
- La fase 3 aporta identificación de affordances: sondeos causales,
  trilateración de un objetivo continuo no enumerado y cambio de contenido al
  cambiar el cuerpo. La ley RBF conocida limita su alcance y deberá
  generalizarse en fases posteriores.
- La fase 4 aporta un automodelo online de consecuencias motoras: predicción
  pasiva de un cuerpo de onda, atribución del residuo mediante copia eferente,
  detección de un actuador dañado y selección contrafactual de respaldo. Es un
  automodelo funcional estrecho, no un yo narrativo.

### Campo social-normativo — «Superyó»

- Variables: daño previsto a otros, violación de normas, reciprocidad,
  confianza y discrepancia entre perspectivas.
- Requiere antes una red de mentalización que prediga creencias y consecuencias
  para otros agentes.
- Propone restricciones y objetivos sociales aprendidos; no será una lista de
  reglas escrita a mano ni un simple término escalar de recompensa.

### Acoplamiento que habrá que probar

Los tres campos evolucionarán simultáneamente sobre grafos y escalas temporales
distintas. Un árbitro ejecutivo recibirá propuestas con urgencia, confianza y
coste previsto. Las predicciones centrales son:

- lesión del campo primario: pérdida selectiva de mantenimiento/urgencia;
- lesión ejecutiva: impulsividad, perseveración y mala planificación;
- lesión social: conducta instrumental preservada pero peor coordinación,
  mentalización y cumplimiento de normas nuevas;
- desacoplamiento temporal: cada lesión debe producir un perfil distinto, no
  sólo una caída global de rendimiento.

Esta arquitectura sólo se construirá después de validar por separado los
componentes necesarios. «Ello», «Yo» y «Superyó» no serán etiquetas colocadas
sobre tres redes arbitrarias, sino hipótesis funcionales sometidas a lesiones y
trasplantes causales.
