# N1b piloto offline — CERRADO SIN VEREDICTO SOBRE LA PREDICCIÓN
### (2026-08-09, 3 rondas, 0 GPU. El control pre-declarado bloqueó las
### tres — y la tercera vez la razón resultó ser ESTRUCTURAL, que es el
### hallazgo.)

## 1. Qué se intentó

Testar la predicción N1b («con pago todo-o-nada, el valor pagaría incluso
con posterior») a coste cero, construyendo entornos de pago/presupuesto
sobre los perfiles correct(n) congelados de la sonda N3, con un control
ancla: reproducir el cero conocido de VG-B0 (valor|posterior = +0.0002)
en pago lineal.

## 2. Las tres rondas (el control funcionando)

| ronda | instrumento | fallo detectado por el control |
|---|---|---|
| R1 | posterior round-robin | brazo posterior débil → valor espurio |
| R2 | índice miope (hazard 1 paso) | el stake compensa la miopía, no mide valor; diseño conflaba cliff y escasez |
| R3 | índice no-miope (Gittins-lite) + 3 celdas factorizadas | el ancla lineal-lote TAMPOCO da cero → razón estructural (§3) |

## 3. El hallazgo: por qué el cero de VG-B0 no es reproducible offline

El brazo posterior de N3 es el **halting nativo con cómputo ELÁSTICO por
instancia** (para-cuando-está-hecho; n̄=5.4 emergente; sin presupuesto
duro compartido). El cero «el posterior satura el valor» es una propiedad
de ESE régimen: si cada instancia ya para al estar lista, el sesgo por
valor solo puede malgastar. Cualquier juego offline con presupuesto duro
compartido es OTRO régimen: bajo escasez vinculante con stakes 8:1, la
asignación consciente del valor gana trivialmente (lo que medimos:
+0.5/instancia — cierto, pero a-priori; no es la predicción del paper).

Y lo más importante: **la predicción N1b real es sobre OBSERVABILIDAD
INTERNA** («acantilados donde desde dentro no se ve si estás cerca») —
una propiedad del PAR (tarea, sustrato). Los perfiles de S₅ vienen de la
familia SUAVE donde el sistema sí se ve pensar (posterior 0.921): con
ellos no se puede emular ceguera interna. El piloto era el instrumento
equivocado para esta pregunta, por construcción.

## 4. Lo que el piloto SÍ deja (requisitos de diseño del N1b real)

1. La familia N1b debe tener **valor todo-o-nada con progreso interno
   mudo**: hasta encontrar la solución no hay señal graduada (tareas tipo
   búsqueda/cerradura: o tienes la llave o no tienes nada). El contraste
   clave: el halting nativo NO debe poder anticipar la llegada (medible:
   AUC del posterior sobre «convergerá pronto» ≈ azar).
2. El cómputo debe seguir siendo **elástico por instancia** (régimen de
   VG-B0): la escasez dura confunde el efecto (paga por aritmética).
3. Predicción refinada y falsable: en esa familia, la jerarquía se
   INVIERTE — ex-ante (dificultad/valor, commitment) recupera terreno
   sobre el posterior, porque el posterior pierde su canal.
4. Coste: familia nueva + entrenamiento (pipeline n2 reutilizable) +
   prereg + panel. Es un capítulo real, no un piloto — y es EL cierre
   falsable del paper, ahora con requisitos de diseño medidos.

## 5. Estado

La predicción N1b queda **abierta y afilada** (no confirmada ni refutada
por esta vía — el paper la mantiene como cierre falsable, ahora con los
requisitos §4). El caso R1→R3 entra en el capítulo de Método: tres
instrumentos defectuosos consecutivos bloqueados por el mismo control
pre-declarado antes de contaminar ninguna conclusión.
