# N1b-LLM — el acantilado mudo en su hogar natural: RESULTADO
### 2026-08-12. Qwen2.5-14B congelado, greedy, 768 instancias ciclo +
### 384 aritmética. Todo offline sobre UNA generación por instancia.

## 1. La escalera medida (punto primario, presupuesto GASTADO emparejado)

| brazo (información que usa) | ciclo (acantilado) | aritmética (suave) |
|---|---|---|
| uniforme | 0.674 | 0.159 |
| + dificultad (clase visible) | 0.754 | 0.182 |
| + **valor ex-ante** (clase × stake) | **0.885** | **0.201** |
| + saber quién llega (oráculo binario) | 0.885 | 0.214 |
| + saber el coste exacto (oráculo-c) | 0.952 | 0.214 |
| **rango total** (oráculo-c − uniforme) | **0.278** | **0.055** |

## 2. Veredictos

**H1 (pre-registrada) — CONFIRMADA con margen.**
valor − dificultad = **+0.1312, IC95 [+0.1124, +0.1502]**, umbral
max(0.02, 4·SE)=0.0386 → 3.4× el umbral, IC lejísimos de 0. Estimador
pre-registrado: pago marginalizado analíticamente sobre el stake +
cross-fitting A↔B + bootstrap pareado 20k. En la familia suave el mismo
contraste da +0.019 (7× menor). **El valor paga bajo compromiso, y paga
mucho más donde el coste es invisible ex-ante.**

**Contraste de familia (pre-registrado, predicción OPUESTA) —
CONFIRMADO.** (oráculo-c − dificultad) = **+0.198 en ciclo vs +0.031 en
aritmética** (6.4×). Y el gate R lo corrobora por la vía del coste:
R²(coste | clase) = 0.26 en ciclo vs 0.75 en aritmética. La familia
acantilado **multiplica por 5 lo que está en juego** en la asignación
(rango 0.278 vs 0.055).

**H2 (pre-registrada) — NO PASA.** oráculo-c − oráculo-binario = +0.067
(umbral 0.0056). Lectura honesta: el comparador que pre-registré es
CLARIVIDENTE (conoce c_i antes de gastar), no un observador mid-flight
— el panel me lo advirtió y lo dejé igual. Ese +0.067 mide el precio de
una clarividencia que en una familia muda nadie puede tener, ni ex-ante
ni mirando. Lo que sí es realizable está medido y es tajante: **el canal
«quién llega» está exactamente vacío** (valor = oráculo-binario = 0.885,
Δ=0.000; censura 0%).

## 3. El hallazgo que no buscábamos: misma estructura, distinta escala

Normalizando cada familia por su propio rango, la fracción capturada
por el mejor ex-ante es **0.759 en el acantilado y 0.764 en la suave** —
indistinguibles. Es decir:

> El acantilado **no cambia el reparto** entre lo decidible a priori y
> lo que exigiría ver el futuro: cambia **cuánto hay que repartir**.

Eso REFUTA la forma fuerte de la predicción N1b (que el acantilado
desplazaría el peso hacia el ex-ante) y a la vez explica por qué el
valor «renace» en él: no porque capture una fracción mayor, sino porque
la tarta es 5× mayor. La predicción original conflaba dos cosas —
magnitud del problema de asignación y estructura del reparto — que estos
datos separan por primera vez.

## 4. Gates (todos sobre las grabaciones, antes de computar H1/H2)

- **R (rango/acantilado)** ✓: corr(coste, distancia) = 1.00; R² por
  clase 0.26 (ciclo) vs 0.75 (arit); censura 0%.
- **S (no salta)** ✓: cadena válida en el **100%** de las correctas —
  Qwen camina hop a hop, nunca resuelve de un vistazo. La premisa
  coste ∝ distancia se sostiene literalmente.
- **M (mudez)**: criterio PRE-REGISTRADO (¿el contenido añade sobre
  (m, clase)?) se cumple — ΔAUC ≤ 0 en todos los horizontes — **pero su
  control positivo, en la forma pre-registrada, FALLA** (la aritmética
  también da ≈0): en ambas familias los tokens emitidos ya resumen el
  avance, así que el instrumento no discrimina. Métrica alternativa
  (NO pre-registrada, umbral fijado el 2026-08-12): predictibilidad
  ABSOLUTA del remanente desde el conjunto observable O = **0.735 en
  ciclo vs 0.874 en aritmética** — el acantilado es menos predecible,
  pero no mudo del todo: el hazard «llevo m sin llegar y L es finito»
  informa. **La mudez estricta queda sin adjudicar** y así se reporta.

## 5. Contabilidad honesta del proceso

Pre-registrado y cumplido: familia, brazos, estimador de H1, umbrales de
H1, contraste aritmético con predicción opuesta, gates R y S, truncado
honesto (self-test 60×20 contra re-parseo literal). No pre-registrado y
declarado como tal: la métrica absoluta del gate M y su umbral. Tres
bugs propios cazados por sus tests antes del veredicto: el self-test de
truncado (resolución de rejilla), el brazo uniforme a 0.000
(comparación sobre array de objetos) y el hueco de integralidad del
multiplicador de Lagrange (sustituido por DP exacto). Coste total:
~6 min de GPU (las generaciones son cortas: 36 tokens de media) + CPU.
