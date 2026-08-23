# FINDINGS TEORÍA — ¿los teoremas gobiernan una clase? Sí, y por eso fallan
### 2026-08-20. Auditoría adversarial de alcance y novedad (8 agentes, 3 búsquedas de literatura independientes).

## La respuesta

**Sí gobiernan una clase, y ese es exactamente el problema.** Ninguna
demostración usa el laplaciano del grafo, la interfaz de interocepción,
f_θ/g_φ, el reasoner ni el transformer: la única identidad que se emplea
es vᵀGv = 0. La generalidad es real. Pero la clase que gobiernan es
«sistemas M–C–K–G lineales invariantes con rigidez y amortiguamiento
SPD», gobernada desde **Thomson & Tait 1879**. El eje NARROWNESS falla; el
eje TRIVIALIDAD acierta de lleno.

## Precedentes, resultado por resultado

| resultado | clase que cubre | precedente |
|---|---|---|
| Dicotomía de colocación (a) | cualquier ü+(C+G)u̇+Ku=0, K,C SPD | **Kelvin–Tait–Chetaev** (Thomson & Tait 1879); es el índice del cap. 6 de Merkin 1997 |
| ídem (b), 1er orden | cualquier γu̇+(K+G)u=0 acretivo | **AntisymmetricRNN** (ICLR 2019) y **A-DGN** (ICLR 2023) — *ya citado en nuestro refs.bib como `gravina2023antisymmetric`* |
| Umbral de flutter | K **y** C ambos escalares (c=0 **y** D=0) | **Bottema 1955** en K=k·I, coeficiente a coeficiente; **Smith 1933** (whirl rotodinámico) |
| Cota IMEX | cualquier M=I+λ(K+G) con parte simétrica coerciva | **norma logarítmica** (Dahlquist 1958, Söderlind 2006); rango numérico (Horn–Johnson); acretivos (Kato); **Bai–Golub–Ng HSS 2003** |

La dicotomía parte (c) **no es un teorema**: mostrar que un candidato de
Lyapunov deja de decrecer no demuestra inestabilidad. La literatura
clásica sí tiene el teorema correspondiente (Bottema–Lakhadanov–Karapetyan,
Bulatović 1999).

## Lo que verifiqué yo en el código, y es peor que la novedad

**1. c > 0 SIEMPRE.** `hbp.py:285` es
`c = cfg.c_max * torch.sigmoid(self.raw_c)`, y la sigmoide nunca vale
cero. Con `c_init=0.4`, `c_max=0.7`, **c es estrictamente positivo en
todos los brazos**. El umbral de flutter «exacto» está demostrado solo
para K=ω₀²I, es decir c=0: **cubre CERO de los sistemas que corrimos**.

**2. G = 0 en el modelo del titular.** `D_max`, `b_adv_max` y
`kdv_beta_max` valen `0.0` por defecto (`hbp.py:73,74,107`) y
`trainer.py` solo los activa para `hbp_kdv` (β=0.1) y `hbp_mix` (D=0.4,
b=0.4). **`hbp_full` no instancia ningún operador antisimétrico.** Con
G=0, todo el aparato de colocación y flutter es **vacuo para el modelo
que produce los números del paper**.

## Lo único genuinamente nuevo, y no está demostrado

La condición discreta de Verlet/Schur–Cohn con coeficientes complejos: es
el único ítem para el que tres búsquedas independientes no hallaron
precedente. Y es justo el que el apéndice B **no demuestra**, mientras el
abstract y la contribución 3 lo venden.

## Una frase del abstract que es falsa

«a guarantee no learned recurrent governor carries» — **Bonassi, Farina &
Scattolini (Syst. Control Lett. 157, 2021)** dan condiciones ISS y δISS
**para una GRU**, verificables post-hoc o imponibles en entrenamiento.

## Veredicto de sede

**TMLR, no JMLR.** Odds: JMLR ahora ≈45% desk-reject y 5-8% de aceptación;
TMLR ≈75-80%. El informe de JMLR se escribe solo, y con nuestras propias
frases: la línea 428 atribuye Kelvin–Tait–Chetaev, la 440 llama a nuestra
hipótesis «the degenerate Merkin case», y el apéndice A llama a las
pruebas «standard energy arguments».

## Reencuadre que sí funciona

Dejar de llamar «certificado del gobernador» a lo que es un **certificado
del integrador**. Bajo ese marco el paper es bueno: *el campo es un
integrador LTI cuya estabilidad es exactamente auditable en runtime; una
GRU solo admite certificados suficientes y conservadores; medimos el
precio de esa restricción sobre la expresividad y es pequeño pero no nulo
(≤0.035)*. Eso convierte el negativo en el resultado: **el coste medido de
la certificabilidad en gobernadores de cómputo aprendidos.**

## Lo que sí sería de tamaño JMLR (4-8 semanas)

El hueco que la literatura deja realmente vacío: un certificado para el
lazo **cerrado y gateado**. Dos mitades: (i) Lyapunov cuadrático común vía
**LMI** sobre la caja de coeficientes —arregla cuatro agujeros a la vez:
certifica la mezcla α, certifica el caso LTV gateado donde ρ(Φ_t)<1 no
vale, sube de radio espectral a cota en NORMA, y sustituye la penalización
blanda por una proyección por construcción—; y (ii) ISS en lazo cerrado
por small-gain sobre el camino interocepción→modulación→host. Verificado
que nadie junta las tres propiedades: REN tiene (c) sin (a,b); CON-ISS
tiene (a)+(c) sin (b); LinOSS/D-LinOSS solo (a). **Caveat honesto: no está
verificado que la LMI cierre sobre nuestra caja real de coeficientes.**

## Arreglos OBLIGATORIOS antes de enviar a cualquier sitio

1. Demostrar la Prop. 1 o rebajarla en el abstract a «criterio, validado
   numéricamente». No anunciar lo que no se demuestra.
2. Reescribir la frase «no learned recurrent governor carries» (falsa).
3. Retirar o demostrar que Props. 1+2 establecen la estabilidad de la
   mezcla α: es falso como implicación (el radio espectral no es convexo)
   y `hbp_mix` se queda hoy **sin certificado alguno**.
4. Corregir «equivalently c=0, or [L,A³]=0»: bajo conmutación con c>0 el
   umbral es suficiente, no exacto. Y declarar la hipótesis silenciosa
   C escalar (D=0).
5. Corregir el Lema 1(d): la condición correcta es ν(1+ρ(A)) < ω₀².
6. Acotar el corolario del apéndice B a F independiente del estado, o
   añadir L_F < ω₀² y comprobar si se cumple (ω₀²=0.04 frente a
   ganancias 0.3 — comprobar antes de afirmar).
7. Borrar la afirmación del apéndice A de que la estabilidad IMEX se sigue
   de Schur–Cohn modo a modo: es incorrecta y contradice al apéndice B.
8. Rebajar la Prop. 2 a «Lema (estándar; se recuerda por completitud)» con
   las citas. No cuesta nada empíricamente y desarma el cargo de
   reetiquetado.
9. Citar Bottema 1955, Smith 1933, Thomson & Tait 1879; corregir la
   atribución de `ascher1995imex`; y añadir la línea entera que falta:
   coRNN, UnICORNN, LinOSS, D-LinOSS, REN, AntisymmetricRNN,
   Kolter–Manek, CON-ISS, Bonassi, LTC. Decir explícitamente que A-DGN
   **es** el Lema 1(b) y que nosotros generalizamos γI → K.
10. Reencuadrar: certificado del **integrador**, no del gobernador.
