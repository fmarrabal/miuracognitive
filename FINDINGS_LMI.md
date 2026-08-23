# FINDINGS LMI — el certificado por Lyapunov común: qué cierra y qué no
### 2026-08-20. `certify_lmi.py`, `certify_lmi_ckpt.py`, `certify_lmi_iss.py`.

Se intentó cerrar el hueco que la auditoría señaló como «de tamaño JMLR»:
Lyapunov cuadrático común vía LMI sobre la caja de coeficientes, más ISS en
lazo cerrado por small-gain. **Cierra una parte y no cierra la otra**, y la
parte que no cierra es la que hacía falta para JMLR.

## Lo que SÍ queda certificado

**Existe P común sobre la envolvente de los checkpoints entrenados.** Con
ω₀∈[0.488,0.529], ζ∈[0.475,0.538], c≤0.371 (leídos de los cinco
checkpoints de `hbp_full`), la LMI cierra con **ρ=0.9517** y κ=2.51. Los
cinco son además certificables por separado, con el peor radio espectral en
**0.7322** — cómodamente dentro.

Eso no es cosmético. Es una cota en **norma**, no en radio espectral, así
que cubre los transitorios no normales que el radio espectral no ve; y por
convexidad y submultiplicatividad de la norma de operador cubre **las
mezclas α y el caso gateado (LTV) dentro de la familia de onda**, que es
exactamente lo que `ρ(Φ_t)<1` no podía cubrir.

## Lo que NO cierra, y es lo que importaba

**1. La caja DECLARADA en `HBPConfig` no es certificable, porque no es
estable.** De sus 64 vértices, **40 divergen** (peor ρ=5.24); en la caja de
operación más estrecha, 16 de 64. El culpable es `c`, que entra en la
rigidez como c²·λmax(L): con c=0.7 sobre la cadena de seis nodos eso vale
≈1.83 y se come el margen del criterio de Verlet. La frontera de
certificabilidad resultó ser **c=0, ω₀≤0.276** — degenerada. Conclusión
honesta: **los topes del código son demasiado laxos**, y lo que mantiene al
modelo lejos de la divergencia es el entrenamiento, no la caja.

**2. La mezcla con la rama difusiva sigue sin certificado.** Sobre la
envolvente entrenada, incluyendo operadores antisimétricos (D≤0.4, b≤0.4,
β≤0.1) y la rama difusiva, max‖Φ‖_P = **1.57**, y **no existe P común ni
para el conjunto ampliado**. El agujero de `hbp_mix` sigue abierto: la LMI
no lo cierra.

**3. El small-gain NO cierra, por un factor ~48.** La condición es
ρ + κ·Δt·L_F < 1, o sea L_F < **0.0192**. La constante de Lipschitz real
del forzamiento, acotada **exactamente desde los pesos** —
L_F ≤ f_gain·‖W₂‖·Lip(SiLU)·‖W₁ₕ‖ con f_gain≈0.058, ‖W₁ₕ‖≈2.1–3.9,
‖W₂‖≈2.8–4.9 — vale entre **0.367 y 0.918**. Ni en el mejor checkpoint se
acerca.

## El mecanismo, que es lo que hace publicable el negativo

El margen de contracción del campo (1−ρ = 0.048) es **demasiado pequeño
frente a la ganancia del forzamiento aprendido** (L_F ≈ 0.9). Ni siquiera
certificando un punto aislado en vez de la envolvente —donde ρ bajaría a
≈0.73 y el umbral subiría a ≈0.13— alcanzaría: seguiría faltando un factor
7. No es que la caja sea un poco ancha; es que la escala del forzamiento y
la del margen están separadas por un orden de magnitud.

**Matiz obligado**: el small-gain es una condición **suficiente**. Que no
se cumpla no prueba que el lazo sea inestable — de hecho no diverge en
entrenamiento. Prueba que **esta técnica no alcanza**, que es distinto y hay
que decirlo así.

## Consecuencia para la sede

**El plan de JMLR no sale.** Lo que buscábamos era la conjunción de segundo
orden + gateado + lazo cerrado, que es el hueco que la literatura deja
vacío. Tenemos el primero y una parte del segundo; el lazo cerrado no, y la
mezcla tampoco. **TMLR sigue siendo el destino**, ahora con un certificado
mejor y una limitación cuantificada en vez de una promesa.

## Lo que sí hay que llevar al paper

1. Sustituir el certificado por radio espectral por el **P común sobre la
   región entrenada** (ρ=0.9517, κ=2.51): es cota en norma y cubre mezcla y
   gating dentro de la rama de onda.
2. Decir que **la caja declarada contiene configuraciones divergentes** y
   estrechar los topes de `HBPConfig` a la región certificada, o declarar
   que la caja es nominal y el certificado operativo es el de runtime.
3. Reportar que `hbp_mix` **no tiene certificado**, ni por radio espectral
   ni por LMI.
4. Reportar el small-gain con sus números: umbral 0.0192 frente a L_F medido
   0.37–0.92, y el mecanismo (margen contra ganancia).
5. Dejar declarado que el lazo por el **host** ni siquiera se intenta: haría
   falta una cota de Lipschitz del transformer respecto de su modulación,
   que no tenemos.
