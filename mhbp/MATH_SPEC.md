# mHBP — Especificación matemática cerrada (Fase 1)

> Multiscale Homeostatic Background Processor: plano de control autonómico
> multiescala. Este documento fija TODAS las decisiones matemáticas de la Fase 1;
> el código (`mhbp/`) la implementa literalmente y los tests la verifican.
> Convención del proyecto: los términos biológicos (homeostasis, alostasis,
> interocepción) son analogías funcionales, no atribuciones de estados mentales.

## 1. Campos y estado

Q = 4 campos homeostáticos, indexados q ∈ {fast_executive, risk_priority,
slow_deliberative, resource_metabolic}. Cada campo:

- grafo regulador propio G_q con N_q nodos, laplaciano combinatorio
  L_q ⪰ 0 (simétrico) y matriz de advección orientada A_q = −A_qᵀ;
- estado h_q ∈ R^{B×N_q×d} (batch B, dimensión latente común d);
- punto de consigna h*_q (aprendible; el de los campos rápidos puede
  desplazarse alostáticamente, §6). *Nota Fase 1: h* es INERTE (la dinámica y
  los actuadores operan sobre u = h − h*); se activa en Fase 3, cuando el
  reasoner lea h = h* + u;*
- desviación u_q := h_q − h*_q y velocidad escalada w_q := τ_q u̇_q;
- escala temporal τ_q > 0 con orden impuesto τ_fast < τ_risk < τ_delib < τ_res.

Config MVP: N = (8, 4, 6, 4) en cadena, d = 8, τ = (1, 4, 8, 32).

## 2. Operadores por campo (por dimensión latente ℓ)

    K_qℓ = ω_qℓ² I + c_q² L_q               (rigidez: reacción + difusión espacial)
    C_qℓ = 2 ζ_qℓ ω_qℓ I + D_q L_q          (disipación: uniforme + estructural)
    G_qℓ = b_q A_q + β_q A_q³               (antisimétrico: advección + dispersión)

Rangos seguros (squash; parámetros crudos en FP32, lección BF16 del HBP):
ω ∈ [ω_min, ω_max] (sigmoid), ζ ≥ ζ_min > 0 (softplus), c ∈ [0, c_max],
D ∈ [0, D_max], b ∈ [0, b_max], β ∈ [0, β_max] (sigmoids). Con ζ_min > 0 y
ω_min > 0: K_qℓ ≻ 0 y C_qℓ ≻ 0 **por construcción**, G_qℓ antisimétrica exacta.

## 3. Acoplamiento entre campos: potencial de interfaz (decisión clave)

El prompt plantea B_qr(u_q − T_qr u_r). La forma general no es simétrica y
destruiría el certificado de energía. **Decisión (la más limpia): acoplamiento
derivado de un potencial.** Interfaz por campo:

    y_q := Ŵ_qᵀ ū_q ∈ R^{d_c},   ū_q := p_qᵀ u_q,  p_q := 1_{N_q}/N_q  (media de nodos)

con Ŵ_q := W_q/‖W_q‖_F la interfaz NORMALIZADA (dirección aprendible; la
ESCALA vive únicamente en κ — sin esta caja, ‖W‖ grande arruina el
condicionamiento del núcleo implícito: hallazgo del panel adversarial).
Energía de acoplamiento sobre las aristas E_H del grafo de campos (topología
por defecto: cadena en orden de escala temporal; ablations: completa, nula):

    E_cpl(U) = ½ Σ_{(q,r)∈E_H} κ_qr ‖y_q − y_r‖²,    κ_qr = κ_max·σ(θ_qr) ≥ 0

Fuerza sobre u_q: f_q = −∂E_cpl/∂u_q. En la variable apilada U, esto añade a la
rigidez el término 𝓑 = Σ_e κ_e M_eᵀ M_e ⪰ 0 (Gram), donde M_e extrae y_q − y_r:
**PSD y ACOTADA (‖𝓑‖₂ ≤ Σ_e κ_e(1/N_q + 1/N_r)) por construcción para
cualesquiera parámetros aprendidos**. 𝓑 es un laplaciano de grafo GENERALIZADO
con pesos matriciales de Gram por arista (no un producto L_H ⊗ R con R fija:
con N_q distintos ese Kronecker ni siquiera es formable).

**Correspondencia exacta con la forma B_qr(u_q − T_qr u_r) del plan** (derivada
y verificada numéricamente a 5e−18 por el árbitro): en el layout plano
(nodo ⊗ dim),

    f_q = −B_qr (u_q − T_qr u_r),   B_qr = κ_qr (p_q p_qᵀ) ⊗ (Ŵ_q Ŵ_qᵀ),
    T_qr = (1_{N_q} p_rᵀ) ⊗ (Ŵ_q(Ŵ_qᵀŴ_q)⁻¹Ŵ_rᵀ)      [pseudoinversa si d < d_c]

con la recíproca (B_rq, T_rq) análoga. El potencial NO genera la familia
B_qr/T_qr arbitraria del plan: es la RESTRICCIÓN DELIBERADA al subconjunto
simétrico de rango ≤ d_c sobre medias de nodos, con reciprocidad impuesta —
el precio exacto de conservar el certificado de Lyapunov global.

La DIRECCIONALIDAD de la jerarquía (recursos → riesgo → deliberativo → rápido)
no se pone en el acoplamiento (que es simétrico y conservativo) sino en la
**alostasis** (§6), que es acotada y no rompe el certificado. El acoplamiento
dirigido no-conservativo queda como ablation de fases posteriores (con
certificado ISS, no de Lyapunov).

## 4. Dinámica continua y energía

En tiempo local s_q = t/τ_q, cada campo satisface (M_q = I en Fase 1):

    u̇_q = w_q / τ_q
    ẇ_q = [ −(C_q + G_q) w_q − K_q u_q + f_q^cpl + F_q ] / τ_q

F_q = forzamiento acotado: interocepción g·tanh(·) (‖F‖ ≤ F_max) + alostasis (§6).

**Energía global**

    E(U, W) = Σ_q ½‖w_q‖² + ½⟨u_q, K_q u_q⟩ + E_cpl(U)

**Proposición 1 (disipación global).** A lo largo de las trayectorias:

    Ė = − Σ_q (1/τ_q) ⟨w_q, C_q w_q⟩ + Σ_q (1/τ_q) ⟨w_q, F_q⟩

*Prueba.* d/dt(½‖w_q‖²) = ⟨w_q, ẇ_q⟩; el término giroscópico no trabaja
(⟨w, G w⟩ = 0 por antisimetría); d/dt(½⟨u,Ku⟩) = (1/τ)⟨w, Ku⟩ cancela el de ẇ;
d/dt E_cpl = Σ_q ⟨∂E_cpl/∂u_q, u̇_q⟩ = −Σ_q (1/τ_q)⟨f_q, w_q⟩ cancela el
acoplamiento. ∎

**Corolario (GAS sin forzamiento).** F ≡ 0 ⇒ Ė = −Σ (1/τ_q)⟨w_q, C_q w_q⟩ ≤ 0,
con igualdad sólo en W = 0. E es radialmente no acotada (𝓚 = blockdiag(K) + 𝓑 ≻ 0
porque blockdiag(K) ≻ 0 y 𝓑 ⪰ 0). En el mayor conjunto invariante contenido en
{W = 0}: ẇ = 0 fuerza 𝓚U = 0 ⇒ U = 0. Por LaSalle, el origen es global y
asintóticamente estable **para cualesquiera valores aprendidos dentro de las
cajas seguras** (estabilidad por construcción, no por penalización). ∎

**Proposición 2 (ISS/BIBS).** Sea ‖F_q‖₂ ≤ F_max por campo (para el encoder
interoceptivo, que acota por entrada a f_max: F_max = f_max·√(N_q d)). De la
Prop. 1: Ė ≤ −(2ζ_min ω_min/τ_max)‖W‖² + (√Q·F_max/τ_min)‖W‖ (Cauchy–Schwarz
sobre los Q campos; el peor τ divide a cada término). Por tanto

    Ė < 0   para   ‖W‖ > R* := √Q · (τ_max/τ_min) · F_max / (2 ζ_min ω_min)

y, con 𝓚 ≻ 0, el estado converge a una bola (ultimate bound) de radio
O(R*/λ_min(𝓚)^{1/2}): el sistema es ISS respecto a F. *(El radio sin los
factores √Q y τ_max/τ_min que figuraba en una versión anterior era falso —
contraejemplos del árbitro con Q=2, τ=(1,1.05); corregido.)* ∎

## 5. Separación de escalas: tres modos

1. **Fijas**: τ = (1, 4, 8, 32).
2. **Aprendibles con orden garantizado**: softmax sobre Q+1 logits — el último
   es un GAP FANTASMA que absorbe la cola hasta 1 (sin él, cumsum(softmax)
   termina en 1 y τ_Q queda clavada en τ_max; hallazgo del code-review) —,
   s_q = (cumsum_q + g·q)/(1 + g·(Q+1)), τ_q = τ_min + (τ_max − τ_min)·s_q.
   Orden estricto y acotación **por construcción** para todo a ∈ R^{Q+1};
   la construcción va en FP64 (en FP32 el cumsum desborda τ_max ~1e−6).
   El init INVIERTE el mapa completo: el modo learnable arranca EXACTAMENTE en
   taus_init (si es factible con la separación g; si no, en el punto factible
   más próximo) — imprescindible para A/B fixed-vs-learnable limpios.
3. **Condicionadas por contexto**: la misma construcción con a ← a + red(ctx)
   (media del lote → τ nominal común; logits saneados con nan_to_num+clamp).
   El integrador se re-prepara cuando llega ctx; el certificado espectral se
   evalúa en el τ nominal. Flag experimental, off por defecto en Fase 1.

## 6. Homeostasis y alostasis

- **Homeostasis**: h*_q aprendible, cuasi-estático (lr bajo).
- **Alostasis**: el setpoint del campo rápido se desplaza en función de los
  campos lentos: Δh*_fast(t) = ε_a · tanh(Ψ(ū_risk, ū_delib, ū_res, ctx)),
  con Ψ una MLP pequeña y ε_a ≪ 1. Implementación: forzamiento
  F_allo = K_fast · Δh*(t). **Precisión (árbitro):** esto equivale EXACTAMENTE
  a desplazar el setpoint solo con 𝓑 = 0; con acoplamiento, el atractor real es
  𝓚⁻¹·embed(K_fast Δh*) (desviación O(‖𝓑‖/λ_min(𝓚)), ~2% en el MVP, y los
  campos lentos también se desplazan). Lo que se garantiza SIEMPRE:
  ‖Δh*‖_∞ ≤ ε_a ⇒ F_allo acotado ⇒ Prop. 2 (ISS): **la alostasis mantiene el
  estado ACOTADO**. Con ganancia alta de Ψ el lazo cerrado (que queda FUERA del
  mapa lineal certificado) puede sostener ciclos límite acotados — verificado
  adversarialmente: "acotado" ≠ "convergencia a punto"; la condición de pequeña
  ganancia del lazo (‖K_fast‖·ε_a·Lip(Ψ) < margen de contracción) es trabajo de
  Fase 2. Anti-bypass: penalización de tasa ‖Δh*_t − Δh*_{t−1}‖² y de norma
  (con Δh*_{t−1} DETACHED: sin encadenar grafos entre backwards), y ablation
  Ψ=0 obligatoria.

## 7. Integrador Cayley-IMEX (principal)

Tick global Δt; paso local h_q = Δt/τ_q. Splitting de Strang:

    (1) w_q ← Q_q(h_q/2) w_q                     [media rotación giroscópica]
    (2) resolver GLOBALMENTE (backward-Euler, θ=1):
          (I + H C_blk + H 𝓚 H) W⁺ = W − H 𝓚 U + H F
          U⁺ = U + H W⁺
        con H = diag(h_q I), C_blk = blockdiag(C_qℓ), 𝓚 = blockdiag(K_qℓ) + 𝓑
    (3) w_q ← Q_q(h_q/2) w_q

**Cayley**: Q(a) = (I + (a/2)G)⁻¹(I − (a/2)G) aproxima exp(−aG) y satisface
QᵀQ = I **exactamente** (G antisimétrica ⇒ (I−X)ᵀ = (I+X) y los factores
conmutan). La parte giroscópica no puede inyectar energía ni en el discreto —
elimina de raíz la condición "genuinamente discreta" (i) del Verlet del HBP
(Prop. 2 del paper original), que era su defecto.

**Parte implícita**: la matriz A = I + H C_blk + H 𝓚 H es simétrica ≻ 0
(H C_blk = diag(h_q C_q) ≻ 0; H𝓚H congruencia de 𝓚 ≻ 0) ⇒ resoluble y el paso
homogéneo es contractivo (backward-Euler sobre un sistema disipativo). El
acoplamiento va DENTRO del núcleo implícito (sin hipótesis de conmutación),
como el IMEX del HBP original (Prop. 3), ahora extendido a campos acoplados
con escalas distintas mediante la congruencia H𝓚H.

**Proposición 3 (estabilidad discreta).**
*(θ = 1, BE — el caso por defecto).* El mapa lineal homogéneo de un tick
Φ = [Cayley/2] ∘ [BE] ∘ [Cayley/2] cumple ρ(Φ) < 1 para todo parámetro en las
cajas seguras (ζ_min > 0): el Cayley es exactamente ortogonal (no toca U ni la
energía) y el BE es estrictamente disipativo; E constante en una órbita forzaría
W⁺ = 0 y punto fijo ⇒ 𝓚U = 0 ⇒ z = 0. Margen empírico bajo ataque: 1−ρ ≥ 2e−4
en 840+ configuraciones adversarias (dt hasta 50, κ al tope, grafos completos).

*(θ < 1, CN — RESONANCIA GIROSCÓPICA; hallazgo crítico del panel).* Con θ=½
existe un modo NEUTRO exacto (ρ = 1, energía conservada) DENTRO de las cajas:
si la media rotación de Cayley alcanza π/2 — condición (h_q/4)·μ = 1 para algún
autovalor iμ de G_q —, Q² tiene autovalor −1 que compone con el modo flip de
Crank–Nicolson (que solo ve el punto medio y queda ciego al amortiguamiento) en
un autovalor +1 genuino. Por ello, para θ < 1 se EXIGE la condición
anti-resonancia   h_q·σ_max(G_q) < 4·(1−δ)   (δ=0.05), verificada en
`prepare()` (rechaza la configuración si se viola). Con ella, ρ(Φ) < 1 también
para θ=½.

*Verificación*: el certificado no se deja como cota analítica conservadora sino
que se calcula EXACTAMENTE sondeando Φ columna a columna (z = [U; W] ∈
R^{2n_tot}, n_tot ≈ 176 en MVP) y tomando ρ por autovalores — cubre
acoplamiento, escalas y el RADIO del caso no-normal. *Matiz no-normal*: ρ < 1
da el decaimiento asintótico; la amplificación TRANSITORIA de un mapa no normal
no la acota ρ sino la disipación de energía (θ=1: E no creciente ⇒ transitorio
acotado en la norma de energía); `transient_growth_check` mide max_k ‖Φ^k‖₂.
Orden de consistencia: 1 (BE); θ=½ da orden ~2 (test de convergencia cubre
ambos). El condicionamiento del núcleo implícito queda acotado por la caja de
κ y la normalización de Ŵ (§3); `structural_certificate` reporta cond(A_θ).

**Integrador de referencia**: Verlet amortiguado (compatibilidad con el HBP
original), coupling explícito — condicionalmente estable; sólo comparación.

## 8. Interocepción y actuadores (contratos de la Fase 1)

- `InteroceptiveSignal`: 26 canales canónicos + máscara de validez; normalización
  running (EMA de media/varianza); el encoder produce F_q = g_q·tanh(U_q s) con
  g_q ≤ F_max (entra en la Prop. 2). En tareas de dificultad no trivial la
  longitud K NO se expone (regla anti-leakage del proyecto).
- Actuadores (MVP 4): sesgo de halting (continuo), profundidad máxima (STE),
  gate de herramienta ∈ [0,1], escala de presupuesto. Forma general
  a = Π_A[a_0 + Σ_q W_q^{act}[ū_q; w̄_q]] con Π_A proyección a intervalo
  (sigmoid escalada) ⇒ acotación por construcción; log de saturación; API de
  intervención causal (congelar campo / fijar actuador / permutar señales).

## 9. Criterios numéricos de aceptación (tests)

| Test | Criterio |
|---|---|
| Cayley ortogonal | ‖QᵀQ − I‖_∞ < 1e−12 (FP64) |
| Identidad de energía continua | residuo Ė-fórmula < 1e−10 (FP64; matrices ensambladas Y autodiff sobre integ.energy) |
| Disipación discreta | F=0 ⇒ E_{t+1} ≤ E_t (+1e−12) en 100% de ≥50 CONFIGS random (params re-aleatorizados), θ=1 |
| Certificado exacto | ρ(Φ) < 1 en ≥40 configs seguras random; ρ ≥ 1 detectado al violar la estructura |
| Anti-resonancia | el repro exacto del ataque (θ=½, dt=20, cajas al borde) es RECHAZADO por prepare() |
| Caja de la interfaz | W~1e6: ‖𝓑‖ acotada, sin crash de Cholesky, ρ < 1 |
| Convergencia | error vs expm(SΔT): pendiente ∈ [0.85, 1.4] (θ=1; BE con prefactor variable), ∈ [1.7, 2.4] (θ=½) |
| Oscilador analítico | error < 1%·envolvente con h = 0.01 (θ=½) |
| gradcheck | FP64, 2 ticks, atol 1e−6; gradiente fluye a todos los grupos activos |
| Batch/determinismo | batch de copias == individual (1e−12); dos runs bitwise |
| Escalas | τ estrictamente creciente para 200 raws random, 3 modos; init learnable == taus_init |
| Conmutación (switched) | cambio de parámetros in-place con estado vivo ⇒ re-prepare automático y energía acotada |
| Transitorio no-normal | max_k ‖Φ^k‖₂ finito y reportado con b,β al tope (θ=1: E no creciente lo acota) |
| Interocepción | canal ausente ⇒ contribución 0 exacta; leak_mask veta; ‖F‖_∞ ≤ f_max bajo ±1e6 |
| Actuadores | dentro de límites bajo estados extremos (±1e3); freeze/override efectivos |
| Checkpoint | trayectoria bitwise tras state_dict + save/load_dynamic_state (incluye alostasis) |
| Certificados puros | stability_report NO altera el estado vivo del modelo |
| CPU/GPU | misma trayectoria FP64 CPU vs CUDA a 1e−9 (skip si no hay GPU) |
| NaN | protección activa: estados finitos o excepción explícita |

## 10. Correspondencia con el HBP original

Con Q=1, τ=1, acoplamiento nulo, Ψ=0 y el integrador Verlet de referencia, el
mHBP se reduce a la rama de onda del HBP (misma K, C, G giroscópica). El avance
de Fase 1 sobre el HBP: (i) Cayley elimina la inyección de energía discreta del
giroscópico; (ii) el acoplamiento certificado entre campos; (iii) escalas
ordenadas por construcción; (iv) alostasis acotada. Todo lo demás (actuadores
ricos, RAG, recursos duales §12 del prompt) es Fase ≥ 4.
