# Metadatos de arXiv — para copiar y pegar

Abstracts en texto plano, con todas las macros expandidas y sin marcado de
LaTeX. arXiv no interpreta macros propias: si quedara una sola sin expandir la
mostraría literal en la página de resumen.

Campos que van **en blanco** en los dos: *Report number* (no hay número local),
*Journal reference* (no está publicado), *External DOI* (no hay DOI de revista)
y **MSC class** (restringido a los archivos de matemáticas). El **ACM class**
solo vale si la categoría PRIMARIA está en el archivo `cs`. El DOI de Zenodo del código **no** va en *External DOI*: ese campo es
solo para la versión de revista del artículo.

---

# 1. Cognición  (`submit/7975103`)

**Categoría primaria:** `cs.LG` · **Cross-list:** `cs.AI`, `cs.CL`

### Title

```
Where Cognition Lives: Dissecting Emergent from Computed Function in a Minimal Complete Cognitive Architecture
```

### Author(s)

```
Francisco M. Arrabal-Campos, Francisco G. Montoya, Alfredo Alcayde, Ignacio Fernandez
```

### Abstract

**1895 caracteres**, dentro del limite de 1920 que impone arXiv. Todas las macros expandidas, sin marcado de LaTeX, un solo parrafo.

```
A cognitive architecture must decide not only how to reason but how long to think and what deserves the effort. We built a minimal but complete system - recurrent reasoner with adaptive halting, homeostatic field, value module - and asked of each part: does it emerge from gradient descent, or must it be computed? Competence emerges. Stopping emerges too, and appears to be worth more than everything decidable in advance, but that appearance is instrumentation: payoff at matched mean compute climbs from 0.467 (uniform) through 0.546 (difficulty) to 0.698 (ex-ante value), and the further climb to 0.921 (posterior self-observation) does not survive audit. PonderNet-style halting returns a halting-weighted mixture of hidden states while forced-depth baselines return one, and the language head is trained on the mixture alone; equalizing the readout annihilates that advantage (residual +0.000 [0.000, 0.000]). Value does not emerge: trained couplings capture zero of a payoff an explicit allocator captures completely (+0.151), so the second-order decisions that do pay must be computed, at least where value is orthogonal to content, as here. On a frozen LLM actuator, self-consistency voting is a measured bound (+0.0236) and inter-sample agreement is nearly worthless as a stopping signal, its mass concentrating on wrong answers. Almost every negative carries a mechanism and a positive control. Executing our own falsifiable prediction, value under commitment pays +0.1312 [+0.1124, +0.1502] in a cliff-cost family, seven times the smooth-family estimate - not because the cliff shifts information toward ex-ante decisions (a post-hoc comparison compatible with equality but imprecisely estimated), but because it multiplies the attainable range fivefold (5.1x [3.4, 8.2]): magnitude and information structure are separate axes, the first measured decisively, the second only bounded.
```

<details>
<summary>Abstract completo del manuscrito (3453 caracteres) — NO cabe en arXiv, se guarda como referencia</summary>

```
A cognitive architecture is more than the module that reasons. It must also decide how long to think, what deserves the effort, and when looking is better than planning. We built a minimal but complete system - a recurrent reasoner with adaptive halting, a homeostatic control field, and a value module - and spent a research program asking one question of every part: does this function emerge from gradient descent, or must it be computed by explicit machinery? The answers are sharp. Competence emerges. Stopping emerges too, and appears to be worth more than everything decidable in advance, but that appearance is instrumentation: in our information hierarchy, payoff at closely matched mean compute climbs from 0.467 (uniform allocation) through 0.546 (difficulty) to 0.698 (ex-ante value), which matches the class-stake oracle to three decimals, an equality that is partly by construction, since our stake sensor is perfect. The further climb to 0.921 (posterior self-observation) does not survive audit: PonderNet-style halting returns a halting-weighted mixture of hidden states while every forced-depth baseline returns a single state, and the language head is trained on the mixture alone. Equalizing the readout annihilates the apparent advantage of native execution (residual +0.000 [0.000, 0.000]), and with the readout held fixed and the budget matched, knowing which instance needs how much compute is worth +0.0011 [+0.0003, +0.0019]. Adding value on top of the posterior likewise buys nothing (+0.0002 +/- 0.0004). Value does not emerge: trained couplings capture zero of an available payoff that an explicit allocator captures completely (+0.151, routing correlation +0.79), while anticipation has no payoff to capture at all in these families (<= +0.001 across 31 configurations), so the second-order decisions that do pay must be computed, at least where value is orthogonal to content as it is here by construction. On a frozen LLM actuator the same instruments show that the standard test-time lever, self-consistency voting, is a measured bound (+0.0236 [+0.0150, +0.0326]) and that inter-sample agreement is nearly worthless as a stopping signal - its mass concentrates on wrong answers - so watching oneself think is nearly worthless in both regimes, a convergence that was invisible while one of the two was measured through a readout the other did not share. Almost every negative result in the program carries its mechanism, the exceptions are declared as unadjudicated, and the protocol that produced them - preregistration, adversarial panels, kill-gates, replication rules, and positive controls for every null we assert - is part of the contribution. We close by executing our own falsifiable prediction. In a cliff-cost family, where the compute an instance needs is invisible until it is spent, value under commitment pays +0.1312 [+0.1124, +0.1502], roughly seven times the point estimate (+0.019) in a smooth family, yet not because the cliff shifts information toward ex-ante decisions (the ex-ante fraction of the attainable range is 0.759 versus 0.763; difference -0.004 [-0.150, +0.241], a post-hoc observation compatible with equality but imprecisely estimated), but because it multiplies the attainable range fivefold (5.1x [3.4, 8.2]). On this evidence the magnitude of an allocation problem and the structure of its information behave as separate axes: the first measured decisively, the second only bounded.
```
</details>


### Comments

```
16 pages, 3 figures. Code, preregistrations and results: https://github.com/fmarrabal/miuracognitive
```

### ACM class

```
I.2.6; I.2.8
```

### MSC class

**En blanco.** MSC-class esta restringido a los archivos de MATEMATICAS; un
envio con primaria `cs.LG` lo rechaza.

---

# 2. Gobernador  (segundo envío)

**Categoría primaria:** `eess.SY` (o `cs.LG` si `eess` te sigue pidiendo aval)
· **Cross-list:** `cs.LG`, `cs.AI`

### Title

```
Can a Dynamic Internal Field Govern a Transformer's Cognition? Certifiability, not Superiority, in Homeostatic Compute Control
```

### Author(s)

```
Francisco M. Arrabal-Campos, Francisco G. Montoya, Alfredo Alcayde, Ignacio Fernandez
```

### Abstract

**1909 caracteres**, dentro del limite de 1920 que impone arXiv. Todas las macros expandidas, sin marcado de LaTeX, un solo parrafo.

```
An intelligent system does not merely reason: it governs its own reasoning - how much to compute, when to stop, which module to activate. Can that role be played by a dynamic internal field - a low-dimensional homeostatic state with explicit physics and certified stability - that modulates cognition without performing it? Ours is a field on the module graph governed by a family of PDEs on the graph Laplacian, advancing with an adaptive-depth reasoner. We certify the stability of the integrator of the whole family - an integrator certificate, not a closed-loop one. New, and proved here: a discrete Schur-Cohn criterion for Verlet with velocity coupling, necessary and sufficient per latent root, with no commutation hypothesis. The answer is threefold: substance no, structure only in part, certifiability yes. The type of the field's physics is irrelevant for accuracy: wave, diffusion, gated mixtures and a 2D Navier-Stokes substrate tie. A twenty-seed preregistered deconfounding campaign bounds the structural claim: at equalized caps the second-order effect is strong in one family (+0.087 [+0.042, +0.132], t=4.0) but is not detected in the other (+0.014 [-0.013, +0.040], n.s.), so part of the original contrast was capacity, not order; and a matched-interface GRU is indistinguishable in the first and nominally exceeds the field in the second (-0.035 [-0.067, -0.002]). What distinguishes the field is not capability but that its one-step operator admits an exact runtime stability check - a difference of kind, not of existence: learned recurrences carry certificates too, sufficient and conservative ones. A kill-gate with a positive control finds no evidence for the field as evidence accumulator (Delta AUC +0.0007 [-0.0065, +0.0079] vs a 0.03 threshold). A dynamic internal field is a viable, certifiable compute governor, but not an enhancer of cognition: it modulates, it does not think.
```

<details>
<summary>Abstract completo del manuscrito (5283 caracteres) — NO cabe en arXiv, se guarda como referencia</summary>

```
An intelligent system does not merely reason: it governs its own reasoning - how much to compute, when to stop, which module to activate. We ask whether that role of metacognitive governor can be played by a dynamic internal field: a low-dimensional homeostatic state, with explicit physics and certified stability, that modulates a transformer's cognition without performing it. Our instance is the Homeostatic Background Processor (HBP): a field defined over the module graph of a transformer and governed by a family of partial differential equations on the graph Laplacian - damped wave (forced Klein-Gordon), its diffusive limit, and a KdV-type dynamics - that evolves along the iterations of an adaptive-depth reasoner and modulates its computation (halting threshold, block gain, memory gates; the interface also exposes a router-bias head, not consumed by the models reported here) through a universal interface of interoception and modulation. We characterize the stability of the integrator of the whole family, and we are explicit that this is an integrator certificate and not a closed-loop one. Two of the three ingredients are classical and we recall them with attribution: a placement dichotomy for antisymmetric operators (gyroscopic in the second-order branch, positional in the first), which is Kelvin-Tait-Chetaev and, in its first-order half, exactly Anti-Symmetric DGN with gamma*I generalized to K; and a coercivity bound giving an unconditionally contractive implicit kernel, which is the logarithmic-norm resolvent estimate. The circulatory branch produces flutter with threshold beta*rho(A^3) < 2*zeta*omega_0^2, exact only when stiffness and damping are both multiples of the identity - a hypothesis none of our runs satisfies, and under which the threshold is Bottema's criterion. What is new is narrower, and we now prove it: a discrete Schur-Cohn criterion with complex coefficients for the Verlet integrator with velocity coupling, necessary and sufficient per latent root and requiring no commutation hypothesis, which shows that a backward-differenced gyroscopic term does not inherit the neutrality it has in continuous time. Empirically (S_5, NC^1-hard; pre-registered protocol, n=10, plus a twenty-seed preregistered deconfounding campaign that partly overturned it), the answer is threefold: substance no, structure only in part, certifiability yes. Substance (no): the type of the field's physics is irrelevant for accuracy - wave, diffusion, gated mixtures, non-local Poisson-type coupling and even a 2D Navier-Stokes flow substrate all give the same accuracy; the evidence is consistent with the gradient laminating every modulation demand to a quasi-static set-point, and incompressible flow hits a physical ceiling (div u = 0 forbids concentrating information). Structure (only in part): the second order of the dynamics endows out-of-distribution compute allocation with a robustness the end-to-end learned halting control (gating_wm) lacks, and - after fixing a BF16 bug that froze the physical parameters - replicates unchanged; but a preregistered deconfounding campaign bounds the claim. With twenty fresh seeds, with the first-order branch rebuilt at equalized caps (which the unconditionally contractive implicit kernel makes safe, so the original caps were an artifact of the explicit integrator), and with a matched-interface GRU replacing the physical integrator, the order effect is strong in one generator family (+0.087 [+0.042, +0.132], t=4.0) and is not detected in the other (+0.014 [-0.013, +0.040], n.s. - an interval that excludes the original v3 estimate but not a small positive effect): part of the original contrast was capacity, not order. The GRU governor is statistically indistinguishable from the field in the first family (+0.006 [-0.051, +0.062], n.s.) and nominally exceeds it in the second (-0.035 [-0.067, -0.002]), with mutual accuracy non-inferiority throughout. Certifiability (yes): what distinguishes the field is therefore not capability but that its stability is provable - placement dichotomy, flutter threshold exact under K = omega_0^2 I, unconditional contraction bound - and, above all, that the one-step operator admits an exact runtime check. That is a difference of kind, not of existence: learned recurrences do carry certificates, and for gated recurrent units in particular there are explicit ISS and incremental-ISS conditions on the weights; they are sufficient and conservative, and obtaining them requires bespoke machinery, whereas the field's spectral radius is read off directly. We conclude that a dynamic internal field is a viable and certifiable compute governor - the "brainstem" of a cognitive architecture - but not an enhancer of cognition: it modulates, it does not think. A kill-gate with a positively controlled probe finds no evidence for the remaining route, the field as temporal evidence accumulator, on twelve frozen solvers of a companion substrate (Delta AUC = +0.0007 [-0.0065, +0.0079] against a 0.03 pass threshold); we read this as the recurrent state already integrating its own history, a mechanism we propose rather than demonstrate. The null, broad and with a mechanism, delimits which role a homeostatic field can and cannot play in an adaptive transformer.
```
</details>


### Comments

```
22 pages, 4 figures. Companion paper: "Where Cognition Lives" (arXiv:XXXX.XXXXX). Code, preregistrations and results: https://github.com/fmarrabal/miuracognitive
```

### ACM class

**En blanco.** El campo ACM-class solo existe para el archivo `cs`; este envio
va con primaria `eess.SY` y lo rechaza, sean cuales sean los codigos. Si
acabas invirtiendo las categorias (primaria `cs.LG`, cross-list `eess.SY`),
entonces si vale: `I.2.6; I.2.8; G.1.8`.

### MSC class

**En blanco.** Restringido a los archivos de MATEMATICAS.

---

## Después de que salgan los identificadores

1. En *Comments* del gobernador, sustituir `arXiv:XXXX.XXXXX` por el
   identificador real de cognición. Si ya está enviado cuando subas el
   segundo, ponlo directamente.
2. Actualizar `arrabal2026cognition` y `arrabal2026governor` en
   `paper/refs.bib` con los dos identificadores, reconstruir con
   `paper/build_all.ps1` y subir la **v2** de ambos.
