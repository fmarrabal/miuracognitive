# Pre-registro — N3: la decisión de cómputo como ACTO EXPLÍCITO

> **v2 — 2026-08-05.** La v1 (git 4ccefda) fue atacada por un panel de 6
> lentes (~30 hallazgos, 9 críticos; cosecha en panel_n3_hallazgos.json),
> tres de ellas con cálculo cuantitativo propio. Adjudicación en §9. GATE
> DURO: nada corre sobre la eval congelada hasta que sonda/fits/λ estén
> fijados con timestamp.

## 1. Hipótesis (re-alcanzada tras el panel)

**HN3**: sobre un solver de competencia general CONGELADO (los 12 ckpts
blind_flat de N2), una asignación explícita de cómputo computada de MODELOS
INTERNOS aprendidos — stakê(input) del sensor certificado + p̂(éxito|input,n)
aprendido de AUTO-SONDAS — supera con margen δ0 a la misma asignación ciega
al valor, a cómputo emparejado. Nota de alcance (lente #ramas): con
features de input, p̂ es una TABLA aprendida de las propias sondas (un
automodelo grueso, no fino por-instancia) — se declara así; la versión con
features de estado (tick-2) queda como ablación opcional post-A.

## 2. Diseño — fase A (forward-only) sobre los 12 ckpts congelados

**Features del asignador (SOLO input; decisión ANTES del forward)**:
emb(slot0) ⊕ emb(slot1) (la señal de stake vive ahí: probe 1.00, N2) +
longitud (K-proxy exacto). Sin piso n≥2, sin doble forward, contabilidad
de cómputo EXACTA (n_total = n_i; el nativo paga lo suyo — A3 justo por
construcción).

1. **Sonda** (por checkpoint; stream 'train' seed 1999, disjunto y con
   dedupe por hash de contenido contra val/eval): 1024 instancias × malla
   n∈{1,2,3,4,5,6,8,10,12,16,20,24} con forced_steps → (input-feats, K,
   stake, n, correcto). **Pre-requisito T-N3a**: fix del caché de valor en
   la ruta forced (hoy value_tick exige active_idx None → el canal queda
   ausente y el sistema sondado NO es el nativo — verificado en código) +
   test de bit-igualdad forced(n) vs prefijo del rollout nativo (n∈{2,3,8,24}).
2. **p̂(éxito | feats, n)**: cabeza MONÓTONA POR ARQUITECTURA
   p̂ = σ(a(f)·(n − b(f))), a>0 vía softplus — mata la minería del argmax
   (optimizer's curse con 24 candidatos) y b(f) es la «profundidad
   requerida» interpretable. Fit POR CHECKPOINT, split POR INSTANCIA
   (group-hash), validación cruzada.
3. **stakê(feats)**: target = payoff de sondas con correcto=1 (ahí payoff
   ∈{1,8} revela el stake sin dividir por p̂; los payoffs de la sonda usan
   el stake VERDADERO ss del generador — declarado). Chequeos VG-N3b:
   AUC ≥ 0.9 held-out Y escala E[ŝ|alto]/E[ŝ|bajo] ∈ [4,12] Y
   |corr(ŝ, K | stake)| < 0.1 (la contaminación por dificultad que la
   lente cuantificó en 3.3× con el target crudo).
4. **Decisión**: n_i = argmax_{n∈1..24} [ŝ_i·p̂_i(n) − λ·n]; λ POR
   checkpoint×brazo en el split 'val' (seed 2999); **emparejado por
   LOTERÍA de λ adyacentes** (Bernoulli por instancia entre λ⁻/λ⁺) hasta
   |ΔE[n̄]| ≤ 0.05 ticks VERIFICADO EN EVAL (la bisección pura salta hasta
   0.83 ticks — medido); ANCOVA payoff~E[n̄] portada de N2 como
   sensibilidad (si el signo cambia → «no interpretable», rama).

**Brazos** (mismos 12 ckpts, eval congelada seed 999 ampliada a m=16384 —
forward-only, el sobremuestreo del estrato alto con reponderación insesgada
baja la varianza 2-3×):
| brazo | decisión | rol |
|---|---|---|
| alloc_uniforme | n = cte | suelo |
| alloc_dificultad | argmax s̄·p̂(n) (ciego al stake) | baseline de A1 |
| **alloc_expl** | argmax ŝ·p̂(n) | el test |
| alloc_regla | n(stake) de dos niveles (sin p̂) | ¿sobra el automodelo? (descriptivo) |
| alloc_oraculo_clase | DP por clase (K×stake) con acc held-out ISOTÓNICA + stake verdadero | techo NO clarividente |
| halting nativo | el PonderNet del ckpt | statu quo (A3) |

El oráculo POR-INSTANCIA (clarividente de la solucionabilidad: elige n_min
de la etiqueta realizada — infla el techo +0.05-0.08, mismo orden que TODO
el headroom, y con eval determinista es degenerado) queda PROHIBIDO en
gates y contrastes; solo descriptivo etiquetado «techo de solucionabilidad».

## 3. Contrastes (primario sobre 6 MEDIAS POR SEED — las 12 celdas
## comparten init por seed, pseudo-replicación declarada; las 12 como
## sensibilidad con ICC reportado)

- **A1 (primario)**: payoff(alloc_expl) − payoff(alloc_dificultad) > **δ0
  = 0.02** (superioridad con margen: en forward-only la significancia es
  gratis — σ_d esperada ~0.004 — y el TAMAÑO es el test; δ0 ≈ ¼ del
  headroom VG1). t pareada por seed unilateral α=0.05 **Y co-primario
  bootstrap POR INSTANCIA estratificado por stake** (la eval común es
  ruido COMPARTIDO entre celdas que el t pareado no ve): ambos deben
  excluir δ0.
- **A2 (lineal, no cociente — la lección C2 de N2 portada)**:
  (expl − dificultad) − 0.5·(oraculo_clase − dificultad) ≥ 0, t pareada
  por seed. [«fracción sustancial» operacionalizada; sin Fieller vacuo]
- **A3 (gatekeeping: solo si A1 pasa; α=0.05 unilateral)**: alloc_expl vs
  halting nativo a E[n̄] emparejado exacto. **Rama A1✓∧A3✗ pre-declarada**:
  «el mecanismo explícito captura headroom sobre asignadores estáticos
  pero NO supera la política por gradiente — adopción BLOQUEADA, HN3
  parcial» (el nativo decide con el estado posterior; las features ex-ante
  no lo tienen — sería un resultado sobre INFORMACIÓN, no mecanismo).
- Suelo de cordura (guardarraíl): alloc_expl ≥ alloc_uniforme (IC).
- Descriptivos: corr(n_i, stake|K); fracción de triage (n=1) por brazo;
  alloc_regla vs alloc_expl; techo de solucionabilidad; payoff(E[n̄]).

## 4. Kill-gates (orden estricto; CPU/GPU-min salvo la sonda)

- **T-N3a (cableado, ANTES de la sonda)**: fix del caché forced + canal de
  valor en régimen idéntico sonda↔nativo + bit-igualdad de prefijos.
- **VG-N3a (automodelo)**: p̂ calibrado en held-out POR INSTANCIA (Brier
  skill ≥ 0 vs tabla de clase de la propia sonda; AUC solo donde ambas
  clases ≥50 muestras). Monotonía: garantizada por arquitectura (el gate
  de monotonía-media de v1 era vulnerable a acc no monótona real).
- **VG-N3b (sensor)**: los tres chequeos del §2.3.
- **VG-N3c (headroom REALIZADO — el que decide)**: sobre las TABLAS de la
  sonda held-out isotónica de los 12 ckpts reales (DP exacto, CPU):
  headroom de VALOR por clase = payoff(oraculo K+stake) −
  payoff(oraculo K-solo) a E[n̄] emparejado, para **e ∈ {3, 4, 5}**
  (excluido e ≥ E[n̄] nativo 5.42 — la escasez no ligaría). Umbral: ≥
  max(0.04, 4·SE_eval). La lente de saturación predice que puede salir
  ROJO (blind_flat satura: el headroom del perfil F3a viene de una clase
  de competencia MÁS DÉBIL — la premisa +0.087 de v1 no es transferible);
  rama roja: «sin headroom de valor a e≥3 EN ESTE SOLVER — la competencia
  disolvió la asignación en este entorno» → hallazgo terminal de la serie
  (cierra N3-en-S₅) y el test del mecanismo exige el entorno con
  acantilados (la deuda N1b), hipótesis nueva y prereg nuevo.
- **VG-N3d (ejecución justa)**: alloc_dificultad con λ bisecado a
  E[n̄]=5.42 rinde ≥ payoff nativo − 2σ en los 12 ckpts (si la ejecución
  forzada per se pierde contra el régimen nativo, A3 se re-etiqueta:
  compararía ejecución, no decisión).
- Piloto σ_d (1 ckpt, minutos) → MDD real por enmienda fechada antes del
  resto.

## 5. Streams y congelación

sonda=seed 1999 (train) / λ y gates=seed 2999 (val) / eval=999 INTOCADA,
m=16384, dedupe por hash contra ambos; p̂, ŝ y λ congelados con timestamp
ANTES de la primera pasada por eval; 12 fits por-checkpoint (sin pooling:
acoplaría los errores de las celdas y rompería el pareado).

## 6. Coste

Sonda 12 ckpts × 12288 forwards forced (~2-3 h GPU) + fits (GPU-min) +
eval 5 brazos × 12 ckpts sobre m=16k (~2 h) ≈ **4-6 h GPU total, cero
re-entrenos**. Fase B (integración online + gating_endo puente): SOLO por
enmienda si A confirma, con presupuesto propio.

## 7-8. Ramas (esqueleto §6-v1 + las nuevas del panel)

A1✓(margen)∧A2✓∧A3✓ → HN3: el mecanismo explícito se adopta; fase B.
A1✓∧A3✗ → adopción bloqueada (información ex-ante insuficiente; se
registra). A1✓∧regla≈expl → el automodelo sobra: se adopta la REGLA (más
simple, misma tesis de mecanismo explícito). A1✗ con gates verdes → ni el
mecanismo explícito captura el headroom con estas features: se audita la
descomposición (ŝ vs p̂ vs λ) pre-declarada. VG-N3c rojo → cierre de
N3-en-S₅ («la competencia barre la asignación»), pivote a la tarea con
acantilados. VG-N3d rojo → re-etiquetado de A3.

## 9. Registro de los 9 críticos → decisión

oráculo clarividente → clase+isotónica, prohibido en gates (×3 lentes);
headroom congelado no transferible → VG-N3c sobre sonda real con menú
{3,4,5} y rama roja terminal (×2); ruta forced apaga el canal y no cachea
→ T-N3a fix+test (×2, verificado en código); piso n≥2/doble forward →
features de input, decisión pre-forward (×2); VG-N3a pasable por K → Brier
skill vs tabla + brazo regla (×1); σ_run inaplicable → δ0=0.02 + piloto
σ_d + primario por seed + bootstrap por instancia (×2); λ-bisección salta
→ lotería + 0.05 ticks + ANCOVA (×2); stakê contaminado por dificultad →
target correcto=1 + chequeos de escala y |corr|<0.1 (×1); A2 cociente →
contraste lineal θ=0.5 (×1).
