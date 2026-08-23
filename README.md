# MiuraCognitive

**English** · [Español](README.es.md)

A research program on **where cognition lives** in a minimal but complete
architecture: a recurrent reasoner with adaptive halting, a homeostatic control
field, and a value module. We asked the same question of every part: does this
function **emerge** from gradient descent, or must it be **computed** by
explicit machinery?

The central object is the **Homeostatic Background Processor (HBP)**: a
low-dimensional internal-state field defined over the module graph of a
transformer, governed by a **family** of partial differential equations on the
graph Laplacian — damped wave (forced Klein–Gordon), its diffusive limit, and a
KdV-type dynamics. It advances **one tick per reasoner iteration** and modulates
the reasoner's computation (halting threshold, block gain, memory gates).

    ∂²h/∂t² + 2ζω₀ ∂h/∂t − c²∇²h + ω₀²(h − h*) = f_θ(h,s) + g_φ(h,x)

## The two papers

| | |
|---|---|
| [`paper/main.tex`](paper/main.tex) | **Where Cognition Lives** — the information hierarchy, what emerges and what must be computed, and the readout artifact |
| [`paper/governor_en.tex`](paper/governor_en.tex) | **The field as a compute governor** — the PDE family, the stability certificates, and the deconfounding campaign |

They are mutual companions. `paper/build_all.ps1` builds all four outputs
(authored and double-blind, for each paper) from the same source via the
`\anonfalse` / `\anontrue` switch.

## What the program concludes

Every claim below points to the findings document that supports it.

### Competence emerges; stopping, not as it appeared

At matched mean compute, payoff climbs from `0.467` (uniform allocation)
through `0.546` (difficulty) to `0.698` (ex-ante value). The apparent next rung
— `0.921`, posterior self-observation — **does not survive audit**: PonderNet-style
halting returns a halting-weighted **mixture** of hidden states, while every
forced-depth baseline returns a single state, and the language head is trained
on the mixture alone.

Equalizing the readout **annihilates** the advantage of native execution:
residual regime `+0.000 [0.000, 0.000]`. With the readout fixed and the budget
matched, knowing which instance needs how much compute is worth
`+0.0011 [+0.0003, +0.0019]`. Adding value on top of the posterior buys nothing
(`+0.0002 ± 0.0004`).

→ [`FINDINGS_READOUT.md`](FINDINGS_READOUT.md)

### Value does not emerge: it has to be computed

Trained couplings capture **zero** of an available payoff that an explicit
allocator captures completely (`+0.151`, routing correlation `+0.79`).
Anticipation, by contrast, has no payoff to capture in these families
(`≤+0.001` across 31 configurations). The caveat matters: here the stake is ⊥
to content **by construction**.

### On a frozen LLM, the same instruments

The self-consistency voting ceiling is a **measured** bound:
`+0.0236 [+0.0150, +0.0326]`. Inter-sample agreement is nearly worthless as a
stopping signal — its mass concentrates on wrong answers.

→ [`mhbp/tasks/llm_gov/FINDINGS_LLM.md`](mhbp/tasks/llm_gov/FINDINGS_LLM.md)

### Our own prediction, executed

In a cliff-cost family, value under commitment pays
`+0.1312 [+0.1124, +0.1502]`, roughly **seven times** the point estimate in the
smooth family (`+0.019`). But **not** because the cliff shifts information
toward ex-ante decisions: because it multiplies the attainable range by
`5.1× [3.4, 8.2]`. The magnitude of an allocation problem and the structure of
its information are separate axes.

### The field: substance no, structure only in part, certifiability yes

- **Substance (no).** The *type* of the field's physics is irrelevant for
  accuracy: wave, diffusion, gated mixtures, non-local Poisson-type coupling,
  and even a 2D Navier–Stokes flow substrate all give the same accuracy.
- **Structure (only in part).** Second order endows out-of-distribution compute
  allocation with a robustness that end-to-end learned halting lacks — but a
  preregistered deconfounding campaign with 20 fresh seeds bounds the claim.
  At equalized caps the order effect is strong in one generator family
  (`+0.087 [+0.042, +0.132]`, t=4.0) and **not detected** in the other
  (`+0.014 [−0.013, +0.040]`, n.s.): part of the original contrast was
  **capacity, not order**. A matched-interface GRU is indistinguishable in the
  first family (`+0.006`, n.s.) and **nominally wins** in the second
  (`−0.035 [−0.067, −0.002]`).
- **Certifiability (yes).** What distinguishes the field is not capability but
  that its stability is **provable**, and above all that the one-step operator
  admits an **exact runtime check**.

→ [`FINDINGS_V4.md`](FINDINGS_V4.md) · [`FINDINGS_TEORIA.md`](FINDINGS_TEORIA.md)

### What is certified, and what is not

A **common P exists over the envelope of the trained checkpoints**
(ω₀∈[0.488,0.529], ζ∈[0.475,0.538], c≤0.371): the LMI closes with `ρ=0.9517`,
κ=2.51, and the worst individual spectral radius is `0.7322`. It is a bound in
**norm**, not in spectral radius, so it covers non-normal transients and, by
convexity and submultiplicativity, mixtures and the gated (LTV) case.

What it does **not** close, plainly:

1. **The box declared in `HBPConfig` is not certifiable, because it is not
   stable**: of its 64 vertices, **40 diverge** (worst `ρ=5.24`). What keeps the
   model away from divergence is training, not the caps in the code.
2. The mixture with the diffusive branch remains uncertified (`max‖Φ‖_P = 1.57`).
3. Small-gain does not close, by a factor of ~48.

It is an **integrator certificate**, not a closed-loop one, and the papers say
so explicitly.

→ [`FINDINGS_LMI.md`](FINDINGS_LMI.md) · `experiments/certify_lmi*.py`,
`experiments/verify_verlet_schurcohn.py`

### The field as evidence accumulator: null with a positive control

A kill-gate with a positively controlled probe finds no evidence for the
remaining route across twelve frozen solvers: `ΔAUC = +0.0007 [−0.0065, +0.0079]`
against a pass threshold of `0.03`. Proposed (not demonstrated) reading: the
recurrent state already integrates its own history.

→ [`mhbp/tasks/reasoner_g0/FINDINGS_N4.md`](mhbp/tasks/reasoner_g0/FINDINGS_N4.md)

## Environment

```powershell
$env:PYTHONPATH="."; $env:PYTHONIOENCODING="utf-8"
```

Conda with PyTorch cu128 (RTX PRO 5000 Blackwell, sm_120). Dependencies in
`requirements.txt` and `environment.yml`. The LLM chapter downloads
Qwen2.5-14B-Instruct (Apache-2.0, ~28 GB) on first run.

## Reproduction

This repository contains **only what the two manuscripts need**: the
architecture code, the scripts that run the experiments they report, the
summary results their numbers come from, and the findings documents and
preregistrations that support them. With that, tables and figures regenerate
without retraining anything.

Not here, and where it lives: the per-instance records and the trained
checkpoints (1.8 GB) are in the archived Zenodo deposit whose DOI the papers
cite. Experimental lines that appear in neither manuscript have been withdrawn
from the public tree, so that what remains is unambiguously the papers' material.

```powershell
python verify_setup.py                    # GPU / BF16 / SDPA, with CPU fallback
python experiments/_audit_hbp.py          # HBP regression: grad(ζ,ω₀)≠0, input-dependent modulation
python eval/diagnostics.py                # live VEI, FFT/oscillation, damping regime
python experiments/example_routine.py     # example: hbp_full solving an S₅ composition
```

### Map: paper result → script → results file

| Result | Script | Results |
|---|---|---|
| Information hierarchy | `mhbp/tasks/reasoner_g0/n3_*.py` | `mhbp/tasks/reasoner_g0/results/` |
| **Readout artifact** | `n3_readout.py`, `n3_readout_fair.py` | `results/n3_readout*.json` |
| Explicit vs coupled allocator | `n3_eval.py`, `n3_sonda.py` | `results/n3_eval.json`, `FINDINGS_N3.md` |
| Cliff (N1b) | `mhbp/tasks/llm_gov/llm_n1b*.py` | `results/llm_n1b_*.json` |
| Self-consistency ceiling | `mhbp/tasks/llm_gov/llm_gsm8k.py` | `results/llm_gsm8k.json` |
| v4 deconfounding | `experiments/benchmark_v4.py` | `results_benchmark_v4/` |
| LMI certificates | `experiments/certify_lmi*.py` | `FINDINGS_LMI.md` |
| Verlet criterion | `experiments/verify_verlet_schurcohn.py` | `FINDINGS_TEORIA.md` |
| Accumulator kill-gate | `mhbp/tasks/reasoner_g0/n4_*.py` | `FINDINGS_N4.md` |

### Full benchmark

```powershell
python experiments/benchmark_v3.py         # replication with learnable physics
python experiments/benchmark_v4.py         # deconfounding: equalized caps + GRU
python experiments/benchmark_report_v4.py  # aggregate table
```

### Variants

| Variant | HBP | Reasoner | WM | What it isolates |
|---|---|---|---|---|
| `vanilla` | – | – | – | fixed-depth transformer baseline |
| `gating` | – | ✓ | – | adaptive recurrence alone |
| `gating_wm` | – | ✓ | ✓ | **control**: recurrence + working memory, no HBP |
| `hbp_first` | 1st | ✓ | ✓ | overdamped limit (relaxation) |
| `hbp_full` | 2nd | ✓ | ✓ | full Verlet (inertia/oscillation) |
| `hbp_gru` | – | ✓ | ✓ | **matched-interface** GRU (the honest rival in v4) |

### Tasks

- `permcomp` — composition of S₅ generators (NC¹-hard; the main one).
  Generators: `adjacent` or `cycle_transp`.
- `runsum` — running sum mod n; solvable in one pass, useful as contrast.
- `recall` — recall with distractors (saturated; historical).

Extrapolation: `--train_max_writes 12 --max_writes 24` trains on K≤12 and
evaluates up to K≤24.

## Structure

```
model/        transformer.py · hbp.py ★ · adaptive_depth.py · working_memory.py · miura.py
data/         synthetic_recall.py
training/     config.py · trainer.py
eval/         diagnostics.py · aggregate.py

mhbp/                        the code behind BOTH papers
  tasks/reasoner_g0/         information hierarchy, readout, allocator, N4 kill-gate
  tasks/llm_gov/             frozen LLM actuator: GSM8K, N1b cliff, levers
  analysis/                  FINDINGS_PHASE2.md and successors

experiments/  benchmark_v2/v3/v4 · certify_lmi*.py · verify_verlet_schurcohn.py
              _audit_hbp.py · the mechanism-null studies

paper/        main.tex · governor_en.tex · refs.bib · figures/ · build_all.ps1
              check_anon.py (verifies double-blinding) · pack_arxiv.py
```

## Implementation notes

- The HBP couples to the reasoner via **soft modulation** `gate = 1 + s·tanh(·)`
  around the identity. A `sigmoid` gate (centred at 0.5) decays the signal and
  breaks the iteration.
- Verlet runs in **FP32**: velocity cancellation in BF16 corrupts the damping
  near equilibrium.
- **BF16 freezing bug (critical, retroactive).** `raw_ζ`, `ω₀`, `c` and `f_gain`
  stayed frozen at their init in every GPU/BF16 run, because the Adam step fell
  below ULP/2. The `pin_fp32()` fix is mandatory after `.to(bf16)`. Results
  predating the fix have the physics **fixed, not learned**.
- The ω₀ caps of `hbp_first` were an **artifact of the explicit integrator**: the
  unconditionally contractive implicit kernel makes them unnecessary, which is
  why the fair v4 arm is `hbp_first_eq` at equalized caps.

## Licence

Own code, documents and results under **MIT** (`LICENSE`). Third-party
material — the GSM8K test set, the actuator's generations, and the LaTeX style
files — keeps its own licence and notice: see [`NOTICE`](NOTICE).
