"""
EJEMPLO CONCRETO DE RUTINA — MiuraCognitive (variante hbp_full)
================================================================
Demuestra el sistema completo resolviendo composiciones en S_5, replicando el
setup del paper (entrena K<=12, evalúa hasta K<=24 -> el bucket 'largo' es
EXTRAPOLACIÓN). Traza legible de:
  1) entrenamiento breve,
  2) un ejemplo concreto: input -> respuesta verdadera vs predicha + nº de
     iteraciones de "pensamiento",
  3) dinámica del HBP (oscilación en evolución libre; VEI activo en inferencia),
  4) cómputo adaptativo: E[n_iter] y accuracy por dificultad + corr(K, n_iter).

Uso:  $env:PYTHONPATH="." ; python example_routine.py
"""
import torch
import numpy as np
from data.synthetic_recall import PermutationCompositionDataset
from training.trainer import build_model
from eval.diagnostics import run_full_diagnostics

torch.manual_seed(0)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
print(f"Dispositivo: {dev} | dtype: {dt}\n")

# Dos rangos de K: entrenamiento (<=12) y evaluación (<=24, con extrapolación).
ds_train = PermutationCompositionDataset(min_ops=2, max_ops=12, seq_len=128, seed=0, gen_set="adjacent")
ds_eval = PermutationCompositionDataset(min_ops=2, max_ops=24, seq_len=128, seed=1, gen_set="adjacent")

model = build_model("hbp_full", ds_train.vocab_size, 128, halt_mod_gain=2.0).to(device=dev, dtype=dt)
active = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(active, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

N_STEPS = 2500
print(f"[1] Entrenando hbp_full ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params) "
      f"{N_STEPS} pasos (composición en S_5, entreno K<=12)...")
model.train()
for step in range(1, N_STEPS + 1):
    lr = 3e-4 * min(1.0, step / 200) * (0.5 * (1 + np.cos(np.pi * min(1.0, step / N_STEPS))))
    for g in opt.param_groups:
        g["lr"] = lr
    inp, tgt, _ = ds_train.batch(32)
    inp, tgt = inp.to(dev), tgt.to(dev)
    _, ld = model(inp, tgt)
    opt.zero_grad(); ld["total"].backward()
    torch.nn.utils.clip_grad_norm_(active, 1.0); opt.step()
    if step % 500 == 0:
        print(f"    paso {step:4d} | loss {ld['total'].item():.3f}")
model.eval()


def build_example(ds, ops):
    seq, running = [], ds.identity
    for j in ops:
        seq.append(ds.gen_offset + j)
        running = ds._compose(ds.gens[j], running)
    seq.append(ds.Q)
    arrow_pos = len(seq)
    seq.append(ds.ARROW)
    full = seq + [ds.PAD] * (ds.seq_len - len(seq))
    return torch.tensor(full, dtype=torch.long), arrow_pos, running


@torch.no_grad()
def run_once(ds, ops):
    tokens, arrow_pos, true_perm = build_example(ds, ops)
    logits, _ = model(tokens.unsqueeze(0).to(dev), None)
    pred_idx = int(logits[0, arrow_pos].argmax().item()) - ds.perm_offset
    pred_perm = ds.perms[pred_idx] if 0 <= pred_idx < len(ds.perms) else None
    return true_perm, pred_perm, float(model._last_n_expected[0].item())


# --------------------------------------------------------------------------- #
# 2) UN ejemplo concreto, narrado
# --------------------------------------------------------------------------- #
ps = lambda p: "".join(str(i) for i in p) if p is not None else "??"
ops = [0, 2, 1, 3, 1]
true_perm, pred_perm, n_iter = run_once(ds_eval, ops)
print("\n[2] EJEMPLO CONCRETO")
print(f"    Estado inicial (identidad):  {ps(ds_eval.identity)}")
print(f"    Operaciones (K={len(ops)}): " + ", ".join(f"t{j}=(swap {j},{j+1})" for j in ops))
print(f"    Permutación VERDADERA:  {ps(true_perm)}")
print(f"    Predicción del modelo:  {ps(pred_perm)}   "
      f"{'CORRECTO' if pred_perm == true_perm else 'incorrecto'}")
print(f"    El reasoner 'pensó' E[n_iter] = {n_iter:.2f} de 10 iteraciones posibles.")

# --------------------------------------------------------------------------- #
# 3) Dinámica del HBP: oscilación (evolución libre) + actividad en inferencia
# --------------------------------------------------------------------------- #
rep = run_full_diagnostics(model.hbp, n_ticks=200, device=dev, dtype=dt)
inf_var = []
with torch.no_grad():
    for _ in range(20):
        inp, _, _ = ds_eval.batch(32)
        model(inp.to(dev), None)
        inf_var.append(model.hbp.state_summary()["vei_variance"])
print("\n[3] DINÁMICA DEL HBP")
print(f"    Evolución libre: frecuencia dominante FFT = {rep['spectrum']['dominant_freq']:.4f} "
      f"(oscila: {rep['spectrum']['has_oscillation']}); ζ medio = {rep['damping']['zeta_mean']:.3f} "
      f"-> {rep['damping']['regime_counts']}")
print(f"    En inferencia: varianza media del VEI = {np.mean(inf_var):.4f} "
      f"(el campo se activa al procesar el input)")

# --------------------------------------------------------------------------- #
# 4) Cómputo adaptativo por dificultad (con corr sobre muchas muestras)
# --------------------------------------------------------------------------- #
print("\n[4] CÓMPUTO ADAPTATIVO Y EXTRAPOLACIÓN")
gen_rng = torch.Generator().manual_seed(123)
Ks, niters, correct = [], [], []
for _ in range(600):
    K = int(torch.randint(2, 25, (1,), generator=gen_rng).item())
    ops = [int(torch.randint(0, ds_eval.n_gen, (1,), generator=gen_rng).item()) for _ in range(K)]
    tp, pp, ni = run_once(ds_eval, ops)
    Ks.append(K); niters.append(ni); correct.append(int(pp == tp))
Ks, niters, correct = np.array(Ks), np.array(niters), np.array(correct)
print(f"    {'bucket':22s} {'accuracy':>9s} {'E[n_iter]':>10s}")
for name, m in [("corto (K<=6, in-dist)", Ks <= 6),
                ("medio (7-13, in-dist)", (Ks >= 7) & (Ks <= 13)),
                ("largo (>=14, EXTRAP.)", Ks >= 14)]:
    print(f"    {name:22s} {correct[m].mean():>9.3f} {niters[m].mean():>10.2f}")
corr = np.corrcoef(Ks, niters)[0, 1]
print(f"\n    corr(K, E[n_iter]) = {corr:+.3f}  "
      f"(>0: el modelo asigna MÁS cómputo a las composiciones más difíciles)")
print("\nFin de la rutina.")
