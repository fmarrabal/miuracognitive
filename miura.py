"""
MiuraCognitive — CLI de uso
============================
Entrena, guarda y usa un modelo MiuraCognitive desde la terminal.

  # Entrenar y guardar un checkpoint (variante y generadores a elegir)
  python miura.py train --variant hbp_full --gens adjacent --steps 2500 --out checkpoints/hbp_full.pt

  # Componer permutaciones de S_5 (índices de generador) y ver el "pensamiento"
  python miura.py compose 0 2 1 3 1 --ckpt checkpoints/hbp_full.pt

  # Diagnóstico del campo homeostático del modelo entrenado
  python miura.py introspect --ckpt checkpoints/hbp_full.pt

  # Ficha del modelo
  python miura.py info --ckpt checkpoints/hbp_full.pt

Ejecutar desde la raíz del paquete con  $env:PYTHONPATH="."  (Windows) o  PYTHONPATH=.
"""
from __future__ import annotations
import argparse
import math
import torch


def _train(args):
    from data.synthetic_recall import PermutationCompositionDataset
    from training.trainer import build_model
    from miura_infer import MiuraModel

    if args.device:
        dev = torch.device(args.device)
    else:
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
    print(f"Entrenando {args.variant} en {dev} ({dt}), {args.steps} pasos, generadores={args.gens}...")

    ds = PermutationCompositionDataset(min_ops=2, max_ops=args.max_ops, seq_len=128,
                                       seed=args.seed, gen_set=args.gens)
    build_args = dict(variant=args.variant, vocab_size=ds.vocab_size, seq_len=128,
                      halt_mod_gain=2.0, max_halt_steps=args.max_halt_steps,
                      ponder_expected_loss=args.ponder_expected_loss)
    model = build_model(**build_args).to(device=dev, dtype=dt)
    if model.cfg.use_hbp:
        model.hbp.pin_fp32()   # FIX BF16: raw físicos congelados por ULP en BF16
    active = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(active, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    model.train()
    for step in range(1, args.steps + 1):
        lr = 3e-4 * min(1.0, step / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        inp, tgt, _ = ds.batch(32)
        _, ld = model(inp.to(dev), tgt.to(dev))
        opt.zero_grad(); ld["total"].backward()
        torch.nn.utils.clip_grad_norm_(active, 1.0); opt.step()
        if step % max(1, args.steps // 10) == 0:
            print(f"  paso {step:5d}/{args.steps} | loss {ld['total'].item():.3f}")

    MiuraModel.save(model, ds, args.out, build_args,
                    meta={"variant": args.variant, "gen_set": args.gens,
                          "steps": args.steps, "max_ops_train": args.max_ops,
                          "seed": args.seed,
                          "ponder_expected_loss": args.ponder_expected_loss})
    print(f"Checkpoint guardado en {args.out}")


def _load(args):
    from miura_infer import MiuraModel
    return MiuraModel.from_checkpoint(args.ckpt, device=args.device)


def _compose(args):
    m = _load(args)
    ng = m.ds.n_gen
    ops = [o for o in args.ops if 0 <= o < ng]
    if len(ops) != len(args.ops):
        print(f"Aviso: generadores válidos = 0..{ng-1}; ignorados {[o for o in args.ops if o not in ops]}")
    r = m.compose(ops)
    ps = lambda p: "".join(map(str, p)) if p else "??"
    gens_desc = ", ".join(f"g{j}" for j in ops)
    print(f"\nEstado inicial (identidad): {ps(m.ds.identity)}")
    print(f"Operaciones (K={len(ops)}): {gens_desc}")
    print(f"Resultado VERDADERO:  {ps(r.true_perm)}")
    print(f"Predicción del modelo: {ps(r.predicted)}   {'CORRECTO' if r.correct else 'incorrecto'}"
          f"  (confianza {r.confidence:.2f})")
    print(f"El reasoner pensó E[n_iter] = {r.n_iter:.2f} iteraciones.")
    if r.physics is not None and r.physics.get("D", 0) + r.physics.get("b", 0) > 0:
        print(f"Física elegida por el HBP: α={r.physics['alpha']:.2f} "
              f"({'onda' if r.physics['alpha'] >= 0.5 else 'difusión'}), "
              f"D={r.physics['D']:.3f} (difusión), b={r.physics['b']:.3f} (advección).")
    if r.trace:
        print("\nTraza del HBP (campo homeostático por tick de pensamiento):")
        print(f"  {'tick':>4s} {'VEI_var':>9s} {'desviación':>11s} {'umbral_halt':>12s} {'mod_norm':>9s}")
        for s in r.trace:
            print(f"  {s['n']:>4d} {s['vei_var']:>9.4f} {s['deviation']:>11.3f} "
                  f"{s['halt_threshold']:>12.3f} {s['reasoner_mod_norm']:>9.3f}")


def _introspect(args):
    m = _load(args)
    d = m.introspect()
    print("\nDiagnóstico del campo homeostático (HBP):")
    if not d["use_hbp"]:
        print("  (esta variante no tiene HBP)")
        return
    print(f"  Varianza media del VEI:   {d['vei_variance']:.4f}")
    print(f"  Frecuencia dominante FFT: {d['dominant_freq']:.4f}  (oscila: {d['oscillates']})")
    print(f"  ζ medio:                  {d['zeta_mean']:.3f}")
    print(f"  Regímenes de amortiguamiento: {d['regimes']}")
    if d.get("gate_physics"):
        p = d["physics"]
        print(f"  Física gateada (α onda↔difusión={p['alpha']:.2f}, "
              f"D={p['D']:.3f}, b={p['b']:.3f})")


def _info(args):
    m = _load(args)
    for k, v in m.info().items():
        print(f"  {k}: {v}")


def main():
    ap = argparse.ArgumentParser(description="MiuraCognitive CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="entrena y guarda un checkpoint")
    t.add_argument("--variant", default="hbp_full",
                   choices=["vanilla", "gating", "gating_wm", "hbp_first", "hbp_full",
                            "hbp_mix", "hbp_kdv", "hbp_elliptic"])
    t.add_argument("--gens", default="adjacent", choices=["adjacent", "cycle_transp"])
    t.add_argument("--steps", type=int, default=2500)
    t.add_argument("--max_ops", type=int, default=16)
    t.add_argument("--max_halt_steps", type=int, default=24)
    t.add_argument("--ponder-expected-loss", action="store_true",
                   help="entrena con E_p[CE por profundidad] en lugar de CE sobre la mezcla")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="checkpoints/miura.pt")
    t.add_argument("--device", default=None, help="cpu | cuda:0 (por defecto: auto)")
    t.set_defaults(fn=_train)

    for name, fn, helptxt in [("compose", _compose, "compón permutaciones y observa el pensamiento"),
                              ("introspect", _introspect, "diagnostica el campo homeostático"),
                              ("info", _info, "ficha del modelo")]:
        p = sub.add_parser(name, help=helptxt)
        if name == "compose":
            p.add_argument("ops", type=int, nargs="+", help="índices de generador a aplicar en orden")
        p.add_argument("--ckpt", default="checkpoints/miura.pt")
        p.add_argument("--device", default=None)
        p.set_defaults(fn=fn)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
