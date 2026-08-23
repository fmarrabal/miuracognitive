"""
mHBP — Demo mínima ejecutable (Fase 1).

  PYTHONPATH=. python -m mhbp.examples.minimal_demo

1. Construye el mHBP de 4 campos (config MVP).
2. Imprime el informe de estabilidad (certificado estructural + ρ exacto + energía).
3. Respuesta a IMPULSO interoceptivo: un golpe de entropía/incertidumbre en t=10
   → cada campo responde a SU escala temporal (separación multiescala visible).
4. Conflicto rápido-vs-recursos en miniatura: sube el coste de tokens a mitad
   de rollout → el campo metabólico se desplaza y (vía alostasis) arrastra el
   setpoint del ejecutivo.
5. Figura con: estados por campo, energía, actuadores. Guardada en examples/.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from mhbp.coupled_fields import MHBPConfig, CoupledMultiscaleHBP
from mhbp.interoception import InteroceptiveSignal
from mhbp.stability.certificates import stability_report

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    torch.manual_seed(0)
    m = CoupledMultiscaleHBP(MHBPConfig()).double()
    m.reset_state(1)

    print("=== mHBP Fase 1 — demo mínima ===")
    rep = stability_report(m)
    print(f"certificado estructural: all_ok={rep['structural']['all_ok']}")
    for f in m.fields:
        s = rep["structural"][f.name]
        print(f"  {f.name:20s} K_min={s['K_min_eig']:.4f}  C_min={s['C_min_eig']:.4f}  "
              f"G_antisym_err={s['G_antisym_err']:.1e}")
    print(f"radio espectral exacto ρ(Φ) = {rep['spectral_radius']:.6f}  (<1: {rep['rho_lt_1']})")
    print(f"disipación discreta: {rep['energy']['violations']}/{rep['energy']['total']} violaciones")
    print(f"τ = {[f'{t:.1f}' for t in m.timescales().tolist()]}")

    # ---------------- rollout con impulso + cambio de recursos ----------------
    # ganancia ILUSTRATIVA de los actuadores (el sistema está sin entrenar; en
    # Fase 2+ estas W las fija el gradiente — aquí solo se visualiza el cableado
    # campo→actuador a través de la proyección acotada)
    with torch.no_grad():
        for W in m.actuators.W:
            W.mul_(20.0)
    T = 400
    m.reset_state(1)
    traces = {f.name: [] for f in m.fields}
    energies, acts_tr = [], {k: [] for k in ("halt_bias", "depth_max", "tool_gate", "budget_scale")}
    with torch.no_grad():
        for t in range(T):
            vals = {"entropy": 0.1, "uncertainty": 0.1, "token_cost": 0.2}
            if 10 <= t < 14:
                vals["entropy"] = 3.0          # IMPULSO de dificultad
                vals["uncertainty"] = 3.0
            if t >= 200:
                vals["token_cost"] = 4.0       # presión de recursos sostenida
            acts, info = m.tick(InteroceptiveSignal(tick=t, values=vals))
            for f in m.fields:
                traces[f.name].append(float(f.u.norm()))
            energies.append(float(info["energy"][0]))
            for k in acts_tr:
                acts_tr[k].append(float(acts[k][0]))

    print("\nrespuesta al impulso (‖u‖ máx por campo y tick del pico):")
    for name, tr in traces.items():
        tt = torch.tensor(tr)
        print(f"  {name:20s} pico={float(tt.max()):.4f} en t={int(tt.argmax())}")

    # ---------------- figura ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for name, tr in traces.items():
        axes[0].plot(tr, label=name, lw=1.4)
    axes[0].axvspan(10, 14, color="red", alpha=0.15, label="impulso")
    axes[0].axvline(200, color="k", ls="--", lw=0.8, label="presión de recursos")
    axes[0].set_ylabel("‖u_q‖"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[0].set_title("mHBP: cada campo responde a su escala temporal (τ = 1, 4, 8, 32)")
    axes[1].plot(energies, color="C4")
    axes[1].set_ylabel("energía E"); axes[1].grid(alpha=0.3)
    from mhbp.actuators import MVP_ACTUATORS
    rng = {sp.name: (sp.lo, sp.hi) for sp in MVP_ACTUATORS}
    for k, tr in acts_tr.items():
        lo, hi = rng[k]
        axes[2].plot([(v - lo) / (hi - lo) for v in tr], label=f"{k} (norm.)", lw=1.2)
    axes[2].set_ylabel("actuadores (fracción del rango)"); axes[2].set_xlabel("tick")
    axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(HERE, "demo_impulse_response.png")
    fig.savefig(p, dpi=140)
    print(f"\nfigura: {p}")


if __name__ == "__main__":
    main()
