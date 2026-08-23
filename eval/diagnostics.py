"""
MiuraCognitive - Diagnósticos del HBP
======================================
Herramientas para verificar que el HBP aprende dinámicas no triviales y
para producir las figuras del paper:

  - Varianza temporal del VEI (¿colapsa al equilibrio?)
  - Espectro FFT del VEI (¿hay frecuencias dominantes -> oscilación?)
  - Trayectoria del VEI ante un input (¿se "activa" con dificultad?)
  - Régimen de amortiguamiento efectivo (sub/crítico/sobre)

Autor: Francisco M. Arrabal (Curro) + asistencia Claude
"""

from __future__ import annotations
import torch
import numpy as np


@torch.no_grad()
def record_vei_trajectory(hbp, n_ticks: int = 100, device=None, dtype=None):
    """Hace evolucionar el HBP n_ticks sin input y registra el VEI.
    Devuelve array (n_ticks, n_nodes, d_h)."""
    device = device or hbp.h_star.device
    dtype = dtype or hbp.h_star.dtype
    hbp.reset_state(batch_size=1, device=device, dtype=dtype)
    # Perturbamos ligeramente para sacar el sistema del reposo exacto
    hbp.h_t = hbp.h_t + 0.1 * torch.randn_like(hbp.h_t)

    traj = []
    zero_intero = torch.zeros(1, hbp.cfg.n_nodes, hbp.cfg.d_intero, device=device, dtype=dtype)
    for _ in range(n_ticks):
        h = hbp.step(zero_intero)
        traj.append(h.detach().float().cpu().squeeze(0).clone())   # (n_nodes, d_h)
    return torch.stack(traj).numpy()   # (n_ticks, n_nodes, d_h)


def vei_variance(traj: np.ndarray) -> dict:
    """Varianza temporal por nodo. Si ~0 el HBP colapsó (fallo)."""
    var_per_node = traj.var(axis=0).mean(axis=-1)   # (n_nodes,)
    return {
        "var_per_node": var_per_node.tolist(),
        "var_mean": float(var_per_node.mean()),
        "collapsed": bool(var_per_node.mean() < 1e-3),
    }


def vei_spectrum(traj: np.ndarray, dt: float = 1.0) -> dict:
    """FFT del VEI promediado. Detecta frecuencias dominantes (oscilación).
    Devuelve la frecuencia dominante y el espectro de potencia."""
    # Promedio sobre nodos y dimensiones -> señal 1D temporal
    signal = traj.mean(axis=(1, 2))                 # (n_ticks,)
    signal = signal - signal.mean()                 # quitar DC
    fft = np.fft.rfft(signal)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=dt)
    # Frecuencia dominante (excluyendo DC en índice 0)
    if len(power) > 1:
        dom_idx = 1 + int(np.argmax(power[1:]))
        dom_freq = float(freqs[dom_idx])
    else:
        dom_freq = 0.0
    return {
        "dominant_freq": dom_freq,
        "has_oscillation": bool(dom_freq > 1e-3),
        "power_spectrum": power.tolist(),
        "freqs": freqs.tolist(),
    }


def damping_regime(hbp) -> dict:
    """Clasifica el régimen de amortiguamiento por dimensión del VEI según ζ.
      ζ<1 subamortiguado, ζ=1 crítico, ζ>1 sobreamortiguado."""
    zeta = hbp.zeta.detach().float().cpu().numpy()
    regimes = []
    for z in zeta:
        if z < 0.95:
            regimes.append("subamortiguado")
        elif z <= 1.05:
            regimes.append("critico")
        else:
            regimes.append("sobreamortiguado")
    from collections import Counter
    return {
        "zeta_values": zeta.tolist(),
        "zeta_mean": float(zeta.mean()),
        "regime_counts": dict(Counter(regimes)),
    }


def run_full_diagnostics(hbp, n_ticks: int = 200, device=None, dtype=None) -> dict:
    """Ejecuta toda la batería de diagnósticos y devuelve un informe."""
    traj = record_vei_trajectory(hbp, n_ticks, device, dtype)
    report = {
        "variance": vei_variance(traj),
        "spectrum": vei_spectrum(traj, dt=hbp.cfg.dt),
        "damping": damping_regime(hbp),
    }
    return report


if __name__ == "__main__":
    # Prueba rápida con un HBP recién inicializado
    from model.hbp import HBPConfig, HomeostaticBackgroundProcessor
    hbp = HomeostaticBackgroundProcessor(HBPConfig(n_nodes=4))
    rep = run_full_diagnostics(hbp, n_ticks=200)
    print("=== Diagnóstico del HBP ===")
    print(f"Varianza media VEI: {rep['variance']['var_mean']:.4f} "
          f"(colapsado: {rep['variance']['collapsed']})")
    print(f"Frecuencia dominante: {rep['spectrum']['dominant_freq']:.4f} "
          f"(oscila: {rep['spectrum']['has_oscillation']})")
    print(f"ζ medio: {rep['damping']['zeta_mean']:.3f} "
          f"-> {rep['damping']['regime_counts']}")
