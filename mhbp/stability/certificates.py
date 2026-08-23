"""
mHBP — Certificados de estabilidad (§7, Prop. 3 de MATH_SPEC.md).

Tres niveles:
 1. `structural_certificate`: K≻0, C≻0, G antisimétrica, 𝓑 PSD y ACOTADA,
    τ ordenadas, margen anti-resonancia, cond(A_θ) — garantizados POR
    CONSTRUCCIÓN; aquí se VERIFICAN numéricamente (paranoia).
 2. `exact_spectral_radius`: ρ(Φ) del mapa lineal homogéneo de UN TICK COMPLETO,
    sondeando el integrador real columna a columna (F=0) — exacto, cubre
    acoplamiento, escalas distintas, Cayley, solve implícito y el radio del caso
    no-normal. (ρ<1 da el DECAY asintótico; el transitorio no-normal lo acota la
    disipación de energía con θ=1 — nivel 3.)
 3. `energy_dissipation_check`: E_{t+1} ≤ E_t sin forzamiento, en trayectorias
    aleatorias (BE + Cayley ortogonal ⇒ debe cumplirse siempre).

TODOS los certificados operan sobre un integrador FP64 construido AD HOC con
`model.build_integrator(dtype=float64)`: NO tocan el estado vivo del modelo
(hallazgo major: la versión anterior hacía reset_state y destruía el episodio)
y certifican la matemática aunque el despliegue sea FP32 (se documenta que la
aritmética FP32 desplegada no queda cubierta).
"""
from __future__ import annotations
import torch

from ..integrators.cayley_imex import CayleyIMEX, field_offsets


@torch.no_grad()
def structural_certificate(model) -> dict:
    """Comprobaciones estructurales del CoupledMultiscaleHBP."""
    out = {}
    dt64 = torch.float64
    for f in model.fields:
        K, C, G = f.operators(dtype=dt64)
        evK = torch.linalg.eigvalsh(K).min()
        evC = torch.linalg.eigvalsh(C).min()
        anti = (G + G.transpose(-1, -2)).abs().max()
        out[f.name] = {"K_min_eig": float(evK), "C_min_eig": float(evC),
                       "G_antisym_err": float(anti),
                       "K_pd": bool(evK > 0), "C_pd": bool(evC > 0)}
    taus = model.timescales()
    out["taus_ordered"] = bool((torch.diff(taus) > 0).all())
    if model.coupling is not None:
        offs, n = field_offsets(list(model.fields))
        B = model.coupling.assemble_B(offs, n, [f.cfg for f in model.fields],
                                      dtype=dt64, device=taus.device)
        evB = torch.linalg.eigvalsh(0.5 * (B + B.T))
        out["coupling_psd"] = bool(evB.min() > -1e-10)
        out["coupling_min_eig"] = float(evB.min())
        out["coupling_norm"] = float(evB.max())          # acotada por la caja de κ/Ŵ
    # margen anti-resonancia y condicionamiento del núcleo implícito
    integ = model.build_integrator(dtype=dt64)
    if isinstance(integ, CayleyIMEX):
        out["antiresonance_margin"] = integ.antiresonance_margin
        A = integ._A_chol @ integ._A_chol.T              # A_θ = L Lᵀ
        ev = torch.linalg.eigvalsh(A)
        out["cond_A_theta"] = float(ev.max() / ev.min())
    out["all_ok"] = all(
        v.get("K_pd", True) and v.get("C_pd", True) and v.get("G_antisym_err", 0) < 1e-10
        for v in out.values() if isinstance(v, dict)
    ) and out["taus_ordered"] and out.get("coupling_psd", True) \
      and out.get("cond_A_theta", 1.0) < 1e12
    return out


@torch.no_grad()
def one_step_matrix(model) -> torch.Tensor:
    """Mapa lineal Φ (2n × 2n) de un tick homogéneo (F=0), por SONDEO del
    integrador real sobre la base canónica de z = [U; W]. FP64. SIN tocar el
    estado del modelo (integrador ad hoc)."""
    integ = model.build_integrator(dtype=torch.float64)
    n = integ.n_tot
    dev = model.fields[0].h_star.device
    Phi = torch.zeros(2 * n, 2 * n, dtype=torch.float64, device=dev)
    # sondeo por bloques (una pasada batched: 2n "muestras")
    Z = torch.eye(2 * n, dtype=torch.float64, device=dev)
    U, W = Z[:, :n], Z[:, n:]
    Up, Wp = integ.step(U, W, None)
    Phi[:n] = Up.T
    Phi[n:] = Wp.T
    return Phi


@torch.no_grad()
def exact_spectral_radius(model) -> float:
    """ρ(Φ) exacto del tick completo. Certificado: ρ < 1."""
    Phi = one_step_matrix(model)
    return float(torch.linalg.eigvals(Phi).abs().max())


@torch.no_grad()
def spectral_margin(model) -> float:
    """1 − ρ(Φ): métrica CONTINUA para el StabilityMonitor (colapsa suavemente
    al acercarse a la variedad de resonancia — recomendación del ataque)."""
    return 1.0 - exact_spectral_radius(model)


@torch.no_grad()
def energy_dissipation_check(model, trials: int = 20, ticks: int = 50,
                             seed: int = 0, tol: float = 1e-12) -> dict:
    """F=0: E debe ser no creciente tick a tick (BE disipativo + Cayley neutro).
    Integrador ad hoc FP64; NO toca el estado del modelo."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    integ = model.build_integrator(dtype=torch.float64)
    n = integ.n_tot
    dev = model.fields[0].h_star.device
    worst, viol, total = 0.0, 0, 0
    for _ in range(trials):
        U = torch.randn(1, n, generator=g).double().to(dev)
        W = torch.randn(1, n, generator=g).double().to(dev)
        E_prev = float(integ.energy(U, W))
        for _t in range(ticks):
            U, W = integ.step(U, W, None)
            E = float(integ.energy(U, W))
            total += 1
            rel = (E - E_prev) / max(abs(E_prev), 1e-300)
            if rel > tol:
                viol += 1
                worst = max(worst, rel)
            E_prev = E
    return {"violations": viol, "total": total, "worst_rel_increase": worst,
            "ok": viol == 0}


@torch.no_grad()
def transient_growth_check(model, kmax: int = 200) -> dict:
    """Crecimiento transitorio no-normal: max_k ‖Φ^k‖₂. Con θ=1 la disipación
    de energía acota el transitorio en la norma de energía; aquí se mide en
    norma-2 (puede superar 1 transitoriamente sin violar la estabilidad)."""
    Phi = one_step_matrix(model)
    M = torch.eye(Phi.shape[0], dtype=Phi.dtype, device=Phi.device)
    peak, peak_k = 1.0, 0
    for k in range(1, kmax + 1):
        M = Phi @ M
        nrm = float(torch.linalg.matrix_norm(M, ord=2))
        if nrm > peak:
            peak, peak_k = nrm, k
    return {"max_transient_2norm": peak, "argmax_k": peak_k,
            "final_2norm": float(torch.linalg.matrix_norm(M, ord=2))}


@torch.no_grad()
def stability_report(model) -> dict:
    """Informe completo (StabilityMonitor de Fase 1). No altera el estado."""
    rep = {"structural": structural_certificate(model)}
    rho = exact_spectral_radius(model)
    rep["spectral_radius"] = rho
    rep["spectral_margin"] = 1.0 - rho
    rep["rho_lt_1"] = rho < 1.0
    rep["energy"] = energy_dissipation_check(model, trials=5, ticks=30)
    rep["ok"] = (rep["structural"]["all_ok"] and rep["rho_lt_1"]
                 and rep["energy"]["ok"])
    return rep
