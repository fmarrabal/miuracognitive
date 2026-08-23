"""Auditoría reproducible de la rama de onda amortiguada del HBP."""

from __future__ import annotations

from dataclasses import asdict
import glob
import json
import math
import os

import torch

from model.aha import AHAControllerConfig, AnticipatoryHomeostaticAgent
from model.budgeted_stream import (
    BudgetedStreamConfig, BudgetedStreamReasoner)
from model.hbp import (
    HBPConfig, HomeostaticBackgroundProcessor, build_chain_laplacian)


def _linear_hbp(dt: float = 1.0) -> HomeostaticBackgroundProcessor:
    cfg = HBPConfig(
        n_nodes=6, d_f=1, d_e=1, d_u=1, d_intero=2,
        dt=dt, order=2, g_phi_gain=0.0, h_clamp=1e6)
    hbp = HomeostaticBackgroundProcessor(cfg).float()
    with torch.no_grad():
        hbp.h_star.zero_()
        for parameter in hbp.f_theta.parameters():
            parameter.zero_()
        hbp.g_phi_proj.weight.zero_()
        hbp.g_phi_proj.bias.zero_()
    hbp.reset_state(1, dtype=torch.float32)
    return hbp


def _zero_intero(hbp: HomeostaticBackgroundProcessor) -> torch.Tensor:
    return torch.zeros(1, hbp.cfg.n_nodes, hbp.cfg.d_intero)


def _checkpoint_rows() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(
            "results_budgeted_stream/checkpoints/hbp_full_seed*.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = BudgetedStreamReasoner(
            BudgetedStreamConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
        rows.append(_checkpoint_metrics(path, model.hbp))
    for path in sorted(glob.glob(
            "results_aha/pilot_v4/checkpoints/hbp_full_seed*.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = AnticipatoryHomeostaticAgent(
            AHAControllerConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["state_dict"])
        rows.append(_checkpoint_metrics(path, model.hbp))
    return rows


def _checkpoint_metrics(path: str,
                        hbp: HomeostaticBackgroundProcessor) -> dict:
    return {
        "path": os.path.abspath(path),
        "stability_penalty": float(hbp.stability_penalty().detach()),
        "exact_spectral_radius": hbp.certificate_spectral_radius(),
        "omega0_min": float(hbp.omega0.detach().min()),
        "omega0_max": float(hbp.omega0.detach().max()),
        "zeta_min": float(hbp.zeta.detach().min()),
        "zeta_max": float(hbp.zeta.detach().max()),
        "wave_speed": float(hbp.c.detach()),
    }


def run_audit() -> dict:
    hbp = _linear_hbp()
    laplacian = hbp.laplacian
    eigvals, eigvecs = torch.linalg.eigh(laplacian)
    expected = torch.tensor([
        2.0 - 2.0 * math.cos(k * math.pi / hbp.cfg.n_nodes)
        for k in range(hbp.cfg.n_nodes)])
    laplacian_error = float((eigvals - expected).abs().max())

    modal_errors = []
    for mode_idx in range(hbp.cfg.n_nodes):
        hbp = _linear_hbp()
        eigvals, eigvecs = torch.linalg.eigh(hbp.laplacian)
        current, previous = 0.7, -0.2
        hbp.h_t.zero_()
        hbp.h_tm1.zero_()
        hbp.h_t[0, :, 0] = current * eigvecs[:, mode_idx]
        hbp.h_tm1[0, :, 0] = previous * eigvecs[:, mode_idx]
        omega = float(hbp.omega0[0].detach())
        zeta = float(hbp.zeta[0].detach())
        q = hbp.cfg.dt ** 2 * (
            omega ** 2 + float(hbp.c.detach()) ** 2
            * float(eigvals[mode_idx]))
        damping = 2.0 * zeta * omega * hbp.cfg.dt
        expected_state = ((2.0 - q - damping) * current
                          - (1.0 - damping) * previous) * eigvecs[:, mode_idx]
        actual = hbp.step(_zero_intero(hbp))[0, :, 0]
        modal_errors.append(float(
            (actual - expected_state).abs().max().detach()))

    hbp = _linear_hbp()
    hbp.h_t.zero_()
    hbp.h_tm1.zero_()
    hbp.h_t[0, 0, 0] = 1.0
    hbp.h_tm1[0, 0, 0] = 1.0
    first = hbp.step(_zero_intero(hbp))[0, :, 0].detach()
    second = hbp.step(_zero_intero(hbp))[0, :, 0].detach()

    torch.manual_seed(41)
    hbp = _linear_hbp()
    displacement = 0.3 * torch.randn_like(hbp.h_t)
    hbp.h_t = displacement.clone()
    hbp.h_tm1 = displacement.clone()
    initial_norm = float(displacement.norm())
    maximum_norm = initial_norm
    for _ in range(120):
        state = hbp.step(_zero_intero(hbp))
        maximum_norm = max(maximum_norm, float(state.norm().detach()))
    decay_ratio = float(hbp.h_t.norm().detach()) / initial_norm

    stable = _linear_hbp(dt=1.0)
    outside_cfl = _linear_hbp(dt=3.0)
    checkpoints = _checkpoint_rows()
    criteria = {
        "laplacian_symmetric": bool(torch.equal(laplacian, laplacian.T)),
        "laplacian_rows_sum_zero": float(laplacian.sum(dim=1).abs().max()) < 1e-7,
        "laplacian_spectrum_error_below_1e-5": laplacian_error < 1e-5,
        "modal_recurrence_error_below_1e-5": max(modal_errors) < 1e-5,
        "one_edge_per_tick_locality": (
            float(first[1]) > 0 and float(first[2:].abs().max()) < 1e-7
            and float(second[2]) > 0
            and float(second[3:].abs().max()) < 1e-7),
        "free_decay_ratio_below_1e-5": decay_ratio < 1e-5,
        "stable_default_radius_below_1": (
            stable.certificate_spectral_radius() < 1.0),
        "outside_cfl_radius_above_1": (
            outside_cfl.certificate_spectral_radius() > 1.0),
        "outside_cfl_penalty_positive": (
            float(outside_cfl.stability_penalty().detach()) > 0),
        "all_checkpoint_penalties_zero": all(
            row["stability_penalty"] == 0.0 for row in checkpoints),
        "all_checkpoint_radii_below_1": all(
            row["exact_spectral_radius"] < 1.0 for row in checkpoints),
    }
    return {
        "status": "pass" if all(criteria.values()) else "fail",
        "implementation": {
            "equation": (
                "u[n+1]=2u[n]-u[n-1]-dt^2(omega0^2 I+c^2 L)u[n]"
                "-2 zeta omega0 dt (u[n]-u[n-1])+dt^2 F[n]"),
            "integrator": "position_Verlet_with_backward_velocity_damping",
            "laplacian": "combinatorial_chain_Neumann_graph_boundary",
            "config": asdict(stable.cfg),
        },
        "metrics": {
            "laplacian_spectrum_max_abs_error": laplacian_error,
            "modal_recurrence_max_abs_error": max(modal_errors),
            "first_tick_state": first.tolist(),
            "second_tick_state": second.tolist(),
            "free_decay_ratio_120_ticks": decay_ratio,
            "free_max_norm_over_initial": maximum_norm / initial_norm,
            "default_exact_spectral_radius": (
                stable.certificate_spectral_radius()),
            "outside_cfl_exact_spectral_radius": (
                outside_cfl.certificate_spectral_radius()),
            "outside_cfl_stability_penalty": float(
                outside_cfl.stability_penalty().detach()),
        },
        "criteria": criteria,
        "checkpoint_audit": checkpoints,
    }


def main() -> None:
    result = run_audit()
    path = "results_wave_audit/audit.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)
    print(json.dumps({
        "status": result["status"],
        "metrics": result["metrics"],
        "criteria": result["criteria"],
        "checkpoint_radii": [
            row["exact_spectral_radius"]
            for row in result["checkpoint_audit"]],
    }, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
