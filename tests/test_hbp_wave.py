import math

import torch

from model.hbp import (
    HBPConfig, HomeostaticBackgroundProcessor, build_chain_laplacian)


def _linear_wave(n_nodes=6, dt=1.0):
    cfg = HBPConfig(
        n_nodes=n_nodes, d_f=1, d_e=1, d_u=1, d_intero=2,
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


def _zero_intero(hbp):
    return torch.zeros(
        1, hbp.cfg.n_nodes, hbp.cfg.d_intero, dtype=torch.float32)


def test_chain_laplacian_is_neumann_graph_laplacian():
    n = 6
    laplacian = build_chain_laplacian(n)
    torch.testing.assert_close(laplacian, laplacian.T, rtol=0, atol=0)
    torch.testing.assert_close(
        laplacian.sum(dim=1), torch.zeros(n), rtol=0, atol=0)
    eigenvalues = torch.linalg.eigvalsh(laplacian)
    expected = torch.tensor([
        2.0 - 2.0 * math.cos(k * math.pi / n) for k in range(n)])
    torch.testing.assert_close(eigenvalues, expected, rtol=1e-5, atol=1e-6)
    assert eigenvalues[0] >= -1e-6


def test_wave_step_matches_exact_modal_recurrence():
    hbp = _linear_wave()
    eigenvalues, eigenvectors = torch.linalg.eigh(hbp.laplacian)
    mode = eigenvectors[:, 3]
    current, previous = 0.7, -0.2
    hbp.h_t.zero_()
    hbp.h_tm1.zero_()
    hbp.h_t[0, :, 0] = current * mode
    hbp.h_tm1[0, :, 0] = previous * mode

    dt = hbp.cfg.dt
    omega = float(hbp.omega0[0].detach())
    zeta = float(hbp.zeta[0].detach())
    speed = float(hbp.c.detach())
    q = dt ** 2 * (omega ** 2 + speed ** 2 * float(eigenvalues[3]))
    damping = 2.0 * zeta * omega * dt
    expected_amplitude = ((2.0 - q - damping) * current
                          - (1.0 - damping) * previous)
    actual = hbp.step(_zero_intero(hbp))[0, :, 0]
    torch.testing.assert_close(
        actual, expected_amplitude * mode, rtol=2e-5, atol=2e-6)


def test_wave_has_local_one_edge_per_tick_propagation():
    hbp = _linear_wave()
    hbp.h_t.zero_()
    hbp.h_tm1.zero_()
    hbp.h_t[0, 0, 0] = 1.0
    hbp.h_tm1[0, 0, 0] = 1.0  # velocidad inicial nula
    first = hbp.step(_zero_intero(hbp))[0, :, 0].detach()
    assert first[1] > 0
    torch.testing.assert_close(first[2:], torch.zeros_like(first[2:]),
                               rtol=0, atol=1e-7)
    second = hbp.step(_zero_intero(hbp))[0, :, 0].detach()
    assert second[2] > 0
    torch.testing.assert_close(second[3:], torch.zeros_like(second[3:]),
                               rtol=0, atol=1e-7)


def test_exact_spectral_certificate_matches_modal_roots():
    hbp = _linear_wave()
    maximum = 0.0
    for omega, zeta in zip(hbp.omega0.detach(), hbp.zeta.detach()):
        damping = 2.0 * float(zeta * omega) * hbp.cfg.dt
        for eigenvalue in torch.linalg.eigvalsh(hbp.laplacian):
            q = hbp.cfg.dt ** 2 * (
                float(omega) ** 2
                + float(hbp.c.detach()) ** 2 * float(eigenvalue))
            a = 2.0 - q - damping
            b = 1.0 - damping
            discriminant = complex(a * a - 4.0 * b) ** 0.5
            roots = ((a + discriminant) / 2.0,
                     (a - discriminant) / 2.0)
            maximum = max(maximum, *(abs(root) for root in roots))
    assert abs(hbp.certificate_spectral_radius() - maximum) < 2e-5
    assert maximum < 1.0
    assert float(hbp.stability_penalty().detach()) == 0.0


def test_damped_unforced_wave_converges_without_clamp_activation():
    torch.manual_seed(41)
    hbp = _linear_wave()
    displacement = 0.3 * torch.randn_like(hbp.h_t)
    hbp.h_t = displacement.clone()
    hbp.h_tm1 = displacement.clone()
    initial = float(displacement.norm())
    maximum = initial
    for _ in range(120):
        state = hbp.step(_zero_intero(hbp))
        maximum = max(maximum, float(state.norm().detach()))
    assert float(hbp.h_t.norm().detach()) < 1e-5 * initial
    assert maximum < 2.0 * initial


def test_certificate_rejects_a_step_beyond_the_wave_cfl_region():
    stable = _linear_wave(dt=1.0)
    unstable = _linear_wave(dt=3.0)
    assert stable.certificate_spectral_radius() < 1.0
    assert float(stable.stability_penalty().detach()) == 0.0
    assert unstable.certificate_spectral_radius() > 1.0
    assert float(unstable.stability_penalty().detach()) > 0.0
