"""Regresión de la verificación adversarial (2026-07-26): cada hallazgo crítico
o major tiene aquí su test con el REPRO EXACTO del atacante. Si alguno de estos
falla, el endurecimiento ha regresionado."""
import torch
import pytest

from mhbp.coupled_fields import MHBPConfig, CoupledMultiscaleHBP, Timescales
from mhbp.field import FieldConfig
from mhbp.interoception import InteroceptiveSignal
from mhbp.stability.certificates import (exact_spectral_radius, stability_report,
                                         structural_certificate,
                                         transient_growth_check)

torch.manual_seed(0)


def adversarial_raws(m):
    """Crudos en los bordes de caja, como el ataque: b,β,ω,c al tope; ζ,D al suelo."""
    for f in m.fields:
        f.params.raw_omega.data.fill_(40.0)
        f.params.raw_zeta.data.fill_(-40.0)
        f.params.raw_c.data.fill_(40.0)
        f.params.raw_D.data.fill_(-40.0)
        f.params.raw_b.data.fill_(40.0)
        f.params.raw_beta.data.fill_(40.0)
    if m.coupling is not None:
        m.coupling.raw_kappa.data.fill_(40.0)
    return m


# ------------------- CRÍTICO 1: resonancia giroscópica θ=½ ------------------- #
def test_resonance_repro_now_rejected():
    """El repro EXACTO del ataque (θ=0.5, dt=20, bordes de caja): prepare() debe
    RECHAZARLO por la condición anti-resonancia, no devolver ρ=1 en silencio."""
    cfg = MHBPConfig(dt=20.0, theta=0.5, coupling_topology="chain",
                     kappa_max=0.5, allostasis=False)
    m = adversarial_raws(CoupledMultiscaleHBP(cfg).double())
    m.reset_state(1)
    with pytest.raises(ValueError, match="anti-resonancia"):
        m.build_integrator()


def test_resonance_variant_rejected():
    """Variante del ataque: dt=10, taus apretadas (0.5,...)."""
    cfg = MHBPConfig(dt=10.0, theta=0.5, taus=(0.5, 0.51, 0.52, 0.53),
                     coupling_topology="none", allostasis=False)
    m = adversarial_raws(CoupledMultiscaleHBP(cfg).double())
    with pytest.raises(ValueError, match="anti-resonancia"):
        m.build_integrator()


def test_same_config_with_theta1_is_stable():
    """La MISMA configuración con θ=1 (BE): permitida y ρ<1 (margen del BE)."""
    cfg = MHBPConfig(dt=20.0, theta=1.0, coupling_topology="chain",
                     kappa_max=0.5, allostasis=False)
    m = adversarial_raws(CoupledMultiscaleHBP(cfg).double())
    rho = exact_spectral_radius(m)
    assert rho < 1.0, f"θ=1 debe ser incondicional (ρ={rho:.6f})"


def test_cn_moderate_dt_still_works():
    """θ=0.5 con dt moderado (rotación ≪ π/2): sigue permitido y estable."""
    cfg = MHBPConfig(dt=0.5, theta=0.5, allostasis=False)
    m = CoupledMultiscaleHBP(cfg).double()
    rho = exact_spectral_radius(m)
    assert rho < 1.0
    integ = m.build_integrator()
    assert integ.antiresonance_margin > 0.9


# ------------------- CRÍTICO 2: W de la interfaz sin caja ------------------- #
def test_unbounded_W_now_safe():
    """Repro del ataque #2 (θ=1 para aislar): W~1e6 y 1e8 → la normalización
    Frobenius mantiene ‖𝓑‖ acotada, el Cholesky no crashea y ρ<1."""
    cfg = MHBPConfig(dt=10.0, theta=1.0, taus=(0.5, 0.51, 0.52, 0.53),
                     coupling_topology="full", kappa_max=50.0, allostasis=False)
    m = adversarial_raws(CoupledMultiscaleHBP(cfg).double())
    for scale in (1e6, 1e8):
        for W in m.coupling.W:
            W.data.normal_(0, scale)
        rho = exact_spectral_radius(m)                 # sin excepción
        assert rho < 1.0, f"ρ={rho:.6f} con ‖W‖~{scale:.0e}"
        s = structural_certificate(m)
        assert s["coupling_psd"] and s["coupling_norm"] < 1e4, s["coupling_norm"]
        assert s["cond_A_theta"] < 1e12


# ------------------- majors de estabilidad/validación ------------------- #
def test_validations():
    with pytest.raises(ValueError):                     # len(taus) != Q
        CoupledMultiscaleHBP(MHBPConfig(taus=(1.0, 4.0, 8.0)))
    with pytest.raises(ValueError):                     # dt <= 0
        CoupledMultiscaleHBP(MHBPConfig(dt=0.0))
    with pytest.raises(ValueError):                     # ζ_min = 0
        fcs = [FieldConfig(name="x", zeta_min=0.0)]
        CoupledMultiscaleHBP(MHBPConfig(taus=(1.0,)), field_cfgs=fcs)


def test_certificates_do_not_touch_state():
    """stability_report ya NO destruye el estado vivo (hallazgo major)."""
    m = CoupledMultiscaleHBP(MHBPConfig()).double()
    m.reset_state(3)
    with torch.no_grad():
        for t in range(5):
            m.tick(InteroceptiveSignal(values={"entropy": 1.0}))
    u_before = [f.u.clone() for f in m.fields]
    tick_before, B_before = m._tick, m._B
    rep = stability_report(m)
    assert rep["ok"]
    assert m._tick == tick_before and m._B == B_before
    for f, u in zip(m.fields, u_before):
        assert torch.equal(f.u, u), "el certificado alteró el estado vivo"


def test_switched_params_auto_reprepare():
    """Cambiar parámetros in-place con estado vivo ⇒ el siguiente tick
    re-ensambla (nada de física vieja) y el sistema conmutado queda acotado."""
    m = CoupledMultiscaleHBP(MHBPConfig(allostasis=False)).double()
    m.reset_state(1)
    with torch.no_grad():
        m.tick(None)
        A_old = m._integ._A_chol.clone()
        m.fields[0].params.raw_omega.add_(2.0)          # "optimizer step"
        m.tick(None)
        assert not torch.allclose(m._integ._A_chol, A_old), \
            "el tick siguió usando la física vieja tras cambiar parámetros"
        # conmutación repetida (switched system): alternar 2 configs × 100 ticks
        energies = []
        for k in range(100):
            m.fields[0].params.raw_omega.add_(1.0 if k % 2 == 0 else -1.0)
            _, info = m.tick(None)
            energies.append(float(info["energy"][0]))
        assert all(torch.isfinite(torch.tensor(energies)))
        assert energies[-1] < 10.0, "sistema conmutado divergiendo"


def test_allostasis_two_episode_training():
    """Fuga de grafo entre episodios (major): 2 episodios con regularizador y
    backward — sin RuntimeError, y regularizador 0 al empezar episodio."""
    m = CoupledMultiscaleHBP(MHBPConfig(allostasis=True)).double()
    for _ep in range(2):
        m.reset_state(2)
        assert float(m.regularizers()["allostasis"]) == 0.0, \
            "regularizador contaminado de un episodio anterior"
        loss = 0.0
        for t in range(3):
            acts, _ = m.tick(InteroceptiveSignal(values={"entropy": 1.0}))
            loss = loss + acts["halt_bias"].pow(2).mean()
        loss = loss + m.regularizers()["allostasis"]
        m.zero_grad()
        loss.backward()                                 # no debe crashear


def test_learnable_init_reproduces_taus():
    """El init del modo learnable arranca EXACTAMENTE en taus_init (major)."""
    ts = Timescales((1.0, 4.0, 8.0, 32.0), mode="learnable")
    taus = ts()
    ref = torch.tensor([1.0, 4.0, 8.0, 32.0]).double()
    assert torch.allclose(taus, ref, atol=1e-6), f"init={taus.tolist()}"


def test_context_mode_integrated():
    """Modo context CABLEADO: tick(ctx=...) funciona, τ ordenadas, ρ<1; ctx no
    finito no produce NaN."""
    cfg = MHBPConfig(timescale_mode="context", ctx_dim=5, allostasis=True)
    m = CoupledMultiscaleHBP(cfg).double()
    m.reset_state(2)
    with torch.no_grad():
        ctx = torch.randn(2, 5)
        acts, info = m.tick(InteroceptiveSignal(values={"entropy": 1.0}), ctx=ctx)
        assert all(torch.isfinite(a).all() for a in acts.values())
        taus = m.timescales(ctx)
        assert (torch.diff(taus) > 0).all()
        # contexto tóxico: inf/nan → τ finitas y ordenadas (guarda nan_to_num)
        bad = torch.tensor([[float("inf"), float("nan"), 1e30, -1e30, 0.0]] * 2)
        taus_bad = m.timescales(bad)
        assert torch.isfinite(taus_bad).all() and (torch.diff(taus_bad) > 0).all()
    assert exact_spectral_radius(m) < 1.0


def test_heterogeneous_d_allostasis():
    """Campos con d distinto + alostasis: ya no crashea (minor confirmado)."""
    fcs = [FieldConfig(name="a", n_nodes=3, d=2), FieldConfig(name="b", n_nodes=2, d=5),
           FieldConfig(name="c", n_nodes=4, d=3)]
    cfg = MHBPConfig(d=2, taus=(1.0, 4.0, 16.0), allostasis=True)
    m = CoupledMultiscaleHBP(cfg, field_cfgs=fcs).double()
    m.reset_state(2)
    with torch.no_grad():
        acts, _ = m.tick(InteroceptiveSignal(values={"entropy": 1.0}))
    assert all(torch.isfinite(a).all() for a in acts.values())


def test_pin_fp32_covers_everything():
    """pin_fp32 re-ancla TODOS los parámetros (major: antes dejaba 25 en bf16)."""
    m = CoupledMultiscaleHBP(MHBPConfig(timescale_mode="learnable"))
    m = m.to(torch.bfloat16).pin_fp32()
    bad = [n for n, p in m.named_parameters() if p.dtype != torch.float32]
    assert not bad, f"parámetros sin re-anclar: {bad}"


def test_dissipation_50_random_configs():
    """Criterio §9 como está escrito: 50 CONFIGS aleatorias (params
    re-aleatorizados), θ=1, F=0 ⇒ E no creciente en todas."""
    from mhbp.stability.certificates import energy_dissipation_check
    for seed in range(50):
        torch.manual_seed(seed)
        cfg = MHBPConfig(dt=0.5, theta=1.0,
                         coupling_topology=["chain", "full", "none"][seed % 3],
                         allostasis=False)
        m = CoupledMultiscaleHBP(cfg).double()
        for f in m.fields:
            for p in f.params.parameters():
                p.data.uniform_(-4, 4)
        rep = energy_dissipation_check(m, trials=2, ticks=15, seed=seed)
        assert rep["ok"], f"config {seed}: {rep}"


def test_energy_identity_autodiff():
    """Criterio §9: Ė por AUTODIFF sobre integ.energy del CayleyIMEX real."""
    m = CoupledMultiscaleHBP(MHBPConfig(allostasis=False)).double()
    integ = m.build_integrator()
    n = integ.n_tot
    taus = m.timescales()
    # T⁻¹ por entrada (mismo layout que hvec)
    Tinv = integ._hvec / m.cfg.dt
    torch.manual_seed(7)
    U = torch.randn(1, n, dtype=torch.float64, requires_grad=True)
    W = torch.randn(1, n, dtype=torch.float64, requires_grad=True)
    E = integ.energy(U, W).sum()
    gU, gW = torch.autograd.grad(E, (U, W))
    F = torch.randn(1, n, dtype=torch.float64)
    # dinámica continua: U̇ = T⁻¹W ; Ẇ = T⁻¹(−(C+G)W − 𝓚U + F)
    from mhbp.integrators.verlet import VerletRef
    vr = VerletRef(); vr.prepare(list(m.fields), taus, m.coupling, m.cfg.dt)
    Udot = Tinv * W.detach()
    Wdot = Tinv * (-(W.detach() @ vr._CG.T) - U.detach() @ vr._K.T + F)
    Edot_auto = (gU * Udot).sum() + (gW * Wdot).sum()
    Csym = 0.5 * (vr._CG + vr._CG.T)
    w = W.detach().squeeze(0)
    Edot_formula = -(w * Tinv) @ (Csym @ w) + (w * Tinv) @ F.squeeze(0)
    assert abs(float(Edot_auto - Edot_formula)) < 1e-9 * max(1.0, abs(float(Edot_auto)))


def test_transient_growth_max_advection():
    """Transitorio no-normal con b,β al TOPE (θ=1): medido y acotado; energía
    no creciente incluso en el caso más no-normal."""
    cfg = MHBPConfig(dt=0.5, theta=1.0, allostasis=False)
    fcs = [FieldConfig(name=f"f{i}", n_nodes=6, d=4, b_init=0.29, b_max=0.3,
                       beta_init=0.09, beta_max=0.1, D_init=0.0, D_max=1e-9)
           for i in range(4)]
    m = CoupledMultiscaleHBP(cfg, field_cfgs=fcs).double()
    for f in m.fields:
        f.params.raw_b.data.fill_(40.0)
        f.params.raw_beta.data.fill_(40.0)
    rep = transient_growth_check(m, kmax=600)
    assert torch.isfinite(torch.tensor(rep["max_transient_2norm"]))
    assert rep["max_transient_2norm"] < 10.0, "transitorio no-normal desbocado"
    # el campo lento (τ=32) contrae a ~e^(−0.011k): a k=600 el bump transitorio
    # (~1.16 en k~100) ya debe estar por debajo de 1
    assert rep["final_2norm"] < 1.0, f"Φ^600 no contraído: {rep}"
    from mhbp.stability.certificates import energy_dissipation_check
    ed = energy_dissipation_check(m, trials=5, ticks=50)
    assert ed["ok"], f"disipación rota con advección máxima: {ed}"


def test_permute_fields_intervention():
    """La intervención 'permutar señales' del §8 existe y cambia las acciones."""
    m = CoupledMultiscaleHBP(MHBPConfig(allostasis=False)).double()
    m.reset_state(1)
    with torch.no_grad():
        for W in m.actuators.W:
            W.mul_(50.0)
        for t in range(5):
            a_ref, _ = m.tick(InteroceptiveSignal(values={"entropy": 2.0}))
        m.actuators.permute_fields([3, 2, 1, 0])
        a_perm, _ = m.actuators(([f.node_mean() for f in m.fields]))
        m.actuators.permute_fields(None)
    assert not torch.allclose(a_ref["halt_bias"], a_perm["halt_bias"]), \
        "permutar campos con estados distintos debe cambiar la acción"
    with pytest.raises(ValueError):
        m.actuators.permute_fields([0, 0, 1, 2])


def test_checkpoint_helpers_including_allostasis():
    """save/load_dynamic_state: trayectoria y REGULARIZADOR idénticos tras resume."""
    def sig(t):
        g = torch.Generator().manual_seed(t)
        return InteroceptiveSignal(values={"entropy": torch.rand(2, generator=g)})

    m1 = CoupledMultiscaleHBP(MHBPConfig()).double()
    m1.reset_state(2)
    with torch.no_grad():
        for t in range(8):
            m1.tick(sig(t))
    ckpt = {"sd": m1.state_dict(), "dyn": m1.save_dynamic_state()}

    m2 = CoupledMultiscaleHBP(MHBPConfig()).double()
    m2.load_state_dict(ckpt["sd"])
    m2.load_dynamic_state(ckpt["dyn"])
    with torch.no_grad():
        for t in range(8, 12):
            a1, _ = m1.tick(sig(t))
            a2, _ = m2.tick(sig(t))
            for k in a1:
                assert torch.equal(a1[k], a2[k])
    r1 = float(m1.regularizers()["allostasis"])
    r2 = float(m2.regularizers()["allostasis"])
    assert abs(r1 - r2) < 1e-12, "regularizador de alostasis diverge tras resume"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sin GPU")
def test_cpu_gpu_consistency():
    """Criterio §27: misma trayectoria FP64 en CPU y CUDA (tolerancia 1e-9)."""
    def run(device):
        torch.manual_seed(11)
        m = CoupledMultiscaleHBP(MHBPConfig(allostasis=False)).double().to(device)
        m.reset_state(2, device=torch.device(device))
        outs = []
        with torch.no_grad():
            for t in range(20):
                g = torch.Generator().manual_seed(t)
                s = InteroceptiveSignal(values={"entropy": torch.rand(2, generator=g)})
                acts, _ = m.tick(s)
                outs.append(acts["halt_bias"].cpu())
        return outs, float(exact_spectral_radius(m))

    o_cpu, rho_cpu = run("cpu")
    o_gpu, rho_gpu = run("cuda")
    for a, b in zip(o_cpu, o_gpu):
        assert torch.allclose(a, b, atol=1e-9), f"CPU vs GPU: {(a-b).abs().max():.2e}"
    assert abs(rho_cpu - rho_gpu) < 1e-9
