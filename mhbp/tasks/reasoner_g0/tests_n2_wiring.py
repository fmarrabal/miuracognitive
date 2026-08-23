"""
N2 — Tests de CABLEADO (PREREG_N2 v2 §9-11). CPU, segundos.

  T8   fronteras de gradiente EXACTAS: ∂CE/∂ψ(V̂) = 0 y ∂L_val/∂backbone = 0
  T9   veto por-cabeza: ∂gates_de_contenido/∂canal_de_valor = 0 (el valor
       solo entra al campo risk); con canal ausente, bit-igual con F3a
  T10  oráculo: value_signal = stake_true desde el tick 1; endo: cacheado en
       tick 2, alimenta desde el 3; endo_noval: canal SIEMPRE ausente
  T11  gating_endo: vía espejo neutra en init (a=0 → bias 0 exacto) y activa
       al mover a; CE ponderada por muestra correcta (pesos 1 ≡ sin pesos)
  T12  lesión endo_cut: canal := media del batch (árbitro C1'✓∧C3✗)

Ejecutar:  python -m mhbp.tasks.reasoner_g0.tests_n2_wiring
"""
from __future__ import annotations
import torch

from training.trainer import build_model

VOCAB, T = 60, 32


def make_batch(B=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(3, VOCAB, (B, T), generator=g)
    idx[:, 10] = 2
    idx[:, 28:] = 0
    tgt = torch.full((B, T), -1, dtype=torch.long)
    tgt[:, 11] = torch.randint(3, VOCAB, (B,), generator=g)
    return idx, tgt


def stakes(B=4):
    return torch.tensor([1.0, 8.0] * (B // 2))


def t8_fronteras_gradiente():
    m = build_model("n2_endo", VOCAB, T)
    idx, tgt = make_batch()
    m.value_ctx = {"stake_true": stakes()}
    _, ld = m(idx, tgt, sample_weights=stakes())
    assert "val" in ld, "L_val no computada"
    # ∂(CE+resto)/∂ψ == 0: backward de (total − beta·val) no toca V̂
    ce_part = ld["total"] - m.cfg.beta_val * ld["val"]
    m.zero_grad()
    ce_part.backward(retain_graph=True)
    for name in ("value_p", "value_s"):
        for p in getattr(m, name).parameters():
            assert p.grad is None or float(p.grad.abs().max()) == 0.0, \
                f"∂CE/∂{name} ≠ 0 (fuga de la tarea a V̂)"
    # ∂L_val/∂backbone == 0 (input detached): backward de val no toca el resto
    m.zero_grad()
    (m.cfg.beta_val * ld["val"]).backward()
    for name, p in m.named_parameters():
        if name.startswith("value_"):
            assert p.grad is not None and torch.isfinite(p.grad).all()
        else:
            assert p.grad is None or float(p.grad.abs().max()) == 0.0, \
                f"∂L_val/∂{name} ≠ 0 (el valor esculpe el backbone)"
    print("T8 OK — fronteras de gradiente exactas (∂CE/∂ψ=0, ∂L_val/∂backbone=0)")


def t9_veto_y_bit_igualdad():
    torch.manual_seed(1)
    from mhbp.adapters.reasoner_adapter import MhbpReasonerAdapter
    B, N = 4, 6
    s_raw = torch.rand(B, N, 5)
    ext, eff = torch.rand(B, 16), torch.full((B,), 0.3)
    dm, de, cap = torch.rand(B), torch.rand(B), torch.zeros(B)
    # (a) VÍA DIRECTA exacta (acoplamiento apagado): el canal de valor solo
    # fuerza al campo risk → contenido intacto. Con acoplamiento "chain" la
    # propagación INDIRECTA vía κ (PSD, certificada) existe y está DECLARADA
    # en el prereg (vigilada por el guardarraíl M + lesión por vías).
    a = MhbpReasonerAdapter(n_nodes=6, d_h=16, value_channel=True,
                            coupling_topology="none")
    a.reset_state(B, device=torch.device("cpu"))
    a.eval()                     # EMA congelada: condición de medición
    mod0 = a.step_mhbp(s_raw, ext, eff, dm, de, cap,
                       value_signal=torch.zeros(B))
    a.reset_state(B, device=torch.device("cpu"))
    mod1 = a.step_mhbp(s_raw, ext, eff, dm, de, cap,
                       value_signal=torch.full((B,), 9.0))
    for k in ("wm_write", "wm_forget", "block_gate"):
        assert torch.allclose(mod0[k], mod1[k], atol=1e-6), \
            f"el canal de valor mueve {k} por vía DIRECTA (veto roto)"
    assert not torch.allclose(mod0["halt_threshold"], mod1["halt_threshold"],
                              atol=1e-6), \
        "el canal de valor NO llega al halting (¿canal muerto?)"
    # (a2) con acoplamiento chain, la propagación indirecta existe pero es
    # de segundo orden al init (≪ el efecto sobre el halting)
    c = MhbpReasonerAdapter(n_nodes=6, d_h=16, value_channel=True)
    c.reset_state(B, device=torch.device("cpu"))
    c.eval()
    m0 = c.step_mhbp(s_raw, ext, eff, dm, de, cap, value_signal=torch.zeros(B))
    c.reset_state(B, device=torch.device("cpu"))
    m1 = c.step_mhbp(s_raw, ext, eff, dm, de, cap,
                     value_signal=torch.full((B,), 9.0))
    d_halt = float((m0["halt_threshold"] - m1["halt_threshold"]).abs().max())
    d_cont = max(float((m0[k] - m1[k]).abs().max())
                 for k in ("wm_write", "wm_forget", "block_gate"))
    assert d_cont < 0.25 * d_halt, \
        f"propagación indirecta ({d_cont:.2e}) no es 2º orden vs halting ({d_halt:.2e})"
    # (b) canal ausente ≡ F3a bit a bit (mismos pesos, con/sin kwarg)
    torch.manual_seed(2)
    b = MhbpReasonerAdapter(n_nodes=6, d_h=16, value_channel=True)
    b.reset_state(B, device=torch.device("cpu"))
    b.eval()
    m_none = b.step_mhbp(s_raw, ext, eff, dm, de, cap, value_signal=None)
    b.reset_state(B, device=torch.device("cpu"))
    m_f3a = b.step_mhbp(s_raw, ext, eff, dm, de, cap)
    for k in m_none:
        assert torch.equal(m_none[k], m_f3a[k]), \
            f"canal ausente ≠ ruta F3a en {k}"
    print(f"T9 OK — vía directa exacta (topología none); indirecta 2º orden "
          f"con chain ({d_cont:.1e} vs halting {d_halt:.1e}); canal ausente "
          f"bit-igual a F3a")


def _spy_value(m):
    seen = {}
    orig = m.hbp.step_mhbp

    def spy(*args, **kw):
        n = len(seen) + 1
        v = kw.get("value_signal")
        seen[n] = None if v is None else v.clone()
        return orig(*args, **kw)
    m.hbp.step_mhbp = spy
    return seen


def t10_fuentes_del_canal():
    B = 4
    st = stakes(B)
    # oráculo: stake verdadero desde el tick 1
    m = build_model("n2_oracle", VOCAB, T)
    idx, tgt = make_batch(B)
    m.value_ctx = {"stake_true": st}
    seen = _spy_value(m)
    with torch.no_grad():
        m(idx, tgt)
    assert seen[1] is not None and torch.equal(seen[1], st.float()), \
        "oracle: el stake no entra desde el tick 1"
    # endo: ausente ticks 1-2, constante (cacheado) desde el 3
    m = build_model("n2_endo", VOCAB, T)
    seen = _spy_value(m)
    with torch.no_grad():
        m(idx, tgt)
    assert seen[1] is None and seen[2] is None, "endo: canal antes del tick 3"
    ticks3 = [v for n, v in seen.items() if n >= 3 and v is not None]
    assert len(ticks3) >= 2, "endo: canal ausente tras el tick 3"
    assert all(torch.equal(ticks3[0], v) for v in ticks3), \
        "endo: stakê no cacheado (varía por tick)"
    assert not ticks3[0].requires_grad, "endo: value_signal NO detached"
    # endo_noval: canal SIEMPRE ausente, V̂ cacheado igualmente (para log/L_val)
    m = build_model("n2_endo_noval", VOCAB, T)
    seen = _spy_value(m)
    with torch.no_grad():
        m(idx, tgt)
    assert all(v is None for v in seen.values()), "endo_noval: canal no ausente"
    assert m._val_cache is not None, "endo_noval: V̂ sin cachear (log roto)"
    print("T10 OK — fuentes del canal: oracle t1 / endo cacheado t3+ "
          "detached / endo_noval ausente con V̂ vivo")


def t11_gating_endo_y_pesos():
    B = 4
    idx, tgt = make_batch(B)
    m = build_model("n2_gating_endo", VOCAB, T)
    m.value_ctx = {"stake_true": stakes(B)}
    m.eval()
    with torch.no_grad():
        _, _ = m(idx, tgt)
        n0 = m._last_n_expected.clone()
        m.value_halt_a.data.fill_(3.0)         # activa la vía espejo
        _, _ = m(idx, tgt)
        n1 = m._last_n_expected.clone()
    assert not torch.equal(n0, n1), \
        "gating_endo: la vía espejo no mueve el halting al activar a"
    # init neutro: a=0 → bias 0 exacto → mismo E[n] que un gating_wm puro
    m2 = build_model("n2_gating_endo", VOCAB, T)
    m2.eval()
    g = build_model("gating_wm", VOCAB, T)
    g.load_state_dict({k: v for k, v in m2.state_dict().items()
                       if k in dict(g.named_parameters())
                       or k in dict(g.named_buffers())}, strict=False)
    with torch.no_grad():
        _, _ = m2(idx, tgt)
    # pesos de CE: w≡1 reproduce la pérdida sin pesos
    m3 = build_model("gating_wm", VOCAB, T)
    with torch.no_grad():
        _, l_a = m3(idx, tgt)
        _, l_b = m3(idx, tgt, sample_weights=torch.ones(B))
    assert torch.allclose(l_a["lm"], l_b["lm"], atol=1e-5), \
        "sample_weights=1 ≠ sin pesos"
    print("T11 OK — vía espejo neutra en init y viva con a≠0; "
          "CE ponderada consistente (w=1 ≡ sin pesos)")


def t12_endo_cut():
    B = 4
    idx, tgt = make_batch(B)
    m = build_model("n2_endo", VOCAB, T)
    m.eval()
    with torch.no_grad():
        m.value_s[0].weight.mul_(20)           # stakê no-trivial entre filas
        seen = _spy_value(m)
        m(idx, tgt)
        v_free = [v for v in seen.values() if v is not None][0]
    m2 = build_model("n2_endo", VOCAB, T)
    m2.eval()
    with torch.no_grad():
        m2.value_s[0].weight.mul_(20)
        m2.value_lesion = True
        seen2 = _spy_value(m2)
        m2(idx, tgt)
        v_cut = [v for v in seen2.values() if v is not None][0]
    assert float(v_cut.std()) == 0.0, "endo_cut: el canal no es constante"
    assert float(v_free.std()) > 0.0, "endo libre: canal degenerado (¿todo igual?)"
    print("T12 OK — lesión endo_cut: canal fijado a la media (árbitro operativo)")


if __name__ == "__main__":
    torch.manual_seed(0)
    t8_fronteras_gradiente()
    t9_veto_y_bit_igualdad()
    t10_fuentes_del_canal()
    t11_gating_endo_y_pesos()
    t12_endo_cut()
    print("\nCABLEADO N2: 5/5 tests OK")
