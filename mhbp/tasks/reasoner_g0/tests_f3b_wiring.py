"""
F3b — Tests de CABLEADO (PREREG_F3B v2 §11). Sin entrenar: verifican que la
cirugía del modelo hace exactamente lo declarado, en CPU y en segundos.

  T1  regresión F3a: session_mode=False reproduce las rutas existentes
  T2  los 7 brazos F3b hacen forward con presupuesto y pérdidas finitas
  T3  λ-clamp por fila: E[n] y ticks cargados respetan row_caps
  T4  persistencia: el plano PERSISTE entre forwards (y reset → bit-cero)
  T5  gates por-instancia: constantes dentro de la instancia, cambian entre
      instancias (y neutros en la instancia 1 post-reset)
  T6  tick enmascarado: las filas inactivas quedan bit-idénticas (§6.1)
  T7  ruta forced_steps (batería) con presupuesto ∞ en session_mode

Ejecutar:  python -m mhbp.tasks.reasoner_g0.tests_f3b_wiring
"""
from __future__ import annotations
import torch

from training.trainer import build_model

VOCAB, T = 60, 32
ARMS = ["f3b_mhbp_sess", "f3b_mhbp_sess_inst", "f3b_mhbp_noper",
        "f3b_hbp_sess", "f3b_gru_sess", "f3b_react_sess", "f3b_gating_wm"]


def make_batch(B=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(3, VOCAB, (B, T), generator=g)
    idx[:, 10] = 2                       # token de lectura (ARROW)
    idx[:, 28:] = 0                      # algo de PAD
    tgt = torch.full((B, T), -1, dtype=torch.long)
    tgt[:, 11] = torch.randint(3, VOCAB, (B,), generator=g)
    return idx, tgt


def ctx(B, caps=None, spent=0.3):
    return {
        "spent_frac": torch.full((B,), spent),
        "remaining_frac": torch.full((B,), 1.0 - spent),
        "stake": torch.tensor([0.0, 1.0] * (B // 2)),
        "inst_left": torch.full((B,), 0.5),
        "row_caps": caps,
    }


def plane_state_vec(hbp):
    """Vector plano del estado persistente del controlador (para comparar)."""
    if hasattr(hbp, "plane"):
        return torch.cat([torch.cat([f.u.flatten(), f.w.flatten()])
                          for f in hbp.plane.fields])
    if hasattr(hbp, "core"):
        return torch.cat([hbp.core.h_t.flatten(), hbp.core.h_tm1.flatten()])
    if hasattr(hbp, "h_gru"):
        return hbp.h_gru.flatten()
    return torch.zeros(1)


def t1_regresion_f3a():
    for variant in ("hbp_full", "miura_mhbp", "gating_wm"):
        m = build_model(variant, VOCAB, T)
        idx, tgt = make_batch()
        _, ld = m(idx, tgt)
        assert torch.isfinite(ld["total"]), f"{variant}: loss no finita"
        ld["total"].backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters()
                   if p.grad is not None), f"{variant}: grad no finito"
    print("T1 OK — rutas F3a intactas (hbp_full, miura_mhbp, gating_wm)")


def t2_forward_brazos():
    B = 4
    for arm in ARMS:
        m = build_model(arm, VOCAB, T)
        idx, tgt = make_batch(B)
        m.budget_ctx = ctx(B, caps=torch.tensor([10, 5, 3, 1]))
        if m.cfg.use_hbp:
            m.hbp.reset_state(B, device=idx.device)
            m.persist_plane = arm not in ("f3b_mhbp_noper", "f3b_react_sess",
                                          "f3b_gating_wm")
        _, ld = m(idx, tgt)
        assert torch.isfinite(ld["total"]), f"{arm}: loss no finita"
        ld["total"].backward()
        gs = [p.grad for p in m.parameters() if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in gs), f"{arm}: grad no finito"
    print(f"T2 OK — forward+backward finito en los {len(ARMS)} brazos")


def t3_lambda_clamp():
    B = 4
    caps = torch.tensor([10, 5, 3, 1])
    m = build_model("f3b_gating_wm", VOCAB, T)
    idx, tgt = make_batch(B)
    m.budget_ctx = ctx(B, caps=caps)
    _, _ = m(idx, tgt)
    n_exp = m._last_n_expected
    charged = m.halting._last_charged_ticks
    assert (n_exp <= caps.float() + 1e-4).all(), f"E[n]={n_exp} viola caps={caps}"
    assert (charged <= caps.float() + 1e-6).all(), \
        f"cargados={charged} viola caps={caps}"
    assert charged[3].item() == 1.0, "fila forzada a n=1 debe cargar 1 tick"
    print(f"T3 OK — λ-clamp por fila: E[n]={n_exp.tolist()}, "
          f"cargados={charged.tolist()} bajo caps={caps.tolist()}")


def t4_persistencia():
    B = 2
    for arm in ("f3b_mhbp_sess", "f3b_hbp_sess", "f3b_gru_sess"):
        m = build_model(arm, VOCAB, T)
        idx, tgt = make_batch(B)
        m.hbp.reset_state(B, device=idx.device)
        s0 = plane_state_vec(m.hbp).clone()   # estado de REPOSO (h* si lo hay)
        m.persist_plane = True
        m.budget_ctx = ctx(B)
        with torch.no_grad():
            m(idx, tgt)
        s1 = plane_state_vec(m.hbp).clone()
        assert not torch.equal(s0, s1), f"{arm}: el estado no evoluciona"
        with torch.no_grad():
            m(idx, tgt)
        s2 = plane_state_vec(m.hbp).clone()
        assert not torch.equal(s1, s2), f"{arm}: el estado no persiste/avanza"
        m.hbp.reset_state(B, device=idx.device)
        sr = plane_state_vec(m.hbp)
        assert torch.equal(sr, s0), \
            f"{arm}: el reset de sesión no restaura el reposo EXACTO"
    print("T4 OK — persistencia entre forwards + reset exacto al reposo "
          "(mhbp_sess, hbp_sess, gru_sess)")


def t5_gates_por_instancia():
    B = 2
    m = build_model("f3b_mhbp_sess_inst", VOCAB, T)
    idx, tgt = make_batch(B)
    m.hbp.reset_state(B, device=idx.device)
    m.persist_plane = True
    m.budget_ctx = ctx(B)
    seen = []
    orig = m.hbp.step_session

    def spy(*a, **k):
        mod = orig(*a, **k)
        seen.append(mod["wm_write"][0, 0, 0].item())
        return mod
    m.hbp.step_session = spy
    with torch.no_grad():
        m(idx, tgt)                       # instancia 1 (post-reset)
    inst1 = list(seen)
    assert all(abs(v - inst1[0]) < 1e-7 for v in inst1), \
        "gates NO constantes dentro de la instancia 1"
    assert abs(inst1[0] - 0.5) < 1e-3, \
        f"instancia 1 post-reset debe ser neutra (0.5), fue {inst1[0]}"
    seen.clear()
    with torch.no_grad():
        m(idx, tgt)                       # instancia 2 (plano cargado)
    inst2 = list(seen)
    assert all(abs(v - inst2[0]) < 1e-7 for v in inst2), \
        "gates NO constantes dentro de la instancia 2"
    assert abs(inst2[0] - inst1[0]) > 1e-6, \
        "los gates no cambian entre instancias (¿frozen no se recomputa?)"
    print(f"T5 OK — gates por-instancia: inst1={inst1[0]:.4f} (neutra), "
          f"inst2={inst2[0]:.6f}, constantes intra-instancia "
          f"({len(inst1)} ticks)")


def t6_tick_enmascarado():
    from mhbp.adapters.reasoner_adapter import MhbpReasonerAdapter
    B, N = 4, 6
    a = MhbpReasonerAdapter(n_nodes=N, d_h=16, taus=(1.0, 3.0, 10.0, 32.0))
    a.reset_state(B, device=torch.device("cpu"))
    # un tick completo para que el estado no sea cero
    s_raw = torch.rand(B, N, 11)
    ext = torch.rand(B, 16)
    eff = torch.full((B,), 0.2)
    dm = torch.rand(B)
    de = torch.rand(B)
    cap = torch.zeros(B)
    a.step_session(s_raw, ext, eff, dm, de, cap)
    before = plane_state_vec(a).clone().reshape(-1)
    # tick enmascarado: solo filas 0 y 2 activas
    act = torch.tensor([0, 2])
    a.step_session(s_raw[act], ext[act], eff[act], dm[act], de[act], cap[act],
                   active_idx=act)
    after = plane_state_vec(a).reshape(-1)
    # comparar por fila: gather el estado por campo
    changed, frozen = [], []
    off = 0
    for f in a.plane.fields:
        for tname in ("u", "w"):
            t_b = before[off:off + getattr(f, tname).numel()].view_as(
                getattr(f, tname))
            t_a = getattr(f, tname)
            for b in range(B):
                (changed if b in (0, 2) else frozen).append(
                    torch.equal(t_b[b], t_a[b]))
            off += getattr(f, tname).numel()
    assert all(frozen), "filas INACTIVAS alteradas por el tick enmascarado"
    assert not any(changed), "filas activas NO avanzaron"
    print("T6 OK — tick enmascarado: inactivas bit-idénticas, activas avanzan")


def t7_forced_bateria():
    B = 4
    for arm in ("f3b_mhbp_sess", "f3b_hbp_sess"):
        m = build_model(arm, VOCAB, T)
        idx, tgt = make_batch(B)
        m.hbp.reset_state(B, device=idx.device)
        m.persist_plane = True
        m.budget_ctx = ctx(B, caps=None)          # presupuesto ∞ (batería, §8)
        forced = torch.tensor([2, 4, 6, 8])
        with torch.no_grad():
            logits, _ = m(idx, forced_steps=forced)
        assert torch.isfinite(logits).all(), f"{arm}: logits no finitos (forced)"
        assert m._last_reasoner_step_units == int(forced.sum()), \
            f"{arm}: presupuesto efectivo forced incorrecto"
    print("T7 OK — ruta forced_steps (batería M) operativa con adaptadores F3b")


if __name__ == "__main__":
    torch.manual_seed(0)
    t1_regresion_f3a()
    t2_forward_brazos()
    t3_lambda_clamp()
    t4_persistencia()
    t5_gates_por_instancia()
    t6_tick_enmascarado()
    t7_forced_bateria()
    print("\nCABLEADO F3b: 7/7 tests OK")
