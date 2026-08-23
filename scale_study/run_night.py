"""
Barrido NOCTURNO del campo homeostático a escala gigante (Estudio A del pre-registro).
Resumible: 1 JSON atómico por celda; salta las hechas. Límite de tiempo duro.

Bloques (orden de ejecución = prioridad científica; el caro B3 va pronto):
  B1  espectro multi-topología multi-N
  B4  estabilidad/flutter a escala (certificado vs rollout real)
  B2  relación de dispersión en el anillo (medida vs analítica)
  B3a SOLITONES núcleo (anillo FFT): emergencia, saturación (H3s), amortiguamiento
  B5  onda amortiguada GIGANTE (propagación, la figura del paper a N=10⁶)
  B3b colisiones de solitones (elasticidad)
  B3c recurrencia FPUT
  B3x solitones a N enorme (llena el presupuesto)

Uso:
  python -m scale_study.run_night --calibrate      # mide tiempos + confirma solitones
  python -m scale_study.run_night --hours 9.5      # barrido completo con tope de 9.5 h
"""
import os, sys, json, time, math, argparse, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from scale_study.field_sim import (
    build_ops, RingOps, Phys, run_wave, run_kdv_ring, kdv_ring_dispersion,
    detect_peaks, track_soliton, flutter_threshold, companion_spectral_radius,
    fwhm, pick_device, F64, F32,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
DEV = pick_device()
T0 = time.time()


def log(msg):
    line = f"[{time.time()-T0:8.1f}s] {msg}"
    print(line, flush=True)
    with open(os.path.join(RES, "progress.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_atomic(cell_id, payload):
    payload["cell_id"] = cell_id
    payload["wall_s"] = round(time.time() - payload.pop("_t_start", time.time()), 2)
    tmp = os.path.join(RES, f".{cell_id}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, os.path.join(RES, f"{cell_id}.json"))


def done(cell_id):
    return os.path.exists(os.path.join(RES, f"{cell_id}.json"))


def sub(seq, n=256):
    """Submuestrea una lista a <=n puntos para guardar compacto."""
    if len(seq) <= n:
        return [round(float(x), 6) for x in seq]
    step = len(seq) / n
    return [round(float(seq[int(i * step)]), 6) for i in range(n)]


def sub_field(t, n=1024):
    """Submuestrea un campo 1D (tensor) a <=n nodos."""
    t = t.flatten()
    if t.numel() <= n:
        return [round(float(x), 5) for x in t.tolist()]
    idx = torch.linspace(0, t.numel() - 1, n).long()
    return [round(float(x), 5) for x in t[idx].tolist()]


# ============================================================================= #
# GENERADORES DE PULSOS
# ============================================================================= #
def gaussian_pulse(N, amp, width_frac, center_frac=0.5, device=DEV, dtype=F64):
    idx = torch.arange(N, device=device, dtype=dtype)
    c = center_frac * N
    w = width_frac * N
    # distancia circular
    d = torch.remainder(idx - c + N / 2, N) - N / 2
    return (amp * torch.exp(-(d / w) ** 2)).unsqueeze(0)


# ============================================================================= #
# CELDAS
# ============================================================================= #
def cell_B1(cfg):
    t_start = time.time()
    kind, N, seed = cfg["kind"], cfg["N"], cfg["seed"]
    ops = build_ops(kind, N, DEV, dtype=F64, seed=seed,
                    degree=cfg.get("degree", 4), k=cfg.get("k", 6), p=cfg.get("p", 0.1))
    # gap espectral algebraico (λ_2) para estructurados: analítico; para sparse: aprox
    out = {"_t_start": t_start, "block": "B1", "kind": kind, "N": N, "seed": seed,
           "rho_L": ops.rho_L, "rho_A": ops.rho_A, "rho_A3": ops.rho_A3,
           "fourier": ops.fourier}
    if kind == "ring":
        out["lambda2"] = 4 * math.sin(math.pi / N) ** 2
    elif kind == "chain":
        out["lambda2"] = 2 - 2 * math.cos(math.pi / N)
    # umbral de flutter para β de referencia
    pref = Phys(omega0=0.5, zeta=0.5, beta=0.05)
    lhs, rhs, stable = flutter_threshold(pref, ops)
    out["flutter_ref"] = {"lhs": lhs, "rhs": rhs, "stable": stable}
    return out


def cell_B2(cfg):
    """Relación de dispersión en el anillo. VECTORIZADO: todos los modos en un
    solo rollout batched (M,N). Mide ω(k) por FFT temporal de la proyección."""
    t_start = time.time()
    N, beta = cfg["N"], cfg["beta"]
    ops = RingOps(N)
    p = Phys(omega0=0.5, zeta=0.0, c=0.4, beta=beta, b=0.0, dt=0.1)
    ks = torch.tensor(cfg["modes"], dtype=F64, device=DEV)          # (M,)
    idx = torch.arange(N, device=DEV, dtype=F64)
    modes = torch.cos(2 * math.pi * ks.unsqueeze(1) * idx.unsqueeze(0) / N)  # (M,N)
    ht = 0.01 * modes.clone(); htm1 = ht.clone()
    T = cfg["ticks"]
    proj = torch.empty(T, ks.numel(), dtype=F64, device=DEV)
    for t in range(T):
        hn = 2 * ht - htm1 + p.dt ** 2 * (
            -(p.omega0 ** 2) * ht - (p.c ** 2) * ops.L(ht)
            - (p.beta / p.dt) * ops.A3(ht - htm1))
        htm1, ht = ht, hn
        proj[t] = (ht * modes).sum(-1)
    proj = proj - proj.mean(0, keepdim=True)
    spct = torch.fft.rfft(proj, dim=0).abs()                        # (T//2+1, M)
    kpk = spct[1:].argmax(0) + 1                                    # (M,)
    freq = (2 * math.pi * kpk.double() / T / p.dt).cpu().tolist()
    lamL = 4 * torch.sin(math.pi * ks / N) ** 2
    K = p.omega0 ** 2 + p.c ** 2 * lamL
    Omega = (torch.acos(torch.clamp(1 - p.dt ** 2 * K / 2, -1.0, 1.0)) / p.dt).cpu().tolist()
    return {"_t_start": t_start, "block": "B2", "N": N, "beta": beta,
            "modes": cfg["modes"], "omega_measured": [round(x, 6) for x in freq],
            "omega_analytic": [round(x, 6) for x in Omega]}


def single_soliton_run(N, amp, beta, nonlin, damp, ticks, dt, b=1.0, gamma=1.0, nu=1.0,
                       n_frames=240):
    """TEST LIMPIO de solitón: inicializa el perfil sech² EXACTO de KdV de amplitud
    `amp` y mide si se propaga sin deformarse. Teoría (u_t+αuu_x+δu_xxx=0, α=2ν/γ,
    δ=8β/γ): v_sol=α·amp/3, W=√(48β/amp) [para ν=γ=1], v_lab=(2b+α·amp/3)·dt."""
    alpha = 2.0 * nu / gamma
    delta = 8.0 * beta / gamma
    W = math.sqrt(max(delta, 1e-9) * 12.0 / (alpha * amp))    # =√(48β/amp) si ν=γ=1
    v_lab_theory = (2.0 * b / gamma + alpha * amp / 3.0) * dt  # nodos/tick
    idx = torch.arange(N, device=DEV, dtype=F64)
    d = torch.remainder(idx - N / 2 + N / 2, N) - N / 2
    u0 = (amp / torch.cosh(d / W) ** 2).unsqueeze(0)
    p = Phys(b=b, beta=beta, nu=nu, gamma=gamma, dt=dt, nonlin=nonlin)
    rec = max(1, ticks // n_frames)
    d0 = 0.5 if damp == "weak" else 0.0
    kd = run_kdv_ring(N, p, u0, ticks, device=DEV, record_every=rec,
                      damp_omega0=d0, damp_c=0.4 * d0)
    frames = [f.flatten().cpu() for f in kd["frames"]]
    track = track_soliton(frames, rec, N)
    vel = track.get("velocity_nodes_per_tick") if track else None
    return {
        "soliton_width_theory": W, "v_lab_theory": v_lab_theory,
        "velocity_measured": vel,
        "vel_err": (abs(vel - v_lab_theory) / (abs(v_lab_theory) + 1e-9)) if vel is not None else None,
        "amp_cv": track.get("amp_cv") if track else None,
        "width_cv": track.get("width_cv") if track else None,
        "amp_mean": track.get("amp_mean") if track else None,
        "amp_ratio_final_initial": (float(frames[-1].max()) / (float(frames[0].max()) + 1e-9)) if frames else None,
        "life_frac": track.get("life_frac") if track else None,
        "n_peaks_final": len(detect_peaks(frames[-1], 0.35)) if frames else 0,
        "finite": bool(math.isfinite(kd["norms"][-1])),
        "mass_drift": abs(kd["mass"][-1] - kd["mass"][0]) / (abs(kd["mass"][0]) + 1e-12),
        "frame_first": sub_field(frames[0], 1024) if frames else None,
        "frame_last": sub_field(frames[-1], 1024) if frames else None,
        "frames_stack": [sub_field(f, 512) for f in frames[::max(1, len(frames)//80)]],
    }


def solitons_run(N, amp, width, beta, nu, nonlin, damp, ticks, dt, b=1.0, gamma=1.0,
                 sign=1.0, n_frames=240):
    """Integra la rama KdV y hace tracking del pico dominante."""
    p = Phys(b=b, beta=beta, nu=nu, gamma=gamma, dt=dt, nonlin=nonlin)
    u0 = gaussian_pulse(N, sign * amp, width)
    rec = max(1, ticks // n_frames)
    d0 = 0.5 if damp == "weak" else 0.0
    kd = run_kdv_ring(N, p, u0, ticks, device=DEV, record_every=rec,
                      damp_omega0=d0, damp_c=(0.4 * d0), damp_zeta=0.0)
    frames = kd["frames"]
    dt_frames = rec
    # nº de solitones en el frame final (picos sobre 0.3·max)
    final = kd["final"].flatten().cpu()
    if sign < 0:
        final = -final
    npk = len(detect_peaks(final, thresh_frac=0.35))
    frames_pos = [(-f.flatten() if sign < 0 else f.flatten()).cpu() for f in frames]
    track = track_soliton(frames_pos, dt_frames, N)
    return {
        "n_solitons_final": npk,
        "track": track,
        "norm_trace": sub(kd["norms"], 200),
        "mass_trace": sub(kd["mass"], 200),
        "finite": bool(math.isfinite(kd["norms"][-1])),
        "frame_first": sub_field(frames_pos[0], 1024) if frames_pos else None,
        "frame_mid": sub_field(frames_pos[len(frames_pos)//2], 1024) if frames_pos else None,
        "frame_last": sub_field(frames_pos[-1], 1024) if frames_pos else None,
    }


def cell_B3s(cfg):
    """Solitón INDIVIDUAL (sech² exacto): test cuantitativo limpio."""
    t_start = time.time()
    r = single_soliton_run(cfg["N"], cfg["amp"], cfg["beta"], cfg["nonlin"],
                           cfg["damp"], cfg["ticks"], cfg["dt"], nu=cfg["nu"])
    r.update({"_t_start": t_start, "block": cfg.get("block", "B3s"),
              **{k: cfg[k] for k in ("N", "amp", "beta", "nu", "nonlin", "damp", "ticks", "dt")}})
    return r


def cell_B3p(cfg):
    """Pulso gaussiano genérico -> emergencia del TREN de solitones (sign +) vs
    dispersión (sign -). Guarda frames para la figura."""
    t_start = time.time()
    r = solitons_run(cfg["N"], cfg["amp"], cfg["width"], cfg["beta"], cfg["nu"],
                     cfg["nonlin"], cfg["damp"], cfg["ticks"], cfg["dt"], sign=cfg["sign"])
    # frames para figura del tren
    r["frames_stack"] = None  # solitons_run no los expone; recomputamos compacto abajo
    r.update({"_t_start": t_start, "block": "B3p", **{k: cfg[k] for k in
              ("N", "amp", "width", "beta", "nu", "nonlin", "damp", "ticks", "dt", "sign")}})
    return r


def sech2_soliton(N, amp, beta, nu, gamma, center_frac, device=DEV, dtype=F64):
    """Perfil sech² EXACTO de KdV de amplitud amp: W=√(48βγ/(ν·amp))."""
    W = math.sqrt(max(8.0 * beta / gamma, 1e-9) * 12.0 / (2.0 * nu / gamma * amp))
    idx = torch.arange(N, device=device, dtype=dtype)
    d = torch.remainder(idx - center_frac * N + N / 2, N) - N / 2
    return (amp / torch.cosh(d / W) ** 2).unsqueeze(0)


def cell_B3b(cfg):
    """Colisión de dos solitones sech² de amplitud distinta (elasticidad). El más
    alto (rápido) alcanza al más bajo (lento) y —si son solitones— emergen intactos."""
    t_start = time.time()
    N, nonlin = cfg["N"], cfg["nonlin"]
    nu = cfg["nu"]
    p = Phys(b=1.0, beta=cfg["beta"], nu=nu, gamma=1.0, dt=cfg["dt"], nonlin=nonlin)
    u0 = (sech2_soliton(N, cfg["amp1"], cfg["beta"], nu, 1.0, 0.25)
          + sech2_soliton(N, cfg["amp2"], cfg["beta"], nu, 1.0, 0.45))
    ticks = cfg["ticks"]; rec = max(1, ticks // 300)
    kd = run_kdv_ring(N, p, u0, ticks, device=DEV, record_every=rec)
    frames = [f.flatten().cpu() for f in kd["frames"]]
    # amplitudes de los dos picos mayores al principio y al final
    def top2(fr):
        pk = detect_peaks(fr, 0.25)
        amps = sorted([float(fr[i]) for i in pk], reverse=True)[:2]
        return amps
    a_start = top2(frames[0]) if frames else []
    a_end = top2(frames[-1]) if frames else []
    return {"_t_start": t_start, "block": "B3b", "N": N, "nonlin": nonlin,
            "amp1": cfg["amp1"], "amp2": cfg["amp2"], "beta": cfg["beta"], "nu": cfg["nu"],
            "amps_start": a_start, "amps_end": a_end,
            "n_start": len(detect_peaks(frames[0], 0.25)) if frames else 0,
            "n_end": len(detect_peaks(frames[-1], 0.25)) if frames else 0,
            "finite": bool(math.isfinite(kd["norms"][-1])),
            "frame_first": sub_field(frames[0], 1024) if frames else None,
            "frame_last": sub_field(frames[-1], 1024) if frames else None,
            "frames_stack": [sub_field(f, 512) for f in frames[::max(1, len(frames)//60)]]}


def cell_B3c(cfg):
    """Recurrencia tipo FPUT: modo coseno de baja frecuencia, integración muy larga."""
    t_start = time.time()
    N = cfg["N"]
    p = Phys(b=1.0, beta=cfg["beta"], nu=cfg["nu"], gamma=1.0, dt=cfg["dt"], nonlin="genuine")
    idx = torch.arange(N, device=DEV, dtype=F64)
    u0 = (cfg["amp"] * torch.cos(2 * math.pi * idx / N)).unsqueeze(0)
    ticks = cfg["ticks"]; rec = max(1, ticks // 400)
    kd = run_kdv_ring(N, p, u0, ticks, device=DEV, record_every=rec)
    # energía en el primer modo de Fourier a lo largo del tiempo (recurrencia)
    e1 = []
    for f in kd["frames"]:
        fh = torch.fft.rfft(f.flatten())
        e1.append(float((fh[1].abs() ** 2)))
    return {"_t_start": t_start, "block": "B3c", "N": N, "beta": cfg["beta"], "nu": cfg["nu"],
            "amp": cfg["amp"], "ticks": ticks, "mode1_energy": sub(e1, 400),
            "finite": bool(math.isfinite(kd["norms"][-1]))}


def cell_B4(cfg):
    t_start = time.time()
    N, beta, place = cfg["N"], cfg["beta"], cfg["placement"]
    ops = RingOps(N)
    p = Phys(omega0=0.5, zeta=cfg["zeta"], c=0.0, beta=beta, dt=cfg["dt"])
    u0 = torch.randn(4, N, device=DEV, dtype=F64) * 0.05
    kd = run_wave(ops, p, u0, cfg["ticks"], placement=place)
    norms = kd["norms"]
    # pendiente de log-norma en la 2ª mitad (tasa de crecimiento/decaimiento)
    half = len(norms) // 2
    seg = [n for n in norms[half:] if n > 0 and math.isfinite(n)]
    if len(seg) > 5:
        import math as _m
        ln = [_m.log(n) for n in seg]
        xs = list(range(len(ln)))
        mx = sum(xs) / len(xs); my = sum(ln) / len(ln)
        slope = sum((xs[i]-mx)*(ln[i]-my) for i in range(len(xs))) / (sum((x-mx)**2 for x in xs)+1e-30)
    else:
        slope = float("inf")
    lhs, rhs, stable = flutter_threshold(p, ops)
    out = {"_t_start": t_start, "block": "B4", "N": N, "beta": beta, "placement": place,
           "zeta": cfg["zeta"], "log_slope": slope, "diverged": not math.isfinite(norms[-1]),
           "flutter": {"lhs": lhs, "rhs": rhs, "cert_stable": stable}}
    if N <= 128:
        out["rho_companion"] = companion_spectral_radius(ops, p, DEV, placement=place)
    return out


def cell_B5(cfg):
    t_start = time.time()
    N = cfg["N"]
    ops = RingOps(N)
    p = Phys(omega0=0.6, zeta=0.08, c=0.5, beta=cfg.get("beta", 0.0), dt=0.5)
    u0 = torch.zeros(1, N, device=DEV, dtype=F64)
    u0[0, N // 2] = 2.5                      # impulso local
    ticks = cfg["ticks"]; rec = max(1, ticks // 300)
    kd = run_wave(ops, p, u0, ticks, record_every=rec)
    # mapa nodo×tick (submuestreado espacialmente) alrededor del impulso
    frames = kd["frames"]
    win = min(N, 2048)
    lo = N // 2 - win // 2
    stack = []
    for f in frames:
        seg = f.flatten()[lo:lo + win].abs()
        stack.append(sub_field(seg, 512))
    return {"_t_start": t_start, "block": "B5", "N": N, "ticks": ticks,
            "norm_trace": sub(kd["norms"], 300), "propagation": stack,
            "finite": bool(math.isfinite(kd["norms"][-1]))}


# ============================================================================= #
# PLAN DEL BARRIDO
# ============================================================================= #
def plan(full=True):
    cells = []   # (cell_id, block_fn, cfg)
    TK = 32000          # ticks estándar de solitón (dt=0.05 -> 1600 u.t.)

    # ============ PRIORIDAD 1 — esencial (~1.5 h) ============
    # --- B1 espectro ---
    Ns_struct = [6, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
    Ns_sparse = [64, 256, 1024, 4096, 16384, 65536]
    for kind in ["chain", "ring", "grid2d"]:
        for N in Ns_struct:
            cells.append((f"B1_{kind}_N{N}", cell_B1, {"kind": kind, "N": N, "seed": 0}))
    for kind in ["random_regular", "ws_smallworld", "expander"]:
        for N in Ns_sparse:
            for seed in [0, 1, 2]:
                cells.append((f"B1_{kind}_N{N}_s{seed}", cell_B1,
                              {"kind": kind, "N": N, "seed": seed}))

    # --- B4 estabilidad/flutter a escala ---
    for N in [64, 256, 4096, 65536, 1048576]:
        for zeta in [0.1, 0.3]:
            for beta in [round(x, 4) for x in _linspace(0.0, 0.30, 11)]:
                for place in ["gyro", "circulatory"]:
                    cells.append((f"B4_N{N}_z{zeta}_b{beta}_{place}", cell_B4,
                                  {"N": N, "zeta": zeta, "beta": beta, "placement": place,
                                   "dt": 0.1, "ticks": 3000}))

    # --- B2 dispersión anillo (vectorizado) ---
    modes = lambda N: sorted(set(int(x) for x in _linspace(2, N // 2 - 2, 40)))
    for N in [256, 1024, 4096, 16384, 65536]:
        for beta in [0.0, 0.05]:
            cells.append((f"B2_N{N}_b{beta}", cell_B2,
                          {"N": N, "beta": beta, "modes": modes(N), "ticks": 4000}))

    # --- B3s SOLITONES núcleo (la joya cuantitativa) ---
    #  small-amp (ν=0.2, tanh≈lineal): genuine ≈ saturated, solitón limpio
    for N in [1024, 4096, 16384, 65536]:
        for amp in [0.3, 0.6]:
            for beta in [0.05, 0.1]:
                for nonlin in ["genuine", "saturated"]:
                    for damp in ["none", "weak"]:
                        cells.append((f"B3s_N{N}_A{amp}_b{beta}_{nonlin}_{damp}", cell_B3s,
                                      {"block": "B3s", "N": N, "amp": amp, "beta": beta, "nu": 0.2,
                                       "nonlin": nonlin, "damp": damp, "ticks": TK, "dt": 0.05}))
    #  large-amp (ν=0.1, tanh SATURA): revela H3s (genuine limpio, saturated degradado)
    for N in [1024, 4096, 16384, 65536]:
        for amp in [1.0, 1.5, 2.0]:
            for nonlin in ["genuine", "saturated"]:
                for damp in ["none", "weak"]:
                    cells.append((f"B3s_N{N}_A{amp}_b0.1_{nonlin}_{damp}", cell_B3s,
                                  {"block": "B3s", "N": N, "amp": amp, "beta": 0.1, "nu": 0.1,
                                   "nonlin": nonlin, "damp": damp, "ticks": TK, "dt": 0.05}))

    # --- B3p tren de solitones vs dispersión (figura; el signo decide) ---
    for N in [4096, 16384]:
        for sign in [1.0, -1.0]:
            for nonlin in ["genuine", "saturated"]:
                cells.append((f"B3p_N{N}_s{int(sign)}_{nonlin}", cell_B3p,
                              {"N": N, "amp": 0.8, "width": 0.02, "beta": 0.1, "nu": 0.2,
                               "nonlin": nonlin, "damp": "none", "ticks": TK, "dt": 0.05, "sign": sign}))

    # --- B5 onda amortiguada GIGANTE (figura del paper a N=10⁶) ---
    for N in [10000, 100000, 1000000]:
        cells.append((f"B5_N{N}", cell_B5, {"N": N, "ticks": 3000}))

    # ============ PRIORIDAD 2 — llena la noche (~5-7 h) ============
    # --- B3b colisiones (elasticidad) ---
    for N in [4096, 16384, 65536, 262144]:
        for nonlin in ["genuine", "saturated"]:
            cells.append((f"B3b_N{N}_{nonlin}", cell_B3b,
                          {"N": N, "amp1": 0.9, "amp2": 0.35, "beta": 0.1, "nu": 0.2,
                           "nonlin": nonlin, "ticks": 40000, "dt": 0.05}))

    # --- B3c recurrencia FPUT (integración muy larga) ---
    for N in [1024, 4096, 16384]:
        for beta in [0.05, 0.1]:
            cells.append((f"B3c_N{N}_b{beta}", cell_B3c,
                          {"N": N, "amp": 0.4, "beta": beta, "nu": 0.2,
                           "ticks": 120000, "dt": 0.05}))

    # --- B3map MAPA DE FASE denso de solitones (amplitud × dispersión) a escala ---
    #  el estudio de relleno principal: dónde hay solitones limpios, genuine vs saturated
    for N in [4096, 16384, 65536, 262144, 1048576]:
        for amp in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]:
            for beta in [0.03, 0.05, 0.075, 0.1, 0.15, 0.2]:
                for nonlin in ["genuine", "saturated"]:
                    cells.append((f"B3map_N{N}_A{amp}_b{beta}_{nonlin}", cell_B3s,
                                  {"block": "B3map", "N": N, "amp": amp, "beta": beta, "nu": 0.15,
                                   "nonlin": nonlin, "damp": "none", "ticks": 20000, "dt": 0.05}))

    # --- B3x solitones a N GIGANTE (coherencia a escala masiva) ---
    if full:
        for N in [131072, 262144, 524288, 1048576, 2097152]:
            for nonlin in ["genuine", "saturated"]:
                for amp in [0.4, 0.8]:
                    cells.append((f"B3x_N{N}_A{amp}_{nonlin}", cell_B3s,
                                  {"block": "B3x", "N": N, "amp": amp, "beta": 0.1, "nu": 0.15,
                                   "nonlin": nonlin, "damp": "none", "ticks": 30000, "dt": 0.05}))

    # --- B3long persistencia/recurrencia larguísima ---
    for N in [2048, 8192]:
        for nonlin in ["genuine", "saturated"]:
            for beta in [0.05, 0.1]:
                cells.append((f"B3long_N{N}_b{beta}_{nonlin}", cell_B3s,
                              {"block": "B3x", "N": N, "amp": 0.5, "beta": beta, "nu": 0.15,
                               "nonlin": nonlin, "damp": "none", "ticks": 200000, "dt": 0.05}))
    return cells


def _linspace(a, b, n):
    if n == 1:
        return [a]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


# ============================================================================= #
# MAIN
# ============================================================================= #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=9.5)
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    if args.calibrate:
        calibrate()
        return

    cells = plan(full=True)
    log(f"PLAN: {len(cells)} celdas. Tope de tiempo: {args.hours} h. device={DEV}")
    n_done = sum(1 for cid, _, _ in cells if done(cid))
    log(f"Ya completadas: {n_done}/{len(cells)}")
    budget = args.hours * 3600
    ncomp = 0
    for cid, fn, cfg in cells:
        if done(cid):
            continue
        if time.time() - T0 > budget:
            log(f"*** TOPE DE TIEMPO alcanzado. Paro limpio. ***")
            break
        try:
            payload = fn(cfg)
            save_atomic(cid, payload)
            ncomp += 1
            if ncomp % 5 == 0 or True:
                log(f"OK {cid}  ({payload.get('wall_s','?')}s)")
        except Exception as e:
            log(f"ERR {cid}: {repr(e)[:120]}")
            save_atomic(cid, {"_t_start": time.time(), "error": repr(e)[:200], "cfg": cfg})
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
    log(f"FIN. Completadas esta corrida: {ncomp}. Total en disco: "
        f"{sum(1 for cid,_,_ in cells if done(cid))}/{len(cells)}")


def calibrate():
    """Mide tiempos por bloque y CONFIRMA que emergen solitones antes de la noche."""
    log("=== CALIBRACIÓN ===")
    samples = [
        ("B1_ring_N1048576", cell_B1, {"kind": "ring", "N": 1048576, "seed": 0}),
        ("B1_expander_N65536", cell_B1, {"kind": "expander", "N": 65536, "seed": 0}),
        ("B4_N65536_gyro", cell_B4, {"N": 65536, "zeta": 0.1, "beta": 0.1,
                                     "placement": "gyro", "dt": 0.1, "ticks": 3000}),
        ("B2_N65536", cell_B2, {"N": 65536, "beta": 0.05,
                                "modes": [int(x) for x in _linspace(2, 65536//2-2, 36)], "ticks": 4000}),
        ("B5_N1000000", cell_B5, {"N": 1000000, "ticks": 3000}),
    ]
    for cid, fn, cfg in samples:
        t = time.time(); r = fn(cfg); dt = time.time() - t
        log(f"  {cid}: {dt:.1f}s")

    log("--- Sondeo de SOLITONES (¿emergen? ¿qué régimen?) ---")
    N = 4096
    for nonlin in ["genuine", "saturated"]:
        for beta in [0.02, 0.05, 0.1]:
            for sign in [1.0, -1.0]:
                t = time.time()
                r = solitons_run(N, 0.8, 0.02, beta, 1.0, nonlin, "none",
                                 ticks=20000, dt=0.02, sign=sign)
                dt = time.time() - t
                tr = r["track"]
                v = tr.get("velocity_nodes_per_tick") if tr else None
                wcv = tr.get("width_cv") if tr else None
                life = tr.get("life_frac") if tr else None
                log(f"  {nonlin:9s} β={beta} sign={sign:+.0f}: "
                    f"n_sol={r['n_solitons_final']} vel={v} width_cv={wcv} "
                    f"life={life} finite={r['finite']} ({dt:.1f}s)")


if __name__ == "__main__":
    main()
