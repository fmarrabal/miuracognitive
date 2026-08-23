"""
Entrenamiento y evaluación de un controlador en el SMA (Fase 2, pre-registrado).

  train_run(name, seed, ...) -> dict con métricas por protocolo (JSON-able).

Protocolo (PREREG_PHASE2.md): 500 pasos Adam lr 3e-3, batch 32, T=256, clip 1.0.
Eval con EPISODIOS FIJOS COMPARTIDOS entre controladores (semillas 9000+k):
contraste pareado por semilla Y por episodio. Oráculo cacheado por protocolo.
"""
from __future__ import annotations
import dataclasses
import hashlib
import json
import os
import time
import torch

from .env import EnvConfig, sample_latents, rollout, tracking_metrics
from .oracle import oracle_actions, ORACLE_VERSION
from .controllers import build_controller

EVAL_PROTOCOLS = {
    "iid": dict(ood=None, T=256, seed=9001),
    "ood_budget_lo": dict(ood="budget_lo", T=256, seed=9002),
    "ood_budget_hi": dict(ood="budget_hi", T=256, seed=9003),
    "ood_riskfreq": dict(ood="riskfreq", T=256, seed=9004),
    "ood_long": dict(ood=None, T=512, seed=9005),
    "e4_step": dict(ood="step", T=256, seed=9006),
    "e2_risk_only": dict(ood="risk_only", T=256, seed=9007),
}
# Métrica primaria (PREREG v3): media de TRES componentes OOD, con los dos
# presupuestos promediados PRIMERO (peso 1/3 al factor presupuesto, no 1/2)
OOD_COMPONENTS = (("ood_budget_lo", "ood_budget_hi"), ("ood_riskfreq",), ("ood_long",))
EVAL_EPISODES = 300


def _cfg_hash(cfg: EnvConfig, spec: dict) -> str:
    """Hash corto de (EnvConfig, protocolo, episodios, versión del oráculo):
    el caché no puede sobrevivir silenciosamente a un cambio de entorno."""
    payload = json.dumps({"cfg": dataclasses.asdict(cfg), "spec": spec,
                          "n": EVAL_EPISODES, "ov": ORACLE_VERSION}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def env_fingerprint() -> str:
    """Huella del ENTORNO EXPERIMENTAL (EnvConfig + fuentes clave): detecta en
    load() la mezcla de corridas de entornos distintos (hallazgo del panel 2b)."""
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha1()
    h.update(json.dumps(dataclasses.asdict(EnvConfig()), sort_keys=True).encode())
    for fn in ("env.py", "controllers.py"):
        with open(os.path.join(here, fn), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


def _oracle_cached(cache_dir: str, proto: str, lat: dict, device,
                   cfg: EnvConfig, spec: dict) -> torch.Tensor:
    os.makedirs(cache_dir, exist_ok=True)
    fp = os.path.join(cache_dir, f"oracle_{proto}_{_cfg_hash(cfg, spec)}.pt")
    if os.path.exists(fp):
        return torch.load(fp, map_location=device, weights_only=True)
    res = oracle_actions(lat)
    assert res["convergence_residual"] < 0.01, \
        f"oráculo no convergido en {proto}: resid={res['convergence_residual']:.4f}"
    J = res["J"]
    tmp = fp + f".{os.getpid()}.tmp"
    torch.save(J.cpu(), tmp)
    os.replace(tmp, fp)
    return J.to(device)


def _step_settling(lat: dict, out: dict) -> dict:
    """Métricas E4 (escalón de presupuesto en t=T/2): violación tras el escalón,
    ventanas hasta re-ajustarse, overshoot."""
    cfg = lat["cfg"]
    w0 = (lat["T"] // 2) // cfg.W
    spend = out["spend_w"]                      # (B,nW)
    Bw = lat["Bw"]
    post_viol = torch.relu(spend[:, w0:] - Bw[:, w0:])
    ok = spend[:, w0:] <= Bw[:, w0:] * 1.05
    # primera ventana post-escalón ya ajustada (por episodio; nW si nunca)
    first_ok = torch.where(ok.any(1), ok.double().argmax(1).double(),
                           torch.full((spend.shape[0],), float(ok.shape[1]),
                                      device=spend.device))
    return {"post_step_viol_mean": float(post_viol.mean()),
            "post_step_overshoot_max": float(post_viol.max()),
            "settling_windows_mean": float(first_ok.mean())}


@torch.no_grad()
def evaluate(controller, device, cache_dir: str) -> dict:
    controller.eval()
    results = {}
    cfg = EnvConfig()
    for proto, spec in EVAL_PROTOCOLS.items():
        lat = sample_latents(cfg, EVAL_EPISODES, spec["T"], spec["seed"], device,
                             ood=spec["ood"])
        with torch.enable_grad():                        # el oráculo necesita grad
            J_or = _oracle_cached(cache_dir, proto, lat, device, cfg, spec)
        out = rollout(controller, lat, device)
        tm = tracking_metrics(lat, out["A"])
        r = {
            "J_mean": float(out["J"].mean()), "J_std": float(out["J"].std()),
            "err": float(out["err"].mean()), "hard": float(out["hard"].mean()),
            "plan": float(out["plan"].mean()), "hold": float(out["hold"].mean()),
            "cost_term": float(out["cost_term"].mean()),
            "viol_rate": float(out["viol_rate"].mean()),
            "viol_mag_mean": float(out["viol_mag_mean"].mean()),
            "viol_mag_p95": float(out["viol_mag_p95"].mean()),
            "oracle_J": float(J_or.mean()),
            "regret": float((out["J"] - J_or).mean()),
            # J POR EPISODIO (pareado exacto por episodio entre controladores;
            # ~300 floats/protocolo — hallazgo del panel)
            "J_episodes": [round(float(x), 6) for x in out["J"].cpu()],
            **tm,
        }
        if proto == "e4_step":
            r.update(_step_settling(lat, out))
        results[proto] = r
    # primaria de 3 componentes (presupuesto promediado primero)
    comp_J, comp_R = [], []
    for group in OOD_COMPONENTS:
        comp_J.append(sum(results[p]["J_mean"] for p in group) / len(group))
        comp_R.append(sum(results[p]["regret"] for p in group) / len(group))
    results["ood_primary_J"] = sum(comp_J) / len(comp_J)
    results["ood_primary_regret"] = sum(comp_R) / len(comp_R)
    # sensibilidad: la media plana de 4 protocolos (variante no primaria)
    flat = ("ood_budget_lo", "ood_budget_hi", "ood_riskfreq", "ood_long")
    results["ood_flat4_J"] = sum(results[p]["J_mean"] for p in flat) / 4
    controller.train()
    return results


def train_run(name: str, seed: int, device_str: str = "cuda",
              steps: int = 500, batch: int = 32, lr: float = 3e-3,
              out_json: str | None = None, cache_dir: str = "results_phase2/oracle_cache",
              log_every: int = 100) -> dict:
    t0 = time.time()
    assert steps < 9000, "steps>=9000 colisionaría con las semillas de eval (9001+)"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    controller = build_controller(name).to(device)
    opt = torch.optim.Adam(controller.parameters(), lr=lr)
    cfg = EnvConfig()
    losses = []
    for step in range(steps):
        lat = sample_latents(cfg, batch, cfg.T, seed * 100000 + step, device)
        out = rollout(controller, lat, device)
        loss = out["J"].mean() + 1e-3 * controller.regularizer()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        if log_every and step % log_every == 0:
            print(f"  [{name} s{seed}] step {step}: J={losses[-1]:.4f}", flush=True)
    res = {
        "controller": name, "seed": seed, "steps": steps,
        "env_fingerprint": env_fingerprint(),
        "params_total": sum(p.numel() for p in controller.parameters()),
        "params_core": controller.core_param_count(),
        "train_loss_first": sum(losses[:20]) / max(len(losses[:20]), 1),
        "train_loss_last": sum(losses[-20:]) / max(len(losses[-20:]), 1),
        "eval": evaluate(controller, device, cache_dir),
        "wall_s": round(time.time() - t0, 1),
    }
    # intervención causal g≡1 (familia gov): la eval es determinista → el
    # pareado episodio-a-episodio con la eval normal sale gratis (panel 2b:
    # sin esto la intervención pre-registrada era inejecutable)
    if hasattr(controller, "modulation_off") and getattr(controller, "gain_core", None):
        controller.modulation_off(True)
        res["eval_mod_off"] = evaluate(controller, device, cache_dir)
        controller.modulation_off(False)
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        tmp = out_json + f".{os.getpid()}.tmp"        # único por proceso (carrera TTL)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(res, f)
        os.replace(tmp, out_json)
        # checkpoint por celda (familia gov y react): habilita las intervenciones
        # finas post-hoc (g-media por episodio, g-barajada) desde el modelo REAL
        if hasattr(controller, "gain_core"):
            ck = out_json.replace(".json", ".ckpt.pt")
            tmpc = ck + f".{os.getpid()}.tmp"
            torch.save(controller.state_dict(), tmpc)
            os.replace(tmpc, ck)
    return res
