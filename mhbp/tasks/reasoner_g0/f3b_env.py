"""
F3b — Generador del ENTORNO DE SESIÓN (PREREG_F3B.md §2).

Una sesión = E=6 instancias de composición en S_5 (cycle_transp) secuenciales,
con tres escalas reales:
  tick      : iteración del reasoner (la gobierna el modelo, no este módulo)
  instancia : una tarea permcomp con dificultad K
  sesión    : presupuesto duro B_total de ticks + REGÍMENES de dificultad
              (distribuciones de K con solape) + STAKES observables ex ante.

Decisiones del prereg implementadas aquí:
  - Regímenes como DISTRIBUCIONES con solape (R0 fácil K∈[8,16] uniforme,
    R1 difícil K∈[12,22] uniforme — provisionales, diales de calibración):
    una instancia NO identifica el régimen; filtrar la historia sí (GS2).
  - Permanencia ESTOCÁSTICA: prob. de cambio p_switch=1/3 por instancia
    (geométrica de media 3) y fase/régimen inicial aleatorio 50/50:
    MI(régimen; posición) ≈ 0 por construcción (verificado en GS3).
  - Stakes: EXACTAMENTE n_high=2 de E=6 instancias a ×4 (resto ×1),
    posiciones permutadas, con un stream RNG INDEPENDIENTE del de regímenes
    (independencia por construcción, verificada por chi² en el self-test).
    El stake es observable EX ANTE vía token reservado (ver stake_token).
  - B_total = CONSTANTE del spec, NO por sesión: un B dependiente de los K
    de la sesión filtraría la dificultad total al modelo (fuga).
  - RNG del entorno PROPIO e independiente del modelo: numpy PCG64 con
    seed_env = seed*1000+7 (train) / +800000 (eval); sesión indexada
    determinista vía SeedSequence(seed_env, spawn_key=(idx,)) — acceso
    aleatorio a la sesión idx sin generar las anteriores.

Las instancias se generan REUTILIZANDO el builder de G0 (build_dataset con
task="permcomp", gen_set="cycle_transp": training/trainer.py →
data/synthetic_recall.py::PermutationCompositionDataset). Mismo vocabulario,
mismo formato de labels (alineadas a la posición de entrada, SIN shift
next-token), mismo ARROW/PAD.

VOCABULARIO EXTENDIDO (tokens de stake, ids RESERVADOS NUEVOS):
  g0 cycle_transp: PAD=0, Q=1, ARROW=2, generadores {3,4}, perms [5,125)
                   → vocab_size = 125.
  STAKE_LOW_ID  = 125  (stake ×1)
  STAKE_HIGH_ID = 126  (stake ×4)
  → F3B_VOCAB_SIZE = 127. Los ckpts F3a (vocab 125) NO ven estos tokens:
  f3b_probe_acc.py usa instancias SIN stake; los brazos F3b se construyen
  con el vocab extendido.

Self-test:  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.f3b_env
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field

import numpy as np
import torch

from training.config import TrainConfig
from training.trainer import build_dataset


# --------------------------------------------------------------------------- #
#  Especificación del entorno (diales de calibración de GS1/GS2 — §10:
#  cualquier cambio es una ronda de calibración commiteada, ≤3)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SessionSpec:
    """RONDA 1 de calibración (2026-08-02, cortafuegos §10; motivo: GS1 FALLÓ
    con los diales provisionales — mordida 38.8%<60%, headroom +0.021<+0.08 —
    y acc(n=1|K)≤0.2 exige K_min≥13 según f3b_acc_profile.json):
      k_easy (8,16)→(13,18) · k_hard (12,22)→(16,22) · B_total 60→56
      (= 0.75 × demanda estacionaria 74.1 con d_ref medido)."""
    E: int = 6                      # instancias por sesión
    k_easy: tuple = (13, 18)        # R0: K ~ U[13..18]  (solape en [16..18])
    k_hard: tuple = (16, 22)        # R1: K ~ U[16..22]
    p_switch: float = 1.0 / 3.0     # prob. de cambio de régimen por instancia
                                    # → permanencia geométrica de media 3
    n_high: int = 2                 # instancias con stake alto por sesión
    stake_high: int = 4
    stake_low: int = 1
    B_total: int = 56               # presupuesto duro de ticks por sesión
                                    # (0.75 × demanda d_ref de la mezcla 50/50)
    # --- ENMIENDA A (2026-08-02, FINDINGS_F3B_GATES §fork, ratificada):
    # stake_mode="regime_corr" acopla el stake al RÉGIMEN por instancia
    # (Bernoulli: P(×alto|difícil)=q_hard > P(×alto|fácil)=q_easy). Crea
    # IRREVERSIBILIDAD económica: la tormenta (régimen difícil) trae stakes
    # altos → anticiparla obliga a ahorrar YA. La composición deja de ser
    # fija (la normalización por máximo de sesión absorbe la varianza).
    # MI(régimen; stake) pasa a ser CANAL DECLARADO, no fuga (GS3 re-scope).
    # "fixed" reproduce el diseño v1/ronda-1 (escaneo del techo, git 304a1a9).
    stake_mode: str = "fixed"
    q_easy: float = 1.0 / 6.0
    q_hard: float = 1.0 / 2.0
    seq_len: int = 128              # el de G0 (paridad con los ckpts F3a)
    n_max: int = 24                 # techo de ticks por instancia (N_max G0)


# --------------------------------------------------------------------------- #
#  Vocabulario extendido: tokens de stake reservados
# --------------------------------------------------------------------------- #
def _g0_dataset(spec: SessionSpec):
    """El builder de G0, importado (no duplicado): mismo vocab y formato."""
    return build_dataset(TrainConfig(task="permcomp", perm_gens="cycle_transp",
                                     min_writes=2, max_writes=spec.n_max,
                                     seq_len=spec.seq_len, seed=0))


_BASE_VOCAB = None            # vocab_size del dataset g0 (125), cacheado


def _base_vocab_size(spec: SessionSpec | None = None) -> int:
    global _BASE_VOCAB
    if _BASE_VOCAB is None:
        _BASE_VOCAB = _g0_dataset(spec or SessionSpec()).vocab_size
    return _BASE_VOCAB


def stake_token(stake: int, spec: SessionSpec | None = None) -> int:
    """Id de vocabulario RESERVADO para el token de stake (observable ex ante).

    Ids NUEVOS a continuación del vocab de g0 (125 para cycle_transp):
      stake ×1 (low)  → 125   (STAKE_LOW_ID)
      stake ×4 (high) → 126   (STAKE_HIGH_ID)
    """
    spec = spec or SessionSpec()
    base = _base_vocab_size(spec)
    if stake == spec.stake_low:
        return base            # 125
    if stake == spec.stake_high:
        return base + 1        # 126
    raise ValueError(f"stake desconocido: {stake}")


def f3b_vocab_size(spec: SessionSpec | None = None) -> int:
    """vocab de g0 + 2 tokens de stake = 127 (cycle_transp)."""
    return _base_vocab_size(spec) + 2


# --------------------------------------------------------------------------- #
#  Streams RNG del entorno (independientes del modelo)
# --------------------------------------------------------------------------- #
def seed_env_train(seed: int) -> int:
    return seed * 1000 + 7


def seed_env_eval(seed: int) -> int:
    return seed * 1000 + 7 + 800_000


def _session_rngs(seed_env: int, idx: int):
    """Tres generadores hijos INDEPENDIENTES por sesión (spawn determinista):
      rng_reg   : cadena de regímenes + K por instancia
      rng_stake : permutación de posiciones de stake (⊥ regímenes por
                  construcción: stream separado)
      rng_tok   : semillas del generador torch de tokens (contenido de la
                  instancia; separado para que el modo 'light' no altere nada)
    """
    ss = np.random.SeedSequence(seed_env, spawn_key=(idx,))
    c_reg, c_stake, c_tok = ss.spawn(3)
    mk = lambda c: np.random.Generator(np.random.PCG64(c))
    return mk(c_reg), mk(c_stake), mk(c_tok)


# --------------------------------------------------------------------------- #
#  Generación de sesiones
# --------------------------------------------------------------------------- #
_DS_CACHE = {}                 # dataset g0 cacheado por seq_len (se re-siembra
                               # y re-acota por instancia; un solo objeto)


def _instance_tokens(spec: SessionSpec, K: int, torch_seed: int):
    """Una instancia cycle_transp de dificultad EXACTA K con el builder de G0.

    Se re-siembra el generador torch del dataset con una semilla extraída del
    stream numpy del ENTORNO (rng_tok) y se fija min_ops=max_ops=K: sample()
    reproduce el formato g0 al bit (tokens, labels densas alineadas a la
    posición, ARROW en K+1, PAD=0)."""
    key = spec.seq_len
    if key not in _DS_CACHE:
        _DS_CACHE[key] = _g0_dataset(spec)
    ds = _DS_CACHE[key]
    ds.min_ops = ds.max_ops = int(K)
    ds.gen.manual_seed(int(torch_seed))
    tokens, labels, k_out = ds.sample()
    assert k_out == K
    assert int(tokens[K + 1]) == ds.ARROW, "ARROW fuera de sitio"
    return tokens, labels


def gen_session(spec: SessionSpec, seed_env: int, idx: int,
                materialize: bool = True) -> dict:
    """Genera la sesión `idx` del stream `seed_env` (determinista, indexada).

    Devuelve un dict con:
      instances             lista de E dicts: tokens (seq_len,), labels
                            (seq_len,), K, regimen, stake, stake_tok,
                            arrow_pos (=K+1, en las coordenadas SIN stake)
      regimen_por_instancia [E] (0=fácil, 1=difícil)
      stake_por_instancia   [E]
      B_total               constante del spec
    Con materialize=False omite tokens/labels (modo ligero para Monte Carlo
    de gates); el resto del stream RNG es IDÉNTICO en ambos modos.
    """
    rng_reg, rng_stake, rng_tok = _session_rngs(seed_env, idx)

    # --- cadena de regímenes: inicial 50/50, cambio con prob p_switch ---
    regs, ks = [], []
    r = int(rng_reg.integers(0, 2))
    for i in range(spec.E):
        if i > 0 and rng_reg.random() < spec.p_switch:
            r = 1 - r
        lo, hi = spec.k_easy if r == 0 else spec.k_hard
        ks.append(int(rng_reg.integers(lo, hi + 1)))
        regs.append(r)

    # --- stakes ---
    if spec.stake_mode == "regime_corr":
        # ENMIENDA A: Bernoulli por instancia condicionada al régimen
        # (q_hard si difícil, q_easy si fácil) — el stake es evidencia
        # PARCIAL y legítima del régimen, y el futuro caro exige ahorro.
        stakes = [spec.stake_high
                  if rng_stake.random() < (spec.q_hard if regs[i] == 1
                                           else spec.q_easy)
                  else spec.stake_low for i in range(spec.E)]
    else:
        # v1: exactamente n_high posiciones a ×4, stream INDEPENDIENTE
        perm = rng_stake.permutation(spec.E)
        stakes = [spec.stake_low] * spec.E
        for p in perm[: spec.n_high]:
            stakes[int(p)] = spec.stake_high

    # --- contenido de las instancias (builder g0, semillas del stream env) ---
    instances = []
    for i in range(spec.E):
        torch_seed = int(rng_tok.integers(0, 2 ** 62))   # se extrae SIEMPRE
        inst = {"K": ks[i], "regimen": regs[i], "stake": stakes[i],
                "stake_tok": stake_token(stakes[i], spec),
                "arrow_pos": ks[i] + 1}
        if materialize:
            tokens, labels = _instance_tokens(spec, ks[i], torch_seed)
            inst["tokens"] = tokens
            inst["labels"] = labels
        instances.append(inst)

    return {"idx": idx, "B_total": spec.B_total, "instances": instances,
            "regimen_por_instancia": regs, "stake_por_instancia": stakes}


def instance_tensors(inst: dict, prepend_stake: bool = True):
    """(inputs, targets, answer_pos) en el formato de entrenamiento de G0
    (inputs = tokens[:-1]; labels alineadas a la posición, SIN shift).

    Con prepend_stake=True antepone el token de stake (observable ex ante,
    §2): la secuencia se desplaza +1 y se recorta el último PAD; answer_pos
    pasa a K+2."""
    tokens, labels = inst["tokens"], inst["labels"]
    if prepend_stake:
        assert int(tokens[-1]) == 0, "el último token no es PAD: no se puede desplazar"
        tokens = torch.cat([torch.tensor([inst["stake_tok"]], dtype=tokens.dtype),
                            tokens[:-1]])
        labels = torch.cat([torch.tensor([-1], dtype=labels.dtype), labels[:-1]])
        ans = inst["arrow_pos"] + 1
    else:
        ans = inst["arrow_pos"]
    return tokens[:-1].contiguous(), labels[:-1].contiguous(), ans


# --------------------------------------------------------------------------- #
#  Hash de paridad entre brazos (§2: assert antes de lanzar la parrilla)
# --------------------------------------------------------------------------- #
def session_stream_hash(spec: SessionSpec, seed_env: int, n: int = 64) -> str:
    """sha256 hex de las primeras n sesiones serializadas (tokens incluidos).
    Los 7 brazos deben ver EXACTAMENTE este stream: el assert de paridad
    compara este hash entre procesos antes de entrenar."""
    h = hashlib.sha256()
    h.update(json.dumps(asdict(spec), sort_keys=True).encode())
    for idx in range(n):
        s = gen_session(spec, seed_env, idx, materialize=True)
        ser = {"idx": s["idx"], "B_total": s["B_total"],
               "reg": s["regimen_por_instancia"],
               "stake": s["stake_por_instancia"],
               "inst": [{"K": i["K"], "stake_tok": i["stake_tok"],
                         "tokens": i["tokens"].tolist(),
                         "labels": i["labels"].tolist()}
                        for i in s["instances"]]}
        h.update(json.dumps(ser, sort_keys=True).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #
def _selftest():
    spec = SessionSpec()
    print(f"SessionSpec: {spec}")
    print(f"vocab g0={_base_vocab_size(spec)}  f3b={f3b_vocab_size(spec)}  "
          f"stake_tok(x1)={stake_token(1)}  stake_tok(x4)={stake_token(4)}")
    ok = True

    # 1) determinismo: misma (seed_env, idx) → misma sesión (bit a bit)
    se = seed_env_train(0)
    a = gen_session(spec, se, 3)
    b = gen_session(spec, se, 3)
    same = (a["regimen_por_instancia"] == b["regimen_por_instancia"]
            and a["stake_por_instancia"] == b["stake_por_instancia"]
            and all(torch.equal(x["tokens"], y["tokens"])
                    and torch.equal(x["labels"], y["labels"])
                    for x, y in zip(a["instances"], b["instances"])))
    print(f"[1] determinismo misma seed+idx: {'OK' if same else 'FAIL'}")
    ok &= same

    # 1b) el modo light reproduce el MISMO stream de regímenes/stakes/K
    c = gen_session(spec, se, 3, materialize=False)
    same_l = (a["regimen_por_instancia"] == c["regimen_por_instancia"]
              and a["stake_por_instancia"] == c["stake_por_instancia"]
              and [i["K"] for i in a["instances"]] == [i["K"] for i in c["instances"]])
    print(f"[1b] paridad materialize=False: {'OK' if same_l else 'FAIL'}")
    ok &= same_l

    # 1c) sesiones distintas difieren; streams train/eval difieren
    d = gen_session(spec, se, 4, materialize=False)
    e = gen_session(spec, seed_env_eval(0), 3, materialize=False)
    dif = ([i["K"] for i in d["instances"]] != [i["K"] for i in a["instances"]]
           or d["regimen_por_instancia"] != a["regimen_por_instancia"])
    dif &= ([i["K"] for i in e["instances"]] != [i["K"] for i in a["instances"]]
            or e["regimen_por_instancia"] != a["regimen_por_instancia"])
    print(f"[1c] idx/stream distintos → sesiones distintas: {'OK' if dif else 'FAIL'}")
    ok &= dif

    # --- estadísticos sobre N sesiones (modo light) ---
    N = 2000
    sess = [gen_session(spec, se, i, materialize=False) for i in range(N)]
    regs = np.array([s["regimen_por_instancia"] for s in sess])     # (N,E)
    stks = np.array([s["stake_por_instancia"] for s in sess])       # (N,E)
    ks = np.array([[i["K"] for i in s["instances"]] for s in sess])  # (N,E)

    # 2) independencia stakes ⊥ régimen: chi² de la tabla 2×2 (por instancia)
    hi = (stks == spec.stake_high).ravel().astype(int)
    rg = regs.ravel()
    tab = np.zeros((2, 2))
    for x, y in zip(rg, hi):
        tab[x, y] += 1
    exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    chi2 = float(((tab - exp) ** 2 / exp).sum())          # df=1: p>0.05 ⇔ chi2<3.84
    corr = float(np.corrcoef(rg, hi)[0, 1])
    ind_ok = chi2 < 3.84
    print(f"[2] stakes⊥régimen: chi2={chi2:.2f} (umbral 3.84) corr={corr:+.4f} "
          f"{'OK' if ind_ok else 'FAIL'}")
    ok &= ind_ok

    # 3) permanencia: tasa de cambio ≈ p_switch → media geométrica 1/p ≈ 3
    sw = (regs[:, 1:] != regs[:, :-1]).mean()
    perm_mean = 1.0 / sw
    perm_ok = abs(sw - spec.p_switch) < 0.02
    print(f"[3] permanencia: p_switch_emp={sw:.3f} (spec {spec.p_switch:.3f}) "
          f"→ media geométrica {perm_mean:.2f} (esperado 3) {'OK' if perm_ok else 'FAIL'}")
    ok &= perm_ok

    # 3b) fase inicial 50/50 y marginal por posición ~0.5 (pre-check de GS3)
    m0 = regs[:, 0].mean()
    mpos = regs.mean(0)
    f_ok = abs(m0 - 0.5) < 0.03 and np.all(np.abs(mpos - 0.5) < 0.04)
    print(f"[3b] régimen inicial={m0:.3f}; marginal por posición="
          f"{np.round(mpos, 3).tolist()} {'OK' if f_ok else 'FAIL'}")
    ok &= f_ok

    # 4) hash estable (dos cómputos idénticos) — es el assert de paridad §2
    h1 = session_stream_hash(spec, se, n=16)
    h2 = session_stream_hash(spec, se, n=16)
    print(f"[4] hash estable: {h1[:16]}... {'OK' if h1 == h2 else 'FAIL'}")
    ok &= (h1 == h2)

    # 5) fracción de sesiones sin transición: (1-p)^(E-1) = (2/3)^5 ≈ 13.2%
    no_tr = float((regs[:, 1:] == regs[:, :-1]).all(1).mean())
    esp = (1 - spec.p_switch) ** (spec.E - 1)
    nt_ok = abs(no_tr - esp) < 0.025
    print(f"[5] sesiones sin transición: {no_tr:.3f} (esperado {esp:.3f}) "
          f"{'OK' if nt_ok else 'FAIL'}")
    ok &= nt_ok

    # 6) composición de stakes y rangos de K por régimen
    comp_ok = bool(np.all((stks == spec.stake_high).sum(1) == spec.n_high))
    k_ok = bool(np.all((ks[regs == 0] >= spec.k_easy[0]) & (ks[regs == 0] <= spec.k_easy[1]))
                and np.all((ks[regs == 1] >= spec.k_hard[0]) & (ks[regs == 1] <= spec.k_hard[1])))
    print(f"[6] 2/6 stakes ×4 por sesión: {'OK' if comp_ok else 'FAIL'}; "
          f"K dentro de rango por régimen: {'OK' if k_ok else 'FAIL'}")
    ok &= comp_ok and k_ok

    # 7) formato de instancia: paridad con g0 + prepend de stake
    inst = a["instances"][0]
    inp, tgt, ans = instance_tensors(inst, prepend_stake=True)
    ds = _DS_CACHE[spec.seq_len]
    fmt_ok = (int(inp[0]) == inst["stake_tok"]
              and int(inp[ans]) == ds.ARROW
              and ds.answer_offset <= int(tgt[ans]) < ds.answer_offset + ds.answer_size
              and int(tgt[0]) == -1)
    inp0, tgt0, ans0 = instance_tensors(inst, prepend_stake=False)
    fmt_ok &= (int(inp0[ans0]) == ds.ARROW and torch.equal(tgt0[ans0], tgt[ans]))
    print(f"[7] formato g0 + stake prepend (ARROW/label en answer_pos): "
          f"{'OK' if fmt_ok else 'FAIL'}")
    ok &= fmt_ok

    print(f"\nhash de las primeras 64 sesiones (train seed 0): "
          f"{session_stream_hash(spec, se, n=64)}")
    print(f"\nSELF-TEST {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
