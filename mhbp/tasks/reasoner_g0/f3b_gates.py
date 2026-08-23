"""
F3b — Escalera de políticas NO aprendidas y kill-gates GS1/GS2/GS3
(PREREG_F3B.md §10, con la contabilidad de presupuesto de §6.1).

Simulación Monte Carlo sobre el generador de sesiones (f3b_env) usando el
perfil de competencia acc(n,K) medido con los ckpts F3a (f3b_probe_acc):
el score esperado de una política de asignación de ticks es
    score = Σ_i stake_i · acc(n_eff_i, K_i) / Σ_i stake_i     (normalizado)
con la contabilidad secuencial del presupuesto: n_eff = min(n_req, restante);
al agotarse, n=1 forzado (idéntica a §6.1).

Políticas (ninguna aprendida; §10 GS1). Las cuatro informadas comparten EL
MISMO asignador greedy por valor marginal (stake·Δacc/Δn, con re-planificación
secuencial en las MPC): difieren SOLO en la información con la que construyen
el valor esperado — el contraste GS2 queda puramente informacional (ver la
nota de DESVIACIÓN junto a _MPC):
  uniforme        B/E por instancia (ciega a todo)
  taper           familia posicional decreciente (rejilla; se reporta la mejor)
  stake_greedy    prioridad ×4 = demanda estacionaria; uniforme dentro de
                  nivel (ciega a K)
  contador        MPC con predictiva ESTACIONARIA en toda posición (null
                  posicional, ciega a K y a la historia)
  ultima_K        MPC con masa puntual en K_{i-1} (persistencia-de-1)
  bayes           MPC con la predictiva del filtro bayesiano EXACTO (modelo
                  generativo verdadero) sobre la historia de K
  oraculo         el mismo asignador con el K verdadero (y stakes) de TODA
                  la sesión (techo perceptivo)

Gates:
  GS1a  mordida: fracción de sesiones con ≥1 forzado bajo consumo por
        DEMANDA (n_req=d_ref(K_i), el proxy de un modelo sin gobierno) ≥60%.
        Nota: bajo la política uniforme estricta (B/E exacto) nunca hay
        forzados por construcción (Σ=B); la mordida del prereg se
        operacionaliza con la demanda d_ref y se reporta además la
        insuficiencia de la asignación uniforme (d_ref(K_i) > B/E).
  GS1b  headroom perceptivo = oraculo − mejor{uniforme,taper,stake_greedy,
        contador}: IC-inferior bootstrap ≥ +0.08. Y oraculo ≥ stake_greedy
        + 0.08.
  GS1c  headroom ≥ 2× MDD implicada por la varianza medida (prereg §10c).
  GS2   bayes > ultima_K (IC-inf ≥ +0.05·rango, rango = oraculo−uniforme)
        Y bayes > contador. Si falla: diagnóstico de qué dial mover.
  GS3   proxies: MI(régimen;posición) y MI(régimen;stake) < 0.05 bits;
        corr(régimen;K propio) se reporta APARTE (es la señal legítima).

Además: varianza por-sesión del score con aciertos Bernoulli (no esperados)
→ N_eval tal que SE_intra ≤ ⅓·sd entre-seeds (proxy 0.03) (§8).

Sin perfil real:  --synthetic  usa un perfil monótono razonable, documentado
como NO válido para el veredicto (solo valida el harness).

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.f3b_gates [--synthetic]
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from .f3b_env import SessionSpec, gen_session, seed_env_train

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PROFILE_PATH = os.path.join(RES, "f3b_acc_profile.json")
OUT_PATH = os.path.join(RES, "f3b_gates.json")

K_MIN, K_MAX = 6, 24          # rejilla del perfil
N_MIN, N_MAX = 1, 24


# --------------------------------------------------------------------------- #
#  Perfil acc(n,K)
# --------------------------------------------------------------------------- #
def load_profile(synthetic: bool):
    """Devuelve (acc, d_ref, origen): acc (24,19) [n-1, K-6]; d_ref dict K→n."""
    if synthetic:
        acc = _synthetic_profile()
        origen = "SINTETICO (NO valido para el veredicto real; solo harness)"
    else:
        if not os.path.exists(PROFILE_PATH):
            raise SystemExit(f"No existe {PROFILE_PATH}. Corre antes "
                             "f3b_probe_acc o usa --synthetic.")
        with open(PROFILE_PATH, encoding="utf-8") as f:
            prof = json.load(f)
        acc = np.zeros((N_MAX, K_MAX - K_MIN + 1))
        for K in range(K_MIN, K_MAX + 1):
            acc[:, K - K_MIN] = np.asarray(prof["acc"][str(K)])
        origen = f"real ({', '.join(prof.get('checkpoints_usados', ['?'])[:2])}... " \
                 f"n_muestras={prof.get('n_muestras')})"
    # d_ref(K): mínimo n con acc(n,K) >= 0.9·acc(24,K)  (ticks-hasta-competencia)
    d_ref = {}
    for K in range(K_MIN, K_MAX + 1):
        col = acc[:, K - K_MIN]
        thr = 0.9 * col[-1]
        d_ref[K] = int(np.argmax(col >= thr)) + 1
    return acc, d_ref, origen


def _synthetic_profile():
    """Perfil monótono plausible: rampa hasta una meseta en n≈0.75·K.
    SOLO para validar el harness; jamás para el veredicto real."""
    acc = np.zeros((N_MAX, K_MAX - K_MIN + 1))
    for K in range(K_MIN, K_MAX + 1):
        plateau = 0.97 - 0.012 * (K - K_MIN)
        d = math.ceil(0.75 * K)
        for n in range(N_MIN, N_MAX + 1):
            s = min(1.0, n / d) ** 1.5
            acc[n - 1, K - K_MIN] = 0.05 + (plateau - 0.05) * s
    return acc


# --------------------------------------------------------------------------- #
#  Contabilidad del presupuesto (§6.1)
# --------------------------------------------------------------------------- #
def ejecutar(n_req, B):
    """Consumo secuencial: n_eff=min(n_req, restante); restante=0 → n=1
    forzado (la instancia corre igualmente con 1 tick, semántica λ:=1).
    Devuelve (n_eff list, forzado list)."""
    rem = int(B)
    n_eff, forz = [], []
    for nr in n_req:
        nr = max(N_MIN, min(N_MAX, int(round(nr))))
        if rem <= 0:
            n_eff.append(1)
            forz.append(True)
        elif rem < nr:
            n_eff.append(rem)          # corte a mitad de instancia
            forz.append(True)
            rem = 0
        else:
            n_eff.append(nr)
            forz.append(False)
            rem -= nr
    return n_eff, forz


# --------------------------------------------------------------------------- #
#  Políticas (devuelven n_req por instancia; la contabilidad va aparte)
# --------------------------------------------------------------------------- #
def _mixture_pmf(spec):
    """pmf estacionaria de K (regímenes 50/50 — cadena simétrica)."""
    pmf = {}
    for (lo, hi) in (spec.k_easy, spec.k_hard):
        for K in range(lo, hi + 1):
            pmf[K] = pmf.get(K, 0.0) + 0.5 / (hi - lo + 1)
    return pmf


def _stationary_demand(spec, d_ref):
    return sum(p * d_ref[K] for K, p in _mixture_pmf(spec).items())


def pol_uniforme(spec, sess, d_ref, acc):
    base = spec.B_total // spec.E
    extra = spec.B_total - base * spec.E
    return [base + (1 if i < extra else 0) for i in range(spec.E)]


def make_taper(gamma=None, decay=None):
    """Schedule posicional decreciente: lineal w_i∝(1−γ·i/(E−1)) o
    exponencial w_i∝decay^i, normalizado a B (enteros, restos mayores)."""
    def pol(spec, sess, d_ref, acc):
        E, B = spec.E, spec.B_total
        if gamma is not None:
            w = np.array([1.0 - gamma * i / (E - 1) for i in range(E)])
        else:
            w = np.array([decay ** i for i in range(E)])
        w = w / w.sum() * B
        n = np.maximum(1, np.floor(w).astype(int))
        for _ in range(B - int(n.sum())):
            i = int(np.argmax(w - n))
            n[i] += 1
        return np.minimum(n, N_MAX).tolist()
    return pol


def pol_stake_greedy(spec, sess, d_ref, acc):
    """Uniforme DENTRO de cada nivel de stake, con prioridad ×4: las
    instancias ×4 reciben su demanda estacionaria E[d_ref] (ciega a K); el
    presupuesto restante se reparte uniforme entre las ×1. Composición fija
    (2/6) y stake ex ante observables → implementable sin mirar K."""
    est = _stationary_demand(spec, d_ref)
    n_hi = max(N_MIN, min(N_MAX, int(round(est))))
    stakes = sess["stake_por_instancia"]
    # composición de la SESIÓN (en stake_mode="regime_corr" no es fija: la
    # política ve todas las posiciones ×alto — clarividente en stakes, ciega
    # a K: cota SUPERIOR de la familia ciega, conservadora para GS1b)
    n_high_s = sum(1 for s in stakes if s == spec.stake_high)
    n_low_total = spec.B_total - n_hi * n_high_s
    n_lo = max(N_MIN, min(N_MAX, int(round(
        n_low_total / max(1, spec.E - n_high_s)))))
    return [n_hi if s == spec.stake_high else n_lo for s in stakes]


# --- asignador COMÚN por valor marginal (el del oráculo, §10 GS1) ---------- #
def _segmentos_concavos(v):
    """Envolvente cóncava superior de v(n), n=1..24, como segmentos
    (rate, steps) con rate DECRECIENTE desde n=1. El greedy por valor
    marginal con lookahead equivale a consumir estos segmentos en orden;
    los saltos aterrizan en puntos donde la envolvente TOCA la v real."""
    hull = [(1, float(v[0]))]
    for n in range(2, N_MAX + 1):
        p = (n, float(v[n - 1]))
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (p[1] - y1) * (x2 - x1) >= (y2 - y1) * (p[0] - x1):
                hull.pop()                       # hull[-1] queda bajo la cuerda
            else:
                break
        hull.append(p)
    segs = []
    for (x1, y1), (x2, y2) in zip(hull, hull[1:]):
        rate = (y2 - y1) / (x2 - x1)
        if rate <= 0:
            break                                # rates decrecientes: fin útil
        segs.append((rate, x2 - x1))
    return segs


def _greedy_alloc(v_list, budget):
    """Asignación greedy por valor marginal: n_i≥1, Σn_i ≤ budget, consumiendo
    los segmentos de mayor rate primero (stake·Δacc/Δn). Es EL MISMO
    asignador para oráculo y políticas MPC: solo cambia la información con
    la que se construye v_i (K verdadero / predictiva / estacionaria)."""
    E = len(v_list)
    n = [1] * E
    rem = budget - E
    if rem <= 0:
        return n                                 # §6.1 corta si ni el mínimo cabe
    segs = []
    for i, v in enumerate(v_list):
        for rate, steps in _segmentos_concavos(v):
            segs.append((rate, i, steps))
    segs.sort(key=lambda s: -s[0])               # estable → orden intra-instancia
    for rate, i, steps in segs:
        if rem <= 0:
            break
        t = min(steps, min(rem, N_MAX - n[i]))
        n[i] += t
        rem -= t
    return n


def pol_oraculo(spec, sess, d_ref, acc):
    """Greedy por valor marginal stake·Δacc/Δn con el K verdadero (y los
    stakes) de TODA la sesión: techo perceptivo de GS1."""
    ks = [i["K"] for i in sess["instances"]]
    stakes = sess["stake_por_instancia"]
    v_list = [stakes[i] * acc[:, ks[i] - K_MIN] for i in range(spec.E)]
    return _greedy_alloc(v_list, spec.B_total)


# --- políticas informadas: MPC con el asignador del oráculo ---------------- #
#  DESVIACIÓN DOCUMENTADA respecto a la fórmula "n_i = demanda estimada":
#  una asignación proporcional-a-demanda + redondeo entero ABSORBE toda la
#  señal (ΔE[d_ref] entre regímenes ≈3 ticks → siempre round(B/E); verificado
#  en la validación sintética: bayes ≡ contador ≡ uniforme). Para que GS2
#  contraste INFORMACIÓN y no aritmética, las tres políticas usan el MISMO
#  asignador greedy del oráculo con re-planificación secuencial (MPC),
#  cambiando SOLO la distribución predictiva de K:
#    contador : estacionaria en todas las posiciones (null posicional)
#    ultima_K : masa puntual en K_{i-1} (persistencia-de-1); futuras estac.
#    bayes    : predictiva del filtro exacto (historia completa); futuras
#               propagadas con T sin evidencia
#  Siguen siendo políticas NO aprendidas con forma cerrada.
class _MPC:
    def __init__(self, kind):
        self.kind = kind                          # "contador"|"ultima_K"|"bayes"
        self._cache_key = None

    def _prep(self, spec, acc):
        p = spec.p_switch
        self.T = np.array([[1 - p, p], [p, 1 - p]])
        self.rangos = (spec.k_easy, spec.k_hard)
        self.em = []
        # acc esperada por régimen: media de acc(:,K) sobre el rango (24,)
        self.accE = []
        for (lo, hi) in self.rangos:
            e = np.zeros(K_MAX + 1)
            e[lo:hi + 1] = 1.0 / (hi - lo + 1)
            self.em.append(e)
            self.accE.append(acc[:, lo - K_MIN:hi - K_MIN + 1].mean(axis=1))
        self.acc_stat = 0.5 * self.accE[0] + 0.5 * self.accE[1]

    def _v_pred(self, pi):
        return pi[0] * self.accE[0] + pi[1] * self.accE[1]

    def _exp_stake(self, spec, pj):
        """Stake esperado bajo la creencia de régimen pj (ENMIENDA A:
        stake_mode='regime_corr' — el stake futuro depende del régimen)."""
        qs = (spec.q_easy, spec.q_hard)
        return sum(pj[r] * (qs[r] * spec.stake_high
                            + (1 - qs[r]) * spec.stake_low) for r in (0, 1))

    def __call__(self, spec, sess, d_ref, acc):
        key = (id(acc), spec.E, spec.stake_mode)
        if key != self._cache_key:
            self._prep(spec, acc)
            self._cache_key = key
        E = spec.E
        corr_mode = spec.stake_mode == "regime_corr"
        ks = [i["K"] for i in sess["instances"]]
        stakes = sess["stake_por_instancia"]
        pi = np.array([0.5, 0.5])                 # predictiva de la instancia 0
        stat = np.array([0.5, 0.5])               # estacionaria (nulls)
        rem = spec.B_total
        out = []
        for i in range(E):
            cnt_left = E - 1 - i
            # stake futuro esperado según la INFORMACIÓN de la política
            if corr_mode:
                exp_stake_stat = self._exp_stake(spec, stat)
            else:
                # v1: composición fija conocida; posiciones no
                n_hi_left = spec.n_high - sum(1 for s in stakes[:i + 1]
                                              if s == spec.stake_high)
                exp_stake_stat = ((n_hi_left * spec.stake_high
                                   + (cnt_left - n_hi_left) * spec.stake_low)
                                  / cnt_left if cnt_left > 0 else 0.0)
            # v de la instancia ACTUAL según la información de la política
            if self.kind == "contador":
                v_cur = self.acc_stat
            elif self.kind == "ultima_K":
                v_cur = (self.acc_stat if i == 0
                         else acc[:, ks[i - 1] - K_MIN])
            else:                                 # bayes
                v_cur = self._v_pred(pi)
            v_list = [stakes[i] * v_cur]
            # v de las futuras: propagación (bayes) o estacionaria
            pj = pi.copy()
            for _ in range(cnt_left):
                if self.kind == "bayes":
                    pj = self.T @ pj
                    es = (self._exp_stake(spec, pj) if corr_mode
                          else exp_stake_stat)
                    v_list.append(es * self._v_pred(pj))
                else:
                    v_list.append(exp_stake_stat * self.acc_stat)
            alloc = _greedy_alloc(v_list, rem)
            n_i = alloc[0]
            out.append(n_i)
            rem = max(0, rem - n_i)
            # actualización del filtro: K experimentado + (enmienda A) stake
            # OBSERVADO de la instancia i (evidencia parcial del régimen)
            lik = np.array([self.em[0][ks[i]], self.em[1][ks[i]]])
            if corr_mode:
                qs = (spec.q_easy, spec.q_hard)
                hi = stakes[i] == spec.stake_high
                lik = lik * np.array([qs[0] if hi else 1 - qs[0],
                                      qs[1] if hi else 1 - qs[1]])
            post = pi * lik
            post = post / post.sum() if post.sum() > 0 else np.array([0.5, 0.5])
            pi = self.T @ post
        return out


pol_contador = _MPC("contador")
pol_ultima_K = _MPC("ultima_K")
pol_bayes = _MPC("bayes")


def pol_demanda(spec, sess, d_ref, acc):
    """Proxy del modelo SIN gobierno: cada instancia pide su demanda d_ref
    (K verdadero). Solo se usa para medir la MORDIDA (GS1a), no compite."""
    return [d_ref[i["K"]] for i in sess["instances"]]


POLITICAS = {
    "uniforme": pol_uniforme,
    "stake_greedy": pol_stake_greedy,
    "contador": pol_contador,
    "ultima_K": pol_ultima_K,
    "bayes": pol_bayes,
    "oraculo": pol_oraculo,
}
TAPER_GRID = ([("taper_lin_%.1f" % g, make_taper(gamma=g)) for g in (0.2, 0.4, 0.6, 0.8, 1.0)]
              + [("taper_exp_%.1f" % d, make_taper(decay=d)) for d in (0.6, 0.7, 0.8, 0.9)])


# --------------------------------------------------------------------------- #
#  Evaluación
# --------------------------------------------------------------------------- #
def score_sesion(spec, sess, n_req, acc, rng_bern=None):
    """Score normalizado de la sesión (§2): esperado con la tabla acc, o con
    aciertos Bernoulli si rng_bern se proporciona (varianza real, §8)."""
    n_eff, forz = ejecutar(n_req, sess["B_total"])
    ks = [i["K"] for i in sess["instances"]]
    stakes = sess["stake_por_instancia"]
    num, den = 0.0, 0.0
    for i in range(spec.E):
        a = acc[n_eff[i] - 1, ks[i] - K_MIN]
        x = a if rng_bern is None else float(rng_bern.random() < a)
        num += stakes[i] * x
        den += stakes[i]
    return num / den, any(forz)


def eval_politica(spec, sessions, pol, acc, d_ref):
    scores, forz = [], []
    for sess in sessions:
        s, f = score_sesion(spec, sess, pol(spec, sess, d_ref, acc), acc)
        scores.append(s)
        forz.append(f)
    return np.array(scores), np.array(forz)


def boot_ci_lower(diffs_fn, n_sessions, n_boot, rng, alpha=0.05):
    """IC-inferior unilateral (percentil alpha) por bootstrap de sesiones.
    diffs_fn(idx) → estadístico sobre el remuestreo idx."""
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_sessions, n_sessions)
        stats[b] = diffs_fn(idx)
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 0.5))


def mi_bits(x, y):
    """MI plug-in en bits entre dos discretas (arrays 1D)."""
    xs, ys = np.unique(x), np.unique(y)
    n = len(x)
    mi = 0.0
    for xv in xs:
        for yv in ys:
            pxy = np.mean((x == xv) & (y == yv))
            if pxy > 0:
                px, py = np.mean(x == xv), np.mean(y == yv)
                mi += pxy * math.log2(pxy / (px * py))
    return max(0.0, mi)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="perfil sintético (solo validación del harness)")
    ap.add_argument("--n_mc", type=int, default=2000)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    spec = SessionSpec()
    acc, d_ref, origen = load_profile(args.synthetic)
    print(f"Perfil: {origen}")
    print(f"d_ref(K): { {K: d_ref[K] for K in sorted(d_ref)} }")
    dem_est = _stationary_demand(spec, d_ref)
    print(f"Demanda estacionaria E[d_ref]={dem_est:.2f} → demanda de sesión "
          f"≈{dem_est * spec.E:.1f} vs B_total={spec.B_total}")

    # --- sesiones Monte Carlo (modo light; stream propio del entorno) ---
    se = seed_env_train(args.seed)
    sessions = [gen_session(spec, se, i, materialize=False)
                for i in range(args.n_mc)]
    rng = np.random.default_rng(12345)

    # --- escalera de políticas ---
    resultados = {}
    for name, pol in POLITICAS.items():
        sc, fz = eval_politica(spec, sessions, pol, acc, d_ref)
        resultados[name] = {"scores": sc, "forzados": fz}
    # familia taper: la mejor de la rejilla
    taper_all = {}
    for name, pol in TAPER_GRID:
        sc, fz = eval_politica(spec, sessions, pol, acc, d_ref)
        taper_all[name] = float(sc.mean())
    best_taper_name = max(taper_all, key=taper_all.get)
    sc, fz = eval_politica(spec, sessions,
                           dict(TAPER_GRID)[best_taper_name], acc, d_ref)
    resultados["taper"] = {"scores": sc, "forzados": fz}
    # demanda (solo mordida)
    sc_dem, fz_dem = eval_politica(spec, sessions, pol_demanda, acc, d_ref)

    orden = ["uniforme", "taper", "stake_greedy", "contador", "ultima_K",
             "bayes", "oraculo"]
    print(f"\n=== Escalera de políticas (n={args.n_mc} sesiones, score "
          f"esperado normalizado) ===")
    for name in orden:
        r = resultados[name]
        extra = f"  [mejor: {best_taper_name}]" if name == "taper" else ""
        print(f"  {name:14s} {r['scores'].mean():.4f} ± "
              f"{r['scores'].std() / math.sqrt(args.n_mc):.4f}   "
              f"forzados en {r['forzados'].mean() * 100:5.1f}% de sesiones{extra}")

    S = {k: resultados[k]["scores"] for k in orden}
    ciegas = ["uniforme", "taper", "stake_greedy", "contador"]

    # ------------------------------------------------------------------ GS1
    mordida_dem = float(fz_dem.mean())
    n_unif = spec.B_total // spec.E
    insuf = np.array([any(d_ref[i["K"]] > n_unif for i in s["instances"])
                      for s in sessions])
    dem_total = np.array([sum(d_ref[i["K"]] for i in s["instances"])
                          > spec.B_total for s in sessions])
    gs1a_ok = mordida_dem >= 0.60

    def headroom(idx):
        return (S["oraculo"][idx].mean()
                - max(S[c][idx].mean() for c in ciegas))
    rngb = np.random.default_rng(777)
    hr_lo, hr_med = boot_ci_lower(headroom, args.n_mc, args.n_boot, rngb)
    gs1b_head_ok = hr_lo >= 0.08

    def d_or_sg(idx):
        return S["oraculo"][idx].mean() - S["stake_greedy"][idx].mean()
    sg_lo, sg_med = boot_ci_lower(d_or_sg, args.n_mc, args.n_boot, rngb)
    gs1b_sg_ok = sg_lo >= 0.08

    # GS1c: headroom ≥ 2× MDD (aprox.: t pareada unilateral α=.05, pot .8 →
    # MDD ≈ 2.5·sd_diff/√n con n = N_eval provisional 256)
    best_c = max(ciegas, key=lambda c: S[c].mean())
    sd_diff = float((S["oraculo"] - S[best_c]).std())
    mdd = 2.5 * sd_diff / math.sqrt(256)
    gs1c_ok = hr_med >= 2 * mdd

    print(f"\n=== GS1 (el presupuesto muerde) ===")
    print(f"  mordida (demanda d_ref, ≥1 forzado): {mordida_dem * 100:.1f}% "
          f"(umbral ≥60%) → {'PASA' if gs1a_ok else 'FALLA'}")
    print(f"    [aux] uniforme insuficiente (algún d_ref>B/E={n_unif}): "
          f"{insuf.mean() * 100:.1f}%;  Σd_ref>B: {dem_total.mean() * 100:.1f}%")
    print(f"  headroom perceptivo oraculo−mejor_ciega[{best_c}]: "
          f"mediana {hr_med:+.4f}, IC-inf {hr_lo:+.4f} (umbral ≥+0.08) "
          f"→ {'PASA' if gs1b_head_ok else 'FALLA'}")
    print(f"  GS1b oraculo−stake_greedy: mediana {sg_med:+.4f}, IC-inf "
          f"{sg_lo:+.4f} (≥+0.08) → {'PASA' if gs1b_sg_ok else 'FALLA'}")
    print(f"  GS1c headroom ≥ 2×MDD: {hr_med:+.4f} vs 2×{mdd:.4f}={2 * mdd:.4f} "
          f"→ {'PASA' if gs1c_ok else 'FALLA'}")

    # ------------------------------------------------------------------ GS2
    rango = float(S["oraculo"].mean() - S["uniforme"].mean())
    umbral_gs2 = 0.05 * rango

    def d_bay_last(idx):
        return S["bayes"][idx].mean() - S["ultima_K"][idx].mean()
    bl_lo, bl_med = boot_ci_lower(d_bay_last, args.n_mc, args.n_boot, rngb)
    gs2a_ok = bl_lo >= umbral_gs2

    def d_bay_cont(idx):
        return S["bayes"][idx].mean() - S["contador"][idx].mean()
    bc_lo, bc_med = boot_ci_lower(d_bay_cont, args.n_mc, args.n_boot, rngb)
    gs2b_ok = bc_lo > 0.0

    print(f"\n=== GS2 (la escala lenta paga con información aprendible) ===")
    print(f"  rango de la escalera (oraculo−uniforme): {rango:.4f} → umbral "
          f"0.05·rango = {umbral_gs2:.4f}")
    print(f"  bayes−ultima_K: mediana {bl_med:+.4f}, IC-inf {bl_lo:+.4f} "
          f"→ {'PASA' if gs2a_ok else 'FALLA'}   "
          f"[abs +0.05: {'sí' if bl_lo >= 0.05 else 'no'}]")
    print(f"  bayes−contador: mediana {bc_med:+.4f}, IC-inf {bc_lo:+.4f} "
          f"(>0) → {'PASA' if gs2b_ok else 'FALLA'}")
    if not (gs2a_ok and gs2b_ok):
        print("  DIAGNÓSTICO GS2: la historia filtrada no paga sobre la "
              "persistencia-de-1. Diales (≤3 rondas commiteadas, decisión "
              "del orquestador):")
        print("   - MÁS solape de K (p.ej. R1 → [10,20]): un K aislado "
              "identifica menos el régimen → filtrar la historia gana más.")
        print("   - MENOS p_switch (p.ej. 1/4): permanencia más larga → la "
              "historia predice mejor el régimen actual.")
        print("   - Revisar la CURVATURA de d_ref: si d_ref es casi plano "
              "entre regímenes, ninguna inferencia de régimen paga (dial: "
              "separar las medias de K o ampliar B para que la diferencia "
              "de demanda importe).")

    # ------------------------------------------------------------------ GS3
    n_gs3 = max(args.n_mc, 2000)
    sess3 = sessions if n_gs3 == args.n_mc else [
        gen_session(spec, se, i, materialize=False) for i in range(n_gs3)]
    regs = np.array([s["regimen_por_instancia"] for s in sess3])
    stks = np.array([s["stake_por_instancia"] for s in sess3])
    ks3 = np.array([[i["K"] for i in s["instances"]] for s in sess3])
    pos = np.tile(np.arange(spec.E), (len(sess3), 1))
    mi_pos = mi_bits(regs.ravel(), pos.ravel())
    mi_stk = mi_bits(regs.ravel(), (stks == spec.stake_high).ravel().astype(int))
    corr_k = float(np.corrcoef(regs.ravel(), ks3.ravel())[0, 1])
    gs3_ok = (mi_pos < 0.05) and (mi_stk < 0.05)
    print(f"\n=== GS3 (sin fugas — proxies, n={len(sess3)} sesiones) ===")
    print(f"  MI(régimen; posición) = {mi_pos:.5f} bits (<0.05) "
          f"→ {'PASA' if mi_pos < 0.05 else 'FALLA'}")
    print(f"  MI(régimen; stake)    = {mi_stk:.5f} bits (<0.05) "
          f"→ {'PASA' if mi_stk < 0.05 else 'FALLA'}")
    print(f"  corr(régimen; K propio) = {corr_k:+.3f}  [señal LEGÍTIMA por "
          f"construcción: no es fuga, es la observación]")

    # ------------------------------------------------- varianza → N_eval (§8)
    rng_b = np.random.default_rng(999)
    sd_bern = {}
    for name in ("uniforme", "bayes", "oraculo"):
        scs = []
        for sess in sessions[:1000]:
            n_req = POLITICAS[name](spec, sess, d_ref, acc)
            s, _ = score_sesion(spec, sess, n_req, acc, rng_bern=rng_b)
            scs.append(s)
        sd_bern[name] = float(np.std(scs))
    sd_between = 0.03                       # proxy sd entre-seeds (§8)
    se_max = sd_between / 3.0
    n_eval = {k: int(math.ceil((v / se_max) ** 2)) for k, v in sd_bern.items()}
    n_eval_rec = max(256, max(n_eval.values()))
    print(f"\n=== Varianza por-sesión (Bernoulli) y N_eval (§8) ===")
    for k in sd_bern:
        print(f"  sd_sesion[{k}]={sd_bern[k]:.4f} → N_eval≥{n_eval[k]}")
    print(f"  N_eval recomendado (SE_intra ≤ ⅓·{sd_between}): {n_eval_rec} "
          f"sesiones/condición/celda (suelo prereg 256)")

    # ------------------------------------------------------------------ salida
    verdicto = {
        "GS1a_mordida": bool(gs1a_ok), "GS1b_headroom": bool(gs1b_head_ok),
        "GS1b_vs_stake_greedy": bool(gs1b_sg_ok), "GS1c_2xMDD": bool(gs1c_ok),
        "GS2_bayes_vs_ultimaK": bool(gs2a_ok),
        "GS2_bayes_vs_contador": bool(gs2b_ok), "GS3_proxies": bool(gs3_ok),
    }
    todo_ok = all(verdicto.values())
    out = {
        "perfil": "sintetico" if args.synthetic else "real",
        "perfil_origen": origen,
        "valido_para_veredicto": not args.synthetic,
        "spec": {k: (list(v) if isinstance(v, tuple) else v)
                 for k, v in vars(spec).items()},
        "n_mc": args.n_mc, "n_boot": args.n_boot,
        "d_ref": {str(k): v for k, v in d_ref.items()},
        "demanda_estacionaria": dem_est,
        "escalera": {k: {"score": float(S[k].mean()),
                         "se": float(S[k].std() / math.sqrt(args.n_mc)),
                         "frac_forzados": float(resultados[k]["forzados"].mean())}
                     for k in orden},
        "taper_grid": taper_all, "mejor_taper": best_taper_name,
        "GS1": {"mordida_demanda": mordida_dem,
                "mordida_uniforme_insuficiente": float(insuf.mean()),
                "frac_demanda_total_excede_B": float(dem_total.mean()),
                "headroom_mediana": hr_med, "headroom_ic_inf": hr_lo,
                "mejor_ciega": best_c,
                "oraculo_menos_stake_greedy_mediana": sg_med,
                "oraculo_menos_stake_greedy_ic_inf": sg_lo,
                "mdd_n256": mdd},
        "GS2": {"rango": rango, "umbral": umbral_gs2,
                "bayes_menos_ultimaK_mediana": bl_med,
                "bayes_menos_ultimaK_ic_inf": bl_lo,
                "bayes_menos_contador_mediana": bc_med,
                "bayes_menos_contador_ic_inf": bc_lo},
        "GS3": {"mi_regimen_posicion_bits": mi_pos,
                "mi_regimen_stake_bits": mi_stk,
                "corr_regimen_K_propio": corr_k, "n_sesiones": len(sess3)},
        "N_eval": {"sd_bernoulli": sd_bern, "por_politica": n_eval,
                   "recomendado": n_eval_rec, "sd_entre_seeds_proxy": sd_between},
        "verdicto": verdicto, "todos_los_gates": bool(todo_ok),
    }
    os.makedirs(RES, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\n=== VEREDICTO ({out['perfil']}) ===")
    for k, v in verdicto.items():
        print(f"  {k:24s} {'PASA' if v else 'FALLA'}")
    print(f"  TODOS: {'PASA' if todo_ok else 'FALLA'}"
          + ("  [PERFIL SINTETICO: NO VALIDO como veredicto real]"
             if args.synthetic else ""))
    print(f"Guardado: {OUT_PATH}")


if __name__ == "__main__":
    main()
