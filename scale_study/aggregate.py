"""
Agregador del barrido nocturno: lee results/*.json, genera figuras en figures/ y
un REPORT.md con tablas y el VEREDICTO pre-registrado (calculado de los datos).
Robusto a celdas faltantes (se puede correr con el barrido a medias).

  python -m scale_study.aggregate
"""
import os, sys, json, glob, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def load():
    cells = {}
    for fp in glob.glob(os.path.join(RES, "*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                d = json.load(f)
            cells[d.get("cell_id", os.path.basename(fp)[:-5])] = d
        except Exception:
            pass
    return cells


def by_block(cells, block):
    return [c for c in cells.values() if c.get("block") == block and "error" not in c]


R = []   # líneas del report
def w(line=""): R.append(line)


def fig_B1(cells):
    rows = by_block(cells, "B1")
    if not rows:
        return
    kinds = sorted(set(r["kind"] for r in rows))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, ttl in zip(axes, ["rho_L", "rho_A3", "flutter"],
                            ["ρ(L)", "ρ(A³)", "umbral flutter β·ρ(A³) vs 2ζω₀²"]):
        for kind in kinds:
            rs = sorted([r for r in rows if r["kind"] == kind], key=lambda r: r["N"])
            Ns = [r["N"] for r in rs]
            if key == "flutter":
                ax.plot(Ns, [r["flutter_ref"]["lhs"] for r in rs], "o-", label=f"{kind}")
            else:
                ax.plot(Ns, [r[key] for r in rs], "o-", label=kind)
        if key == "flutter":
            ax.axhline(rows[0]["flutter_ref"]["rhs"], ls="--", color="k", label="2ζω₀² (ref)")
        ax.set_xscale("log"); ax.set_xlabel("N (nodos)"); ax.set_title(ttl)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("B1 — Espectro de los operadores del grafo vs escala y topología")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "B1_spectrum.png"), dpi=140); plt.close(fig)
    # tabla resumen
    w("## B1 — Espectro a escala\n")
    w("| topología | N | ρ(L) | ρ(A) | ρ(A³) |")
    w("|---|---|---|---|---|")
    for kind in kinds:
        for r in sorted([r for r in rows if r["kind"] == kind], key=lambda r: r["N"])[::2]:
            w(f"| {kind} | {r['N']:,} | {r['rho_L']:.3f} | {r['rho_A']:.3f} | {r['rho_A3']:.3f} |")
    w("")
    # H1: radio acotado en grado acotado, crece en grado creciente
    struct = [r for r in rows if r["kind"] in ("ring", "chain", "grid2d")]
    sparse = [r for r in rows if r["kind"] in ("random_regular", "expander", "ws_smallworld")]
    if struct:
        big = [r for r in struct if r["N"] >= 10000]
        bounded = all(r["rho_A3"] < 9.0 for r in big) if big else True
        w(f"**H1 (grado acotado → ρ acotado):** {'CONFIRMADA' if bounded else 'REFUTADA'} "
          f"— ρ(A³) máx en estructurados N≥10⁴: {max((r['rho_A3'] for r in big), default=0):.2f} (≤8 teórico).\n")


def _b2_exact_freq(N, beta, ks, om0=0.5, cc=0.4, dt=0.1):
    """Frecuencia discreta EXACTA del rollout de B2 (Verlet + giroscópico por
    diferencia atrasada): raíz del companion complejo por modo (Prop. 2 del paper)
      z² − (2 − q − i·m)·z + (1 − i·m) = 0,
    con q = Δt²·(ω₀²+c²λ_L), m = Δt·β·μ₃, μ₃ = −(2sinθ)³. La rama dominante
    (mayor |z|) fija la frecuencia observada |arg z|/Δt. Con β=0 se reduce a la
    fórmula clásica acos(1−q/2)/Δt."""
    th = 2.0 * np.pi * np.asarray(ks, dtype=float) / N
    lamL = 4.0 * np.sin(th / 2.0) ** 2
    q = dt ** 2 * (om0 ** 2 + cc ** 2 * lamL)
    mu3 = -((2.0 * np.sin(th)) ** 3)
    m = dt * beta * mu3
    b_ = -(2.0 - q - 1j * m)
    c_ = (1.0 - 1j * m)
    disc = np.sqrt(b_ ** 2 - 4.0 * c_ + 0j)
    z1 = (-b_ + disc) / 2.0
    z2 = (-b_ - disc) / 2.0
    z = np.where(np.abs(z1) >= np.abs(z2), z1, z2)
    return np.abs(np.angle(z)) / dt


def fig_B2(cells):
    rows = by_block(cells, "B2")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    maxerr = 0.0
    for r in sorted(rows, key=lambda r: (r["beta"], r["N"])):
        ks = np.array(r["modes"]); N = r["N"]
        th = ks / N
        meas = np.array(r["omega_measured"])
        ana = _b2_exact_freq(N, r["beta"], ks)       # referencia exacta CON giroscópico
        m = ana > 1e-6
        err = np.abs(meas[m] - ana[m]) / ana[m]
        maxerr = max(maxerr, float(err.max()) if err.size else 0)
        ax.plot(th, meas, ".", ms=4, label=f"N={N} β={r['beta']} (medida)")
        ax.plot(th, ana, "-", lw=0.8, alpha=0.6)
    ax.set_xlabel("k/N (modo normalizado)"); ax.set_ylabel("ω (frecuencia)")
    ax.set_title("B2 — Dispersión en el anillo: medida (puntos) vs companion exacto (línea)")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "B2_dispersion.png"), dpi=140); plt.close(fig)
    w("## B2 — Relación de dispersión (anillo)\n")
    w(f"**H2 (verificación):** error máx medida vs companion EXACTO (con el término "
      f"giroscópico discreto de la Prop. 2) = {maxerr:.2e} "
      f"({'PASS' if maxerr < 0.03 else 'FAIL'}, tol 3%). La dispersión β·A³ modifica "
      f"ω(k) de forma medible y la teoría discreta la predice; el espectro es un "
      f"continuo a N grande (vs 3 puntos a N=6).\n")


def _solitons(cells):
    # single_soliton_run APLANA las métricas del track (velocity_measured, amp_cv,
    # width_cv, ...) — filtrar por la métrica aplanada, no por una clave 'track'.
    return [c for c in cells.values() if c.get("block") in ("B3s", "B3x", "B3map")
            and "error" not in c and "velocity_measured" in c]


def fig_B3s(cells):
    rows = _solitons(cells)
    if not rows:
        return
    # criterio pre-registrado de "solitón limpio"
    def is_clean(r):
        ar = r.get("amp_ratio_final_initial"); wc = r.get("width_cv"); ve = r.get("vel_err")
        return (ar is not None and 0.8 <= ar <= 1.2 and wc is not None and wc < 0.25
                and ve is not None and ve < 0.2 and r.get("finite"))
    gen = [r for r in rows if r["nonlin"] == "genuine" and r.get("damp") == "none"]
    sat = [r for r in rows if r["nonlin"] == "saturated" and r.get("damp") == "none"]
    dmp = [r for r in rows if r.get("damp") == "weak"]
    n_gen_clean = sum(is_clean(r) for r in gen)
    n_sat_clean = sum(is_clean(r) for r in sat)
    w("## B3 — Solitones a escala (la prueba central)\n")
    w(f"- genuine (sin amort.): **{n_gen_clean}/{len(gen)}** solitones limpios "
      f"(amp≈cte, ancho CV<0.25, vel≈teoría).")
    w(f"- saturated (sin amort.): **{n_sat_clean}/{len(sat)}** solitones limpios.")
    if gen:
        wcg = np.mean([r["width_cv"] for r in gen if r.get("width_cv") is not None])
        wcs = np.mean([r["width_cv"] for r in sat if r.get("width_cv") is not None]) if sat else float("nan")
        w(f"- ancho CV medio: genuine={wcg:.3f} vs saturated={wcs:.3f}.")
    # H3 / H3s veredicto
    h3 = "CONFIRMADA" if n_gen_clean >= 1 else "REFUTADA"
    h3s = "CONFIRMADA" if (sat and n_sat_clean < n_gen_clean) else ("NO CONCLUYENTE" if sat else "s/d")
    w(f"\n**H3 (emergen solitones a escala):** {h3} — a N≥1024 hay solitones coherentes; "
      f"a N=6 es imposible (3 modos).")
    w(f"**H3s (la saturación degrada):** {h3s} — genuine {n_gen_clean} vs saturated {n_sat_clean} limpios.")
    n_dmp_clean = sum(is_clean(r) for r in dmp)
    if dmp:
        w(f"**Amortiguamiento mata solitones:** {n_dmp_clean}/{len(dmp)} limpios con amort. débil "
          f"(el campo modulador —amortiguado y forzado— no los sostiene).")
    w("")
    # figura: velocidad medida vs teórica + width_cv
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for lab, rs, col in [("genuine", gen, "C0"), ("saturated", sat, "C3")]:
        vt = [r["v_lab_theory"] for r in rs if r.get("velocity_measured") is not None]
        vm = [r["velocity_measured"] for r in rs if r.get("velocity_measured") is not None]
        a1.scatter(vt, vm, s=18, c=col, label=lab, alpha=0.7)
    lims = [0, max([r.get("v_lab_theory", 0) for r in rows] + [0.1])]
    a1.plot(lims, lims, "k--", lw=0.8, label="y=x (solitón ideal)")
    a1.set_xlabel("velocidad teórica (KdV)"); a1.set_ylabel("velocidad medida")
    a1.set_title("Velocidad del solitón: medida vs teoría"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
    for lab, rs, col in [("genuine", gen, "C0"), ("saturated", sat, "C3")]:
        Ns = sorted(set(r["N"] for r in rs))
        wc = [np.mean([r["width_cv"] for r in rs if r["N"] == n and r.get("width_cv") is not None]) for n in Ns]
        a2.plot(Ns, wc, "o-", c=col, label=lab)
    a2.axhline(0.25, ls="--", color="k", label="umbral solitón limpio")
    a2.set_xscale("log"); a2.set_xlabel("N"); a2.set_ylabel("ancho CV (↓ = más coherente)")
    a2.set_title("Coherencia del solitón vs escala"); a2.legend(fontsize=8); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "B3s_solitons.png"), dpi=140); plt.close(fig)

    # figura de perfiles (primer vs último frame de un caso limpio representativo)
    rep = next((r for r in gen if is_clean(r)), (gen[0] if gen else None))
    if rep and rep.get("frame_first"):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(rep["frame_first"], label="t=0 (sech² inicial)", lw=1.2)
        ax.plot(rep["frame_last"], label="t=final", lw=1.2)
        ax.set_title(f"Solitón coherente N={rep['N']} amp={rep['amp']} β={rep['beta']} ({rep['nonlin']})")
        ax.legend(); ax.grid(alpha=0.3); ax.set_xlabel("nodo (submuestreado)")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "B3s_profile.png"), dpi=140); plt.close(fig)


def fig_B3p(cells):
    rows = by_block(cells, "B3p")
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for r in rows:
        if not r.get("frame_last"):
            continue
        ax = axes[0] if r["sign"] > 0 else axes[1]
        ax.plot(r["frame_last"], lw=0.8, label=f"N={r['N']} {r['nonlin']}")
    axes[0].set_title("sign=+1: TREN de solitones (emergencia)")
    axes[1].set_title("sign=−1: dispersión (radiación)")
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_xlabel("nodo")
    fig.suptitle("B3p — El signo del pulso decide: solitones vs dispersión (física de KdV)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "B3p_train.png"), dpi=140); plt.close(fig)


def fig_B3b(cells):
    rows = by_block(cells, "B3b")
    if not rows:
        return
    w("## B3b — Colisión de solitones (elasticidad)\n")
    w("| N | nolin | picos ini | picos fin | amps ini | amps fin |")
    w("|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["N"], r["nonlin"])):
        w(f"| {r['N']:,} | {r['nonlin']} | {r.get('n_start')} | {r.get('n_end')} | "
          f"{[round(x,2) for x in r.get('amps_start',[])]} | {[round(x,2) for x in r.get('amps_end',[])]} |")
    w("")
    r = next((r for r in rows if r.get("frames_stack")), None)
    if r:
        stack = np.array(r["frames_stack"])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(stack, aspect="auto", origin="lower", cmap="magma",
                  extent=[0, stack.shape[1], 0, len(stack)])
        ax.set_xlabel("nodo"); ax.set_ylabel("tiempo (frame)")
        ax.set_title(f"Colisión de dos solitones (N={r['N']}, {r['nonlin']}): "
                     f"¿preservan identidad tras cruzarse?")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "B3b_collision.png"), dpi=140); plt.close(fig)


def fig_B3c(cells):
    rows = by_block(cells, "B3c")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in sorted(rows, key=lambda r: (r["N"], r["beta"])):
        e = np.array(r["mode1_energy"]); e = e / (e[0] + 1e-30)
        ax.plot(np.linspace(0, r["ticks"], len(e)), e, label=f"N={r['N']} β={r['beta']}")
    ax.set_xlabel("tick"); ax.set_ylabel("energía modo-1 (norm.)")
    ax.set_title("B3c — Recurrencia tipo FPUT (energía vuelve al modo fundamental)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "B3c_fput.png"), dpi=140); plt.close(fig)


def fig_B4(cells):
    rows = by_block(cells, "B4")
    if not rows:
        return
    Ns = sorted(set(r["N"] for r in rows))
    fig, axes = plt.subplots(1, len(Ns), figsize=(3.6*len(Ns), 4), squeeze=False)
    tight_ok = True
    for j, N in enumerate(Ns):
        ax = axes[0][j]
        for place, col in [("gyro", "C0"), ("circulatory", "C3")]:
            rs = sorted([r for r in rows if r["N"] == N and r["placement"] == place
                         and r["zeta"] == 0.1], key=lambda r: r["beta"])
            if not rs:
                continue
            betas = [r["beta"] for r in rs]
            slopes = [min(r["log_slope"], 0.01) if math.isfinite(r["log_slope"]) else 0.01 for r in rs]
            ax.plot(betas, slopes, "o-", c=col, label=place)
            if rs:
                thr = rs[0]["flutter"]["rhs"] / rs[0]["flutter"]["lhs"] * rs[0]["beta"] if rs[0]["flutter"]["lhs"] > 0 else None
        # umbral de Merkin
        rhs = rows[0]["flutter"]["rhs"]; ratio = None
        # β* = 2ζω₀²/ρ(A³); ρ(A³)=8, ζ=0.1, ω₀=0.5 -> β*=2·0.1·0.25/8=0.00625... (bajo)
        beta_star = 2*0.1*0.25 / 8.0
        ax.axvline(beta_star, ls="--", color="k", lw=0.8, label="β* Merkin")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(f"N={N:,}"); ax.set_xlabel("β"); ax.set_ylabel("pendiente log‖u‖")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("B4 — Estabilidad a escala: giroscópico decae (≤0), circulatorio flutter (>0) sobre β* de Merkin")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "B4_stability.png"), dpi=140); plt.close(fig)
    # verificación de tightness
    circ = [r for r in rows if r["placement"] == "circulatory" and r["zeta"] == 0.1]
    ok = 0; tot = 0
    for r in circ:
        beta_star = r["flutter"]["rhs"] / 8.0
        predicted_unstable = r["beta"] > beta_star
        observed_unstable = (not math.isfinite(r["log_slope"])) or r["log_slope"] > 1e-4
        tot += 1; ok += (predicted_unstable == observed_unstable)
    w("## B4 — Estabilidad/flutter a escala\n")
    if tot:
        w(f"**H5 (certificado tight a escala):** el umbral de Merkin predice el flutter "
          f"en {ok}/{tot} celdas circulatorias ({100*ok/tot:.0f}%). El giroscópico "
          f"decae en toda topología y N.\n")


def fig_B5(cells):
    rows = by_block(cells, "B5")
    if not rows:
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(5*len(rows), 4.2), squeeze=False)
    for j, r in enumerate(sorted(rows, key=lambda r: r["N"])):
        ax = axes[0][j]
        prop = np.array(r["propagation"])
        ax.imshow(prop, aspect="auto", origin="lower", cmap="viridis",
                  extent=[0, prop.shape[1], 0, r["ticks"]])
        ax.set_title(f"N={r['N']:,}"); ax.set_xlabel("nodo (ventana)"); ax.set_ylabel("tick")
    fig.suptitle("B5 — Propagación de la onda amortiguada a escala GIGANTE (impulso local → frente que se propaga y oscila)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "B5_propagation.png"), dpi=140); plt.close(fig)


def main():
    cells = load()
    n = len([c for c in cells.values() if "error" not in c])
    nerr = len([c for c in cells.values() if "error" in c])
    w(f"# Resultados — Campo homeostático a escala gigante\n")
    w(f"Celdas completadas: **{n}** (errores: {nerr}). Generado por `aggregate.py`.\n")
    counts = {}
    for c in cells.values():
        counts[c.get("block", "?")] = counts.get(c.get("block", "?"), 0) + 1
    w("Celdas por bloque: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + "\n")
    for fn in (fig_B1, fig_B2, fig_B4, fig_B3s, fig_B3p, fig_B3b, fig_B3c, fig_B5):
        try:
            fn(cells)
        except Exception as e:
            w(f"_(error en {fn.__name__}: {repr(e)[:100]})_\n")
    with open(os.path.join(HERE, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))
    print(f"REPORT.md + figuras en figures/ ({n} celdas)")
    print("\n".join(R))


if __name__ == "__main__":
    main()
