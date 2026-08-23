"""
CIERRE DEL CERTIFICADO: (a) la MEZCLA bajo el P comun de la region entrenada,
y (b) la constante de Lipschitz REAL del forzamiento, para el small-gain.

La cadena que se certifica es:
  - Existe P comun sobre la envolvente de los checkpoints entrenados
    (certify_lmi_ckpt.py): ||Phi||_P <= rho para todo vertice.
  - La norma de operador es CONVEXA => toda mezcla alpha hereda la cota, y
    SUBMULTIPLICATIVA => el producto LTV del caso gateado tambien.
  - El unico forzamiento que depende del estado es
    f_gain * f_theta([h ; s]), con f_theta = Linear -> SiLU -> Linear -> Tanh.
    Su Lipschitz respecto de h se acota EXACTAMENTE por
        L_F <= f_gain * ||W2||_2 * Lip(SiLU) * ||W1_h||_2 * Lip(tanh)
    con Lip(SiLU) = 1.0998 (maximo de la derivada) y Lip(tanh) = 1.
    (g_phi depende de ext_force, NO de h: no entra en la ganancia de lazo.)
  - Small-gain: ISS si rho + kappa*dt*L_F < 1.

CAVEAT DECLARADO: esto certifica el lazo INTERNO del campo. El lazo que pasa
por el host (h -> modulacion -> transformer -> interocepcion -> f_theta) tiene
una sensibilidad que no se acota analiticamente desde los pesos del campo; se
reporta aparte como hueco, no se da por cerrado.

  PYTHONPATH=. python certify_lmi_iss.py
"""
import glob
import io
import os
import sys

import numpy as np
import torch

from certify_lmi import mapas, mejor_rho, norma_P, rho_vertices
from certify_lmi_ckpt import lee_ckpts

# (codificacion via PYTHONIOENCODING=utf-8)

LIP_SILU = 1.0998


def lipschitz_forzamiento(pats=("checkpoints/*.pt",), lim=20):
    """Cota EXACTA de L_F por checkpoint, desde los pesos."""
    filas = []
    for pat in pats:
        for p in sorted(glob.glob(pat))[:lim]:
            try:
                obj = torch.load(p, map_location="cpu", weights_only=True)
            except Exception:
                continue
            sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
            if not isinstance(sd, dict):
                continue
            k1 = [k for k in sd if k.endswith("f_theta.0.weight")]
            k2 = [k for k in sd if k.endswith("f_theta.2.weight")]
            kg = [k for k in sd if k.endswith("raw_f_gain")]
            ko = [k for k in sd if k.endswith("raw_omega0")]
            if not (k1 and k2 and kg and ko):
                continue
            W1 = sd[k1[0]].float()              # (2 d_h, d_h + d_intero)
            W2 = sd[k2[0]].float()              # (d_h, 2 d_h)
            d_h = W2.shape[0]
            W1h = W1[:, :d_h]                   # solo la parte que ve a h
            f_gain = 0.3 * torch.sigmoid(sd[kg[0]].float())
            LF = (float(f_gain) * float(torch.linalg.matrix_norm(W2, 2))
                  * LIP_SILU * float(torch.linalg.matrix_norm(W1h, 2)))
            filas.append((os.path.basename(p), float(f_gain),
                          float(torch.linalg.matrix_norm(W1h, 2)),
                          float(torch.linalg.matrix_norm(W2, 2)), LF))
    return filas


if __name__ == "__main__":
    dt = 1.0
    ck = lee_ckpts()
    if not ck:
        print("sin checkpoints")
        sys.exit(0)
    om = (min(f["om"][0] for f in ck), max(f["om"][1] for f in ck))
    ze = (min(f["ze"][0] for f in ck), max(f["ze"][1] for f in ck))
    cm = max(f["c"] for f in ck)

    print("=" * 70)
    print("1. P COMUN SOBRE LA ENVOLVENTE ENTRENADA")
    print("=" * 70)
    Ph = mapas(dt, om, ze, (0.0, cm))
    rho, P = mejor_rho(Ph)
    if P is None:
        print("  no cierra")
        sys.exit(0)
    _, kappa = norma_P(Ph[0], P)
    print(f"  omega0{tuple(round(x,3) for x in om)}"
          f"  zeta{tuple(round(x,3) for x in ze)}  c<={cm:.3f}")
    print(f"  peor radio espectral en vertices : {rho_vertices(Ph):.4f}")
    print(f"  tasa comun certificada rho       : {rho:.4f}")
    print(f"  kappa                            : {kappa:.2f}")

    print("\n" + "=" * 70)
    print("2. LA MEZCLA, BAJO EL MISMO P (lo que el radio espectral no cierra)")
    print("=" * 70)
    Pmix = mapas(dt, om, ze, (0.0, cm), D=(0.0, 0.4), b=(0.0, 0.4),
                 be=(0.0, 0.1), con_dif=True)
    peor = max(norma_P(M, P)[0] for M in Pmix)
    print(f"  mapas evaluados (onda + difusiva + operadores antisim.): "
          f"{len(Pmix)}")
    print(f"  max ||Phi||_P                                         : "
          f"{peor:.4f}")
    if peor < 1:
        print("  => MEZCLA CERTIFICADA. Por convexidad de la norma de")
        print("     operador, toda combinacion alpha hereda la cota; por")
        print("     submultiplicatividad, tambien el producto LTV gateado.")
    else:
        rho2, P2 = mejor_rho(Pmix)
        if P2 is not None:
            k2 = norma_P(Pmix[0], P2)[1]
            print(f"  con el P del subconjunto no basta, pero existe P propio")
            print(f"  para el conjunto AMPLIADO: rho={rho2:.4f}, kappa={k2:.2f}")
            print("  => MEZCLA CERTIFICADA con ese P.")
            rho, P, kappa = rho2, P2, k2
        else:
            print("  => la mezcla NO cierra ni con P propio.")

    print("\n" + "=" * 70)
    print("3. LIPSCHITZ REAL DEL FORZAMIENTO (cota exacta desde los pesos)")
    print("=" * 70)
    lf = lipschitz_forzamiento()
    if not lf:
        print("  no se pudieron leer los pesos de f_theta")
    else:
        print(f"  {'checkpoint':44s} {'f_gain':>7s} {'|W1h|':>7s}"
              f" {'|W2|':>7s} {'L_F':>9s}")
        for n, g, w1, w2, L in lf:
            print(f"  {n[:44]:44s} {g:7.4f} {w1:7.3f} {w2:7.3f} {L:9.5f}")
        LFmax = max(x[4] for x in lf)
        print(f"\n  L_F maximo observado: {LFmax:.5f}")

        print("\n" + "=" * 70)
        print("4. SMALL-GAIN: ISS DEL LAZO INTERNO DEL CAMPO")
        print("=" * 70)
        umbral = (1 - rho) / kappa
        v = rho + kappa * dt * LFmax
        print(f"  condicion : rho + kappa*dt*L_F < 1")
        print(f"  numeros   : {rho:.4f} + {kappa:.2f}*{dt}*{LFmax:.5f}"
              f" = {v:.4f}")
        print(f"  umbral    : L_F < {umbral:.5f}")
        print(f"  margen    : {umbral/LFmax:.1f}x" if LFmax > 0 else "")
        print(f"\n  => {'ISS CERTIFICADO' if v < 1 else 'NO CIERRA'}")

    print("\n" + "=" * 70)
    print("5. LO QUE ESTO NO CERTIFICA (declarado, no escondido)")
    print("=" * 70)
    print("  El lazo que pasa por el HOST -- h -> modulacion -> transformer ->")
    print("  interocepcion s -> f_theta(.,s) -> h -- tiene una sensibilidad")
    print("  que no se acota desde los pesos del campo. Lo certificado aqui es")
    print("  el lazo INTERNO. Cerrar el externo exige una cota de Lipschitz")
    print("  del host respecto de su modulacion, que no tenemos.")
