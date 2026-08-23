"""
LA PREGUNTA DECISIVA DEL CERTIFICADO LMI: no si la caja DECLARADA se puede
certificar (ya sabemos que no: contiene configuraciones divergentes), sino si
existe un ENTORNO CERTIFICADO alrededor del punto de operacion que los modelos
entrenados ocupan de verdad.

Si la respuesta es si, el teorema aplica a lo que corrimos y la conclusion es
que los TOPES de HBPConfig hay que estrecharlos. Si es no, el certificado por
Lyapunov comun no es la via y hay que decirlo.

  PYTHONPATH=. python certify_lmi_ckpt.py
"""
import glob
import io
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from certify_lmi import (factible, mapas, mejor_rho, norma_P, rho_vertices)

# (la codificacion se fija con PYTHONIOENCODING=utf-8; envolver sys.stdout
#  aqui rompe a quien importe este modulo)


def lee_ckpts(pats=("checkpoints/*.pt", "mhbp/tasks/reasoner_g0/ckpts/*.pt"),
              lim=80):
    """Los checkpoints anidan los pesos bajo la clave 'state_dict'."""
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
            ko = [k for k in sd if k.endswith("raw_omega0")]
            kz = [k for k in sd if k.endswith("raw_zeta")]
            kc = [k for k in sd if k.endswith("raw_c")]
            if not (ko and kz and kc):
                continue
            om = 0.2 + 1.6 * torch.sigmoid(sd[ko[0]].float())
            ze = 0.05 + F.softplus(sd[kz[0]].float())
            cc = 0.7 * torch.sigmoid(sd[kc[0]].float())
            filas.append({"f": os.path.basename(p),
                          "om": (float(om.min()), float(om.max())),
                          "ze": (float(ze.min()), float(ze.max())),
                          "c": float(cc.max())})
    return filas


if __name__ == "__main__":
    dt = 1.0
    ck = lee_ckpts()
    print("=" * 70)
    print("1. PUNTO DE OPERACION REAL DE LOS MODELOS ENTRENADOS")
    print("=" * 70)
    if not ck:
        print("  no se pudo leer ninguno")
        sys.exit(0)
    om_lo = min(f["om"][0] for f in ck)
    om_hi = max(f["om"][1] for f in ck)
    ze_lo = min(f["ze"][0] for f in ck)
    ze_hi = max(f["ze"][1] for f in ck)
    c_hi = max(f["c"] for f in ck)
    print(f"  {len(ck)} checkpoints leidos")
    print(f"    omega0 en [{om_lo:.3f}, {om_hi:.3f}]")
    print(f"    zeta   en [{ze_lo:.3f}, {ze_hi:.3f}]")
    print(f"    c      <= {c_hi:.3f}")
    for f in ck[:6]:
        print(f"      {f['f'][:44]:44s} om[{f['om'][0]:.2f},{f['om'][1]:.2f}]"
              f" ze[{f['ze'][0]:.2f},{f['ze'][1]:.2f}] c={f['c']:.3f}")

    print("\n" + "=" * 70)
    print("2. ES ESTABLE EL PUNTO DE OPERACION, SIN CAJA?")
    print("=" * 70)
    Ph = mapas(dt, (om_lo, om_hi), (ze_lo, ze_hi), (0.0, c_hi))
    r = rho_vertices(Ph)
    malos = sum(1 for M in Ph if np.max(np.abs(np.linalg.eigvals(M))) >= 1.0)
    print(f"  caja envolvente de lo entrenado: {malos}/{len(Ph)} vertices"
          f" divergentes, peor rho={r:.4f}")

    print("\n" + "=" * 70)
    print("3. HAY P COMUN SOBRE LA ENVOLVENTE DE LO ENTRENADO?")
    print("=" * 70)
    if malos:
        print("  NO: la propia envolvente contiene vertices divergentes.")
        print("  (recuerda que la envolvente es conservadora: desacopla")
        print("   omega0 de zeta y toma el peor c de todos los checkpoints)")
    else:
        rho, P = mejor_rho(Ph)
        if P is None:
            print("  NO: sin vertices divergentes, pero la LMI no cierra.")
        else:
            _, kappa = norma_P(Ph[0], P)
            peor = max(norma_P(M, P)[0] for M in Ph)
            print(f"  SI. rho={rho:.4f}  max||Phi||_P={peor:.4f}"
                  f"  kappa={kappa:.2f}")
            print(f"  => small-gain: ISS si L_F < {(1-rho)/kappa:.5f}")

    print("\n" + "=" * 70)
    print("4. Y POR CHECKPOINT INDIVIDUAL (el caso mas favorable posible)")
    print("=" * 70)
    ok = 0
    for f in ck:
        Pi = mapas(dt, f["om"], f["ze"], (0.0, f["c"]))
        ri = rho_vertices(Pi)
        est = ri < 1.0
        cert = est and factible(Pi)
        ok += cert
        if len(ck) <= 12 or cert:
            print(f"    {f['f'][:44]:44s} rho={ri:.3f}"
                  f"  {'CERTIFICABLE' if cert else ('estable' if est else 'DIVERGE')}")
    print(f"\n  certificables por si solos: {ok}/{len(ck)}")
