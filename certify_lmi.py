"""
CERTIFICADO DE LYAPUNOV COMUN (LMI) + ISS EN LAZO CERRADO POR SMALL-GAIN.

Cierra los cuatro agujeros que el radio espectral NO puede cerrar:

  1. LA MEZCLA ALPHA. rho no es convexo: de rho(Phi_onda)<1 y rho(Phi_dif)<1
     no se sigue nada sobre alpha*Phi_onda+(1-alpha)*Phi_dif. Con un P COMUN
     si, porque ||P^{1/2} Phi P^{-1/2}|| <= rho para cada rama y la NORMA DE
     OPERADOR ES CONVEXA: toda combinacion convexa hereda la cota.
  2. EL CASO GATEADO (LTV). rho(Phi_t)<1 para cada t no dice nada del
     producto; ||Phi_t||_P <= rho para todo t da ||prod Phi_t||_P <= rho^T.
  3. LOS TRANSITORIOS NO NORMALES, invisibles al radio espectral.
  4. EL LAZO CERRADO, via small-gain con las constantes de Lipschitz de las
     cabezas de interocepcion y modulacion.

CLAVE TECNICA. Phi no es afin en (omega0, zeta, c) —aparecen omega0^2,
zeta*omega0 y c^2— luego los vertices de esa caja no cubririan su interior.
Se reparametriza en coordenadas DERIVADAS (k=dt^2 omega0^2, ch=2 dt zeta
omega0, cc=dt^2 c^2, dw=dt D, bb=dt b, be=dt beta), en las que Phi SI es
afin: cada punto de la caja es combinacion convexa de vertices y el chequeo
en vertices certifica el interior. La caja derivada CONTIENE a la fisica
(omega0 y zeta se desacoplan), asi que el certificado es conservador.

  PYTHONPATH=. python certify_lmi.py
"""
import glob
import itertools
import os
import warnings

import cvxpy as cp
import numpy as np

warnings.filterwarnings("ignore")


# ------------------------------------------------------------- operadores
def operadores(N=6):
    L = np.zeros((N, N))
    for i in range(N - 1):
        L[i, i] += 1.0
        L[i + 1, i + 1] += 1.0
        L[i, i + 1] -= 1.0
        L[i + 1, i] -= 1.0
    A = np.zeros((N, N))
    for i in range(N - 1):
        A[i, i + 1] = 0.5
        A[i + 1, i] = -0.5
    return L, A, A @ A @ A


def phi_onda(k, ch, cc, dw, bb, be, L, A, A3):
    """Companion 2N x 2N del Verlet de posicion. AFIN en los seis argumentos."""
    N = L.shape[0]
    I = np.eye(N)
    K = k * I + cc * L
    C = ch * I + dw * L
    G = bb * A + be * A3
    return np.vstack([np.hstack([2 * I - K - C - G, -(I - C - G)]),
                      np.hstack([I, np.zeros((N, N))])])


def phi_dif(k, cc, dw, bb, be, gam, L, A, A3):
    """Rama difusiva IMEX, elevada al mismo espacio 2N para poder mezclar."""
    N = L.shape[0]
    I = np.eye(N)
    M = I + (k * I + cc * L + bb * A + be * A3) / gam
    S = np.linalg.solve(M, I)
    return np.vstack([np.hstack([S, np.zeros((N, N))]),
                      np.hstack([I, np.zeros((N, N))])])


def caja_derivada(dt, om, ze, c, D=(0, 0), b=(0, 0), be=(0, 0)):
    return [(dt ** 2 * om[0] ** 2, dt ** 2 * om[1] ** 2),
            (2 * dt * ze[0] * om[0], 2 * dt * ze[1] * om[1]),
            (dt ** 2 * c[0] ** 2, dt ** 2 * c[1] ** 2),
            (dt * D[0], dt * D[1]), (dt * b[0], dt * b[1]),
            (dt * be[0], dt * be[1])]


def vertices(caja):
    return list(itertools.product(*[(a, b) for a, b in caja]))


def mapas(dt, om, ze, c, D=(0, 0), b=(0, 0), be=(0, 0),
          con_dif=False, gam=(0.1, 4.0), N=6):
    L, A, A3 = operadores(N)
    vs = vertices(caja_derivada(dt, om, ze, c, D, b, be))
    Phis = [phi_onda(*v, L, A, A3) for v in vs]
    if con_dif:
        for k, ch, cc, dw, bb, bee in vs:
            for g in gam:
                Phis.append(phi_dif(k, cc, dw, bb, bee, g, L, A, A3))
    return Phis


def rho_vertices(Phis):
    return max(float(np.max(np.abs(np.linalg.eigvals(Ph)))) for Ph in Phis)


# -------------------------------------------------------------------- LMI
def lmi(Phis, rho2):
    """Existe P >= I con Phi^T P Phi <= rho2 P para todos? Devuelve P o None.
    La factibilidad se re-verifica a mano: no nos fiamos del status del solver."""
    n = Phis[0].shape[0]
    P = cp.Variable((n, n), symmetric=True)
    cons = [P >> np.eye(n)]
    for Ph in Phis:
        cons.append(cp.bmat([[rho2 * P, Ph.T @ P], [P @ Ph, P]])
                    >> 1e-8 * np.eye(2 * n))
    try:
        cp.Problem(cp.Minimize(cp.trace(P)), cons).solve(
            solver=cp.CLARABEL, verbose=False)
    except Exception:
        return None
    if P.value is None:
        return None
    Pv = P.value
    if np.min(np.linalg.eigvalsh(Pv)) <= 0:
        return None
    for Ph in Phis:
        if np.max(np.linalg.eigvalsh(Ph.T @ Pv @ Ph - rho2 * Pv)) > 1e-6:
            return None
    return Pv


def factible(Phis, rho=0.999):
    return lmi(Phis, rho ** 2) is not None


def mejor_rho(Phis, iters=10):
    """Biseccion fina. Solo en la caja ganadora, no durante el barrido."""
    P0 = lmi(Phis, 0.999 ** 2)
    if P0 is None:
        return None, None
    lo, hi, best, Pb = 0.05, 0.999, 0.999, P0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        P = lmi(Phis, mid ** 2)
        if P is not None:
            best, Pb, hi = mid, P, mid
        else:
            lo = mid
    return best, Pb


def norma_P(M, P):
    w, V = np.linalg.eigh(P)
    R = V @ np.diag(np.sqrt(w)) @ V.T
    Ri = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    return (float(np.linalg.norm(R @ M @ Ri, 2)),
            float(np.sqrt(w.max() / w.min())))


# --------------------------------------------------------------- barrido
def frontera(dt=1.0, ze=(0.2, 0.55), om_min=0.2,
             cs=(0.0, 0.2, 0.4, 0.55, 0.7), N=6, pasos=7):
    """Mayor omega0_max certificable para cada c_max. Durante el barrido solo
    se pregunta FACTIBILIDAD (un SDP), no la mejor tasa."""
    filas, mejor = [], None
    for cm in cs:
        lo, hi, top = om_min + 1e-3, 1.8, None
        for _ in range(pasos):
            mid = 0.5 * (lo + hi)
            Ph = mapas(dt, (om_min, mid), ze, (0.0, cm), N=N)
            if rho_vertices(Ph) >= 1.0 or not factible(Ph):
                hi = mid
            else:
                top, lo = mid, mid
        filas.append((cm, top))
        if top is not None and (mejor is None
                                or top * max(cm, 1e-3) >
                                mejor[1] * max(mejor[0], 1e-3)):
            mejor = (cm, top)
    return filas, mejor


# ---------------------------------------------------------- checkpoints
def checkpoints_reales(lim=60):
    import torch
    import torch.nn.functional as F
    filas = []
    for pat in ("checkpoints/*.pt", "mhbp/tasks/reasoner_g0/ckpts/*.pt"):
        for p in sorted(glob.glob(pat))[:lim]:
            try:
                sd = torch.load(p, map_location="cpu", weights_only=True)
            except Exception:
                continue
            ko = [k for k in sd if k.endswith("raw_omega0")]
            kz = [k for k in sd if k.endswith("raw_zeta")]
            kc = [k for k in sd if k.endswith("raw_c")]
            if not (ko and kz and kc):
                continue
            om = 0.2 + 1.6 * torch.sigmoid(sd[ko[0]].float())
            ze = 0.05 + F.softplus(sd[kz[0]].float())
            cc = 0.7 * torch.sigmoid(sd[kc[0]].float())
            filas.append((os.path.basename(p), float(om.min()),
                          float(om.max()), float(ze.min()), float(ze.max()),
                          float(cc.max())))
    return filas


# -------------------------------------------------------------------- main
if __name__ == "__main__":
    dt, ze = 1.0, (0.2, 0.55)
    print("=" * 70)
    print("1. DIAGNOSTICO: la caja declarada, tal cual esta en HBPConfig")
    print("=" * 70)
    for nombre, om, c in (("declarada", (0.2, 1.8), (0.0, 0.7)),
                          ("operacion", (0.2, 1.0), (0.0, 0.7))):
        Ph = mapas(dt, om, ze, c)
        malos = sum(1 for M in Ph
                    if np.max(np.abs(np.linalg.eigvals(M))) >= 1.0)
        print(f"  {nombre:10s} omega0{om} c{c}: {malos}/{len(Ph)} vertices YA "
              f"divergentes (peor rho={rho_vertices(Ph):.3f})")
    print("  => con vertices de rho>=1 la LMI no puede cerrar: no es fallo del")
    print("     metodo sino que la caja contiene configuraciones divergentes.")
    print("     El culpable es c, que entra en la rigidez como c^2*lmax(L).")

    print("\n" + "=" * 70)
    print("2. FRONTERA DE CERTIFICABILIDAD en (c_max, omega0_max)")
    print("=" * 70)
    filas, mejor = frontera(dt, ze)
    print(f"  zeta{ze}, dt={dt}")
    print(f"    {'c_max':>7s}  {'omega0_max certificable':>24s}")
    for cm, om_max in filas:
        print(f"    {cm:7.3f}  "
              + (f"{om_max:24.3f}" if om_max else f"{'ninguno':>24s}"))

    res = None
    if mejor:
        cm, om_max = mejor
        Ph = mapas(dt, (0.2, om_max), ze, (0.0, cm))
        rho, P = mejor_rho(Ph)
        if P is not None:
            normas = [norma_P(M, P)[0] for M in Ph]
            _, kappa = norma_P(Ph[0], P)
            res = {"rho": rho, "P": P, "kappa": kappa, "c_max": cm,
                   "om_max": om_max, "ze": ze}
            print(f"\n  CAJA CERTIFICADA: c<={cm:.3f}, omega0<={om_max:.3f}, "
                  f"zeta en {ze}")
            print(f"    tasa comun rho          : {rho:.4f}")
            print(f"    max ||Phi||_P en vertices: {max(normas):.4f}")
            print(f"    kappa = sqrt(lmax/lmin) : {kappa:.2f}")
            print("    => certifica la MEZCLA alpha y el caso GATEADO (LTV):")
            print("       la norma de operador es convexa y submultiplicativa.")

            print("\n  Comprobacion de la mezcla con la rama difusiva bajo el")
            print("  MISMO P (el agujero que el radio espectral no cierra):")
            Pm = mapas(dt, (0.2, om_max), ze, (0.0, cm), con_dif=True)
            peor = max(norma_P(M, P)[0] for M in Pm)
            print(f"    max ||Phi||_P sobre onda + difusiva: {peor:.4f}"
                  f"  {'OK' if peor < 1 else 'NO CIERRA'}")

    print("\n" + "=" * 70)
    print("3. DONDE CAEN LOS CHECKPOINTS ENTRENADOS")
    print("=" * 70)
    ck = checkpoints_reales()
    if not ck:
        print("  (ninguno con parametros fisicos)")
    else:
        om_hi = max(f[2] for f in ck)
        ze_lo = min(f[3] for f in ck)
        c_hi = max(f[5] for f in ck)
        print(f"  checkpoints leidos: {len(ck)}")
        print(f"    omega0 <= {om_hi:.3f}   zeta >= {ze_lo:.3f}   "
              f"c <= {c_hi:.3f}")
        for f in ck[:5]:
            print(f"      {f[0][:46]:46s} om<={f[2]:.2f} ze>={f[3]:.2f} "
                  f"c={f[5]:.2f}")
        if res:
            dentro = (om_hi <= res["om_max"] and ze_lo >= res["ze"][0]
                      and c_hi <= res["c_max"])
            print(f"\n  caja certificada: omega0<={res['om_max']:.3f}, "
                  f"zeta>={res['ze'][0]}, c<={res['c_max']:.3f}")
            print(f"  VEREDICTO: los entrenados "
                  f"{'CAEN DENTRO' if dentro else 'NO CAEN DENTRO'}")

    print("\n" + "=" * 70)
    print("4. ISS DEL LAZO CERRADO (small-gain)")
    print("=" * 70)
    if res is None:
        print("  sin caja certificada no hay tasa que usar")
    else:
        print("  El campo contrae a tasa rho en ||.||_P; el forzamiento vuelve")
        print("  por interocepcion -> host -> modulacion con Lipschitz L_F.")
        print("  Condicion de small-gain:  rho + kappa*dt*L_F < 1")
        for LF in (0.01, 0.02, 0.05, 0.1, 0.3):
            v = res["rho"] + res["kappa"] * dt * LF
            print(f"    L_F={LF:<5}: {v:.4f}  "
                  f"{'ISS CERTIFICADO' if v < 1 else 'no cierra'}")
        umbral = (1 - res["rho"]) / res["kappa"]
        print(f"\n  => lazo cerrado ISS si  L_F < {umbral:.5f}")
        print(f"     Cotas duras del codigo: f_gain<=0.3 y g_phi_gain=0.3,")
        print(f"     ambas tras tanh, luego L_F <= 0.3*||W_h||; el certificado")
        print(f"     se sostiene si ||W_h|| < {umbral / 0.3:.5f}.")
