"""
VERIFICACION DE LA PROPOSICION DE VERLET/SCHUR-COHN (el unico resultado del
paper sin precedente hallado, y el unico que no estaba demostrado).

Lo que se comprueba, en tres niveles:

  N1. SIMBOLICO. Que el criterio de Schur-Cohn para el cuadratico complejo
      z^2 + a1 z + a0 con a0=(1-g)-i*mu, a1=-(2-q-g)+i*mu equivale EXACTAMENTE
      a las condiciones (i) y (ii) del paper. Se comprueba que:
        |a0|^2 < 1                      <=>  (i)  mu^2 < g(2-g)
        |a1 - conj(a1) a0| < 1 - |a0|^2 <=>  (ii) q(g^2+mu^2) < 2g(g(2-g)-mu^2)
      y ademas que son NECESARIAS Y SUFICIENTES (el paper las daba como
      suficientes: se infravaloraban).

  N2. ESCALAR NUMERICO. Que el criterio predice |z|<1 sin un solo fallo sobre
      una malla densa de (q,g,mu), comparando contra las raices exactas.

  N3. MATRICIAL NUMERICO. Que la reduccion a escalar por RAIZ LATENTE vale SIN
      diagonalizacion simultanea: se generan K,C,G aleatorias que NO conmutan,
      se construye la companion 2N x 2N, y se comprueba (a) que cada autovalor
      satisface el polinomio escalar con los cocientes de Rayleigh de su propio
      vector latente, y (b) que la version por CAJA es suficiente.

  PYTHONPATH=. python verify_verlet_schurcohn.py
"""
import itertools

import numpy as np
import sympy as sp

rng = np.random.default_rng(20260820)


# ---------------------------------------------------------------- N1
def n1_simbolico():
    q, g, mu = sp.symbols("q g mu", real=True)
    a0 = (1 - g) - sp.I * mu
    a1 = -(2 - q - g) + sp.I * mu

    # Schur-Cohn grado 2 (Cohn): |a0|<1 y |a1 - conj(a1) a0| < 1 - |a0|^2
    cond_A = sp.simplify(sp.expand(sp.Abs(a0) ** 2))          # |a0|^2
    objetivo_i = sp.expand(1 - (g * (2 - g) - mu ** 2))       # 1 - (i)
    ok_i = sp.simplify(cond_A - objetivo_i) == 0

    w = sp.expand(a1 - sp.conjugate(a1) * a0)
    re_w = sp.simplify(sp.re(sp.expand(w)))
    im_w = sp.simplify(sp.im(sp.expand(w)))
    D = g * (2 - g) - mu ** 2                                  # = 1 - |a0|^2
    ok_D = sp.simplify(sp.expand(1 - cond_A) - D) == 0

    # |w|^2 < D^2  <=>  (ii), tras dividir por q>0
    lhs = sp.expand(re_w ** 2 + im_w ** 2 - D ** 2)
    ii = sp.expand(q * (g ** 2 + mu ** 2) - 2 * g * D)         # (ii) < 0
    # lhs deberia ser exactamente q * ii
    ok_ii = sp.simplify(sp.expand(lhs - q * ii)) == 0

    print("=== N1 simbolico ===")
    print(f"  Re(a1 - conj(a1)a0) = {sp.simplify(re_w)}")
    print(f"  Im(a1 - conj(a1)a0) = {sp.simplify(im_w)}")
    print(f"  |a0|^2 = 1 - (i)                      : {ok_i}")
    print(f"  1 - |a0|^2 = g(2-g) - mu^2            : {ok_D}")
    print(f"  |w|^2 - D^2 = q * [(ii) reordenada]   : {ok_ii}")
    return ok_i and ok_D and ok_ii


# ---------------------------------------------------------------- criterio
def criterio(q, g, mu):
    """(i) y (ii) del paper."""
    D = g * (2 - g) - mu ** 2
    return (D > 0) and (q * (g ** 2 + mu ** 2) < 2 * g * D)


def raices(q, g, mu):
    a1 = -(2 - q - g) + 1j * mu
    a0 = (1 - g) - 1j * mu
    return np.roots([1.0, a1, a0])


# ---------------------------------------------------------------- N2
def n2_escalar(n=120):
    print("\n=== N2 escalar numerico ===")
    fallos = 0
    total = 0
    peor = 0.0
    for q in np.linspace(1e-4, 6.0, n):
        for g in np.linspace(1e-4, 1.9, n):
            for mu in np.linspace(-1.5, 1.5, 21):
                total += 1
                pred = criterio(q, g, mu)
                rho = np.max(np.abs(raices(q, g, mu)))
                real = rho < 1.0
                if pred != real:
                    # solo cuenta como fallo si no esta en el borde numerico
                    if abs(rho - 1.0) > 1e-9:
                        fallos += 1
                        peor = max(peor, abs(rho - 1.0))
    print(f"  celdas probadas      : {total}")
    print(f"  discrepancias        : {fallos}")
    print(f"  peor |rho-1| en fallo: {peor:.2e}")
    print("  => el criterio es NECESARIO Y SUFICIENTE, no solo suficiente"
          if fallos == 0 else "  => REVISAR")
    return fallos == 0


# ---------------------------------------------------------------- N3
def companion(K, C, G, dt):
    """Verlet de posicion con terminos de velocidad, en forma companion."""
    N = K.shape[0]
    I = np.eye(N)
    B = 2 * I - dt ** 2 * K - dt * (C + G)
    A_ = -(I - dt * (C + G))
    top = np.hstack([B, A_])
    bot = np.hstack([I, np.zeros((N, N))])
    return np.vstack([top, bot])


def matrices(N, escala=1.0):
    """K simetrica definida positiva, C simetrica definida positiva, G real
    antisimetrica; SIN imponer que conmuten."""
    M1 = rng.normal(size=(N, N))
    K = M1 @ M1.T / N + 0.5 * np.eye(N)
    M2 = rng.normal(size=(N, N))
    C = (M2 @ M2.T / N + 0.2 * np.eye(N)) * escala
    M3 = rng.normal(size=(N, N))
    G = (M3 - M3.T) / 2 * escala
    return K, C, G


def n3_matricial(pruebas=400, N=6):
    print("\n=== N3 matricial numerico (K, C, G que NO conmutan) ===")
    max_res_latente = 0.0
    conmutan = 0
    caja_suf_ok = 0
    caja_aplicable = 0
    for _ in range(pruebas):
        dt = rng.uniform(0.05, 0.9)
        K, C, G = matrices(N, escala=rng.uniform(0.2, 1.5))
        if np.linalg.norm(K @ G - G @ K) < 1e-10:
            conmutan += 1

        Phi = companion(K, C, G, dt)
        vals, vecs = np.linalg.eig(Phi)
        rho = np.max(np.abs(vals))

        # (a) cada autovalor cumple el polinomio escalar con los cocientes de
        #     Rayleigh de su PROPIO vector latente (la mitad superior del
        #     autovector de la companion es el vector latente x)
        for z, v in zip(vals, vecs.T):
            x = v[:N]
            nx = np.linalg.norm(x)
            if nx < 1e-12:
                continue
            x = x / nx
            qq = dt ** 2 * np.real(np.conj(x) @ K @ x)
            gg = dt * np.real(np.conj(x) @ C @ x)
            mm = dt * np.imag(np.conj(x) @ G @ x)
            res = abs(z ** 2 - (2 - qq - gg - 1j * mm) * z + (1 - gg - 1j * mm))
            max_res_latente = max(max_res_latente, res)

        # (b) version por CAJA: rangos de los cocientes de Rayleigh
        q_max = dt ** 2 * np.max(np.linalg.eigvalsh(K))
        g_lo = dt * np.min(np.linalg.eigvalsh(C))
        g_hi = dt * np.max(np.linalg.eigvalsh(C))
        mu_max = dt * np.max(np.abs(np.linalg.eigvals(G).imag))

        # (i) minimo de g(2-g) en el intervalo: en el extremo mas lejano de 1
        gi = min(g_lo * (2 - g_lo), g_hi * (2 - g_hi))
        cond_i = mu_max ** 2 < gi

        # (ii) minimizar F(g) = 2g(g(2-g)-mu^2) - q(g^2+mu^2) sobre [g_lo,g_hi]
        #      F es cubica: minimo en extremos O en punto critico interior
        def F(gv):
            return 2 * gv * (gv * (2 - gv) - mu_max ** 2) \
                - q_max * (gv ** 2 + mu_max ** 2)
        cands = [g_lo, g_hi]
        a, b, c = -6.0, (8 - 2 * q_max), -2 * mu_max ** 2
        disc = b * b - 4 * a * c
        if disc >= 0:
            for r in ((-b + np.sqrt(disc)) / (2 * a),
                      (-b - np.sqrt(disc)) / (2 * a)):
                if g_lo <= r <= g_hi:
                    cands.append(r)
        cond_ii = min(F(gv) for gv in cands) > 0

        if cond_i and cond_ii:
            caja_aplicable += 1
            if rho < 1.0 + 1e-12:
                caja_suf_ok += 1

    print(f"  pruebas                         : {pruebas}  (N={N})")
    print(f"  pares K,G que conmutaban        : {conmutan}  "
          f"(0 = la reduccion NO usa diagonalizacion simultanea)")
    print(f"  max residuo del polinomio latente: {max_res_latente:.2e}")
    print(f"  casos en que la CAJA certifica  : {caja_aplicable}")
    print(f"  de esos, rho<1 de verdad        : {caja_suf_ok}")
    ok = (max_res_latente < 1e-8) and (caja_suf_ok == caja_aplicable)
    print("  => reduccion latente EXACTA y criterio por caja SUFICIENTE"
          if ok else "  => REVISAR")
    return ok


if __name__ == "__main__":
    r1 = n1_simbolico()
    r2 = n2_escalar()
    r3 = n3_matricial()
    print("\n" + "=" * 62)
    print("VEREDICTO:", "TODO VERIFICADO" if (r1 and r2 and r3)
          else "HAY ALGO QUE FALLA")
