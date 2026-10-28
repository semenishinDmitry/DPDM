"""Independent re-implementation of Kobozeva (2025) Eqs. (11) in NumPy/SciPy.

Shares no code with the C++ library: it re-derives the disk structure (Eqs. 12-13),
drag laws (Eqs. 1-2) and the dimensionless equations of motion directly from the
paper, and integrates them with SciPy's LSODA/Radau. Used to generate
cross-check values (tests/cpp/reference_data.hpp) and by tests/python.

Agreement between this script and the C++ code verifies the *implementation*;
both share the interpretation choices documented in docs/physics_model.md.
"""
import numpy as np
from scipy.integrate import solve_ivp

AU = 1.495978707e13
GM_SUN = 1.3271244e26
KB = 1.380649e-16
MP = 1.67262192369e-24
YR = 3.15576e7
V0 = np.sqrt(GM_SUN / AU)
T0 = AU / V0


def make_rhs(a_cm, beta=1 - 0.9987**2, mu=2.3, sigma=1e-16, drag="both", coag=False, sed=True,
             Yd=0.01, rhod=3.0):
    def T(r):
        return 240.0 * r ** (-9 / 8)

    def H(r):
        return 0.03 * r ** (15 / 16)

    def rho(r, z):
        return 2.5e-10 * r ** (-21 / 16) * np.exp(-z * z / (2 * H(r) ** 2))

    def ur(r):
        return -30.0 * r ** (-5 / 8) / V0

    def tau(r, z, a):
        rh = rho(r, z)
        vth = np.sqrt(8 * KB * T(r) / (np.pi * mu * MP))
        te = rhod * a / (rh * vth)
        lam = mu * MP / (rh * sigma)
        eta = vth * lam * rh / 3
        ts = 2 / 9 * rhod * a * a / eta
        stokes = drag == "stokes" or (drag == "both" and a >= 1.5 * lam)
        return (ts if stokes else te) / T0, rh

    def f(t, y):
        r, vr, j, z, vz, ah = y
        a = ah * a_cm
        zz = z if sed else 0.0
        ts, rh = tau(r, zz, a)
        uph = np.sqrt(1 - beta) / np.sqrt(r)
        u = ur(r)
        vph = j / r
        dvz = vz if sed else 0.0
        da = Yd * rh * AU / (4 * rhod * a_cm) * np.sqrt((vr - u) ** 2 + (vph - uph) ** 2 + dvz**2) if coag else 0.0
        return [vr, j * j / r**3 - 1 / r**2 - (vr - u) / ts, -r / ts * (vph - uph),
                vz if sed else 0.0, (-zz / r**3 - vz / ts) if sed else 0.0, da]

    def y0(r_init, z_H=3.0):
        z0 = z_H * H(r_init) if sed else 0.0
        return [r_init, 0.0, np.sqrt(r_init), z0, 0.0, 1.0]

    return f, y0


def drift_time_yr(a_cm, r_init, rin=0.05, rtol=1e-10, method="LSODA", **kw):
    f, y0 = make_rhs(a_cm, **kw)

    def ev(t, y):
        return y[0] - rin
    ev.terminal = True
    ev.direction = -1
    sol = solve_ivp(f, [0, 1e12], y0(r_init), method=method, rtol=rtol,
                    atol=[1e-13, 1e-17, 1e-13, 1e-13, 1e-17, 1e-13], events=ev)
    assert sol.status == 1, sol.message
    return sol.t_events[0][0] * T0 / YR, sol.y_events[0][0]
