# Physical model

Source: V. A. Kobozeva, *Numerical modeling of the dust particle dynamics in the accretion
disks of young stars with fossil magnetic field*, final qualification work, SPbU, 2025
(supervisor S. A. Khaibrakhmanov). Equation numbers below refer to that text.

This document separates **(P)** facts stated in the paper from **(I)** implementation
decisions made here. Every implemented term is mapped to its source.

## 1. Problem

A single spherical grain (radius `a`, material density `rho_d`) moves in the gas disk
around a T Tauri star of mass `M`. Forces: stellar gravity (thin-disk approximation),
gas drag, centrifugal and Coriolis forces (P, Sec. 2.1, Eqs. 8-9). The grain may grow
by sweeping up dust (coagulation, P, Eqs. 7, 10).

## 2. State variables and units

| index | symbol | meaning | code unit |
|---|---|---|---|
| 0 | `r` | cylindrical radius | `r0 = 1 au` |
| 1 | `v_r` | radial velocity | `v0 = sqrt(G Msun / r0)` |
| 2 | `j = r v_phi` | specific angular momentum | `j0 = r0 v0` |
| 3 | `z` | height above midplane | `r0` |
| 4 | `v_z` | vertical velocity | `v0` |
| 5 | `a` | grain radius | `a0 = a_init` |

Time unit `t0 = r0 / v0 = 5.0226e6 s = 0.15916 yr` (= 1/(2 pi) yr to 1e-4).

(P) The paper uses `v0 = v_K(1 au) = 2.976e6 cm/s` (Sec. 2.2). (I) We compute
`v0 = sqrt(GM_sun / au)` = 2.97847e6 cm/s from IAU nominal constants so that the
dimensionless gravity is exactly `M/r^2` (difference 0.08 %). `a0` is "a0" in the
paper without a value (I: `a0 = a_init`, so `a(0) = 1`, good scaling).

## 3. Equations of motion (P, Eq. 11)

```
dr/dt   = v_r
dv_r/dt = j^2/r^3 - M/r^2 - (v_r - u_r)/tau_s
dj/dt   = -(r/tau_s) (j/r - u_phi)
dz/dt   = v_z
dv_z/dt = -M z/r^3 - v_z/tau_s
da/dt   = Y_d rho r0 / (4 rho_d a0) * sqrt((v_r-u_r)^2 + (v_phi-u_phi)^2 + v_z^2)   (Eq. 10)
```

`M` = stellar mass in Msun (paper: M = 1, so the gravity terms read `1/r^2`, `z/r^3`).
Implemented in `src/dust/model.cpp`, `DustModel::rhs`.

Initial conditions (P, Sec. 2.2/2.6): `r = r_init`, `z = 3 H(r_init)`, `v_r = v_z = 0`,
`j = r_init v_K(r_init)` (Keplerian), `a = a_init`.

Options (I): `sedimentation = false` freezes `z = v_z = 0` (radial drift only; this is what
the paper's `jord++` does for the coagulation runs, Sec. 2.5); `coagulation = false`
sets `da/dt = 0`.

## 4. Gas drag (P, Sec. 1.4)

| quantity | formula | source |
|---|---|---|
| thermal speed | `v_th = sqrt(8 k T / (pi mu m_H))` | p. 15 |
| Epstein stopping time | `tau_Ep = rho_d a / (rho v_th)` | Eq. (1) |
| dynamic viscosity | `eta = v_th lambda rho / 3` | p. 16 |
| Stokes stopping time (Re < 1) | `tau_St = (2/9) rho_d a^2 / eta` | Eq. (2) |
| regime switch | Stokes for `a / lambda >= 3/2` | p. 16 |
| mean free path | `lambda = mu m_H / (rho sigma_mol)` | (I), see A3 |

`tau_St / tau_Ep = 2a / (3 lambda)`, so the switch is continuous in `tau`. Drag law is
selectable (`epstein`, `stokes`, `both`). The Reynolds number `Re = 2 a rho |dv| / eta`
is diagnosed; a warning is issued when the Stokes law is used with `Re > 1` (the paper
lists the higher-Re drag coefficients but uses only Eq. (2)).

## 5. Disk model (P, Sec. 2.3)

Analytic solution of the Dudorov-Khaibrakhmanov model, with
`alpha_0.01 = alpha/0.01`, `Mdot_8 = Mdot / 1e-8 Msun/yr`, `M = M/Msun`, `r` in au:

```
T     = 240      alpha^-1/4 Mdot^1/2 M^3/8   r^-9/8   K          (Eq. 12)
rho_0 = 2.5e-10  alpha^-5/8 Mdot^1/4 M^7/16  r^-21/16 g/cm^3     (Eq. 13)
u_r   = -30      alpha^3/4  Mdot^1/2 M^1/8   r^-5/8   cm/s
H     = 0.03     alpha^-1/8 Mdot^1/4 M^-5/16 r^15/16  au
rho(r, z) = rho_0(r) exp(-z^2 / (2 H^2))                          (Eq. 19)
u_phi = v_K sqrt(1 - beta_K)                                      (Eq. 5)
```

Analytic disk: constant `beta_K = 1 - (u_phi/v_K)^2` with `u_phi/v_K = 0.9987`
(P, Sec. 2.6, Table 5 `vphi2vk`), i.e. `beta_K = 0.0025983`.

Further disk models (I): `self_consistent` (beta_K from Eq. 16, i.e. gas pressure gradient
plus magnetic tension, with optional pressure bumps or a dead-zone edge) and `tabulated`
(full radial profiles, e.g. Belmondo output). See `docs/science_extensions.md`.

Tabulated `beta_K(r)` (I): a CSV of `(r_au, beta)` is interpolated linearly in `log r`
and held constant outside its range (flagged in the run summary). The structure
`T, rho, u_r, H` stays analytic in this mode.

## 6. Parameters (reference values, P, Sec. 2.6 and Table 5)

| parameter | value | source |
|---|---|---|
| `a_init` | 1e-4, 0.1, 1, 10, 100 cm | Table 5 |
| `rho_d` | 3 g/cm^3 | Sec. 2.6 |
| `Y_d` | 0.01 | Sec. 2.6 |
| `M` | 1 Msun | Sec. 2.6 |
| `alpha` | 0.01 | Sec. 2.6 |
| `Mdot` | 1e-8 Msun/yr | Sec. 2.6 |
| `u_phi/v_K` | 0.9987 | Sec. 2.6 |
| `r_init` | 1, 10, 50, 100 au | Sec. 2.6 |
| `z_init` | 3 H | Sec. 2.6 |
| `r_in` | 0.05 au (0.05-0.1) | Sec. 2.6, 3.4 |
| `mu` | 2.3 | (I) A2 |
| `sigma_mol` | 1e-16 cm^2 | (I) A3 |
| `settle_eps` | 0.005 | (I) A4 |

Physical constants live only in `include/dpd/dust/constants.hpp`.

## 7. Derived diagnostics

- **Drift time** `t_drift`: time to reach `r = r_in` (terminal event, P Sec. 3.3).
- **Settling time**: the paper does not define it. Two event-based diagnostics are
  reported: *first entry* into `|z| <= eps z_init` and *last entry* (time after which `|z|`
  stays in the band). See A4.
- `St = tau_s Omega_K`, `max |v_r|` and the radius where it occurs, `a_final`, `a_max`,
  maximum `Re` in the Stokes regime.

## 8. Ambiguities and how they were resolved

**A1. Stationary drift formula.** (P) Eq. (26) writes `tau_s` where Eq. (6) writes `St`;
the derivation uses `d(r v_K)/dr = v_K/2` which holds only at `r = 1`. The two coincide
at 1 au. (I) The ODE system (11) is integrated as stated; the analytic check in the tests
uses the correct form `v_r = (u_r/St - beta v_K)/(St + 1/St)`, `St = tau_s Omega_K`.

**A2. Mean molecular weight.** Not stated. `mu = 2.3` reproduces Eq. (3) exactly
(`tau_Ep = 8.1 s` for 1 um at 1 au) and `H = c_T/Omega_K = 0.031 au`.

**A3. Mean free path / collision cross-section.** Not stated, but it decides the
Epstein-Stokes switch. Two independent statements of the paper fix it: the switch for a
1 m body occurs at `r ~ 0.5 au` (Sec. 3.4) and `a(tau_s = 1) = 127.7 cm` in the Stokes
regime at 1 au (Sec. 2.4). Both require `lambda(1 au, midplane) ~ 1.6-1.7 m`, i.e.
`sigma_mol ~ 1e-16 cm^2` with `lambda = mu m_H/(rho sigma)`. With this value the four
1 m drift times of Table 2 agree within 2.5 %. The standard H2 value `2e-15 cm^2` gives
`lambda ~ 8 cm` and contradicts both statements (e.g. 1 m from 1 au drifts in 1240 yr
instead of 98 yr). The parameter is configurable (`physics.sigma_mol_cm2`).

**A4. Settling criterion.** Not stated. With `eps = 0.005` the first-entry time matches
6 of the 8 Table 1 values within 3 % (and the 1 m at 1 au value matches the last-entry
time), so the paper apparently stops at the first entry into a band of 0.5 % of
`z_init`; for fast oscillating bodies a sampled check can miss the band, which explains
why 1 m at 1 au corresponds to the final settling.

**A5. Inner edge.** 0.05 au (used for the reference case, Sec. 3.4); configurable.

**A6. Numerical disk ("Belmondo") tables** are not available. The GD/MHD rows of Table 2
and Figs. 4, 7 cannot be reproduced; `disk.model = "tabulated_beta"` accepts such data
when available.

## 9. Validity limits

- Thin disk (`z^2 << r^2`), vertically isothermal, gas `u_z = 0` (P).
- No turbulence, no fragmentation, no charging, no size distribution (P, Sec. 1.5).
- Stokes law valid only for `Re < 1` (warning otherwise).
- Coagulation Eq. (10) assumes perfect sticking; the paper notes velocities above
  1-10 m/s should fragment. No fragmentation barrier is applied (as in the paper).
- `z` decays exponentially and becomes subnormal (`~1e-300`) after settling; this is
  physical (no clamping) and harmless.

See `docs/validation.md` for the comparison and `docs/reference_case.md` for the
reproducibility instructions and the list of discrepancies.
