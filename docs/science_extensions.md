# Science extensions: traps, turbulent mixing, charged grains

Three extensions of the Kobozeva (2025) model. All of them are **(I)** additions made
here, not part of the paper. Each section states the physics, how it is validated, the
first results, and what is still missing for a publishable result.

Reproduce every number and figure in this document:

```bash
python scripts/trap_map.py   # -> docs/science/traps.md, trap_map_A_w.png, trap_vs_field.png
python scripts/settling.py   # -> docs/science/settling.md, settling.png
python scripts/hall_map.py   # -> docs/science/charged_grains.md, hall_map.png
```

## 1. Dust traps in disks with a fossil field

### Physics

The disk structure gains two models (`disk.model`):

- `self_consistent`: the paper's analytic disk, but beta_K is computed from Eq. (16)
  instead of being a constant:
  `beta_K = [ -(r/rho) dp/dr + r B_z B_r / (4 pi Sigma) ] / v_K^2`, with `Sigma = rho H`.
  The midplane density can be perturbed by a Gaussian bump or a dead-zone edge,
  `f = 1 + A (1 + tanh((r - r_b)/w)) / 2`. With A > 0 the gas piles up outside r_b,
  which is the inner edge of a dead zone, where a pressure maximum forms (Kretke & Lin
  2007; Dzyurkevich et al. 2010). Steady accretion is kept by `u_r ~ 1/f`.
- `tabulated`: full radial profiles (`r_au, T_K, rho_gcc, ur_cms, H_au` and either
  `beta` or `Bz_G, Br_G`, from which beta follows by Eq. 16). This is the entry point
  for the "Belmondo" MHD solutions; nothing else in the code has to change.

The fossil field is `B_z = B0 (r/au)^-p`, `B_r = q B_z` (`[field]` section), or taken
from the table.

**Trap condition.** At an equilibrium of Eqs. (11) (`v_r = 0`, `dj/dt = 0`) we have
`beta_K v_K = u_r / St`. This is exactly the zero of the stationary drift velocity,
`v_r = (u_r/St - beta_K v_K) / (St + 1/St)` (Eq. 6). Traps are therefore found by root
finding (`drift_zeros`) and confirmed by integrating the ODEs: the particle settles at
the root to 1e-6. The smallest trapped grain follows by bisection in size
(`min_trapped_size`).

### Validation

- With no perturbation and no field, the self-consistent beta_K equals
  `(c_T/v_K)^2 (21/16 + 9/8)`, which is 0.00237 at 1 au. The paper's constant 0.0026
  agrees to 10 %, so its choice is physically consistent.
- The zero-head-wind radius equals the pressure maximum (independent golden-section
  search, 1e-6).
- The ODE equilibrium equals the stationary root (1e-6). Grains below `a_min` drift
  through to `r_in`.
- A finely tabulated copy of the self-consistent disk with B_z, B_r columns reproduces
  beta (Eq. 16 with finite differences) to 2e-3 and the trap radii to 1e-4.
- The magnetic-tension term matches the closed formula to 1e-10.
- The analytic Jacobian, now including all disk derivatives, is tested against central
  differences.

### First results (illustrative disk, edge at 10 au; `docs/science/traps.md`)

- A trap needs a sharp enough edge: A > 0.64 for w = 1 au, A > 4.6 for w = 3 au.
- For A = 2, w = 1 au, grains above about 1 mm are trapped, at 9.9 au (1 mm) to
  10.75 au (1 m). Smaller grains are carried through by the accretion flow.
- **Fossil field.** Magnetic tension raises beta_K. As it grows, the trap moves inward
  and the smallest trapped size increases. Above B0 ~ 2.85 G (B_z(10 au) ~ 0.16 G) the
  trap disappears. Prediction: rings at dead-zone edges should be weaker or absent in
  strongly magnetized disks, with an inward-dragged field (B_z B_r > 0).

### Needed for a paper

- Belmondo profiles (with and without field), including the actual dead-zone
  boundaries. Load them with `disk.model = "tabulated"`.
- The sign and geometry of B_r. Tension with B_z B_r < 0 would deepen traps instead.
- Grain growth with a fragmentation barrier. Traps only matter for grains that can
  grow past `a_min`.

## 2. Vertical turbulent mixing (stochastic)

### Physics and numerics

There is a new core component, `dpd::ode::simulate`: Ito SDEs with diagonal noise and
three schemes (Euler-Maruyama, Milstein, and a linear-implicit Euler that stays stable
when the drift is stiff). Ensembles run in parallel, with one counter-seeded
xoshiro256++ stream per path, so results do not depend on the thread count.

The dust model (`VerticalDiffusion`, `dustppd diffuse`, `dp.vertical_diffusion`) is

```
dz = [ v_s + dD/dz + D d ln rho_g/dz ] dt + sqrt(2D) dW,
v_s = -Omega St(z) z,   D = alpha_t Omega H^2 / (1 + St^2)
```

following Ciesla (2010) and Youdin & Lithwick (2007). St(z) comes from the model's
drag law. In the Epstein regime it grows as `exp(z^2/2H^2)`, which makes the drift
stiff high above the midplane, hence the implicit scheme. The exact stationary state
(zero flux) is `d ln(rho_d/rho_g)/dz = v_s/D`; it is computed by quadrature as the
reference.

### Validation

- Ornstein-Uhlenbeck: the stationary variances of the *discrete* schemes match their
  closed forms, `sigma^2/(theta(2 - theta h))` for EM and `sigma^2/(theta(2 + theta h))`
  for the linear-implicit scheme (within 5 sigma, 200 000 paths). The transient weak
  error decreases with dt.
- Geometric Brownian motion against the exact solution on the same Wiener path: measured
  strong orders 0.5 (EM) and 1.0 (Milstein).
- Dust: the simulated H_d matches the exact stationary solution within statistics plus
  the O(dt) bias in all tested cases. For constant St (Stokes drag) the exact solution
  reproduces Dubrulle et al. (1995) to 1e-6.

### First results (`docs/science/settling.md`, r = 100 au)

- Simulated H_d agrees with the exact solution to about 0.5 % (20 000 paths,
  0.2-0.4 s per point).
- **The Dubrulle formula `H_d = H sqrt(alpha/(alpha + St))` overestimates H_d by up to
  16.5 % when St ~ alpha**, because St grows with height. For strongly settled mm grains
  (H_d/H ~ 0.1) the error is below 1 %. Since H_d ~ sqrt(alpha), an alpha inferred with
  the formula where H_d/H ~ 0.6 is biased by up to ~35 %.
- Inversion: `python scripts/settling.py --invert <h_d/H> --a <cm> --r <au>`. For
  example, h_d/H = 0.1 for 1 mm grains at 100 au gives alpha_t = 9.3e-5.

### Needed for a paper

- Observed dust scale heights of edge-on disks, such as HH 30 and Oph 163131. Take them
  from the papers (Villenave et al. 2020, 2022, among others); none are hard-coded here.
- A size distribution and a radiative-transfer comparison: the mm-wavelength emission
  height is not H_d of a single size.
- Optionally radial diffusion with the same SDE machinery.

## 3. Charged grains and the Lorentz force

### Physics

- The grain charge is the orbital-motion-limited (OML) equilibrium of electron and ion
  collection (Spitzer 1941; Draine & Sutin 1987): `exp(psi) sqrt(m_i/m_e) = 1 - psi`,
  `Z = psi a k T / e^2`, with psi = -3.86 for m_i = 29 amu (HCO+).
- Field lines move with the gas, so the equations of motion gain the force
  `(Z e / m c)(v - u) x B`, with B = (B_r, 0, B_z).
- The force does no work in the gas frame, which is tested exactly. The Jacobian
  includes all Lorentz terms and is tested against finite differences.
- The diagnostic is the Hall parameter `Gamma = (Z e B / m c) tau_s`.

### Limitations, stated in the code and the outputs

- OML neglects electron depletion by grains (the Havnes effect), which only lowers |Z|.
  The computed Lorentz force is therefore an **upper bound**.
- The continuous charge is meaningless for |Z| < 1, i.e. nm grains in the cold outer
  disk; these regions are masked.
- The field is a power law (illustrative).

### First result (`docs/science/charged_grains.md`)

In the disk body the Lorentz force is negligible even with the upper-bound charge:
Gamma_mid ~ 1e-5 to 1e-9 for 0.01-10 um grains at 1 au and B0 = 0.1-1 G. Grains become
tied to the field (Gamma > 1) only above z ~ 5-6 H, in the disk atmosphere. Drift and
settling in the disk interior can therefore ignore grain charge. This is a bounding
result. A positive result needs the disk atmosphere and the ionization model
(temperature inversion, X-ray / UV ionization), which the Dudorov-Khaibrakhmanov model
provides, plus a non-OML charging model with electron depletion.

## Summary of scientific value

| direction | status | strongest result so far | what makes it publishable |
|---|---|---|---|
| traps | ready for real profiles | a fossil field with inward-bent lines weakens and erases dead-zone traps (threshold B_z(10 au) ~ 0.16 G in the illustrative disk) | Belmondo profiles with and without field; growth and fragmentation |
| turbulent mixing | ready | the Dubrulle formula is biased by up to 16.5 % in H_d (~35 % in alpha) where St ~ alpha | comparison with edge-on observations, size distribution |
| charged grains | bounding result | Lorentz force negligible below ~5 H for 0.01-10 um grains | disk atmosphere and ionization model, Havnes-limited charge |
