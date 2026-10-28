# dustppd

A C++20 ODE solver library (adaptive DOPRI5, variable-order BDF with dense-LU or
Newton-Krylov/GMRES, automatic stiffness detection and switching) and an implementation
of the dust-dynamics model of **V. A. Kobozeva (2025, SPbU)**, *"Numerical modeling of
the dust particle dynamics in the accretion disks of young stars with fossil magnetic
field"*: sedimentation, radial drift and coagulation of a single grain in an
analytic accretion disk. With a CLI, Python bindings, a GUI, tests and benchmarks.

## Quick start

Requirements: CMake >= 3.21, a C++20 compiler (GCC >= 11, Clang >= 14, MSVC 2022),
Ninja (optional), network access on first configure (toml++ and Catch2 are fetched).

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build                                   # C++ tests (~3 s)
./build/dustppd run configs/reference/paper_1m_100au.toml -o ref.csv -s ref.json
```

Output of the reference case (1 m body released at r = 100 au, z = 3H):

```
t_drift        : 304661 yr                 (paper: 306 kyr)
t_settle first : 251.147 yr                (paper: 252 yr)
max |v_r|      : 54.6303 m/s at r = 0.363038 au
```

### Command line

```bash
dustppd run <config.toml> [-o traj.csv] [-s summary.json] [--set section.key=value]...
dustppd sweep <config.toml> --param particle.a_init_cm --values 1e-4,0.1,1,10,100 --threads 8
dustppd traps <config.toml> [--sizes 1e-3,0.1,1] [--rmin 1 --rmax 100]   # trap radii, minimum trapped size
dustppd diffuse <config.toml> --r 100 --a 0.1 --alpha 1e-4 [--paths 20000]  # stochastic vertical mixing
dustppd show-config <config.toml> [--set ...]      # print the effective configuration
dustppd info                                       # SIMD backends, code units
```

Example: a 1 cm grain from 10 au with coagulation, radial drift only:

```bash
./build/dustppd run configs/reference/paper_1m_100au.toml \
  --set particle.a_init_cm=1.0 --set particle.r_init_au=10 \
  --set physics.coagulation=true --set physics.sedimentation=false -o run.csv
```

All configuration keys are documented in `configs/reference/paper_1m_100au.toml` and
`docs/physics_model.md`. Trajectory CSV columns: `t_yr, r_au, z_au, vr_cms, vz_cms,
vphi_cms, a_cm, St, tau_s, stokes, lambda_cm, Re, rho_gcc, T_K, H_au, ur_cms, uphi_cms,
beta, dv_cms`.

## Python

```bash
python -m venv .venv && . .venv/bin/activate
pip install ".[gui,test]"
pytest tests/python
```

```python
import dustppd as dp

res = dp.run(dp.reference_config(), check=True)     # GIL released during the run
s, cols = res["summary"], res["columns"]             # dict of NumPy arrays
print(s["t_drift_yr"], s["stats"]["n_steps"], s["stats"]["switches"])

cfg = dp.reference_config(["particle.r_init_au=10.0"])
summaries = dp.sweep(cfg, "a_init_cm", [1e-4, 1e-2, 1, 100], threads=4)   # parallel

# generic ODEs
r = dp.solve(lambda t, y: -50 * y, (0, 1), [1.0], method="auto", rtol=1e-8, atol=[1e-12])
r = dp.solve_system(dp.robertson(), (0, 4e6), [1, 0, 0], method="bdf", rtol=1e-8,
                    atol=[1e-14, 1e-18, 1e-14])
```

Failures are reported in `status`/`message` (`run(..., check=True)` raises
`SolverError`); exceptions raised inside a Python RHS propagate. Arrays returned to
Python own their memory (trajectories are moved without copying).

## GUI

```bash
pip install .
dustppd-gui            # or: python -m dustppd.gui   [--port 8765] [--no-browser]
```

A local server (standard library only, bound to 127.0.0.1) opens a tab in the default
browser. Tabs:

- **Trajectory**: scenario presets (paper reference case, micron grain with growth,
  dead-zone trap, trap with fossil field, charged grain), all parameters, run/cancel with
  progress, summary cards, four selectable plots (r(t), z(r), v_r, a, St, Re, beta_K,
  Hall, Z, ...), disk profiles with stationary drift and trap markers, solver statistics,
  CSV export, TOML load/save.
- **Mixing**: stochastic vertical mixing (SDE ensemble) against the exact stationary
  solution and the Dubrulle formula.
- **Model and methods**: the equations and a short description of the numerical methods.

Plots are drawn in the page (zoom by dragging, double-click to reset, hover for values,
SVG download); no internet connection is needed. Ctrl/Cmd+Enter starts a run.

## C++ API

```cpp
#include "dpd/ode/integrate.hpp"
struct Decay : dpd::ode::OdeSystem {
    dpd::ode::Index size() const override { return 1; }
    void rhs(double, std::span<const double> y, std::span<double> f) const override { f[0] = -y[0]; }
};
dpd::ode::Options opt;            // method = Auto, rtol, atol, events, t_eval, ...
auto res = dpd::ode::integrate(Decay{}, 0.0, std::vector<double>{1.0}, 5.0, opt);
// res.status, res.t, res.y, res.stats (steps, rejections, RHS, Jacobians, switches ...)
```

Link `dpd::ode` (core) and `dpd::dust` (model). See `docs/architecture.md`.

## Documentation

| file | content |
|---|---|
| `docs/physics_model.md` | equations, units, parameters, mapping to the paper, ambiguities |
| `docs/numerical_methods.md` | methods, error control, stiffness switching, FP safeguards |
| `docs/reference_case.md` | reproducing the paper, **known discrepancies** |
| `docs/validation.md` | generated comparison tables and figures (`scripts/compare_paper.py`) |
| `docs/architecture.md` | code structure and design decisions |
| `docs/testing.md` | test suite, reference data provenance, tolerance policy |
| `docs/performance.md` | benchmarks, SIMD, linear-solver crossover |
| `docs/science_extensions.md` | dust traps with a fossil field, stochastic turbulent mixing, charged grains |
| `docs/gpu.md` | experimental Metal/CUDA ensemble benchmark and conclusions |

## Status and limitations

Production: DOPRI5, BDF (1-5) with dense LU, auto switching explicit -> implicit,
Newton-Krylov/GMRES (tested on the provided systems), event location, the dust model
with the analytic disk. Verification only: RK4.

Extensions (see `docs/science_extensions.md`): self-consistent and tabulated disks with
dust-trap analysis, a stochastic (SDE) solver with vertical turbulent mixing of dust, OML
grain charging and the Lorentz force of the fossil field.

Not implemented: switching back implicit -> explicit, backward integration, DAEs,
the Belmondo disk itself (no data; full profile tables can be loaded),
fragmentation, turbulence, charged grains. Two model parameters are not given in the
paper and were calibrated (`sigma_mol`, `settle_eps`; `docs/physics_model.md`).
The AVX2 kernels are only compiled/tested in x86 CI, not on the development machine.

License: MIT.
