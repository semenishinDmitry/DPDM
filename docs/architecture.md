# Architecture

```
include/dpd/ode/   numerical core (no physics, no I/O, no Python)
  types.hpp          Real, DenseMatrix, Method, Status, Stats
  system.hpp         OdeSystem (rhs, optional jacobian/jvp, in_domain), Preconditioner
  options.hpp        Options (tolerances, method, switching, Newton/GMRES, events, output)
  kernels.hpp        vector kernels + backend dispatch (scalar/generic/NEON/AVX2)
  linalg.hpp         DenseLU, Gmres, FunctionRef
  jacobian.hpp       finite-difference Jacobian and J*v
  stepper.hpp        Stepper interface; Dopri5, Rk4, Bdf
  integrate.hpp      integrate(): output, events, auto switching
  test_problems.hpp  Robertson, Van der Pol, reaction-diffusion, failure problems
include/dpd/dust/  physical model
  constants.hpp      the only place with physical constants
  disk.hpp           analytic disk (+ tabulated beta_K)
  model.hpp          DustModel : OdeSystem, drag laws, diagnostics
  config.hpp         RunConfig <-> TOML
  run.hpp            run(), run_ensemble(), summaries (events -> drift/settling times)
  io.hpp             CSV / JSON output
apps/              dustppd CLI
python/            pybind11 module dustppd._core + package (gui.py: browser GUI server, web/: page)
tests/cpp, tests/python, tests/reference (independent SciPy model, reference generator)
bench/             benchmark harness
scripts/           compare_paper.py
```

CMake targets: `dpd_ode` (core), `dpd_dust` (model + config), `dustppd` (CLI),
`dpd_tests`, `dpd_alloc_audit`, `dpd_bench`, `_core` (Python, `DPD_BUILD_PYTHON=ON`).

## Key decisions

- **Runtime polymorphism for problems (`OdeSystem` virtual interface)**, not templates:
  a virtual call per RHS evaluation is negligible next to the RHS cost (the dust RHS is
  17 ns with several `exp/log/sqrt`), it keeps the library a normal compiled library,
  and Python-defined systems plug in the same way.
- **Steppers own all workspace**, allocated once in the constructor; `integrate()`
  allocates only for output and events. The allocation audit
  (`tests/cpp/test_alloc.cpp`, global `operator new` counter) verifies that the number of
  allocations does not depend on the number of steps.
- **Counters live in `Stats`**, updated through `StepContext`; every result carries
  steps, rejections (by cause), RHS/Jacobian/LU/Newton/GMRES counts, h range, switches.
- **Failures are statuses, not exceptions.** Exceptions only for invalid configuration
  (C++/Python) and exceptions raised by user callbacks (propagated to Python).
- **Physics separate from numerics**: `dpd_dust` depends on `dpd_ode`, never the other
  way. The GUI and Python live outside both.
- **Thread-compatibility**: `OdeSystem::rhs` is `const`; ensembles run one integration
  per thread with no shared mutable state (results are bitwise identical to serial runs,
  tested). Cancellation/progress use atomics.
- **Dependencies** (pinned, fetched by CMake): toml++ 3.4.0 (MIT, parsing only),
  Catch2 3.7.1 (BSL, tests). Python: pybind11, numpy; matplotlib only for the scripts. The GUI uses the standard library.

## Adding another ODE system

Implement `OdeSystem` (`size`, `rhs`, optionally `jacobian`/`jvp`/`in_domain`) and call
`dpd::ode::integrate`. For large stiff systems pass `linear_solver = Gmres` and,
ideally, a `Preconditioner`. From Python, `dustppd.solve(fun, t_span, y0, ...)` accepts
any callable (slower: the GIL is re-acquired for every RHS call).
