# Performance

Measured with `./build/bench/dpd_bench` (Release, `-O3 -ffp-contract=off`), Apple M5
(10 cores), macOS 27, Apple clang 21. Best of 3-7 repetitions after a warm-up. Wall times
are machine-specific; accuracy is reported next to every timing.

## Time to reproduce the paper

All 40 runs behind `docs/validation.md` (Tables 1, 2, 4, Fig. 12, reference case):

| | serial | 10 threads |
|---|---|---|
| before the analytic Jacobian | 1.73 s | 1.62 s |
| now | **0.10 s** | **0.02 s** |

Before, one run took 1.63 s of the total: 1 um grain from 1 au with coagulation
(Table 4). The coagulation rate depends on `|v - u|`, which for micron grains is ~4e-9 in
code units, smaller than the finite-difference perturbation of `j` (~1.5e-8). The FD
Jacobian of that row was therefore wrong and Newton failed 1.9 million times
(1.5 M steps). With the analytic Jacobian (`DustModel::jacobian`, tested against
central differences) the run needs 576 steps and 0.9 ms; the result is unchanged
(7049.03 yr). The slowest remaining run is the 1 m body from 100 au with sedimentation
(~8 ms, 12 000 steps: weakly damped vertical oscillations).

## Dust model, complete runs

Error = relative error of `t_drift` vs BDF at rtol = 1e-12.

| case | method | rtol | wall | rel. error | steps | RHS | switch |
|---|---|---|---|---|---|---|---|
| 1 um from 10 au (stiff) | auto | 1e-8 | 0.60 ms | 1.6e-8 | 683 | 3587 | yes |
| 1 um from 10 au | bdf | 1e-8 | 0.69 ms | 2.0e-8 | 767 | 3540 | - |
| 1 um from 10 au | dopri5 | - | not runnable (~1e11 steps) | | | | |
| 1 cm from 10 au | auto | 1e-8 | 1.09 ms | 6.8e-8 | 1315 | 8046 | yes |
| 1 cm from 10 au | bdf | 1e-8 | 1.20 ms | 7.0e-8 | 1245 | 6728 | - |
| 1 cm from 10 au | dopri5 | 1e-8 | 1650 ms | 2.3e-9 | 3.5 M | 21 M | - |
| 1 m from 100 au (reference) | auto | 1e-8 | 7.3 ms | 4.1e-8 | 10671 | 47817 | yes |
| 1 m from 100 au | bdf | 1e-8 | 10.4 ms | 4.1e-8 | 14254 | 44177 | - |
| 1 m from 100 au | dopri5 | 1e-8 | 11.8 ms | 1.7e-10 | 25547 | 154979 | - |

The paper reports that its code stalled for micron and millimetre grains; with stiffness
detection these runs take under 1 ms. RHS evaluation: 17.4 ns (57 M evaluations/s).

## Classic problems

| problem | method | rtol | wall | rel. error | steps |
|---|---|---|---|---|---|
| Robertson, T = 4e6 | bdf | 1e-9 | 0.39 ms | 2.1e-8 | 1277 |
| Robertson | auto | 1e-9 | 0.35 ms | 2.2e-8 | 1211 |
| Van der Pol mu = 1000, T = 3000 | bdf | 1e-6 | 266 ms | 2.9e-3 | 1.28 M |
| Van der Pol mu = 1000 | auto | 1e-6 | 630 ms | 2.9e-3 | 1.28 M |
| harmonic oscillator, T = 100 | dopri5 / auto | 1e-10 | 0.30 ms | 7e-10 | 3390 |

Van der Pol mu = 1000 on [0, 3000] needs ~1.2 M BDF steps at rtol 1e-6; SciPy's BDF
needs 1.23 M on the same problem, so this is a property of the formulation, not of the
implementation. Auto is slower there because the explicit phase before the switch is long
(the spectral-radius check fires only once `h rho >= 1`).

## Linear solvers in BDF (1-D reaction-diffusion, T = 0.1, rtol 1e-6)

| n | dense LU | GMRES | GMRES + tridiagonal preconditioner |
|---|---|---|---|
| 50 | 0.22 ms | 0.15 ms | 0.10 ms |
| 200 | 3.2 ms | 2.5 ms | 0.36 ms |
| 800 | 58 ms | 71 ms | 1.45 ms |
| 1600 | 250 ms | 366 ms | 2.9 ms |

Max difference between LU and preconditioned GMRES solutions: <= 1e-9. Without a
preconditioner Krylov iterations grow ~ n and dense LU wins again beyond n ~ 400.

## Ensembles

64 grain sizes (1 um - 1 m) from 10 au: 1 thread 96 ms, 4 threads 28 ms (3.4x),
10 threads 18 ms (5.4x; individual runs are 0.3-15 ms, so load balance limits scaling).

## SIMD

Kernels (ns per call, NEON vs compiler-vectorized "generic" vs scalar with
vectorization disabled):

| n | kernel | scalar | generic | NEON |
|---|---|---|---|---|
| 6 | lincomb (6 terms) | 11.4 | 10.9 | 6.3 |
| 1024 | lincomb | 1733 | 621 | 570 |
| 1024 | wrms_err | 644 | 408 | 235 |
| 65536 | wrms_err | 42411 | 26125 | 16482 |
| 65536 | dot | 38163 | 10490 | 13919 |

Honest summary: the compiler already auto-vectorizes the generic loops well; explicit
NEON gains ~1.1-1.8x on norms and multi-term combinations and is slower than generic for
`dot` at large n. **For the 6-component dust model SIMD gives no measurable end-to-end
benefit** - the cost is in the RHS (transcendental functions). The useful parallelism for
this problem is across trajectories (ensembles). The AVX2 path is selected at runtime
only after a CPU check (`__builtin_cpu_supports`), compiled with function-level target
attributes, and is not benchmarked here (ARM host). Agreement of all backends with the
scalar reference is tested.

## Memory

No allocation in stepping loops (allocation audit: the same number of allocations for
runs differing by 100x in steps). Output storage grows with the saved steps;
`save_steps = false` with `t_eval` keeps memory constant.
