# Numerical methods

All methods integrate forward in time (`t_end >= t0`). Status of each component:

| component | status |
|---|---|
| DOPRI5 (adaptive explicit) | production |
| BDF orders 1-5 + dense LU Newton | production |
| BDF + Newton-Krylov (GMRES, matrix-free) | production for tested systems; preconditioner interface user-supplied |
| automatic explicit -> implicit switching | production (one-way) |
| RK4 fixed step | verification only (no error control) |
| implicit -> explicit switch back | **not implemented** |
| backward integration, DAEs, sparse direct solvers | not implemented |

## Error norm

For each step the scaled RMS norm is used:

```
scale_i = atol_i + rtol * max(|y_old_i|, |y_new_i|)
err     = sqrt( (1/n) sum_i (e_i / scale_i)^2 )        step accepted if err <= 1
```

`atol` may be a scalar or a per-component vector (the dust model uses per-component
values because `r ~ 1-100`, `v_r ~ 1e-7-1e-3`, `z -> 0`). For components that approach
zero the absolute part dominates: relative accuracy is lost once `|y| < atol / rtol`, which
is intended. `rtol < 10 eps` is rejected as invalid input.

The tolerance controls the *local* error. Global errors are typically 10-100 x `rtol`;
for slowly decaying solutions relative errors add up over the steps (bound ~ `n_steps * rtol`).

## DOPRI5

Dormand & Prince 5(4) with FSAL, coefficients as in Hairer-Norsett-Wanner `dopri5.f`;
4th-order continuous extension; PI step-size controller (beta = 0.04, safety 0.9, step
ratio limited to [0.2, 10], no increase right after a rejection); Hairer's initial step
selection. The state update `y_new = y + h sum b_i k_i` uses compensated (Kahan)
summation, which removes the O(N eps) drift over long integrations (tested over 1e6 steps).
Time is advanced as `t_new = t + h` and the step actually used is `h = t_new - t`.

## BDF

Variable-order (1-5), variable-step BDF in the fixed-leading-coefficient form of
Shampine & Reichelt (MATLAB `ode15s` with `kappa = 0`), structured like SciPy's `BDF`:
difference array `D`, step-size change by `D <- (R U)^T D`, local error
`C_{k+1} * d`, order selection from the error estimates of orders k-1, k, k+1 after
`k+1` steps of constant size, step factors in [0.2, 10].

Two fixes beyond the SciPy formulation:

- **Grid consistency.** When `t_new - t` (representable) differs from the intended
  `h`, `D` is rescaled to the actual step. Without this, at tight tolerances and large
  `|t|` the relative mismatch `ulp(t)/h` feeds back into rejections and the step size
  collapses (observed on Van der Pol at `rtol = 1e-10`, `t ~ 1024`; regression test).
- **Invalid Jacobians.** A Jacobian containing NaN/Inf (e.g. evaluated at a predictor
  outside the model domain) is replaced by one at the last accepted state.

### Nonlinear solver

Simplified Newton iteration on `(I - c J) dy = c f(y) - psi - d`, at most 4 iterations,
convergence-rate test `rate^(m-k)/(1-rate) * |dy| > tol` -> abort,
`rate/(1-rate) * |dy| < tol` -> converged, with `tol = max(10 eps/rtol, min(0.03, sqrt(rtol)))`.
On failure the Jacobian is refreshed once (at the predictor), then the step is halved.
Jacobian: analytic if provided (the dust model provides one, see `performance.md` for why
finite differences fail there), otherwise forward differences column by column with the
CVODE-style increment `max(sqrt(eps)|y_j|, 1000 |h| eps n ||f|| * w_j^-1)`.

### Linear solvers

- **Dense LU** with partial pivoting; matrices whose pivot ratio `min|u_ii|/max|u_ii|`
  is below `16 n eps` or which contain non-finite entries are reported singular
  (-> step rejected, after 60 consecutive failures `LinearSolverFailure`).
- **GMRES(m)** (restart 30 by default), modified Gram-Schmidt with one DGKS
  re-orthogonalisation pass, Givens rotations, right preconditioning. Used matrix-free:
  `J v` is analytic (`OdeSystem::jvp`) or `(f(y + e v) - f(y))/e` with
  `e = sqrt(eps)(1 + |y|)/|v|`. The system is solved in the error-weighted space
  `D (I - cJ) D^-1`, `D = diag(1/scale)`, with relative residual 0.05 (inexact Newton).
  This is a true Newton-Krylov iteration (Jacobian at the current iterate).
- Crossover (see `performance.md`): unpreconditioned GMRES beats dense LU only for
  roughly 50 < n < 400 on the diffusion test problem; with a good preconditioner it
  wins by 80x at n = 1600. For the 6-component dust model dense LU is used.

## Stiffness detection and switching (`method = auto`)

Start with DOPRI5. After every accepted step three indicators are evaluated; any of
them flags stiffness, the solver switches to BDF once and records the reason:

1. **Hairer's test**: `h |lambda| > 3.25` (estimate from stages 6/7) for 15 accepted
   steps, reset by 6 non-stiff steps.
2. **Pinned step size**: at least 80 % of the last 50 accepted steps have
   `h |lambda| >= 0.7 * 3.25`. The PI controller tends to settle *below* the stability
   boundary (dust model: `h lambda ~ 2.8` steadily), which criterion 1 never sees.
3. **Spectral radius**: every 25 steps `rho(J)` is estimated with 5 power iterations
   (FD `J v`, ~5 extra RHS calls); 3 consecutive checks with `h rho >= 1` flag stiffness.
   An *active* mode with `h|lambda| >= 1` has local error `~ (h lambda)^6/720` and fails
   the error test for `rtol <~ 1e-3`, so accepted steps at `h rho >= 1` mean the dominant
   modes have decayed. Needed for Van der Pol, where criterion 1's directional estimate
   reads only 0-1.5 while `h rho = 1.5`.

The BDF solver is started from the current state with the current step size, so
tolerances and the physical state are preserved. In addition, if DOPRI5 fails with step
underflow or non-finite values, BDF is tried once (`fallback_to_implicit`).

There is **no switch back** to the explicit method (oscillation between methods is
therefore impossible). Tests: stiff problems switch exactly once, non-stiff problems
(oscillator, exponential, Van der Pol mu = 1) never switch for rtol 1e-4..1e-10.

## Events

`g(t, y) = 0` is checked at each step end and optionally at `event_samples` interior
points (dense output); roots are found with Brent's method to ~2 eps|t|. Terminal events
stop the integration at the root. An even number of roots inside one sub-interval is not
detected. The dust model uses `r - r_in` (terminal) and `z -+ eps z_init` with 4 interior
samples; two monotone functions are used instead of `z^2 - band^2`, which misses fast
passes through the band.

## Floating-point safeguards

| situation | handling |
|---|---|
| NaN/Inf in stages or RHS | step rejected, h /4 (explicit) or /2 (BDF); `NonFiniteState` after repeated failure |
| model domain (`OdeSystem::in_domain`, e.g. r > 0, a > 0) | step rejected, h halved; `DomainViolation` after repeated failure; values are never clipped |
| step below `10 ulp(t)` or user `h_min` | `StepSizeUnderflow` |
| `t + h == t` | `TimeResolutionLost` |
| increments lost (`y + dy == y`, `dy != 0`) | counted (`n_lost_increment`); with Kahan summation the increment is carried; only if *all* increments are lost for `max_stagnant_steps` (1000) consecutive steps -> `TimeResolutionLost` |
| subnormals | not flushed (the dust `z` becomes subnormal after settling; harmless) |
| FP contraction | `-ffp-contract=off` by default, so scalar results do not depend on the compiler's FMA fusion; explicit FMA only in NEON/AVX2 kernels; `-ffast-math` is never used |

The last valid state (`t_last`, `y_last`) is always returned with the failure status
and a message.

## Scaling

The dust equations are integrated in the dimensionless variables of the paper (au, v_K(1au),
t0, a/a_init), which keeps all state components within ~1e-7..1e2 and makes per-component
`atol` meaningful. Outputs are converted back to CGS / au / yr.
