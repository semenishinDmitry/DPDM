// Solver configuration.
#pragma once

#include <atomic>
#include <functional>

#include "dpd/ode/types.hpp"

namespace dpd::ode {

class Preconditioner;

struct Options {
    Method method = Method::Auto;

    // Error control: scale_i = atol_i + rtol * max(|y_old_i|, |y_new_i|).
    // atol has either one entry (applied to all components) or n entries.
    Real rtol = 1e-6;
    std::vector<Real> atol{1e-9};

    Real h0 = 0;          // initial step; 0 = automatic (Hairer's algorithm)
    Real h_max = kInf;    // |h| upper bound
    Real h_min = 0;       // |h| lower bound (in addition to round-off bound)
    std::uint64_t max_steps = 50'000'000;

    // RK4 only: fixed step size (required, > 0).
    Real rk4_step = 0;

    // Explicit stiffness detection (Hairer & Wanner, DOPRI5): stiffness is
    // flagged after `stiff_detect_steps` accepted steps with h*|lambda| > stiff_hlamb
    // (interrupted only by `nonstiff_reset_steps` consecutive non-stiff steps).
    int stiff_detect_steps = 15;
    int nonstiff_reset_steps = 6;
    Real stiff_hlamb = 3.25;
    // Second criterion (stability-limited steps): stiffness is also flagged when at
    // least `stiff_window_fraction` of the last `stiff_window` accepted steps had
    // h*|lambda| >= stiff_boundary_fraction * stiff_hlamb, i.e. the step size is
    // pinned near the explicit stability boundary by the controller.
    int stiff_window = 50;
    Real stiff_window_fraction = 0.8;
    Real stiff_boundary_fraction = 0.7;
    // Third criterion (spectral radius): every `stiff_rho_interval` accepted steps
    // rho(J) is estimated by `stiff_rho_iterations` power iterations with
    // finite-difference J*v products; stiffness is flagged after
    // `stiff_rho_checks` consecutive checks with h * rho >= stiff_rho_threshold.
    // Rationale: an *active* mode with h|lambda| >= 1 has local error ~ (h lambda)^6/720
    // and cannot pass the error test for rtol <~ 1e-3, so repeatedly accepted steps
    // with h*rho >= 1 imply the dominant modes have decayed (stiffness). 0 disables.
    int stiff_rho_interval = 25;
    int stiff_rho_iterations = 5;
    int stiff_rho_checks = 3;
    Real stiff_rho_threshold = 1.0;
    // Auto mode: if the explicit method fails (step underflow / too many
    // rejections) fall back to BDF once instead of failing.
    bool fallback_to_implicit = true;

    // BDF / Newton
    int bdf_max_order = 5;
    int newton_max_iter = 4;
    Real newton_tol = 0;  // 0 = automatic: max(10 eps / rtol, min(0.03, sqrt(rtol)))
    LinearSolverKind linear_solver = LinearSolverKind::DenseLU;
    int gmres_restart = 30;
    int gmres_max_iter = 200;
    Real gmres_rel_tol = 0.05;  // inexact Newton forcing term (relative residual)
    Preconditioner* preconditioner = nullptr;  // optional, not owned

    // Event detection: g is checked for sign changes at the step end and, if
    // event_samples > 0, additionally at that many equally spaced interior
    // points of each step (dense output). Sign changes inside a sub-interval
    // with an even number of roots are still not detectable.
    int event_samples = 0;

    // Output
    bool save_steps = true;   // record every accepted step
    std::vector<Real> t_eval; // additionally/alternatively record at these times (dense output)
    // Lost-increment detection: also fail (Status::TimeResolutionLost) when
    // all non-zero increments are lost for this many consecutive steps. 0 = never.
    std::uint64_t max_stagnant_steps = 1000;

    // Cooperative cancellation and progress (both optional, not owned).
    const std::atomic<bool>* cancel = nullptr;
    std::atomic<double>* progress_t = nullptr;  // updated with current t after each step
    // Optional observer called after every accepted step with (t, y) (not in
    // the stepper hot loop; must not throw).
    std::function<void(Real, std::span<const Real>)> on_step;
};

}  // namespace dpd::ode
