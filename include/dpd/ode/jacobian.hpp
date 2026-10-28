// Finite-difference Jacobians and Jacobian-vector products.
#pragma once

#include "dpd/ode/system.hpp"

namespace dpd::ode {

// Forward-difference dense Jacobian, column by column.
// Perturbation: delta_j = sqrt(eps) * max(|y_j|, typ_j), then rounded so that
// (y_j + delta_j) - y_j is exactly representable. `typ` provides a typical
// magnitude per component (e.g. atol/rtol) so components near zero are still
// perturbed meaningfully. f0 = f(t, y) must be supplied.
// work_y, work_f: caller-provided scratch of size n. Returns RHS evaluations used.
std::uint64_t fd_jacobian(const OdeSystem& sys, Real t, std::span<const Real> y, std::span<const Real> f0,
                          std::span<const Real> typ, DenseMatrix& J, std::span<Real> work_y,
                          std::span<Real> work_f);

// Directional derivative J(t,y) v ~= (f(t, y + e v) - f(t, y)) / e with
// e = sqrt(eps) * (1 + ||y||_2) / ||v||_2 (Brown & Saad 1990). f0 = f(t, y).
// Returns RHS evaluations used (0 if v == 0).
std::uint64_t fd_jvp(const OdeSystem& sys, Real t, std::span<const Real> y, std::span<const Real> f0,
                     std::span<const Real> v, std::span<Real> Jv, std::span<Real> work_y);

}  // namespace dpd::ode
