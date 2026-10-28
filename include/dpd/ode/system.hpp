// Problem definition interface. Every ODE solved by the library implements
// OdeSystem; numerical methods depend only on this interface.
#pragma once

#include "dpd/ode/types.hpp"

namespace dpd::ode {

class OdeSystem {
public:
    virtual ~OdeSystem() = default;

    [[nodiscard]] virtual Index size() const = 0;

    // dydt = f(t, y). Must not allocate; must be thread-compatible (const).
    virtual void rhs(Real t, std::span<const Real> y, std::span<Real> dydt) const = 0;

    // Optional analytic Jacobian J(i,j) = d f_i / d y_j (row-major, n x n).
    [[nodiscard]] virtual bool has_jacobian() const { return false; }
    virtual void jacobian(Real /*t*/, std::span<const Real> /*y*/, DenseMatrix& /*J*/) const {}

    // Optional analytic Jacobian-vector product Jv = J(t, y) v.
    [[nodiscard]] virtual bool has_jvp() const { return false; }
    virtual void jvp(Real /*t*/, std::span<const Real> /*y*/, std::span<const Real> /*v*/,
                     std::span<Real> /*Jv*/) const {}

    // Physical admissibility of a state (e.g. positive radius). A step whose
    // result is outside the domain is rejected and retried with a smaller h;
    // it is never silently clipped.
    [[nodiscard]] virtual bool in_domain(Real /*t*/, std::span<const Real> /*y*/) const { return true; }
};

// Optional preconditioner for Newton-Krylov: z ~= (I - c J)^{-1} r.
// setup() is called whenever the iteration matrix changes (new c or new J).
class Preconditioner {
public:
    virtual ~Preconditioner() = default;
    virtual void setup(Real t, std::span<const Real> y, Real c) = 0;
    virtual void apply(std::span<const Real> r, std::span<Real> z) const = 0;
};

}  // namespace dpd::ode
