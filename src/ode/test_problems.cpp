#include "dpd/ode/test_problems.hpp"

#include <numbers>

namespace dpd::ode::problems {

void LinearSystem::rhs(Real, std::span<const Real> y, std::span<Real> f) const {
    const Index n = A_.rows();
    for (Index i = 0; i < n; ++i) {
        Real s = 0;
        for (Index j = 0; j < n; ++j) s += A_(i, j) * y[j];
        f[i] = s;
    }
}

void Robertson::rhs(Real, std::span<const Real> y, std::span<Real> f) const {
    const Real r1 = 0.04 * y[0];
    const Real r2 = 1e4 * y[1] * y[2];
    const Real r3 = 3e7 * y[1] * y[1];
    f[0] = -r1 + r2;
    f[1] = r1 - r2 - r3;
    f[2] = r3;
}

void Robertson::jacobian(Real, std::span<const Real> y, DenseMatrix& J) const {
    J(0, 0) = -0.04;
    J(0, 1) = 1e4 * y[2];
    J(0, 2) = 1e4 * y[1];
    J(1, 0) = 0.04;
    J(1, 1) = -1e4 * y[2] - 6e7 * y[1];
    J(1, 2) = -1e4 * y[1];
    J(2, 0) = 0;
    J(2, 1) = 6e7 * y[1];
    J(2, 2) = 0;
}

void ReactionDiffusion1D::rhs(Real, std::span<const Real> u, std::span<Real> f) const {
    const Real c = d_ / (dx_ * dx_);
    for (Index i = 0; i < n_; ++i) {
        const Real ul = i > 0 ? u[i - 1] : Real{0};
        const Real ur = i + 1 < n_ ? u[i + 1] : Real{0};
        f[i] = c * (ul - 2 * u[i] + ur) - k_ * u[i] * u[i];
    }
}

void ReactionDiffusion1D::jacobian(Real, std::span<const Real> u, DenseMatrix& J) const {
    const Real c = d_ / (dx_ * dx_);
    J.fill(0);
    for (Index i = 0; i < n_; ++i) {
        J(i, i) = -2 * c - 2 * k_ * u[i];
        if (i > 0) J(i, i - 1) = c;
        if (i + 1 < n_) J(i, i + 1) = c;
    }
}

void ReactionDiffusion1D::jvp(Real, std::span<const Real> u, std::span<const Real> v, std::span<Real> Jv) const {
    const Real c = d_ / (dx_ * dx_);
    for (Index i = 0; i < n_; ++i) {
        const Real vl = i > 0 ? v[i - 1] : Real{0};
        const Real vr = i + 1 < n_ ? v[i + 1] : Real{0};
        Jv[i] = c * (vl - 2 * v[i] + vr) - 2 * k_ * u[i] * v[i];
    }
}

std::vector<Real> ReactionDiffusion1D::initial_condition() const {
    std::vector<Real> u(n_);
    for (Index i = 0; i < n_; ++i) u[i] = std::sin(std::numbers::pi * dx_ * static_cast<Real>(i + 1));
    return u;
}

TridiagonalRDPreconditioner::TridiagonalRDPreconditioner(Index n, Real d, Real k)
    : n_(n), d_(d), k_(k), dx_(Real{1} / static_cast<Real>(n + 1)), a_(n), b_(n), cc_(n), cp_(n), dp_(n) {}

void TridiagonalRDPreconditioner::setup(Real, std::span<const Real> y, Real c) {
    const Real s = d_ / (dx_ * dx_);
    for (Index i = 0; i < n_; ++i) {
        a_[i] = -c * s;
        cc_[i] = -c * s;
        b_[i] = 1 - c * (-2 * s - 2 * k_ * y[i]);
    }
    ++setups_;
}

void TridiagonalRDPreconditioner::apply(std::span<const Real> r, std::span<Real> z) const {
    // Thomas algorithm (diagonally dominant for c >= 0, u >= 0).
    cp_[0] = cc_[0] / b_[0];
    dp_[0] = r[0] / b_[0];
    for (Index i = 1; i < n_; ++i) {
        const Real m = b_[i] - a_[i] * cp_[i - 1];
        cp_[i] = cc_[i] / m;
        dp_[i] = (r[i] - a_[i] * dp_[i - 1]) / m;
    }
    z[n_ - 1] = dp_[n_ - 1];
    for (Index i = n_ - 1; i-- > 0;) z[i] = dp_[i] - cp_[i] * z[i + 1];
}

}  // namespace dpd::ode::problems
