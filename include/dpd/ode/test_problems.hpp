// Standard ODE test problems with known solutions or established references.
// Used by the test suite, the benchmarks and the Python bindings.
#pragma once

#include <cmath>

#include "dpd/ode/system.hpp"

namespace dpd::ode::problems {

// y_i' = lambda_i y_i  ->  y_i(t) = y_i(0) exp(lambda_i t)
class ExponentialSystem final : public OdeSystem {
public:
    explicit ExponentialSystem(std::vector<Real> lambda) : lambda_(std::move(lambda)) {}
    [[nodiscard]] Index size() const override { return lambda_.size(); }
    void rhs(Real, std::span<const Real> y, std::span<Real> f) const override {
        for (Index i = 0; i < lambda_.size(); ++i) f[i] = lambda_[i] * y[i];
    }
    [[nodiscard]] bool has_jacobian() const override { return true; }
    void jacobian(Real, std::span<const Real>, DenseMatrix& J) const override {
        J.fill(0);
        for (Index i = 0; i < lambda_.size(); ++i) J(i, i) = lambda_[i];
    }
    [[nodiscard]] Real exact(Index i, Real t, Real y0) const { return y0 * std::exp(lambda_[i] * t); }

private:
    std::vector<Real> lambda_;
};

// x'' = -omega^2 x as a first-order system; energy E = v^2 + omega^2 x^2 is invariant.
class HarmonicOscillator final : public OdeSystem {
public:
    explicit HarmonicOscillator(Real omega) : w_(omega) {}
    [[nodiscard]] Index size() const override { return 2; }
    void rhs(Real, std::span<const Real> y, std::span<Real> f) const override {
        f[0] = y[1];
        f[1] = -w_ * w_ * y[0];
    }
    [[nodiscard]] Real omega() const { return w_; }

private:
    Real w_;
};

// y' = A y with a constant dense matrix (row-major).
class LinearSystem final : public OdeSystem {
public:
    explicit LinearSystem(DenseMatrix A) : A_(std::move(A)) {}
    [[nodiscard]] Index size() const override { return A_.rows(); }
    void rhs(Real, std::span<const Real> y, std::span<Real> f) const override;
    [[nodiscard]] bool has_jacobian() const override { return true; }
    void jacobian(Real, std::span<const Real>, DenseMatrix& J) const override { J = A_; }
    [[nodiscard]] const DenseMatrix& matrix() const { return A_; }

private:
    DenseMatrix A_;
};

// Robertson chemical kinetics (Robertson 1966; Hairer & Wanner, Solving ODEs II):
// y1' = -0.04 y1 + 1e4 y2 y3; y2' = 0.04 y1 - 1e4 y2 y3 - 3e7 y2^2; y3' = 3e7 y2^2.
// Invariant: y1 + y2 + y3 = 1. Solution components are non-negative.
class Robertson final : public OdeSystem {
public:
    [[nodiscard]] Index size() const override { return 3; }
    void rhs(Real, std::span<const Real> y, std::span<Real> f) const override;
    [[nodiscard]] bool has_jacobian() const override { return analytic_jac_; }
    void jacobian(Real, std::span<const Real> y, DenseMatrix& J) const override;
    void set_analytic_jacobian(bool on) { analytic_jac_ = on; }

private:
    bool analytic_jac_ = true;
};

// Van der Pol oscillator: y1' = y2, y2' = mu ((1 - y1^2) y2 - y1).
class VanDerPol final : public OdeSystem {
public:
    explicit VanDerPol(Real mu) : mu_(mu) {}
    [[nodiscard]] Index size() const override { return 2; }
    void rhs(Real, std::span<const Real> y, std::span<Real> f) const override {
        f[0] = y[1];
        f[1] = mu_ * ((1 - y[0] * y[0]) * y[1] - y[0]);
    }
    [[nodiscard]] bool has_jacobian() const override { return true; }
    void jacobian(Real, std::span<const Real> y, DenseMatrix& J) const override {
        J(0, 0) = 0;
        J(0, 1) = 1;
        J(1, 0) = mu_ * (-2 * y[0] * y[1] - 1);
        J(1, 1) = mu_ * (1 - y[0] * y[0]);
    }

private:
    Real mu_;
};

// 1-D reaction-diffusion u_t = d u_xx - k u^2 on (0,1), u(0)=u(1)=0, N interior
// points (second-order central differences). Stiffness ratio grows like N^2.
// Provides an analytic Jacobian-vector product and a tridiagonal (exact)
// preconditioner for (I - c J), intended for Newton-Krylov tests.
class ReactionDiffusion1D final : public OdeSystem {
public:
    ReactionDiffusion1D(Index n, Real d, Real k) : n_(n), d_(d), k_(k), dx_(Real{1} / static_cast<Real>(n + 1)) {}
    [[nodiscard]] Index size() const override { return n_; }
    void rhs(Real, std::span<const Real> u, std::span<Real> f) const override;
    [[nodiscard]] bool has_jacobian() const override { return true; }
    void jacobian(Real, std::span<const Real> u, DenseMatrix& J) const override;
    [[nodiscard]] bool has_jvp() const override { return true; }
    void jvp(Real, std::span<const Real> u, std::span<const Real> v, std::span<Real> Jv) const override;
    [[nodiscard]] std::vector<Real> initial_condition() const;  // sin(pi x)

private:
    Index n_;
    Real d_, k_, dx_;
};

// Exact tridiagonal preconditioner for ReactionDiffusion1D: solves (I - c J) z = r.
class TridiagonalRDPreconditioner final : public Preconditioner {
public:
    TridiagonalRDPreconditioner(Index n, Real d, Real k);
    void setup(Real t, std::span<const Real> y, Real c) override;
    void apply(std::span<const Real> r, std::span<Real> z) const override;
    [[nodiscard]] int setups() const { return setups_; }

private:
    Index n_;
    Real d_, k_, dx_;
    std::vector<Real> a_, b_, cc_;          // sub, main, super diagonals
    mutable std::vector<Real> cp_, dp_;     // Thomas scratch
    int setups_ = 0;
};

// y' = y^2, y(0) = 1: solution 1/(1 - t) blows up at t = 1.
class BlowUp final : public OdeSystem {
public:
    [[nodiscard]] Index size() const override { return 1; }
    void rhs(Real, std::span<const Real> y, std::span<Real> f) const override { f[0] = y[0] * y[0]; }
};

// y' = -1, y(0) = 1, but the RHS is evaluated as -sqrt(y)/sqrt(y): it is NaN
// for y <= 0, i.e. for t >= 1. With domain checking, y <= 0 is reported as
// outside the model domain instead.
class NanBeyondZero final : public OdeSystem {
public:
    explicit NanBeyondZero(bool domain_check) : domain_(domain_check) {}
    [[nodiscard]] Index size() const override { return 1; }
    void rhs(Real, std::span<const Real> y, std::span<Real> f) const override {
        f[0] = -std::sqrt(y[0]) / std::sqrt(y[0]);
    }
    [[nodiscard]] bool in_domain(Real, std::span<const Real> y) const override { return !domain_ || y[0] > 0; }

private:
    bool domain_;
};

// y' = eps_inc (constant). With y(0) = 1 and eps_inc * h far below ulp(1),
// every increment is lost to rounding: exercises stagnation detection.
class TinyIncrement final : public OdeSystem {
public:
    explicit TinyIncrement(Real rate) : rate_(rate) {}
    [[nodiscard]] Index size() const override { return 1; }
    void rhs(Real, std::span<const Real>, std::span<Real> f) const override { f[0] = rate_; }

private:
    Real rate_;
};

}  // namespace dpd::ode::problems
