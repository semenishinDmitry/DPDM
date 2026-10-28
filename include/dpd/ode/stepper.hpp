// One-step interface shared by all integration methods.
#pragma once

#include <cmath>
#include <memory>
#include <string>

#include "dpd/ode/linalg.hpp"
#include "dpd/ode/options.hpp"
#include "dpd/ode/system.hpp"

namespace dpd::ode {

// Everything a stepper needs besides its own workspace. Counters are updated here.
struct StepContext {
    const OdeSystem& sys;
    const Options& opt;
    Stats& stats;
    std::vector<Real> atol;  // expanded to n entries

    StepContext(const OdeSystem& s, const Options& o, Stats& st);
    void rhs(Real t, std::span<const Real> y, std::span<Real> f) const {
        ++stats.n_rhs;
        sys.rhs(t, y, f);
    }
    // Smallest step that still changes t by a meaningful amount.
    [[nodiscard]] static Real roundoff_step(Real t) noexcept {
        const Real at = std::abs(t);
        return 10 * (std::nextafter(at, kInf) - at);
    }
};

[[nodiscard]] bool all_finite(std::span<const Real> v) noexcept;

class Stepper {
public:
    virtual ~Stepper() = default;
    [[nodiscard]] virtual Method method() const = 0;
    // Start (or restart) integration from (t0, y0). h_guess <= 0: automatic.
    virtual void init(Real t0, std::span<const Real> y0, Real h_guess) = 0;
    // Take exactly one accepted step, never stepping past t_bound.
    // Returns Status::Success when a step was accepted; any other status is a failure.
    virtual Status step(Real t_bound) = 0;
    [[nodiscard]] virtual Real t() const = 0;
    [[nodiscard]] virtual Real t_old() const = 0;
    [[nodiscard]] virtual std::span<const Real> y() const = 0;
    // Continuous extension valid on [t_old(), t()].
    virtual void dense(Real t, std::span<Real> out) const = 0;
    // Magnitude of the step proposed for the next attempt.
    [[nodiscard]] virtual Real h_next() const = 0;
    [[nodiscard]] virtual bool stiffness_detected() const { return false; }
    [[nodiscard]] virtual std::string stiffness_reason() const { return {}; }
    // Free-form detail about the last failure.
    [[nodiscard]] virtual std::string failure_detail() const { return {}; }
};

// Dormand-Prince 5(4) with PI step-size control, 4th-order dense output and
// Hairer's stiffness detector. Compensated (Kahan) summation of the state update.
class Dopri5 final : public Stepper {
public:
    explicit Dopri5(StepContext& ctx);
    [[nodiscard]] Method method() const override { return Method::Dopri5; }
    void init(Real t0, std::span<const Real> y0, Real h_guess) override;
    Status step(Real t_bound) override;
    [[nodiscard]] Real t() const override { return t_; }
    [[nodiscard]] Real t_old() const override { return t_old_; }
    [[nodiscard]] std::span<const Real> y() const override { return y_; }
    void dense(Real t, std::span<Real> out) const override;
    [[nodiscard]] Real h_next() const override { return h_; }
    [[nodiscard]] bool stiffness_detected() const override { return stiff_; }
    [[nodiscard]] std::string stiffness_reason() const override { return stiff_reason_; }
    [[nodiscard]] std::string failure_detail() const override { return detail_; }
    [[nodiscard]] Real last_hlamb() const { return hlamb_; }
    [[nodiscard]] Real last_rho() const { return rho_last_; }

private:
    Real initial_step();
    Real estimate_rho(Real t);
    StepContext& c_;
    Index n_;
    Real t_ = 0, t_old_ = 0, h_ = 0, h_done_ = 0, facold_ = 1e-4, hlamb_ = 0;
    bool last_rejected_ = false, stiff_ = false;
    int iasti_ = 0, nonsti_ = 0;
    std::uint64_t window_bits_ = 0;  // last <= 64 steps: 1 = stability-limited
    int window_fill_ = 0;
    // (lost increments are carried by the compensated summation)
    int rho_counter_ = 0, rho_hits_ = 0;
    Real rho_last_ = 0;
    std::vector<Real> pv_, pjv_, pwork_;  // power-iteration workspace
    std::string stiff_reason_, detail_;
    std::vector<Real> y_, ynew_, comp_, compnew_, ytmp_, ysti_, delta_, err_;
    std::vector<Real> k1_, k2_, k3_, k4_, k5_, k6_, k7_;
    std::vector<Real> r1_, r2_, r3_, r4_, r5_;  // dense output coefficients
};

// Classical fixed-step RK4 with cubic Hermite dense output (verification only).
class Rk4 final : public Stepper {
public:
    explicit Rk4(StepContext& ctx);
    [[nodiscard]] Method method() const override { return Method::Rk4; }
    void init(Real t0, std::span<const Real> y0, Real h_guess) override;
    Status step(Real t_bound) override;
    [[nodiscard]] Real t() const override { return t_; }
    [[nodiscard]] Real t_old() const override { return t_old_; }
    [[nodiscard]] std::span<const Real> y() const override { return y_; }
    void dense(Real t, std::span<Real> out) const override;
    [[nodiscard]] Real h_next() const override { return h_; }

private:
    StepContext& c_;
    Index n_;
    Real t_ = 0, t_old_ = 0, h_ = 0;
    std::vector<Real> y_, yold_, f_, fold_, k2_, k3_, k4_, ytmp_, comp_;
};

// Variable-order (1..5), variable-step BDF in the fixed-leading-coefficient
// (Nordsieck-like difference array) form of Shampine & Reichelt (MATLAB ode15s
// with kappa = 0), as also used by SciPy's BDF. Newton iteration either with a
// dense LU of (I - c J) (simplified Newton, Jacobian reuse) or matrix-free
// inexact Newton-Krylov with restarted GMRES.
class Bdf final : public Stepper {
public:
    explicit Bdf(StepContext& ctx);
    [[nodiscard]] Method method() const override { return Method::Bdf; }
    void init(Real t0, std::span<const Real> y0, Real h_guess) override;
    Status step(Real t_bound) override;
    [[nodiscard]] Real t() const override { return t_; }
    [[nodiscard]] Real t_old() const override { return t_old_; }
    [[nodiscard]] std::span<const Real> y() const override { return y_; }
    void dense(Real t, std::span<Real> out) const override;
    [[nodiscard]] Real h_next() const override { return h_abs_; }
    [[nodiscard]] int order() const { return order_; }
    [[nodiscard]] std::string failure_detail() const override { return detail_; }

    static constexpr int kMaxOrder = 5;

private:
    void change_D(int order, Real factor);
    void compute_jacobian(Real t, std::span<const Real> y);
    void evaluate_jacobian(Real t, std::span<const Real> y);
    bool factor_iteration_matrix(Real c);
    // Returns true if converged; n_iter receives the iteration count.
    bool solve_newton(Real t_new, Real c, Real tol, int& n_iter, bool& linear_failure);
    Real initial_step(std::span<const Real> f0);

    StepContext& c_;
    Index n_;
    int max_order_;
    Real t_ = 0, t_old_ = 0, h_abs_ = 0, h_abs_old_ = 0, newton_tol_ = 0;
    int order_ = 1, n_equal_steps_ = 0, dense_order_ = 1;
    bool lu_valid_ = false, jac_current_ = false;
    Real lu_c_ = 0;
    std::vector<Real> lost_acc_;  // signed sum of lost increments per component
    std::string detail_;
    Real gamma_[kMaxOrder + 1]{}, alpha_[kMaxOrder + 1]{}, error_const_[kMaxOrder + 2]{};
    DenseMatrix D_;      // (kMaxOrder + 3) x n difference array
    DenseMatrix Dtmp_;   // (kMaxOrder + 1) x n scratch for change_D
    DenseMatrix Ddense_; // (kMaxOrder + 1) x n copy for dense output
    DenseMatrix J_, M_;  // Jacobian and iteration matrix I - c J
    DenseLU lu_;
    Gmres gmres_;
    std::vector<Real> y_, y_predict_, psi_, d_, y_new_, f_, dy_, scale_, err_, rhs_, w1_, w2_, w3_, typ_;
};

std::unique_ptr<Stepper> make_stepper(Method m, StepContext& ctx);

}  // namespace dpd::ode
