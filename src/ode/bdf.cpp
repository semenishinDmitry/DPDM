// Variable-order variable-step BDF (fixed-leading-coefficient form).
//
// Algorithm follows Shampine & Reichelt, "The MATLAB ODE Suite", SIAM J. Sci.
// Comput. 18 (1997), with kappa = 0 (pure BDF), in the formulation used by
// SciPy's `BDF` class (difference array D, step-size change by the R*U
// transformation, order selection from estimated errors of orders k-1, k, k+1).
#include <algorithm>
#include <cmath>

#include "dpd/ode/jacobian.hpp"
#include "dpd/ode/kernels.hpp"
#include "dpd/ode/stepper.hpp"

namespace dpd::ode {

namespace {
constexpr Real kMinFactor = 0.2;
constexpr Real kMaxFactor = 10;
constexpr int kMaxConsecutiveBad = 60;

// R = cumprod over rows of M, M[0][*] = 1, M[i][j] = (i - 1 - factor * j) / i (i, j >= 1), M[i][0] = 0.
void compute_R(int order, Real factor, Real (&R)[Bdf::kMaxOrder + 1][Bdf::kMaxOrder + 1]) {
    for (int j = 0; j <= order; ++j) R[0][j] = 1;
    for (int i = 1; i <= order; ++i) {
        R[i][0] = 0;
        for (int j = 1; j <= order; ++j) R[i][j] = R[i - 1][j] * ((i - 1 - factor * j) / i);
    }
}
}  // namespace

Bdf::Bdf(StepContext& ctx)
    : c_(ctx), n_(ctx.sys.size()), max_order_(std::clamp(ctx.opt.bdf_max_order, 1, kMaxOrder)) {
    D_.resize(kMaxOrder + 3, n_);
    Dtmp_.resize(kMaxOrder + 1, n_);
    Ddense_.resize(kMaxOrder + 1, n_);
    J_.resize(n_, n_);
    if (ctx.opt.linear_solver == LinearSolverKind::DenseLU) {
        M_.resize(n_, n_);
        lu_.resize(n_);
    } else {
        gmres_.resize(n_, ctx.opt.gmres_restart);
    }
    for (auto* v : {&y_, &y_predict_, &psi_, &d_, &y_new_, &f_, &dy_, &scale_, &err_, &rhs_, &w1_, &w2_, &w3_, &typ_})
        v->assign(n_, 0);
    gamma_[0] = 0;
    for (int k = 1; k <= kMaxOrder; ++k) gamma_[k] = gamma_[k - 1] + Real{1} / k;
    for (int k = 0; k <= kMaxOrder; ++k) alpha_[k] = gamma_[k];  // (1 - kappa) * gamma, kappa = 0
    for (int k = 0; k <= kMaxOrder + 1; ++k) error_const_[k] = Real{1} / (k + 1);
}

void Bdf::change_D(int order, Real factor) {
    Real R[kMaxOrder + 1][kMaxOrder + 1], U[kMaxOrder + 1][kMaxOrder + 1], RU[kMaxOrder + 1][kMaxOrder + 1];
    compute_R(order, factor, R);
    compute_R(order, 1, U);
    for (int i = 0; i <= order; ++i)
        for (int j = 0; j <= order; ++j) {
            Real s = 0;
            for (int k = 0; k <= order; ++k) s += R[i][k] * U[k][j];
            RU[i][j] = s;
        }
    // D[:order+1] = RU^T D[:order+1]
    for (int i = 0; i <= order; ++i) {
        auto out = Dtmp_.row(static_cast<Index>(i));
        std::fill(out.begin(), out.end(), Real{0});
        for (int k = 0; k <= order; ++k) kernels::axpy(RU[k][i], D_.row(static_cast<Index>(k)).data(), out.data(), n_);
    }
    for (int i = 0; i <= order; ++i) {
        auto src = Dtmp_.row(static_cast<Index>(i));
        std::copy(src.begin(), src.end(), D_.row(static_cast<Index>(i)).begin());
    }
    lu_valid_ = false;
}

void Bdf::compute_jacobian(Real t, std::span<const Real> y) {
    evaluate_jacobian(t, y);
    // A Jacobian evaluated at a predictor where the model is undefined (NaN/Inf)
    // is useless; fall back to the last accepted state, which is always valid.
    if (!all_finite(std::span<const Real>(J_.data(), n_ * n_)) && (t != t_ || !std::equal(y.begin(), y.end(), y_.begin())))
        evaluate_jacobian(t_, y_);
    jac_current_ = true;
    lu_valid_ = false;
}

void Bdf::evaluate_jacobian(Real t, std::span<const Real> y) {
    auto& st = c_.stats;
    ++st.n_jac;
    if (c_.sys.has_jacobian()) {
        c_.sys.jacobian(t, y, J_);
    } else {
        c_.rhs(t, y, w3_);
        // CVODE-style minimum increment: 1000 |h| eps n ||f||_wrms, in units of the error weights.
        for (Index i = 0; i < n_; ++i) scale_[i] = c_.atol[i] + c_.opt.rtol * std::abs(y[i]);
        const Real fnorm = kernels::wrms(w3_.data(), scale_.data(), n_);
        Real min_inc = 1000 * h_abs_ * kEps * static_cast<Real>(n_) * fnorm;
        if (!(min_inc > 0) || !std::isfinite(min_inc)) min_inc = 1;
        const Real sq = std::sqrt(kEps);
        for (Index i = 0; i < n_; ++i) typ_[i] = min_inc * scale_[i] / sq;
        st.n_rhs += fd_jacobian(c_.sys, t, y, w3_, typ_, J_, w1_, w2_);
    }
}

bool Bdf::factor_iteration_matrix(Real c) {
    for (Index i = 0; i < n_; ++i)
        for (Index j = 0; j < n_; ++j) M_(i, j) = (i == j ? Real{1} : Real{0}) - c * J_(i, j);
    ++c_.stats.n_lu;
    lu_c_ = c;
    lu_valid_ = lu_.factor(M_);
    return lu_valid_;
}

bool Bdf::solve_newton(Real t_new, Real c, Real tol, int& n_iter, bool& linear_failure) {
    const auto& opt = c_.opt;
    auto& st = c_.stats;
    const int maxit = std::max(1, opt.newton_max_iter);
    const bool krylov = opt.linear_solver == LinearSolverKind::Gmres;
    linear_failure = false;
    std::fill(d_.begin(), d_.end(), Real{0});
    std::copy(y_predict_.begin(), y_predict_.end(), y_new_.begin());
    Real dy_norm_old = -1;
    bool converged = false;
    int k = 0;
    for (; k < maxit; ++k) {
        ++st.n_newton_iter;
        c_.rhs(t_new, y_new_, f_);
        if (!all_finite(f_)) break;
        for (Index i = 0; i < n_; ++i) rhs_[i] = c * f_[i] - psi_[i] - d_[i];
        if (!krylov) {
            std::copy(rhs_.begin(), rhs_.end(), dy_.begin());
            lu_.solve(dy_);
        } else {
            // Scaled system: (D (I - cJ) D^{-1}) (D dy) = D rhs, D = diag(1/scale).
            for (Index i = 0; i < n_; ++i) w3_[i] = rhs_[i] / scale_[i];
            std::fill(err_.begin(), err_.end(), Real{0});  // err_ holds u = D dy
            auto A = [&](std::span<const Real> u, std::span<Real> out) {
                for (Index i = 0; i < n_; ++i) w1_[i] = u[i] * scale_[i];
                ++st.n_jvp;
                if (c_.sys.has_jvp()) {
                    c_.sys.jvp(t_new, y_new_, w1_, w2_);
                } else {
                    st.n_rhs += fd_jvp(c_.sys, t_new, y_new_, f_, w1_, w2_, dy_);
                }
                for (Index i = 0; i < n_; ++i) out[i] = u[i] - c * w2_[i] / scale_[i];
            };
            auto P = [&](std::span<const Real> u, std::span<Real> out) {
                for (Index i = 0; i < n_; ++i) w1_[i] = u[i] * scale_[i];
                opt.preconditioner->apply(w1_, out);
                for (Index i = 0; i < n_; ++i) out[i] /= scale_[i];
            };
            const LinearOp Pref(P);
            const GmresResult gr = gmres_.solve(A, opt.preconditioner ? &Pref : nullptr, w3_, err_,
                                                opt.gmres_rel_tol, opt.gmres_max_iter);
            st.n_linear_iter += static_cast<std::uint64_t>(gr.iterations);
            if (!gr.converged) {
                linear_failure = true;
                detail_ = "GMRES did not converge (rel. residual " + std::to_string(gr.rel_residual) + ")";
                break;
            }
            for (Index i = 0; i < n_; ++i) dy_[i] = err_[i] * scale_[i];
        }
        const Real dy_norm = kernels::wrms(dy_.data(), scale_.data(), n_);
        if (!std::isfinite(dy_norm)) break;
        Real rate = -1;
        if (dy_norm_old > 0) rate = dy_norm / dy_norm_old;
        if (rate >= 0 && (rate >= 1 || std::pow(rate, maxit - k) / (1 - rate) * dy_norm > tol)) break;
        for (Index i = 0; i < n_; ++i) {
            y_new_[i] += dy_[i];
            d_[i] += dy_[i];
        }
        if (dy_norm == 0 || (rate >= 0 && rate / (1 - rate) * dy_norm < tol)) {
            converged = true;
            break;
        }
        dy_norm_old = dy_norm;
    }
    n_iter = std::min(k + 1, maxit);
    return converged;
}

Real Bdf::initial_step(std::span<const Real> f0) {
    // Hairer/SciPy select_initial_step with error order 1.
    for (Index i = 0; i < n_; ++i) scale_[i] = c_.atol[i] + c_.opt.rtol * std::abs(y_[i]);
    const Real d0 = kernels::wrms(y_.data(), scale_.data(), n_);
    const Real d1 = kernels::wrms(f0.data(), scale_.data(), n_);
    Real h0 = (d0 < 1e-5 || d1 < 1e-5) ? 1e-6 : 0.01 * d0 / d1;
    h0 = std::min(h0, c_.opt.h_max);
    for (Index i = 0; i < n_; ++i) w1_[i] = y_[i] + h0 * f0[i];
    c_.rhs(t_ + h0, w1_, w2_);
    for (Index i = 0; i < n_; ++i) w2_[i] -= f0[i];
    Real d2 = kernels::wrms(w2_.data(), scale_.data(), n_) / h0;
    if (!std::isfinite(d2)) d2 = 1 / h0;
    const Real h1 = (d1 <= 1e-15 && d2 <= 1e-15) ? std::max(1e-6, h0 * 1e-3) : std::sqrt(0.01 / std::max(d1, d2));
    return std::min({100 * h0, h1, c_.opt.h_max});
}

void Bdf::init(Real t0, std::span<const Real> y0, Real h_guess) {
    t_ = t_old_ = t0;
    std::copy(y0.begin(), y0.end(), y_.begin());
    c_.rhs(t_, y_, f_);
    std::copy(f_.begin(), f_.end(), w3_.begin());
    h_abs_ = h_guess > 0 ? std::min(h_guess, c_.opt.h_max) : initial_step(w3_);
    h_abs_old_ = 0;
    D_.fill(0);
    for (Index i = 0; i < n_; ++i) {
        D_(0, i) = y_[i];
        D_(1, i) = w3_[i] * h_abs_;
    }
    order_ = 1;
    dense_order_ = 1;
    n_equal_steps_ = 0;
    lu_valid_ = false;
    stagnant_ = 0;
    const Real rtol = c_.opt.rtol;
    newton_tol_ = c_.opt.newton_tol > 0 ? c_.opt.newton_tol : std::max(10 * kEps / rtol, std::min(0.03, std::sqrt(rtol)));
    if (c_.opt.linear_solver == LinearSolverKind::DenseLU) compute_jacobian(t_, y_);
    jac_current_ = false;
    Ddense_.fill(0);
    for (Index i = 0; i < n_; ++i) Ddense_(0, i) = y_[i];
}

Status Bdf::step(Real t_bound) {
    const auto& opt = c_.opt;
    auto& st = c_.stats;
    const bool krylov = opt.linear_solver == LinearSolverKind::Gmres;
    const Real min_step = std::max(StepContext::roundoff_step(t_), opt.h_min);
    if (h_abs_ > opt.h_max) {
        change_D(order_, opt.h_max / h_abs_);
        h_abs_ = opt.h_max;
        n_equal_steps_ = 0;
    } else if (h_abs_ < min_step) {
        change_D(order_, min_step / h_abs_);
        h_abs_ = min_step;
        n_equal_steps_ = 0;
    }
    jac_current_ = false;
    int bad = 0, n_iter = 0;
    Real t_new = t_, error_norm = 0;
    while (true) {
        if (h_abs_ < min_step && t_ + h_abs_ < t_bound) {
            detail_ = "step size " + std::to_string(h_abs_) + " below minimum at t=" + std::to_string(t_);
            return Status::StepSizeUnderflow;
        }
        t_new = t_ + h_abs_;
        if (t_new >= t_bound) {
            t_new = t_bound;
            change_D(order_, (t_new - t_) / h_abs_);
            h_abs_ = t_new - t_;  // D now describes this step exactly
            n_equal_steps_ = 0;
        }
        const Real h = t_new - t_;
        if (h <= 0) {
            detail_ = "t + h == t at t=" + std::to_string(t_);
            return Status::TimeResolutionLost;
        }
        // The representable step t_new - t may differ from h_abs_ by up to
        // ulp(t) / h relative. D must describe the grid actually used: otherwise
        // the predictor error grows like ulp(t)/h, causes rejections, smaller h
        // and an even larger relative mismatch (step-size collapse at tight rtol
        // and large |t|). Rescaling does not count as a step-size change for
        // order selection.
        if (h != h_abs_) change_D(order_, h / h_abs_);
        h_abs_ = h;

        std::fill(y_predict_.begin(), y_predict_.end(), Real{0});
        for (int k = 0; k <= order_; ++k) kernels::axpy(1, D_.row(static_cast<Index>(k)).data(), y_predict_.data(), n_);
        for (Index i = 0; i < n_; ++i) scale_[i] = c_.atol[i] + opt.rtol * std::abs(y_predict_[i]);
        std::fill(psi_.begin(), psi_.end(), Real{0});
        for (int k = 1; k <= order_; ++k)
            kernels::axpy(gamma_[k] / alpha_[order_], D_.row(static_cast<Index>(k)).data(), psi_.data(), n_);
        const Real c = h / alpha_[order_];

        bool converged = false, linear_failure = false;
        while (true) {
            if (!krylov) {
                if (!lu_valid_ || lu_c_ != c) {
                    if (!factor_iteration_matrix(c)) {
                        linear_failure = true;
                        detail_ = "iteration matrix I - cJ singular to working precision (pivot ratio " +
                                  std::to_string(lu_.pivot_ratio()) + ")";
                    }
                }
            } else if (opt.preconditioner && lu_c_ != c) {
                opt.preconditioner->setup(t_new, y_predict_, c);
                lu_c_ = c;
            }
            if (!linear_failure) converged = solve_newton(t_new, c, newton_tol_, n_iter, linear_failure);
            if (converged || krylov || jac_current_) break;
            compute_jacobian(t_new, y_predict_);  // stale Jacobian: refresh once and retry
            linear_failure = false;
        }
        bool reject = false;
        if (!converged) {
            ++st.n_reject_newton;
            reject = true;
        } else if (!c_.sys.in_domain(t_new, y_new_)) {
            ++st.n_reject_domain;
            reject = true;
        }
        if (reject) {
            if (++bad > kMaxConsecutiveBad) {
                if (linear_failure) return Status::LinearSolverFailure;
                if (!converged) {
                    if (detail_.empty()) detail_ = "Newton iteration failed to converge at t=" + std::to_string(t_);
                    return Status::NewtonFailure;
                }
                detail_ = "state leaves the model domain at t=" + std::to_string(t_);
                return Status::DomainViolation;
            }
            h_abs_ *= 0.5;
            change_D(order_, 0.5);
            n_equal_steps_ = 0;
            continue;
        }

        const Real safety = 0.9 * (2 * opt.newton_max_iter + 1) / (2 * opt.newton_max_iter + n_iter);
        for (Index i = 0; i < n_; ++i) {
            scale_[i] = c_.atol[i] + opt.rtol * std::abs(y_new_[i]);
            err_[i] = error_const_[order_] * d_[i];
        }
        error_norm = kernels::wrms(err_.data(), scale_.data(), n_);
        if (!std::isfinite(error_norm) || !all_finite(y_new_)) {
            ++st.n_reject_nonfinite;
            if (++bad > kMaxConsecutiveBad) return Status::NonFiniteState;
            h_abs_ *= 0.5;
            change_D(order_, 0.5);
            n_equal_steps_ = 0;
            continue;
        }
        if (error_norm > 1) {
            ++st.n_reject_error;
            const Real factor = std::max(kMinFactor, safety * std::pow(error_norm, -1.0 / (order_ + 1)));
            h_abs_ *= factor;
            change_D(order_, factor);
            n_equal_steps_ = 0;
            // Newton converged, so the LU is only invalidated because c changes.
            continue;
        }
        // accepted
        ++n_equal_steps_;
        // lost increments (uses the last Newton RHS f_ as derivative estimate)
        std::uint64_t lost = 0, nonzero = 0;
        for (Index i = 0; i < n_; ++i)
            if (f_[i] != 0 && h * f_[i] != 0) {
                ++nonzero;
                if (y_new_[i] == y_[i] && std::abs(h * f_[i]) > kEps * scale_[i]) ++lost;
            }
        if (lost) ++st.n_lost_increment;
        stagnant_ = (nonzero > 0 && lost == nonzero) ? stagnant_ + 1 : 0;

        t_old_ = t_;
        t_ = t_new;
        std::copy(y_new_.begin(), y_new_.end(), y_.begin());
        h_abs_old_ = h;
        const auto o = static_cast<Index>(order_);
        for (Index i = 0; i < n_; ++i) {
            D_(o + 2, i) = d_[i] - D_(o + 1, i);
            D_(o + 1, i) = d_[i];
        }
        for (int i = order_; i >= 0; --i)
            kernels::axpy(1, D_.row(static_cast<Index>(i) + 1).data(), D_.row(static_cast<Index>(i)).data(), n_);
        dense_order_ = order_;
        for (int k = 0; k <= order_; ++k) {
            auto src = D_.row(static_cast<Index>(k));
            std::copy(src.begin(), src.end(), Ddense_.row(static_cast<Index>(k)).begin());
        }
        ++st.n_steps;
        st.h_min = std::min(st.h_min, h);
        st.h_max = std::max(st.h_max, h);
        st.h_last = h;
        if (opt.max_stagnant_steps && stagnant_ >= opt.max_stagnant_steps) {
            detail_ = "all state increments lost to rounding for " + std::to_string(stagnant_) + " steps";
            return Status::TimeResolutionLost;
        }

        if (n_equal_steps_ < order_ + 1) return Status::Success;

        Real em = kInf, ep = kInf;
        if (order_ > 1) {
            for (Index i = 0; i < n_; ++i) err_[i] = error_const_[order_ - 1] * D_(o, i);
            em = kernels::wrms(err_.data(), scale_.data(), n_);
        }
        if (order_ < max_order_) {
            for (Index i = 0; i < n_; ++i) err_[i] = error_const_[order_ + 1] * D_(o + 2, i);
            ep = kernels::wrms(err_.data(), scale_.data(), n_);
        }
        const Real norms[3] = {em, error_norm, ep};
        Real factors[3];
        for (int k = 0; k < 3; ++k)
            factors[k] = norms[k] == 0 ? kInf : std::pow(norms[k], -1.0 / (order_ + k));
        const int best = static_cast<int>(std::max_element(factors, factors + 3) - factors);
        order_ += best - 1;
        const Real factor = std::min(kMaxFactor, safety * factors[best]);
        h_abs_ *= factor;
        change_D(order_, factor);
        n_equal_steps_ = 0;
        return Status::Success;
    }
}

void Bdf::dense(Real t, std::span<Real> out) const {
    // Newton backward-difference interpolant on the equally spaced grid
    // t_, t_ - h, ..., t_ - k h (h = last step), stored in Ddense_.
    auto d0 = Ddense_.row(0);
    std::copy(d0.begin(), d0.end(), out.begin());
    const Real h = h_abs_old_;
    if (h == 0) return;
    Real p = 1;
    for (int k = 0; k < dense_order_; ++k) {
        p *= (t - (t_ - h * k)) / (h * (k + 1));
        kernels::axpy(p, Ddense_.row(static_cast<Index>(k) + 1).data(), out.data(), n_);
    }
}

std::unique_ptr<Stepper> make_stepper(Method m, StepContext& ctx) {
    switch (m) {
        case Method::Dopri5: return std::make_unique<Dopri5>(ctx);
        case Method::Rk4: return std::make_unique<Rk4>(ctx);
        case Method::Bdf: return std::make_unique<Bdf>(ctx);
        case Method::Auto: return std::make_unique<Dopri5>(ctx);
    }
    return nullptr;
}

}  // namespace dpd::ode
