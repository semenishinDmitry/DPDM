// Explicit Runge-Kutta steppers: Dormand-Prince 5(4) and classical RK4.
#include <algorithm>
#include <bit>
#include <cmath>

#include "dpd/ode/jacobian.hpp"
#include "dpd/ode/kernels.hpp"
#include "dpd/ode/stepper.hpp"

namespace dpd::ode {

StepContext::StepContext(const OdeSystem& s, const Options& o, Stats& st) : sys(s), opt(o), stats(st) {
    const Index n = s.size();
    if (o.atol.size() == n)
        atol = o.atol;
    else
        atol.assign(n, o.atol.empty() ? Real{0} : o.atol.front());
}

bool all_finite(std::span<const Real> v) noexcept {
    for (Real x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

namespace {

// Dormand & Prince (1980) coefficients; dense output from Hairer, Norsett & Wanner (dopri5.f).
constexpr Real c2 = 1.0 / 5, c3 = 3.0 / 10, c4 = 4.0 / 5, c5 = 8.0 / 9;
constexpr Real a21 = 1.0 / 5;
constexpr Real a31 = 3.0 / 40, a32 = 9.0 / 40;
constexpr Real a41 = 44.0 / 45, a42 = -56.0 / 15, a43 = 32.0 / 9;
constexpr Real a51 = 19372.0 / 6561, a52 = -25360.0 / 2187, a53 = 64448.0 / 6561, a54 = -212.0 / 729;
constexpr Real a61 = 9017.0 / 3168, a62 = -355.0 / 33, a63 = 46732.0 / 5247, a64 = 49.0 / 176,
               a65 = -5103.0 / 18656;
constexpr Real a71 = 35.0 / 384, a73 = 500.0 / 1113, a74 = 125.0 / 192, a75 = -2187.0 / 6784, a76 = 11.0 / 84;
constexpr Real e1 = 71.0 / 57600, e3 = -71.0 / 16695, e4 = 71.0 / 1920, e5 = -17253.0 / 339200,
               e6 = 22.0 / 525, e7 = -1.0 / 40;
constexpr Real d1 = -12715105075.0 / 11282082432.0, d3 = 87487479700.0 / 32700410799.0,
               d4 = -10690763975.0 / 1880347072.0, d5 = 701980252875.0 / 199316789632.0,
               d6 = -1453857185.0 / 822651844.0, d7 = 69997945.0 / 29380423.0;

// PI controller parameters (Hairer & Wanner, dopri5.f defaults).
constexpr Real kBeta = 0.04;
constexpr Real kExpo1 = 0.2 - kBeta * 0.75;
constexpr Real kSafe = 0.9;
constexpr Real kFacc1 = 1.0 / 0.2;  // max step decrease 5x
constexpr Real kFacc2 = 1.0 / 10.0; // max step increase 10x
constexpr int kMaxConsecutiveBad = 50;

}  // namespace

Dopri5::Dopri5(StepContext& ctx) : c_(ctx), n_(ctx.sys.size()) {
    for (auto* v : {&y_, &ynew_, &comp_, &compnew_, &ytmp_, &ysti_, &delta_, &err_, &k1_, &k2_, &k3_, &k4_, &k5_,
                    &k6_, &k7_, &r1_, &r2_, &r3_, &r4_, &r5_, &pv_, &pjv_, &pwork_})
        v->assign(n_, 0);
}

Real Dopri5::estimate_rho(Real t) {
    // Power iteration on J(t, ynew) with finite-difference products; the start
    // vector k7 - k6 is already biased towards the stiff direction.
    for (Index i = 0; i < n_; ++i) pv_[i] = k7_[i] - k6_[i];
    Real vn = std::sqrt(kernels::dot(pv_.data(), pv_.data(), n_));
    if (!(vn > 0) || !std::isfinite(vn)) {
        for (Index i = 0; i < n_; ++i) pv_[i] = 1.0 + 0.1 * static_cast<Real>(i % 7);
        vn = std::sqrt(kernels::dot(pv_.data(), pv_.data(), n_));
    }
    Real rho = 0;
    for (int it = 0; it < c_.opt.stiff_rho_iterations; ++it) {
        for (Index i = 0; i < n_; ++i) pv_[i] /= vn;
        c_.stats.n_rhs += fd_jvp(c_.sys, t, ynew_, k7_, pv_, pjv_, pwork_);
        const Real jn = std::sqrt(kernels::dot(pjv_.data(), pjv_.data(), n_));
        if (!(jn > 0) || !std::isfinite(jn)) return rho;
        rho = jn;  // ||J v|| with ||v|| = 1
        std::swap(pv_, pjv_);
        vn = jn;
    }
    return rho;
}

Real Dopri5::initial_step() {
    // Hairer, Norsett & Wanner, Solving ODEs I, Sec. II.4 (hinit in dopri5.f).
    const auto& opt = c_.opt;
    for (Index i = 0; i < n_; ++i) ytmp_[i] = c_.atol[i] + opt.rtol * std::abs(y_[i]);
    const Real dnf = kernels::wrms(k1_.data(), ytmp_.data(), n_);
    const Real dny = kernels::wrms(y_.data(), ytmp_.data(), n_);
    Real h = (dnf <= 1e-10 || dny <= 1e-10) ? 1e-6 : 0.01 * dny / dnf;
    h = std::min(h, opt.h_max);
    for (Index i = 0; i < n_; ++i) k3_[i] = y_[i] + h * k1_[i];
    c_.rhs(t_ + h, k3_, k2_);
    for (Index i = 0; i < n_; ++i) k3_[i] = k2_[i] - k1_[i];
    Real der2 = kernels::wrms(k3_.data(), ytmp_.data(), n_) / h;
    if (!std::isfinite(der2)) der2 = 1 / h;
    const Real der12 = std::max(std::abs(der2), dnf);
    const Real h1 = der12 <= 1e-15 ? std::max(1e-6, h * 1e-3) : std::pow(0.01 / der12, 1.0 / 5.0);
    return std::min({100 * h, h1, opt.h_max});
}

void Dopri5::init(Real t0, std::span<const Real> y0, Real h_guess) {
    t_ = t_old_ = t0;
    std::copy(y0.begin(), y0.end(), y_.begin());
    std::fill(comp_.begin(), comp_.end(), Real{0});
    c_.rhs(t_, y_, k1_);
    h_ = h_guess > 0 ? std::min(h_guess, c_.opt.h_max) : initial_step();
    facold_ = 1e-4;
    last_rejected_ = stiff_ = false;
    iasti_ = nonsti_ = 0;
    window_bits_ = 0;
    window_fill_ = 0;
    rho_counter_ = rho_hits_ = 0;
    rho_last_ = 0;
    // Dense output on the degenerate interval [t0, t0]: constant.
    r1_ = y_;
    std::fill(r2_.begin(), r2_.end(), 0);
    std::fill(r3_.begin(), r3_.end(), 0);
    std::fill(r4_.begin(), r4_.end(), 0);
    std::fill(r5_.begin(), r5_.end(), 0);
    h_done_ = 0;
}

Status Dopri5::step(Real t_bound) {
    const auto& opt = c_.opt;
    auto& st = c_.stats;
    int bad = 0;
    while (true) {
        Real h = std::min(h_, opt.h_max);
        const Real remaining = t_bound - t_;
        if (h >= remaining) h = remaining;
        const Real hmin_round = StepContext::roundoff_step(t_);
        if (h < std::max(hmin_round, opt.h_min) && h < remaining) {
            detail_ = "step size " + std::to_string(h) + " below minimum at t=" + std::to_string(t_);
            return Status::StepSizeUnderflow;
        }
        const Real t_new = (h == remaining) ? t_bound : t_ + h;
        if (t_new == t_) {
            detail_ = "t + h == t at t=" + std::to_string(t_);
            return Status::TimeResolutionLost;
        }
        h = t_new - t_;  // the step actually represented by the grid points

        const Real* K[7];
        Real cc[7];
        // stage 2
        cc[0] = h * a21; K[0] = k1_.data();
        kernels::lincomb(ytmp_.data(), y_.data(), cc, K, 1, n_);
        c_.rhs(t_ + c2 * h, ytmp_, k2_);
        // stage 3
        cc[0] = h * a31; cc[1] = h * a32; K[1] = k2_.data();
        kernels::lincomb(ytmp_.data(), y_.data(), cc, K, 2, n_);
        c_.rhs(t_ + c3 * h, ytmp_, k3_);
        // stage 4
        cc[0] = h * a41; cc[1] = h * a42; cc[2] = h * a43; K[2] = k3_.data();
        kernels::lincomb(ytmp_.data(), y_.data(), cc, K, 3, n_);
        c_.rhs(t_ + c4 * h, ytmp_, k4_);
        // stage 5
        cc[0] = h * a51; cc[1] = h * a52; cc[2] = h * a53; cc[3] = h * a54; K[3] = k4_.data();
        kernels::lincomb(ytmp_.data(), y_.data(), cc, K, 4, n_);
        c_.rhs(t_ + c5 * h, ytmp_, k5_);
        // stage 6
        cc[0] = h * a61; cc[1] = h * a62; cc[2] = h * a63; cc[3] = h * a64; cc[4] = h * a65; K[4] = k5_.data();
        kernels::lincomb(ysti_.data(), y_.data(), cc, K, 5, n_);
        c_.rhs(t_new, ysti_, k6_);
        // increment delta = h * sum b_j k_j (b = a7j, b2 = 0)
        {
            const Real cb[5] = {h * a71, h * a73, h * a74, h * a75, h * a76};
            const Real* Kb[5] = {k1_.data(), k3_.data(), k4_.data(), k5_.data(), k6_.data()};
            std::fill(delta_.begin(), delta_.end(), Real{0});
            kernels::lincomb(delta_.data(), delta_.data(), cb, Kb, 5, n_);
        }
        // compensated update ynew = y + delta
        for (Index i = 0; i < n_; ++i) {
            const Real corr = delta_[i] - comp_[i];
            const Real yn = y_[i] + corr;
            compnew_[i] = (yn - y_[i]) - corr;
            ynew_[i] = yn;
        }
        c_.rhs(t_new, ynew_, k7_);
        bool finite = all_finite(ynew_) && all_finite(k7_);
        Real err = kInf;
        if (finite) {
            const Real ce[6] = {h * e1, h * e3, h * e4, h * e5, h * e6, h * e7};
            const Real* Ke[6] = {k1_.data(), k3_.data(), k4_.data(), k5_.data(), k6_.data(), k7_.data()};
            std::fill(err_.begin(), err_.end(), Real{0});
            kernels::lincomb(err_.data(), err_.data(), ce, Ke, 6, n_);
            err = kernels::wrms_err(err_.data(), y_.data(), ynew_.data(), c_.atol.data(), true, opt.rtol, n_);
            finite = std::isfinite(err);
        }
        if (!finite) {
            ++st.n_reject_nonfinite;
            if (++bad > kMaxConsecutiveBad) {
                detail_ = "non-finite values persist after step reduction at t=" + std::to_string(t_);
                return Status::NonFiniteState;
            }
            h_ = h * 0.25;
            last_rejected_ = true;
            continue;
        }
        if (!c_.sys.in_domain(t_new, ynew_)) {
            ++st.n_reject_domain;
            if (++bad > kMaxConsecutiveBad) {
                detail_ = "state leaves the model domain at t=" + std::to_string(t_);
                return Status::DomainViolation;
            }
            h_ = h * 0.5;
            last_rejected_ = true;
            continue;
        }

        const Real fac11 = std::pow(err, kExpo1);
        if (err <= 1) {
            // accepted
            Real fac = fac11 / std::pow(facold_, kBeta);
            fac = std::max(kFacc2, std::min(kFacc1, fac / kSafe));
            Real hnew = h / fac;
            facold_ = std::max(err, Real{1e-4});

            // Stiffness detection (dopri5.f): h * |lambda| estimate from stages 6/7.
            Real stnum = 0, stden = 0;
            for (Index i = 0; i < n_; ++i) {
                const Real a = k7_[i] - k6_[i];
                const Real b = ynew_[i] - ysti_[i];
                stnum += a * a;
                stden += b * b;
            }
            hlamb_ = stden > 0 ? h * std::sqrt(stnum / stden) : 0;
            if (hlamb_ > opt.stiff_hlamb) {
                nonsti_ = 0;
                if (++iasti_ >= opt.stiff_detect_steps && !stiff_) {
                    stiff_ = true;
                    stiff_reason_ = "explicit stability limit: h*|lambda| > " + std::to_string(opt.stiff_hlamb) +
                                    " for " + std::to_string(iasti_) + " accepted steps (last " +
                                    std::to_string(hlamb_) + ")";
                }
            } else if (++nonsti_ >= opt.nonstiff_reset_steps) {
                iasti_ = 0;
            }
            const int W = std::clamp(opt.stiff_window, 1, 64);
            const std::uint64_t mask = W == 64 ? ~std::uint64_t{0} : ((std::uint64_t{1} << W) - 1);
            const bool limited = hlamb_ >= opt.stiff_boundary_fraction * opt.stiff_hlamb;
            window_bits_ = ((window_bits_ << 1) | (limited ? 1u : 0u)) & mask;
            window_fill_ = std::min(window_fill_ + 1, W);
            if (!stiff_ && window_fill_ == W) {
                const int cnt = std::popcount(window_bits_);
                if (cnt >= opt.stiff_window_fraction * W) {
                    stiff_ = true;
                    stiff_reason_ = "step size pinned at explicit stability boundary: " + std::to_string(cnt) + " of last " +
                                    std::to_string(W) + " steps had h*|lambda| >= " +
                                    std::to_string(opt.stiff_boundary_fraction * opt.stiff_hlamb);
                }
            }

            if (!stiff_ && opt.stiff_rho_interval > 0 && ++rho_counter_ >= opt.stiff_rho_interval) {
                rho_counter_ = 0;
                rho_last_ = estimate_rho(t_new);
                if (h * rho_last_ >= opt.stiff_rho_threshold) {
                    if (++rho_hits_ >= opt.stiff_rho_checks) {
                        stiff_ = true;
                        stiff_reason_ = "h * rho(J) = " + std::to_string(h * rho_last_) + " >= " +
                                        std::to_string(opt.stiff_rho_threshold) + " in " + std::to_string(rho_hits_) +
                                        " consecutive spectral-radius checks";
                    }
                } else {
                    rho_hits_ = 0;
                }
            }

            // Dense output coefficients.
            for (Index i = 0; i < n_; ++i) {
                const Real ydiff = ynew_[i] - y_[i];
                const Real bspl = h * k1_[i] - ydiff;
                r1_[i] = y_[i];
                r2_[i] = ydiff;
                r3_[i] = bspl;
                r4_[i] = ydiff - h * k7_[i] - bspl;
                r5_[i] = h * (d1 * k1_[i] + d3 * k3_[i] + d4 * k4_[i] + d5 * k5_[i] + d6 * k6_[i] + d7 * k7_[i]);
            }

            // Lost-increment diagnostics (the lost part stays in comp_, so this is
            // not an error for DOPRI5).
            for (Index i = 0; i < n_; ++i)
                if (delta_[i] != 0 && ynew_[i] == y_[i]) {
                    ++st.n_lost_increment;
                    break;
                }

            std::swap(y_, ynew_);
            std::swap(comp_, compnew_);
            std::swap(k1_, k7_);
            t_old_ = t_;
            t_ = t_new;
            h_done_ = h;
            if (last_rejected_) hnew = std::min(hnew, h);
            last_rejected_ = false;
            h_ = hnew;
            ++st.n_steps;
            st.h_min = std::min(st.h_min, h);
            st.h_max = std::max(st.h_max, h);
            st.h_last = h;
            return Status::Success;
        }
        // rejected
        ++st.n_reject_error;
        h_ = h / std::min(kFacc1, fac11 / kSafe);
        last_rejected_ = true;
        if (++bad > 10 * kMaxConsecutiveBad && h_ < 1e3 * StepContext::roundoff_step(t_)) {
            detail_ = "repeated error-test failures at t=" + std::to_string(t_);
            return Status::StepSizeUnderflow;
        }
    }
}

void Dopri5::dense(Real t, std::span<Real> out) const {
    if (h_done_ == 0) {
        std::copy(y_.begin(), y_.end(), out.begin());
        return;
    }
    const Real th = (t - t_old_) / h_done_;
    const Real th1 = 1 - th;
    for (Index i = 0; i < n_; ++i)
        out[i] = r1_[i] + th * (r2_[i] + th1 * (r3_[i] + th * (r4_[i] + th1 * r5_[i])));
}

// ------------------------------------------------------------------- RK4 ---

Rk4::Rk4(StepContext& ctx) : c_(ctx), n_(ctx.sys.size()) {
    for (auto* v : {&y_, &yold_, &f_, &fold_, &k2_, &k3_, &k4_, &ytmp_, &comp_}) v->assign(n_, 0);
}

void Rk4::init(Real t0, std::span<const Real> y0, Real h_guess) {
    t_ = t_old_ = t0;
    std::copy(y0.begin(), y0.end(), y_.begin());
    yold_ = y_;
    std::fill(comp_.begin(), comp_.end(), Real{0});
    c_.rhs(t_, y_, f_);
    fold_ = f_;
    h_ = c_.opt.rk4_step > 0 ? c_.opt.rk4_step : h_guess;
}

Status Rk4::step(Real t_bound) {
    if (!(h_ > 0)) return Status::InvalidInput;
    const Real remaining = t_bound - t_;
    const Real h0 = std::min(h_, remaining);
    const Real t_new = (h0 == remaining) ? t_bound : t_ + h0;
    if (t_new == t_) return Status::TimeResolutionLost;
    const Real h = t_new - t_;
    for (Index i = 0; i < n_; ++i) ytmp_[i] = y_[i] + 0.5 * h * f_[i];
    c_.rhs(t_ + 0.5 * h, ytmp_, k2_);
    for (Index i = 0; i < n_; ++i) ytmp_[i] = y_[i] + 0.5 * h * k2_[i];
    c_.rhs(t_ + 0.5 * h, ytmp_, k3_);
    for (Index i = 0; i < n_; ++i) ytmp_[i] = y_[i] + h * k3_[i];
    c_.rhs(t_new, ytmp_, k4_);
    yold_ = y_;
    fold_ = f_;
    for (Index i = 0; i < n_; ++i) {
        const Real delta = h / 6 * (f_[i] + 2 * k2_[i] + 2 * k3_[i] + k4_[i]);
        const Real corr = delta - comp_[i];
        const Real yn = y_[i] + corr;
        comp_[i] = (yn - y_[i]) - corr;
        y_[i] = yn;
    }
    c_.rhs(t_new, y_, f_);
    if (!all_finite(y_) || !all_finite(f_)) return Status::NonFiniteState;
    t_old_ = t_;
    t_ = t_new;
    ++c_.stats.n_steps;
    c_.stats.h_min = std::min(c_.stats.h_min, h);
    c_.stats.h_max = std::max(c_.stats.h_max, h);
    c_.stats.h_last = h;
    return Status::Success;
}

void Rk4::dense(Real t, std::span<Real> out) const {
    const Real h = t_ - t_old_;
    if (h == 0) {
        std::copy(y_.begin(), y_.end(), out.begin());
        return;
    }
    // cubic Hermite interpolation
    const Real s = (t - t_old_) / h;
    const Real h00 = (1 + 2 * s) * (1 - s) * (1 - s), h10 = s * (1 - s) * (1 - s);
    const Real h01 = s * s * (3 - 2 * s), h11 = s * s * (s - 1);
    for (Index i = 0; i < n_; ++i) out[i] = h00 * yold_[i] + h10 * h * fold_[i] + h01 * y_[i] + h11 * h * f_[i];
}

}  // namespace dpd::ode
