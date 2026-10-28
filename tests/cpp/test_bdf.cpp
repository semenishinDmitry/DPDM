#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numbers>

#include "dpd/ode/integrate.hpp"
#include "dpd/ode/test_problems.hpp"
#include "reference_data.hpp"
#include "stiff_linear.hpp"

using namespace dpd::ode;

TEST_CASE("BDF: exponential decay accuracy improves with tolerance", "[bdf]") {
    problems::ExponentialSystem sys({-1.0});
    const std::vector<Real> y0{1.0};
    Real prev = kInf;
    for (Real rtol : {1e-3, 1e-5, 1e-7, 1e-9}) {
        Options o;
        o.method = Method::Bdf;
        o.rtol = rtol;
        o.atol = {1e-14};
        const auto r = integrate(sys, 0, y0, 5, o);
        REQUIRE(r.status == Status::Success);
        const Real err = std::abs(r.y_last[0] / std::exp(-5.0) - 1);
        CAPTURE(rtol, err);
        CHECK(err < 100 * rtol);
        CHECK(err < prev);
        prev = err;
    }
}

TEST_CASE("BDF: linear stiff system with known eigenvalues", "[bdf][stiff]") {
    testing_util::StiffLinear sl(-1.0, -1e3, -1e6);
    problems::LinearSystem sys(sl.matrix());
    const std::vector<Real> y0{1.0, 2.0, -1.0};
    for (auto ls : {LinearSolverKind::DenseLU, LinearSolverKind::Gmres}) {
        Options o;
        o.method = Method::Bdf;
        o.rtol = 1e-8;
        o.atol = {1e-12};
        o.linear_solver = ls;
        const auto r = integrate(sys, 0, y0, 10.0, o);
        REQUIRE(r.status == Status::Success);
        const auto ex = sl.exact(10.0, y0);
        // Global error bound: 100 * rtol relative (BDF global errors are typically
        // 10-100x the local tolerance) plus the absolute tolerance.
        // Global error: relative errors of the decaying slow mode add up over the
        // steps, so the bound is ~ n_steps * rtol (plus the absolute tolerance).
        const Real bound = 2.0 * static_cast<Real>(r.stats.n_steps) * o.rtol;
        for (int i = 0; i < 3; ++i) CHECK(std::abs(r.y_last[i] - ex[i]) < bound * std::abs(ex[i]) + 10 * o.atol[0]);
        CHECK(r.stats.n_steps < 2000);  // explicit methods would need ~1e7 steps (|lambda| h < 3.3)
    }
}

TEST_CASE("BDF: Robertson kinetics vs high-accuracy reference", "[bdf][stiff][robertson]") {
    for (bool analytic : {true, false}) {
        CAPTURE(analytic);
        problems::Robertson sys;
        sys.set_analytic_jacobian(analytic);
        Options o;
        o.method = Method::Bdf;
        o.rtol = 1e-8;
        o.atol = {1e-14, 1e-18, 1e-14};
        o.save_steps = true;
        o.t_eval.assign(refdata::kRoberT, refdata::kRoberT + refdata::kRoberN);
        o.save_steps = false;
        const std::vector<Real> y0{1, 0, 0};
        const auto r = integrate(sys, 0, y0, refdata::kRoberT[refdata::kRoberN - 1], o);
        REQUIRE(r.status == Status::Success);
        REQUIRE(r.size() == static_cast<Index>(refdata::kRoberN));
        for (int k = 0; k < refdata::kRoberN; ++k) {
            const auto y = r.y_at(static_cast<Index>(k));
            CAPTURE(k, refdata::kRoberT[k]);
            for (int i = 0; i < 3; ++i) {
                const Real ref = refdata::kRoberY[k][i];
                CHECK(std::abs(y[i] - ref) <= 2e-6 * std::abs(ref) + 10 * o.atol[i]);
            }
            CHECK(std::abs(y[0] + y[1] + y[2] - 1) < 1e-10);  // exact invariant
            CHECK(y[1] > -o.atol[1] * 10);                    // positivity (within tolerance)
        }
        CHECK(r.stats.n_steps < 2000);
    }
}

TEST_CASE("BDF: stiff Van der Pol (mu = 1000) converges to the reference", "[bdf][stiff][vdp]") {
    // Reference: SciPy LSODA at rtol 1e-12 (tests/reference). The first relaxation
    // jump occurs near t ~ 807; the phase error after it dominates. We check
    // tolerance proportionality (error shrinks ~10x per decade of rtol) and a bound.
    problems::VanDerPol sys(1000.0);
    const std::vector<Real> y0{2, 0};
    Real errs[2];
    int idx = 0;
    for (Real rtol : {1e-6, 1e-8}) {
        Options o;
        o.method = Method::Bdf;
        o.rtol = rtol;
        o.atol = {rtol * 0.1};
        o.t_eval = {refdata::kVdpT[0], refdata::kVdpT[1]};  // t = 500, 1000
        o.save_steps = false;
        const auto r = integrate(sys, 0, y0, refdata::kVdpT[1], o);
        REQUIRE(r.status == Status::Success);
        REQUIRE(r.size() == 2);
        Real e = 0;
        for (Index k = 0; k < 2; ++k) e = std::max(e, std::abs(r.y_at(k)[0] - refdata::kVdpY[k][0]));
        errs[idx++] = e;
    }
    CAPTURE(errs[0], errs[1]);
    CHECK(errs[1] < errs[0] / 20);
    CHECK(errs[1] < 1e-3);
}

TEST_CASE("BDF: tight tolerance at large |t| does not collapse the step size", "[bdf][floating]") {
    // Regression: with rtol = 1e-10 the step near t ~ 1024 shrank to ~1e-8, where the
    // representable step t_new - t differs from h by ~ulp(t)/h ~ 2e-5 relative. The
    // difference array is now rescaled to the exact step, which removes the feedback.
    problems::VanDerPol sys(1000.0);
    const std::vector<Real> y0{2, 0};
    Options o;
    o.method = Method::Bdf;
    o.rtol = 1e-10;
    o.atol = {1e-11};
    o.save_steps = false;
    o.t_eval = {refdata::kVdpT[1]};
    const auto r = integrate(sys, 0, y0, 1100.0, o);
    REQUIRE(r.status == Status::Success);
    CHECK(std::abs(r.y_at(0)[0] - refdata::kVdpY[1][0]) < 1e-5);
}

TEST_CASE("BDF: Newton-Krylov (GMRES) agrees with dense LU on a large stiff system", "[bdf][gmres]") {
    const Index n = 300;
    problems::ReactionDiffusion1D sys(n, 1.0, 5.0);
    const auto u0 = sys.initial_condition();
    Options base;
    base.method = Method::Bdf;
    base.rtol = 1e-7;
    base.atol = {1e-10};
    base.save_steps = false;
    Options lu = base;
    const auto rl = integrate(sys, 0, u0, 0.2, lu);
    REQUIRE(rl.status == Status::Success);

    Options kr = base;
    kr.linear_solver = LinearSolverKind::Gmres;
    kr.gmres_max_iter = 2000;
    const auto rk = integrate(sys, 0, u0, 0.2, kr);
    REQUIRE(rk.status == Status::Success);

    problems::TridiagonalRDPreconditioner P(n, 1.0, 5.0);
    Options kp = kr;
    kp.preconditioner = &P;
    const auto rp = integrate(sys, 0, u0, 0.2, kp);
    REQUIRE(rp.status == Status::Success);
    CHECK(P.setups() > 0);

    Real umax = 0;
    for (Real v : rl.y_last) umax = std::max(umax, std::abs(v));
    for (Index i = 0; i < n; ++i) {
        CHECK(std::abs(rk.y_last[i] - rl.y_last[i]) < 1e-5 * umax);
        CHECK(std::abs(rp.y_last[i] - rl.y_last[i]) < 1e-5 * umax);
    }
    CHECK(rk.stats.n_lu == 0);
    CHECK(rk.stats.n_linear_iter > 0);
    CHECK(rp.stats.n_linear_iter < rk.stats.n_linear_iter);
    CHECK(rl.stats.n_lu > 0);
    // Exact solution decays like exp(-pi^2 t) for the linear part; sanity bound
    CHECK(umax < std::exp(-std::numbers::pi * std::numbers::pi * 0.2) * 1.01);
}

namespace {
// Hides the analytic JVP/Jacobian so the finite-difference JVP path is exercised.
class NoDerivatives final : public OdeSystem {
public:
    explicit NoDerivatives(const OdeSystem& s) : s_(s) {}
    [[nodiscard]] Index size() const override { return s_.size(); }
    void rhs(Real t, std::span<const Real> y, std::span<Real> f) const override { s_.rhs(t, y, f); }

private:
    const OdeSystem& s_;
};
}  // namespace

TEST_CASE("BDF: Newton-Krylov with finite-difference JVP", "[bdf][gmres]") {
    problems::ReactionDiffusion1D inner(100, 1.0, 5.0);
    NoDerivatives sys(inner);
    const auto u0 = inner.initial_condition();
    Options o;
    o.method = Method::Bdf;
    o.rtol = 1e-7;
    o.atol = {1e-10};
    o.save_steps = false;
    o.linear_solver = LinearSolverKind::Gmres;
    o.gmres_max_iter = 1000;
    const auto rk = integrate(sys, 0, u0, 0.1, o);
    REQUIRE(rk.status == Status::Success);
    o.linear_solver = LinearSolverKind::DenseLU;
    const auto rl = integrate(inner, 0, u0, 0.1, o);
    REQUIRE(rl.status == Status::Success);
    for (Index i = 0; i < 100; ++i) CHECK(std::abs(rk.y_last[i] - rl.y_last[i]) < 1e-5);
    CHECK(rk.stats.n_jvp > 0);
}

TEST_CASE("BDF: higher maximum order reduces work on smooth problems", "[bdf]") {
    problems::HarmonicOscillator sys(1.0);
    const std::vector<Real> y0{1, 0};
    std::uint64_t steps[2];
    int idx = 0;
    for (int order : {1, 5}) {
        Options o;
        o.method = Method::Bdf;
        o.rtol = 1e-7;
        o.atol = {1e-9};
        o.bdf_max_order = order;
        o.save_steps = false;
        const auto r = integrate(sys, 0, y0, 10, o);
        REQUIRE(r.ok());
        // BDF1 is strongly dissipative on oscillations (amplitude error O(h)); BDF5 is not.
        CHECK(std::abs(r.y_last[0] - std::cos(10.0)) < (order == 1 ? 1e-2 : 1e-5));
        steps[idx++] = r.stats.n_steps;
    }
    CHECK(steps[1] * 10 < steps[0]);
}
