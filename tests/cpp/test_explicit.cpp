#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numbers>

#include "dpd/ode/integrate.hpp"
#include "dpd/ode/stepper.hpp"
#include "dpd/ode/test_problems.hpp"

using namespace dpd::ode;

TEST_CASE("DOPRI5: exponential decay and growth within tolerance", "[dopri5]") {
    problems::ExponentialSystem sys({-1.0, 0.5});
    const std::vector<Real> y0{1.0, 2.0};
    Real prev_err = kInf;
    for (Real rtol : {1e-4, 1e-6, 1e-8, 1e-10, 1e-12}) {
        Options o;
        o.method = Method::Dopri5;
        o.rtol = rtol;
        o.atol = {rtol * 1e-3};
        const auto r = integrate(sys, 0, y0, 10, o);
        REQUIRE(r.status == Status::Success);
        const Real e0 = std::abs(r.y_last[0] - sys.exact(0, 10, 1.0)) / sys.exact(0, 10, 1.0);
        const Real e1 = std::abs(r.y_last[1] - sys.exact(1, 10, 2.0)) / sys.exact(1, 10, 2.0);
        const Real err = std::max(e0, e1);
        CAPTURE(rtol, err);
        CHECK(err < 50 * rtol);       // global error is a modest multiple of the local tolerance
        CHECK(err < prev_err);        // tighter tolerance -> better agreement
        prev_err = err;
        CHECK(r.stats.switches.empty());
    }
}

TEST_CASE("DOPRI5: local error has order 5 (error ~ h^6)", "[dopri5][order]") {
    problems::ExponentialSystem sys({-1.0});
    Options o;
    o.method = Method::Dopri5;
    o.rtol = 1e-2;  // loose: every forced step is accepted
    o.atol = {1e-2};
    Stats st;
    StepContext ctx(sys, o, st);
    std::vector<Real> errs;
    for (Real h : {0.4, 0.2, 0.1, 0.05}) {
        Dopri5 dp(ctx);
        const std::vector<Real> y0{1.0};
        dp.init(0, y0, h);
        REQUIRE(dp.step(h) == Status::Success);
        REQUIRE(dp.t() == h);
        errs.push_back(std::abs(dp.y()[0] - std::exp(-h)));
    }
    for (std::size_t i = 1; i < errs.size(); ++i) {
        const Real p = std::log2(errs[i - 1] / errs[i]);
        CAPTURE(i, p);
        CHECK(p > 5.6);
        CHECK(p < 6.4);
    }
}

TEST_CASE("RK4: global error has order 4", "[rk4][order]") {
    problems::HarmonicOscillator sys(2.0);
    const std::vector<Real> y0{1.0, 0.0};
    std::vector<Real> errs;
    for (Real h : {0.02, 0.01, 0.005}) {
        Options o;
        o.method = Method::Rk4;
        o.rk4_step = h;
        o.save_steps = false;
        const auto r = integrate(sys, 0, y0, 5.0, o);
        REQUIRE(r.status == Status::Success);
        errs.push_back(std::abs(r.y_last[0] - std::cos(2.0 * 5.0)));
    }
    for (std::size_t i = 1; i < errs.size(); ++i) {
        const Real p = std::log2(errs[i - 1] / errs[i]);
        CAPTURE(p);
        CHECK(p > 3.8);
        CHECK(p < 4.2);
    }
}

TEST_CASE("DOPRI5: harmonic oscillator phase and energy over many periods", "[dopri5]") {
    const Real w = 3.0;
    problems::HarmonicOscillator sys(w);
    const std::vector<Real> y0{1.0, 0.0};
    Options o;
    o.method = Method::Dopri5;
    o.rtol = 1e-10;
    o.atol = {1e-12};
    const Real T = 50 * 2 * std::numbers::pi / w;
    const auto r = integrate(sys, 0, y0, T, o);
    REQUIRE(r.ok());
    CHECK(std::abs(r.y_last[0] - 1.0) < 1e-7);
    CHECK(std::abs(r.y_last[1]) < 3e-7);
    Real emax = 0;
    for (Index i = 0; i < r.size(); ++i) {
        const auto y = r.y_at(i);
        emax = std::max(emax, std::abs(y[1] * y[1] + w * w * y[0] * y[0] - w * w));
    }
    CHECK(emax / (w * w) < 1e-7);  // invariant preserved to tolerance level (no symplecticity claimed)
}

TEST_CASE("DOPRI5: dense output is accurate between steps", "[dopri5][dense]") {
    problems::HarmonicOscillator sys(1.0);
    const std::vector<Real> y0{0.0, 1.0};
    Options o;
    o.method = Method::Dopri5;
    o.rtol = 1e-8;
    o.atol = {1e-10};
    o.save_steps = false;
    for (int i = 0; i <= 200; ++i) o.t_eval.push_back(0.05 * i);
    const auto r = integrate(sys, 0, y0, 10.0, o);
    REQUIRE(r.ok());
    REQUIRE(r.size() == 201);
    for (Index i = 0; i < r.size(); ++i) CHECK(std::abs(r.y_at(i)[0] - std::sin(r.t[i])) < 5e-7);
}

TEST_CASE("components of very different magnitude with per-component atol", "[dopri5][scaling]") {
    // y0 ~ 1e10 growing slowly, y1 ~ 1e-10 decaying fast: absolute tolerances per component.
    problems::ExponentialSystem sys({0.1, -5.0});
    const std::vector<Real> y0{1e10, 1e-10};
    Options o;
    o.method = Method::Dopri5;
    o.rtol = 1e-9;
    o.atol = {1e-3, 1e-25};
    const auto r = integrate(sys, 0, y0, 4.0, o);
    REQUIRE(r.ok());
    CHECK(std::abs(r.y_last[0] / sys.exact(0, 4, 1e10) - 1) < 1e-8);
    CHECK(std::abs(r.y_last[1] / sys.exact(1, 4, 1e-10) - 1) < 1e-7);
}

TEST_CASE("solution decaying to (nearly) zero keeps relative accuracy while above atol", "[dopri5][scaling]") {
    problems::ExponentialSystem sys({-1.0});
    const std::vector<Real> y0{1.0};
    Options o;
    o.method = Method::Dopri5;
    o.rtol = 1e-8;
    o.atol = {1e-30};
    const auto r = integrate(sys, 0, y0, 60.0, o);  // y(60) ~ 8.8e-27
    REQUIRE(r.ok());
    // The controller bounds the *local* relative error per step by ~rtol; for this
    // problem global errors add up linearly, so the bound is ~ n_steps * rtol.
    const Real rel = std::abs(r.y_last[0] / std::exp(-60.0) - 1);
    CAPTURE(rel, r.stats.n_steps);
    CHECK(rel < 2.0 * static_cast<Real>(r.stats.n_steps) * o.rtol);
    CHECK(rel < 1e-4);
}

TEST_CASE("compensated summation removes round-off accumulation over 1e6 steps", "[rk4][floating]") {
    // y' = 3e-11, 1e6 RK4 steps of h = 1: each increment 3e-11 is not a multiple of ulp(1).
    problems::TinyIncrement sys(3e-11);
    const std::vector<Real> y0{1.0};
    Options o;
    o.method = Method::Rk4;
    o.rk4_step = 1.0;
    o.save_steps = false;
    o.max_steps = 2'000'000;
    const auto r = integrate(sys, 0, y0, 1e6, o);
    REQUIRE(r.ok());
    const Real exact = 1.0 + 3e-5;
    // naive summation would be off by up to ~1e6 * ulp/2 ~ 1e-10
    CHECK(std::abs(r.y_last[0] - exact) <= 4 * kEps);
}

TEST_CASE("a deliberately overlarge initial step is rejected and recovered", "[dopri5][failure]") {
    problems::ExponentialSystem sys({-1000.0});
    const std::vector<Real> y0{1.0};
    Options o;
    o.method = Method::Dopri5;
    o.rtol = 1e-6;
    o.atol = {1e-12};
    o.h0 = 10.0;  // h * lambda = -1e4: far outside the stability region
    const auto r = integrate(sys, 0, y0, 0.01, o);
    REQUIRE(r.ok());
    CHECK(r.stats.n_reject_error + r.stats.n_reject_nonfinite > 0);
    CHECK(std::abs(r.y_last[0] - std::exp(-10.0)) < 1e-6 * std::exp(-10.0) * 100 + 1e-12);
}
