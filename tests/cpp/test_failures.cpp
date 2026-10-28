#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <cmath>

#include "dpd/ode/integrate.hpp"
#include "dpd/ode/test_problems.hpp"

using namespace dpd::ode;

TEST_CASE("finite-time blow-up is reported as step-size underflow", "[failure]") {
    problems::BlowUp sys;
    const std::vector<Real> y0{1.0};
    for (Method m : {Method::Dopri5, Method::Bdf, Method::Auto}) {
        CAPTURE(to_string(m));
        Options o;
        o.method = m;
        o.rtol = 1e-8;
        o.atol = {1e-10};
        const auto r = integrate(sys, 0, y0, 2.0, o);
        CHECK_FALSE(r.ok());
        CHECK((r.status == Status::StepSizeUnderflow || r.status == Status::NonFiniteState ||
               r.status == Status::NewtonFailure));
        // The solver may stop marginally past the singularity at t = 1 (the
        // numerical solution is finite there); it must not get far beyond it.
        CHECK(r.t_last < 1.0 + 1e-6);
        CHECK(r.t_last > 0.999);
        CHECK(std::isfinite(r.y_last[0]));
        CHECK_FALSE(r.message.empty());
    }
}

TEST_CASE("NaN from the RHS is never propagated into accepted states", "[failure][floating]") {
    for (bool domain : {false, true}) {
        CAPTURE(domain);
        problems::NanBeyondZero sys(domain);
        const std::vector<Real> y0{1.0};
        for (Method m : {Method::Dopri5, Method::Bdf}) {
            CAPTURE(to_string(m));
            Options o;
            o.method = m;
            o.fallback_to_implicit = false;
            const auto r = integrate(sys, 0, y0, 2.0, o);
            CHECK_FALSE(r.ok());
            if (domain)
                CHECK((r.status == Status::DomainViolation || r.status == Status::StepSizeUnderflow ||
                       r.status == Status::NewtonFailure));
            else
                CHECK((r.status == Status::NonFiniteState || r.status == Status::StepSizeUnderflow ||
                       r.status == Status::NewtonFailure));
            CHECK(r.t_last <= 1.0 + 1e-6);
            CHECK(r.t_last > 0.9);
            CHECK(std::isfinite(r.y_last[0]));
            for (Real v : r.y) CHECK(std::isfinite(v));
        }
    }
}

TEST_CASE("lost increments are detected; persistent stagnation is an error", "[failure][floating]") {
    // y' = 1e-30, y(0) = 1: every increment (1e-30 * h) is far below ulp(1) / 2.
    problems::TinyIncrement sys(1e-30);
    const std::vector<Real> y0{1.0};
    Options o;
    o.method = Method::Rk4;
    o.rk4_step = 1.0;
    o.max_stagnant_steps = 50;
    const auto r = integrate(sys, 0, y0, 1000, o);
    // RK4 has no stagnation check (fixed step, verification only) -> runs through
    CHECK(r.status == Status::Success);
    o.method = Method::Dopri5;
    o.h_max = 1.0;
    o.rtol = 1e-6;
    o.atol = {1e-6};
    const auto r2 = integrate(sys, 0, y0, 1000, o);
    CHECK(r2.status == Status::TimeResolutionLost);
    CHECK(r2.stats.n_lost_increment >= 50);
    CHECK(r2.message.find("lost to rounding") != std::string::npos);
    o.max_stagnant_steps = 0;  // diagnostic only
    const auto r3 = integrate(sys, 0, y0, 1000, o);
    CHECK(r3.status == Status::Success);
    CHECK(r3.stats.n_lost_increment > 0);
}

TEST_CASE("step underflow when |t| is so large that the needed h is unrepresentable", "[failure][floating]") {
    problems::ExponentialSystem sys({-1e3});
    const std::vector<Real> y0{1.0};
    Options o;
    o.method = Method::Dopri5;
    o.rtol = 1e-8;
    o.atol = {1e-12};
    o.fallback_to_implicit = false;
    const Real t0 = 1e20;  // spacing of doubles near 1e20 is 16384
    const auto r = integrate(sys, t0, y0, t0 + 1e6, o);
    CHECK((r.status == Status::StepSizeUnderflow || r.status == Status::TimeResolutionLost));
}

TEST_CASE("max_steps, cancellation and invalid input", "[failure]") {
    problems::HarmonicOscillator sys(1.0);
    const std::vector<Real> y0{1, 0};
    Options o;
    o.max_steps = 10;
    auto r = integrate(sys, 0, y0, 1000, o);
    CHECK(r.status == Status::MaxStepsExceeded);
    CHECK(r.stats.n_steps == 10);

    std::atomic<bool> cancel{true};
    Options oc;
    oc.cancel = &cancel;
    r = integrate(sys, 0, y0, 1000, oc);
    CHECK(r.status == Status::Cancelled);

    Options bad;
    bad.rtol = 0;
    CHECK(integrate(sys, 0, y0, 1, bad).status == Status::InvalidInput);
    bad = Options{};
    bad.atol = {1, 2, 3};
    CHECK(integrate(sys, 0, y0, 1, bad).status == Status::InvalidInput);
    bad = Options{};
    CHECK(integrate(sys, 1, y0, 0, bad).status == Status::InvalidInput);
    const std::vector<Real> ynan{std::nan(""), 0};
    CHECK(integrate(sys, 0, ynan, 1, bad).status == Status::InvalidInput);
    bad.method = Method::Rk4;
    CHECK(integrate(sys, 0, y0, 1, bad).status == Status::InvalidInput);  // rk4_step missing
    bad = Options{};
    bad.t_eval = {0.5, 0.2};
    CHECK(integrate(sys, 0, y0, 1, bad).status == Status::InvalidInput);
}
