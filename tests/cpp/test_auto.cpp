#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numbers>

#include "dpd/ode/integrate.hpp"
#include "dpd/ode/test_problems.hpp"
#include "reference_data.hpp"
#include "stiff_linear.hpp"

using namespace dpd::ode;

TEST_CASE("auto: switches once to BDF on a stiff linear system and stays accurate", "[auto][stiff]") {
    testing_util::StiffLinear sl(-1.0, -1e2, -1e5);
    problems::LinearSystem sys(sl.matrix());
    const std::vector<Real> y0{1.0, 2.0, -1.0};
    Options o;
    o.method = Method::Auto;
    o.rtol = 1e-8;
    o.atol = {1e-12};
    o.save_steps = false;
    const auto ra = integrate(sys, 0, y0, 20.0, o);
    REQUIRE(ra.status == Status::Success);
    REQUIRE(ra.stats.switches.size() == 1);
    CHECK(ra.stats.switches[0].from == Method::Dopri5);
    CHECK(ra.stats.switches[0].to == Method::Bdf);
    CHECK_FALSE(ra.stats.switches[0].reason.empty());
    CHECK(ra.final_method == Method::Bdf);
    const auto ex = sl.exact(20.0, y0);
    const Real bound = 2.0 * static_cast<Real>(ra.stats.n_steps) * o.rtol;  // see test_bdf.cpp
    for (int i = 0; i < 3; ++i) CHECK(std::abs(ra.y_last[i] - ex[i]) < bound * std::abs(ex[i]) + 10 * o.atol[0]);

    o.method = Method::Dopri5;
    const auto re = integrate(sys, 0, y0, 20.0, o);
    REQUIRE(re.status == Status::Success);
    CHECK(ra.stats.n_rhs * 10 < re.stats.n_rhs);  // switching pays off
}

TEST_CASE("auto: no switching on non-stiff problems", "[auto]") {
    for (Real rtol : {1e-4, 1e-6, 1e-8, 1e-10}) {
        CAPTURE(rtol);
        Options o;
        o.method = Method::Auto;
        o.rtol = rtol;
        o.atol = {rtol * 1e-2};
        o.save_steps = false;
        problems::HarmonicOscillator ho(2.0);
        const std::vector<Real> y0{1, 0};
        auto r = integrate(ho, 0, y0, 100, o);
        REQUIRE(r.ok());
        CHECK(r.stats.switches.empty());
        problems::ExponentialSystem ex({-0.5, 0.2});
        const std::vector<Real> y1{1, 1};
        r = integrate(ex, 0, y1, 20, o);
        REQUIRE(r.ok());
        CHECK(r.stats.switches.empty());
        problems::VanDerPol vdp(1.0);
        const std::vector<Real> y2{2, 0};
        r = integrate(vdp, 0, y2, 30, o);
        REQUIRE(r.ok());
        CHECK(r.stats.switches.empty());
    }
}

TEST_CASE("auto: stiff benchmarks are detected and solved", "[auto][stiff]") {
    {
        problems::Robertson rob;
        Options o;
        o.rtol = 1e-8;
        o.atol = {1e-14, 1e-18, 1e-14};
        o.save_steps = false;
        o.t_eval = {refdata::kRoberT[refdata::kRoberN - 1]};
        const std::vector<Real> y0{1, 0, 0};
        const auto r = integrate(rob, 0, y0, refdata::kRoberT[refdata::kRoberN - 1], o);
        REQUIRE(r.status == Status::Success);
        CHECK(r.stats.switches.size() == 1);
        for (int i = 0; i < 3; ++i) {
            const Real ref = refdata::kRoberY[refdata::kRoberN - 1][i];
            CHECK(std::abs(r.y_last[i] - ref) <= 2e-6 * std::abs(ref) + 1e-13);
        }
    }
    {
        // Auto must be as accurate as BDF-only at the same tolerance.
        problems::VanDerPol vdp(1000.0);
        Options o;
        o.rtol = 1e-7;
        o.atol = {1e-8};
        o.save_steps = false;
        o.t_eval = {refdata::kVdpT[0]};
        const std::vector<Real> y0{2, 0};
        const auto r = integrate(vdp, 0, y0, refdata::kVdpT[0], o);
        REQUIRE(r.status == Status::Success);
        CHECK(r.stats.switches.size() == 1);
        o.method = Method::Bdf;
        const auto rb = integrate(vdp, 0, y0, refdata::kVdpT[0], o);
        REQUIRE(rb.status == Status::Success);
        const Real ea = std::abs(r.y_at(0)[0] - refdata::kVdpY[0][0]);
        const Real eb = std::abs(rb.y_at(0)[0] - refdata::kVdpY[0][0]);
        CAPTURE(ea, eb);
        CHECK(ea < 3 * eb + 1e-7);
        CHECK(ea < 1e-3);
    }
}

TEST_CASE("auto: explicit failure falls back to BDF once", "[auto][failure]") {
    // A huge user-supplied minimum step makes the explicit method fail immediately on
    // a stiff problem; with fallback enabled the run completes with BDF.
    testing_util::StiffLinear sl(-1.0, -1e3, -1e6);
    problems::LinearSystem sys(sl.matrix());
    // Only the slow mode is excited: the solution is smooth, so BDF can respect h_min.
    const std::vector<Real> y0{sl.Q(0, 0), sl.Q(1, 0), sl.Q(2, 0)};
    Options o;
    o.rtol = 1e-6;
    o.atol = {1e-10};
    o.stiff_detect_steps = 1000000;  // disable detection so only the fallback can trigger
    o.stiff_window = 64;
    o.stiff_window_fraction = 2.0;
    o.stiff_rho_interval = 0;
    o.h_min = 1e-4;  // explicit stability needs h < 3.3e-6
    o.save_steps = false;
    auto r = integrate(sys, 0, y0, 1.0, o);
    std::string reasons;
    for (const auto& w : r.stats.switches) reasons += w.reason + "; ";
    CAPTURE(r.message, r.stats.switches.size(), r.t_last, reasons);
    REQUIRE(r.status == Status::Success);
    REQUIRE(r.stats.switches.size() == 1);
    CHECK(r.stats.switches[0].reason.find("explicit method failed") != std::string::npos);
    const auto ex = sl.exact(1.0, y0);
    for (int i = 0; i < 3; ++i) CHECK(std::abs(r.y_last[i] - ex[i]) < 1e-5);
    o.fallback_to_implicit = false;
    r = integrate(sys, 0, y0, 1.0, o);
    CHECK(r.status == Status::StepSizeUnderflow);
}
