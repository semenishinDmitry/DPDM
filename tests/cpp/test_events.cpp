#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numbers>

#include "dpd/ode/integrate.hpp"
#include "dpd/ode/test_problems.hpp"

using namespace dpd::ode;

TEST_CASE("terminal event located to near machine precision", "[events]") {
    problems::ExponentialSystem sys({-1.0});
    const std::vector<Real> y0{1.0};
    for (Method m : {Method::Dopri5, Method::Bdf}) {
        CAPTURE(to_string(m));
        Options o;
        o.method = m;
        o.rtol = 1e-11;
        o.atol = {1e-14};
        std::vector<Event> ev{{[](Real, std::span<const Real> y) { return y[0] - 0.5; }, -1, true, "half"}};
        const auto r = integrate(sys, 0, y0, 10, o, ev);
        REQUIRE(r.status == Status::TerminalEvent);
        REQUIRE(r.events.size() == 1);
        CHECK(std::abs(r.events[0].t - std::log(2.0)) < 1e-9);
        CHECK(r.t.back() == r.events[0].t);
        CHECK(r.t_last == r.events[0].t);
        CHECK(std::abs(r.y_last[0] - 0.5) < 1e-10);
    }
}

TEST_CASE("non-terminal events with direction filtering", "[events]") {
    problems::HarmonicOscillator sys(1.0);  // x = cos t
    const std::vector<Real> y0{1, 0};
    Options o;
    o.rtol = 1e-10;
    o.atol = {1e-12};
    std::vector<Event> ev{{[](Real, std::span<const Real> y) { return y[0]; }, 0, false, "any"},
                          {[](Real, std::span<const Real> y) { return y[0]; }, 1, false, "up"},
                          {[](Real, std::span<const Real> y) { return y[0]; }, -1, false, "down"}};
    const auto r = integrate(sys, 0, y0, 10 * std::numbers::pi, o, ev);
    REQUIRE(r.status == Status::Success);
    int n_any = 0, n_up = 0, n_down = 0;
    for (const auto& h : r.events) {
        // zeros of cos t: (k + 1/2) pi
        const Real k = h.t / std::numbers::pi - 0.5;
        CHECK(std::abs(k - std::round(k)) < 1e-8);
        if (h.event == 0) ++n_any;
        if (h.event == 1) { ++n_up; CHECK(h.direction == 1); }
        if (h.event == 2) { ++n_down; CHECK(h.direction == -1); }
    }
    CHECK(n_any == 10);
    CHECK(n_up == 5);
    CHECK(n_down == 5);
}
