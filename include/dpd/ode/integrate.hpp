// High-level integration driver: output handling, event location and
// automatic explicit -> implicit switching.
#pragma once

#include <functional>
#include <string>

#include "dpd/ode/options.hpp"
#include "dpd/ode/system.hpp"

namespace dpd::ode {

// Root of g(t, y) = 0. direction: +1 only upward crossings (g: - -> +),
// -1 only downward, 0 both. Terminal events stop the integration at the root.
struct Event {
    std::function<Real(Real, std::span<const Real>)> g;
    int direction = 0;
    bool terminal = false;
    std::string name;
};

struct EventHit {
    Index event;  // index into the events list
    Real t;
    int direction;  // +1 upward, -1 downward
    std::vector<Real> y;
};

struct Result {
    Status status = Status::InvalidInput;
    std::string message;
    Index n = 0;                 // system size
    std::vector<Real> t;         // output times
    std::vector<Real> y;         // row-major, t.size() x n
    std::vector<EventHit> events;
    Real t_last = 0;             // last valid time (event time for terminal events)
    std::vector<Real> y_last;    // state at t_last
    Stats stats;
    Method final_method = Method::Dopri5;

    [[nodiscard]] Index size() const noexcept { return t.size(); }
    [[nodiscard]] std::span<const Real> y_at(Index i) const { return {y.data() + i * n, n}; }
    [[nodiscard]] bool ok() const noexcept { return is_success(status); }
};

// Integrates y' = f(t, y) from t0 to t_end (t_end > t0; forward integration only).
Result integrate(const OdeSystem& sys, Real t0, std::span<const Real> y0, Real t_end, const Options& opt,
                 std::span<const Event> events = {});

}  // namespace dpd::ode
