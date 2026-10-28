#include "dpd/ode/integrate.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

#include "dpd/ode/stepper.hpp"

namespace dpd::ode {

std::string_view to_string(Status s) noexcept {
    switch (s) {
        case Status::Success: return "success";
        case Status::TerminalEvent: return "terminal_event";
        case Status::MaxStepsExceeded: return "max_steps_exceeded";
        case Status::StepSizeUnderflow: return "step_size_underflow";
        case Status::TimeResolutionLost: return "time_resolution_lost";
        case Status::NonFiniteState: return "non_finite_state";
        case Status::NewtonFailure: return "newton_failure";
        case Status::LinearSolverFailure: return "linear_solver_failure";
        case Status::DomainViolation: return "domain_violation";
        case Status::Cancelled: return "cancelled";
        case Status::InvalidInput: return "invalid_input";
    }
    return "unknown";
}

std::string_view to_string(Method m) noexcept {
    switch (m) {
        case Method::Dopri5: return "dopri5";
        case Method::Rk4: return "rk4";
        case Method::Bdf: return "bdf";
        case Method::Auto: return "auto";
    }
    return "unknown";
}

std::string_view to_string(LinearSolverKind k) noexcept {
    return k == LinearSolverKind::DenseLU ? "dense_lu" : "gmres";
}

bool is_success(Status s) noexcept { return s == Status::Success || s == Status::TerminalEvent; }

namespace {

// Brent's method for g(t) = 0 on [a, b] with g(a), g(b) of opposite sign (or g(b) == 0).
template <class G>
Real brent_root(G&& g, Real a, Real b, Real fa, Real fb) {
    if (fb == 0) return b;
    if (fa == 0) return a;
    Real c = a, fc = fa, d = b - a, e = d;
    for (int iter = 0; iter < 200; ++iter) {
        if ((fb > 0) == (fc > 0)) {
            c = a;
            fc = fa;
            d = e = b - a;
        }
        if (std::abs(fc) < std::abs(fb)) {
            a = b; b = c; c = a;
            fa = fb; fb = fc; fc = fa;
        }
        const Real tol = 2 * kEps * std::abs(b) + 1e-300;
        const Real m = 0.5 * (c - b);
        if (std::abs(m) <= tol || fb == 0) return b;
        if (std::abs(e) >= tol && std::abs(fa) > std::abs(fb)) {
            Real p, q, r;
            const Real s = fb / fa;
            if (a == c) {
                p = 2 * m * s;
                q = 1 - s;
            } else {
                q = fa / fc;
                r = fb / fc;
                p = s * (2 * m * q * (q - r) - (b - a) * (r - 1));
                q = (q - 1) * (r - 1) * (s - 1);
            }
            if (p > 0) q = -q; else p = -p;
            if (2 * p < std::min(3 * m * q - std::abs(tol * q), std::abs(e * q))) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = d;
            }
        } else {
            d = m;
            e = d;
        }
        a = b;
        fa = fb;
        b += (std::abs(d) > tol) ? d : (m > 0 ? tol : -tol);
        fb = g(b);
    }
    return b;
}

Status validate(const OdeSystem& sys, Real t0, std::span<const Real> y0, Real t_end, const Options& opt,
                std::string& msg) {
    const Index n = sys.size();
    if (n == 0) { msg = "system size is zero"; return Status::InvalidInput; }
    if (y0.size() != n) { msg = "y0 size does not match system size"; return Status::InvalidInput; }
    if (!all_finite(y0)) { msg = "y0 contains non-finite values"; return Status::InvalidInput; }
    if (!std::isfinite(t0) || !std::isfinite(t_end)) { msg = "t0/t_end must be finite"; return Status::InvalidInput; }
    if (t_end < t0) { msg = "only forward integration (t_end >= t0) is supported"; return Status::InvalidInput; }
    if (!(opt.rtol >= 10 * kEps) && opt.method != Method::Rk4) {
        msg = "rtol must be >= 10 * machine epsilon";
        return Status::InvalidInput;
    }
    if (opt.atol.empty() || (opt.atol.size() != 1 && opt.atol.size() != n)) {
        msg = "atol must have 1 or n entries";
        return Status::InvalidInput;
    }
    for (Real a : opt.atol)
        if (!(a >= 0) || !std::isfinite(a)) { msg = "atol entries must be finite and >= 0"; return Status::InvalidInput; }
    if (!(opt.h_max > 0)) { msg = "h_max must be > 0"; return Status::InvalidInput; }
    if (opt.method == Method::Rk4 && !(opt.rk4_step > 0)) { msg = "rk4_step must be > 0"; return Status::InvalidInput; }
    if (opt.linear_solver == LinearSolverKind::Gmres && opt.gmres_restart < 1) {
        msg = "gmres_restart must be >= 1";
        return Status::InvalidInput;
    }
    if (!std::is_sorted(opt.t_eval.begin(), opt.t_eval.end())) { msg = "t_eval must be sorted"; return Status::InvalidInput; }
    return Status::Success;
}

}  // namespace

Result integrate(const OdeSystem& sys, Real t0, std::span<const Real> y0, Real t_end, const Options& opt,
                 std::span<const Event> events) {
    Result res;
    const Index n = sys.size();
    res.n = n;
    if (const Status v = validate(sys, t0, y0, t_end, opt, res.message); v != Status::Success) {
        res.status = v;
        return res;
    }
    StepContext ctx(sys, opt, res.stats);
    const Method start = opt.method == Method::Auto ? Method::Dopri5 : opt.method;
    std::unique_ptr<Stepper> stepper = make_stepper(start, ctx);
    stepper->init(t0, y0, opt.h0);

    auto record = [&](Real t, std::span<const Real> y) {
        res.t.push_back(t);
        res.y.insert(res.y.end(), y.begin(), y.end());
    };
    std::size_t ie = 0;
    while (ie < opt.t_eval.size() && opt.t_eval[ie] < t0) ++ie;
    if (opt.save_steps || (ie < opt.t_eval.size() && opt.t_eval[ie] == t0)) record(t0, y0);
    while (ie < opt.t_eval.size() && opt.t_eval[ie] == t0) ++ie;

    std::vector<Real> ybuf(n), g_old(events.size());
    for (std::size_t k = 0; k < events.size(); ++k) g_old[k] = events[k].g(t0, y0);
    std::vector<EventHit> hits;
    bool fallback_used = false;
    res.status = Status::Success;

    auto switch_to_bdf = [&](std::string reason) {
        const Real t = stepper->t();
        const Real h = stepper->h_next();
        std::vector<Real> y(stepper->y().begin(), stepper->y().end());
        res.stats.switches.push_back({t, stepper->method(), Method::Bdf, std::move(reason)});
        stepper = make_stepper(Method::Bdf, ctx);
        stepper->init(t, y, h);
    };

    while (stepper->t() < t_end) {
        if (opt.cancel && opt.cancel->load(std::memory_order_relaxed)) {
            res.status = Status::Cancelled;
            res.message = "cancelled by caller";
            break;
        }
        if (res.stats.n_steps >= opt.max_steps) {
            res.status = Status::MaxStepsExceeded;
            res.message = "max_steps reached at t=" + std::to_string(stepper->t());
            break;
        }
        const Status s = stepper->step(t_end);
        if (s != Status::Success) {
            if (opt.method == Method::Auto && opt.fallback_to_implicit && !fallback_used &&
                stepper->method() == Method::Dopri5 &&
                (s == Status::StepSizeUnderflow || s == Status::NonFiniteState)) {
                fallback_used = true;
                switch_to_bdf("explicit method failed (" + std::string(to_string(s)) + "): " +
                              stepper->failure_detail());
                continue;
            }
            res.status = s;
            res.message = stepper->failure_detail();
            break;
        }
        const Real ta = stepper->t_old();
        const Real tb = stepper->t();
        const auto yb = stepper->y();

        // Event detection on [ta, tb].
        hits.clear();
        Real t_stop = tb;
        bool terminal = false;
        const int nsub = std::max(0, opt.event_samples) + 1;
        for (std::size_t k = 0; k < events.size(); ++k) {
            const Event& ev = events[k];
            auto gt = [&](Real t) {
                stepper->dense(t, ybuf);
                return ev.g(t, ybuf);
            };
            Real sa = ta, ga = g_old[k];
            for (int q = 1; q <= nsub; ++q) {
                const Real sb = q == nsub ? tb : ta + (tb - ta) * q / nsub;
                const Real gb = q == nsub ? ev.g(tb, yb) : gt(sb);
                const bool up = ga < 0 && gb >= 0;
                const bool down = ga > 0 && gb <= 0;
                if ((up && ev.direction >= 0) || (down && ev.direction <= 0)) {
                    const Real tr = brent_root(gt, sa, sb, ga, gb);
                    stepper->dense(tr, ybuf);
                    hits.push_back({k, tr, up ? 1 : -1, ybuf});
                }
                sa = sb;
                ga = gb;
            }
            g_old[k] = ga;
        }
        std::sort(hits.begin(), hits.end(), [](const EventHit& a, const EventHit& b) { return a.t < b.t; });
        for (auto& h : hits) {
            if (terminal) break;
            ++res.stats.n_events;
            res.events.push_back(h);
            if (events[h.event].terminal) {
                terminal = true;
                t_stop = h.t;
            }
        }

        while (ie < opt.t_eval.size() && opt.t_eval[ie] <= t_stop) {
            stepper->dense(opt.t_eval[ie], ybuf);
            record(opt.t_eval[ie], ybuf);
            ++ie;
        }
        if (opt.save_steps && (res.t.empty() || res.t.back() < t_stop)) {
            if (terminal) {
                stepper->dense(t_stop, ybuf);
                record(t_stop, ybuf);
            } else {
                record(tb, yb);
            }
        }
        if (opt.progress_t) opt.progress_t->store(t_stop, std::memory_order_relaxed);
        if (opt.on_step) opt.on_step(tb, yb);
        if (terminal) {
            res.status = Status::TerminalEvent;
            res.message = "terminal event '" + events[res.events.back().event].name + "'";
            break;
        }
        if (opt.method == Method::Auto && stepper->method() == Method::Dopri5 && stepper->stiffness_detected())
            switch_to_bdf(stepper->stiffness_reason());
    }
    // The last valid state is always available, even on failure or with t_eval-only output.
    if (res.status == Status::TerminalEvent) {
        res.t_last = res.events.back().t;
        res.y_last = res.events.back().y;
    } else {
        res.t_last = stepper->t();
        res.y_last.assign(stepper->y().begin(), stepper->y().end());
    }
    res.final_method = stepper->method();
    return res;
}

}  // namespace dpd::ode
