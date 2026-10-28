#include "dpd/dust/run.hpp"

#include <chrono>
#include <cmath>
#include <thread>

namespace dpd::dust {

namespace {
constexpr double kInf_ = std::numeric_limits<double>::infinity();
}  // namespace

ode::Options make_options(const RunConfig& cfg) {
    ode::Options o;
    o.method = cfg.solver.method;
    o.rtol = cfg.solver.rtol;
    o.atol = cfg.solver.atol;
    o.h_max = cfg.solver.h_max;
    o.max_steps = cfg.solver.max_steps;
    o.linear_solver = cfg.solver.linear_solver;
    o.bdf_max_order = cfg.solver.bdf_max_order;
    o.rk4_step = cfg.solver.rk4_step;
    o.save_steps = true;  // summary quantities are evaluated on the accepted steps
    return o;
}

RunResult run(const RunConfig& cfg, const RunControl& ctl, bool keep_trajectory) {
    const auto wall0 = std::chrono::steady_clock::now();
    RunResult out;
    RunSummary& S = out.summary;
    const DustModel model(cfg.disk, cfg.grain);
    const Units& u = model.units();
    const double z0 = cfg.grain.sedimentation ? cfg.z_init_H * model.disk().scale_height(cfg.r_init_au) : 0.0;
    const std::vector<double> y0 = model.initial_state(cfg.r_init_au, z0, cfg.vr_init_cms, cfg.vz_init_cms);

    std::vector<ode::Event> events;
    const double r_in = cfg.r_in_au;
    events.push_back({[r_in](double, std::span<const double> y) { return y[DustModel::R] - r_in; }, -1, true,
                      "inner_edge"});
    const bool settle = cfg.grain.sedimentation && z0 != 0;
    const double band = cfg.settle_eps * std::abs(z0);
    if (settle) {
        // Two monotone event functions (z - band, z + band): a fast pass through
        // the band within one step changes the sign of both, which a single
        // z^2 - band^2 function would miss.
        events.push_back({[band](double, std::span<const double> y) { return y[DustModel::Z] - band; }, 0, false,
                          "settle_upper"});
        events.push_back({[band](double, std::span<const double> y) { return y[DustModel::Z] + band; }, 0, false,
                          "settle_lower"});
    }

    ode::Options opt = make_options(cfg);
    opt.event_samples = 4;
    opt.cancel = ctl.cancel;
    if (ctl.progress) {
        const double span_r = cfg.r_init_au - r_in;
        auto* prog = ctl.progress;
        opt.on_step = [prog, span_r, r0 = cfg.r_init_au](double, std::span<const double> y) {
            const double f = (r0 - y[DustModel::R]) / span_r;
            prog->store(std::clamp(f, 0.0, 1.0), std::memory_order_relaxed);
        };
    }
    const double t_end = cfg.t_end_yr / u.t0_yr();
    ode::Result res = ode::integrate(model, 0.0, y0, t_end, opt, events);

    S.status = res.status;
    S.message = res.message;
    S.stats = res.stats;
    S.final_method = res.final_method;
    S.t_final_yr = res.t_last * u.t0_yr();
    if (!res.y_last.empty()) {
        S.r_final_au = res.y_last[DustModel::R];
        S.a_final_cm = res.y_last[DustModel::A] * u.a0_cm;
    }
    // Events: entering the band = z - band crossing downward or z + band crossing upward.
    int last_in = -1, last_out = -1;
    for (std::size_t k = 0; k < res.events.size(); ++k) {
        const auto& e = res.events[k];
        if (e.event == 0) {
            S.reached_inner_edge = true;
            S.t_drift_yr = e.t * u.t0_yr();
            continue;
        }
        const bool enter = (e.event == 1 && e.direction < 0) || (e.event == 2 && e.direction > 0);
        if (enter) {
            if (std::isnan(S.t_settle_first_yr)) S.t_settle_first_yr = e.t * u.t0_yr();
            last_in = static_cast<int>(k);
        } else {
            last_out = static_cast<int>(k);
        }
    }
    if (settle && last_in > last_out && !res.y_last.empty() && std::abs(res.y_last[DustModel::Z]) <= band)
        S.t_settle_last_yr = res.events[static_cast<std::size_t>(last_in)].t * u.t0_yr();

    // Trapping detection
    if (res.status == ode::Status::Success && !S.reached_inner_edge && res.t.size() > 2) {
        const double t_end_code = res.t.back();
        double rmin = kInf_, rmax = -kInf_;
        for (std::size_t i = 0; i < res.t.size(); ++i)
            if (res.t[i] >= 0.9 * t_end_code) {
                rmin = std::min(rmin, res.y_at(i)[DustModel::R]);
                rmax = std::max(rmax, res.y_at(i)[DustModel::R]);
            }
        if (rmax - rmin < 1e-3 * rmax) {
            S.trapped = true;
            S.r_trap_au = res.y_last[DustModel::R];
        }
    }

    // Trajectory-wide quantities
    const std::size_t n = res.t.size();
    double vmax = -1, amax = 0, remax = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto yi = res.y_at(i);
        const Diagnostics D = diagnose(model, res.t[i], yi);
        if (std::abs(D.vr_cms) > vmax) {
            vmax = std::abs(D.vr_cms);
            S.r_at_vr_max_au = D.r_au;
        }
        amax = std::max(amax, D.a_cm);
        if (D.stokes) {
            S.stokes_regime_seen = true;
            remax = std::max(remax, D.Re);
        } else {
            S.epstein_regime_seen = true;
        }
        if (model.disk().beta_extrapolated(D.r_au)) S.beta_extrapolated = true;
        if (keep_trajectory) out.diag.push_back(D);
    }
    S.vr_max_cms = vmax;
    S.a_max_cm = amax;
    S.max_Re = S.stokes_regime_seen ? remax : kNaN;
    if (S.stokes_regime_seen && remax > 1)
        S.warnings.push_back("Stokes drag used with Re up to " + std::to_string(remax) +
                             " > 1: Eq. (2) is outside its validity range (paper uses Re < 1 law only)");
    if (S.beta_extrapolated) S.warnings.push_back("tabulated beta_K evaluated outside its radial range (held constant)");
    if (!S.reached_inner_edge && is_success(S.status) && !S.trapped)
        S.warnings.push_back("particle did not reach r_in before t_end");
    if (keep_trajectory) {
        out.t = std::move(res.t);
        out.y = std::move(res.y);
    }
    S.wall_time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
    return out;
}

std::vector<RunResult> run_ensemble(const std::vector<RunConfig>& cfgs, unsigned threads, bool keep,
                                    const std::atomic<bool>* cancel) {
    std::vector<RunResult> out(cfgs.size());
    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
    threads = std::min<unsigned>(threads, static_cast<unsigned>(std::max<std::size_t>(1, cfgs.size())));
    std::atomic<std::size_t> next{0};
    auto worker = [&] {
        RunControl ctl;
        ctl.cancel = cancel;
        for (std::size_t i; (i = next.fetch_add(1)) < cfgs.size();) {
            try {
                out[i] = run(cfgs[i], ctl, keep);
            } catch (const std::exception& e) {
                out[i].summary.status = ode::Status::InvalidInput;
                out[i].summary.message = e.what();
            }
        }
    };
    if (threads == 1) {
        worker();
        return out;
    }
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (unsigned k = 0; k < threads; ++k) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
    return out;
}

}  // namespace dpd::dust
