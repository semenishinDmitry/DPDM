// Running the dust model: events (inner edge, settling), summary quantities,
// diagnostics and parallel ensembles.
#pragma once

#include <atomic>
#include <limits>

#include "dpd/dust/config.hpp"
#include "dpd/ode/integrate.hpp"

namespace dpd::dust {

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct RunSummary {
    ode::Status status = ode::Status::InvalidInput;
    std::string message;
    bool reached_inner_edge = false;
    double t_final_yr = kNaN;
    double r_final_au = kNaN;
    double a_final_cm = kNaN;
    double t_drift_yr = kNaN;          // time to reach r_in (NaN if not reached)
    double t_settle_first_yr = kNaN;   // first entry into |z| <= eps z_init
    double t_settle_last_yr = kNaN;    // last entry, if |z| stays in the band to the end
    double vr_max_cms = kNaN;          // max |v_r| over the recorded trajectory
    double r_at_vr_max_au = kNaN;
    double a_max_cm = kNaN;
    double max_Re = kNaN;              // Stokes-law validity requires Re < 1
    bool stokes_regime_seen = false;
    bool epstein_regime_seen = false;
    bool beta_extrapolated = false;    // tabulated beta used outside its radial range
    std::vector<std::string> warnings;
    ode::Stats stats;
    ode::Method final_method = ode::Method::Dopri5;
    double wall_time_s = 0;
};

struct RunResult {
    RunSummary summary;
    std::vector<double> t;   // code time units
    std::vector<double> y;   // row-major (t.size() x 6), code units
    std::vector<Diagnostics> diag;  // derived physical quantities per row (CGS / au / yr)
};

struct RunControl {
    const std::atomic<bool>* cancel = nullptr;
    std::atomic<double>* progress = nullptr;  // fraction in [0, 1] (radial progress towards r_in)
};

[[nodiscard]] ode::Options make_options(const RunConfig& cfg);
[[nodiscard]] RunResult run(const RunConfig& cfg, const RunControl& ctl = {}, bool keep_trajectory = true);

// Runs independent configurations concurrently (threads = 0: hardware concurrency).
[[nodiscard]] std::vector<RunResult> run_ensemble(const std::vector<RunConfig>& cfgs, unsigned threads,
                                                  bool keep_trajectories, const std::atomic<bool>* cancel = nullptr);

}  // namespace dpd::dust
