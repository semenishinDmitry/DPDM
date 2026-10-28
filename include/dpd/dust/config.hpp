// Run configuration for the dust model (TOML I/O).
#pragma once

#include <limits>
#include <string>
#include <vector>

#include "dpd/dust/model.hpp"
#include "dpd/ode/types.hpp"

namespace dpd::dust {

struct SolverConfig {
    ode::Method method = ode::Method::Auto;
    double rtol = 1e-8;
    // Per-component absolute tolerances in code units [r, v_r, j, z, v_z, a/a0].
    std::vector<double> atol{1e-12, 1e-16, 1e-12, 1e-12, 1e-16, 1e-12};
    double h_max = std::numeric_limits<double>::infinity();  // code time units
    std::uint64_t max_steps = 20'000'000;
    ode::LinearSolverKind linear_solver = ode::LinearSolverKind::DenseLU;
    int bdf_max_order = 5;
    double rk4_step = 0;  // code time units, only for method = rk4
};

struct RunConfig {
    std::string name = "run";
    std::string description;
    DiskParams disk;
    GrainParams grain;
    double r_init_au = 10.0;
    double z_init_H = 3.0;        // z(0) = z_init_H * H(r_init) (Sec. 2.6)
    double vr_init_cms = 0.0;
    double vz_init_cms = 0.0;
    double r_in_au = 0.05;        // inner disk edge: terminal event (Sec. 2.6: 0.05-0.1 au)
    double t_end_yr = 1e9;        // safety limit; runs normally end at r_in
    double settle_eps = 0.005;    // settled when |z| <= settle_eps * z_init (calibrated; see docs)
    SolverConfig solver;
    bool save_steps = true;
    std::string beta_table_path;  // CSV (r_au, beta_K) for disk.model = "tabulated_beta"
};

// Parse/serialize. Relative paths inside the file are resolved against base_dir.
[[nodiscard]] RunConfig load_config(const std::string& path, const std::vector<std::string>& overrides = {});
[[nodiscard]] RunConfig parse_config(const std::string& toml_text, const std::string& base_dir = ".",
                                     const std::vector<std::string>& overrides = {});
[[nodiscard]] std::string to_toml(const RunConfig& cfg);
void save_config(const RunConfig& cfg, const std::string& path);

// Two-column CSV (r_au, beta); '#' comments and a non-numeric header line allowed.
void load_beta_table(const std::string& path, DiskParams& disk);

[[nodiscard]] ode::Method method_from_string(const std::string& s);
[[nodiscard]] ode::LinearSolverKind linear_solver_from_string(const std::string& s);

}  // namespace dpd::dust
