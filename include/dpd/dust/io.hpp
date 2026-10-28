// Output of dust-model runs (CSV trajectories, JSON summaries).
#pragma once

#include <iosfwd>
#include <string>

#include "dpd/dust/run.hpp"

namespace dpd::dust {

// Columns: t_yr, r_au, z_au, vr_cms, vz_cms, vphi_cms, a_cm, St, tau_s, stokes,
// lambda_cm, Re, rho_gcc, T_K, H_au, ur_cms, uphi_cms, beta, dv_cms
void write_csv(const RunResult& r, std::ostream& os);
void write_csv(const RunResult& r, const std::string& path);
void write_summary_json(const RunResult& r, const RunConfig& cfg, std::ostream& os);
void write_summary_json(const RunResult& r, const RunConfig& cfg, const std::string& path);

}  // namespace dpd::dust
