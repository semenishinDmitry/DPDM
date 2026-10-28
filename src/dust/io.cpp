#include "dpd/dust/io.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace dpd::dust {

void write_csv(const RunResult& r, std::ostream& os) {
    os << "t_yr,r_au,z_au,vr_cms,vz_cms,vphi_cms,a_cm,St,tau_s,stokes,lambda_cm,Re,rho_gcc,T_K,H_au,ur_cms,"
          "uphi_cms,beta,dv_cms,Z_charge,hall\n";
    os << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const Diagnostics& d : r.diag) {
        os << d.t_yr << ',' << d.r_au << ',' << d.z_au << ',' << d.vr_cms << ',' << d.vz_cms << ',' << d.vphi_cms
           << ',' << d.a_cm << ',' << d.St << ',' << d.tau_s << ',' << (d.stokes ? 1 : 0) << ',' << d.lambda_cm
           << ',' << d.Re << ',' << d.rho << ',' << d.T << ',' << d.H_au << ',' << d.ur_cms << ',' << d.uphi_cms
           << ',' << d.beta << ',' << d.dv_cms << ',' << d.Z_charge << ',' << d.hall << '\n';
    }
}

void write_csv(const RunResult& r, const std::string& path) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot write '" + path + "'");
    write_csv(r, f);
}

namespace {
void num(std::ostream& os, double v) {
    if (std::isfinite(v))
        os << v;
    else
        os << "null";
}
std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        if (c == '\n') {
            o += "\\n";
            continue;
        }
        o += c;
    }
    return o;
}
}  // namespace

void write_summary_json(const RunResult& r, const RunConfig& cfg, std::ostream& os) {
    const RunSummary& s = r.summary;
    const auto& st = s.stats;
    os << std::setprecision(12);
    os << "{\n  \"name\": \"" << esc(cfg.name) << "\",\n";
    os << "  \"status\": \"" << ode::to_string(s.status) << "\",\n";
    os << "  \"message\": \"" << esc(s.message) << "\",\n";
    os << "  \"reached_inner_edge\": " << (s.reached_inner_edge ? "true" : "false") << ",\n";
    os << "  \"t_final_yr\": "; num(os, s.t_final_yr); os << ",\n";
    os << "  \"r_final_au\": "; num(os, s.r_final_au); os << ",\n";
    os << "  \"a_final_cm\": "; num(os, s.a_final_cm); os << ",\n";
    os << "  \"t_drift_yr\": "; num(os, s.t_drift_yr); os << ",\n";
    os << "  \"t_settle_first_yr\": "; num(os, s.t_settle_first_yr); os << ",\n";
    os << "  \"t_settle_last_yr\": "; num(os, s.t_settle_last_yr); os << ",\n";
    os << "  \"vr_max_cms\": "; num(os, s.vr_max_cms); os << ",\n";
    os << "  \"r_at_vr_max_au\": "; num(os, s.r_at_vr_max_au); os << ",\n";
    os << "  \"a_max_cm\": "; num(os, s.a_max_cm); os << ",\n";
    os << "  \"max_Re\": "; num(os, s.max_Re); os << ",\n";
    os << "  \"trapped\": " << (s.trapped ? "true" : "false") << ",\n";
    os << "  \"r_trap_au\": "; num(os, s.r_trap_au); os << ",\n";
    os << "  \"stokes_regime_seen\": " << (s.stokes_regime_seen ? "true" : "false") << ",\n";
    os << "  \"final_method\": \"" << ode::to_string(s.final_method) << "\",\n";
    os << "  \"wall_time_s\": " << s.wall_time_s << ",\n";
    os << "  \"warnings\": [";
    for (std::size_t i = 0; i < s.warnings.size(); ++i) os << (i ? ", " : "") << '"' << esc(s.warnings[i]) << '"';
    os << "],\n  \"stats\": {\"n_rhs\": " << st.n_rhs << ", \"n_jac\": " << st.n_jac << ", \"n_lu\": " << st.n_lu
       << ", \"n_steps\": " << st.n_steps << ", \"n_reject_error\": " << st.n_reject_error
       << ", \"n_reject_newton\": " << st.n_reject_newton << ", \"n_newton_iter\": " << st.n_newton_iter
       << ", \"n_linear_iter\": " << st.n_linear_iter << ", \"n_lost_increment\": " << st.n_lost_increment
       << ", \"h_min\": ";
    num(os, st.h_min);
    os << ", \"h_max\": ";
    num(os, st.h_max);
    os << ", \"switches\": [";
    for (std::size_t i = 0; i < st.switches.size(); ++i) {
        const auto& w = st.switches[i];
        os << (i ? ", " : "") << "{\"t_yr\": " << w.t * make_units(1).t0_yr() << ", \"from\": \"" << ode::to_string(w.from)
           << "\", \"to\": \"" << ode::to_string(w.to) << "\", \"reason\": \"" << esc(w.reason) << "\"}";
    }
    os << "]}\n}\n";
}

void write_summary_json(const RunResult& r, const RunConfig& cfg, const std::string& path) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot write '" + path + "'");
    write_summary_json(r, cfg, f);
}

}  // namespace dpd::dust
