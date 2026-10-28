// dustppd: command-line driver for the dust-dynamics model.
//
//   dustppd run <config.toml> [-o traj.csv] [-s summary.json] [--set key=value]... [-q]
//   dustppd sweep <config.toml> --param <key> --values v1,v2,... [--threads N] [-s summary.csv] [--set ...]
//   dustppd traps <config.toml> [--sizes a1,a2,...] [--rmin R] [--rmax R] [--set ...]
//   dustppd diffuse <config.toml> [--r R] [--a A] [--alpha ALPHA] [--paths N] [--seed S] [-o z.csv] [--set ...]
//   dustppd show-config <config.toml> [--set ...]
//   dustppd info
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "dpd/dust/diffusion.hpp"
#include "dpd/dust/io.hpp"
#include "dpd/dust/traps.hpp"
#include "dpd/dust/run.hpp"
#include "dpd/ode/kernels.hpp"

using namespace dpd;

namespace {

int usage() {
    std::cerr << "usage:\n"
                 "  dustppd run <config.toml> [-o traj.csv] [-s summary.json] [--set section.key=value]... [-q]\n"
                 "  dustppd sweep <config.toml> --param section.key --values v1,v2,... [--threads N]\n"
                 "                [-s summary.csv] [--set section.key=value]...\n"
                 "  dustppd traps <config.toml> [--sizes a1,a2,...] [--rmin R] [--rmax R] [--set ...]\n"
                 "  dustppd diffuse <config.toml> [--r R_au] [--a A_cm] [--alpha ALPHA] [--paths N] [--seed S]\n"
                 "                  [-o z_final.csv] [--set ...]\n"
                 "  dustppd show-config <config.toml> [--set section.key=value]...\n"
                 "  dustppd info\n";
    return 2;
}

void print_summary(const dust::RunConfig& cfg, const dust::RunResult& r, std::ostream& os) {
    const auto& s = r.summary;
    const auto& st = s.stats;
    os << std::setprecision(6);
    os << "run            : " << cfg.name << "\n";
    os << "status         : " << ode::to_string(s.status) << (s.message.empty() ? "" : " (" + s.message + ")") << "\n";
    os << "a_init         : " << cfg.grain.a_init_cm << " cm, r_init = " << cfg.r_init_au << " au, drag = "
       << dust::to_string(cfg.grain.drag) << ", coagulation = " << (cfg.grain.coagulation ? "on" : "off")
       << ", sedimentation = " << (cfg.grain.sedimentation ? "on" : "off") << "\n";
    os << "t_drift        : " << s.t_drift_yr << " yr" << (s.reached_inner_edge ? "" : " (inner edge not reached)") << "\n";
    os << "t_settle first : " << s.t_settle_first_yr << " yr, settled (last entry): " << s.t_settle_last_yr << " yr\n";
    os << "max |v_r|      : " << s.vr_max_cms / 100 << " m/s at r = " << s.r_at_vr_max_au << " au\n";
    if (s.trapped) os << "trapped        : at r = " << s.r_trap_au << " au\n";
    os << "a_final        : " << s.a_final_cm << " cm (max " << s.a_max_cm << " cm)\n";
    os << "solver         : final method " << ode::to_string(s.final_method) << ", steps " << st.n_steps
       << ", rejected " << st.n_reject_error << "+" << st.n_reject_newton << ", rhs " << st.n_rhs << ", jac "
       << st.n_jac << ", lu " << st.n_lu << ", wall " << s.wall_time_s << " s\n";
    for (const auto& w : st.switches)
        os << "switch         : t = " << w.t * dust::make_units(1).t0_yr() << " yr " << ode::to_string(w.from) << " -> "
           << ode::to_string(w.to) << ": " << w.reason << "\n";
    for (const auto& w : s.warnings) os << "WARNING        : " << w << "\n";
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep))
        if (!item.empty()) out.push_back(item);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    const std::string cmd = argv[1];
    try {
        if (cmd == "info") {
            std::cout << "dustppd 0.1.1\nSIMD kernel backends available:";
            for (auto b : ode::kernels::available_backends()) std::cout << ' ' << ode::kernels::to_string(b);
            std::cout << "\nactive backend: " << ode::kernels::to_string(ode::kernels::active_backend()) << "\n";
            const auto u = dust::make_units(1);
            std::cout << std::setprecision(10) << "code units: r0 = " << u.r0_cm << " cm, v0 = " << u.v0_cms
                      << " cm/s, t0 = " << u.t0_s << " s = " << u.t0_yr() << " yr\n";
            return 0;
        }
        if (argc < 3) return usage();
        const std::string cfg_path = argv[2];
        std::string out_csv, out_summary, param, values, sizes = "1e-4,1e-3,1e-2,0.1,1,10,100";
        double rmin = 0.1, rmax = 300, d_r = NAN, d_a = NAN, d_alpha = NAN;
        std::size_t paths = 20000;
        std::uint64_t seed = 1;
        std::vector<std::string> overrides;
        unsigned threads = 0;
        bool quiet = false;
        for (int i = 3; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value after " + a);
                return argv[++i];
            };
            if (a == "-o" || a == "--output") out_csv = next();
            else if (a == "-s" || a == "--summary") out_summary = next();
            else if (a == "--set") overrides.push_back(next());
            else if (a == "--param") param = next();
            else if (a == "--values") values = next();
            else if (a == "--threads") threads = static_cast<unsigned>(std::stoul(next()));
            else if (a == "-q" || a == "--quiet") quiet = true;
            else if (a == "--sizes") sizes = next();
            else if (a == "--rmin") rmin = std::stod(next());
            else if (a == "--rmax") rmax = std::stod(next());
            else if (a == "--r") d_r = std::stod(next());
            else if (a == "--a") d_a = std::stod(next());
            else if (a == "--alpha") d_alpha = std::stod(next());
            else if (a == "--paths") paths = std::stoul(next());
            else if (a == "--seed") seed = std::stoull(next());
            else throw std::invalid_argument("unknown argument '" + a + "'");
        }
        const dust::RunConfig cfg = dust::load_config(cfg_path, overrides);

        if (cmd == "show-config") {
            std::cout << dust::to_toml(cfg);
            return 0;
        }
        if (cmd == "traps") {
            const dust::Disk disk(cfg.disk);
            std::cout << std::setprecision(6) << "disk model: " << dust::to_string(cfg.disk.model)
                      << ", perturbation: " << dust::to_string(cfg.disk.perturbation) << "\n";
            std::cout << "zero head wind (beta_K: - -> +) in [" << rmin << ", " << rmax << "] au:";
            for (double r : dust::zero_headwind_radii(disk, rmin, rmax)) std::cout << ' ' << r;
            std::cout << "\n";
            for (const auto& sv : split(sizes, ',')) {
                const double a = std::stod(sv);
                dust::GrainParams g = cfg.grain;
                g.a_init_cm = a;
                const dust::DustModel m(cfg.disk, g);
                std::cout << "a = " << a << " cm: drift zeros";
                const auto zs = dust::drift_zeros(m, a, rmin, rmax);
                if (zs.empty()) std::cout << " none (drifts through)";
                for (const auto& z : zs) std::cout << ' ' << z.r_au << (z.stable ? " (trap)" : " (unstable)");
                std::cout << "\n";
            }
            const dust::DustModel m(cfg.disk, cfg.grain);
            std::cout << "minimum trapped radius: " << dust::min_trapped_size_cm(m, rmin, rmax) << " cm\n";
            return 0;
        }
        if (cmd == "diffuse") {
            dust::DiffusionRunConfig dc;
            dc.r_au = std::isfinite(d_r) ? d_r : cfg.r_init_au;
            dc.a_cm = std::isfinite(d_a) ? d_a : cfg.grain.a_init_cm;
            dc.alpha_turb = std::isfinite(d_alpha) ? d_alpha : cfg.disk.alpha;
            dc.n_paths = paths;
            dc.seed = seed;
            const auto r = dust::run_vertical_diffusion(cfg.disk, cfg.grain, dc);
            std::cout << std::setprecision(6) << "r = " << dc.r_au << " au, a = " << dc.a_cm
                      << " cm, alpha_t = " << dc.alpha_turb << ", St_mid = " << r.St_mid << ", H = " << r.H_au << " au\n"
                      << "status: " << ode::to_string(r.status) << " " << r.message << "\n"
                      << "H_d simulated : " << r.Hd_simulated_au << " +- " << r.Hd_stat_error_au << " au ("
                      << r.Hd_simulated_au / r.H_au << " H)\n"
                      << "H_d exact     : " << r.Hd_exact_au << " au (zero-flux stationary solution)\n"
                      << "H_d Dubrulle  : " << r.Hd_dubrulle_au << " au (H sqrt(alpha/(alpha+St)))\n"
                      << "paths " << paths << ", steps " << r.n_steps << ", wall " << r.wall_time_s << " s\n";
            for (const auto& w : r.warnings) std::cout << "WARNING: " << w << "\n";
            if (!out_csv.empty()) {
                std::ofstream f(out_csv);
                f << "z_au\n" << std::setprecision(17);
                for (double z : r.z_final_au) f << z << "\n";
            }
            return ode::is_success(r.status) ? 0 : 1;
        }
        if (cmd == "run") {
            const dust::RunResult r = dust::run(cfg);
            if (!quiet) print_summary(cfg, r, std::cout);
            if (!out_csv.empty()) dust::write_csv(r, out_csv);
            if (!out_summary.empty()) dust::write_summary_json(r, cfg, out_summary);
            return ode::is_success(r.summary.status) ? 0 : 1;
        }
        if (cmd == "sweep") {
            if (param.empty() || values.empty()) return usage();
            std::vector<dust::RunConfig> cfgs;
            const auto vals = split(values, ',');
            for (const auto& v : vals) {
                auto ov = overrides;
                ov.push_back(param + "=" + v);
                cfgs.push_back(dust::load_config(cfg_path, ov));
                cfgs.back().name = cfg.name + "[" + param + "=" + v + "]";
            }
            const auto results = dust::run_ensemble(cfgs, threads, false);
            std::ostream* os = &std::cout;
            std::ofstream f;
            if (!out_summary.empty()) {
                f.open(out_summary);
                os = &f;
            }
            *os << std::setprecision(8);
            *os << param << ",status,t_drift_yr,t_settle_first_yr,t_settle_last_yr,vr_max_cms,r_at_vr_max_au,"
                           "a_final_cm,n_steps,n_rhs,wall_s\n";
            int rc = 0;
            for (std::size_t i = 0; i < results.size(); ++i) {
                const auto& s = results[i].summary;
                if (!ode::is_success(s.status)) rc = 1;
                *os << vals[i] << ',' << ode::to_string(s.status) << ',' << s.t_drift_yr << ',' << s.t_settle_first_yr
                    << ',' << s.t_settle_last_yr << ',' << s.vr_max_cms << ',' << s.r_at_vr_max_au << ','
                    << s.a_final_cm << ',' << s.stats.n_steps << ',' << s.stats.n_rhs << ',' << s.wall_time_s << '\n';
            }
            return rc;
        }
        return usage();
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
