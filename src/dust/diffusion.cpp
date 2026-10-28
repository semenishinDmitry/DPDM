#include "dpd/dust/diffusion.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace dpd::dust {

VerticalDiffusion::VerticalDiffusion(const DustModel& model, double r_au, double a_cm, double alpha_turb)
    : m_(model), r_(r_au), a_(a_cm), alpha_(alpha_turb) {
    if (!(r_au > 0) || !(a_cm > 0) || !(alpha_turb > 0))
        throw std::invalid_argument("vertical diffusion: r, a and alpha_turb must be > 0");
    om_ = model.v_kepler(r_) / r_;
    H_ = model.disk().scale_height(r_);
}

double VerticalDiffusion::St(double z) const { return m_.drag(r_, z, a_).tau / m_.units().t0_s * om_; }

double VerticalDiffusion::D(double z) const {
    const double s = St(z);
    return alpha_ * om_ * H_ * H_ / (1 + s * s);
}

double VerticalDiffusion::settling_velocity(double z) const { return -om_ * St(z) * z; }

void VerticalDiffusion::drift(double, std::span<const double> y, std::span<double> f) const {
    const double z = y[0];
    const double s = St(z);
    const double d = alpha_ * om_ * H_ * H_ / (1 + s * s);
    // dSt/dz = St * d ln tau/dz; Epstein: tau ~ 1/rho -> +z/H^2; Stokes: independent of rho
    const bool stokes = m_.drag(r_, z, a_).stokes;
    const double dln_st = stokes ? 0.0 : z / (H_ * H_);
    const double dD = d * (-2 * s * s / (1 + s * s)) * dln_st;
    f[0] = -om_ * s * z + dD + d * (-z / (H_ * H_));
}

void VerticalDiffusion::diffusion(double, std::span<const double> y, std::span<double> g) const {
    g[0] = std::sqrt(2 * D(y[0]));
}

double VerticalDiffusion::log_ratio(double z) const {
    // integral_0^z v_s / D dz' by Gauss-Legendre (8 points per sub-interval)
    static const double x[8] = {-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
                                0.1834346424956498,  0.5255324099163290,  0.7966664774136267,  0.9602898564975363};
    static const double w[8] = {0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
                                0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763};
    const int nsub = 16;
    const double h = z / nsub;
    double s = 0;
    for (int k = 0; k < nsub; ++k) {
        const double c = (k + 0.5) * h;
        for (int q = 0; q < 8; ++q) {
            const double zz = c + 0.5 * h * x[q];
            s += w[q] * 0.5 * h * settling_velocity(zz) / D(zz);
        }
    }
    return s;
}

std::vector<double> VerticalDiffusion::stationary_density(const std::vector<double>& z) const {
    std::vector<double> p(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) p[i] = std::exp(-0.5 * z[i] * z[i] / (H_ * H_) + log_ratio(z[i]));
    // normalize with the trapezoidal rule on the given grid
    double norm = 0;
    for (std::size_t i = 1; i < z.size(); ++i) norm += 0.5 * (p[i] + p[i - 1]) * (z[i] - z[i - 1]);
    for (auto& v : p) v /= norm;
    return p;
}

double VerticalDiffusion::stationary_Hd() const {
    // <z^2> = int z^2 p dz / int p dz over [0, 6H] (symmetric), Simpson with 4000 intervals
    const int n = 4000;
    const double zmax = 6 * H_, h = zmax / n;
    double num = 0, den = 0;
    for (int i = 0; i <= n; ++i) {
        const double z = i * h;
        const double p = std::exp(-0.5 * z * z / (H_ * H_) + log_ratio(z));
        const double wgt = (i == 0 || i == n) ? 1 : (i % 2 ? 4 : 2);
        num += wgt * z * z * p;
        den += wgt * p;
    }
    return std::sqrt(num / den);
}

double VerticalDiffusion::dubrulle_Hd() const { return H_ * std::sqrt(alpha_ / (alpha_ + St(0))); }

double VerticalDiffusion::relaxation_time() const { return 1.0 / (om_ * (alpha_ + St(0))); }

DiffusionRunResult run_vertical_diffusion(const DiskParams& disk, const GrainParams& grain,
                                          const DiffusionRunConfig& cfg, const std::atomic<bool>* cancel) {
    const auto w0 = std::chrono::steady_clock::now();
    DiffusionRunResult out;
    GrainParams g = grain;
    g.a_init_cm = cfg.a_cm;
    const DustModel model(disk, g);
    const VerticalDiffusion sys(model, cfg.r_au, cfg.a_cm, cfg.alpha_turb);
    out.H_au = sys.H();
    out.St_mid = sys.St(0);
    out.Hd_exact_au = sys.stationary_Hd();
    out.Hd_dubrulle_au = sys.dubrulle_Hd();
    if (out.St_mid > 0.3)
        out.warnings.push_back("St_mid = " + std::to_string(out.St_mid) +
                               " > 0.3: the terminal-velocity approximation of the settling term is inaccurate");
    const double trel = sys.relaxation_time();
    ode::SdeOptions o;
    o.scheme = cfg.scheme;
    o.dt = cfg.dt_relax * trel;
    o.seed = cfg.seed;
    o.n_paths = cfg.n_paths;
    o.threads = cfg.threads;
    o.cancel = cancel;
    for (std::size_t k = 1; k <= cfg.n_out; ++k) o.t_out.push_back(cfg.t_end_relax * trel * static_cast<double>(k) / static_cast<double>(cfg.n_out));
    const std::vector<double> y0{cfg.z_init_H * sys.H()};
    const ode::SdeResult r = ode::simulate(sys, 0.0, y0, o);
    out.status = r.status;
    out.message = r.message;
    out.n_steps = r.n_steps;
    out.n_failed_paths = r.n_failed_paths;
    const double t0yr = model.units().t0_yr();
    const std::size_t K = o.t_out.size();
    double acc = 0, acc4 = 0;
    std::size_t cnt = 0;
    for (std::size_t k = 0; k < K; ++k) {
        double s2 = 0;
        std::size_t c = 0;
        for (std::size_t p = 0; p < r.n_paths; ++p) {
            const double z = r.at(p, k, 0);
            if (!std::isfinite(z)) continue;
            s2 += z * z;
            ++c;
            if (k >= K - K / 5) {
                acc += z * z;
                acc4 += z * z * z * z;
                ++cnt;
            }
        }
        out.t_yr.push_back(o.t_out[k] * t0yr);
        out.z_rms_au.push_back(c ? std::sqrt(s2 / static_cast<double>(c)) : NAN);
    }
    for (std::size_t p = 0; p < r.n_paths; ++p) out.z_final_au.push_back(r.at(p, K - 1, 0));
    if (cnt) {
        const double m2 = acc / static_cast<double>(cnt), m4 = acc4 / static_cast<double>(cnt);
        out.Hd_simulated_au = std::sqrt(m2);
        // samples from consecutive outputs are correlated; use the number of paths only
        const double var_m2 = (m4 - m2 * m2) / static_cast<double>(r.n_paths);
        out.Hd_stat_error_au = 0.5 * std::sqrt(var_m2) / out.Hd_simulated_au;
    }
    out.wall_time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
    return out;
}

}  // namespace dpd::dust
