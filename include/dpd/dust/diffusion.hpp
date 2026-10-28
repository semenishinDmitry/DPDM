// Vertical turbulent mixing of dust at a fixed radius (I: extension of the paper).
//
// Lagrangian (Ito) form of the advection-diffusion equation for the dust density
// with gas density rho_g(z) as the reference (Ciesla 2010; Charnoz et al. 2011):
//   dz = [ v_s + dD/dz + D d ln rho_g / dz ] dt + sqrt(2 D) dW,
//   v_s = -Omega_K^2 tau_s z = -Omega_K St(z) z   (terminal settling velocity, St < ~0.3)
//   D   = D_g / (1 + St^2),  D_g = alpha_t Omega_K H^2   (Youdin & Lithwick 2007)
// St(z) uses the drag law of the model (grows as exp(z^2 / 2H^2) in the Epstein regime).
// Zero-flux stationary state: d ln(rho_d / rho_g) / dz = v_s / D, computed here by
// quadrature as an exact reference; for constant St it is Gaussian with
//   H_d = H sqrt(alpha_t / (alpha_t + St))   (Dubrulle et al. 1995).
#pragma once

#include "dpd/dust/model.hpp"
#include "dpd/ode/sde.hpp"

namespace dpd::dust {

class VerticalDiffusion final : public ode::SdeSystem {
public:
    // Times in code units t0, z in au.
    VerticalDiffusion(const DustModel& model, double r_au, double a_cm, double alpha_turb);
    [[nodiscard]] ode::Index size() const override { return 1; }
    void drift(double t, std::span<const double> y, std::span<double> f) const override;
    void diffusion(double t, std::span<const double> y, std::span<double> g) const override;

    [[nodiscard]] double omega() const { return om_; }          // code units
    [[nodiscard]] double H() const { return H_; }               // au
    [[nodiscard]] double St(double z) const;
    [[nodiscard]] double D(double z) const;                     // au^2 / t0
    [[nodiscard]] double settling_velocity(double z) const;     // au / t0
    // Exact stationary dust scale height sqrt(<z^2>) from the zero-flux solution.
    [[nodiscard]] double stationary_Hd() const;
    // Stationary density on a grid (normalized to unit integral).
    [[nodiscard]] std::vector<double> stationary_density(const std::vector<double>& z) const;
    // Dubrulle et al. (1995) estimate with the midplane Stokes number.
    [[nodiscard]] double dubrulle_Hd() const;
    // Relaxation time 1 / (Omega (alpha_t + St_mid)) in code units.
    [[nodiscard]] double relaxation_time() const;

private:
    double log_ratio(double z) const;  // ln(rho_d / rho_g) relative to z = 0
    const DustModel& m_;
    double r_, a_, alpha_, om_, H_;
};

struct DiffusionRunConfig {
    double r_au = 100.0;
    double a_cm = 0.1;
    double alpha_turb = 1e-4;
    double z_init_H = 0.0;
    std::size_t n_paths = 20000;
    double t_end_relax = 10.0;     // in units of the relaxation time
    double dt_relax = 0.02;        // step in units of the relaxation time
    std::size_t n_out = 50;
    std::uint64_t seed = 1;
    unsigned threads = 0;
    ode::SdeScheme scheme = ode::SdeScheme::LinearImplicitEuler;
};

struct DiffusionRunResult {
    ode::Status status = ode::Status::InvalidInput;
    std::string message;
    std::vector<double> t_yr;        // output times
    std::vector<double> z_rms_au;    // sqrt(<z^2>) over paths at each output time
    std::vector<double> z_final_au;  // final positions of all paths
    double H_au = 0, St_mid = 0;
    double Hd_simulated_au = 0;      // sqrt(<z^2>) over the last 20 % of outputs
    double Hd_exact_au = 0;          // zero-flux quadrature
    double Hd_dubrulle_au = 0;
    double Hd_stat_error_au = 0;     // 1-sigma sampling error of Hd_simulated
    std::uint64_t n_steps = 0;
    std::size_t n_failed_paths = 0;
    double wall_time_s = 0;
    std::vector<std::string> warnings;
};

[[nodiscard]] DiffusionRunResult run_vertical_diffusion(const DiskParams& disk, const GrainParams& grain,
                                                        const DiffusionRunConfig& cfg,
                                                        const std::atomic<bool>* cancel = nullptr);

}  // namespace dpd::dust
