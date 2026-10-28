// Gas disk structure (Kobozeva 2025, Sec. 2.3; Dudorov & Khaibrakhmanov model).
//
// Four structure models:
//   Analytic       paper's analytic solution, Eqs. (12)-(13), constant beta_K (Sec. 2.6)
//   TabulatedBeta  analytic structure, beta_K(r) from a table
//   SelfConsistent analytic structure (optionally perturbed by a pressure bump or a
//                  dead-zone edge) with beta_K computed from Eq. (16): gas pressure
//                  gradient + magnetic tension of the fossil field
//   Tabulated      full radial profiles T, rho, u_r, H (+ beta_K, or B_z/B_r from
//                  which beta_K follows from Eq. (16)), e.g. output of the "Belmondo" code
#pragma once

#include <string>
#include <vector>

namespace dpd::dust {

enum class DiskModel { Analytic, TabulatedBeta, SelfConsistent, Tabulated };
enum class Perturbation { None, Gaussian, DeadZoneEdge };

// Large-scale fossil magnetic field (I: power law, parameters to be replaced by
// the Dudorov-Khaibrakhmanov solution): B_z = B0 (r / 1 au)^-exponent, B_r = q B_z,
// B_phi = 0, independent of z (thin disk).
struct FieldParams {
    double B0_G = 0.0;
    double exponent = 1.5;
    double br_over_bz = 0.0;
};

// Full radial profile table (all vectors the same length, r increasing).
struct ProfileTable {
    std::vector<double> r_au, T_K, rho_gcc, ur_cms, H_au;
    std::vector<double> beta;        // optional
    std::vector<double> Bz_G, Br_G;  // optional (used for beta via Eq. 16 and the Lorentz force)
};

struct DiskParams {
    DiskModel model = DiskModel::Analytic;

    // Analytic solution, normalizations at r = 1 au for alpha = 0.01,
    // Mdot = 1e-8 Msun/yr, M = 1 Msun (paper p. 26).
    double T1 = 240.0;       // K
    double rho1 = 2.5e-10;   // g cm^-3 (midplane)
    double ur1 = -30.0;      // cm s^-1 (accretion velocity)
    double H1 = 0.03;        // au
    double alpha = 0.01;     // turbulence parameter
    double mdot = 1.0;       // accretion rate in units of 1e-8 Msun/yr
    double mstar = 1.0;      // stellar mass in Msun
    double mu = 2.3;         // mean molecular weight (pressure p = rho k T / (mu m_H))

    // Analytic model: u_phi = v_K sqrt(1 - beta_K), beta_K = 1 - vphi2vk^2.
    double vphi2vk = 0.9987;
    // TabulatedBeta: beta_K(r), linear in log r, constant outside the table.
    std::vector<double> beta_r_au;
    std::vector<double> beta_values;
    std::string beta_source;

    // SelfConsistent: midplane density multiplied by f(r) >= 0;
    //   Gaussian:      f = 1 + A exp(-(r - r_b)^2 / (2 w^2))
    //   DeadZoneEdge:  f = 1 + A (1 + tanh((r - r_b) / w)) / 2
    //     A > 0: gas piles up outside r_b = inner edge of a dead zone (pressure
    //     maximum, Kretke & Lin 2007; Dzyurkevich et al. 2010); A < 0: outer edge.
    // Steady accretion: u_r scales as 1/f (Mdot = -2 pi r Sigma u_r constant).
    Perturbation perturbation = Perturbation::None;
    double bump_amplitude = 0.0;
    double bump_r_au = 10.0;
    double bump_width_au = 1.0;
    bool magnetic_tension = true;  // include r B_z B_r / (4 pi Sigma) in beta (SelfConsistent)

    FieldParams field;
    ProfileTable table;            // Tabulated
    std::string table_source;
};

class Disk {
public:
    explicit Disk(const DiskParams& p);

    // r, z in au.
    [[nodiscard]] double temperature(double r) const;          // K
    [[nodiscard]] double rho_mid(double r) const;              // g cm^-3
    [[nodiscard]] double scale_height(double r) const;         // au
    [[nodiscard]] double rho(double r, double z) const;        // g cm^-3, Eq. (19)
    [[nodiscard]] double ur_cgs(double r) const;               // cm s^-1
    [[nodiscard]] double beta(double r) const;                 // dimensionless
    [[nodiscard]] double dbeta_dr(double r) const;             // 1/au
    [[nodiscard]] double pressure_mid(double r) const;         // dyn cm^-2
    [[nodiscard]] double Bz(double r) const;                   // G
    [[nodiscard]] double Br(double r) const;                   // G
    [[nodiscard]] bool beta_extrapolated(double r) const;      // outside a table range
    [[nodiscard]] bool tabulated() const {
        return p_.model == DiskModel::TabulatedBeta || p_.model == DiskModel::Tabulated;
    }
    [[nodiscard]] bool has_field() const;
    [[nodiscard]] const DiskParams& params() const { return p_; }

    struct Local {
        double T;        // K
        double rho;      // g cm^-3 at (r, z)
        double H;        // au
        double ur;       // cm s^-1
        double beta;
    };
    [[nodiscard]] Local local(double r, double z) const;

    // Radial derivatives at the midplane (for analytic Jacobians):
    // d ln T/dr, d ln rho_mid/dr, d ln H/dr, d u_r/dr [cm/s/au], d beta/dr, d ln B/dr.
    struct Derivs {
        double dlnT, dlnrho, dlnH, dur, dbeta, dlnB;
    };
    [[nodiscard]] Derivs derivs(double r) const;

private:
    [[nodiscard]] double f_pert(double r) const;     // density multiplier
    [[nodiscard]] double dlnf_dr(double r) const;
    [[nodiscard]] double d2lnf_dr2(double r) const;
    [[nodiscard]] double beta_self_consistent(double r) const;
    [[nodiscard]] double tab_interp(const std::vector<double>& v, double lr, bool log_values) const;

    DiskParams p_;
    double cT_, cRho_, cUr_, cH_;
    std::vector<double> log_r_;   // tabulated beta or profile radii
    std::vector<double> tab_beta_;
};

[[nodiscard]] const char* to_string(DiskModel m) noexcept;
[[nodiscard]] DiskModel disk_model_from_string(const std::string& s);
[[nodiscard]] const char* to_string(Perturbation p) noexcept;
[[nodiscard]] Perturbation perturbation_from_string(const std::string& s);

// CSV with header naming the columns: r_au, T_K, rho_gcc, ur_cms, H_au and optionally
// beta, Bz_G, Br_G (any order; '#' comments allowed).
[[nodiscard]] ProfileTable load_profile_table(const std::string& path);

}  // namespace dpd::dust
