// Dust-particle dynamics in a gas disk: Kobozeva (2025) Eqs. (11) (dimensionless).
//
// State vector (dimensionless, see units below):
//   y[0] = r      radial coordinate            [r0 = 1 au]
//   y[1] = v_r    radial velocity              [v0 = sqrt(G Msun / au)]
//   y[2] = j      specific angular momentum    [j0 = r0 v0]
//   y[3] = z      height above the midplane    [r0]
//   y[4] = v_z    vertical velocity            [v0]
//   y[5] = a      grain radius                 [a0 = a_init]
// Time unit t0 = r0 / v0 (= 1/(2 pi) yr to 0.01%).
#pragma once

#include <cstdint>

#include "dpd/dust/disk.hpp"
#include "dpd/ode/system.hpp"

namespace dpd::dust {

enum class DragLaw : std::uint8_t {
    Epstein,  // Eq. (1)
    Stokes,   // Eq. (2), Re < 1 (linear Stokes) only
    Both,     // Epstein for a/lambda < 3/2, Stokes otherwise (Sec. 1.4); continuous at the switch
};

struct Units {
    double r0_cm;   // 1 au
    double v0_cms;  // sqrt(G Msun / r0)
    double t0_s;    // r0 / v0
    double a0_cm;   // grain-radius unit (= a_init)
    [[nodiscard]] double t0_yr() const;
};
[[nodiscard]] Units make_units(double a0_cm);

struct GrainParams {
    double rho_d = 3.0;           // g cm^-3, material density (Sec. 2.6)
    double a_init_cm = 1.0;       // initial radius
    DragLaw drag = DragLaw::Both;
    bool coagulation = false;     // Eq. (10)
    bool sedimentation = true;    // false: z = v_z = 0 (radial drift only, as in jord++)
    double dust_to_gas = 0.01;    // Y_d (Sec. 2.6)
    double mu = 2.3;              // mean molecular weight (reproduces Eq. (3); see docs)
    // Molecular collision cross-section for lambda = mu m_p / (rho sigma).
    // NOT stated in the paper; default calibrated to the paper's statements
    // (see docs/physics_model.md, "Ambiguities"). Standard H2 value is 2e-15.
    double sigma_mol_cm2 = 1e-16;
};

// Local drag quantities (CGS) at a point.
struct DragState {
    double vth;      // cm/s, mean thermal speed sqrt(8 k T / (pi mu m_p))
    double lambda;   // cm, mean free path
    double tau_ep;   // s, Eq. (1)
    double tau_st;   // s, Eq. (2)
    double tau;      // s, active law
    bool stokes;     // active law is Stokes
    double eta;      // g cm^-1 s^-1, dynamic viscosity v_th lambda rho / 3
};

class DustModel final : public ode::OdeSystem {
public:
    static constexpr ode::Index kSize = 6;
    enum Component : int { R = 0, VR = 1, J = 2, Z = 3, VZ = 4, A = 5 };

    DustModel(const DiskParams& disk, const GrainParams& grain);

    [[nodiscard]] ode::Index size() const override { return kSize; }
    void rhs(double t, std::span<const double> y, std::span<double> dydt) const override;
    [[nodiscard]] bool in_domain(double t, std::span<const double> y) const override;

    [[nodiscard]] DragState drag(double r_au, double z_au, double a_cm) const;
    [[nodiscard]] DragState drag_at(const Disk::Local& L, double a_cm) const;
    // Dimensionless stopping time tau_s / t0.
    [[nodiscard]] double tau_dimless(double r_au, double z_au, double a_cm) const;

    // Keplerian speed and gas azimuthal speed, dimensionless.
    [[nodiscard]] double v_kepler(double r) const;
    [[nodiscard]] double u_phi(double r) const;

    // Initial state for (r_init, z_init) [au], velocities [cm/s]; j = r v_K(r_init).
    [[nodiscard]] std::vector<double> initial_state(double r_init_au, double z_init_au, double vr_cms,
                                                    double vz_cms) const;

    [[nodiscard]] const Disk& disk() const { return disk_; }
    [[nodiscard]] const GrainParams& grain() const { return g_; }
    [[nodiscard]] const Units& units() const { return u_; }

private:
    Disk disk_;
    GrainParams g_;
    Units u_;
    double gm_;           // G M_star in code units (= mstar)
    double vth_coef_;     // sqrt(8 k / (pi mu m_p))
    double lambda_coef_;  // mu m_p / sigma
    double coag_coef_;    // Y_d r0 / (4 rho_d a0)
};

// Derived per-point diagnostics (for output and validation).
struct Diagnostics {
    double t_yr, r_au, z_au, vr_cms, vz_cms, vphi_cms, a_cm;
    double T, rho, H_au, ur_cms, uphi_cms, beta;
    double vth, lambda_cm, tau_s, St, Re, dv_cms;
    bool stokes;
};
[[nodiscard]] Diagnostics diagnose(const DustModel& m, double t, std::span<const double> y);

[[nodiscard]] const char* to_string(DragLaw d) noexcept;
[[nodiscard]] DragLaw drag_from_string(const std::string& s);

}  // namespace dpd::dust
