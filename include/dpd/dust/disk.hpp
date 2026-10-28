// Gas disk structure (Kobozeva 2025, Sec. 2.3; Dudorov & Khaibrakhmanov model).
#pragma once

#include <string>
#include <vector>

namespace dpd::dust {

struct DiskParams {
    // Analytic solution of the disk equations, Eqs. (12)-(13) and the u_r, H
    // expressions below them (paper p. 26). Normalizations at r = 1 au for
    // alpha = 0.01, Mdot = 1e-8 Msun/yr, M = 1 Msun:
    double T1 = 240.0;       // K
    double rho1 = 2.5e-10;   // g cm^-3 (midplane)
    double ur1 = -30.0;      // cm s^-1 (accretion velocity)
    double H1 = 0.03;        // au
    double alpha = 0.01;     // turbulence parameter
    double mdot = 1.0;       // accretion rate in units of 1e-8 Msun/yr
    double mstar = 1.0;      // stellar mass in Msun

    // Deviation from Keplerian rotation, u_phi = v_K sqrt(1 - beta_K), Eq. (5).
    // Analytic disk: constant beta_K = 1 - vphi2vk^2 (paper: u_phi/v_K = 0.9987,
    // beta_K ~= 0.0026, Sec. 2.6). Tabulated: beta_K(r) interpolated linearly in
    // log r from (beta_r_au, beta_values); constant extrapolation outside the table.
    double vphi2vk = 0.9987;
    std::vector<double> beta_r_au;
    std::vector<double> beta_values;
    std::string beta_source;  // provenance label for tabulated data
};

class Disk {
public:
    explicit Disk(const DiskParams& p);

    // r, z in au.
    [[nodiscard]] double temperature(double r) const;          // K, Eq. (12)
    [[nodiscard]] double rho_mid(double r) const;              // g cm^-3, Eq. (13)
    [[nodiscard]] double scale_height(double r) const;         // au
    [[nodiscard]] double rho(double r, double z) const;        // g cm^-3, Eq. (19)
    [[nodiscard]] double ur_cgs(double r) const;               // cm s^-1
    [[nodiscard]] double beta(double r) const;                 // dimensionless
    [[nodiscard]] bool beta_extrapolated(double r) const;      // outside table range
    [[nodiscard]] bool tabulated() const { return !p_.beta_r_au.empty(); }
    [[nodiscard]] const DiskParams& params() const { return p_; }

    // All structure quantities at once (single log(r) evaluation; RHS hot path).
    struct Local {
        double T;        // K
        double rho;      // g cm^-3
        double H;        // au
        double ur;       // cm s^-1
        double beta;
    };
    [[nodiscard]] Local local(double r, double z) const;

private:
    DiskParams p_;
    double cT_, cRho_, cUr_, cH_;  // parameter-dependent prefactors
    std::vector<double> log_r_;    // log of beta table radii
};

}  // namespace dpd::dust
