#include "dpd/dust/model.hpp"

#include <cmath>
#include <stdexcept>

#include "dpd/dust/constants.hpp"

namespace dpd::dust {

double Units::t0_yr() const { return t0_s / cgs::year; }

Units make_units(double a0_cm) {
    Units u{};
    u.r0_cm = cgs::au;
    u.v0_cms = std::sqrt(cgs::GM_sun / cgs::au);
    u.t0_s = u.r0_cm / u.v0_cms;
    u.a0_cm = a0_cm;
    return u;
}

const char* to_string(DragLaw d) noexcept {
    switch (d) {
        case DragLaw::Epstein: return "epstein";
        case DragLaw::Stokes: return "stokes";
        case DragLaw::Both: return "both";
    }
    return "?";
}

DragLaw drag_from_string(const std::string& s) {
    if (s == "epstein") return DragLaw::Epstein;
    if (s == "stokes") return DragLaw::Stokes;
    if (s == "both") return DragLaw::Both;
    throw std::invalid_argument("unknown drag law '" + s + "' (expected epstein|stokes|both)");
}

DustModel::DustModel(const DiskParams& disk, const GrainParams& grain)
    : disk_(disk), g_(grain), u_(make_units(grain.a_init_cm)) {
    if (!(g_.rho_d > 0) || !(g_.a_init_cm > 0) || !(g_.mu > 0) || !(g_.sigma_mol_cm2 > 0) || !(g_.dust_to_gas >= 0))
        throw std::invalid_argument("grain: rho_d, a_init, mu, sigma_mol must be > 0 and dust_to_gas >= 0");
    gm_ = disk.mstar;
    vth_coef_ = std::sqrt(8 * cgs::k_B / (cgs::pi * g_.mu * cgs::m_p));
    lambda_coef_ = g_.mu * cgs::m_p / g_.sigma_mol_cm2;
    coag_coef_ = g_.dust_to_gas * u_.r0_cm / (4 * g_.rho_d * u_.a0_cm);
    if (g_.charge == ChargeModel::OML) {
        if (!(g_.ion_mass_amu > 0)) throw std::invalid_argument("grain: ion_mass_amu must be > 0");
        psi_ = oml_potential(g_.ion_mass_amu);
    }
    lorentz_ = g_.charge != ChargeModel::None && disk_.has_field();
}

double oml_potential(double ion_mass_amu) {
    // Solve exp(psi) * sqrt(m_i/m_e) = 1 - psi for psi < 0 (Newton, monotone).
    const double s = std::sqrt(ion_mass_amu * cgs::amu / cgs::m_e);
    double psi = -std::log(s);
    for (int it = 0; it < 50; ++it) {
        const double g = std::exp(psi) * s - (1 - psi);
        const double dg = std::exp(psi) * s + 1;
        const double step = g / dg;
        psi -= step;
        if (std::abs(step) < 1e-15) break;
    }
    return psi;
}

double DustModel::charge_number(double T, double a_cm) const {
    return psi_ * a_cm * cgs::k_B * T / (cgs::e_esu * cgs::e_esu);
}

double DustModel::lorentz_rate(double T, double a_cm, double B) const {
    // (Z e / m c) B with m = 4/3 pi a^3 rho_d
    const double m = 4.0 / 3.0 * cgs::pi * a_cm * a_cm * a_cm * g_.rho_d;
    return charge_number(T, a_cm) * cgs::e_esu * B / (m * cgs::c_light);
}

double DustModel::v_kepler(double r) const { return std::sqrt(gm_ / r); }
double DustModel::u_phi(double r) const { return v_kepler(r) * std::sqrt(1 - disk_.beta(r)); }

DragState DustModel::drag_at(const Disk::Local& L, double a_cm) const {
    DragState d{};
    d.vth = vth_coef_ * std::sqrt(L.T);
    d.lambda = lambda_coef_ / L.rho;
    d.eta = d.vth * d.lambda * L.rho / 3;
    d.tau_ep = g_.rho_d * a_cm / (L.rho * d.vth);                   // Eq. (1)
    d.tau_st = 2.0 / 9.0 * g_.rho_d * a_cm * a_cm / d.eta;           // Eq. (2)
    switch (g_.drag) {
        case DragLaw::Epstein: d.stokes = false; break;
        case DragLaw::Stokes: d.stokes = true; break;
        case DragLaw::Both: d.stokes = a_cm >= 1.5 * d.lambda; break;  // a/lambda = 3/2 (Sec. 1.4)
    }
    d.tau = d.stokes ? d.tau_st : d.tau_ep;
    return d;
}

DragState DustModel::drag(double r_au, double z_au, double a_cm) const {
    return drag_at(disk_.local(r_au, z_au), a_cm);
}

double DustModel::tau_dimless(double r_au, double z_au, double a_cm) const {
    return drag(r_au, z_au, a_cm).tau / u_.t0_s;
}

void DustModel::rhs(double, std::span<const double> y, std::span<double> f) const {
    const double r = y[R], vr = y[VR], j = y[J];
    const double z = g_.sedimentation ? y[Z] : 0.0;
    const double vz = g_.sedimentation ? y[VZ] : 0.0;
    const double a_cm = y[A] * u_.a0_cm;

    const Disk::Local L = disk_.local(r, z);
    const DragState d = drag_at(L, a_cm);
    const double inv_tau = u_.t0_s / d.tau;  // dimensionless 1/tau_s
    const double ur = L.ur / u_.v0_cms;
    const double uphi = std::sqrt(gm_ / r) * std::sqrt(1 - L.beta);
    const double vphi = j / r;
    const double r2 = r * r, r3 = r2 * r;

    f[R] = vr;
    f[VR] = j * j / r3 - gm_ / r2 - (vr - ur) * inv_tau;
    f[J] = -r * inv_tau * (vphi - uphi);
    if (g_.sedimentation) {
        f[Z] = vz;
        f[VZ] = -gm_ * z / r3 - vz * inv_tau;
    } else {
        f[Z] = 0;
        f[VZ] = 0;
    }
    if (lorentz_) {
        // (Z e / m c) (v - u) x B, B = (B_r, 0, B_z), gas velocity u = (u_r, u_phi, 0)
        const double wz_ = lorentz_rate(L.T, a_cm, disk_.Bz(r)) * u_.t0_s;
        const double wr_ = lorentz_rate(L.T, a_cm, disk_.Br(r)) * u_.t0_s;
        const double dr_ = vr - ur, dp_ = vphi - uphi;
        f[VR] += wz_ * dp_;
        f[J] += r * (wr_ * vz - wz_ * dr_);
        if (g_.sedimentation) f[VZ] += -wr_ * dp_;
    }
    if (g_.coagulation) {
        const double dvr = vr - ur, dvp = vphi - uphi;
        f[A] = coag_coef_ * L.rho * std::sqrt(dvr * dvr + dvp * dvp + vz * vz);  // Eq. (10)
    } else {
        f[A] = 0;
    }
}

void DustModel::jacobian(double, std::span<const double> y, ode::DenseMatrix& Jm) const {
    const bool sed = g_.sedimentation;
    const double r = y[R], vr = y[VR], j = y[J];
    const double z = sed ? y[Z] : 0.0;
    const double vz = sed ? y[VZ] : 0.0;
    const double ah = y[A];
    const double a_cm = ah * u_.a0_cm;
    const Disk::Local L = disk_.local(r, z);
    const DragState d = drag_at(L, a_cm);
    const double inv = u_.t0_s / d.tau;
    const Disk::Derivs dd = disk_.derivs(r);
    const double ur = L.ur / u_.v0_cms;
    const double dur_dr = dd.dur / u_.v0_cms;
    const double sb = std::sqrt(1 - L.beta);
    const double vk = std::sqrt(gm_ / r);
    const double uphi = vk * sb;
    const double duphi_dr = uphi * (-0.5 / r) - vk * dd.dbeta / (2 * sb);
    // log-derivatives of rho, v_th and tau
    const double q2 = (z / L.H) * (z / L.H);
    const double dlnrho_dr = dd.dlnrho + q2 * dd.dlnH;
    const double dlnrho_dz = -z / (L.H * L.H);
    const double dlnvth_dr = 0.5 * dd.dlnT;
    double dlntau_dr, dlntau_dz, dlntau_da;
    if (d.stokes) {  // tau_St ~ a^2 / v_th (independent of rho)
        dlntau_dr = -dlnvth_dr;
        dlntau_dz = 0;
        dlntau_da = 2 / ah;
    } else {         // tau_Ep ~ a / (rho v_th)
        dlntau_dr = -dlnrho_dr - dlnvth_dr;
        dlntau_dz = -dlnrho_dz;
        dlntau_da = 1 / ah;
    }
    const double dinv_dr = -inv * dlntau_dr, dinv_dz = -inv * dlntau_dz, dinv_da = -inv * dlntau_da;
    const double r2 = r * r, r3 = r2 * r, r4 = r3 * r;
    const double dvr = vr - ur;
    const double w = j - r * uphi;  // r (v_phi - u_phi)
    Jm.fill(0);
    Jm(R, VR) = 1;
    Jm(VR, R) = -3 * j * j / r4 + 2 * gm_ / r3 + dur_dr * inv - dvr * dinv_dr;
    Jm(VR, VR) = -inv;
    Jm(VR, J) = 2 * j / r3;
    Jm(VR, A) = -dvr * dinv_da;
    Jm(J, R) = -dinv_dr * w + inv * (uphi + r * duphi_dr);
    Jm(J, J) = -inv;
    Jm(J, A) = -dinv_da * w;
    if (sed) {
        Jm(VR, Z) = -dvr * dinv_dz;
        Jm(J, Z) = -dinv_dz * w;
        Jm(Z, VZ) = 1;
        Jm(VZ, R) = 3 * gm_ * z / r4 - vz * dinv_dr;
        Jm(VZ, Z) = -gm_ / r3 - vz * dinv_dz;
        Jm(VZ, VZ) = -inv;
        Jm(VZ, A) = -vz * dinv_da;
    }
    if (lorentz_) {
        // omega = (Z e / m c) B t0 ~ T B / a^2 (Z ~ a T, m ~ a^3)
        const double wz_ = lorentz_rate(L.T, a_cm, disk_.Bz(r)) * u_.t0_s;
        const double wr_ = lorentz_rate(L.T, a_cm, disk_.Br(r)) * u_.t0_s;
        const double dlnw = dd.dlnT + dd.dlnB;
        const double dwz_dr = wz_ * dlnw, dwr_dr = wr_ * dlnw;
        const double dwz_da = -2 * wz_ / ah, dwr_da = -2 * wr_ / ah;
        const double dp_ = j / r - uphi;
        const double ddp_dr = -j / r2 - duphi_dr;
        // f_VR += wz dp
        Jm(VR, R) += dwz_dr * dp_ + wz_ * ddp_dr;
        Jm(VR, J) += wz_ / r;
        Jm(VR, A) += dwz_da * dp_;
        // f_J += r (wr vz - wz dvr)
        Jm(J, R) += (wr_ * vz - wz_ * dvr) + r * (dwr_dr * vz - dwz_dr * dvr + wz_ * dur_dr);
        Jm(J, VR) += -r * wz_;
        Jm(J, A) += r * (dwr_da * vz - dwz_da * dvr);
        if (sed) {
            Jm(J, VZ) += r * wr_;
            // f_VZ += -wr dp
            Jm(VZ, R) += -dwr_dr * dp_ - wr_ * ddp_dr;
            Jm(VZ, J) += -wr_ / r;
            Jm(VZ, A) += -dwr_da * dp_;
        }
    }
    if (g_.coagulation) {
        const double dvp = j / r - uphi;
        const double D = std::sqrt(dvr * dvr + dvp * dvp + vz * vz);
        const double crho = coag_coef_ * L.rho;
        Jm(A, R) = crho * dlnrho_dr * D;
        if (sed) Jm(A, Z) = crho * dlnrho_dz * D;
        if (D > 0) {
            Jm(A, R) += crho / D * (dvr * (-dur_dr) + dvp * (-j / r2 - duphi_dr));
            Jm(A, VR) = crho * dvr / D;
            Jm(A, J) = crho * dvp / (D * r);
            if (sed) Jm(A, VZ) = crho * vz / D;
        }
    }
}

bool DustModel::in_domain(double, std::span<const double> y) const {
    return y[R] > 0 && y[A] > 0 && std::isfinite(y[R]) && std::isfinite(y[A]);
}

std::vector<double> DustModel::initial_state(double r_init_au, double z_init_au, double vr_cms, double vz_cms) const {
    std::vector<double> y(kSize);
    y[R] = r_init_au;
    y[VR] = vr_cms / u_.v0_cms;
    y[J] = r_init_au * v_kepler(r_init_au);
    y[Z] = g_.sedimentation ? z_init_au : 0.0;
    y[VZ] = g_.sedimentation ? vz_cms / u_.v0_cms : 0.0;
    y[A] = g_.a_init_cm / u_.a0_cm;
    return y;
}

Diagnostics diagnose(const DustModel& m, double t, std::span<const double> y) {
    const Units& u = m.units();
    Diagnostics D{};
    const double z = m.grain().sedimentation ? y[DustModel::Z] : 0.0;
    D.t_yr = t * u.t0_yr();
    D.r_au = y[DustModel::R];
    D.z_au = z;
    D.vr_cms = y[DustModel::VR] * u.v0_cms;
    D.vz_cms = (m.grain().sedimentation ? y[DustModel::VZ] : 0.0) * u.v0_cms;
    D.vphi_cms = y[DustModel::J] / D.r_au * u.v0_cms;
    D.a_cm = y[DustModel::A] * u.a0_cm;
    const Disk::Local L = m.disk().local(D.r_au, z);
    D.T = L.T;
    D.rho = L.rho;
    D.H_au = L.H;
    D.ur_cms = L.ur;
    D.beta = L.beta;
    D.uphi_cms = m.u_phi(D.r_au) * u.v0_cms;
    const DragState d = m.drag_at(L, D.a_cm);
    D.vth = d.vth;
    D.lambda_cm = d.lambda;
    D.tau_s = d.tau;
    D.stokes = d.stokes;
    D.St = d.tau / u.t0_s * m.v_kepler(D.r_au) / D.r_au;  // tau_s * Omega_K
    const double dvr = D.vr_cms - D.ur_cms, dvp = D.vphi_cms - D.uphi_cms;
    D.dv_cms = std::sqrt(dvr * dvr + dvp * dvp + D.vz_cms * D.vz_cms);
    D.Re = 2 * D.a_cm * D.rho * D.dv_cms / d.eta;
    D.Z_charge = m.charge_number(D.T, D.a_cm);
    const double Bmag = std::hypot(m.disk().Bz(D.r_au), m.disk().Br(D.r_au));
    D.hall = std::abs(m.lorentz_rate(D.T, D.a_cm, Bmag)) * d.tau;
    return D;
}

}  // namespace dpd::dust
