#include "dpd/dust/disk.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "dpd/dust/constants.hpp"

namespace dpd::dust {

const char* to_string(DiskModel m) noexcept {
    switch (m) {
        case DiskModel::Analytic: return "analytic";
        case DiskModel::TabulatedBeta: return "tabulated_beta";
        case DiskModel::SelfConsistent: return "self_consistent";
        case DiskModel::Tabulated: return "tabulated";
    }
    return "?";
}

DiskModel disk_model_from_string(const std::string& s) {
    if (s == "analytic") return DiskModel::Analytic;
    if (s == "tabulated_beta") return DiskModel::TabulatedBeta;
    if (s == "self_consistent") return DiskModel::SelfConsistent;
    if (s == "tabulated") return DiskModel::Tabulated;
    throw std::invalid_argument("unknown disk.model '" + s +
                                "' (expected analytic|tabulated_beta|self_consistent|tabulated)");
}

const char* to_string(Perturbation p) noexcept {
    switch (p) {
        case Perturbation::None: return "none";
        case Perturbation::Gaussian: return "gaussian";
        case Perturbation::DeadZoneEdge: return "dead_zone_edge";
    }
    return "?";
}

Perturbation perturbation_from_string(const std::string& s) {
    if (s == "none") return Perturbation::None;
    if (s == "gaussian") return Perturbation::Gaussian;
    if (s == "dead_zone_edge") return Perturbation::DeadZoneEdge;
    throw std::invalid_argument("unknown disk.perturbation '" + s + "' (expected none|gaussian|dead_zone_edge)");
}

ProfileTable load_profile_table(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open profile table '" + path + "'");
    ProfileTable t;
    std::vector<std::string> cols;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        for (char& c : line)
            if (c == ',' || c == ';' || c == '\t') c = ' ';
        std::istringstream ss(line);
        if (cols.empty()) {
            std::string name;
            while (ss >> name) cols.push_back(name);
            continue;
        }
        std::map<std::string, double> row;
        for (const auto& c : cols) {
            double v;
            if (!(ss >> v)) throw std::runtime_error("profile table '" + path + "': short row");
            row[c] = v;
        }
        auto need = [&](const char* k) {
            auto it = row.find(k);
            if (it == row.end()) throw std::runtime_error(std::string("profile table: missing column ") + k);
            return it->second;
        };
        t.r_au.push_back(need("r_au"));
        t.T_K.push_back(need("T_K"));
        t.rho_gcc.push_back(need("rho_gcc"));
        t.ur_cms.push_back(need("ur_cms"));
        t.H_au.push_back(need("H_au"));
        if (row.count("beta")) t.beta.push_back(row["beta"]);
        if (row.count("Bz_G")) t.Bz_G.push_back(row["Bz_G"]);
        if (row.count("Br_G")) t.Br_G.push_back(row["Br_G"]);
    }
    if (t.r_au.size() < 3) throw std::runtime_error("profile table '" + path + "' needs >= 3 rows");
    return t;
}

namespace {
bool increasing_positive(const std::vector<double>& r) {
    if (r.empty() || !(r.front() > 0)) return false;
    for (std::size_t i = 1; i < r.size(); ++i)
        if (!(r[i] > r[i - 1])) return false;
    return true;
}
}  // namespace

Disk::Disk(const DiskParams& p) : p_(p) {
    if (!(p.alpha > 0) || !(p.mdot > 0) || !(p.mstar > 0) || !(p.mu > 0))
        throw std::invalid_argument("disk: alpha, mdot, mstar and mu must be > 0");
    if (!(p.vphi2vk > 0) || !(p.vphi2vk <= 1.0)) throw std::invalid_argument("disk: vphi2vk must be in (0, 1]");
    if (p.perturbation != Perturbation::None && !(p.bump_amplitude > -1.0 && p.bump_width_au > 0))
        throw std::invalid_argument("disk: perturbation needs amplitude > -1 and width > 0");
    const double a = p.alpha / 0.01, md = p.mdot, m = p.mstar;
    cT_ = p.T1 * std::pow(a, -0.25) * std::pow(md, 0.5) * std::pow(m, 3.0 / 8.0);
    cRho_ = p.rho1 * std::pow(a, -5.0 / 8.0) * std::pow(md, 0.25) * std::pow(m, 7.0 / 16.0);
    cUr_ = p.ur1 * std::pow(a, 0.75) * std::pow(md, 0.5) * std::pow(m, 1.0 / 8.0);
    cH_ = p.H1 * std::pow(a, -1.0 / 8.0) * std::pow(md, 0.25) * std::pow(m, -5.0 / 16.0);

    if (p.model == DiskModel::TabulatedBeta) {
        if (p.beta_r_au.size() != p.beta_values.size() || p.beta_r_au.size() < 2 || !increasing_positive(p.beta_r_au))
            throw std::invalid_argument("disk: beta table needs >= 2 increasing positive radii with values");
        for (double b : p.beta_values)
            if (!(b < 1.0)) throw std::invalid_argument("disk: beta values must be < 1");
        for (double r : p.beta_r_au) log_r_.push_back(std::log(r));
        tab_beta_ = p.beta_values;
    } else if (p.model == DiskModel::Tabulated) {
        const auto& t = p.table;
        const std::size_t n = t.r_au.size();
        if (n < 3 || !increasing_positive(t.r_au) || t.T_K.size() != n || t.rho_gcc.size() != n ||
            t.ur_cms.size() != n || t.H_au.size() != n)
            throw std::invalid_argument("disk: profile table needs >= 3 increasing radii and equal-length columns");
        for (std::size_t i = 0; i < n; ++i)
            if (!(t.T_K[i] > 0 && t.rho_gcc[i] > 0 && t.H_au[i] > 0))
                throw std::invalid_argument("disk: profile table T, rho, H must be > 0");
        for (double r : t.r_au) log_r_.push_back(std::log(r));
        if (!t.beta.empty()) {
            if (t.beta.size() != n) throw std::invalid_argument("disk: beta column length mismatch");
            tab_beta_ = t.beta;
        } else {
            // Eq. (16): beta = [ -(r/rho) dp/dr + r Bz Br / (4 pi Sigma) ] / v_K^2, Sigma = rho H.
            const bool hasB = t.Bz_G.size() == n && t.Br_G.size() == n;
            tab_beta_.resize(n);
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t i0 = i == 0 ? 0 : i - 1, i1 = i + 1 == n ? n - 1 : i + 1;
                const double lp0 = std::log(t.rho_gcc[i0] * t.T_K[i0]), lp1 = std::log(t.rho_gcc[i1] * t.T_K[i1]);
                const double dlnp = (lp1 - lp0) / (log_r_[i1] - log_r_[i0]);
                const double r_cm = t.r_au[i] * cgs::au;
                const double vk2 = cgs::GM_sun * p.mstar / r_cm;
                const double cT2 = cgs::k_B * t.T_K[i] / (p.mu * cgs::m_p);
                double b = -cT2 * dlnp / vk2;
                if (hasB) b += r_cm * t.Bz_G[i] * t.Br_G[i] / (4 * cgs::pi * t.rho_gcc[i] * t.H_au[i] * cgs::au) / vk2;
                tab_beta_[i] = b;
            }
        }
        for (double b : tab_beta_)
            if (!(b < 1.0)) throw std::invalid_argument("disk: beta must be < 1");
    }
}

double Disk::tab_interp(const std::vector<double>& v, double lr, bool log_values) const {
    if (lr <= log_r_.front()) return v.front();
    if (lr >= log_r_.back()) return v.back();
    const auto it = std::upper_bound(log_r_.begin(), log_r_.end(), lr);
    const auto i = static_cast<std::size_t>(it - log_r_.begin());
    const double w = (lr - log_r_[i - 1]) / (log_r_[i] - log_r_[i - 1]);
    if (log_values) return std::exp(std::log(v[i - 1]) + w * (std::log(v[i]) - std::log(v[i - 1])));
    return v[i - 1] + w * (v[i] - v[i - 1]);
}

double Disk::f_pert(double r) const {
    if (p_.model != DiskModel::SelfConsistent || p_.perturbation == Perturbation::None) return 1.0;
    const double x = (r - p_.bump_r_au) / p_.bump_width_au;
    if (p_.perturbation == Perturbation::Gaussian) return 1.0 + p_.bump_amplitude * std::exp(-0.5 * x * x);
    return 1.0 + 0.5 * p_.bump_amplitude * (1.0 + std::tanh(x));
}

double Disk::dlnf_dr(double r) const {
    if (p_.model != DiskModel::SelfConsistent || p_.perturbation == Perturbation::None) return 0.0;
    const double w = p_.bump_width_au, x = (r - p_.bump_r_au) / w, A = p_.bump_amplitude;
    double df;
    if (p_.perturbation == Perturbation::Gaussian) {
        df = -A * x * std::exp(-0.5 * x * x) / w;
    } else {
        const double s = 1.0 / std::cosh(x);
        df = 0.5 * A * s * s / w;
    }
    return df / f_pert(r);
}

double Disk::d2lnf_dr2(double r) const {
    if (p_.model != DiskModel::SelfConsistent || p_.perturbation == Perturbation::None) return 0.0;
    const double w = p_.bump_width_au, x = (r - p_.bump_r_au) / w, A = p_.bump_amplitude;
    double d2f;
    if (p_.perturbation == Perturbation::Gaussian) {
        d2f = A * (x * x - 1) * std::exp(-0.5 * x * x) / (w * w);
    } else {
        const double s = 1.0 / std::cosh(x);
        d2f = -A * s * s * std::tanh(x) / (w * w);
    }
    const double g = dlnf_dr(r);
    return d2f / f_pert(r) - g * g;
}

bool Disk::has_field() const {
    if (p_.model == DiskModel::Tabulated && !p_.table.Bz_G.empty()) return true;
    return p_.field.B0_G != 0.0;
}

double Disk::Bz(double r) const {
    if (p_.model == DiskModel::Tabulated && p_.table.Bz_G.size() == p_.table.r_au.size())
        return tab_interp(p_.table.Bz_G, std::log(r), false);
    return p_.field.B0_G * std::pow(r, -p_.field.exponent);
}

double Disk::Br(double r) const {
    if (p_.model == DiskModel::Tabulated && p_.table.Br_G.size() == p_.table.r_au.size())
        return tab_interp(p_.table.Br_G, std::log(r), false);
    return p_.field.br_over_bz * Bz(r);
}

double Disk::temperature(double r) const {
    if (p_.model == DiskModel::Tabulated) return tab_interp(p_.table.T_K, std::log(r), true);
    return cT_ * std::pow(r, -9.0 / 8.0);
}

double Disk::rho_mid(double r) const {
    if (p_.model == DiskModel::Tabulated) return tab_interp(p_.table.rho_gcc, std::log(r), true);
    return cRho_ * std::pow(r, -21.0 / 16.0) * f_pert(r);
}

double Disk::scale_height(double r) const {
    if (p_.model == DiskModel::Tabulated) return tab_interp(p_.table.H_au, std::log(r), true);
    return cH_ * std::pow(r, 15.0 / 16.0);
}

double Disk::ur_cgs(double r) const {
    if (p_.model == DiskModel::Tabulated) return tab_interp(p_.table.ur_cms, std::log(r), false);
    return cUr_ * std::pow(r, -5.0 / 8.0) / f_pert(r);
}

double Disk::rho(double r, double z) const {
    const double H = scale_height(r);
    return rho_mid(r) * std::exp(-z * z / (2 * H * H));
}

double Disk::pressure_mid(double r) const { return rho_mid(r) * cgs::k_B * temperature(r) / (p_.mu * cgs::m_p); }

double Disk::beta_self_consistent(double r) const {
    const double r_cm = r * cgs::au;
    const double vk2 = cgs::GM_sun * p_.mstar / r_cm;
    const double cT2 = cgs::k_B * temperature(r) / (p_.mu * cgs::m_p);
    const double dlnp_dlnr = -21.0 / 16.0 - 9.0 / 8.0 + r * dlnf_dr(r);
    double b = -cT2 * dlnp_dlnr / vk2;
    if (p_.magnetic_tension && p_.field.B0_G != 0.0 && p_.field.br_over_bz != 0.0) {
        const double sigma = rho_mid(r) * scale_height(r) * cgs::au;
        b += r_cm * Bz(r) * Br(r) / (4 * cgs::pi * sigma) / vk2;
    }
    return b;
}

double Disk::beta(double r) const {
    switch (p_.model) {
        case DiskModel::Analytic: return 1.0 - p_.vphi2vk * p_.vphi2vk;
        case DiskModel::TabulatedBeta:
        case DiskModel::Tabulated: return tab_interp(tab_beta_, std::log(r), false);
        case DiskModel::SelfConsistent: return beta_self_consistent(r);
    }
    return 0;
}

double Disk::dbeta_dr(double r) const {
    if (p_.model == DiskModel::Analytic) return 0.0;
    if (tabulated()) {
        const double lr = std::log(r);
        if (lr <= log_r_.front() || lr >= log_r_.back()) return 0.0;
        const auto it = std::upper_bound(log_r_.begin(), log_r_.end(), lr);
        const auto i = static_cast<std::size_t>(it - log_r_.begin());
        return (tab_beta_[i] - tab_beta_[i - 1]) / (log_r_[i] - log_r_[i - 1]) / r;
    }
    // SelfConsistent: smooth -> central difference (truncation ~1e-8 relative)
    const double h = 1e-4 * r;
    return (beta(r + h) - beta(r - h)) / (2 * h);
}

bool Disk::beta_extrapolated(double r) const {
    if (!tabulated()) return false;
    const double lr = std::log(r);
    return lr < log_r_.front() || lr > log_r_.back();
}

Disk::Local Disk::local(double r, double z) const {
    Local L{};
    if (p_.model == DiskModel::Tabulated) {
        const double lr = std::log(r);
        L.T = tab_interp(p_.table.T_K, lr, true);
        L.H = tab_interp(p_.table.H_au, lr, true);
        L.ur = tab_interp(p_.table.ur_cms, lr, false);
        const double q = z / L.H;
        L.rho = tab_interp(p_.table.rho_gcc, lr, true) * std::exp(-0.5 * q * q);
        L.beta = tab_interp(tab_beta_, lr, false);
        return L;
    }
    const double lr = std::log(r);
    const double f = f_pert(r);
    L.T = cT_ * std::exp(-9.0 / 8.0 * lr);
    L.H = cH_ * std::exp(15.0 / 16.0 * lr);
    L.ur = cUr_ * std::exp(-5.0 / 8.0 * lr) / f;
    const double q = z / L.H;
    L.rho = cRho_ * f * std::exp(-21.0 / 16.0 * lr - 0.5 * q * q);
    L.beta = beta(r);
    return L;
}

Disk::Derivs Disk::derivs(double r) const {
    Derivs d{};
    if (p_.model != DiskModel::Tabulated) {
        const double g = dlnf_dr(r);
        d.dlnT = -9.0 / 8.0 / r;
        d.dlnrho = -21.0 / 16.0 / r + g;
        d.dlnH = 15.0 / 16.0 / r;
        d.dur = ur_cgs(r) * (-5.0 / 8.0 / r - g);
        d.dbeta = dbeta_dr(r);
        d.dlnB = -p_.field.exponent / r;
        return d;
    }
    // Tables: piecewise derivatives via central differences in log r.
    const double h = 1e-6;
    const double rp = r * (1 + h), rm = r * (1 - h), dr = rp - rm;
    d.dlnT = (std::log(temperature(rp)) - std::log(temperature(rm))) / dr;
    d.dlnrho = (std::log(rho_mid(rp)) - std::log(rho_mid(rm))) / dr;
    d.dlnH = (std::log(scale_height(rp)) - std::log(scale_height(rm))) / dr;
    d.dur = (ur_cgs(rp) - ur_cgs(rm)) / dr;
    d.dbeta = dbeta_dr(r);
    const double bp = Bz(rp), bm = Bz(rm);
    d.dlnB = (bp != 0 && bm != 0 && (bp > 0) == (bm > 0)) ? (std::log(std::abs(bp)) - std::log(std::abs(bm))) / dr : 0.0;
    return d;
}

}  // namespace dpd::dust
