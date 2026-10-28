#include "dpd/dust/disk.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dpd::dust {

Disk::Disk(const DiskParams& p) : p_(p) {
    if (!(p.alpha > 0) || !(p.mdot > 0) || !(p.mstar > 0))
        throw std::invalid_argument("disk: alpha, mdot and mstar must be > 0");
    if (!(p.vphi2vk > 0) || !(p.vphi2vk <= 1.0))
        throw std::invalid_argument("disk: vphi2vk must be in (0, 1]");
    const double a = p.alpha / 0.01;
    const double md = p.mdot;
    const double m = p.mstar;
    cT_ = p.T1 * std::pow(a, -0.25) * std::pow(md, 0.5) * std::pow(m, 3.0 / 8.0);
    cRho_ = p.rho1 * std::pow(a, -5.0 / 8.0) * std::pow(md, 0.25) * std::pow(m, 7.0 / 16.0);
    cUr_ = p.ur1 * std::pow(a, 0.75) * std::pow(md, 0.5) * std::pow(m, 1.0 / 8.0);
    cH_ = p.H1 * std::pow(a, -1.0 / 8.0) * std::pow(md, 0.25) * std::pow(m, -5.0 / 16.0);
    if (!p.beta_r_au.empty()) {
        if (p.beta_r_au.size() != p.beta_values.size() || p.beta_r_au.size() < 2)
            throw std::invalid_argument("disk: beta table needs >= 2 (r, beta) pairs of equal length");
        if (!std::is_sorted(p.beta_r_au.begin(), p.beta_r_au.end()) || !(p.beta_r_au.front() > 0))
            throw std::invalid_argument("disk: beta table radii must be positive and increasing");
        for (double b : p.beta_values)
            if (!(b < 1.0)) throw std::invalid_argument("disk: beta values must be < 1");
        log_r_.resize(p.beta_r_au.size());
        std::transform(p.beta_r_au.begin(), p.beta_r_au.end(), log_r_.begin(), [](double r) { return std::log(r); });
    }
}

double Disk::temperature(double r) const { return cT_ * std::pow(r, -9.0 / 8.0); }
double Disk::rho_mid(double r) const { return cRho_ * std::pow(r, -21.0 / 16.0); }
double Disk::scale_height(double r) const { return cH_ * std::pow(r, 15.0 / 16.0); }
double Disk::ur_cgs(double r) const { return cUr_ * std::pow(r, -5.0 / 8.0); }

double Disk::rho(double r, double z) const {
    const double H = scale_height(r);
    return rho_mid(r) * std::exp(-z * z / (2 * H * H));
}

double Disk::beta(double r) const {
    if (log_r_.empty()) return 1.0 - p_.vphi2vk * p_.vphi2vk;
    const double lr = std::log(r);
    if (lr <= log_r_.front()) return p_.beta_values.front();
    if (lr >= log_r_.back()) return p_.beta_values.back();
    const auto it = std::upper_bound(log_r_.begin(), log_r_.end(), lr);
    const auto i = static_cast<std::size_t>(it - log_r_.begin());
    const double w = (lr - log_r_[i - 1]) / (log_r_[i] - log_r_[i - 1]);
    return p_.beta_values[i - 1] + w * (p_.beta_values[i] - p_.beta_values[i - 1]);
}

double Disk::dbeta_dr(double r) const {
    if (log_r_.empty()) return 0.0;
    const double lr = std::log(r);
    if (lr <= log_r_.front() || lr >= log_r_.back()) return 0.0;
    const auto it = std::upper_bound(log_r_.begin(), log_r_.end(), lr);
    const auto i = static_cast<std::size_t>(it - log_r_.begin());
    // beta is linear in log r on each interval
    return (p_.beta_values[i] - p_.beta_values[i - 1]) / (log_r_[i] - log_r_[i - 1]) / r;
}

bool Disk::beta_extrapolated(double r) const {
    return !log_r_.empty() && (r < p_.beta_r_au.front() || r > p_.beta_r_au.back());
}

Disk::Local Disk::local(double r, double z) const {
    const double lr = std::log(r);
    Local L{};
    L.T = cT_ * std::exp(-9.0 / 8.0 * lr);
    L.H = cH_ * std::exp(15.0 / 16.0 * lr);
    L.ur = cUr_ * std::exp(-5.0 / 8.0 * lr);
    const double q = z / L.H;
    L.rho = cRho_ * std::exp(-21.0 / 16.0 * lr - 0.5 * q * q);
    L.beta = beta(r);
    return L;
}

}  // namespace dpd::dust
