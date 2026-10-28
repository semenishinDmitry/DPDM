#include "dpd/dust/traps.hpp"

#include <cmath>
#include <limits>

#include "dpd/dust/constants.hpp"

namespace dpd::dust {

double stationary_vr_cms(const DustModel& m, double r_au, double a_cm) {
    const Units& u = m.units();
    const Disk& d = m.disk();
    const double vk = m.v_kepler(r_au);                // code units
    const double om = vk / r_au;
    const double St = m.drag(r_au, 0.0, a_cm).tau / u.t0_s * om;
    const double ur = d.ur_cgs(r_au) / u.v0_cms;
    const double vr = (ur / St - d.beta(r_au) * vk) / (St + 1.0 / St);
    return vr * u.v0_cms;
}

namespace {
template <class F>
double bisect(F&& f, double a, double b, double fa) {
    for (int it = 0; it < 200; ++it) {
        const double c = std::sqrt(a * b);  // geometric midpoint (r > 0)
        const double fc = f(c);
        if ((fc > 0) == (fa > 0)) {
            a = c;
            fa = fc;
        } else {
            b = c;
        }
        if (b - a <= 1e-12 * b) break;
    }
    return std::sqrt(a * b);
}
}  // namespace

std::vector<DriftZero> drift_zeros(const DustModel& m, double a_cm, double r_min, double r_max, int n) {
    std::vector<DriftZero> out;
    auto f = [&](double r) { return stationary_vr_cms(m, r, a_cm); };
    const double q = std::log(r_max / r_min) / n;
    double r0 = r_min, f0 = f(r0);
    for (int i = 1; i <= n; ++i) {
        const double r1 = r_min * std::exp(q * i);
        const double f1 = f(r1);
        if (f0 != 0 && f1 != 0 && (f0 > 0) != (f1 > 0)) out.push_back({bisect(f, r0, r1, f0), f0 > 0});
        r0 = r1;
        f0 = f1;
    }
    return out;
}

std::vector<double> zero_headwind_radii(const Disk& d, double r_min, double r_max, int n) {
    std::vector<double> out;
    auto f = [&](double r) { return d.beta(r); };
    const double q = std::log(r_max / r_min) / n;
    double r0 = r_min, f0 = f(r0);
    for (int i = 1; i <= n; ++i) {
        const double r1 = r_min * std::exp(q * i);
        const double f1 = f(r1);
        if (f0 < 0 && f1 >= 0) out.push_back(bisect(f, r0, r1, f0));
        r0 = r1;
        f0 = f1;
    }
    return out;
}

double min_trapped_size_cm(const DustModel& m, double r_min, double r_max, double a_lo, double a_hi) {
    auto trapped = [&](double a) {
        for (const auto& z : drift_zeros(m, a, r_min, r_max, 1500))
            if (z.stable) return true;
        return false;
    };
    if (!trapped(a_hi)) return std::numeric_limits<double>::quiet_NaN();
    if (trapped(a_lo)) return a_lo;
    for (int it = 0; it < 60; ++it) {
        const double c = std::sqrt(a_lo * a_hi);
        (trapped(c) ? a_hi : a_lo) = c;
        if (a_hi / a_lo < 1.0 + 1e-4) break;
    }
    return a_hi;
}

}  // namespace dpd::dust
