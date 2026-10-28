// Dust traps: radii where the radial drift of a grain of given size vanishes.
//
// Equilibrium of Eqs. (11) at the midplane (v_r = 0, dj/dt = 0) requires
//   beta_K(r) v_K = u_r / St,   St = tau_s Omega_K,
// which is exactly the zero of the stationary drift velocity (paper Eq. 6;
// Nakagawa et al. 1986; Takeuchi & Lin 2002)
//   v_r = (u_r / St - beta_K v_K) / (St + 1 / St).
// A zero with v_r > 0 inside and v_r < 0 outside is a stable trap.
#pragma once

#include <vector>

#include "dpd/dust/model.hpp"

namespace dpd::dust {

struct DriftZero {
    double r_au;
    bool stable;  // v_r: + inside, - outside
};

// Stationary midplane drift velocity [cm/s] of a grain of radius a_cm at r_au.
[[nodiscard]] double stationary_vr_cms(const DustModel& m, double r_au, double a_cm);

// All zeros of the stationary drift velocity in [r_min, r_max] (log-spaced scan
// with n points, each bracket refined by bisection to relative 1e-12).
[[nodiscard]] std::vector<DriftZero> drift_zeros(const DustModel& m, double a_cm, double r_min, double r_max,
                                                 int n = 4000);

// Radii where beta_K changes sign from negative (inside) to positive (outside):
// zero head wind, i.e. the trap location in the limit St -> infinity
// (pressure maximum, shifted by magnetic tension when the field is included).
[[nodiscard]] std::vector<double> zero_headwind_radii(const Disk& d, double r_min, double r_max, int n = 4000);

// Smallest grain radius (in [a_lo, a_hi]) that has a stable trap in [r_min, r_max];
// NaN if even a_hi is not trapped. Bisection in log a (trapping is monotone in a
// for St <= 1: larger grains feel the pressure gradient more strongly than the
// inward gas flow).
[[nodiscard]] double min_trapped_size_cm(const DustModel& m, double r_min, double r_max, double a_lo = 1e-5,
                                         double a_hi = 1e3);

}  // namespace dpd::dust
