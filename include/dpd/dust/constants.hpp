// Physical constants (CGS). Every constant used by the dust model is defined
// here and only here.
#pragma once

namespace dpd::dust::cgs {

inline constexpr double au = 1.495978707e13;        // cm (IAU 2012, exact)
inline constexpr double GM_sun = 1.3271244e26;      // cm^3 s^-2 (IAU 2015 nominal)
inline constexpr double k_B = 1.380649e-16;         // erg K^-1 (SI 2019, exact)
inline constexpr double m_p = 1.67262192369e-24;    // g (CODATA 2018); paper: "m_H - proton mass"
inline constexpr double year = 3.15576e7;           // s (Julian year)
inline constexpr double pi = 3.141592653589793238462643383279502884;

}  // namespace dpd::dust::cgs
