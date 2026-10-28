// Per-trajectory integrator shared verbatim by the CPU (C++, float/double),
// Metal (MSL, float only: Apple GPUs have no FP64) and CUDA (float/double).
//
// Model: radial drift of a grain in the analytic disk (paper Eqs. 11, 12-13,
// drag Eqs. 1-2 with the Epstein/Stokes switch), sedimentation and coagulation
// off -- identical equations to dpd::dust::DustModel with sedimentation=false.
// Integrator: adaptive DOPRI5 (same tableau as the library), simple I-controller,
// terminal event r = r_in located by linear interpolation inside the last step,
// time accumulated with Kahan summation (needed in float32: ulp(3e4) ~ 2e-3).
// EXPERIMENTAL: not part of the production library.
#pragma once

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
using namespace metal;
#define DPD_FN inline
#define DPD_THREAD thread
#elif defined(__CUDACC__)
#define DPD_FN __host__ __device__ inline
#define DPD_THREAD
#else
#include <algorithm>
#include <cmath>
using std::exp; using std::fabs; using std::isfinite; using std::log; using std::max; using std::min;
using std::pow; using std::sqrt;
#define DPD_FN inline
#define DPD_THREAD
#endif

template <class T>
struct EnsParams {
    T cT, cRho, cUr;      // T(1au)[K], rho(1au)[g/cc], u_r(1au)/v0 (alpha, mdot, M folded in)
    T uphi_fac;           // sqrt(1 - beta_K)
    T vth_coef;           // sqrt(8 k / (pi mu m_H))  [cm/s/K^0.5]
    T lam_coef;           // mu m_H / sigma            [g/cm^2]
    T t0;                 // code time unit [s]
    T rho_d;              // g/cc
    T r_in;               // au
    T rtol, atol_r, atol_v, atol_j;
    T h_max;
    int max_steps;
    int drag_both;        // 1: Epstein/Stokes switch at a/lambda = 3/2, 0: Epstein only
};

template <class T>
struct EnsResult {
    T t_drift;   // code units (multiply by t0/yr)
    int steps;
    int status;  // 0 reached r_in, 1 max steps, 2 step underflow/non-finite
};

template <class T>
DPD_FN void ens_rhs(DPD_THREAD const EnsParams<T>& p, T a_cm, T r, T vr, T j, DPD_THREAD T& dr, DPD_THREAD T& dvr,
                    DPD_THREAD T& dj) {
    const T lr = log(r);
    const T temp = p.cT * exp(T(-1.125) * lr);
    const T rho = p.cRho * exp(T(-1.3125) * lr);
    const T ur = p.cUr * exp(T(-0.625) * lr);
    const T vth = p.vth_coef * sqrt(temp);
    T tau = p.rho_d * a_cm / (rho * vth) / p.t0;  // Epstein, Eq. (1)
    if (p.drag_both) {
        const T lam = p.lam_coef / rho;
        if (a_cm >= T(1.5) * lam) tau = tau * (T(2) * a_cm / (T(3) * lam));  // Stokes, Eq. (2)
    }
    const T inv_tau = T(1) / tau;
    const T sr = sqrt(r);
    const T uphi = p.uphi_fac / sr;
    const T r2 = r * r;
    dr = vr;
    dvr = j * j / (r2 * r) - T(1) / r2 - (vr - ur) * inv_tau;
    dj = -r * inv_tau * (j / r - uphi);
}

template <class T>
DPD_FN EnsResult<T> ens_integrate(DPD_THREAD const EnsParams<T>& p, T a_cm, T r0) {
    // Dormand-Prince 5(4)
    const T c2 = T(0.2), c3 = T(0.3), c4 = T(0.8), c5 = T(8.0 / 9.0);
    const T a21 = T(0.2);
    const T a31 = T(3.0 / 40), a32 = T(9.0 / 40);
    const T a41 = T(44.0 / 45), a42 = T(-56.0 / 15), a43 = T(32.0 / 9);
    const T a51 = T(19372.0 / 6561), a52 = T(-25360.0 / 2187), a53 = T(64448.0 / 6561), a54 = T(-212.0 / 729);
    const T a61 = T(9017.0 / 3168), a62 = T(-355.0 / 33), a63 = T(46732.0 / 5247), a64 = T(49.0 / 176),
            a65 = T(-5103.0 / 18656);
    const T b1 = T(35.0 / 384), b3 = T(500.0 / 1113), b4 = T(125.0 / 192), b5 = T(-2187.0 / 6784), b6 = T(11.0 / 84);
    const T e1 = T(71.0 / 57600), e3 = T(-71.0 / 16695), e4 = T(71.0 / 1920), e5 = T(-17253.0 / 339200),
            e6 = T(22.0 / 525), e7 = T(-1.0 / 40);
    (void)c2; (void)c3; (void)c4; (void)c5;  // autonomous system

    EnsResult<T> res;
    res.t_drift = T(0);
    res.steps = 0;
    res.status = 1;
    T y0 = r0, y1 = T(0), y2 = sqrt(r0);  // r, v_r, j (Keplerian start, M = 1)
    T t = T(0), tc = T(0);                // Kahan-compensated time
    T h = T(1e-3);
    T k1[3], k2[3], k3[3], k4[3], k5[3], k6[3], k7[3];
    ens_rhs(p, a_cm, y0, y1, y2, k1[0], k1[1], k1[2]);
    int fails = 0;
    for (int it = 0; it < p.max_steps; ++it) {
        h = min(h, p.h_max);
        ens_rhs(p, a_cm, y0 + h * a21 * k1[0], y1 + h * a21 * k1[1], y2 + h * a21 * k1[2], k2[0], k2[1], k2[2]);
        ens_rhs(p, a_cm, y0 + h * (a31 * k1[0] + a32 * k2[0]), y1 + h * (a31 * k1[1] + a32 * k2[1]),
                y2 + h * (a31 * k1[2] + a32 * k2[2]), k3[0], k3[1], k3[2]);
        ens_rhs(p, a_cm, y0 + h * (a41 * k1[0] + a42 * k2[0] + a43 * k3[0]),
                y1 + h * (a41 * k1[1] + a42 * k2[1] + a43 * k3[1]), y2 + h * (a41 * k1[2] + a42 * k2[2] + a43 * k3[2]),
                k4[0], k4[1], k4[2]);
        ens_rhs(p, a_cm, y0 + h * (a51 * k1[0] + a52 * k2[0] + a53 * k3[0] + a54 * k4[0]),
                y1 + h * (a51 * k1[1] + a52 * k2[1] + a53 * k3[1] + a54 * k4[1]),
                y2 + h * (a51 * k1[2] + a52 * k2[2] + a53 * k3[2] + a54 * k4[2]), k5[0], k5[1], k5[2]);
        ens_rhs(p, a_cm, y0 + h * (a61 * k1[0] + a62 * k2[0] + a63 * k3[0] + a64 * k4[0] + a65 * k5[0]),
                y1 + h * (a61 * k1[1] + a62 * k2[1] + a63 * k3[1] + a64 * k4[1] + a65 * k5[1]),
                y2 + h * (a61 * k1[2] + a62 * k2[2] + a63 * k3[2] + a64 * k4[2] + a65 * k5[2]), k6[0], k6[1], k6[2]);
        T yn[3];
        yn[0] = y0 + h * (b1 * k1[0] + b3 * k3[0] + b4 * k4[0] + b5 * k5[0] + b6 * k6[0]);
        yn[1] = y1 + h * (b1 * k1[1] + b3 * k3[1] + b4 * k4[1] + b5 * k5[1] + b6 * k6[1]);
        yn[2] = y2 + h * (b1 * k1[2] + b3 * k3[2] + b4 * k4[2] + b5 * k5[2] + b6 * k6[2]);
        T err = T(0);
        bool ok = yn[0] > T(0);
        if (ok) {
            ens_rhs(p, a_cm, yn[0], yn[1], yn[2], k7[0], k7[1], k7[2]);
            const T yo[3] = {y0, y1, y2};
            const T at[3] = {p.atol_r, p.atol_v, p.atol_j};
            for (int i = 0; i < 3; ++i) {
                const T e = h * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
                const T sc = at[i] + p.rtol * max(fabs(yo[i]), fabs(yn[i]));
                err += (e / sc) * (e / sc);
            }
            err = sqrt(err / T(3));
            ok = isfinite(err);
        }
        if (!ok) {
            h *= T(0.25);
            if (++fails > 60) { res.status = 2; break; }
            continue;
        }
        T fac = T(0.9) * pow(max(err, T(1e-10)), T(-0.2));
        fac = min(T(5), max(T(0.2), fac));
        if (err <= T(1)) {
            fails = 0;
            ++res.steps;
            if (yn[0] <= p.r_in) {  // terminal event: linear interpolation in r within the step
                const T frac = (y0 - p.r_in) / (y0 - yn[0]);
                res.t_drift = t + (frac * h - tc);
                res.status = 0;
                break;
            }
            const T dt = h - tc;  // Kahan
            const T tn = t + dt;
            tc = (tn - t) - dt;
            t = tn;
            y0 = yn[0]; y1 = yn[1]; y2 = yn[2];
            k1[0] = k7[0]; k1[1] = k7[1]; k1[2] = k7[2];
            h *= fac;
        } else {
            h *= min(T(1), fac);
            if (h < T(1e-12)) { res.status = 2; break; }
        }
    }
    if (res.status == 1) res.t_drift = t;
    return res;
}
