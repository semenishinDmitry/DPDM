// Explicit ARM64 Advanced SIMD (NEON) kernels. NEON is architecturally
// mandatory on AArch64, so no runtime check is needed; on other targets this
// translation unit only provides a null table.
#include "dpd/ode/kernels.hpp"

#if (defined(__aarch64__) || defined(_M_ARM64)) && !defined(DPD_DISABLE_NEON)
#include <arm_neon.h>

#include <cmath>

namespace dpd::ode::kernels {
namespace {

void n_axpy(Real a, const Real* x, Real* y, Index n) noexcept {
    const float64x2_t va = vdupq_n_f64(a);
    Index i = 0;
    for (; i + 4 <= n; i += 4) {
        float64x2_t y0 = vld1q_f64(y + i), y1 = vld1q_f64(y + i + 2);
        // Separate multiply and add (no FMA) so axpy matches the scalar result bitwise.
        y0 = vaddq_f64(y0, vmulq_f64(va, vld1q_f64(x + i)));
        y1 = vaddq_f64(y1, vmulq_f64(va, vld1q_f64(x + i + 2)));
        vst1q_f64(y + i, y0);
        vst1q_f64(y + i + 2, y1);
    }
    for (; i < n; ++i) y[i] += a * x[i];
}

void n_lincomb(Real* out, const Real* y, const Real* c, const Real* const* K, int m, Index n) noexcept {
    Index i = 0;
    for (; i + 4 <= n; i += 4) {
        float64x2_t a0 = vld1q_f64(y + i), a1 = vld1q_f64(y + i + 2);
        for (int k = 0; k < m; ++k) {
            const float64x2_t ck = vdupq_n_f64(c[k]);
            a0 = vfmaq_f64(a0, ck, vld1q_f64(K[k] + i));
            a1 = vfmaq_f64(a1, ck, vld1q_f64(K[k] + i + 2));
        }
        vst1q_f64(out + i, a0);
        vst1q_f64(out + i + 2, a1);
    }
    for (; i < n; ++i) {
        Real acc = y[i];
        for (int k = 0; k < m; ++k) acc = std::fma(c[k], K[k][i], acc);
        out[i] = acc;
    }
}

Real n_wrms_err(const Real* e, const Real* a, const Real* b, const Real* atol, bool vec, Real rtol,
                Index n) noexcept {
    if (n == 0) return 0;
    const float64x2_t vr = vdupq_n_f64(rtol);
    const float64x2_t va0 = vdupq_n_f64(atol[0]);
    float64x2_t s0 = vdupq_n_f64(0), s1 = vdupq_n_f64(0);
    Index i = 0;
    for (; i + 4 <= n; i += 4) {
        const float64x2_t at0 = vec ? vld1q_f64(atol + i) : va0;
        const float64x2_t at1 = vec ? vld1q_f64(atol + i + 2) : va0;
        const float64x2_t m0 = vmaxq_f64(vabsq_f64(vld1q_f64(a + i)), vabsq_f64(vld1q_f64(b + i)));
        const float64x2_t m1 = vmaxq_f64(vabsq_f64(vld1q_f64(a + i + 2)), vabsq_f64(vld1q_f64(b + i + 2)));
        const float64x2_t q0 = vdivq_f64(vld1q_f64(e + i), vaddq_f64(at0, vmulq_f64(vr, m0)));
        const float64x2_t q1 = vdivq_f64(vld1q_f64(e + i + 2), vaddq_f64(at1, vmulq_f64(vr, m1)));
        s0 = vfmaq_f64(s0, q0, q0);
        s1 = vfmaq_f64(s1, q1, q1);
    }
    Real sum = vaddvq_f64(vaddq_f64(s0, s1));
    for (; i < n; ++i) {
        const Real sc = (vec ? atol[i] : atol[0]) + rtol * std::max(std::abs(a[i]), std::abs(b[i]));
        const Real q = e[i] / sc;
        sum += q * q;
    }
    return n ? std::sqrt(sum / static_cast<Real>(n)) : Real{0};
}

Real n_wrms(const Real* x, const Real* sc, Index n) noexcept {
    float64x2_t s0 = vdupq_n_f64(0), s1 = vdupq_n_f64(0);
    Index i = 0;
    for (; i + 4 <= n; i += 4) {
        const float64x2_t q0 = vdivq_f64(vld1q_f64(x + i), vld1q_f64(sc + i));
        const float64x2_t q1 = vdivq_f64(vld1q_f64(x + i + 2), vld1q_f64(sc + i + 2));
        s0 = vfmaq_f64(s0, q0, q0);
        s1 = vfmaq_f64(s1, q1, q1);
    }
    Real sum = vaddvq_f64(vaddq_f64(s0, s1));
    for (; i < n; ++i) {
        const Real q = x[i] / sc[i];
        sum += q * q;
    }
    return n ? std::sqrt(sum / static_cast<Real>(n)) : Real{0};
}

Real n_dot(const Real* x, const Real* y, Index n) noexcept {
    float64x2_t s0 = vdupq_n_f64(0), s1 = vdupq_n_f64(0);
    Index i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 = vfmaq_f64(s0, vld1q_f64(x + i), vld1q_f64(y + i));
        s1 = vfmaq_f64(s1, vld1q_f64(x + i + 2), vld1q_f64(y + i + 2));
    }
    Real sum = vaddvq_f64(vaddq_f64(s0, s1));
    for (; i < n; ++i) sum += x[i] * y[i];
    return sum;
}

constexpr Table kNeon{n_axpy, n_lincomb, n_wrms_err, n_wrms, n_dot};

}  // namespace

namespace detail {
const Table* neon_table() noexcept { return &kNeon; }
}  // namespace detail
}  // namespace dpd::ode::kernels

#else

namespace dpd::ode::kernels::detail {
const Table* neon_table() noexcept { return nullptr; }
}  // namespace dpd::ode::kernels::detail

#endif
