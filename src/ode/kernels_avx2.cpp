// Explicit AVX2 + FMA kernels for x86-64.
//
// These functions are compiled with function-level target attributes rather
// than a global -mavx2 flag, so the rest of the library stays baseline x86-64.
// They are only ever called after detail::cpu_has_avx2_fma() returned true.
#include "dpd/ode/kernels.hpp"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__)) && \
    !defined(DPD_DISABLE_AVX2)
#include <immintrin.h>

#include <cmath>

#define DPD_AVX2 __attribute__((target("avx2,fma")))

namespace dpd::ode::kernels {
namespace {

DPD_AVX2 inline __m256d vabs(__m256d v) { return _mm256_andnot_pd(_mm256_set1_pd(-0.0), v); }

DPD_AVX2 inline Real hsum(__m256d v) {
    const __m128d lo = _mm256_castpd256_pd128(v);
    const __m128d hi = _mm256_extractf128_pd(v, 1);
    const __m128d s = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)));
}

DPD_AVX2 void x_axpy(Real a, const Real* x, Real* y, Index n) noexcept {
    const __m256d va = _mm256_set1_pd(a);
    Index i = 0;
    for (; i + 4 <= n; i += 4)
        _mm256_storeu_pd(y + i, _mm256_add_pd(_mm256_loadu_pd(y + i), _mm256_mul_pd(va, _mm256_loadu_pd(x + i))));
    for (; i < n; ++i) y[i] += a * x[i];
}

DPD_AVX2 void x_lincomb(Real* out, const Real* y, const Real* c, const Real* const* K, int m,
                        Index n) noexcept {
    Index i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256d a0 = _mm256_loadu_pd(y + i), a1 = _mm256_loadu_pd(y + i + 4);
        for (int k = 0; k < m; ++k) {
            const __m256d ck = _mm256_set1_pd(c[k]);
            a0 = _mm256_fmadd_pd(ck, _mm256_loadu_pd(K[k] + i), a0);
            a1 = _mm256_fmadd_pd(ck, _mm256_loadu_pd(K[k] + i + 4), a1);
        }
        _mm256_storeu_pd(out + i, a0);
        _mm256_storeu_pd(out + i + 4, a1);
    }
    for (; i < n; ++i) {
        Real acc = y[i];
        for (int k = 0; k < m; ++k) acc = std::fma(c[k], K[k][i], acc);
        out[i] = acc;
    }
}

DPD_AVX2 Real x_wrms_err(const Real* e, const Real* a, const Real* b, const Real* atol, bool vec,
                         Real rtol, Index n) noexcept {
    if (n == 0) return 0;
    const __m256d vr = _mm256_set1_pd(rtol);
    const __m256d va0 = _mm256_set1_pd(atol[0]);
    __m256d s = _mm256_setzero_pd();
    Index i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d at = vec ? _mm256_loadu_pd(atol + i) : va0;
        const __m256d mx = _mm256_max_pd(vabs(_mm256_loadu_pd(a + i)), vabs(_mm256_loadu_pd(b + i)));
        const __m256d q = _mm256_div_pd(_mm256_loadu_pd(e + i), _mm256_fmadd_pd(vr, mx, at));
        s = _mm256_fmadd_pd(q, q, s);
    }
    Real sum = hsum(s);
    for (; i < n; ++i) {
        const Real sc = (vec ? atol[i] : atol[0]) + rtol * std::max(std::abs(a[i]), std::abs(b[i]));
        const Real q = e[i] / sc;
        sum += q * q;
    }
    return n ? std::sqrt(sum / static_cast<Real>(n)) : Real{0};
}

DPD_AVX2 Real x_wrms(const Real* x, const Real* sc, Index n) noexcept {
    __m256d s = _mm256_setzero_pd();
    Index i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d q = _mm256_div_pd(_mm256_loadu_pd(x + i), _mm256_loadu_pd(sc + i));
        s = _mm256_fmadd_pd(q, q, s);
    }
    Real sum = hsum(s);
    for (; i < n; ++i) {
        const Real q = x[i] / sc[i];
        sum += q * q;
    }
    return n ? std::sqrt(sum / static_cast<Real>(n)) : Real{0};
}

DPD_AVX2 Real x_dot(const Real* x, const Real* y, Index n) noexcept {
    __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
    Index i = 0;
    for (; i + 8 <= n; i += 8) {
        s0 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i), _mm256_loadu_pd(y + i), s0);
        s1 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + 4), _mm256_loadu_pd(y + i + 4), s1);
    }
    Real sum = hsum(_mm256_add_pd(s0, s1));
    for (; i < n; ++i) sum += x[i] * y[i];
    return sum;
}

constexpr Table kAvx2{x_axpy, x_lincomb, x_wrms_err, x_wrms, x_dot};

}  // namespace

namespace detail {
const Table* avx2_table() noexcept { return &kAvx2; }
bool cpu_has_avx2_fma() noexcept {
    // __builtin_cpu_supports also verifies OS support for the YMM state (XCR0).
    static const bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return ok;
}
}  // namespace detail
}  // namespace dpd::ode::kernels

#else

namespace dpd::ode::kernels::detail {
const Table* avx2_table() noexcept { return nullptr; }
bool cpu_has_avx2_fma() noexcept { return false; }
}  // namespace dpd::ode::kernels::detail

#endif
