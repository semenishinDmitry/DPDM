// Scalar / generic kernels and backend dispatch.
#include "dpd/ode/kernels.hpp"

#include <atomic>
#include <cmath>

#if defined(__clang__)
#define DPD_NOVEC _Pragma("clang loop vectorize(disable) interleave(disable)")
#define DPD_NOVEC_FN
#elif defined(__GNUC__)
#define DPD_NOVEC
#define DPD_NOVEC_FN __attribute__((optimize("no-tree-vectorize")))
#elif defined(_MSC_VER)
#define DPD_NOVEC __pragma(loop(no_vector))
#define DPD_NOVEC_FN
#else
#define DPD_NOVEC
#define DPD_NOVEC_FN
#endif

namespace dpd::ode::kernels {

namespace {

// ---------------------------------------------------------------- scalar ---
DPD_NOVEC_FN void s_axpy(Real a, const Real* x, Real* y, Index n) noexcept {
    DPD_NOVEC
    for (Index i = 0; i < n; ++i) y[i] += a * x[i];
}

DPD_NOVEC_FN void s_lincomb(Real* out, const Real* y, const Real* c, const Real* const* K, int m,
                            Index n) noexcept {
    DPD_NOVEC
    for (Index i = 0; i < n; ++i) {
        Real acc = y[i];
        for (int k = 0; k < m; ++k) acc += c[k] * K[k][i];
        out[i] = acc;
    }
}

DPD_NOVEC_FN Real s_wrms_err(const Real* e, const Real* a, const Real* b, const Real* atol, bool vec,
                             Real rtol, Index n) noexcept {
    Real sum = 0;
    DPD_NOVEC
    for (Index i = 0; i < n; ++i) {
        const Real sc = (vec ? atol[i] : atol[0]) + rtol * std::max(std::abs(a[i]), std::abs(b[i]));
        const Real q = e[i] / sc;
        sum += q * q;
    }
    return n ? std::sqrt(sum / static_cast<Real>(n)) : Real{0};
}

DPD_NOVEC_FN Real s_wrms(const Real* x, const Real* s, Index n) noexcept {
    Real sum = 0;
    DPD_NOVEC
    for (Index i = 0; i < n; ++i) {
        const Real q = x[i] / s[i];
        sum += q * q;
    }
    return n ? std::sqrt(sum / static_cast<Real>(n)) : Real{0};
}

DPD_NOVEC_FN Real s_dot(const Real* x, const Real* y, Index n) noexcept {
    Real sum = 0;
    DPD_NOVEC
    for (Index i = 0; i < n; ++i) sum += x[i] * y[i];
    return sum;
}

// --------------------------------------------------------------- generic ---
// Same arithmetic as scalar but the compiler is free to vectorize. Reductions
// use 4 independent partial sums so that vectorization does not require
// reassociation permission (-ffast-math is never used).
void g_axpy(Real a, const Real* x, Real* y, Index n) noexcept {
    for (Index i = 0; i < n; ++i) y[i] += a * x[i];
}

void g_lincomb(Real* out, const Real* y, const Real* c, const Real* const* K, int m, Index n) noexcept {
    if (out != y)
        for (Index i = 0; i < n; ++i) out[i] = y[i];
    for (int k = 0; k < m; ++k) {
        const Real ck = c[k];
        const Real* kk = K[k];
        for (Index i = 0; i < n; ++i) out[i] += ck * kk[i];
    }
}

Real g_wrms_err(const Real* e, const Real* a, const Real* b, const Real* atol, bool vec, Real rtol,
                Index n) noexcept {
    Real s[4] = {0, 0, 0, 0};
    Index i = 0;
    for (; i + 4 <= n; i += 4) {
        for (int l = 0; l < 4; ++l) {
            const Index j = i + static_cast<Index>(l);
            const Real sc = (vec ? atol[j] : atol[0]) + rtol * std::max(std::abs(a[j]), std::abs(b[j]));
            const Real q = e[j] / sc;
            s[l] += q * q;
        }
    }
    for (; i < n; ++i) {
        const Real sc = (vec ? atol[i] : atol[0]) + rtol * std::max(std::abs(a[i]), std::abs(b[i]));
        const Real q = e[i] / sc;
        s[0] += q * q;
    }
    const Real sum = (s[0] + s[1]) + (s[2] + s[3]);
    return n ? std::sqrt(sum / static_cast<Real>(n)) : Real{0};
}

Real g_wrms(const Real* x, const Real* sc, Index n) noexcept {
    Real s[4] = {0, 0, 0, 0};
    Index i = 0;
    for (; i + 4 <= n; i += 4)
        for (int l = 0; l < 4; ++l) {
            const Real q = x[i + l] / sc[i + l];
            s[l] += q * q;
        }
    for (; i < n; ++i) {
        const Real q = x[i] / sc[i];
        s[0] += q * q;
    }
    const Real sum = (s[0] + s[1]) + (s[2] + s[3]);
    return n ? std::sqrt(sum / static_cast<Real>(n)) : Real{0};
}

Real g_dot(const Real* x, const Real* y, Index n) noexcept {
    Real s[4] = {0, 0, 0, 0};
    Index i = 0;
    for (; i + 4 <= n; i += 4)
        for (int l = 0; l < 4; ++l) s[l] += x[i + l] * y[i + l];
    for (; i < n; ++i) s[0] += x[i] * y[i];
    return (s[0] + s[1]) + (s[2] + s[3]);
}

constexpr Table kScalar{s_axpy, s_lincomb, s_wrms_err, s_wrms, s_dot};
constexpr Table kGeneric{g_axpy, g_lincomb, g_wrms_err, g_wrms, g_dot};

std::atomic<const Table*> g_active{nullptr};

const Table* resolve_default() noexcept {
    const Table* t = table(default_backend());
    return t ? t : &kGeneric;
}

inline const Table& active() noexcept {
    const Table* t = g_active.load(std::memory_order_acquire);
    if (!t) {
        t = resolve_default();
        g_active.store(t, std::memory_order_release);
    }
    return *t;
}

}  // namespace

namespace detail {
const Table* scalar_table() noexcept { return &kScalar; }
const Table* generic_table() noexcept { return &kGeneric; }
}  // namespace detail

std::string_view to_string(Backend b) noexcept {
    switch (b) {
        case Backend::Scalar: return "scalar";
        case Backend::Generic: return "generic";
        case Backend::Neon: return "neon";
        case Backend::Avx2: return "avx2";
    }
    return "unknown";
}

const Table* table(Backend b) noexcept {
    switch (b) {
        case Backend::Scalar: return detail::scalar_table();
        case Backend::Generic: return detail::generic_table();
        case Backend::Neon: return detail::neon_table();
        case Backend::Avx2: return detail::cpu_has_avx2_fma() ? detail::avx2_table() : nullptr;
    }
    return nullptr;
}

std::vector<Backend> available_backends() {
    std::vector<Backend> out;
    for (Backend b : {Backend::Scalar, Backend::Generic, Backend::Neon, Backend::Avx2})
        if (table(b)) out.push_back(b);
    return out;
}

Backend default_backend() noexcept {
    if (table(Backend::Avx2)) return Backend::Avx2;
    if (table(Backend::Neon)) return Backend::Neon;
    return Backend::Generic;
}

bool set_backend(Backend b) noexcept {
    const Table* t = table(b);
    if (!t) return false;
    g_active.store(t, std::memory_order_release);
    return true;
}

Backend active_backend() noexcept {
    const Table* t = &active();
    for (Backend b : {Backend::Scalar, Backend::Generic, Backend::Neon, Backend::Avx2})
        if (table(b) == t) return b;
    return Backend::Generic;
}

void axpy(Real a, const Real* x, Real* y, Index n) noexcept { active().axpy(a, x, y, n); }
void lincomb(Real* out, const Real* y, const Real* c, const Real* const* K, int m, Index n) noexcept {
    active().lincomb(out, y, c, K, m, n);
}
Real wrms_err(const Real* e, const Real* a, const Real* b, const Real* atol, bool v, Real rtol,
              Index n) noexcept {
    return active().wrms_err(e, a, b, atol, v, rtol, n);
}
Real wrms(const Real* x, const Real* s, Index n) noexcept { return active().wrms(x, s, n); }
Real dot(const Real* x, const Real* y, Index n) noexcept { return active().dot(x, y, n); }

}  // namespace dpd::ode::kernels
