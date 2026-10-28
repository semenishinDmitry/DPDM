// Vector kernels used by the integrators' hot loops.
//
// Several backends are provided:
//   Scalar  - plain loops with compiler auto-vectorization explicitly disabled
//             (the bit-for-bit reference implementation),
//   Generic - plain loops; the compiler may auto-vectorize for the baseline ISA,
//   Neon    - explicit ARM64 Advanced SIMD intrinsics (aarch64 builds only),
//   Avx2    - explicit AVX2+FMA intrinsics, compiled with function-level target
//             attributes and selected only after a runtime CPU check.
//
// Element-wise kernels (axpy, lincomb) give identical results in every backend
// except where FMA contraction is used (Avx2/Neon lincomb use FMA; documented
// difference <= 1 ulp per term). Reductions (wrms, dot) use different summation
// orders, so results agree only to a few ulps (see tests/cpp/test_kernels.cpp).
#pragma once

#include <string_view>
#include <vector>

#include "dpd/ode/types.hpp"

namespace dpd::ode::kernels {

enum class Backend { Scalar, Generic, Neon, Avx2 };

[[nodiscard]] std::string_view to_string(Backend b) noexcept;
[[nodiscard]] std::vector<Backend> available_backends();
// Selects the backend used by subsequent calls (process-wide). Returns false
// (and leaves the selection unchanged) if the backend is unavailable here.
bool set_backend(Backend b) noexcept;
[[nodiscard]] Backend active_backend() noexcept;
// Best available backend: Avx2 if the CPU supports it, Neon on aarch64, else Generic.
[[nodiscard]] Backend default_backend() noexcept;

// y += a * x
void axpy(Real a, const Real* x, Real* y, Index n) noexcept;
// out = y + sum_{k<m} c[k] * K[k]   (m <= 8; out may alias y)
void lincomb(Real* out, const Real* y, const Real* c, const Real* const* K, int m, Index n) noexcept;
// sqrt( (1/n) sum_i (e_i / (atol_i + rtol * max(|a_i|, |b_i|)))^2 )
// atol has n entries if atol_is_vector, otherwise atol[0] is used for all.
Real wrms_err(const Real* e, const Real* a, const Real* b, const Real* atol, bool atol_is_vector,
              Real rtol, Index n) noexcept;
// sqrt( (1/n) sum_i (x_i / s_i)^2 )
Real wrms(const Real* x, const Real* s, Index n) noexcept;
Real dot(const Real* x, const Real* y, Index n) noexcept;

// Function table of one backend (exposed for tests/benchmarks).
struct Table {
    void (*axpy)(Real, const Real*, Real*, Index) noexcept;
    void (*lincomb)(Real*, const Real*, const Real*, const Real* const*, int, Index) noexcept;
    Real (*wrms_err)(const Real*, const Real*, const Real*, const Real*, bool, Real, Index) noexcept;
    Real (*wrms)(const Real*, const Real*, Index) noexcept;
    Real (*dot)(const Real*, const Real*, Index) noexcept;
};
// nullptr if unavailable on this build/CPU.
[[nodiscard]] const Table* table(Backend b) noexcept;

namespace detail {
const Table* scalar_table() noexcept;
const Table* generic_table() noexcept;
const Table* neon_table() noexcept;  // nullptr unless built for aarch64
const Table* avx2_table() noexcept;  // nullptr unless built for x86-64 with GCC/Clang
bool cpu_has_avx2_fma() noexcept;
}  // namespace detail

}  // namespace dpd::ode::kernels
