// Stochastic differential equations with diagonal noise (Ito):
//   dY_i = f_i(t, Y) dt + g_i(t, Y) dW_i,   W_i independent Wiener processes.
//
// Fixed-step schemes:
//   EulerMaruyama        weak order 1, strong 1/2
//   Milstein             strong order 1 (diagonal noise; uses dg_i/dy_i)
//   LinearImplicitEuler  Y+ = Y + (f dt + g dW) / (1 - dt df_i/dy_i): weak order 1,
//                        stable for stiff dissipative drift (df_i/dy_i << 0)
// Ensembles run in parallel; every path has its own counter-seeded xoshiro256++
// generator, so results are bitwise independent of the number of threads.
#pragma once

#include <atomic>
#include <cmath>
#include <cstdint>
#include <string>

#include "dpd/ode/types.hpp"

namespace dpd::ode {

class SdeSystem {
public:
    virtual ~SdeSystem() = default;
    [[nodiscard]] virtual Index size() const = 0;
    virtual void drift(Real t, std::span<const Real> y, std::span<Real> f) const = 0;
    virtual void diffusion(Real t, std::span<const Real> y, std::span<Real> g) const = 0;
    // Optional diagonal derivatives d f_i / d y_i and d g_i / d y_i; central
    // finite differences are used when not provided.
    [[nodiscard]] virtual bool has_derivatives() const { return false; }
    virtual void drift_diag_jacobian(Real, std::span<const Real>, std::span<Real>) const {}
    virtual void diffusion_derivative(Real, std::span<const Real>, std::span<Real>) const {}
};

enum class SdeScheme : std::uint8_t { EulerMaruyama, Milstein, LinearImplicitEuler };
[[nodiscard]] std::string_view to_string(SdeScheme s) noexcept;

// xoshiro256++ (Blackman & Vigna) seeded through splitmix64; standard normals
// by the Box-Muller transform.
class Rng {
public:
    Rng(std::uint64_t seed, std::uint64_t stream);
    std::uint64_t next() noexcept;
    Real uniform() noexcept;  // (0, 1)
    Real normal() noexcept;

private:
    std::uint64_t s_[4];
    Real spare_ = 0;
    bool has_spare_ = false;
};

struct SdeOptions {
    SdeScheme scheme = SdeScheme::EulerMaruyama;
    Real dt = 1e-3;                 // fixed step (the last step to each output time is shortened)
    std::uint64_t seed = 1;
    std::size_t n_paths = 1000;
    unsigned threads = 0;           // 0 = hardware concurrency
    std::vector<Real> t_out;        // increasing output times (> t0); required
    const std::atomic<bool>* cancel = nullptr;
};

struct SdeResult {
    Status status = Status::InvalidInput;
    std::string message;
    Index n = 0;
    std::size_t n_paths = 0;
    std::vector<Real> t_out;
    std::vector<Real> y;            // [path][output][component], row-major
    std::size_t n_failed_paths = 0; // paths that produced non-finite values (stored as NaN)
    std::uint64_t n_steps = 0;      // total steps over all paths

    [[nodiscard]] Real at(std::size_t path, std::size_t k, Index i) const {
        return y[(path * t_out.size() + k) * n + i];
    }
};

[[nodiscard]] SdeResult simulate(const SdeSystem& sys, Real t0, std::span<const Real> y0, const SdeOptions& opt);

// One path from t0 to t_end with step dt; returns false on non-finite values.
// If W != nullptr it receives the accumulated Wiener increments W(t_end) (testing
// of strong convergence against exact solutions).
bool simulate_path(const SdeSystem& sys, SdeScheme scheme, Real t0, std::span<Real> y, Real t_end, Real dt, Rng& rng,
                   std::span<Real> W = {});

}  // namespace dpd::ode
