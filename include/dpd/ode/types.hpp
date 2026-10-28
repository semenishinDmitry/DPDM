// Core value types shared by every part of the ODE library.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace dpd::ode {

using Real = double;
using Index = std::size_t;

inline constexpr Real kEps = std::numeric_limits<Real>::epsilon();
inline constexpr Real kInf = std::numeric_limits<Real>::infinity();

// Dense row-major matrix with contiguous storage. Intentionally minimal: the
// library only needs storage, element access and a raw pointer for kernels.
class DenseMatrix {
public:
    DenseMatrix() = default;
    DenseMatrix(Index rows, Index cols) : rows_(rows), cols_(cols), data_(rows * cols, Real{0}) {}

    void resize(Index rows, Index cols) {
        rows_ = rows;
        cols_ = cols;
        data_.assign(rows * cols, Real{0});
    }
    void fill(Real v) { std::fill(data_.begin(), data_.end(), v); }

    [[nodiscard]] Index rows() const noexcept { return rows_; }
    [[nodiscard]] Index cols() const noexcept { return cols_; }
    Real& operator()(Index i, Index j) noexcept { return data_[i * cols_ + j]; }
    Real operator()(Index i, Index j) const noexcept { return data_[i * cols_ + j]; }
    std::span<Real> row(Index i) noexcept { return {data_.data() + i * cols_, cols_}; }
    std::span<const Real> row(Index i) const noexcept { return {data_.data() + i * cols_, cols_}; }
    Real* data() noexcept { return data_.data(); }
    const Real* data() const noexcept { return data_.data(); }

private:
    Index rows_ = 0;
    Index cols_ = 0;
    std::vector<Real> data_;
};

enum class Method : std::uint8_t {
    Dopri5,  // adaptive explicit Dormand-Prince 5(4)
    Rk4,     // classical fixed-step RK4 (verification only)
    Bdf,     // variable-order variable-step BDF (orders 1..5)
    Auto,    // start with Dopri5, switch to BDF when stiffness is detected
};

enum class LinearSolverKind : std::uint8_t {
    DenseLU,  // direct LU with partial pivoting (small systems)
    Gmres,    // matrix-free Newton-Krylov with restarted GMRES
};

enum class Status : std::uint8_t {
    Success,             // reached t_end
    TerminalEvent,       // stopped by a terminal event
    MaxStepsExceeded,
    StepSizeUnderflow,   // h fell below the representable resolution at t
    TimeResolutionLost,  // t + h == t
    NonFiniteState,      // RHS produced NaN/Inf that step reduction could not cure
    NewtonFailure,       // nonlinear solver failed repeatedly
    LinearSolverFailure, // singular / ill-conditioned iteration matrix
    DomainViolation,     // model's in_domain() rejected states repeatedly
    Cancelled,
    InvalidInput,
};

[[nodiscard]] std::string_view to_string(Status s) noexcept;
[[nodiscard]] std::string_view to_string(Method m) noexcept;
[[nodiscard]] std::string_view to_string(LinearSolverKind k) noexcept;
[[nodiscard]] bool is_success(Status s) noexcept;

struct MethodSwitch {
    Real t;
    Method from;
    Method to;
    std::string reason;
};

struct Stats {
    std::uint64_t n_rhs = 0;            // RHS evaluations (including FD Jacobians)
    std::uint64_t n_jac = 0;            // Jacobian evaluations (analytic or FD)
    std::uint64_t n_jvp = 0;            // Jacobian-vector products
    std::uint64_t n_lu = 0;             // LU factorizations
    std::uint64_t n_steps = 0;          // accepted steps
    std::uint64_t n_reject_error = 0;   // rejected by the local error test
    std::uint64_t n_reject_newton = 0;  // rejected because Newton did not converge
    std::uint64_t n_reject_nonfinite = 0;
    std::uint64_t n_reject_domain = 0;
    std::uint64_t n_newton_iter = 0;
    std::uint64_t n_linear_iter = 0;    // GMRES inner iterations
    std::uint64_t n_lost_increment = 0; // accepted steps where some y_i + dy_i == y_i, dy_i != 0
    std::uint64_t n_events = 0;
    Real h_min = kInf;                  // smallest accepted step
    Real h_max = 0;                     // largest accepted step
    Real h_last = 0;
    std::vector<MethodSwitch> switches;
};

}  // namespace dpd::ode
