// Linear algebra used by the implicit integrator: dense LU and restarted GMRES.
#pragma once

#include <type_traits>
#include <utility>

#include "dpd/ode/types.hpp"

namespace dpd::ode {

// Non-owning, non-allocating callable reference (like the proposed std::function_ref).
template <class Sig>
class FunctionRef;

template <class R, class... Args>
class FunctionRef<R(Args...)> {
public:
    template <class F, class = std::enable_if_t<!std::is_same_v<std::decay_t<F>, FunctionRef>>>
    FunctionRef(F&& f) noexcept  // NOLINT(google-explicit-constructor)
        : obj_(const_cast<void*>(static_cast<const void*>(&f))),
          call_([](void* o, Args... a) -> R { return (*static_cast<std::remove_reference_t<F>*>(o))(std::forward<Args>(a)...); }) {}
    R operator()(Args... a) const { return call_(obj_, std::forward<Args>(a)...); }

private:
    void* obj_;
    R (*call_)(void*, Args...);
};

using LinearOp = FunctionRef<void(std::span<const Real>, std::span<Real>)>;

// LU factorization with partial pivoting, P A = L U. Storage is reused across
// factorizations of the same size (no allocation after resize()).
class DenseLU {
public:
    void resize(Index n);
    // Returns false if A is singular to working precision or contains non-finite values.
    bool factor(const DenseMatrix& A);
    // Solves A x = b in place. Precondition: last factor() returned true.
    void solve(std::span<Real> b) const;
    // min |U_ii| / max |U_ii|: a cheap (not rigorous) conditioning indicator.
    [[nodiscard]] Real pivot_ratio() const noexcept { return pivot_ratio_; }
    [[nodiscard]] Index size() const noexcept { return n_; }

private:
    Index n_ = 0;
    DenseMatrix lu_;
    std::vector<Index> piv_;
    Real pivot_ratio_ = 0;
};

struct GmresResult {
    bool converged = false;
    int iterations = 0;       // total inner iterations (matrix-vector products)
    Real residual = 0;        // final ||b - A x||_2 (recursively updated estimate)
    Real rel_residual = 0;    // residual / ||b||_2
    bool breakdown = false;   // lucky/unlucky breakdown occurred
};

// Restarted GMRES(m) with modified Gram-Schmidt (+ one DGKS re-orthogonalization
// pass when cancellation is detected), Givens rotations, and optional right
// preconditioning: A M^{-1} u = b, x = M^{-1} u.
class Gmres {
public:
    void resize(Index n, int restart);
    // x: initial guess on input, solution on output.
    GmresResult solve(LinearOp A, const LinearOp* M, std::span<const Real> b, std::span<Real> x,
                      Real rel_tol, int max_iter);

private:
    Index n_ = 0;
    int m_ = 0;
    DenseMatrix V_;   // (m+1) x n Krylov basis
    DenseMatrix Z_;   // m x n preconditioned basis (right preconditioning)
    DenseMatrix H_;   // (m+1) x m Hessenberg
    std::vector<Real> cs_, sn_, g_, yv_, w_, r_;
};

}  // namespace dpd::ode
