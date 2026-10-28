#include <catch2/catch_test_macros.hpp>
#include <random>

#include "dpd/ode/jacobian.hpp"
#include "dpd/ode/linalg.hpp"
#include "dpd/ode/test_problems.hpp"

using namespace dpd::ode;

namespace {
DenseMatrix random_matrix(Index n, std::mt19937_64& g, Real diag_boost) {
    std::uniform_real_distribution<Real> d(-1, 1);
    DenseMatrix A(n, n);
    for (Index i = 0; i < n; ++i)
        for (Index j = 0; j < n; ++j) A(i, j) = d(g) + (i == j ? diag_boost : 0);
    return A;
}
std::vector<Real> matvec(const DenseMatrix& A, std::span<const Real> x) {
    std::vector<Real> y(A.rows(), 0);
    for (Index i = 0; i < A.rows(); ++i)
        for (Index j = 0; j < A.cols(); ++j) y[i] += A(i, j) * x[j];
    return y;
}
Real nrm(std::span<const Real> v) {
    Real s = 0;
    for (Real x : v) s += x * x;
    return std::sqrt(s);
}
}  // namespace

TEST_CASE("dense LU solves well-conditioned systems", "[linalg]") {
    std::mt19937_64 g(1);
    for (Index n : {1u, 2u, 5u, 17u, 64u}) {
        const DenseMatrix A = random_matrix(n, g, 0.0 + static_cast<Real>(n) * 0.5);
        std::vector<Real> x(n);
        for (Index i = 0; i < n; ++i) x[i] = std::sin(static_cast<Real>(i) + 1);
        auto b = matvec(A, x);
        DenseLU lu;
        lu.resize(n);
        REQUIRE(lu.factor(A));
        lu.solve(b);
        for (Index i = 0; i < n; ++i) CHECK(std::abs(b[i] - x[i]) < 1e-11);
        CHECK(lu.pivot_ratio() > 0);
    }
}

TEST_CASE("dense LU rejects singular and non-finite matrices", "[linalg][failure]") {
    DenseMatrix A(3, 3);
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 2; A(1, 1) = 4; A(1, 2) = 6;  // dependent row
    A(2, 0) = 1; A(2, 1) = 0; A(2, 2) = 1;
    DenseLU lu;
    lu.resize(3);
    CHECK_FALSE(lu.factor(A));
    DenseMatrix B(2, 2);
    B(0, 0) = std::nan(""); B(1, 1) = 1;
    CHECK_FALSE(lu.factor(B));
    // nearly singular (pivot ratio below 16 n eps) is rejected too
    DenseMatrix C(2, 2);
    C(0, 0) = 1; C(1, 1) = 1e-17;
    CHECK_FALSE(lu.factor(C));
}

TEST_CASE("GMRES converges to the direct solution", "[linalg][gmres]") {
    std::mt19937_64 g(7);
    const Index n = 120;
    const DenseMatrix A = random_matrix(n, g, 12.0);  // nonsymmetric, well conditioned
    std::vector<Real> b(n);
    for (Index i = 0; i < n; ++i) b[i] = std::cos(0.3 * static_cast<Real>(i));
    auto xd = b;
    DenseLU lu;
    lu.resize(n);
    REQUIRE(lu.factor(A));
    lu.solve(xd);
    auto op = [&](std::span<const Real> v, std::span<Real> out) {
        auto r = matvec(A, v);
        std::copy(r.begin(), r.end(), out.begin());
    };
    for (int restart : {5, 30, 120}) {
        CAPTURE(restart);
        Gmres gm;
        gm.resize(n, restart);
        std::vector<Real> x(n, 0);
        const auto res = gm.solve(op, nullptr, b, x, 1e-12, 2000);
        CHECK(res.converged);
        std::vector<Real> diff(n);
        for (Index i = 0; i < n; ++i) diff[i] = x[i] - xd[i];
        CHECK(nrm(diff) <= 1e-9 * nrm(xd));
        // true residual matches the reported one
        auto Ax = matvec(A, x);
        for (Index i = 0; i < n; ++i) Ax[i] = b[i] - Ax[i];
        CHECK(nrm(Ax) <= 1e-11 * nrm(b) * 10);
    }
}

TEST_CASE("GMRES right preconditioning reduces iterations", "[linalg][gmres]") {
    const Index n = 400;
    problems::ReactionDiffusion1D rd(n, 1.0, 1.0);
    auto u = rd.initial_condition();
    const Real c = 1e-3;  // I - cJ with stiff diffusion
    std::vector<Real> tmp(n);
    auto op = [&](std::span<const Real> v, std::span<Real> out) {
        rd.jvp(0, u, v, tmp);
        for (Index i = 0; i < n; ++i) out[i] = v[i] - c * tmp[i];
    };
    problems::TridiagonalRDPreconditioner P(n, 1.0, 1.0);
    P.setup(0, u, c);
    auto pre = [&](std::span<const Real> r, std::span<Real> z) { P.apply(r, z); };
    const LinearOp Pref(pre);
    std::vector<Real> b(n, 1.0);
    Gmres gm;
    gm.resize(n, 50);
    std::vector<Real> x1(n, 0), x2(n, 0);
    const auto r1 = gm.solve(op, nullptr, b, x1, 1e-10, 5000);
    const auto r2 = gm.solve(op, &Pref, b, x2, 1e-10, 5000);
    CHECK(r1.converged);
    CHECK(r2.converged);
    CHECK(r2.iterations <= 2);           // exact preconditioner
    CHECK(r2.iterations < r1.iterations);
    for (Index i = 0; i < n; ++i) CHECK(std::abs(x1[i] - x2[i]) < 1e-7 * (1 + std::abs(x2[i])));
}

TEST_CASE("GMRES handles a zero right-hand side and reports non-convergence", "[linalg][gmres][failure]") {
    const Index n = 50;
    Gmres gm;
    gm.resize(n, 3);
    std::vector<Real> b(n, 0), x(n, 1);
    auto id = [](std::span<const Real> v, std::span<Real> o) { std::copy(v.begin(), v.end(), o.begin()); };
    CHECK(gm.solve(id, nullptr, b, x, 1e-10, 10).converged);
    for (Real v : x) CHECK(v == 0);
    // Rotation-like operator: GMRES(3) with 6 iterations cannot reach 1e-14 on a cyclic shift.
    auto shift = [](std::span<const Real> v, std::span<Real> o) {
        const Index n = v.size();
        for (Index i = 0; i < n; ++i) o[(i + 1) % n] = v[i];
    };
    std::vector<Real> b2(n, 0);
    b2[0] = 1;
    std::vector<Real> x2(n, 0);
    const auto r = gm.solve(shift, nullptr, b2, x2, 1e-14, 6);
    CHECK_FALSE(r.converged);
    CHECK(r.iterations == 6);
}

TEST_CASE("finite-difference Jacobian and JVP match analytic derivatives", "[jacobian]") {
    problems::Robertson rob;
    const std::vector<Real> y{0.9, 3e-5, 0.1};
    DenseMatrix Ja(3, 3), Jf(3, 3);
    rob.jacobian(0, y, Ja);
    std::vector<Real> f0(3), wy(3), wf(3), typ{1e-6, 1e-10, 1e-6};
    rob.rhs(0, y, f0);
    fd_jacobian(rob, 0, y, f0, typ, Jf, wy, wf);
    for (Index i = 0; i < 3; ++i)
        for (Index j = 0; j < 3; ++j) CHECK(std::abs(Jf(i, j) - Ja(i, j)) <= 1e-6 * (1 + std::abs(Ja(i, j))));
    const std::vector<Real> v{0.3, -1e-4, 0.2};
    std::vector<Real> jv(3), ex(3, 0);
    fd_jvp(rob, 0, y, f0, v, jv, wy);
    for (Index i = 0; i < 3; ++i)
        for (Index j = 0; j < 3; ++j) ex[i] += Ja(i, j) * v[j];
    for (Index i = 0; i < 3; ++i) CHECK(std::abs(jv[i] - ex[i]) <= 1e-5 * (1 + std::abs(ex[i])));
}
