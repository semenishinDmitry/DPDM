// Linear stiff test system y' = A y with A = Q diag(lambda) Q^T, Q a fixed
// rotation: eigenvalues are known exactly, so is the solution.
#pragma once
#include <cmath>

#include "dpd/ode/test_problems.hpp"

namespace testing_util {
using dpd::ode::DenseMatrix;
using dpd::ode::Real;

inline DenseMatrix rotation3() {
    const Real a = 0.3, b = 0.7, c = 1.1;
    DenseMatrix Rx(3, 3), Ry(3, 3), Rz(3, 3), Q(3, 3), T(3, 3);
    Rx(0, 0) = 1; Rx(1, 1) = std::cos(a); Rx(1, 2) = -std::sin(a); Rx(2, 1) = std::sin(a); Rx(2, 2) = std::cos(a);
    Ry(1, 1) = 1; Ry(0, 0) = std::cos(b); Ry(0, 2) = std::sin(b); Ry(2, 0) = -std::sin(b); Ry(2, 2) = std::cos(b);
    Rz(2, 2) = 1; Rz(0, 0) = std::cos(c); Rz(0, 1) = -std::sin(c); Rz(1, 0) = std::sin(c); Rz(1, 1) = std::cos(c);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) T(i, j) += Rx(i, k) * Ry(k, j);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) Q(i, j) += T(i, k) * Rz(k, j);
    return Q;
}

struct StiffLinear {
    DenseMatrix Q = rotation3();
    Real lam[3];
    explicit StiffLinear(Real l0, Real l1, Real l2) : lam{l0, l1, l2} {}
    [[nodiscard]] DenseMatrix matrix() const {
        DenseMatrix A(3, 3);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k) A(i, j) += Q(i, k) * lam[k] * Q(j, k);
        return A;
    }
    // y(t) = Q exp(Lambda t) Q^T y0
    [[nodiscard]] std::vector<Real> exact(Real t, const std::vector<Real>& y0) const {
        Real w[3] = {0, 0, 0};
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 3; ++j) w[k] += Q(j, k) * y0[j];
        std::vector<Real> y(3, 0);
        for (int i = 0; i < 3; ++i)
            for (int k = 0; k < 3; ++k) y[i] += Q(i, k) * std::exp(lam[k] * t) * w[k];
        return y;
    }
};
}  // namespace testing_util
