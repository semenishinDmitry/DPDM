#include "dpd/ode/jacobian.hpp"

#include <cmath>

#include "dpd/ode/kernels.hpp"

namespace dpd::ode {

std::uint64_t fd_jacobian(const OdeSystem& sys, Real t, std::span<const Real> y, std::span<const Real> f0,
                          std::span<const Real> typ, DenseMatrix& J, std::span<Real> work_y,
                          std::span<Real> work_f) {
    const Index n = y.size();
    if (J.rows() != n || J.cols() != n) J.resize(n, n);
    const Real sq = std::sqrt(kEps);
    std::copy(y.begin(), y.end(), work_y.begin());
    for (Index j = 0; j < n; ++j) {
        const Real yj = y[j];
        Real d = sq * std::max(std::abs(yj), typ[j]);
        if (d == 0) d = sq;
        const Real yp = yj + d;
        d = yp - yj;  // exactly representable step
        work_y[j] = yp;
        sys.rhs(t, work_y, work_f);
        work_y[j] = yj;
        const Real inv = Real{1} / d;
        for (Index i = 0; i < n; ++i) J(i, j) = (work_f[i] - f0[i]) * inv;
    }
    return n;
}

std::uint64_t fd_jvp(const OdeSystem& sys, Real t, std::span<const Real> y, std::span<const Real> f0,
                     std::span<const Real> v, std::span<Real> Jv, std::span<Real> work_y) {
    const Index n = y.size();
    const Real vnorm = std::sqrt(kernels::dot(v.data(), v.data(), n));
    if (vnorm == 0) {
        std::fill(Jv.begin(), Jv.end(), Real{0});
        return 0;
    }
    const Real ynorm = std::sqrt(kernels::dot(y.data(), y.data(), n));
    const Real e = std::sqrt(kEps) * (1 + ynorm) / vnorm;
    for (Index i = 0; i < n; ++i) work_y[i] = y[i] + e * v[i];
    sys.rhs(t, work_y, Jv);
    const Real inv = Real{1} / e;
    for (Index i = 0; i < n; ++i) Jv[i] = (Jv[i] - f0[i]) * inv;
    return 1;
}

}  // namespace dpd::ode
