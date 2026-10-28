#include "dpd/ode/linalg.hpp"

#include <cmath>

#include "dpd/ode/kernels.hpp"

namespace dpd::ode {

void DenseLU::resize(Index n) {
    n_ = n;
    lu_.resize(n, n);
    piv_.assign(n, 0);
}

bool DenseLU::factor(const DenseMatrix& A) {
    if (A.rows() != n_ || A.cols() != n_) resize(A.rows());
    const Index n = n_;
    std::copy(A.data(), A.data() + n * n, lu_.data());
    Real umax = 0;
    Real umin = kInf;
    for (Index k = 0; k < n; ++k) {
        Index p = k;
        Real amax = std::abs(lu_(k, k));
        for (Index i = k + 1; i < n; ++i) {
            const Real v = std::abs(lu_(i, k));
            if (v > amax) {
                amax = v;
                p = i;
            }
        }
        piv_[k] = p;
        if (!(amax > 0) || !std::isfinite(amax)) {  // also catches NaN
            pivot_ratio_ = 0;
            return false;
        }
        if (p != k)
            for (Index j = 0; j < n; ++j) std::swap(lu_(k, j), lu_(p, j));
        const Real inv = Real{1} / lu_(k, k);
        Real* rowk = lu_.data() + k * n;
        for (Index i = k + 1; i < n; ++i) {
            Real* rowi = lu_.data() + i * n;
            const Real lik = rowi[k] * inv;
            rowi[k] = lik;
            if (lik != 0) kernels::axpy(-lik, rowk + k + 1, rowi + k + 1, n - k - 1);
        }
        umax = std::max(umax, amax);
        umin = std::min(umin, amax);
    }
    pivot_ratio_ = n ? umin / umax : 1;
    // Reject matrices singular to working precision.
    return pivot_ratio_ > 16 * kEps * static_cast<Real>(n);
}

void DenseLU::solve(std::span<Real> b) const {
    const Index n = n_;
    for (Index k = 0; k < n; ++k)
        if (piv_[k] != k) std::swap(b[k], b[piv_[k]]);
    for (Index i = 1; i < n; ++i) {
        const Real* row = lu_.data() + i * n;
        Real s = b[i];
        for (Index j = 0; j < i; ++j) s -= row[j] * b[j];
        b[i] = s;
    }
    for (Index ii = n; ii-- > 0;) {
        const Real* row = lu_.data() + ii * n;
        Real s = b[ii];
        for (Index j = ii + 1; j < n; ++j) s -= row[j] * b[j];
        b[ii] = s / row[ii];
    }
}

void Gmres::resize(Index n, int restart) {
    n_ = n;
    m_ = std::max(1, restart);
    V_.resize(static_cast<Index>(m_) + 1, n);
    Z_.resize(static_cast<Index>(m_), n);
    H_.resize(static_cast<Index>(m_) + 1, static_cast<Index>(m_));
    cs_.assign(m_, 0);
    sn_.assign(m_, 0);
    g_.assign(m_ + 1, 0);
    yv_.assign(m_, 0);
    w_.assign(n, 0);
    r_.assign(n, 0);
}

namespace {
inline Real nrm2(std::span<const Real> v) { return std::sqrt(kernels::dot(v.data(), v.data(), v.size())); }
}  // namespace

GmresResult Gmres::solve(LinearOp A, const LinearOp* M, std::span<const Real> b, std::span<Real> x,
                         Real rel_tol, int max_iter) {
    GmresResult res;
    const Index n = n_;
    const Real bnorm = nrm2(b);
    if (bnorm == 0) {
        std::fill(x.begin(), x.end(), Real{0});
        res.converged = true;
        return res;
    }
    const Real target = rel_tol * bnorm;
    int total = 0;
    while (true) {
        // r = b - A x
        A(x, r_);
        for (Index i = 0; i < n; ++i) r_[i] = b[i] - r_[i];
        Real beta = nrm2(r_);
        res.residual = beta;
        res.rel_residual = beta / bnorm;
        if (beta <= target) {
            res.converged = true;
            break;
        }
        if (total >= max_iter) break;
        {
            auto v0 = V_.row(0);
            for (Index i = 0; i < n; ++i) v0[i] = r_[i] / beta;
        }
        std::fill(g_.begin(), g_.end(), Real{0});
        g_[0] = beta;
        int j = 0;
        bool done = false;
        for (; j < m_ && total < max_iter; ++j, ++total) {
            auto vj = V_.row(static_cast<Index>(j));
            if (M) {
                auto zj = Z_.row(static_cast<Index>(j));
                (*M)(vj, zj);
                A(zj, w_);
            } else {
                A(vj, w_);
            }
            const Real wnorm0 = nrm2(w_);
            for (int pass = 0; pass < 2; ++pass) {
                for (int i = 0; i <= j; ++i) {
                    auto vi = V_.row(static_cast<Index>(i));
                    const Real h = kernels::dot(w_.data(), vi.data(), n);
                    H_(static_cast<Index>(i), static_cast<Index>(j)) += h;
                    kernels::axpy(-h, vi.data(), w_.data(), n);
                }
                // DGKS criterion: re-orthogonalize only if severe cancellation occurred.
                if (pass == 0 && nrm2(w_) > 0.7 * wnorm0) break;
            }
            const Real hnext = nrm2(w_);
            H_(static_cast<Index>(j) + 1, static_cast<Index>(j)) = hnext;
            // Apply previous rotations to column j.
            for (int i = 0; i < j; ++i) {
                const Real a = H_(static_cast<Index>(i), static_cast<Index>(j));
                const Real c = H_(static_cast<Index>(i) + 1, static_cast<Index>(j));
                H_(static_cast<Index>(i), static_cast<Index>(j)) = cs_[i] * a + sn_[i] * c;
                H_(static_cast<Index>(i) + 1, static_cast<Index>(j)) = -sn_[i] * a + cs_[i] * c;
            }
            const Real a = H_(static_cast<Index>(j), static_cast<Index>(j));
            const Real c = H_(static_cast<Index>(j) + 1, static_cast<Index>(j));
            const Real den = std::hypot(a, c);
            if (den == 0) {
                res.breakdown = true;
                done = true;
                break;
            }
            cs_[j] = a / den;
            sn_[j] = c / den;
            H_(static_cast<Index>(j), static_cast<Index>(j)) = den;
            H_(static_cast<Index>(j) + 1, static_cast<Index>(j)) = 0;
            g_[j + 1] = -sn_[j] * g_[j];
            g_[j] = cs_[j] * g_[j];
            if (hnext > 0) {
                auto vn = V_.row(static_cast<Index>(j) + 1);
                for (Index i = 0; i < n; ++i) vn[i] = w_[i] / hnext;
            } else {
                res.breakdown = true;  // invariant subspace found: solution exact in this basis
                ++j;
                ++total;
                done = true;
                break;
            }
            if (std::abs(g_[j + 1]) <= target) {
                ++j;
                ++total;
                done = true;
                break;
            }
        }
        // Solve the j x j upper-triangular system H y = g.
        for (int i = j - 1; i >= 0; --i) {
            Real s = g_[i];
            for (int k = i + 1; k < j; ++k) s -= H_(static_cast<Index>(i), static_cast<Index>(k)) * yv_[k];
            yv_[i] = s / H_(static_cast<Index>(i), static_cast<Index>(i));
        }
        for (int i = 0; i < j; ++i) {
            auto basis = M ? Z_.row(static_cast<Index>(i)) : V_.row(static_cast<Index>(i));
            kernels::axpy(yv_[i], basis.data(), x.data(), n);
        }
        H_.fill(0);
        res.iterations = total;
        if (done && res.breakdown) {
            // Recompute the true residual once and stop.
            A(x, r_);
            for (Index i = 0; i < n; ++i) r_[i] = b[i] - r_[i];
            res.residual = nrm2(r_);
            res.rel_residual = res.residual / bnorm;
            res.converged = res.residual <= target * 10;  // allow round-off slack after breakdown
            return res;
        }
        if (total >= max_iter && !done) {
            A(x, r_);
            for (Index i = 0; i < n; ++i) r_[i] = b[i] - r_[i];
            res.residual = nrm2(r_);
            res.rel_residual = res.residual / bnorm;
            res.converged = res.residual <= target;
            return res;
        }
        // otherwise restart (or verify convergence on the next loop iteration)
    }
    res.iterations = total;
    return res;
}

}  // namespace dpd::ode
