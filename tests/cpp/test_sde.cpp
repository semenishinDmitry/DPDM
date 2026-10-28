// Stochastic integrators and the vertical-diffusion dust model.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "dpd/dust/diffusion.hpp"
#include "dpd/ode/sde.hpp"

using namespace dpd;
using namespace dpd::ode;
using Catch::Matchers::WithinRel;

namespace {
// dX = -theta X dt + sigma dW
class OU final : public SdeSystem {
public:
    OU(Real th, Real s) : th_(th), s_(s) {}
    [[nodiscard]] Index size() const override { return 1; }
    void drift(Real, std::span<const Real> y, std::span<Real> f) const override { f[0] = -th_ * y[0]; }
    void diffusion(Real, std::span<const Real>, std::span<Real> g) const override { g[0] = s_; }

private:
    Real th_, s_;
};
// dX = mu X dt + sigma X dW
class GBM final : public SdeSystem {
public:
    GBM(Real mu, Real s) : mu_(mu), s_(s) {}
    [[nodiscard]] Index size() const override { return 1; }
    void drift(Real, std::span<const Real> y, std::span<Real> f) const override { f[0] = mu_ * y[0]; }
    void diffusion(Real, std::span<const Real> y, std::span<Real> g) const override { g[0] = s_ * y[0]; }

private:
    Real mu_, s_;
};

void moments(const SdeResult& r, std::size_t k, Real& mean, Real& var) {
    Real s = 0, s2 = 0;
    for (std::size_t p = 0; p < r.n_paths; ++p) {
        const Real x = r.at(p, k, 0);
        s += x;
        s2 += x * x;
    }
    const Real n = static_cast<Real>(r.n_paths);
    mean = s / n;
    var = s2 / n - mean * mean;
}
}  // namespace

TEST_CASE("RNG: standard normal moments and stream independence", "[sde][rng]") {
    Rng a(42, 0), b(42, 1), c(42, 0);
    const int n = 400000;
    Real s = 0, s2 = 0, s4 = 0, cross = 0;
    bool same = true;
    for (int i = 0; i < n; ++i) {
        const Real x = a.normal(), y = b.normal();
        if (x != c.normal()) same = false;
        s += x;
        s2 += x * x;
        s4 += x * x * x * x;
        cross += x * y;
    }
    CHECK(same);  // reproducible
    CHECK(std::abs(s / n) < 5 / std::sqrt(Real(n)));
    CHECK(std::abs(s2 / n - 1) < 5 * std::sqrt(2.0 / n));
    CHECK(std::abs(s4 / n - 3) < 5 * std::sqrt(96.0 / n));
    CHECK(std::abs(cross / n) < 5 / std::sqrt(Real(n)));
}

TEST_CASE("Ornstein-Uhlenbeck: stationary variances of the discrete schemes are exact", "[sde][ou]") {
    // For dt = h, x = theta h:
    //   Euler-Maruyama         var = sigma^2 / (theta (2 - x))
    //   linear-implicit Euler  var = sigma^2 / (theta (2 + x))
    const Real th = 2.0, sg = 0.5, h = 0.1;
    const OU sys(th, sg);
    for (auto sch : {SdeScheme::EulerMaruyama, SdeScheme::LinearImplicitEuler}) {
        SdeOptions o;
        o.scheme = sch;
        o.dt = h;
        o.n_paths = 200000;
        o.t_out = {20.0};  // 40 relaxation times
        const std::vector<Real> y0{1.0};
        const auto r = simulate(sys, 0, y0, o);
        REQUIRE(r.status == Status::Success);
        Real m, v;
        moments(r, 0, m, v);
        const Real x = th * h;
        const Real expect = sg * sg / (th * (sch == SdeScheme::EulerMaruyama ? 2 - x : 2 + x));
        CAPTURE(to_string(sch), v, expect);
        CHECK(std::abs(m) < 5 * std::sqrt(expect / static_cast<Real>(o.n_paths)));
        CHECK(std::abs(v - expect) < 5 * expect * std::sqrt(2.0 / static_cast<Real>(o.n_paths)));
    }
}

TEST_CASE("Ornstein-Uhlenbeck: transient mean and variance converge with dt", "[sde][ou]") {
    const Real th = 1.0, sg = 1.0, T = 0.7;
    const OU sys(th, sg);
    const Real m_ex = 2.0 * std::exp(-th * T), v_ex = sg * sg / (2 * th) * (1 - std::exp(-2 * th * T));
    Real err_prev = 1e9;
    for (Real h : {0.1, 0.025}) {
        SdeOptions o;
        o.dt = h;
        o.n_paths = 400000;
        o.t_out = {T};
        const std::vector<Real> y0{2.0};
        const auto r = simulate(sys, 0, y0, o);
        Real m, v;
        moments(r, 0, m, v);
        const Real err = std::abs(m - m_ex) + std::abs(v - v_ex);
        CAPTURE(h, m, m_ex, v, v_ex);
        CHECK(err < err_prev);  // weak error decreases with dt
        err_prev = err;
    }
    CHECK(err_prev < 0.02);
}

TEST_CASE("GBM: strong orders 1/2 (Euler-Maruyama) and 1 (Milstein)", "[sde][strong]") {
    const Real mu = 0.5, sg = 0.8, T = 1.0;
    const GBM sys(mu, sg);
    for (auto sch : {SdeScheme::EulerMaruyama, SdeScheme::Milstein}) {
        std::vector<Real> errs;
        for (Real h : {1.0 / 64, 1.0 / 256, 1.0 / 1024}) {
            Real e = 0;
            const int np = 4000;
            for (int p = 0; p < np; ++p) {
                Rng rng(7, static_cast<std::uint64_t>(p));
                std::vector<Real> y{1.0}, W{0.0};
                REQUIRE(simulate_path(sys, sch, 0, y, T, h, rng, W));
                const Real exact = std::exp((mu - 0.5 * sg * sg) * T + sg * W[0]);
                e += std::abs(y[0] - exact);
            }
            errs.push_back(e / np);
        }
        const Real order = std::log(errs[0] / errs[2]) / std::log(16.0);
        CAPTURE(to_string(sch), order, errs[0], errs[2]);
        if (sch == SdeScheme::EulerMaruyama) {
            CHECK(order > 0.4);
            CHECK(order < 0.65);
        } else {
            CHECK(order > 0.9);
            CHECK(order < 1.1);
        }
    }
}

TEST_CASE("SDE ensembles are bitwise independent of the thread count", "[sde]") {
    const OU sys(1.0, 0.3);
    SdeOptions o;
    o.dt = 0.01;
    o.n_paths = 3000;
    o.t_out = {0.5, 1.0};
    const std::vector<Real> y0{1.0};
    o.threads = 1;
    const auto a = simulate(sys, 0, y0, o);
    o.threads = 8;
    const auto b = simulate(sys, 0, y0, o);
    CHECK(a.y == b.y);
    o.seed = 2;
    const auto c = simulate(sys, 0, y0, o);
    CHECK(a.y != c.y);
}

TEST_CASE("vertical dust diffusion reaches the exact zero-flux stationary state", "[sde][dust]") {
    using namespace dpd::dust;
    for (double a : {1e-2, 1.0}) {
        for (double al : {1e-4, 1e-3}) {
            DiffusionRunConfig c;
            c.r_au = 100;
            c.a_cm = a;
            c.alpha_turb = al;
            c.n_paths = 20000;
            const auto r = run_vertical_diffusion(DiskParams{}, GrainParams{}, c);
            REQUIRE(r.status == Status::Success);
            // statistics (4 sigma) + first-order time-step bias (~ dt_relax / 2 = 1 %)
            CAPTURE(a, al, r.Hd_simulated_au, r.Hd_exact_au, r.Hd_stat_error_au);
            CHECK(std::abs(r.Hd_simulated_au - r.Hd_exact_au) < 4 * r.Hd_stat_error_au + 0.01 * r.Hd_exact_au);
        }
    }
}

TEST_CASE("stationary solution: Dubrulle limit for constant St, deviation when St grows with z", "[sde][dust]") {
    using namespace dpd::dust;
    // Stokes drag: tau independent of gas density -> St constant -> exactly Gaussian.
    GrainParams g;
    g.drag = DragLaw::Stokes;
    g.a_init_cm = 1.0;
    const DustModel ms(DiskParams{}, g);
    const VerticalDiffusion vs(ms, 50.0, 1.0, 1e-3);
    CHECK_THAT(vs.stationary_Hd(), WithinRel(vs.dubrulle_Hd(), 1e-6));
    // Epstein with St ~ alpha: dust above the midplane settles faster -> thinner layer
    g.drag = DragLaw::Epstein;
    g.a_init_cm = 0.01;
    const DustModel me(DiskParams{}, g);
    const VerticalDiffusion ve(me, 100.0, 0.01, 1e-3);
    CHECK(ve.stationary_Hd() < 0.9 * ve.dubrulle_Hd());
    // and the density integrates to one
    std::vector<double> z;
    for (int i = -3000; i <= 3000; ++i) z.push_back(i * 6 * ve.H() / 3000);
    const auto p = ve.stationary_density(z);
    double s = 0;
    for (std::size_t i = 1; i < z.size(); ++i) s += 0.5 * (p[i] + p[i - 1]) * (z[i] - z[i - 1]);
    CHECK_THAT(s, WithinRel(1.0, 1e-12));
}
