#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <random>

#include "dpd/ode/kernels.hpp"

using namespace dpd::ode;
using Catch::Matchers::WithinRel;
namespace k = dpd::ode::kernels;

namespace {
std::vector<Real> rnd(std::size_t n, std::mt19937_64& g, Real lo = -1, Real hi = 1) {
    std::uniform_real_distribution<Real> d(lo, hi);
    std::vector<Real> v(n);
    for (auto& x : v) x = d(g);
    return v;
}
}  // namespace

TEST_CASE("every available backend agrees with the scalar reference", "[kernels]") {
    std::mt19937_64 g(42);
    const auto* ref = k::table(k::Backend::Scalar);
    REQUIRE(ref != nullptr);
    for (auto b : k::available_backends()) {
        const auto* t = k::table(b);
        REQUIRE(t != nullptr);
        for (std::size_t n : {0u, 1u, 2u, 3u, 4u, 5u, 7u, 8u, 9u, 16u, 17u, 33u, 1000u}) {
            CAPTURE(k::to_string(b), n);
            auto x = rnd(n, g), y = rnd(n, g), e = rnd(n, g), a = rnd(n, g), bb = rnd(n, g);
            auto atol = rnd(n, g, 1e-8, 1e-6), s = rnd(n, g, 0.5, 2);
            // axpy: no FMA in any backend -> bitwise identical
            auto y1 = y, y2 = y;
            ref->axpy(0.37, x.data(), y1.data(), n);
            t->axpy(0.37, x.data(), y2.data(), n);
            CHECK(y1 == y2);
            // lincomb: SIMD backends use FMA -> agree to a few ulps
            std::vector<std::vector<Real>> K;
            for (int m = 0; m < 6; ++m) K.push_back(rnd(n, g));
            const Real* Kp[6] = {K[0].data(), K[1].data(), K[2].data(), K[3].data(), K[4].data(), K[5].data()};
            const Real c[6] = {0.1, -0.7, 2.5, 1e-3, -3.0, 0.25};
            std::vector<Real> o1(n), o2(n);
            ref->lincomb(o1.data(), y.data(), c, Kp, 6, n);
            t->lincomb(o2.data(), y.data(), c, Kp, 6, n);
            for (std::size_t i = 0; i < n; ++i) CHECK(std::abs(o1[i] - o2[i]) <= 1e-14 * (1 + std::abs(o1[i])));
            // in-place lincomb (out aliases y)
            auto yy = y;
            t->lincomb(yy.data(), yy.data(), c, Kp, 6, n);
            for (std::size_t i = 0; i < n; ++i) CHECK(std::abs(o1[i] - yy[i]) <= 1e-14 * (1 + std::abs(o1[i])));
            // reductions: different summation order
            for (bool vec : {true, false}) {
                const Real r1 = ref->wrms_err(e.data(), a.data(), bb.data(), atol.data(), vec, 1e-3, n);
                const Real r2 = t->wrms_err(e.data(), a.data(), bb.data(), atol.data(), vec, 1e-3, n);
                CHECK_THAT(r2, WithinRel(r1, 1e-13));
            }
            CHECK_THAT(t->wrms(x.data(), s.data(), n), WithinRel(ref->wrms(x.data(), s.data(), n), 1e-13));
            const Real d1 = ref->dot(x.data(), y.data(), n), d2 = t->dot(x.data(), y.data(), n);
            Real l1 = 0;
            for (std::size_t i = 0; i < n; ++i) l1 += std::abs(x[i] * y[i]);
            CHECK(std::abs(d1 - d2) <= 1e-14 * (1 + l1));
        }
    }
}

TEST_CASE("wrms_err implements the documented scaled norm", "[kernels]") {
    const std::vector<Real> e{1e-6, -2e-6}, a{1.0, -3.0}, b{2.0, 1.0}, atol{1e-6, 1e-7};
    const Real rtol = 1e-4;
    const Real s0 = 1e-6 + 1e-4 * 2.0, s1 = 1e-7 + 1e-4 * 3.0;
    const Real expect = std::sqrt(((1e-6 / s0) * (1e-6 / s0) + (2e-6 / s1) * (2e-6 / s1)) / 2);
    for (auto bk : k::available_backends()) {
        CAPTURE(k::to_string(bk));
        CHECK_THAT(k::table(bk)->wrms_err(e.data(), a.data(), b.data(), atol.data(), true, rtol, 2),
                   WithinRel(expect, 1e-14));
    }
}

TEST_CASE("backend selection", "[kernels]") {
    const auto prev = k::active_backend();
    CHECK(k::set_backend(k::Backend::Scalar));
    CHECK(k::active_backend() == k::Backend::Scalar);
    CHECK(k::set_backend(k::Backend::Generic));
    for (auto b : {k::Backend::Neon, k::Backend::Avx2}) {
        const bool avail = k::table(b) != nullptr;
        CHECK(k::set_backend(b) == avail);
    }
#if defined(__aarch64__) && !defined(DPD_DISABLE_NEON)
    CHECK(k::table(k::Backend::Neon) != nullptr);
    CHECK(k::table(k::Backend::Avx2) == nullptr);
#endif
    k::set_backend(prev);
}
