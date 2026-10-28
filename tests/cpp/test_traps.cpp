// Self-consistent / tabulated disks and dust traps.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "dpd/dust/constants.hpp"
#include "dpd/dust/run.hpp"
#include "dpd/dust/traps.hpp"

using namespace dpd;
using namespace dpd::dust;
using Catch::Matchers::WithinRel;

namespace {
DiskParams dead_zone(double A = 2.0, double rb = 10.0, double w = 1.0) {
    DiskParams d;
    d.model = DiskModel::SelfConsistent;
    d.perturbation = Perturbation::DeadZoneEdge;
    d.bump_amplitude = A;
    d.bump_r_au = rb;
    d.bump_width_au = w;
    return d;
}
double cT2_over_vk2(const Disk& d, double r) {
    return cgs::k_B * d.temperature(r) / (2.3 * cgs::m_p) / (cgs::GM_sun / (r * cgs::au));
}
}  // namespace

TEST_CASE("self-consistent beta_K from Eq. (16) without perturbation or field", "[traps][disk]") {
    DiskParams p;
    p.model = DiskModel::SelfConsistent;
    const Disk d(p);
    for (double r : {0.3, 1.0, 10.0, 100.0}) {
        // -(r/rho) dp/dr / v_K^2 = (c_T/v_K)^2 (21/16 + 9/8) for rho ~ r^-21/16, T ~ r^-9/8
        CHECK_THAT(d.beta(r), WithinRel(cT2_over_vk2(d, r) * (21.0 / 16.0 + 9.0 / 8.0), 1e-12));
    }
    // consistent with the paper's constant 0.0026 to ~10 % at 1 au
    CHECK(d.beta(1.0) > 0.0021);
    CHECK(d.beta(1.0) < 0.0026);
}

TEST_CASE("magnetic tension adds r Bz Br / (4 pi Sigma v_K^2) to beta_K", "[traps][disk][field]") {
    DiskParams p;
    p.model = DiskModel::SelfConsistent;
    const Disk d0(p);
    p.field.B0_G = 0.5;
    p.field.exponent = 1.25;
    p.field.br_over_bz = 0.4;
    const Disk d1(p);
    for (double r : {1.0, 10.0}) {
        const double rcm = r * cgs::au;
        const double Bz = 0.5 * std::pow(r, -1.25), Br = 0.4 * Bz;
        const double sigma = d1.rho_mid(r) * d1.scale_height(r) * cgs::au;
        const double expect = rcm * Bz * Br / (4 * cgs::pi * sigma) / (cgs::GM_sun / rcm);
        CHECK_THAT(d1.beta(r) - d0.beta(r), WithinRel(expect, 1e-10));
        CHECK(d1.Bz(r) == Bz);
    }
    p.magnetic_tension = false;
    CHECK(Disk(p).beta(3.0) == d0.beta(3.0));
}

TEST_CASE("dead-zone inner edge: zero head wind at the pressure maximum", "[traps]") {
    const Disk d(dead_zone());
    const auto zh = zero_headwind_radii(d, 1, 100);
    REQUIRE(zh.size() == 1);
    // independent: maximum of the midplane pressure (golden-section search)
    double a = 9.0, b = 13.0;
    const double g = (std::sqrt(5.0) - 1) / 2;
    for (int i = 0; i < 200; ++i) {
        const double c = b - g * (b - a), e = a + g * (b - a);
        (d.pressure_mid(c) > d.pressure_mid(e) ? b : a) = (d.pressure_mid(c) > d.pressure_mid(e) ? e : c);
    }
    CHECK_THAT(zh[0], WithinRel(0.5 * (a + b), 1e-6));
}

TEST_CASE("trap radius from the stationary drift equals the ODE equilibrium", "[traps][ode]") {
    RunConfig c;
    c.disk = dead_zone();
    c.grain.a_init_cm = 1.0;
    c.grain.sedimentation = false;
    c.r_init_au = 15.0;
    c.t_end_yr = 3e6;
    const auto res = run(c, {}, false);
    REQUIRE(res.summary.status == ode::Status::Success);
    REQUIRE(res.summary.trapped);
    const DustModel m(c.disk, c.grain);
    const auto zs = drift_zeros(m, 1.0, 1, 100);
    double rt = NAN;
    for (const auto& z : zs)
        if (z.stable) rt = z.r_au;
    CHECK_THAT(res.summary.r_trap_au, WithinRel(rt, 1e-6));
    // large grains are trapped close to the zero-head-wind radius
    CHECK(std::abs(rt - zero_headwind_radii(m.disk(), 1, 100)[0]) < 0.2);
}

TEST_CASE("minimum trapped size separates trapped and passing grains", "[traps]") {
    GrainParams g;
    const DiskParams dp = dead_zone();
    const DustModel m(dp, g);
    const double amin = min_trapped_size_cm(m, 1, 100);
    REQUIRE(std::isfinite(amin));
    auto has_trap = [&](double a) {
        for (const auto& z : drift_zeros(m, a, 1, 100))
            if (z.stable) return true;
        return false;
    };
    CHECK(has_trap(amin * 1.01));
    CHECK_FALSE(has_trap(amin * 0.99));
    // a grain well below the threshold drifts through the trap to the inner edge (ODE)
    RunConfig c;
    c.disk = dp;
    c.grain.a_init_cm = amin / 10;
    c.grain.sedimentation = false;
    c.r_init_au = 15.0;
    const auto res = run(c, {}, false);
    CHECK(res.summary.reached_inner_edge);
    // ... and the trap disappears without the perturbation
    DiskParams flat;
    flat.model = DiskModel::SelfConsistent;
    CHECK(std::isnan(min_trapped_size_cm(DustModel(flat, g), 1, 100)));
}

TEST_CASE("tabulated profile reproduces the self-consistent disk (beta via Eq. 16)", "[traps][disk][table]") {
    DiskParams p = dead_zone();
    p.field.B0_G = 0.2;
    p.field.br_over_bz = 0.5;
    const Disk ref(p);
    ProfileTable t;
    const int n = 4000;
    for (int i = 0; i < n; ++i) {
        const double r = 0.5 * std::pow(400.0, i / double(n - 1));
        t.r_au.push_back(r);
        t.T_K.push_back(ref.temperature(r));
        t.rho_gcc.push_back(ref.rho_mid(r));
        t.ur_cms.push_back(ref.ur_cgs(r));
        t.H_au.push_back(ref.scale_height(r));
        t.Bz_G.push_back(ref.Bz(r));
        t.Br_G.push_back(ref.Br(r));
    }
    DiskParams q;
    q.model = DiskModel::Tabulated;
    q.table = t;
    const Disk tab(q);
    for (double r : {1.0, 5.0, 9.5, 10.3, 11.0, 30.0}) {
        CAPTURE(r);
        CHECK_THAT(tab.temperature(r), WithinRel(ref.temperature(r), 1e-6));
        CHECK_THAT(tab.rho_mid(r), WithinRel(ref.rho_mid(r), 1e-5));
        CHECK(std::abs(tab.beta(r) - ref.beta(r)) < 2e-3 * std::abs(ref.beta(r)) + 1e-6);
    }
    // trap radius agrees
    GrainParams g;
    const auto z1 = drift_zeros(DustModel(p, g), 1.0, 1, 100);
    const auto z2 = drift_zeros(DustModel(q, g), 1.0, 1, 100);
    REQUIRE(z1.size() == z2.size());
    for (std::size_t i = 0; i < z1.size(); ++i) CHECK_THAT(z2[i].r_au, WithinRel(z1[i].r_au, 1e-4));
}
