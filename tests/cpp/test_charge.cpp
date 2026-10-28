// Grain charging (OML) and the Lorentz force of the fossil field.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "dpd/dust/constants.hpp"
#include "dpd/dust/run.hpp"

using namespace dpd;
using namespace dpd::dust;
using Catch::Matchers::WithinRel;

TEST_CASE("OML surface potential solves the current balance", "[charge]") {
    for (double mi : {1.0, 24.0, 29.0}) {
        // independent bisection of exp(psi) sqrt(m_i/m_e) = 1 - psi
        const double s = std::sqrt(mi * cgs::amu / cgs::m_e);
        double a = -30, b = 0;
        for (int i = 0; i < 200; ++i) {
            const double c = 0.5 * (a + b);
            (std::exp(c) * s - (1 - c) > 0 ? b : a) = c;
        }
        CHECK_THAT(oml_potential(mi), WithinRel(0.5 * (a + b), 1e-12));
    }
    CHECK(oml_potential(29.0) < -3.5);
    CHECK(oml_potential(29.0) > -4.5);
}

namespace {
DiskParams magnetized(double B0 = 1.0, double q = 0.5) {
    DiskParams d;
    d.model = DiskModel::SelfConsistent;
    d.perturbation = Perturbation::Gaussian;
    d.bump_amplitude = 0.5;
    d.bump_r_au = 5.0;
    d.bump_width_au = 1.0;
    d.field.B0_G = B0;
    d.field.exponent = 1.25;
    d.field.br_over_bz = q;
    return d;
}
}  // namespace

TEST_CASE("Lorentz force does no work in the gas frame", "[charge][lorentz]") {
    GrainParams g;
    g.a_init_cm = 1e-5;
    g.charge = ChargeModel::OML;
    const DustModel mc(magnetized(), g);
    GrainParams g0 = g;
    g0.charge = ChargeModel::None;
    const DustModel m0(magnetized(), g0);
    REQUIRE(mc.lorentz_active());
    auto y = mc.initial_state(3.0, 0.2, 0, 0);
    y[DustModel::VR] = 3e-4;
    y[DustModel::J] *= 1.002;
    y[DustModel::VZ] = -1e-4;
    std::vector<double> fc(6), f0(6);
    mc.rhs(0, y, fc);
    m0.rhs(0, y, f0);
    const double r = y[0];
    const double ur = mc.disk().ur_cgs(r) / mc.units().v0_cms;
    const double wr = y[1] - ur, wp = y[2] / r - mc.u_phi(r), wz = y[4];
    const double ar = fc[1] - f0[1], ap = (fc[2] - f0[2]) / r, az = fc[4] - f0[4];
    const double amag = std::sqrt(ar * ar + ap * ap + az * az), wmag = std::sqrt(wr * wr + wp * wp + wz * wz);
    REQUIRE(amag > 0);
    CHECK(std::abs(ar * wr + ap * wp + az * wz) <= 1e-9 * amag * wmag);
}

TEST_CASE("charge number and Hall parameter diagnostics", "[charge]") {
    RunConfig c;
    c.disk = magnetized(0.3, 0.0);
    c.grain.a_init_cm = 1e-4;
    c.grain.charge = ChargeModel::OML;
    c.r_init_au = 2.0;
    c.r_in_au = 1.9;
    const auto res = run(c);
    REQUIRE(!res.diag.empty());
    const auto& D = res.diag.front();
    const double Z = oml_potential(29.0) * 1e-4 * cgs::k_B * D.T / (cgs::e_esu * cgs::e_esu);
    CHECK_THAT(D.Z_charge, WithinRel(Z, 1e-12));
    const double m = 4.0 / 3.0 * cgs::pi * 1e-12 * 3.0;
    const double B = 0.3 * std::pow(D.r_au, -1.25);
    CHECK_THAT(D.hall, WithinRel(std::abs(Z) * cgs::e_esu * B / (m * cgs::c_light) * D.tau_s, 1e-10));
}

TEST_CASE("analytic Jacobian with Lorentz force matches finite differences", "[charge][jacobian]") {
    for (bool sed : {true, false}) {
        GrainParams g;
        g.a_init_cm = 3e-5;
        g.charge = ChargeModel::OML;
        g.sedimentation = sed;
        g.coagulation = true;
        const DustModel m(magnetized(2.0, 0.7), g);
        auto y = m.initial_state(4.5, sed ? 0.1 : 0.0, 0, 0);
        y[DustModel::VR] = 2e-4;
        y[DustModel::J] *= 1.003;
        if (sed) y[DustModel::VZ] = -1e-4;
        y[DustModel::A] = 1.2;
        ode::DenseMatrix J(6, 6);
        m.jacobian(0, y, J);
        std::vector<double> fp(6), fm(6);
        for (int k = 0; k < 6; ++k) {
            if (!sed && (k == DustModel::Z || k == DustModel::VZ)) continue;
            const double h = 1e-6 * std::max(std::abs(y[k]), 1e-3);
            auto yp = y, ym = y;
            yp[k] += h;
            ym[k] -= h;
            m.rhs(0, yp, fp);
            m.rhs(0, ym, fm);
            for (int i = 0; i < 6; ++i) {
                const double fd = (fp[i] - fm[i]) / (2 * h);
                CAPTURE(sed, i, k, fd, J(i, k));
                CHECK(std::abs(J(i, k) - fd) <= 1e-5 * std::abs(fd) + 1e-7 * (1 + std::abs(J(i, i))));
            }
        }
    }
}

TEST_CASE("Lorentz force vanishes without charge or field; small Hall parameter changes little", "[charge]") {
    RunConfig c;
    c.disk = magnetized(0.1, 0.3);
    c.grain.a_init_cm = 1e-4;
    c.grain.sedimentation = false;
    c.r_init_au = 3.0;
    c.r_in_au = 1.0;
    const auto r0 = run(c, {}, false);
    c.grain.charge = ChargeModel::OML;
    const auto r1 = run(c, {}, false);
    REQUIRE(r0.summary.reached_inner_edge);
    REQUIRE(r1.summary.reached_inner_edge);
    // midplane, micron grain, 0.1 G: Hall parameter ~1e-9 -> identical drift to 1e-6
    CHECK_THAT(r1.summary.t_drift_yr, WithinRel(r0.summary.t_drift_yr, 1e-6));
}
