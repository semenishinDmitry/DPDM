// Tests of the physical model against the paper's formulas and independent references.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "dpd/dust/constants.hpp"
#include "dpd/dust/run.hpp"
#include "dpd/ode/test_problems.hpp"
#include "reference_data.hpp"

using namespace dpd;
using namespace dpd::dust;
using Catch::Matchers::WithinRel;

namespace {
RunConfig base_config() {
    RunConfig c;
    c.disk = DiskParams{};
    c.grain = GrainParams{};
    return c;
}
}  // namespace

TEST_CASE("disk: analytic profiles reproduce Eqs. (12)-(13) and the u_r, H laws", "[dust][disk]") {
    Disk d(DiskParams{});
    CHECK_THAT(d.temperature(1), WithinRel(240.0, 1e-14));
    CHECK_THAT(d.rho_mid(1), WithinRel(2.5e-10, 1e-14));
    CHECK_THAT(d.ur_cgs(1), WithinRel(-30.0, 1e-14));
    CHECK_THAT(d.scale_height(1), WithinRel(0.03, 1e-14));
    CHECK_THAT(d.temperature(10), WithinRel(240.0 * std::pow(10.0, -9.0 / 8.0), 1e-14));
    CHECK_THAT(d.rho_mid(10), WithinRel(2.5e-10 * std::pow(10.0, -21.0 / 16.0), 1e-14));
    CHECK_THAT(d.rho(10, d.scale_height(10)), WithinRel(d.rho_mid(10) * std::exp(-0.5), 1e-14));  // Eq. (19)
    CHECK_THAT(d.beta(5), WithinRel(1 - 0.9987 * 0.9987, 1e-14));
    // parameter scaling: alpha_0.01^(-1/4) Mdot^(1/2) M^(3/8)
    DiskParams p;
    p.alpha = 0.04;
    p.mdot = 4;
    p.mstar = 2;
    Disk d2(p);
    CHECK_THAT(d2.temperature(1), WithinRel(240.0 * std::pow(4.0, -0.25) * 2.0 * std::pow(2.0, 3.0 / 8.0), 1e-14));
    // local() equals the individual functions
    const auto L = d.local(7.3, 0.11);
    CHECK_THAT(L.T, WithinRel(d.temperature(7.3), 1e-13));
    CHECK_THAT(L.rho, WithinRel(d.rho(7.3, 0.11), 1e-13));
    CHECK_THAT(L.H, WithinRel(d.scale_height(7.3), 1e-13));
    CHECK_THAT(L.ur, WithinRel(d.ur_cgs(7.3), 1e-13));
}

TEST_CASE("disk: H from the power law is consistent with H = c_T / Omega_K", "[dust][disk]") {
    Disk d(DiskParams{});
    for (double r : {1.0, 10.0, 100.0}) {
        const double cT = std::sqrt(cgs::k_B * d.temperature(r) / (2.3 * cgs::m_p));
        const double Om = std::sqrt(cgs::GM_sun / std::pow(r * cgs::au, 3));
        CHECK_THAT(cT / Om / cgs::au, WithinRel(d.scale_height(r), 0.05));  // power law is a rounded fit
    }
}

TEST_CASE("stopping time reproduces Eq. (3) and Eq. (20)", "[dust][drag]") {
    GrainParams g;
    g.drag = DragLaw::Epstein;
    g.a_init_cm = 1e-4;
    DustModel m(DiskParams{}, g);
    const auto d = m.drag(1.0, 0.0, 1e-4);
    CHECK_THAT(d.tau_ep, WithinRel(8.1, 0.01));                 // Eq. (3): ~8.1 s
    CHECK_THAT(d.tau_ep / m.units().t0_s, WithinRel(1.6e-6, 0.02));  // Eq. (20)
    // linear in a (Epstein), quadratic (Stokes)
    CHECK_THAT(m.drag(1, 0, 2e-4).tau_ep, WithinRel(2 * d.tau_ep, 1e-14));
    CHECK_THAT(m.drag(1, 0, 2e-4).tau_st, WithinRel(4 * d.tau_st, 1e-14));
    // tau_St / tau_Ep = 2 a / (3 lambda): equal at a = 1.5 lambda (Sec. 1.4)
    CHECK_THAT(d.tau_st / d.tau_ep, WithinRel(2 * 1e-4 / (3 * d.lambda), 1e-12));
}

TEST_CASE("drag law switch is continuous at a / lambda = 3/2", "[dust][drag]") {
    GrainParams g;
    g.drag = DragLaw::Both;
    DustModel m(DiskParams{}, g);
    const double lam = m.drag(1, 0, 1).lambda;  // independent of a
    const auto below = m.drag(1, 0, 1.5 * lam * (1 - 1e-9));
    const auto above = m.drag(1, 0, 1.5 * lam * (1 + 1e-9));
    CHECK_FALSE(below.stokes);
    CHECK(above.stokes);
    CHECK_THAT(above.tau, WithinRel(below.tau, 1e-8));
    // Calibrated sigma: lambda(1 au, midplane) ~ 1.5-1.7 m (docs/physics_model.md A3)
    CHECK(lam > 140);
    CHECK(lam < 180);
}

TEST_CASE("code units: t0 = au / sqrt(G Msun / au) ~= 1/(2 pi) yr", "[dust][units]") {
    const auto u = make_units(1.0);
    CHECK_THAT(u.v0_cms, WithinRel(2.976e6, 1e-3));  // paper value 2.976e6 cm/s
    CHECK_THAT(u.t0_yr(), WithinRel(1 / (2 * cgs::pi), 2e-4));
}

TEST_CASE("RHS at the initial state matches hand-evaluated Eqs. (11)", "[dust][rhs]") {
    DiskParams dp;
    GrainParams g;
    g.a_init_cm = 1.0;
    g.coagulation = true;
    g.drag = DragLaw::Epstein;
    DustModel m(dp, g);
    const double r = 10, z = 0.4;
    const auto y = m.initial_state(r, z, 0, 0);
    std::vector<double> f(6);
    m.rhs(0, y, f);
    // independent evaluation from the paper's formulas (CGS -> code units)
    const double T = 240 * std::pow(r, -9.0 / 8.0);
    const double H = 0.03 * std::pow(r, 15.0 / 16.0);
    const double rho = 2.5e-10 * std::pow(r, -21.0 / 16.0) * std::exp(-z * z / (2 * H * H));
    const double vth = std::sqrt(8 * cgs::k_B * T / (cgs::pi * 2.3 * cgs::m_p));
    const auto u = make_units(1.0);
    const double tau = 3.0 * 1.0 / (rho * vth) / u.t0_s;
    const double ur = -30 * std::pow(r, -5.0 / 8.0) / u.v0_cms;
    const double vk = 1 / std::sqrt(r), uphi = vk * 0.9987;
    const double j = r * vk;
    CHECK(f[0] == 0.0);
    CHECK_THAT(f[1], WithinRel(j * j / (r * r * r) - 1 / (r * r) + ur / tau, 1e-12));
    CHECK_THAT(f[2], WithinRel(-r / tau * (j / r - uphi), 1e-12));
    CHECK(f[3] == 0.0);
    CHECK_THAT(f[4], WithinRel(-z / (r * r * r), 1e-12));
    const double da = 0.01 * rho * cgs::au / (4 * 3.0 * 1.0) * std::sqrt(ur * ur + (j / r - uphi) * (j / r - uphi));
    CHECK_THAT(f[5], WithinRel(da, 1e-12));
}

TEST_CASE("model switches: sedimentation and coagulation off", "[dust][rhs]") {
    GrainParams g;
    g.sedimentation = false;
    g.coagulation = false;
    DustModel m(DiskParams{}, g);
    auto y = m.initial_state(10, 0.5, 0, 0);
    CHECK(y[3] == 0.0);
    std::vector<double> f(6);
    m.rhs(0, y, f);
    CHECK(f[3] == 0.0);
    CHECK(f[4] == 0.0);
    CHECK(f[5] == 0.0);
}

TEST_CASE("stationary radial drift of a tightly coupled grain matches the analytic law", "[dust][drift]") {
    // Correct stationary solution (Takeuchi & Lin 2002; paper Eq. (6)):
    //   v_r = (u_r / St - beta v_K) / (St + 1 / St),  St = tau_s Omega_K.
    // (Paper Eq. (26) writes tau_s instead of St; identical at r = 1 au, see docs.)
    for (double a : {0.1, 1.0, 10.0}) {
        RunConfig c = base_config();
        c.grain.a_init_cm = a;
        c.grain.sedimentation = false;
        c.grain.drag = DragLaw::Epstein;
        c.r_init_au = 5.0;
        c.r_in_au = 4.0;
        c.solver.rtol = 1e-10;
        const auto res = run(c);
        REQUIRE(res.summary.reached_inner_edge);
        DustModel m(c.disk, c.grain);
        // check rows after the initial transient
        int checked = 0;
        for (std::size_t i = 0; i < res.diag.size(); ++i) {
            const auto& D = res.diag[i];
            if (D.r_au > 4.8 || D.r_au < 4.1) continue;
            const double St = D.St;
            const double vk = std::sqrt(cgs::GM_sun / (D.r_au * cgs::au));
            const double vr = (D.ur_cms / St - D.beta * vk) / (St + 1 / St);
            CAPTURE(a, D.r_au, St);
            CHECK_THAT(D.vr_cms, WithinRel(vr, 2e-3));
            ++checked;
        }
        CHECK(checked > 3);
    }
}

TEST_CASE("no headwind (beta = 0): micron grains drift with the gas (analytic time)", "[dust][drift]") {
    RunConfig c = base_config();
    c.disk.vphi2vk = 1.0;
    c.grain.a_init_cm = 1e-4;
    c.grain.sedimentation = false;
    c.r_init_au = 10;
    c.r_in_au = 0.05;
    const auto res = run(c, {}, false);
    REQUIRE(res.summary.reached_inner_edge);
    // t = int dr / |u_r| = (r0 / 30 cm/s) (r_i^(13/8) - r_in^(13/8)) / (13/8)
    const double t = cgs::au / 30.0 * (std::pow(10.0, 1.625) - std::pow(0.05, 1.625)) / 1.625 / cgs::year;
    CHECK_THAT(res.summary.t_drift_yr, WithinRel(t, 1e-4));
}

TEST_CASE("coagulation: radius grows monotonically; Y_d = 0 disables growth", "[dust][coag]") {
    RunConfig c = base_config();
    c.grain.a_init_cm = 0.1;
    c.grain.coagulation = true;
    c.grain.sedimentation = false;
    c.r_init_au = 10;
    auto res = run(c);
    REQUIRE(res.summary.reached_inner_edge);
    for (std::size_t i = 1; i < res.diag.size(); ++i) CHECK(res.diag[i].a_cm >= res.diag[i - 1].a_cm);
    CHECK(res.summary.a_final_cm > 1.0);
    c.grain.dust_to_gas = 0;
    res = run(c, {}, false);
    CHECK(res.summary.a_final_cm == 0.1);
}

TEST_CASE("physical constraints along a 2D trajectory", "[dust][constraints]") {
    RunConfig c = base_config();
    c.grain.a_init_cm = 100;
    c.r_init_au = 10;
    const auto res = run(c);
    REQUIRE(res.summary.reached_inner_edge);
    const double z0 = 3 * Disk(c.disk).scale_height(10);
    for (const auto& D : res.diag) {
        CHECK(D.r_au > 0);
        CHECK(std::abs(D.z_au) <= z0 * (1 + 1e-9));  // damped oscillation never exceeds z_init
        CHECK(D.a_cm == 100.0);
        CHECK(D.vphi_cms > 0);
    }
}

TEST_CASE("tabulated beta: constant table reproduces the analytic disk; extrapolation flagged", "[dust][disk]") {
    RunConfig c = base_config();
    c.grain.a_init_cm = 100;
    c.grain.sedimentation = false;
    c.r_init_au = 10;
    const auto ra = run(c, {}, false);
    const double b = 1 - 0.9987 * 0.9987;
    c.disk.beta_r_au = {1.0, 10.0, 100.0};
    c.disk.beta_values = {b, b, b};
    const auto rt = run(c, {}, false);
    CHECK_THAT(rt.summary.t_drift_yr, WithinRel(ra.summary.t_drift_yr, 1e-12));
    CHECK(rt.summary.beta_extrapolated);  // r < 1 au is outside the table
    CHECK_FALSE(ra.summary.beta_extrapolated);
}

TEST_CASE("cross-check against the independent SciPy implementation", "[dust][crosscheck]") {
    for (const auto& ref : refdata::kDust) {
        RunConfig c = base_config();
        c.grain.a_init_cm = ref.a_cm;
        c.r_init_au = ref.r_init_au;
        c.grain.sedimentation = ref.sed;
        c.grain.coagulation = ref.coag;
        c.grain.drag = drag_from_string(ref.drag);
        c.solver.rtol = 1e-10;
        c.solver.atol = {1e-13, 1e-17, 1e-13, 1e-13, 1e-17, 1e-13};
        const auto res = run(c, {}, false);
        CAPTURE(ref.a_cm, ref.r_init_au, ref.sed, ref.coag, ref.drag);
        REQUIRE(res.summary.reached_inner_edge);
        CHECK_THAT(res.summary.t_drift_yr, WithinRel(ref.t_drift_yr, 2e-6));
        CHECK_THAT(res.summary.a_final_cm / ref.a_cm, WithinRel(ref.a_final_rel, 2e-6));
    }
}

TEST_CASE("solver methods agree on the dust model (auto, BDF, DOPRI5)", "[dust][crossmethod]") {
    RunConfig c = base_config();
    c.grain.a_init_cm = 10;
    c.r_init_au = 1;
    c.solver.rtol = 1e-10;
    double t[3];
    int k = 0;
    for (auto m : {ode::Method::Auto, ode::Method::Bdf, ode::Method::Dopri5}) {
        c.solver.method = m;
        const auto res = run(c, {}, false);
        REQUIRE(res.summary.reached_inner_edge);
        t[k++] = res.summary.t_drift_yr;
    }
    CHECK_THAT(t[1], WithinRel(t[0], 1e-6));
    CHECK_THAT(t[2], WithinRel(t[0], 1e-6));
}

TEST_CASE("ensemble runs equal serial runs", "[dust][ensemble]") {
    std::vector<RunConfig> cfgs;
    for (double a : {1e-4, 0.1, 1.0, 10.0, 100.0}) {
        RunConfig c = base_config();
        c.grain.a_init_cm = a;
        c.r_init_au = 10;
        cfgs.push_back(c);
    }
    const auto par = run_ensemble(cfgs, 4, false);
    for (std::size_t i = 0; i < cfgs.size(); ++i) {
        const auto ser = run(cfgs[i], {}, false);
        CHECK(par[i].summary.t_drift_yr == ser.summary.t_drift_yr);  // bitwise: same code path
        CHECK(par[i].summary.stats.n_rhs == ser.summary.stats.n_rhs);
    }
}

TEST_CASE("analytic sedimentation solutions of Sec. 3.2.1 (constant tau_s)", "[dust][sedimentation]") {
    // z'' + z'/tau + z/r^3 = 0, z(0) = 1, z'(0) = 0, with time in units of tau_s:
    // q = 4 tau^2 / r^3; closed forms from the paper (p. 39-40).
    const double r = 1.0;
    for (double q : {0.01, 0.5, 1.0, 5.0, 100.0}) {
        CAPTURE(q);
        const double tau = std::sqrt(q * r * r * r) / 2;
        ode::DenseMatrix A(2, 2);
        A(0, 1) = 1;
        A(1, 0) = -1 / (r * r * r);
        A(1, 1) = -1 / tau;
        ode::problems::LinearSystem sys(A);
        ode::Options o;
        o.rtol = 1e-10;
        o.atol = {1e-13};
        o.save_steps = false;
        for (double s : {0.3, 1.0, 3.0, 10.0}) o.t_eval.push_back(s * tau);
        const std::vector<double> y0{1, 0};
        const auto res = ode::integrate(sys, 0, y0, 10.0 * tau, o);
        REQUIRE(res.ok());
        for (ode::Index k = 0; k < res.size(); ++k) {
            const double t = res.t[k] / tau;  // paper's t is in units of tau_s
            double z;
            if (q < 1) {
                const double s = std::sqrt(1 - q);
                z = ((1 + s) * std::exp(-t * (1 - s) / 2) - (1 - s) * std::exp(-t * (1 + s) / 2)) / (2 * s);
            } else if (q == 1) {
                z = std::exp(-t / 2) * (1 + t / 2);
            } else {
                const double s = std::sqrt(q - 1);
                z = std::exp(-t / 2) * (std::cos(t * s / 2) + std::sin(t * s / 2) / s);
            }
            CAPTURE(t);
            CHECK(std::abs(res.y_at(k)[0] - z) < 1e-8);
        }
    }
}
