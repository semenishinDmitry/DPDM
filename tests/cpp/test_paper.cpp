// Regression against values published in Kobozeva (2025).
//
// Provenance: numbers are copied from the paper's Tables 1, 2, 4 and the text of
// Sec. 3.4 / 3.5.2. They are printed with 2-3 significant figures, i.e. a rounding
// uncertainty of up to ~2%; tolerances below are 3-5% for times (rounding + the two
// calibrated parameters sigma_mol and settle_eps, docs/physics_model.md A3/A4) and
// 10% for values read from the text describing figures.
// Cases the paper reports but this implementation does NOT reproduce are listed in
// docs/reference_case.md ("Known discrepancies") and are deliberately not asserted.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "dpd/dust/run.hpp"

using namespace dpd::dust;
using Catch::Matchers::WithinRel;

namespace {
RunConfig paper_config(double a_cm, double r_au, bool sed = true, bool coag = false) {
    RunConfig c;  // defaults are the paper's parameters (Sec. 2.6)
    c.grain.a_init_cm = a_cm;
    c.r_init_au = r_au;
    c.grain.sedimentation = sed;
    c.grain.coagulation = coag;
    return c;
}
}  // namespace

TEST_CASE("Table 1: sedimentation times (analytic disk)", "[paper][table1]") {
    struct Case { double r, a, t_paper; };
    // First entry into |z| <= 0.005 z_init.
    for (const Case c : {Case{1, 1, 42.5}, Case{1, 10, 3.8}, Case{10, 1, 564}, Case{10, 100, 8.1},
                         Case{50, 100, 89.5}, Case{100, 100, 252}}) {
        CAPTURE(c.r, c.a);
        const auto res = run(paper_config(c.a, c.r), {}, false);
        REQUIRE(dpd::ode::is_success(res.summary.status));
        CHECK_THAT(res.summary.t_settle_first_yr, WithinRel(c.t_paper, 0.05));
    }
    // 1 m at 1 au: the paper's value corresponds to final settling (last entry), see docs.
    const auto res = run(paper_config(100, 1), {}, false);
    CHECK_THAT(res.summary.t_settle_last_yr, WithinRel(2.7, 0.05));
}

TEST_CASE("Table 2: radial drift times (analytic disk, no coagulation)", "[paper][table2]") {
    struct Case { double r, a, t_paper; };
    for (const Case c : {Case{1, 1, 2.4e3}, Case{1, 10, 322}, Case{1, 100, 98.2}, Case{10, 1, 40e3},
                         Case{10, 10, 4.9e3}, Case{10, 100, 4.5e3}, Case{50, 100, 85e3}, Case{100, 100, 306e3}}) {
        CAPTURE(c.r, c.a);
        const auto res = run(paper_config(c.a, c.r), {}, false);
        REQUIRE(res.summary.reached_inner_edge);
        CHECK_THAT(res.summary.t_drift_yr, WithinRel(c.t_paper, 0.04));
    }
}

TEST_CASE("Table 4: drift times with coagulation (radial drift only, as in jord++)", "[paper][table4]") {
    struct Case { double r, a, t_paper; };
    for (const Case c : {Case{1, 1e-4, 7.1e3}, Case{1, 0.1, 2.9e3}, Case{1, 1, 1.2e3}, Case{1, 10, 262},
                         Case{10, 0.1, 67e3}, Case{10, 1, 22e3}, Case{10, 10, 4.3e3}, Case{50, 0.1, 556e3},
                         Case{50, 1, 165e3}}) {
        CAPTURE(c.r, c.a);
        const auto res = run(paper_config(c.a, c.r, false, true), {}, false);
        REQUIRE(res.summary.reached_inner_edge);
        CHECK_THAT(res.summary.t_drift_yr, WithinRel(c.t_paper, 0.03));
    }
}

TEST_CASE("Sec. 3.5.2 / Fig. 12: growth from 10 au with coagulation", "[paper][fig12]") {
    // a_init = 1 cm -> 21 cm in 22 kyr; 1 dm -> 30 cm in 4.3 kyr; 1 m: 4.5 kyr.
    auto r1 = run(paper_config(1, 10, false, true), {}, false);
    CHECK_THAT(r1.summary.a_final_cm, WithinRel(21.0, 0.05));
    auto r2 = run(paper_config(10, 10, false, true), {}, false);
    CHECK_THAT(r2.summary.a_final_cm, WithinRel(30.0, 0.05));
    CHECK_THAT(r2.summary.t_drift_yr, WithinRel(4.3e3, 0.03));
    auto r3 = run(paper_config(100, 10, false, true), {}, false);
    CHECK_THAT(r3.summary.t_drift_yr, WithinRel(4.5e3, 0.03));
}

TEST_CASE("Sec. 3.4 / Fig. 7: 1 m body from 100 au (reference case)", "[paper][fig7]") {
    const auto res = run(paper_config(100, 100));
    REQUIRE(res.summary.reached_inner_edge);
    CHECK_THAT(res.summary.t_drift_yr, WithinRel(306e3, 0.02));      // "approximately 306 kyr"
    CHECK_THAT(res.summary.t_settle_first_yr, WithinRel(252, 0.02));  // Table 1 / Sec. 3.3
    CHECK_THAT(res.summary.vr_max_cms / 100, WithinRel(58.5, 0.10));  // peak drift speed
    CHECK(res.summary.r_at_vr_max_au > 0.25);                          // paper: r ~= 0.31 au
    CHECK(res.summary.r_at_vr_max_au < 0.45);
    // Qualitative behaviour (Sec. 3.3): fast settling with damped oscillations, then drift.
    int sign_changes = 0;
    for (std::size_t i = 1; i < res.diag.size(); ++i)
        if ((res.diag[i].z_au > 0) != (res.diag[i - 1].z_au > 0) && res.diag[i].t_yr < 2e3) ++sign_changes;
    CHECK(sign_changes >= 4);
    CHECK(res.summary.t_settle_first_yr < 1e-2 * res.summary.t_drift_yr);
}
