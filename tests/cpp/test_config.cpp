#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>

#include "dpd/dust/config.hpp"

using namespace dpd;
using namespace dpd::dust;

static const std::string kRef = std::string(DPD_SOURCE_DIR) + "/configs/reference/paper_1m_100au.toml";

TEST_CASE("reference configuration parses with the paper's parameters", "[config]") {
    const RunConfig c = load_config(kRef);
    CHECK(c.grain.a_init_cm == 100.0);
    CHECK(c.r_init_au == 100.0);
    CHECK(c.z_init_H == 3.0);
    CHECK(c.grain.rho_d == 3.0);
    CHECK(c.grain.dust_to_gas == 0.01);
    CHECK(c.disk.alpha == 0.01);
    CHECK(c.disk.mdot == 1.0);
    CHECK(c.disk.vphi2vk == 0.9987);
    CHECK(c.r_in_au == 0.05);
    CHECK(c.grain.drag == DragLaw::Both);
    CHECK(c.solver.method == ode::Method::Auto);
    CHECK(c.solver.atol.size() == 6);
}

TEST_CASE("overrides, unknown keys and invalid values", "[config]") {
    const RunConfig c = load_config(kRef, {"particle.a_init_cm=0.1", "physics.drag=\"epstein\"", "solver.method=bdf"});
    CHECK(c.grain.a_init_cm == 0.1);
    CHECK(c.grain.drag == DragLaw::Epstein);
    CHECK(c.solver.method == ode::Method::Bdf);
    CHECK_THROWS(load_config(kRef, {"physics.not_a_key=1"}));
    CHECK_THROWS(load_config(kRef, {"bogus.section=1"}));
    CHECK_THROWS(load_config(kRef, {"physics.drag=\"newtonian\""}));
    CHECK_THROWS(load_config(kRef, {"particle.r_init_au=0.01"}));  // inside r_in
    CHECK_THROWS(load_config(kRef, {"solver.atol=[1,2]"}));
    CHECK_THROWS(parse_config("[run]\nname = "));                   // syntax error
    CHECK_THROWS(load_config("/nonexistent/file.toml"));
}

TEST_CASE("to_toml round trip", "[config]") {
    const RunConfig a = load_config(kRef, {"physics.coagulation=true", "solver.rtol=1e-7"});
    const RunConfig b = parse_config(to_toml(a));
    CHECK(b.grain.coagulation);
    CHECK(b.solver.rtol == 1e-7);
    CHECK(b.grain.a_init_cm == a.grain.a_init_cm);
    CHECK(b.solver.atol == a.solver.atol);
    CHECK(b.disk.vphi2vk == a.disk.vphi2vk);
    CHECK(b.name == a.name);
    // strings with quotes/backslashes/newlines and tiny/huge floats survive a round trip
    RunConfig c = a;
    c.name = "odd \"name\" with \\ and\nnewline";
    c.solver.atol = {1e-300, 1e-16, 1.0, 12345.0, 0.0, 1e300};  // (toml++ cannot parse subnormals)
    c.t_end_yr = 1e12;
    const RunConfig d = parse_config(to_toml(c));
    CHECK(d.name == c.name);
    CHECK(d.solver.atol == c.solver.atol);
    CHECK(d.t_end_yr == c.t_end_yr);
}

TEST_CASE("beta table loading", "[config]") {
    const auto dir = std::filesystem::temp_directory_path() / "dpd_cfg_test";
    std::filesystem::create_directories(dir);
    {
        std::ofstream f(dir / "beta.csv");
        f << "# test table\nr_au,beta\n0.1,0.003\n1,0.01\n100,0.05\n";
    }
    {
        std::ofstream f(dir / "cfg.toml");
        f << "[disk]\nmodel = \"tabulated_beta\"\nbeta_table = \"beta.csv\"\n[particle]\nr_init_au = 10.0\n";
    }
    const RunConfig c = load_config((dir / "cfg.toml").string());
    REQUIRE(c.disk.beta_r_au.size() == 3);
    CHECK(c.disk.beta_values[2] == 0.05);
    const Disk d(c.disk);
    CHECK(std::abs(d.beta(10) - 0.03) < 1e-12);  // linear in log r
    CHECK(d.beta(1000) == 0.05);
    CHECK(d.beta_extrapolated(1000));
    std::filesystem::remove_all(dir);
}
