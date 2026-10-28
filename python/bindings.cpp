// pybind11 bindings: dustppd._core
//
// Ownership/lifetime: all results are returned as new NumPy arrays. Trajectory
// arrays (t, y) take ownership of the C++ vectors without copying (capsule);
// per-row diagnostics are converted to column arrays (one copy). Long-running
// C++ calls release the GIL; Python RHS callbacks re-acquire it.
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <memory>

#include "dpd/dust/config.hpp"
#include "dpd/dust/diffusion.hpp"
#include "dpd/dust/traps.hpp"
#include "dpd/dust/io.hpp"
#include "dpd/dust/run.hpp"
#include "dpd/ode/integrate.hpp"
#include "dpd/ode/kernels.hpp"
#include "dpd/ode/test_problems.hpp"

namespace py = pybind11;
using namespace dpd;

namespace {

template <class T>
py::array_t<T> to_numpy(std::vector<T>&& v, std::vector<py::ssize_t> shape) {
    auto* heap = new std::vector<T>(std::move(v));
    py::capsule owner(heap, [](void* p) { delete static_cast<std::vector<T>*>(p); });
    std::vector<py::ssize_t> strides(shape.size());
    py::ssize_t s = sizeof(T);
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = s;
        s *= shape[i];
    }
    return py::array_t<T>(shape, strides, heap->data(), owner);
}

py::dict stats_dict(const ode::Stats& s) {
    py::dict d;
    d["n_rhs"] = s.n_rhs;
    d["n_jac"] = s.n_jac;
    d["n_jvp"] = s.n_jvp;
    d["n_lu"] = s.n_lu;
    d["n_steps"] = s.n_steps;
    d["n_reject_error"] = s.n_reject_error;
    d["n_reject_newton"] = s.n_reject_newton;
    d["n_reject_nonfinite"] = s.n_reject_nonfinite;
    d["n_reject_domain"] = s.n_reject_domain;
    d["n_newton_iter"] = s.n_newton_iter;
    d["n_linear_iter"] = s.n_linear_iter;
    d["n_lost_increment"] = s.n_lost_increment;
    d["n_events"] = s.n_events;
    d["h_min"] = s.h_min;
    d["h_max"] = s.h_max;
    py::list sw;
    for (const auto& w : s.switches) {
        py::dict e;
        e["t"] = w.t;
        e["from"] = std::string(ode::to_string(w.from));
        e["to"] = std::string(ode::to_string(w.to));
        e["reason"] = w.reason;
        sw.append(e);
    }
    d["switches"] = sw;
    return d;
}

ode::Method method_of(const std::string& s) { return dust::method_from_string(s); }

// ODE system defined by Python callables. Calls re-acquire the GIL.
class PyOdeSystem final : public ode::OdeSystem {
public:
    PyOdeSystem(py::function f, std::size_t n, py::object jac) : f_(std::move(f)), jac_(std::move(jac)), n_(n) {}
    [[nodiscard]] ode::Index size() const override { return n_; }
    void rhs(double t, std::span<const double> y, std::span<double> dydt) const override {
        py::gil_scoped_acquire gil;
        py::array_t<double> ya(static_cast<py::ssize_t>(n_), y.data());  // copy: callee may keep it
        py::array_t<double, py::array::c_style | py::array::forcecast> r = f_(t, ya);
        if (static_cast<std::size_t>(r.size()) != n_) throw std::runtime_error("rhs returned wrong length");
        std::copy(r.data(), r.data() + n_, dydt.begin());
    }
    [[nodiscard]] bool has_jacobian() const override { return !jac_.is_none(); }
    void jacobian(double t, std::span<const double> y, ode::DenseMatrix& J) const override {
        py::gil_scoped_acquire gil;
        py::array_t<double> ya(static_cast<py::ssize_t>(n_), y.data());
        py::array_t<double, py::array::c_style | py::array::forcecast> r = jac_(t, ya);
        if (static_cast<std::size_t>(r.size()) != n_ * n_) throw std::runtime_error("jac returned wrong shape");
        std::copy(r.data(), r.data() + n_ * n_, J.data());
    }

private:
    py::function f_;
    py::object jac_;
    std::size_t n_;
};

ode::Options make_opts(const std::string& method, double rtol, const std::vector<double>& atol, double h0,
                       double h_max, std::uint64_t max_steps, const std::string& linear_solver, int bdf_max_order,
                       double rk4_step, const std::vector<double>& t_eval, bool save_steps) {
    ode::Options o;
    o.method = method_of(method);
    o.rtol = rtol;
    o.atol = atol;
    o.h0 = h0;
    o.h_max = h_max;
    o.max_steps = max_steps;
    o.linear_solver = dust::linear_solver_from_string(linear_solver);
    o.bdf_max_order = bdf_max_order;
    o.rk4_step = rk4_step;
    o.t_eval = t_eval;
    o.save_steps = save_steps;
    return o;
}

py::dict result_dict(ode::Result&& r) {
    py::dict d;
    const auto n = static_cast<py::ssize_t>(r.n);
    const auto m = static_cast<py::ssize_t>(r.t.size());
    d["status"] = std::string(ode::to_string(r.status));
    d["success"] = r.ok();
    d["message"] = r.message;
    d["t"] = to_numpy(std::move(r.t), {m});
    d["y"] = to_numpy(std::move(r.y), {m, n});
    d["t_last"] = r.t_last;
    d["y_last"] = to_numpy(std::move(r.y_last), {n});
    d["stats"] = stats_dict(r.stats);
    d["final_method"] = std::string(ode::to_string(r.final_method));
    return d;
}

struct RunControl {
    std::atomic<bool> cancel{false};
    std::atomic<double> progress{0.0};
};

py::dict summary_dict(const dust::RunSummary& s) {
    py::dict d;
    d["status"] = std::string(ode::to_string(s.status));
    d["success"] = ode::is_success(s.status);
    d["message"] = s.message;
    d["reached_inner_edge"] = s.reached_inner_edge;
    d["t_final_yr"] = s.t_final_yr;
    d["r_final_au"] = s.r_final_au;
    d["a_final_cm"] = s.a_final_cm;
    d["t_drift_yr"] = s.t_drift_yr;
    d["t_settle_first_yr"] = s.t_settle_first_yr;
    d["t_settle_last_yr"] = s.t_settle_last_yr;
    d["vr_max_cms"] = s.vr_max_cms;
    d["r_at_vr_max_au"] = s.r_at_vr_max_au;
    d["a_max_cm"] = s.a_max_cm;
    d["max_Re"] = s.max_Re;
    d["stokes_regime_seen"] = s.stokes_regime_seen;
    d["epstein_regime_seen"] = s.epstein_regime_seen;
    d["beta_extrapolated"] = s.beta_extrapolated;
    d["trapped"] = s.trapped;
    d["r_trap_au"] = s.r_trap_au;
    d["warnings"] = s.warnings;
    d["stats"] = stats_dict(s.stats);
    d["final_method"] = std::string(ode::to_string(s.final_method));
    d["wall_time_s"] = s.wall_time_s;
    return d;
}

py::dict run_dict(dust::RunResult&& r) {
    py::dict d;
    d["summary"] = summary_dict(r.summary);
    const auto m = static_cast<py::ssize_t>(r.t.size());
    d["t_code"] = to_numpy(std::move(r.t), {m});
    d["y_code"] = to_numpy(std::move(r.y), {m, 6});
    py::dict cols;
    const std::size_t k = r.diag.size();
#define DPD_COL(name, field)                                         \
    {                                                                \
        std::vector<double> v(k);                                    \
        for (std::size_t i = 0; i < k; ++i) v[i] = r.diag[i].field;  \
        cols[name] = to_numpy(std::move(v), {static_cast<py::ssize_t>(k)}); \
    }
    DPD_COL("t_yr", t_yr) DPD_COL("r_au", r_au) DPD_COL("z_au", z_au) DPD_COL("vr_cms", vr_cms)
    DPD_COL("vz_cms", vz_cms) DPD_COL("vphi_cms", vphi_cms) DPD_COL("a_cm", a_cm) DPD_COL("St", St)
    DPD_COL("tau_s", tau_s) DPD_COL("lambda_cm", lambda_cm) DPD_COL("Re", Re) DPD_COL("rho_gcc", rho)
    DPD_COL("T_K", T) DPD_COL("H_au", H_au) DPD_COL("ur_cms", ur_cms) DPD_COL("uphi_cms", uphi_cms)
    DPD_COL("beta", beta) DPD_COL("dv_cms", dv_cms) DPD_COL("stokes", stokes) DPD_COL("Z_charge", Z_charge)
    DPD_COL("hall", hall)
#undef DPD_COL
    d["columns"] = cols;
    return d;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "dustppd: ODE solvers and the Kobozeva (2025) dust-dynamics model (C++ core)";

    // ---------------------------------------------------------- kernels ---
    m.def("available_backends", [] {
        std::vector<std::string> out;
        for (auto b : ode::kernels::available_backends()) out.emplace_back(ode::kernels::to_string(b));
        return out;
    });
    m.def("active_backend", [] { return std::string(ode::kernels::to_string(ode::kernels::active_backend())); });
    m.def("set_backend", [](const std::string& name) {
        for (auto b : {ode::kernels::Backend::Scalar, ode::kernels::Backend::Generic, ode::kernels::Backend::Neon,
                       ode::kernels::Backend::Avx2})
            if (ode::kernels::to_string(b) == name) return ode::kernels::set_backend(b);
        throw py::value_error("unknown backend '" + name + "'");
    });

    // -------------------------------------------------- generic ODE API ---
    py::class_<ode::OdeSystem, std::shared_ptr<ode::OdeSystem>>(m, "OdeSystem")
        .def_property_readonly("size", &ode::OdeSystem::size)
        .def("rhs", [](const ode::OdeSystem& s, double t, py::array_t<double, py::array::c_style | py::array::forcecast> y) {
            if (static_cast<std::size_t>(y.size()) != s.size()) throw py::value_error("y has wrong length");
            std::vector<double> f(s.size());
            s.rhs(t, std::span<const double>(y.data(), s.size()), f);
            return to_numpy(std::move(f), {static_cast<py::ssize_t>(s.size())});
        });
    m.def("robertson", [] { return std::shared_ptr<ode::OdeSystem>(std::make_shared<ode::problems::Robertson>()); });
    m.def("van_der_pol", [](double mu) {
        return std::shared_ptr<ode::OdeSystem>(std::make_shared<ode::problems::VanDerPol>(mu));
    }, py::arg("mu"));
    m.def("harmonic_oscillator", [](double w) {
        return std::shared_ptr<ode::OdeSystem>(std::make_shared<ode::problems::HarmonicOscillator>(w));
    }, py::arg("omega"));
    m.def("reaction_diffusion", [](std::size_t n, double d, double k) {
        return std::shared_ptr<ode::OdeSystem>(std::make_shared<ode::problems::ReactionDiffusion1D>(n, d, k));
    }, py::arg("n"), py::arg("d") = 1.0, py::arg("k") = 1.0);

    auto solve_impl = [](const ode::OdeSystem& sys, std::pair<double, double> span, std::vector<double> y0,
                         const ode::Options& o) {
        ode::Result r;
        {
            py::gil_scoped_release nogil;
            r = ode::integrate(sys, span.first, y0, span.second, o);
        }
        return result_dict(std::move(r));
    };

    m.def("solve_system",
          [solve_impl](const std::shared_ptr<ode::OdeSystem>& sys, std::pair<double, double> t_span, std::vector<double> y0,
                       const std::string& method, double rtol, std::vector<double> atol, double h0, double h_max,
                       std::uint64_t max_steps, const std::string& linear_solver, int bdf_max_order, double rk4_step,
                       std::vector<double> t_eval, bool save_steps) {
              const auto o = make_opts(method, rtol, atol, h0, h_max, max_steps, linear_solver, bdf_max_order, rk4_step,
                                       t_eval, save_steps);
              return solve_impl(*sys, t_span, std::move(y0), o);
          },
          py::arg("system"), py::arg("t_span"), py::arg("y0"), py::arg("method") = "auto", py::arg("rtol") = 1e-6,
          py::arg("atol") = std::vector<double>{1e-9}, py::arg("h0") = 0.0,
          py::arg("h_max") = std::numeric_limits<double>::infinity(), py::arg("max_steps") = 50'000'000ULL,
          py::arg("linear_solver") = "dense_lu", py::arg("bdf_max_order") = 5, py::arg("rk4_step") = 0.0,
          py::arg("t_eval") = std::vector<double>{}, py::arg("save_steps") = true,
          "Integrate a built-in C++ system (GIL released).");

    m.def("solve",
          [solve_impl](py::function fun, std::pair<double, double> t_span, std::vector<double> y0, const std::string& method,
                       double rtol, std::vector<double> atol, py::object jac, double h0, double h_max,
                       std::uint64_t max_steps, int bdf_max_order, double rk4_step, std::vector<double> t_eval,
                       bool save_steps) {
              PyOdeSystem sys(std::move(fun), y0.size(), std::move(jac));
              const auto o = make_opts(method, rtol, atol, h0, h_max, max_steps, "dense_lu", bdf_max_order, rk4_step,
                                       t_eval, save_steps);
              return solve_impl(sys, t_span, std::move(y0), o);
          },
          py::arg("fun"), py::arg("t_span"), py::arg("y0"), py::arg("method") = "auto", py::arg("rtol") = 1e-6,
          py::arg("atol") = std::vector<double>{1e-9}, py::arg("jac") = py::none(), py::arg("h0") = 0.0,
          py::arg("h_max") = std::numeric_limits<double>::infinity(), py::arg("max_steps") = 50'000'000ULL,
          py::arg("bdf_max_order") = 5, py::arg("rk4_step") = 0.0, py::arg("t_eval") = std::vector<double>{},
          py::arg("save_steps") = true,
          "Integrate fun(t, y) -> dy/dt given as a Python callable. Exceptions raised by fun propagate.");

    // -------------------------------------------------------- dust model ---
    py::class_<dust::RunConfig>(m, "RunConfig")
        .def(py::init<>())
        .def_readwrite("name", &dust::RunConfig::name)
        .def_readwrite("description", &dust::RunConfig::description)
        .def_readwrite("r_init_au", &dust::RunConfig::r_init_au)
        .def_readwrite("z_init_H", &dust::RunConfig::z_init_H)
        .def_readwrite("vr_init_cms", &dust::RunConfig::vr_init_cms)
        .def_readwrite("vz_init_cms", &dust::RunConfig::vz_init_cms)
        .def_readwrite("r_in_au", &dust::RunConfig::r_in_au)
        .def_readwrite("t_end_yr", &dust::RunConfig::t_end_yr)
        .def_readwrite("settle_eps", &dust::RunConfig::settle_eps)
        .def_property("a_init_cm", [](const dust::RunConfig& c) { return c.grain.a_init_cm; },
                      [](dust::RunConfig& c, double v) { c.grain.a_init_cm = v; })
        .def_property("rho_d", [](const dust::RunConfig& c) { return c.grain.rho_d; },
                      [](dust::RunConfig& c, double v) { c.grain.rho_d = v; })
        .def_property("drag", [](const dust::RunConfig& c) { return std::string(dust::to_string(c.grain.drag)); },
                      [](dust::RunConfig& c, const std::string& v) { c.grain.drag = dust::drag_from_string(v); })
        .def_property("coagulation", [](const dust::RunConfig& c) { return c.grain.coagulation; },
                      [](dust::RunConfig& c, bool v) { c.grain.coagulation = v; })
        .def_property("sedimentation", [](const dust::RunConfig& c) { return c.grain.sedimentation; },
                      [](dust::RunConfig& c, bool v) { c.grain.sedimentation = v; })
        .def_property("dust_to_gas", [](const dust::RunConfig& c) { return c.grain.dust_to_gas; },
                      [](dust::RunConfig& c, double v) { c.grain.dust_to_gas = v; })
        .def_property("mu", [](const dust::RunConfig& c) { return c.grain.mu; },
                      [](dust::RunConfig& c, double v) { c.grain.mu = v; })
        .def_property("sigma_mol_cm2", [](const dust::RunConfig& c) { return c.grain.sigma_mol_cm2; },
                      [](dust::RunConfig& c, double v) { c.grain.sigma_mol_cm2 = v; })
        .def_property("alpha", [](const dust::RunConfig& c) { return c.disk.alpha; },
                      [](dust::RunConfig& c, double v) { c.disk.alpha = v; })
        .def_property("mdot", [](const dust::RunConfig& c) { return c.disk.mdot; },
                      [](dust::RunConfig& c, double v) { c.disk.mdot = v; })
        .def_property("mstar", [](const dust::RunConfig& c) { return c.disk.mstar; },
                      [](dust::RunConfig& c, double v) { c.disk.mstar = v; })
        .def_property("vphi2vk", [](const dust::RunConfig& c) { return c.disk.vphi2vk; },
                      [](dust::RunConfig& c, double v) { c.disk.vphi2vk = v; })
        .def_property("beta_table", [](const dust::RunConfig& c) { return c.beta_table_path; },
                      [](dust::RunConfig& c, const std::string& p) {
                          if (p.empty()) {
                              c.beta_table_path.clear();
                              c.disk.beta_r_au.clear();
                              c.disk.beta_values.clear();
                              if (c.disk.model == dust::DiskModel::TabulatedBeta) c.disk.model = dust::DiskModel::Analytic;
                          } else {
                              dust::load_beta_table(p, c.disk);
                              c.disk.model = dust::DiskModel::TabulatedBeta;
                              c.beta_table_path = p;
                          }
                      })
        .def_property("method", [](const dust::RunConfig& c) { return std::string(ode::to_string(c.solver.method)); },
                      [](dust::RunConfig& c, const std::string& v) { c.solver.method = method_of(v); })
        .def_property("rtol", [](const dust::RunConfig& c) { return c.solver.rtol; },
                      [](dust::RunConfig& c, double v) { c.solver.rtol = v; })
        .def_property("atol", [](const dust::RunConfig& c) { return c.solver.atol; },
                      [](dust::RunConfig& c, std::vector<double> v) { c.solver.atol = std::move(v); })
        .def_property("max_steps", [](const dust::RunConfig& c) { return c.solver.max_steps; },
                      [](dust::RunConfig& c, std::uint64_t v) { c.solver.max_steps = v; })
        .def_property("linear_solver",
                      [](const dust::RunConfig& c) { return std::string(ode::to_string(c.solver.linear_solver)); },
                      [](dust::RunConfig& c, const std::string& v) {
                          c.solver.linear_solver = dust::linear_solver_from_string(v);
                      })
        .def_property("disk_model", [](const dust::RunConfig& c) { return std::string(dust::to_string(c.disk.model)); },
                      [](dust::RunConfig& c, const std::string& v) { c.disk.model = dust::disk_model_from_string(v); })
        .def_property("perturbation",
                      [](const dust::RunConfig& c) { return std::string(dust::to_string(c.disk.perturbation)); },
                      [](dust::RunConfig& c, const std::string& v) {
                          c.disk.perturbation = dust::perturbation_from_string(v);
                      })
        .def_property("bump_amplitude", [](const dust::RunConfig& c) { return c.disk.bump_amplitude; },
                      [](dust::RunConfig& c, double v) { c.disk.bump_amplitude = v; })
        .def_property("bump_r_au", [](const dust::RunConfig& c) { return c.disk.bump_r_au; },
                      [](dust::RunConfig& c, double v) { c.disk.bump_r_au = v; })
        .def_property("bump_width_au", [](const dust::RunConfig& c) { return c.disk.bump_width_au; },
                      [](dust::RunConfig& c, double v) { c.disk.bump_width_au = v; })
        .def_property("magnetic_tension", [](const dust::RunConfig& c) { return c.disk.magnetic_tension; },
                      [](dust::RunConfig& c, bool v) { c.disk.magnetic_tension = v; })
        .def_property("B0_G", [](const dust::RunConfig& c) { return c.disk.field.B0_G; },
                      [](dust::RunConfig& c, double v) { c.disk.field.B0_G = v; })
        .def_property("field_exponent", [](const dust::RunConfig& c) { return c.disk.field.exponent; },
                      [](dust::RunConfig& c, double v) { c.disk.field.exponent = v; })
        .def_property("br_over_bz", [](const dust::RunConfig& c) { return c.disk.field.br_over_bz; },
                      [](dust::RunConfig& c, double v) { c.disk.field.br_over_bz = v; })
        .def_property("charge",
                      [](const dust::RunConfig& c) {
                          return std::string(c.grain.charge == dust::ChargeModel::OML ? "oml" : "none");
                      },
                      [](dust::RunConfig& c, const std::string& v) {
                          if (v == "oml") c.grain.charge = dust::ChargeModel::OML;
                          else if (v == "none") c.grain.charge = dust::ChargeModel::None;
                          else throw py::value_error("charge must be 'none' or 'oml'");
                      })
        .def_property("ion_mass_amu", [](const dust::RunConfig& c) { return c.grain.ion_mass_amu; },
                      [](dust::RunConfig& c, double v) { c.grain.ion_mass_amu = v; })
        .def_property("profile_table", [](const dust::RunConfig& c) { return c.profile_table_path; },
                      [](dust::RunConfig& c, const std::string& p) {
                          c.disk.table = dust::load_profile_table(p);
                          c.disk.model = dust::DiskModel::Tabulated;
                          c.profile_table_path = p;
                      })
        .def("to_toml", [](const dust::RunConfig& c) { return dust::to_toml(c); })
        .def("copy", [](const dust::RunConfig& c) { return c; })
        .def("__repr__", [](const dust::RunConfig& c) {
            return "<RunConfig '" + c.name + "' a=" + std::to_string(c.grain.a_init_cm) + " cm r=" +
                   std::to_string(c.r_init_au) + " au>";
        });

    m.def("load_config", &dust::load_config, py::arg("path"), py::arg("overrides") = std::vector<std::string>{});
    m.def("parse_config", &dust::parse_config, py::arg("text"), py::arg("base_dir") = ".",
          py::arg("overrides") = std::vector<std::string>{});
    m.def("save_config", &dust::save_config, py::arg("config"), py::arg("path"));

    py::class_<RunControl, std::shared_ptr<RunControl>>(m, "RunControl", "Thread-safe cancel flag and progress.")
        .def(py::init<>())
        .def("cancel", [](RunControl& c) { c.cancel.store(true); })
        .def_property_readonly("cancelled", [](const RunControl& c) { return c.cancel.load(); })
        .def_property_readonly("progress", [](const RunControl& c) { return c.progress.load(); });

    m.def("run",
          [](const dust::RunConfig& cfg, std::shared_ptr<RunControl> ctl, bool keep_trajectory) {
              dust::RunResult r;
              {
                  py::gil_scoped_release nogil;
                  dust::RunControl c;
                  if (ctl) {
                      c.cancel = &ctl->cancel;
                      c.progress = &ctl->progress;
                  }
                  r = dust::run(cfg, c, keep_trajectory);
              }
              return run_dict(std::move(r));
          },
          py::arg("config"), py::arg("control") = nullptr, py::arg("keep_trajectory") = true,
          "Run one dust-model simulation (GIL released). Returns {'summary', 'columns', 't_code', 'y_code'}.");

    m.def("run_ensemble",
          [](const std::vector<dust::RunConfig>& cfgs, unsigned threads) {
              std::vector<dust::RunResult> rs;
              {
                  py::gil_scoped_release nogil;
                  rs = dust::run_ensemble(cfgs, threads, false);
              }
              py::list out;
              for (auto& r : rs) out.append(summary_dict(r.summary));
              return out;
          },
          py::arg("configs"), py::arg("threads") = 0u, "Run independent configurations in parallel; returns summaries.");

    m.def("write_csv", [](const dust::RunConfig& cfg, const std::string& path) {
        dust::RunResult r;
        {
            py::gil_scoped_release nogil;
            r = dust::run(cfg);
        }
        dust::write_csv(r, path);
        return summary_dict(r.summary);
    }, py::arg("config"), py::arg("path"), "Run and write the trajectory CSV (same columns as the CLI).");

    // ------------------------------------------------- disk structure / traps ---
    m.def("disk_profile", [](const dust::RunConfig& cfg, std::vector<double> r) {
        const dust::Disk d(cfg.disk);
        const std::size_t n = r.size();
        std::vector<double> T(n), rho(n), H(n), ur(n), beta(n), p(n), Bz(n), Br(n);
        for (std::size_t i = 0; i < n; ++i) {
            T[i] = d.temperature(r[i]);
            rho[i] = d.rho_mid(r[i]);
            H[i] = d.scale_height(r[i]);
            ur[i] = d.ur_cgs(r[i]);
            beta[i] = d.beta(r[i]);
            p[i] = d.pressure_mid(r[i]);
            Bz[i] = d.Bz(r[i]);
            Br[i] = d.Br(r[i]);
        }
        const auto N = static_cast<py::ssize_t>(n);
        py::dict out;
        out["r_au"] = to_numpy(std::move(r), {N});
        out["T_K"] = to_numpy(std::move(T), {N});
        out["rho_gcc"] = to_numpy(std::move(rho), {N});
        out["H_au"] = to_numpy(std::move(H), {N});
        out["ur_cms"] = to_numpy(std::move(ur), {N});
        out["beta"] = to_numpy(std::move(beta), {N});
        out["p_mid"] = to_numpy(std::move(p), {N});
        out["Bz_G"] = to_numpy(std::move(Bz), {N});
        out["Br_G"] = to_numpy(std::move(Br), {N});
        return out;
    }, py::arg("config"), py::arg("r_au"), "Midplane disk structure on the given radii.");

    m.def("stationary_vr", [](const dust::RunConfig& cfg, std::vector<double> r, double a_cm) {
        dust::GrainParams g = cfg.grain;
        g.a_init_cm = a_cm;
        const dust::DustModel mdl(cfg.disk, g);
        std::vector<double> v(r.size());
        for (std::size_t i = 0; i < r.size(); ++i) v[i] = dust::stationary_vr_cms(mdl, r[i], a_cm);
        return to_numpy(std::move(v), {static_cast<py::ssize_t>(r.size())});
    }, py::arg("config"), py::arg("r_au"), py::arg("a_cm"), "Stationary midplane drift velocity [cm/s] (paper Eq. 6).");

    m.def("drift_zeros", [](const dust::RunConfig& cfg, double a_cm, double r_min, double r_max) {
        dust::GrainParams g = cfg.grain;
        g.a_init_cm = a_cm;
        py::list out;
        for (const auto& z : dust::drift_zeros(dust::DustModel(cfg.disk, g), a_cm, r_min, r_max)) {
            py::dict d;
            d["r_au"] = z.r_au;
            d["stable"] = z.stable;
            out.append(d);
        }
        return out;
    }, py::arg("config"), py::arg("a_cm"), py::arg("r_min") = 0.1, py::arg("r_max") = 300.0,
       "Radii where the stationary drift vanishes; 'stable' marks traps.");

    m.def("zero_headwind_radii", [](const dust::RunConfig& cfg, double r_min, double r_max) {
        return dust::zero_headwind_radii(dust::Disk(cfg.disk), r_min, r_max);
    }, py::arg("config"), py::arg("r_min") = 0.1, py::arg("r_max") = 300.0);

    m.def("min_trapped_size", [](const dust::RunConfig& cfg, double r_min, double r_max, double a_lo, double a_hi) {
        double out;
        {
            py::gil_scoped_release nogil;
            out = dust::min_trapped_size_cm(dust::DustModel(cfg.disk, cfg.grain), r_min, r_max, a_lo, a_hi);
        }
        return out;
    }, py::arg("config"), py::arg("r_min") = 0.1, py::arg("r_max") = 300.0, py::arg("a_lo") = 1e-5,
       py::arg("a_hi") = 1e3, "Smallest trapped grain radius [cm] (NaN if none).");

    m.def("hall_parameter", [](const dust::RunConfig& cfg, std::vector<double> r, std::vector<double> z, double a_cm) {
        if (r.size() != z.size()) throw py::value_error("r and z must have equal length");
        dust::GrainParams g = cfg.grain;
        g.a_init_cm = a_cm;
        g.charge = dust::ChargeModel::OML;
        const dust::DustModel mdl(cfg.disk, g);
        std::vector<double> out(r.size());
        for (std::size_t i = 0; i < r.size(); ++i) {
            const auto L = mdl.disk().local(r[i], z[i]);
            const double B = std::hypot(mdl.disk().Bz(r[i]), mdl.disk().Br(r[i]));
            out[i] = std::abs(mdl.lorentz_rate(L.T, a_cm, B)) * mdl.drag_at(L, a_cm).tau;
        }
        return to_numpy(std::move(out), {static_cast<py::ssize_t>(r.size())});
    }, py::arg("config"), py::arg("r_au"), py::arg("z_au"), py::arg("a_cm"),
       "Hall parameter (Z e B / m c) tau_s of an OML-charged grain at points (r, z).");

    m.def("oml_potential", &dust::oml_potential, py::arg("ion_mass_amu") = 29.0);

    // ------------------------------------------------------ vertical diffusion ---
    m.def("vertical_diffusion",
          [](const dust::RunConfig& cfg, double r_au, double a_cm, double alpha_turb, std::size_t n_paths,
             std::uint64_t seed, double t_end_relax, double dt_relax, double z_init_H, const std::string& scheme,
             unsigned threads) {
              dust::DiffusionRunConfig dc;
              dc.r_au = r_au;
              dc.a_cm = a_cm;
              dc.alpha_turb = alpha_turb > 0 ? alpha_turb : cfg.disk.alpha;
              dc.n_paths = n_paths;
              dc.seed = seed;
              dc.t_end_relax = t_end_relax;
              dc.dt_relax = dt_relax;
              dc.z_init_H = z_init_H;
              dc.threads = threads;
              if (scheme == "euler_maruyama") dc.scheme = ode::SdeScheme::EulerMaruyama;
              else if (scheme == "milstein") dc.scheme = ode::SdeScheme::Milstein;
              else if (scheme == "linear_implicit_euler") dc.scheme = ode::SdeScheme::LinearImplicitEuler;
              else throw py::value_error("scheme: euler_maruyama | milstein | linear_implicit_euler");
              dust::DiffusionRunResult r;
              {
                  py::gil_scoped_release nogil;
                  r = dust::run_vertical_diffusion(cfg.disk, cfg.grain, dc);
              }
              py::dict d;
              d["status"] = std::string(ode::to_string(r.status));
              d["message"] = r.message;
              d["H_au"] = r.H_au;
              d["St_mid"] = r.St_mid;
              d["Hd_simulated_au"] = r.Hd_simulated_au;
              d["Hd_stat_error_au"] = r.Hd_stat_error_au;
              d["Hd_exact_au"] = r.Hd_exact_au;
              d["Hd_dubrulle_au"] = r.Hd_dubrulle_au;
              const auto K = static_cast<py::ssize_t>(r.t_yr.size());
              d["t_yr"] = to_numpy(std::move(r.t_yr), {K});
              d["z_rms_au"] = to_numpy(std::move(r.z_rms_au), {K});
              const auto P = static_cast<py::ssize_t>(r.z_final_au.size());
              d["z_final_au"] = to_numpy(std::move(r.z_final_au), {P});
              d["n_steps"] = r.n_steps;
              d["n_failed_paths"] = r.n_failed_paths;
              d["wall_time_s"] = r.wall_time_s;
              d["warnings"] = r.warnings;
              return d;
          },
          py::arg("config"), py::arg("r_au"), py::arg("a_cm"), py::arg("alpha_turb") = 0.0,
          py::arg("n_paths") = 20000, py::arg("seed") = 1, py::arg("t_end_relax") = 10.0,
          py::arg("dt_relax") = 0.02, py::arg("z_init_H") = 0.0, py::arg("scheme") = "linear_implicit_euler",
          py::arg("threads") = 0u,
          "Stochastic vertical mixing of grains at radius r (GIL released). alpha_turb <= 0 uses the disk alpha.");

    m.def("stationary_dust_density", [](const dust::RunConfig& cfg, double r_au, double a_cm, double alpha_turb,
                                        std::vector<double> z) {
        dust::GrainParams g = cfg.grain;
        g.a_init_cm = a_cm;
        const dust::DustModel mdl(cfg.disk, g);
        const dust::VerticalDiffusion vd(mdl, r_au, a_cm, alpha_turb);
        auto p = vd.stationary_density(z);
        py::dict d;
        d["Hd_exact_au"] = vd.stationary_Hd();
        d["Hd_dubrulle_au"] = vd.dubrulle_Hd();
        d["H_au"] = vd.H();
        d["St_mid"] = vd.St(0);
        d["density"] = to_numpy(std::move(p), {static_cast<py::ssize_t>(z.size())});
        return d;
    }, py::arg("config"), py::arg("r_au"), py::arg("a_cm"), py::arg("alpha_turb"), py::arg("z_au"),
       "Exact zero-flux stationary vertical dust distribution (normalized on the grid).");

    m.def("code_units", [] {
        const auto u = dust::make_units(1.0);
        py::dict d;
        d["r0_cm"] = u.r0_cm;
        d["v0_cms"] = u.v0_cms;
        d["t0_s"] = u.t0_s;
        d["t0_yr"] = u.t0_yr();
        return d;
    });
}
