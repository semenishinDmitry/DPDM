#include "dpd/dust/config.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <toml++/toml.hpp>

namespace dpd::dust {

namespace fs = std::filesystem;

ode::Method method_from_string(const std::string& s) {
    if (s == "auto") return ode::Method::Auto;
    if (s == "dopri5") return ode::Method::Dopri5;
    if (s == "bdf") return ode::Method::Bdf;
    if (s == "rk4") return ode::Method::Rk4;
    throw std::invalid_argument("unknown solver method '" + s + "' (expected auto|dopri5|bdf|rk4)");
}

ode::LinearSolverKind linear_solver_from_string(const std::string& s) {
    if (s == "dense_lu") return ode::LinearSolverKind::DenseLU;
    if (s == "gmres") return ode::LinearSolverKind::Gmres;
    throw std::invalid_argument("unknown linear solver '" + s + "' (expected dense_lu|gmres)");
}

namespace {

// Applies "section.key=value" overrides, where value is a TOML value literal
// (strings must be quoted, e.g. physics.drag="epstein").
void apply_override(toml::table& tbl, const std::string& ov) {
    const auto eq = ov.find('=');
    if (eq == std::string::npos) throw std::invalid_argument("override '" + ov + "' must be key=value");
    const std::string key = ov.substr(0, eq);
    const std::string val = ov.substr(eq + 1);
    toml::table parsed;
    try {
        parsed = toml::parse("v = " + val);
    } catch (const toml::parse_error&) {
        parsed = toml::parse("v = \"" + val + "\"");  // allow unquoted strings
    }
    toml::table* cur = &tbl;
    std::size_t start = 0;
    while (true) {
        const auto dot = key.find('.', start);
        const std::string part = key.substr(start, dot == std::string::npos ? std::string::npos : dot - start);
        if (dot == std::string::npos) {
            cur->insert_or_assign(part, *parsed.get("v"));
            break;
        }
        if (!cur->contains(part)) cur->insert(part, toml::table{});
        cur = (*cur)[part].as_table();
        if (!cur) throw std::invalid_argument("override '" + ov + "': '" + part + "' is not a table");
        start = dot + 1;
    }
}

template <class T>
void get(const toml::table& t, std::string_view sec, std::string_view key, T& out) {
    if (auto v = t[sec][key].value<T>()) out = *v;
}

void check_known(const toml::table& t) {
    static const std::vector<std::pair<std::string, std::vector<std::string>>> known = {
        {"run", {"name", "description", "t_end_yr"}},
        {"particle", {"a_init_cm", "rho_d", "r_init_au", "z_init_H", "vr_init_cms", "vz_init_cms"}},
        {"physics", {"drag", "coagulation", "sedimentation", "dust_to_gas", "sigma_mol_cm2", "mu", "r_in_au",
                     "settle_eps"}},
        {"disk", {"model", "alpha", "mdot", "mstar", "vphi2vk", "beta_table", "T1", "rho1", "ur1", "H1"}},
        {"solver", {"method", "rtol", "atol", "h_max", "max_steps", "linear_solver", "bdf_max_order", "rk4_step"}},
        {"output", {"save_steps"}},
    };
    for (auto&& [sec, node] : t) {
        // Copy the name: capturing a structured binding in a lambda needs clang >= 16
        // (Apple clang 15, Xcode 15, rejects it).
        const std::string sec_name(sec.str());
        auto it = std::find_if(known.begin(), known.end(), [&](auto& p) { return p.first == sec_name; });
        if (it == known.end()) throw std::invalid_argument("unknown config section [" + sec_name + "]");
        if (const auto* tab = node.as_table())
            for (auto&& [k, _] : *tab)
                if (std::find(it->second.begin(), it->second.end(), std::string(k.str())) == it->second.end())
                    throw std::invalid_argument("unknown config key " + sec_name + "." +
                                                std::string(k.str()));
    }
}

}  // namespace

void load_beta_table(const std::string& path, DiskParams& disk) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open beta table '" + path + "'");
    disk.beta_r_au.clear();
    disk.beta_values.clear();
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        for (char& c : line)
            if (c == ',' || c == ';' || c == '\t') c = ' ';
        std::istringstream ss(line);
        double r, b;
        if (!(ss >> r >> b)) continue;  // header
        disk.beta_r_au.push_back(r);
        disk.beta_values.push_back(b);
    }
    disk.beta_source = path;
    if (disk.beta_r_au.size() < 2) throw std::runtime_error("beta table '" + path + "' has fewer than 2 rows");
}

RunConfig parse_config(const std::string& text, const std::string& base_dir, const std::vector<std::string>& ovs) {
    toml::table t;
    try {
        t = toml::parse(text);
    } catch (const toml::parse_error& e) {
        std::ostringstream os;
        os << "TOML parse error: " << e.description() << " at line " << e.source().begin.line;
        throw std::invalid_argument(os.str());
    }
    for (const auto& ov : ovs) apply_override(t, ov);
    check_known(t);

    RunConfig c;
    get(t, "run", "name", c.name);
    get(t, "run", "description", c.description);
    get(t, "run", "t_end_yr", c.t_end_yr);

    get(t, "particle", "a_init_cm", c.grain.a_init_cm);
    get(t, "particle", "rho_d", c.grain.rho_d);
    get(t, "particle", "r_init_au", c.r_init_au);
    get(t, "particle", "z_init_H", c.z_init_H);
    get(t, "particle", "vr_init_cms", c.vr_init_cms);
    get(t, "particle", "vz_init_cms", c.vz_init_cms);

    std::string drag = to_string(c.grain.drag);
    get(t, "physics", "drag", drag);
    c.grain.drag = drag_from_string(drag);
    get(t, "physics", "coagulation", c.grain.coagulation);
    get(t, "physics", "sedimentation", c.grain.sedimentation);
    get(t, "physics", "dust_to_gas", c.grain.dust_to_gas);
    get(t, "physics", "sigma_mol_cm2", c.grain.sigma_mol_cm2);
    get(t, "physics", "mu", c.grain.mu);
    get(t, "physics", "r_in_au", c.r_in_au);
    get(t, "physics", "settle_eps", c.settle_eps);

    std::string model = "analytic";
    get(t, "disk", "model", model);
    get(t, "disk", "alpha", c.disk.alpha);
    get(t, "disk", "mdot", c.disk.mdot);
    get(t, "disk", "mstar", c.disk.mstar);
    get(t, "disk", "vphi2vk", c.disk.vphi2vk);
    get(t, "disk", "T1", c.disk.T1);
    get(t, "disk", "rho1", c.disk.rho1);
    get(t, "disk", "ur1", c.disk.ur1);
    get(t, "disk", "H1", c.disk.H1);
    get(t, "disk", "beta_table", c.beta_table_path);
    if (model == "tabulated_beta") {
        if (c.beta_table_path.empty()) throw std::invalid_argument("disk.model = tabulated_beta needs disk.beta_table");
        fs::path p(c.beta_table_path);
        if (p.is_relative()) p = fs::path(base_dir) / p;
        c.beta_table_path = p.lexically_normal().string();
        load_beta_table(c.beta_table_path, c.disk);
    } else if (model != "analytic") {
        throw std::invalid_argument("unknown disk.model '" + model + "' (expected analytic|tabulated_beta)");
    } else {
        c.beta_table_path.clear();
    }

    std::string method = std::string(ode::to_string(c.solver.method));
    get(t, "solver", "method", method);
    c.solver.method = method_from_string(method);
    get(t, "solver", "rtol", c.solver.rtol);
    if (auto* arr = t["solver"]["atol"].as_array()) {
        c.solver.atol.clear();
        for (auto& v : *arr) c.solver.atol.push_back(v.value<double>().value_or(0.0));
    } else if (auto v = t["solver"]["atol"].value<double>()) {
        c.solver.atol.assign(DustModel::kSize, *v);
    }
    get(t, "solver", "h_max", c.solver.h_max);
    if (auto v = t["solver"]["max_steps"].value<int64_t>()) c.solver.max_steps = static_cast<std::uint64_t>(*v);
    std::string ls = std::string(ode::to_string(c.solver.linear_solver));
    get(t, "solver", "linear_solver", ls);
    c.solver.linear_solver = linear_solver_from_string(ls);
    if (auto v = t["solver"]["bdf_max_order"].value<int64_t>()) c.solver.bdf_max_order = static_cast<int>(*v);
    get(t, "solver", "rk4_step", c.solver.rk4_step);
    get(t, "output", "save_steps", c.save_steps);

    if (c.solver.atol.size() != 1 && c.solver.atol.size() != DustModel::kSize)
        throw std::invalid_argument("solver.atol must have 1 or 6 entries");
    if (!(c.r_init_au > c.r_in_au) || !(c.r_in_au > 0))
        throw std::invalid_argument("particle.r_init_au must exceed physics.r_in_au > 0");
    if (!(c.t_end_yr > 0)) throw std::invalid_argument("run.t_end_yr must be > 0");
    if (!(c.settle_eps > 0 && c.settle_eps < 1)) throw std::invalid_argument("physics.settle_eps must be in (0,1)");
    return c;
}

RunConfig load_config(const std::string& path, const std::vector<std::string>& overrides) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open config '" + path + "'");
    std::stringstream ss;
    ss << in.rdbuf();
    const fs::path dir = fs::path(path).parent_path();
    return parse_config(ss.str(), dir.empty() ? "." : dir.string(), overrides);
}

namespace {
// Minimal TOML writer for the flat RunConfig schema. (toml++'s formatter is not
// used: v3.4.0 has undefined behaviour when sizing floats with negative
// exponents, found by UBSan.) Doubles use 17 significant digits and always carry
// a '.' or exponent so that they read back as TOML floats.
std::string tv(double v) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    std::ostringstream os;
    os << std::setprecision(17) << v;
    std::string s = os.str();
    if (s.find_first_of(".eE") == std::string::npos) s += ".0";
    return s;
}
std::string tv(bool b) { return b ? "true" : "false"; }
std::string tv(const std::string& s) {
    std::string o = "\"";
    for (char ch : s) {
        switch (ch) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\t': o += "\\t"; break;
            default: o += ch;
        }
    }
    return o + "\"";
}
std::string tv(const char* s) { return tv(std::string(s)); }
std::string tv(std::string_view s) { return tv(std::string(s)); }
}  // namespace

std::string to_toml(const RunConfig& c) {
    std::ostringstream os;
    auto kv = [&os](const char* k, const std::string& v) { os << k << " = " << v << "\n"; };
    os << "[run]\n";
    kv("name", tv(c.name));
    kv("description", tv(c.description));
    kv("t_end_yr", tv(c.t_end_yr));
    os << "\n[particle]\n";
    kv("a_init_cm", tv(c.grain.a_init_cm));
    kv("rho_d", tv(c.grain.rho_d));
    kv("r_init_au", tv(c.r_init_au));
    kv("z_init_H", tv(c.z_init_H));
    kv("vr_init_cms", tv(c.vr_init_cms));
    kv("vz_init_cms", tv(c.vz_init_cms));
    os << "\n[physics]\n";
    kv("drag", tv(to_string(c.grain.drag)));
    kv("coagulation", tv(c.grain.coagulation));
    kv("sedimentation", tv(c.grain.sedimentation));
    kv("dust_to_gas", tv(c.grain.dust_to_gas));
    kv("sigma_mol_cm2", tv(c.grain.sigma_mol_cm2));
    kv("mu", tv(c.grain.mu));
    kv("r_in_au", tv(c.r_in_au));
    kv("settle_eps", tv(c.settle_eps));
    os << "\n[disk]\n";
    kv("model", tv(c.beta_table_path.empty() ? "analytic" : "tabulated_beta"));
    if (!c.beta_table_path.empty()) kv("beta_table", tv(c.beta_table_path));
    kv("alpha", tv(c.disk.alpha));
    kv("mdot", tv(c.disk.mdot));
    kv("mstar", tv(c.disk.mstar));
    kv("vphi2vk", tv(c.disk.vphi2vk));
    kv("T1", tv(c.disk.T1));
    kv("rho1", tv(c.disk.rho1));
    kv("ur1", tv(c.disk.ur1));
    kv("H1", tv(c.disk.H1));
    os << "\n[solver]\n";
    kv("method", tv(ode::to_string(c.solver.method)));
    kv("rtol", tv(c.solver.rtol));
    std::string atol = "[";
    for (std::size_t i = 0; i < c.solver.atol.size(); ++i) atol += (i ? ", " : "") + tv(c.solver.atol[i]);
    kv("atol", atol + "]");
    if (std::isfinite(c.solver.h_max)) kv("h_max", tv(c.solver.h_max));
    kv("max_steps", std::to_string(c.solver.max_steps));
    kv("linear_solver", tv(ode::to_string(c.solver.linear_solver)));
    kv("bdf_max_order", std::to_string(c.solver.bdf_max_order));
    if (c.solver.rk4_step > 0) kv("rk4_step", tv(c.solver.rk4_step));
    os << "\n[output]\n";
    kv("save_steps", tv(c.save_steps));
    return os.str();
}

void save_config(const RunConfig& cfg, const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write config '" + path + "'");
    out << to_toml(cfg);
}

}  // namespace dpd::dust
