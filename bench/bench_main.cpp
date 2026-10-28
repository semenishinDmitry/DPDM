// Benchmarks: kernels, RHS throughput, complete integrations, linear-solver
// crossover and ensembles. Output is Markdown. Usage: dpd_bench [--quick]
//
// Timing: steady_clock, best of R repetitions after one warm-up run. Accuracy
// is always reported next to time: performance alone is not evidence of correctness.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>

#include "dpd/dust/run.hpp"
#include "dpd/ode/integrate.hpp"
#include "dpd/ode/kernels.hpp"
#include "dpd/ode/test_problems.hpp"

using namespace dpd;
using namespace dpd::ode;
using clk = std::chrono::steady_clock;

namespace {
bool g_quick = false;

double best_of(int reps, const std::function<void()>& f) {
    f();  // warm-up
    double best = 1e300;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = clk::now();
        f();
        best = std::min(best, std::chrono::duration<double>(clk::now() - t0).count());
    }
    return best;
}

volatile double g_sink = 0;

void bench_kernels() {
    std::printf("\n## Vector kernels (ns per call, best of 7)\n\n");
    std::printf("| n | backend | axpy | lincomb(m=6) | wrms_err | dot |\n|---|---|---|---|---|---|\n");
    std::mt19937_64 g(3);
    std::uniform_real_distribution<double> d(0.5, 1.5);
    for (std::size_t n : {6u, 64u, 1024u, 65536u}) {
        std::vector<double> x(n), y(n), e(n), a(n), b(n), at(n, 1e-8);
        std::vector<std::vector<double>> K(6, std::vector<double>(n));
        for (auto* v : {&x, &y, &e, &a, &b}) for (auto& q : *v) q = d(g);
        for (auto& v : K) for (auto& q : v) q = d(g);
        const double* Kp[6] = {K[0].data(), K[1].data(), K[2].data(), K[3].data(), K[4].data(), K[5].data()};
        const double c[6] = {0.1, 0.2, 0.3, -0.1, -0.2, 0.05};
        const long iters = std::max<long>(20, static_cast<long>((g_quick ? 2e6 : 2e7) / static_cast<double>(n)));
        for (auto bk : kernels::available_backends()) {
            const auto* t = kernels::table(bk);
            auto per = [&](const std::function<void()>& f) { return best_of(7, [&] { for (long i = 0; i < iters; ++i) f(); }) / static_cast<double>(iters) * 1e9; };
            const double t_axpy = per([&] { t->axpy(1e-9, x.data(), y.data(), n); });
            const double t_lin = per([&] { t->lincomb(y.data(), x.data(), c, Kp, 6, n); });
            const double t_wrms = per([&] { g_sink = g_sink + t->wrms_err(e.data(), a.data(), b.data(), at.data(), true, 1e-6, n); });
            const double t_dot = per([&] { g_sink = g_sink + t->dot(x.data(), y.data(), n); });
            std::printf("| %zu | %s | %.1f | %.1f | %.1f | %.1f |\n", n, std::string(kernels::to_string(bk)).c_str(),
                        t_axpy, t_lin, t_wrms, t_dot);
        }
    }
}

void bench_rhs() {
    std::printf("\n## Dust-model RHS evaluation\n\n");
    dust::GrainParams gp;
    gp.coagulation = true;
    dust::DustModel m(dust::DiskParams{}, gp);
    auto y = m.initial_state(10, 0.3, 0, 0);
    std::vector<double> f(6);
    const long n = g_quick ? 1'000'000 : 10'000'000;
    const double t = best_of(5, [&] {
        for (long i = 0; i < n; ++i) {
            y[0] = 10 + 1e-9 * static_cast<double>(i & 1023);
            m.rhs(0, y, f);
            g_sink = g_sink + f[1];
        }
    });
    std::printf("RHS (6 components, sedimentation + coagulation): %.1f ns/eval (%.1f M evals/s)\n",
                t / static_cast<double>(n) * 1e9, static_cast<double>(n) / t / 1e6);
}

struct RunRow {
    std::string problem, method;
    double rtol, wall, err;
    Stats st;
    Status status;
};

void print_rows(const std::vector<RunRow>& rows) {
    std::printf("| problem | method | rtol | status | wall [ms] | rel. error | steps | rejected | RHS | Jac | LU | Newton it. | lin. it. | switches |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for (const auto& r : rows)
        std::printf("| %s | %s | %.0e | %s | %.3f | %.2e | %llu | %llu | %llu | %llu | %llu | %llu | %llu | %zu |\n",
                    r.problem.c_str(), r.method.c_str(), r.rtol, std::string(to_string(r.status)).c_str(), r.wall * 1e3,
                    r.err, (unsigned long long)r.st.n_steps,
                    (unsigned long long)(r.st.n_reject_error + r.st.n_reject_newton + r.st.n_reject_nonfinite),
                    (unsigned long long)r.st.n_rhs, (unsigned long long)r.st.n_jac, (unsigned long long)r.st.n_lu,
                    (unsigned long long)r.st.n_newton_iter, (unsigned long long)r.st.n_linear_iter, r.st.switches.size());
}

void bench_dust() {
    std::printf("\n## Dust model: complete integrations (error of t_drift vs BDF at rtol=1e-12)\n\n");
    std::vector<RunRow> rows;
    struct C { const char* name; double a, r; bool sed; };
    const C cases[] = {{"1 um from 10 au (stiff)", 1e-4, 10, true},
                       {"1 cm from 10 au", 1, 10, true},
                       {"1 m from 100 au (reference)", 100, 100, true}};
    for (const C& c : cases) {
        dust::RunConfig cfg;
        cfg.grain.a_init_cm = c.a;
        cfg.r_init_au = c.r;
        cfg.grain.sedimentation = c.sed;
        cfg.solver.method = Method::Bdf;
        cfg.solver.rtol = 1e-12;
        cfg.solver.atol = {1e-14, 1e-18, 1e-14, 1e-14, 1e-18, 1e-14};
        const double tref = dust::run(cfg, {}, false).summary.t_drift_yr;
        for (Method m : {Method::Auto, Method::Bdf, Method::Dopri5}) {
            for (double rtol : {1e-6, 1e-8, 1e-10}) {
                if (m == Method::Dopri5 && c.a < 1e-2) continue;  // explicit: ~1e11 steps, not runnable
                cfg.solver.method = m;
                cfg.solver.rtol = rtol;
                cfg.solver.atol = {1e-4 * rtol, 1e-8 * rtol, 1e-4 * rtol, 1e-4 * rtol, 1e-8 * rtol, 1e-4 * rtol};
                dust::RunResult res;
                const double w = best_of(g_quick ? 1 : 3, [&] { res = dust::run(cfg, {}, false); });
                rows.push_back({c.name, std::string(to_string(m)), rtol, w,
                                std::abs(res.summary.t_drift_yr / tref - 1), res.summary.stats, res.summary.status});
            }
        }
    }
    print_rows(rows);
}

template <class Exact>
void bench_problem(const char* name, const OdeSystem& sys, const std::vector<Real>& y0, Real T, Exact&& exact_err,
                   std::vector<Method> methods, std::vector<Real> rtols, std::vector<RunRow>& rows) {
    for (Method m : methods)
        for (Real rtol : rtols) {
            Options o;
            o.method = m;
            o.rtol = rtol;
            o.atol = {rtol * 1e-4};
            o.save_steps = false;
            o.max_steps = 20'000'000;
            Result r;
            const double w = best_of(g_quick ? 1 : 3, [&] { r = integrate(sys, 0, y0, T, o); });
            rows.push_back({name, std::string(to_string(m)), rtol, w, r.ok() ? exact_err(r.y_last) : NAN, r.stats, r.status});
        }
}

void bench_classic() {
    std::printf("\n## Classic problems: explicit vs implicit vs auto\n\n");
    std::vector<RunRow> rows;
    problems::HarmonicOscillator ho(1.0);
    bench_problem("harmonic oscillator, T=100", ho, {1, 0}, 100.0,
                  [](std::span<const Real> y) { return std::abs(y[0] - std::cos(100.0)); },
                  {Method::Dopri5, Method::Bdf, Method::Auto}, {1e-6, 1e-10}, rows);
    problems::VanDerPol vdp(1000);
    // reference: BDF at rtol 1e-12
    Options ro;
    ro.method = Method::Bdf;
    ro.rtol = 1e-12;
    ro.atol = {1e-14};
    ro.save_steps = false;
    const auto vref = integrate(vdp, 0, std::vector<Real>{2, 0}, 3000, ro).y_last;
    bench_problem("Van der Pol mu=1000, T=3000", vdp, {2, 0}, 3000.0,
                  [&](std::span<const Real> y) { return std::abs(y[0] - vref[0]) / std::abs(vref[0]); },
                  {Method::Bdf, Method::Auto}, {1e-6, 1e-9}, rows);
    problems::Robertson rob;
    ro.atol = {1e-16, 1e-20, 1e-16};
    const auto rref = integrate(rob, 0, std::vector<Real>{1, 0, 0}, 4e6, ro).y_last;
    bench_problem("Robertson, T=4e6", rob, {1, 0, 0}, 4e6,
                  [&](std::span<const Real> y) { return std::abs(y[0] - rref[0]) / rref[0]; },
                  {Method::Bdf, Method::Auto}, {1e-6, 1e-9}, rows);
    print_rows(rows);
    std::printf("\nDOPRI5 on Van der Pol mu=1000 / Robertson is omitted: it is stability-limited to ~1e7-1e10 steps.\n");
}

void bench_linear_solvers() {
    std::printf("\n## BDF linear solver crossover: 1-D reaction-diffusion (T=0.1, rtol=1e-6)\n\n");
    std::printf("| N | dense LU [ms] | GMRES [ms] | GMRES+tridiag precond [ms] | max |diff| LU vs GMRES+P | lin. it. (no P / P) |\n|---|---|---|---|---|---|\n");
    std::vector<Index> sizes = g_quick ? std::vector<Index>{50, 200, 800} : std::vector<Index>{25, 50, 100, 200, 400, 800, 1600};
    for (Index n : sizes) {
        problems::ReactionDiffusion1D sys(n, 1.0, 5.0);
        const auto u0 = sys.initial_condition();
        Options o;
        o.method = Method::Bdf;
        o.rtol = 1e-6;
        o.atol = {1e-9};
        o.save_steps = false;
        o.gmres_max_iter = 5000;
        Result rl, rk, rp;
        const double tl = best_of(g_quick ? 1 : 3, [&] { rl = integrate(sys, 0, u0, 0.1, o); });
        Options ok = o;
        ok.linear_solver = LinearSolverKind::Gmres;
        const double tk = best_of(g_quick ? 1 : 3, [&] { rk = integrate(sys, 0, u0, 0.1, ok); });
        problems::TridiagonalRDPreconditioner P(n, 1.0, 5.0);
        ok.preconditioner = &P;
        const double tp = best_of(g_quick ? 1 : 3, [&] { rp = integrate(sys, 0, u0, 0.1, ok); });
        double dmax = 0;
        for (Index i = 0; i < n; ++i) dmax = std::max(dmax, std::abs(rl.y_last[i] - rp.y_last[i]));
        std::printf("| %zu | %.2f | %.2f | %.2f | %.1e | %llu / %llu |\n", n, tl * 1e3, tk * 1e3, tp * 1e3, dmax,
                    (unsigned long long)rk.stats.n_linear_iter, (unsigned long long)rp.stats.n_linear_iter);
    }
}

void bench_ensemble() {
    std::printf("\n## Ensemble (parameter sweep): 64 grain sizes 1 um - 1 m from 10 au, sedimentation on\n\n");
    std::vector<dust::RunConfig> cfgs;
    const int N = g_quick ? 16 : 64;
    for (int i = 0; i < N; ++i) {
        dust::RunConfig c;
        c.grain.a_init_cm = 1e-4 * std::pow(1e6, i / double(N - 1));
        c.r_init_au = 10;
        cfgs.push_back(c);
    }
    std::printf("| threads | wall [s] | speedup |\n|---|---|---|\n");
    double t1 = 0;
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    for (unsigned th : {1u, 2u, 4u, 8u, hw}) {
        if (th > hw) continue;
        const double t = best_of(g_quick ? 1 : 2, [&] { (void)dust::run_ensemble(cfgs, th, false); });
        if (th == 1) t1 = t;
        std::printf("| %u | %.3f | %.2f |\n", th, t, t1 / t);
        if (th == hw) break;
    }
}
}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--quick")) g_quick = true;
    std::printf("# dpd benchmarks%s\n\nactive kernel backend: %s; hardware threads: %u\n", g_quick ? " (quick)" : "",
                std::string(kernels::to_string(kernels::active_backend())).c_str(), std::thread::hardware_concurrency());
    bench_kernels();
    bench_rhs();
    bench_dust();
    bench_classic();
    bench_linear_solvers();
    bench_ensemble();
    return 0;
}
