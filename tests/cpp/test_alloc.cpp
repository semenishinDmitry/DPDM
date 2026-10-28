// Allocation audit: the number of heap allocations performed by integrate()
// must not grow with the number of steps (no allocation in the hot loop).
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>

#include "dpd/dust/model.hpp"
#include "dpd/ode/integrate.hpp"
#include "dpd/ode/test_problems.hpp"

static std::atomic<long> g_allocs{0};

void* operator new(std::size_t n) {
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

using namespace dpd::ode;

namespace {
int failures = 0;

long count(const OdeSystem& sys, std::span<const Real> y0, Real t_end, Options o, std::uint64_t* steps) {
    o.save_steps = false;
    const long before = g_allocs.load();
    const Result r = integrate(sys, 0, y0, t_end, o);
    const long n = g_allocs.load() - before;
    *steps = r.stats.n_steps;
    if (!r.ok()) {
        std::printf("  integration failed: %s\n", std::string(to_string(r.status)).c_str());
        ++failures;
    }
    return n;
}

void audit(const char* name, const OdeSystem& sys, std::span<const Real> y0, Real t1, Real t2, const Options& o) {
    std::uint64_t s1 = 0, s2 = 0;
    const long a1 = count(sys, y0, t1, o, &s1);
    const long a2 = count(sys, y0, t2, o, &s2);
    const bool ok = a1 == a2 && s2 >= s1 + 200;  // hundreds more steps, same allocations
    std::printf("%-44s steps %8llu -> %8llu   allocations %4ld -> %4ld   %s\n", name, (unsigned long long)s1,
                (unsigned long long)s2, a1, a2, ok ? "OK" : "FAIL");
    if (!ok) ++failures;
}
}  // namespace

int main() {
    Options o;
    o.rtol = 1e-8;
    o.atol = {1e-10};
    o.method = Method::Dopri5;
    problems::HarmonicOscillator ho(1.0);
    const std::vector<Real> y_ho{1, 0};
    audit("dopri5 / harmonic oscillator", ho, y_ho, 10, 1000, o);

    o.method = Method::Bdf;
    problems::Robertson rob;
    const std::vector<Real> y_rob{1, 0, 0};
    audit("bdf+LU / Robertson", rob, y_rob, 1e2, 1e6, o);

    problems::ReactionDiffusion1D rd(200, 1.0, 1.0);
    const auto u0 = rd.initial_condition();
    o.linear_solver = LinearSolverKind::Gmres;
    audit("bdf+GMRES / reaction-diffusion N=200", rd, u0, 0.01, 1.0, o);
    o.linear_solver = LinearSolverKind::DenseLU;

    o.method = Method::Auto;
    dpd::dust::GrainParams g;
    g.a_init_cm = 1.0;
    dpd::dust::DustModel dm(dpd::dust::DiskParams{}, g);
    const auto yd = dm.initial_state(10, 0.5, 0, 0);
    o.atol = {1e-12, 1e-16, 1e-12, 1e-12, 1e-16, 1e-12};
    audit("auto / dust model (1 cm, 10 au)", dm, yd, 2e3, 2e5, o);  // both runs include the explicit->BDF switch

    std::printf(failures ? "ALLOCATION AUDIT FAILED\n" : "allocation audit passed\n");
    return failures ? 1 : 0;
}
