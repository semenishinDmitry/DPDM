// Ensemble benchmark: Metal GPU (float32) vs CPU (same integrator in float32 and
// float64, std::thread) vs the production dpd::dust library (auto DOPRI5/BDF, float64).
// EXPERIMENTAL. Build: cmake -DDPD_BUILD_GPU_EXPERIMENTS=ON (macOS only).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "dpd/dust/constants.hpp"
#include "dpd/dust/run.hpp"
#include "ensemble_kernel.h"

using clk = std::chrono::steady_clock;
namespace cgs = dpd::dust::cgs;

namespace {

template <class T>
EnsParams<T> make_params(T rtol) {
    const auto u = dpd::dust::make_units(1.0);
    const dpd::dust::DiskParams d;   // paper defaults
    const dpd::dust::GrainParams g;  // paper defaults (mu 2.3, sigma 1e-16, rho_d 3)
    EnsParams<T> p{};
    p.cT = T(d.T1);
    p.cRho = T(d.rho1);
    p.cUr = T(d.ur1 / u.v0_cms);
    p.uphi_fac = T(d.vphi2vk);
    p.vth_coef = T(std::sqrt(8 * cgs::k_B / (cgs::pi * g.mu * cgs::m_p)));
    p.lam_coef = T(g.mu * cgs::m_p / g.sigma_mol_cm2);
    p.t0 = T(u.t0_s);
    p.rho_d = T(g.rho_d);
    p.r_in = T(0.05);
    p.rtol = rtol;
    p.atol_r = rtol * T(1e-3);
    p.atol_v = rtol * T(1e-7);
    p.atol_j = rtol * T(1e-3);
    p.h_max = T(1e30);
    p.max_steps = 5'000'000;
    p.drag_both = 1;
    return p;
}

template <class T>
double cpu_run(const EnsParams<T>& p, const std::vector<double>& a, const std::vector<double>& r,
               std::vector<double>& t_out, unsigned threads) {
    const std::size_t n = a.size();
    t_out.assign(n, NAN);
    std::atomic<std::size_t> next{0};
    const auto t0 = clk::now();
    std::vector<std::thread> pool;
    for (unsigned k = 0; k < threads; ++k)
        pool.emplace_back([&] {
            for (std::size_t i; (i = next.fetch_add(1)) < n;) {
                const auto res = ens_integrate<T>(p, T(a[i]), T(r[i]));
                t_out[i] = res.status == 0 ? double(res.t_drift) : NAN;
            }
        });
    for (auto& th : pool) th.join();
    return std::chrono::duration<double>(clk::now() - t0).count();
}

struct Gpu {
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
    id<MTLComputePipelineState> pso;
    double compile_s;
};

Gpu make_gpu(const std::string& header_path) {
    Gpu g{};
    g.dev = MTLCreateSystemDefaultDevice();
    if (!g.dev) throw std::runtime_error("no Metal device");
    std::ifstream f(header_path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string src = ss.str();
    src += R"(
kernel void ens(device const EnsParams<float>* p [[buffer(0)]], device const float* a [[buffer(1)]],
                device const float* r0 [[buffer(2)]], device EnsResult<float>* out [[buffer(3)]],
                constant uint& n [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    EnsParams<float> pp = *p;
    out[gid] = ens_integrate<float>(pp, a[gid], r0[gid]);
}
)";
    const auto t0 = clk::now();
    MTLCompileOptions* opt = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) {
        opt.mathMode = MTLMathModeSafe;  // no fast-math: IEEE behaviour as on the CPU
    }
    NSError* err = nil;
    id<MTLLibrary> lib = [g.dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:opt error:&err];
    if (!lib) throw std::runtime_error(std::string("MSL compile failed: ") + err.localizedDescription.UTF8String);
    id<MTLFunction> fn = [lib newFunctionWithName:@"ens"];
    g.pso = [g.dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g.pso) throw std::runtime_error("pipeline creation failed");
    g.q = [g.dev newCommandQueue];
    g.compile_s = std::chrono::duration<double>(clk::now() - t0).count();
    return g;
}

// Returns {kernel time (GPU timestamps), total wall incl. upload/download}.
std::pair<double, double> gpu_run(Gpu& g, const EnsParams<float>& p, const std::vector<double>& a,
                                  const std::vector<double>& r, std::vector<double>& t_out) {
    const uint32_t n = static_cast<uint32_t>(a.size());
    const auto w0 = clk::now();
    id<MTLBuffer> ba = [g.dev newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> br = [g.dev newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo = [g.dev newBufferWithLength:n * sizeof(EnsResult<float>) options:MTLResourceStorageModeShared];
    float* pa = static_cast<float*>(ba.contents);
    float* pr = static_cast<float*>(br.contents);
    for (uint32_t i = 0; i < n; ++i) {
        pa[i] = float(a[i]);
        pr[i] = float(r[i]);
    }
    id<MTLCommandBuffer> cb = [g.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g.pso];
    [enc setBytes:&p length:sizeof(p) atIndex:0];
    [enc setBuffer:ba offset:0 atIndex:1];
    [enc setBuffer:br offset:0 atIndex:2];
    [enc setBuffer:bo offset:0 atIndex:3];
    [enc setBytes:&n length:sizeof(n) atIndex:4];
    const NSUInteger tg = std::min<NSUInteger>(64, g.pso.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) throw std::runtime_error(std::string("GPU error: ") + cb.error.localizedDescription.UTF8String);
    const double kernel = cb.GPUEndTime - cb.GPUStartTime;
    const auto* out = static_cast<const EnsResult<float>*>(bo.contents);
    t_out.resize(n);
    for (uint32_t i = 0; i < n; ++i) t_out[i] = out[i].status == 0 ? double(out[i].t_drift) : NAN;
    return {kernel, std::chrono::duration<double>(clk::now() - w0).count()};
}

struct ErrStats {
    double median, max;
    int failed;
};
ErrStats errors(const std::vector<double>& t, const std::vector<std::size_t>& idx, const std::vector<double>& ref) {
    std::vector<double> e;
    int failed = 0;
    for (std::size_t k = 0; k < idx.size(); ++k) {
        const double v = t[idx[k]];
        if (!std::isfinite(v)) { ++failed; continue; }
        e.push_back(std::abs(v / ref[k] - 1));
    }
    std::sort(e.begin(), e.end());
    return {e.empty() ? NAN : e[e.size() / 2], e.empty() ? NAN : e.back(), failed};
}

}  // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::size_t> sizes = {100, 1000, 10000, 100000};
        if (argc > 1 && std::string(argv[1]) == "--quick") sizes = {100, 1000, 10000};
        Gpu gpu = make_gpu(DPD_GPU_HEADER);
        std::printf("# GPU ensemble experiment\n\nGPU: %s; CPU threads: %u; MSL compile: %.0f ms\n",
                    gpu.dev.name.UTF8String, threads, gpu.compile_s * 1e3);
        std::printf("Workload: radial drift (no sedimentation/coagulation), a log-uniform 10 cm - 1 m, "
                    "r_init log-uniform 1-10 au, until r = 0.05 au.\n\n");
        // Deterministic sample
        const std::size_t nmax = sizes.back();
        std::vector<double> a(nmax), r(nmax);
        uint64_t s = 12345;
        auto rnd = [&] { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return double(s >> 11) / double(1ULL << 53); };
        for (std::size_t i = 0; i < nmax; ++i) {
            a[i] = 10.0 * std::pow(10.0, rnd());
            r[i] = std::pow(10.0, rnd());
        }
        // Reference: production library, float64, rtol 1e-10, on a 200-trajectory subset.
        std::vector<std::size_t> idx;
        for (std::size_t i = 0; i < 200; ++i) idx.push_back(i * 37 % 100);
        std::vector<dpd::dust::RunConfig> rc;
        for (std::size_t i : idx) {
            dpd::dust::RunConfig c;
            c.grain.a_init_cm = a[i];
            c.r_init_au = r[i];
            c.grain.sedimentation = false;
            c.solver.rtol = 1e-10;
            c.solver.atol = {1e-14, 1e-18, 1e-14, 1e-14, 1e-18, 1e-14};
            rc.push_back(c);
        }
        std::vector<double> ref;
        for (auto& res : dpd::dust::run_ensemble(rc, threads, false)) ref.push_back(res.summary.t_drift_yr);
        const double t0_yr = dpd::dust::make_units(1.0).t0_yr();

        std::printf("| N | variant | precision | rtol | wall [ms] | traj/s | median rel. err | max rel. err | failed |\n");
        std::printf("|---|---|---|---|---|---|---|---|---|\n");
        auto line = [&](std::size_t n, const char* v, const char* prec, double rtol, double w, std::vector<double> t,
                        bool code_units) {
            if (code_units) for (auto& x : t) x *= t0_yr;
            const auto e = errors(t, idx, ref);
            std::printf("| %zu | %s | %s | %.0e | %.2f | %.3g | %.1e | %.1e | %d |\n", n, v, prec, rtol, w * 1e3,
                        double(n) / w, e.median, e.max, e.failed);
            std::fflush(stdout);
        };
        for (std::size_t n : sizes) {
            std::vector<double> an(a.begin(), a.begin() + long(n)), rn(r.begin(), r.begin() + long(n)), t;
            // production library at several tolerances (match accuracy, not rtol)
            for (double rt : {1e-4, 1e-5, 1e-6}) {
                std::vector<dpd::dust::RunConfig> cfgs(n);
                for (std::size_t i = 0; i < n; ++i) {
                    cfgs[i].grain.a_init_cm = an[i];
                    cfgs[i].r_init_au = rn[i];
                    cfgs[i].grain.sedimentation = false;
                    cfgs[i].solver.rtol = rt;
                    cfgs[i].solver.atol = {rt * 1e-3, rt * 1e-7, rt * 1e-3, rt * 1e-3, rt * 1e-7, rt * 1e-3};
                }
                const auto w0 = clk::now();
                const auto rr = dpd::dust::run_ensemble(cfgs, threads, false);
                const double w = std::chrono::duration<double>(clk::now() - w0).count();
                t.resize(n);
                for (std::size_t i = 0; i < n; ++i) t[i] = rr[i].summary.t_drift_yr;
                line(n, "CPU library (auto DOPRI5/BDF)", "fp64", rt, w, t, false);
            }
            double w = cpu_run<double>(make_params<double>(1e-6), an, rn, t, threads);
            line(n, "CPU same kernel", "fp64", 1e-6, w, t, true);
            w = cpu_run<float>(make_params<float>(1e-4f), an, rn, t, threads);
            line(n, "CPU same kernel", "fp32", 1e-4, w, t, true);
            for (float rt : {1e-4f, 1e-6f}) {
                (void)gpu_run(gpu, make_params<float>(rt), an, rn, t);  // warm-up
                const auto [kern, total] = gpu_run(gpu, make_params<float>(rt), an, rn, t);
                line(n, "Metal GPU (kernel only)", "fp32", rt, kern, t, true);
                line(n, "Metal GPU (incl. transfers)", "fp32", rt, total, t, true);
            }
        }
    }
    return 0;
}
