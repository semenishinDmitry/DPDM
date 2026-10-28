// Ensemble benchmark on an NVIDIA GPU (float32 and float64) vs the same
// integrator on the host. EXPERIMENTAL and NOT RUN by the authors: the
// development machine has no NVIDIA GPU. CI only checks that it compiles.
// Build: cmake -DDPD_BUILD_GPU_EXPERIMENTS=ON (with a CUDA toolkit), or
//        nvcc -O3 -std=c++17 cuda_ensemble_bench.cu -o cuda_ensemble_bench
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "ensemble_kernel.h"

#define CUDA_CHECK(x)                                                                  \
    do {                                                                               \
        cudaError_t e_ = (x);                                                          \
        if (e_ != cudaSuccess) {                                                       \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                              \
        }                                                                              \
    } while (0)

namespace {
// Same constants as include/dpd/dust/constants.hpp and the paper defaults.
constexpr double kAu = 1.495978707e13, kGM = 1.3271244e26, kKB = 1.380649e-16, kMP = 1.67262192369e-24;
constexpr double kPi = 3.141592653589793, kYear = 3.15576e7;

template <class T>
EnsParams<T> make_params(T rtol) {
    const double v0 = std::sqrt(kGM / kAu), mu = 2.3, sigma = 1e-16;
    EnsParams<T> p{};
    p.cT = T(240.0);
    p.cRho = T(2.5e-10);
    p.cUr = T(-30.0 / v0);
    p.uphi_fac = T(0.9987);
    p.vth_coef = T(std::sqrt(8 * kKB / (kPi * mu * kMP)));
    p.lam_coef = T(mu * kMP / sigma);
    p.t0 = T(kAu / v0);
    p.rho_d = T(3.0);
    p.r_in = T(0.05);
    p.rtol = rtol;
    p.atol_r = rtol * T(1e-3);
    p.atol_v = rtol * T(1e-7);
    p.atol_j = rtol * T(1e-3);
    p.h_max = T(1e30);
    p.max_steps = 5000000;
    p.drag_both = 1;
    return p;
}

template <class T>
__global__ void ens_kernel(EnsParams<T> p, const T* a, const T* r0, EnsResult<T>* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ens_integrate<T>(p, a[i], r0[i]);
}

template <class T>
double run_gpu(const std::vector<double>& a, const std::vector<double>& r, T rtol, std::vector<double>& t) {
    const int n = static_cast<int>(a.size());
    std::vector<T> ah(a.begin(), a.end()), rh(r.begin(), r.end());
    T *ad, *rd;
    EnsResult<T>* od;
    CUDA_CHECK(cudaMalloc(&ad, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&rd, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&od, n * sizeof(EnsResult<T>)));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    CUDA_CHECK(cudaMemcpy(ad, ah.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(rd, rh.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    ens_kernel<T><<<(n + 127) / 128, 128>>>(make_params<T>(rtol), ad, rd, od, n);
    CUDA_CHECK(cudaGetLastError());
    std::vector<EnsResult<T>> oh(n);
    CUDA_CHECK(cudaMemcpy(oh.data(), od, n * sizeof(EnsResult<T>), cudaMemcpyDeviceToHost));
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    const double t0_yr = kAu / std::sqrt(kGM / kAu) / kYear;
    t.resize(n);
    for (int i = 0; i < n; ++i) t[i] = oh[i].status == 0 ? double(oh[i].t_drift) * t0_yr : NAN;
    cudaFree(ad);
    cudaFree(rd);
    cudaFree(od);
    return ms * 1e-3;
}
}  // namespace

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s\n\n| N | precision | rtol | wall incl. transfers [ms] | median rel. err vs host fp64 rtol 1e-10 |\n"
                "|---|---|---|---|---|\n", prop.name);
    const int nmax = 100000;
    std::vector<double> a(nmax), r(nmax);
    uint64_t s = 12345;
    auto rnd = [&] { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return double(s >> 11) / double(1ULL << 53); };
    for (int i = 0; i < nmax; ++i) {
        a[i] = 10.0 * std::pow(10.0, rnd());
        r[i] = std::pow(10.0, rnd());
    }
    // Host reference on a subset (same integrator, fp64, tight tolerance).
    const auto pref = make_params<double>(1e-10);
    const double t0_yr = kAu / std::sqrt(kGM / kAu) / kYear;
    std::vector<double> ref(100);
    for (int i = 0; i < 100; ++i) ref[i] = ens_integrate<double>(pref, a[i], r[i]).t_drift * t0_yr;
    for (int n : {1000, 10000, 100000}) {
        std::vector<double> an(a.begin(), a.begin() + n), rn(r.begin(), r.begin() + n), t;
        for (int prec = 0; prec < 2; ++prec) {
            const double w = prec ? run_gpu<double>(an, rn, 1e-6, t) : run_gpu<float>(an, rn, 1e-4f, t);
            std::vector<double> e;
            for (int i = 0; i < 100; ++i)
                if (std::isfinite(t[i])) e.push_back(std::fabs(t[i] / ref[i] - 1));
            std::sort(e.begin(), e.end());
            std::printf("| %d | %s | %g | %.2f | %.1e |\n", n, prec ? "fp64" : "fp32", prec ? 1e-6 : 1e-4, w * 1e3,
                        e.empty() ? NAN : e[e.size() / 2]);
        }
    }
    return 0;
}
