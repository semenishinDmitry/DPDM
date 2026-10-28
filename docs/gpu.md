# GPU experiment (Metal / CUDA)

**Status: experimental, not part of the library.** Code in `experiments/gpu/`, built with
`-DDPD_BUILD_GPU_EXPERIMENTS=ON`. Raw output: `docs/gpu_experiment_results.md`.

## Design

One trajectory per GPU thread. The integrator (adaptive DOPRI5 + radial drift of the
dust model, i.e. `DustModel` with `sedimentation = false`) is a single header,
`experiments/gpu/ensemble_kernel.h`, compiled unchanged as C++ (float and double), as
Metal Shading Language and as CUDA. This makes the comparison apples to apples.

Workload: 100 to 100 000 grains, `a` log-uniform 10 cm - 1 m, `r_init` log-uniform
1-10 au, integrated until `r = 0.05 au`. Error: median relative error of `t_drift` vs the
production library at rtol = 1e-10 (200-trajectory subset).

## Results (Apple M5: 10 CPU threads vs 10-core GPU, Metal 4)

| N | CPU library fp64, rtol 1e-6 (err 1.2e-5) | Metal fp32, rtol 1e-4 (err 1.5e-5) | GPU vs CPU |
|---|---|---|---|
| 100 | 16 ms | 148 ms | 9x **slower** |
| 1 000 | 171 ms | 163 ms | 1.05x |
| 10 000 | 1.68 s | 0.61 s | 2.7x faster |
| 100 000 | 17.2 s | 5.3 s | 3.2x faster |

Same integrator on both sides (N = 100 000): CPU fp32 35.7 s, CPU fp64 25.7 s,
Metal fp32 5.3 s (6.7x vs CPU fp32).

## Findings

1. **Single trajectory: no speedup.** The ODE has 3-6 components and its steps are
   sequential; a GPU thread is ~10x slower than a CPU core, and every kernel launch costs
   ~0.1 ms. The production runs take 0.3-15 ms on the CPU anyway.
2. **Ensembles: up to ~3x** at equal accuracy, starting from ~1 000 trajectories.
   Throughput saturates at ~1.9e4 trajectories/s; thread divergence (trajectories
   differ in step count by orders of magnitude) limits GPU efficiency.
3. **Precision is the hard limit on Apple GPUs: no FP64 in Metal.** In float32 the
   error stalls at ~1.5e-5 whatever the tolerance (rtol 1e-6 is 4x slower, not more
   accurate): `j/r - u_phi` is a difference of nearly equal numbers (`beta ~ 2.6e-3`), and
   time needs Kahan summation (`ulp(3e4) ~ 2e-3` in float). The CPU library reaches 1e-10
   in double; the GPU cannot.
4. **Stiff grains (1 um - 1 mm) are not covered**: the GPU kernel is explicit. A
   per-thread BDF with a small dense LU is feasible but not implemented.
5. **CUDA** (`cuda_ensemble_bench.cu`, float and double) is **not executed**: no NVIDIA
   GPU on the development machine. CI only checks that it compiles. On data-centre GPUs
   (full-rate FP64) it should remove limitation 3; consumer GPUs run FP64 at 1/32-1/64
   of the FP32 rate.

## Recommendation

Not worth integrating into the library for this model: the realistic workloads
(single runs, sweeps of 10-1000 cases) take milliseconds to seconds on the CPU, where
the GPU gives no gain or is slower, and fp32 cannot reach the accuracy the paper's
comparisons need. Worth revisiting only for very large ensembles (>= 1e5 trajectories,
e.g. population synthesis) on hardware with fast FP64 (CUDA, data-centre GPUs).
