# GPU ensemble experiment

GPU: Apple M5; CPU threads: 10; MSL compile: 1 ms
Workload: radial drift (no sedimentation/coagulation), a log-uniform 10 cm - 1 m, r_init log-uniform 1-10 au, until r = 0.05 au.

| N | variant | precision | rtol | wall [ms] | traj/s | median rel. err | max rel. err | failed |
|---|---|---|---|---|---|---|---|---|
| 100 | CPU library (auto DOPRI5/BDF) | fp64 | 1e-04 | 6.00 | 1.67e+04 | 1.0e-03 | 2.0e-03 | 0 |
| 100 | CPU library (auto DOPRI5/BDF) | fp64 | 1e-05 | 9.69 | 1.03e+04 | 1.2e-04 | 3.7e-04 | 0 |
| 100 | CPU library (auto DOPRI5/BDF) | fp64 | 1e-06 | 16.47 | 6.07e+03 | 1.2e-05 | 3.2e-05 | 0 |
| 100 | CPU same kernel | fp64 | 1e-06 | 25.28 | 3.96e+03 | 1.2e-10 | 1.5e-09 | 0 |
| 100 | CPU same kernel | fp32 | 1e-04 | 34.78 | 2.88e+03 | 1.7e-05 | 2.1e-05 | 0 |
| 100 | Metal GPU (kernel only) | fp32 | 1e-04 | 147.60 | 678 | 1.5e-05 | 1.9e-05 | 0 |
| 100 | Metal GPU (incl. transfers) | fp32 | 1e-04 | 147.83 | 676 | 1.5e-05 | 1.9e-05 | 0 |
| 100 | Metal GPU (kernel only) | fp32 | 1e-06 | 559.00 | 179 | 1.7e-05 | 3.5e-05 | 0 |
| 100 | Metal GPU (incl. transfers) | fp32 | 1e-06 | 559.25 | 179 | 1.7e-05 | 3.5e-05 | 0 |
| 1000 | CPU library (auto DOPRI5/BDF) | fp64 | 1e-04 | 57.27 | 1.75e+04 | 1.0e-03 | 2.0e-03 | 0 |
| 1000 | CPU library (auto DOPRI5/BDF) | fp64 | 1e-05 | 96.04 | 1.04e+04 | 1.2e-04 | 3.7e-04 | 0 |
| 1000 | CPU library (auto DOPRI5/BDF) | fp64 | 1e-06 | 171.23 | 5.84e+03 | 1.2e-05 | 3.2e-05 | 0 |
| 1000 | CPU same kernel | fp64 | 1e-06 | 243.10 | 4.11e+03 | 1.2e-10 | 1.5e-09 | 0 |
| 1000 | CPU same kernel | fp32 | 1e-04 | 336.14 | 2.97e+03 | 1.7e-05 | 2.1e-05 | 0 |
| 1000 | Metal GPU (kernel only) | fp32 | 1e-04 | 162.75 | 6.14e+03 | 1.5e-05 | 1.9e-05 | 0 |
| 1000 | Metal GPU (incl. transfers) | fp32 | 1e-04 | 163.01 | 6.13e+03 | 1.5e-05 | 1.9e-05 | 0 |
| 1000 | Metal GPU (kernel only) | fp32 | 1e-06 | 624.23 | 1.6e+03 | 1.7e-05 | 3.5e-05 | 0 |
| 1000 | Metal GPU (incl. transfers) | fp32 | 1e-06 | 624.54 | 1.6e+03 | 1.7e-05 | 3.5e-05 | 0 |
| 10000 | CPU library (auto DOPRI5/BDF) | fp64 | 1e-04 | 577.55 | 1.73e+04 | 1.0e-03 | 2.0e-03 | 0 |
| 10000 | CPU library (auto DOPRI5/BDF) | fp64 | 1e-05 | 972.01 | 1.03e+04 | 1.2e-04 | 3.7e-04 | 0 |
| 10000 | CPU library (auto DOPRI5/BDF) | fp64 | 1e-06 | 1683.93 | 5.94e+03 | 1.2e-05 | 3.2e-05 | 0 |
| 10000 | CPU same kernel | fp64 | 1e-06 | 2568.96 | 3.89e+03 | 1.2e-10 | 1.5e-09 | 0 |
| 10000 | CPU same kernel | fp32 | 1e-04 | 3576.35 | 2.8e+03 | 1.7e-05 | 2.1e-05 | 0 |
| 10000 | Metal GPU (kernel only) | fp32 | 1e-04 | 612.41 | 1.63e+04 | 1.5e-05 | 1.9e-05 | 0 |
| 10000 | Metal GPU (incl. transfers) | fp32 | 1e-04 | 612.73 | 1.63e+04 | 1.5e-05 | 1.9e-05 | 0 |
| 10000 | Metal GPU (kernel only) | fp32 | 1e-06 | 2323.11 | 4.3e+03 | 1.7e-05 | 3.5e-05 | 0 |
| 10000 | Metal GPU (incl. transfers) | fp32 | 1e-06 | 2323.70 | 4.3e+03 | 1.7e-05 | 3.5e-05 | 0 |
| 100000 | CPU library (auto DOPRI5/BDF) | fp64 | 1e-04 | 5828.59 | 1.72e+04 | 1.0e-03 | 2.0e-03 | 0 |
| 100000 | CPU library (auto DOPRI5/BDF) | fp64 | 1e-05 | 9654.42 | 1.04e+04 | 1.2e-04 | 3.7e-04 | 0 |
| 100000 | CPU library (auto DOPRI5/BDF) | fp64 | 1e-06 | 17172.32 | 5.82e+03 | 1.2e-05 | 3.2e-05 | 0 |
| 100000 | CPU same kernel | fp64 | 1e-06 | 25747.87 | 3.88e+03 | 1.2e-10 | 1.5e-09 | 0 |
| 100000 | CPU same kernel | fp32 | 1e-04 | 35720.69 | 2.8e+03 | 1.7e-05 | 2.1e-05 | 0 |
| 100000 | Metal GPU (kernel only) | fp32 | 1e-04 | 5296.08 | 1.89e+04 | 1.5e-05 | 1.9e-05 | 0 |
| 100000 | Metal GPU (incl. transfers) | fp32 | 1e-04 | 5297.32 | 1.89e+04 | 1.5e-05 | 1.9e-05 | 0 |
| 100000 | Metal GPU (kernel only) | fp32 | 1e-06 | 20209.18 | 4.95e+03 | 1.7e-05 | 3.5e-05 | 0 |
| 100000 | Metal GPU (incl. transfers) | fp32 | 1e-06 | 20210.73 | 4.95e+03 | 1.7e-05 | 3.5e-05 | 0 |
