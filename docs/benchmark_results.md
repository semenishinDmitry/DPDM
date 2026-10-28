# Raw benchmark output (Apple M5, macOS, Apple clang 21, Release)

active kernel backend: neon; hardware threads: 10

## Vector kernels (ns per call, best of 7)

| n | backend | axpy | lincomb(m=6) | wrms_err | dot |
|---|---|---|---|---|---|
| 6 | scalar | 2.3 | 11.4 | 4.0 | 2.3 |
| 6 | generic | 2.3 | 10.9 | 3.5 | 2.1 |
| 6 | neon | 3.1 | 6.3 | 3.6 | 2.1 |
| 64 | scalar | 16.4 | 104.7 | 33.6 | 18.2 |
| 64 | generic | 7.5 | 45.2 | 27.2 | 9.0 |
| 64 | neon | 7.6 | 36.4 | 17.0 | 6.6 |
| 1024 | scalar | 245.7 | 1733.0 | 643.5 | 541.7 |
| 1024 | generic | 93.0 | 620.8 | 407.8 | 145.7 |
| 1024 | neon | 99.4 | 569.7 | 234.7 | 173.2 |
| 65536 | scalar | 15354.6 | 135362.3 | 42410.5 | 38163.4 |
| 65536 | generic | 10055.5 | 68893.7 | 26124.6 | 10489.5 |
| 65536 | neon | 9975.1 | 56055.2 | 16481.6 | 13918.6 |

## Dust-model RHS evaluation

RHS (6 components, sedimentation + coagulation): 17.4 ns/eval (57.3 M evals/s)

## Dust model: complete integrations (error of t_drift vs BDF at rtol=1e-12)

| problem | method | rtol | status | wall [ms] | rel. error | steps | rejected | RHS | Jac | LU | Newton it. | lin. it. | switches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 um from 10 au (stiff) | auto | 1e-06 | terminal_event | 0.315 | 7.90e-07 | 343 | 19 | 1905 | 73 | 145 | 1051 | 0 | 1 |
| 1 um from 10 au (stiff) | auto | 1e-08 | terminal_event | 0.603 | 1.61e-08 | 683 | 2 | 3587 | 129 | 243 | 2126 | 0 | 1 |
| 1 um from 10 au (stiff) | auto | 1e-10 | terminal_event | 1.301 | 1.97e-10 | 1521 | 2 | 7177 | 174 | 411 | 4876 | 0 | 1 |
| 1 um from 10 au (stiff) | bdf | 1e-06 | terminal_event | 0.357 | 8.82e-07 | 393 | 19 | 1836 | 74 | 171 | 1316 | 0 | 0 |
| 1 um from 10 au (stiff) | bdf | 1e-08 | terminal_event | 0.685 | 2.04e-08 | 767 | 0 | 3540 | 131 | 284 | 2621 | 0 | 0 |
| 1 um from 10 au (stiff) | bdf | 1e-10 | terminal_event | 1.484 | 1.93e-10 | 1679 | 1 | 7082 | 176 | 488 | 5848 | 0 | 0 |
| 1 cm from 10 au | auto | 1e-06 | terminal_event | 0.624 | 4.80e-06 | 685 | 190 | 4099 | 189 | 500 | 1992 | 0 | 1 |
| 1 cm from 10 au | auto | 1e-08 | terminal_event | 1.087 | 6.79e-08 | 1315 | 158 | 8046 | 310 | 637 | 3238 | 0 | 1 |
| 1 cm from 10 au | auto | 1e-10 | terminal_event | 2.062 | 1.38e-09 | 3431 | 5 | 21173 | 436 | 620 | 3238 | 0 | 1 |
| 1 cm from 10 au | bdf | 1e-06 | terminal_event | 0.667 | 4.31e-06 | 706 | 160 | 3679 | 179 | 497 | 2424 | 0 | 0 |
| 1 cm from 10 au | bdf | 1e-08 | terminal_event | 1.197 | 6.97e-08 | 1245 | 154 | 6728 | 315 | 711 | 4521 | 0 | 0 |
| 1 cm from 10 au | bdf | 1e-10 | terminal_event | 1.626 | 1.34e-09 | 1887 | 1 | 9118 | 444 | 787 | 6008 | 0 | 0 |
| 1 cm from 10 au | dopri5 | 1e-06 | terminal_event | 1591.343 | 2.15e-09 | 3440418 | 77578 | 21108003 | 0 | 0 | 0 | 0 | 0 |
| 1 cm from 10 au | dopri5 | 1e-08 | terminal_event | 1650.216 | 2.26e-09 | 3503150 | 0 | 21018987 | 0 | 0 | 0 | 0 | 0 |
| 1 cm from 10 au | dopri5 | 1e-10 | terminal_event | 1911.187 | 2.26e-09 | 4068937 | 0 | 24414104 | 0 | 0 | 0 | 0 | 0 |
| 1 m from 100 au (reference) | auto | 1e-06 | terminal_event | 4.146 | 5.53e-06 | 5741 | 1110 | 27545 | 784 | 3133 | 13956 | 0 | 1 |
| 1 m from 100 au (reference) | auto | 1e-08 | terminal_event | 7.264 | 4.06e-08 | 10671 | 1153 | 47817 | 720 | 3401 | 22179 | 0 | 1 |
| 1 m from 100 au (reference) | auto | 1e-10 | terminal_event | 14.764 | 4.37e-11 | 22326 | 1213 | 101696 | 1240 | 5069 | 40912 | 0 | 1 |
| 1 m from 100 au (reference) | bdf | 1e-06 | terminal_event | 5.758 | 6.67e-06 | 7769 | 1319 | 27160 | 813 | 4098 | 21467 | 0 | 0 |
| 1 m from 100 au (reference) | bdf | 1e-08 | terminal_event | 10.423 | 4.13e-08 | 14254 | 1444 | 44177 | 885 | 5123 | 37980 | 0 | 0 |
| 1 m from 100 au (reference) | bdf | 1e-10 | terminal_event | 19.505 | 9.79e-11 | 27190 | 1277 | 78686 | 1319 | 7501 | 69451 | 0 | 0 |
| 1 m from 100 au (reference) | dopri5 | 1e-06 | terminal_event | 4.889 | 2.13e-08 | 10649 | 72 | 64578 | 0 | 0 | 0 | 0 | 0 |
| 1 m from 100 au (reference) | dopri5 | 1e-08 | terminal_event | 11.775 | 1.71e-10 | 25547 | 175 | 154979 | 0 | 0 | 0 | 0 | 0 |
| 1 m from 100 au (reference) | dopri5 | 1e-10 | terminal_event | 29.154 | 2.47e-11 | 62699 | 184 | 378965 | 0 | 0 | 0 | 0 | 0 |

## Classic problems: explicit vs implicit vs auto

| problem | method | rtol | status | wall [ms] | rel. error | steps | rejected | RHS | Jac | LU | Newton it. | lin. it. | switches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| harmonic oscillator, T=100 | dopri5 | 1e-06 | success | 0.051 | 5.95e-06 | 515 | 63 | 3570 | 0 | 0 | 0 | 0 | 0 |
| harmonic oscillator, T=100 | dopri5 | 1e-10 | success | 0.302 | 7.35e-10 | 3390 | 54 | 21341 | 0 | 0 | 0 | 0 | 0 |
| harmonic oscillator, T=100 | bdf | 1e-06 | success | 0.199 | 1.38e-04 | 1079 | 81 | 2325 | 1 | 228 | 2320 | 0 | 0 |
| harmonic oscillator, T=100 | bdf | 1e-10 | success | 0.741 | 9.78e-08 | 4352 | 40 | 8789 | 1 | 768 | 8784 | 0 | 0 |
| harmonic oscillator, T=100 | auto | 1e-06 | success | 0.050 | 5.95e-06 | 515 | 63 | 3570 | 0 | 0 | 0 | 0 | 0 |
| harmonic oscillator, T=100 | auto | 1e-10 | success | 0.302 | 7.35e-10 | 3390 | 54 | 21341 | 0 | 0 | 0 | 0 | 0 |
| Van der Pol mu=1000, T=3000 | bdf | 1e-06 | success | 266.122 | 2.90e-03 | 1280912 | 41657 | 3783100 | 39283 | 276200 | 3783098 | 0 | 0 |
| Van der Pol mu=1000, T=3000 | bdf | 1e-09 | success | 849.428 | 5.31e-06 | 4159320 | 22423 | 13161790 | 55111 | 762461 | 13161788 | 0 | 0 |
| Van der Pol mu=1000, T=3000 | auto | 1e-06 | success | 630.127 | 2.90e-03 | 1280891 | 41662 | 3783403 | 39280 | 276175 | 3782952 | 0 | 1 |
| Van der Pol mu=1000, T=3000 | auto | 1e-09 | success | 1752.638 | 5.32e-06 | 4159433 | 22407 | 13163314 | 55103 | 762433 | 13162375 | 0 | 1 |
| Robertson, T=4e6 | bdf | 1e-06 | success | 0.136 | 5.33e-06 | 459 | 1 | 1293 | 11 | 120 | 1291 | 0 | 0 |
| Robertson, T=4e6 | bdf | 1e-09 | success | 0.391 | 2.14e-08 | 1277 | 1 | 4130 | 17 | 264 | 4128 | 0 | 0 |
| Robertson, T=4e6 | auto | 1e-06 | success | 0.124 | 4.99e-06 | 427 | 6 | 1395 | 10 | 100 | 1105 | 0 | 1 |
| Robertson, T=4e6 | auto | 1e-09 | success | 0.353 | 2.15e-08 | 1211 | 8 | 4373 | 16 | 221 | 3577 | 0 | 1 |

DOPRI5 on Van der Pol mu=1000 / Robertson is omitted: it is stability-limited to ~1e7-1e10 steps.

## BDF linear solver crossover: 1-D reaction-diffusion (T=0.1, rtol=1e-6)

| N | dense LU [ms] | GMRES [ms] | GMRES+tridiag precond [ms] | max |diff| LU vs GMRES+P | lin. it. (no P / P) |
|---|---|---|---|---|---|
| 25 | 0.07 | 0.07 | 0.06 | 7.9e-10 | 260 / 102 |
| 50 | 0.22 | 0.15 | 0.10 | 6.6e-10 | 436 / 110 |
| 100 | 0.84 | 0.47 | 0.17 | 7.9e-10 | 810 / 110 |
| 200 | 3.22 | 2.46 | 0.36 | 5.8e-10 | 1562 / 114 |
| 400 | 13.86 | 13.16 | 0.92 | 9.9e-10 | 3384 / 114 |
| 800 | 57.99 | 71.03 | 1.45 | 9.8e-10 | 7952 / 114 |
| 1600 | 250.15 | 365.62 | 2.90 | 1.0e-09 | 20652 / 114 |

## Ensemble (parameter sweep): 64 grain sizes 1 um - 1 m from 10 au, sedimentation on

| threads | wall [s] | speedup |
|---|---|---|
| 1 | 0.096 | 1.00 |
| 2 | 0.052 | 1.86 |
| 4 | 0.028 | 3.44 |
| 8 | 0.019 | 5.00 |
| 10 | 0.018 | 5.36 |
