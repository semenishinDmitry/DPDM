"""Traps, vertical diffusion and charged grains through the Python API."""
import math

import numpy as np
import pytest

import dustppd as dp


def dead_zone_cfg():
    c = dp.RunConfig()
    c.disk_model = "self_consistent"
    c.perturbation = "dead_zone_edge"
    c.bump_amplitude, c.bump_r_au, c.bump_width_au = 2.0, 10.0, 1.0
    return c


def test_disk_profile_and_traps():
    c = dead_zone_cfg()
    prof = dp.disk_profile(c, np.geomspace(9, 13, 400))  # local pressure maximum near the edge
    p = prof["p_mid"]
    r_pmax = prof["r_au"][np.argmax(p)]
    (r0,) = dp.zero_headwind_radii(c, 1, 100)
    assert abs(r0 - r_pmax) < 0.05          # zero head wind at the pressure maximum (grid resolution)
    zs = dp.drift_zeros(c, 1.0, 1, 100)
    traps = [z["r_au"] for z in zs if z["stable"]]
    assert len(traps) == 1 and abs(traps[0] - r0) < 0.2
    amin = dp.min_trapped_size(c, 1, 100)
    assert 0.01 < amin < 1.0
    v = dp.stationary_vr(c, [traps[0] * 0.98, traps[0] * 1.02], 1.0)
    assert v[0] > 0 > v[1]                  # converging flow around the trap


def test_trap_run_matches_root():
    c = dead_zone_cfg()
    c.a_init_cm, c.r_init_au, c.sedimentation, c.t_end_yr = 1.0, 15.0, False, 3e6
    s = dp.run(c, keep_trajectory=False)["summary"]
    assert s["trapped"]
    trap = [z["r_au"] for z in dp.drift_zeros(c, 1.0, 1, 100) if z["stable"]][0]
    assert s["r_trap_au"] == pytest.approx(trap, rel=1e-6)


def test_vertical_diffusion_matches_exact_solution():
    c = dp.RunConfig()
    r = dp.vertical_diffusion(c, r_au=100.0, a_cm=0.1, alpha_turb=1e-4, n_paths=20000)
    assert r["status"] == "success"
    assert abs(r["Hd_simulated_au"] - r["Hd_exact_au"]) < 4 * r["Hd_stat_error_au"] + 0.01 * r["Hd_exact_au"]
    z = np.linspace(-5 * r["H_au"], 5 * r["H_au"], 2001)
    st = dp.stationary_dust_density(c, 100.0, 0.1, 1e-4, z)
    assert np.trapz(st["density"], z) == pytest.approx(1.0, rel=1e-9)
    assert st["Hd_exact_au"] == pytest.approx(r["Hd_exact_au"], rel=1e-12)
    # reproducible with the same seed
    r2 = dp.vertical_diffusion(c, r_au=100.0, a_cm=0.1, alpha_turb=1e-4, n_paths=20000)
    assert np.array_equal(r["z_final_au"], r2["z_final_au"])


def test_charge_and_hall_parameter():
    psi = dp.oml_potential(29.0)
    assert -4.5 < psi < -3.5
    c = dp.RunConfig()
    c.disk_model = "self_consistent"
    c.B0_G, c.field_exponent, c.br_over_bz = 1.0, 1.25, 0.3
    H1 = dp.disk_profile(c, [1.0])["H_au"][0]
    g_mid, g_up = dp.hall_parameter(c, [1.0, 1.0], [0.0, 5 * H1], 1e-5)
    assert g_up > 1e4 * g_mid               # rho falls by exp(-12.5) -> tau_s grows
    # Gamma ~ 1/a at fixed position (Z ~ a, m ~ a^3, tau ~ a)
    g1, g2 = dp.hall_parameter(c, [1.0, 1.0], [0.0, 0.0], 1e-5), dp.hall_parameter(c, [1.0, 1.0], [0.0, 0.0], 1e-4)
    assert g1[0] / g2[0] == pytest.approx(10.0, rel=1e-9)
    c.charge = "oml"
    c.a_init_cm, c.r_init_au, c.r_in_au = 1e-4, 2.0, 1.5
    res = dp.run(c)
    assert np.all(res["columns"]["Z_charge"] < 0)
