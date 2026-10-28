"""Tests of the installed dustppd package (C++ core through pybind11)."""
import math
import threading
import time

import numpy as np
import pytest

import dustppd as dp


def test_backends():
    names = dp.available_backends()
    assert "scalar" in names and "generic" in names
    assert dp.active_backend() in names
    prev = dp.active_backend()
    assert dp.set_backend("scalar")
    assert dp.active_backend() == "scalar"
    dp.set_backend(prev)
    with pytest.raises(ValueError):
        dp.set_backend("sse9000")


@pytest.mark.parametrize("method", ["dopri5", "bdf", "auto"])
def test_solve_python_rhs(method):
    res = dp.solve(lambda t, y: -y, (0.0, 2.0), [1.0, 2.0], method=method, rtol=1e-9, atol=[1e-12])
    assert res["success"], res["message"]
    np.testing.assert_allclose(res["y_last"], [math.exp(-2), 2 * math.exp(-2)], rtol=1e-6)
    assert res["y"].shape == (len(res["t"]), 2)
    assert res["stats"]["n_steps"] > 0


def test_t_eval_and_jacobian():
    lam = -50.0
    res = dp.solve(lambda t, y: lam * y, (0, 1), [1.0], method="bdf", rtol=1e-8, atol=[1e-12],
                   jac=lambda t, y: np.array([[lam]]), t_eval=[0.1, 0.5, 1.0], save_steps=False)
    assert res["success"]
    np.testing.assert_allclose(res["t"], [0.1, 0.5, 1.0])
    np.testing.assert_allclose(res["y"][:, 0], np.exp(lam * np.array([0.1, 0.5, 1.0])), rtol=1e-5, atol=1e-12)  # solver atol
    assert res["stats"]["n_jac"] >= 1


def test_callback_exception_propagates():
    def bad(t, y):
        if t > 0.5:
            raise ZeroDivisionError("model failure")
        return -y

    with pytest.raises(ZeroDivisionError, match="model failure"):
        dp.solve(bad, (0, 1), [1.0])


def test_failure_is_reported_not_raised():
    res = dp.solve(lambda t, y: y * y, (0, 2), [1.0], method="dopri5")
    assert not res["success"]
    assert res["status"] in ("step_size_underflow", "non_finite_state")
    assert 0.999 < res["t_last"] < 1.0 + 1e-6


def test_builtin_robertson_matches_scipy():
    scipy = pytest.importorskip("scipy.integrate")
    res = dp.solve_system(dp.robertson(), (0, 4e5), [1, 0, 0], method="bdf", rtol=1e-8,
                          atol=[1e-14, 1e-18, 1e-14], save_steps=False)
    ref = scipy.solve_ivp(lambda t, y: [-0.04 * y[0] + 1e4 * y[1] * y[2],
                                        0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2, 3e7 * y[1] ** 2],
                          (0, 4e5), [1, 0, 0], method="Radau", rtol=1e-11, atol=[1e-16, 1e-20, 1e-16])
    np.testing.assert_allclose(res["y_last"], ref.y[:, -1], rtol=2e-6, atol=1e-16)
    assert abs(res["y_last"].sum() - 1) < 1e-10


def test_gmres_path():
    sys = dp.reaction_diffusion(200, 1.0, 5.0)
    x = np.linspace(0, 1, 202)[1:-1]
    u0 = np.sin(np.pi * x)
    a = dp.solve_system(sys, (0, 0.1), u0, method="bdf", rtol=1e-7, atol=[1e-10], save_steps=False)
    b = dp.solve_system(sys, (0, 0.1), u0, method="bdf", rtol=1e-7, atol=[1e-10], save_steps=False,
                        linear_solver="gmres")
    assert a["success"] and b["success"]
    np.testing.assert_allclose(a["y_last"], b["y_last"], atol=1e-6)
    assert b["stats"]["n_lu"] == 0 and b["stats"]["n_linear_iter"] > 0


def test_reference_case_runs_and_matches_paper():
    cfg = dp.reference_config()
    assert cfg.a_init_cm == 100.0 and cfg.r_init_au == 100.0
    res = dp.run(cfg, check=True)
    s = res["summary"]
    assert s["reached_inner_edge"]
    assert s["t_drift_yr"] == pytest.approx(306e3, rel=0.02)       # paper Sec. 3.4
    assert s["t_settle_first_yr"] == pytest.approx(252, rel=0.02)  # paper Table 1
    cols = res["columns"]
    assert set(["t_yr", "r_au", "z_au", "vr_cms", "a_cm", "St"]).issubset(cols)
    assert len(cols["t_yr"]) == len(res["t_code"]) == res["y_code"].shape[0]
    assert np.all(np.diff(cols["t_yr"]) > 0)


def test_cpp_matches_independent_scipy_model():
    dms = pytest.importorskip("dust_model_scipy")
    t_ref, _ = dms.drift_time_yr(10.0, 1.0, sed=False, drag="both", rtol=1e-10)
    cfg = dp.RunConfig()
    cfg.a_init_cm = 10.0
    cfg.r_init_au = 1.0
    cfg.sedimentation = False
    cfg.rtol = 1e-10
    cfg.atol = [1e-13, 1e-17, 1e-13, 1e-13, 1e-17, 1e-13]
    s = dp.run(cfg, keep_trajectory=False)["summary"]
    assert s["t_drift_yr"] == pytest.approx(t_ref, rel=2e-6)


def test_config_roundtrip_and_validation(tmp_path):
    cfg = dp.reference_config(["physics.coagulation=true"])
    assert cfg.coagulation
    p = tmp_path / "c.toml"
    dp.save_config(cfg, str(p))
    cfg2 = dp.load_config(str(p))
    assert cfg2.coagulation and cfg2.a_init_cm == cfg.a_init_cm
    with pytest.raises(Exception):
        dp.load_config(str(p), ["physics.bogus=1"])
    with pytest.raises(ValueError):
        cfg2.drag = "newtonian"


def test_sweep_and_ensemble_equal_serial():
    cfg = dp.reference_config(["particle.r_init_au=10.0"])
    sizes = [1e-4, 1e-2, 1.0, 100.0]
    par = dp.sweep(cfg, "a_init_cm", sizes, threads=4)
    for a, s in zip(sizes, par):
        c = cfg.copy()
        c.a_init_cm = a
        ser = dp.run(c, keep_trajectory=False)["summary"]
        assert s["t_drift_yr"] == ser["t_drift_yr"]
    # larger grains (up to ~1 m) drift faster
    t = [s["t_drift_yr"] for s in par]
    assert t[0] > t[1] > t[2]


def test_cancel_and_progress_from_another_thread():
    cfg = dp.reference_config(["particle.a_init_cm=1e-4", "particle.r_init_au=100.0", "solver.method=\"dopri5\"",
                               "solver.max_steps=2000000000"])
    ctl = dp.RunControl()
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("r", dp.run(cfg, ctl, keep_trajectory=False)))
    th.start()
    time.sleep(0.3)
    assert th.is_alive(), "explicit run of a micron grain should take long"
    ctl.cancel()
    th.join(timeout=10)
    assert not th.is_alive()
    assert out["r"]["summary"]["status"] == "cancelled"
    assert 0.0 <= ctl.progress <= 1.0
