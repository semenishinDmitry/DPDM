"""Non-interactive smoke test of the GUI: edit parameters, run in the worker
thread, wait for completion through the Tk event loop, plot, export, cancel.
Skipped when no display is available."""
import time

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")
pytest.importorskip("matplotlib")


@pytest.fixture
def app():
    gui = pytest.importorskip("dustppd.gui")
    try:
        a = gui.App()
    except tk.TclError as e:  # no display
        pytest.skip(f"no display: {e}")
    a.withdraw()
    yield a, gui
    a.destroy()


def _wait(a, timeout=30.0):
    t0 = time.time()
    while a.worker is not None and (a.worker.is_alive() or not a.done_q.empty()) and time.time() - t0 < timeout:
        a.update()
        time.sleep(0.02)
    for _ in range(20):  # let _poll consume the result
        a.update()
        time.sleep(0.02)


def test_gui_run_plot_export(app, tmp_path, monkeypatch):
    a, gui = app
    a.vars["r_init_au"].set("10.0")
    a.vars["a_init_cm"].set("100.0")
    a.on_run()
    _wait(a)
    assert a.result is not None
    s = a.result["summary"]
    assert s["reached_inner_edge"]
    assert "t_drift" in a.text.get("1.0", "end")
    assert len(a.fig.axes) == 4
    out = tmp_path / "traj.csv"
    monkeypatch.setattr(gui.filedialog, "asksaveasfilename", lambda **kw: str(out))
    a.on_export()
    data = np.genfromtxt(out, delimiter=",", names=True)
    assert len(data) == len(a.result["columns"]["t_yr"])
    assert abs(data["t_yr"][-1] - s["t_drift_yr"]) < 1e-6 * s["t_drift_yr"]
    # save and reload the configuration
    cfg_path = tmp_path / "cfg.toml"
    monkeypatch.setattr(gui.filedialog, "asksaveasfilename", lambda **kw: str(cfg_path))
    a.on_save()
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **kw: str(cfg_path))
    a.on_load()
    assert float(a.vars["r_init_au"].get()) == 10.0


def test_gui_invalid_input_and_cancel(app, monkeypatch):
    a, gui = app
    errors = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda title, msg: errors.append(msg))
    monkeypatch.setattr(gui.messagebox, "showwarning", lambda title, msg: errors.append(msg))
    a.vars["a_init_cm"].set("not-a-number")
    a.on_run()
    assert errors and "a_init" in errors[0]
    # A run that cannot finish quickly (explicit method on a micron grain) is cancelled.
    a.vars["a_init_cm"].set("1e-4")
    a.vars["method"].set("dopri5")
    a.vars["max_steps"].set("2000000000")
    a.on_run()
    for _ in range(10):
        a.update()
        time.sleep(0.02)
    a.on_cancel()
    _wait(a)
    assert a.result["summary"]["status"] == "cancelled"


def test_gui_dead_zone_trap(app):
    a, gui = app
    a.vars["disk_model"].set("self_consistent")
    a.vars["perturbation"].set("dead_zone_edge")
    a.vars["bump_amplitude"].set("2.0")
    a.vars["bump_r_au"].set("10.0")
    a.vars["bump_width_au"].set("1.0")
    a.vars["a_init_cm"].set("1.0")
    a.vars["r_init_au"].set("15.0")
    a.vars["t_end_yr"].set("3e6")
    a.vars["sedimentation"].set(False)
    a.on_run()
    _wait(a)
    s = a.result["summary"]
    assert s["trapped"] and abs(s["r_trap_au"] - 10.67) < 0.01
    assert "trapped" in a.text.get("1.0", "end")
