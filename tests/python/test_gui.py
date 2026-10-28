"""Browser GUI: the HTTP API behind the page (no browser or display needed).
Runs a real server on a free port and drives it like the page does."""
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest


@pytest.fixture(scope="module")
def server():
    gui = pytest.importorskip("dustppd.gui")
    srv = gui.make_server(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read()


def _post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _wait(base, jid, timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = json.loads(_get(base, f"/api/status?id={jid}")[2])
        if st["done"]:
            return st
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_page_and_static_files(server):
    code, ctype, body = _get(server, "/")
    assert code == 200 and ctype.startswith("text/html") and b"dustppd" in body
    for name in ("app.js", "style.css"):
        code, _, body = _get(server, f"/static/{name}")
        assert code == 200 and len(body) > 1000
    with pytest.raises(urllib.error.HTTPError):
        _get(server, "/static/../gui.py")


def test_schema_defaults_are_reference_case(server):
    s = json.loads(_get(server, "/api/schema")[2])
    d = s["defaults"]
    assert d["a_init_cm"] == 100.0 and d["r_init_au"] == 100.0 and d["sedimentation"] is True
    keys = {f["key"] for g in s["groups"] for f in g["fields"]}
    assert set(d) == keys
    assert {"reference", "dead_zone_trap"} <= {p["id"] for p in s["presets"]}


def test_reference_run_summary_plots_and_csv(server):
    params = _post(server, "/api/preset", {"id": "reference"})[1]["params"]
    params["r_init_au"] = "10"  # string values from the form are accepted
    code, r = _post(server, "/api/run", {"params": params})
    assert code == 200
    st = _wait(server, r["id"])
    assert "error" not in st
    res = st["result"]
    s = res["summary"]
    assert s["reached_inner_edge"] and s["t_drift_yr"] > 0
    cols = res["columns"]
    assert len(cols["t_yr"]) == len(cols["r_au"]) <= 4001
    assert res["disk"]["beta"] and len(res["disk"]["r_au"]) == 500
    # The CSV has the full (not decimated) trajectory.
    _, ctype, body = _get(server, f"/api/csv?id={r['id']}")
    assert ctype.startswith("text/csv")
    data = np.genfromtxt(body.decode().splitlines(), delimiter=",", names=True)
    assert len(data) == res["n_points"]
    assert abs(data["t_yr"][-1] - s["t_drift_yr"]) < 1e-6 * s["t_drift_yr"]


def test_dead_zone_trap_preset(server):
    params = _post(server, "/api/preset", {"id": "dead_zone_trap"})[1]["params"]
    st = _wait(server, _post(server, "/api/run", {"params": params})[1]["id"])
    s = st["result"]["summary"]
    assert s["trapped"] and abs(s["r_trap_au"] - 10.67) < 0.01
    traps = st["result"]["disk"]["traps"]
    assert len(traps) == 1 and abs(traps[0]["r_au"] - s["r_trap_au"]) < 1e-3


def test_invalid_input_is_reported(server):
    params = json.loads(_get(server, "/api/schema")[2])["defaults"]
    code, r = _post(server, "/api/run", {"params": {**params, "a_init_cm": "not-a-number"}})
    assert code == 400 and "начальный радиус" in r["error"]
    code, r = _post(server, "/api/run", {"params": {**params, "disk_model": "tabulated", "table_path": ""}})
    assert code == 400 and "tabulated" in r["error"]


def test_cancel(server):
    params = json.loads(_get(server, "/api/schema")[2])["defaults"]
    params.update(a_init_cm=1e-4, method="dopri5", max_steps=2_000_000_000)
    jid = _post(server, "/api/run", {"params": params})[1]["id"]
    time.sleep(0.2)
    _post(server, "/api/cancel", {"id": jid})
    assert _wait(server, jid)["result"]["summary"]["status"] == "cancelled"


def test_toml_round_trip(server):
    params = _post(server, "/api/preset", {"id": "trap_with_field"})[1]["params"]
    text = _post(server, "/api/to_toml", {"params": params})[1]["text"]
    back = _post(server, "/api/parse_toml", {"text": text})[1]["params"]
    for k in ("a_init_cm", "r_init_au", "disk_model", "perturbation", "bump_amplitude", "B0_G", "br_over_bz"):
        assert back[k] == params[k], k


def test_vertical_mixing(server):
    params = json.loads(_get(server, "/api/schema")[2])["defaults"]
    jid = _post(server, "/api/diffusion", {"params": params, "options": {
        "r_au": 100, "a_cm": 0.01, "alpha_t": 1e-3, "n_paths": 5000}})[1]["id"]
    d = _wait(server, jid)["result"]
    # Same case as docs/science/settling.md: exact 0.6215, Dubrulle 0.7241.
    assert abs(d["Hd_exact"] - 0.6215) < 2e-3 and abs(d["Hd_dubrulle"] - 0.7241) < 2e-3
    assert abs(d["Hd_sim"] - d["Hd_exact"]) < 5 * d["Hd_err"] + 0.01
