"""Browser GUI for configuring, running and inspecting dust-model runs.

Launch:  dustppd-gui   (or ``python -m dustppd.gui``)

A small HTTP server (standard library only) listens on 127.0.0.1 and the default
browser opens the page. Simulations run in worker threads; the C++ core releases
the GIL, so the page stays responsive and runs can be cancelled.

Options: ``--port N`` (default: a free port), ``--no-browser``.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np

import dustppd as dp

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

# Parameter form: (group, [(RunConfig attribute, label, unit, kind, choices)]).
# kind: "float" | "int" | "bool" | "choice" | "path"
FIELDS: List[Tuple[str, List[Tuple[str, str, str, str, Optional[List[str]]]]]] = [
    ("Частица", [
        ("a_init_cm", "начальный радиус a", "см", "float", None),
        ("rho_d", "плотность вещества ρ_d", "г/см³", "float", None),
        ("r_init_au", "начальный радиус орбиты r₀", "а.е.", "float", None),
        ("z_init_H", "начальная высота z₀", "H(r₀)", "float", None),
        ("vr_init_cms", "начальная v_r", "см/с", "float", None),
        ("vz_init_cms", "начальная v_z", "см/с", "float", None),
    ]),
    ("Физика", [
        ("drag", "закон трения", "", "choice", ["both", "epstein", "stokes"]),
        ("coagulation", "коагуляция (рост, ур. 10)", "", "bool", None),
        ("sedimentation", "оседание (движение по z)", "", "bool", None),
        ("dust_to_gas", "доля пыли Y_d", "", "float", None),
        ("mu", "молекулярный вес μ", "", "float", None),
        ("sigma_mol_cm2", "сечение молекул σ", "см²", "float", None),
        ("r_in_au", "внутренняя граница r_in", "а.е.", "float", None),
        ("settle_eps", "полоса осаждения ε", "z₀", "float", None),
        ("t_end_yr", "время счёта t_end", "лет", "float", None),
    ]),
    ("Диск", [
        ("disk_model", "модель диска", "", "choice", ["analytic", "self_consistent", "tabulated_beta", "tabulated"]),
        ("table_path", "файл таблицы (для tabulated)", "", "path", None),
        ("alpha", "α", "", "float", None),
        ("mdot", "Ṁ", "10⁻⁸ M☉/год", "float", None),
        ("mstar", "M★", "M☉", "float", None),
        ("vphi2vk", "u_φ / v_K (для analytic)", "", "float", None),
        ("perturbation", "возмущение", "", "choice", ["none", "gaussian", "dead_zone_edge"]),
        ("bump_amplitude", "амплитуда A", "", "float", None),
        ("bump_r_au", "положение r_b", "а.е.", "float", None),
        ("bump_width_au", "ширина w", "а.е.", "float", None),
    ]),
    ("Поле и заряд", [
        ("B0_G", "B₀ на 1 а.е.", "Гс", "float", None),
        ("field_exponent", "показатель p (B ∝ r⁻ᵖ)", "", "float", None),
        ("br_over_bz", "B_r / B_z", "", "float", None),
        ("magnetic_tension", "магнитное натяжение в β", "", "bool", None),
        ("charge", "заряд пылинки", "", "choice", ["none", "oml"]),
        ("ion_mass_amu", "масса иона", "а.е.м.", "float", None),
    ]),
    ("Решатель", [
        ("method", "метод", "", "choice", ["auto", "dopri5", "bdf", "rk4"]),
        ("linear_solver", "линейный решатель", "", "choice", ["dense_lu", "gmres"]),
        ("rtol", "относительный допуск rtol", "", "float", None),
        ("max_steps", "максимум шагов", "", "int", None),
    ]),
]

PRESETS = {
    "reference": ("Опорный случай статьи: 1 м из 100 а.е.", []),
    "micron_growth": ("1 мкм с 1 а.е., рост и дрейф", [
        "particle.a_init_cm=1e-4", "particle.r_init_au=1", "physics.coagulation=true",
        "physics.sedimentation=false"]),
    "dead_zone_trap": ("Ловушка на краю мёртвой зоны: 1 см с 15 а.е.", [
        "particle.a_init_cm=1", "particle.r_init_au=15", "physics.sedimentation=false",
        "run.t_end_yr=3e6", 'disk.model="self_consistent"', 'disk.perturbation="dead_zone_edge"',
        "disk.bump_amplitude=2", "disk.bump_r_au=10", "disk.bump_width_au=1"]),
    "trap_with_field": ("Та же ловушка с остаточным полем B₀ = 2 Гс", [
        "particle.a_init_cm=1", "particle.r_init_au=15", "physics.sedimentation=false",
        "run.t_end_yr=3e6", 'disk.model="self_consistent"', 'disk.perturbation="dead_zone_edge"',
        "disk.bump_amplitude=2", "disk.bump_r_au=10", "disk.bump_width_au=1",
        "field.B0_G=2", "field.exponent=1.25", "field.br_over_bz=0.5"]),
    "charged": ("Заряженная пылинка 1 мкм в поле 1 Гс", [
        "particle.a_init_cm=1e-4", "particle.r_init_au=1", 'physics.charge="oml"',
        "field.B0_G=1", "field.exponent=1.25", "field.br_over_bz=0.5", "run.t_end_yr=2e4"]),
}

MAX_POINTS = 4000


def _clean(x: Any) -> Any:
    """JSON-safe copy: NaN / inf -> None, numpy -> python."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def config_to_params(cfg: dp.RunConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for _, fields in FIELDS:
        for attr, _l, _u, _k, _c in fields:
            if attr == "table_path":
                out[attr] = cfg.beta_table if cfg.disk_model == "tabulated_beta" else cfg.profile_table
            else:
                out[attr] = getattr(cfg, attr)
    return _clean(out)


def params_to_config(params: Dict[str, Any]) -> dp.RunConfig:
    """Build and validate a RunConfig from form values (validated like the CLI/TOML path)."""
    cfg = dp.reference_config()
    model = str(params.get("disk_model", cfg.disk_model))
    for _, fields in FIELDS:
        for attr, label, _u, kind, choices in fields:
            if attr in ("table_path", "disk_model") or attr not in params:
                continue
            raw = params[attr]
            try:
                if kind == "float":
                    val: Any = float(raw)
                    if not math.isfinite(val):
                        raise ValueError
                elif kind == "int":
                    val = int(float(raw))
                elif kind == "bool":
                    val = bool(raw)
                else:
                    val = str(raw)
                    if choices and val not in choices:
                        raise ValueError
            except (TypeError, ValueError):
                raise ValueError(f"«{label}»: недопустимое значение '{raw}'") from None
            setattr(cfg, attr, val)
    path = str(params.get("table_path") or "").strip()
    if model == "tabulated_beta":
        if not path:
            raise ValueError("для tabulated_beta укажите файл таблицы β(r)")
        cfg.beta_table = path
    elif model == "tabulated":
        if not path:
            raise ValueError("для tabulated укажите файл профилей диска")
        cfg.profile_table = path
    else:
        cfg.beta_table = ""
        cfg.disk_model = model
    checked = dp.parse_config(cfg.to_toml())
    # parse_config re-reads table files relative to "."; keep the loaded tables.
    return checked if model not in ("tabulated_beta", "tabulated") else cfg


def _decimate(cols: Dict[str, np.ndarray]) -> Dict[str, Any]:
    n = len(next(iter(cols.values()))) if cols else 0
    if n > MAX_POINTS:
        idx = np.unique(np.concatenate([np.linspace(0, n - 1, MAX_POINTS).astype(int), [n - 1]]))
    else:
        idx = slice(None)
    return {k: _clean(np.asarray(v)[idx]) for k, v in cols.items()}


def disk_overview(cfg: dp.RunConfig, a_cm: float) -> Dict[str, Any]:
    r_lo = max(cfg.r_in_au, 1e-3)
    r_hi = max(2.0 * cfg.r_init_au, 10.0 * r_lo, cfg.bump_r_au * 2 if cfg.perturbation != "none" else 0.0)
    r = np.logspace(math.log10(r_lo), math.log10(r_hi), 500)
    prof = dp.disk_profile(cfg, r.tolist())
    out = {k: v for k, v in prof.items()}
    out["vr_stat_cms"] = dp.stationary_vr(cfg, r.tolist(), a_cm)
    out["traps"] = [z for z in dp.drift_zeros(cfg, a_cm, r_lo, r_hi) if z["stable"]]
    out["headwind_zero_au"] = list(dp.zero_headwind_radii(cfg, r_lo, r_hi))
    return _clean(out)


class Job:
    def __init__(self, kind: str) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.control = dp.RunControl() if kind == "run" else None
        self.result: Optional[Dict[str, Any]] = None
        self.payload: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.done = threading.Event()


class State:
    def __init__(self) -> None:
        self.jobs: Dict[str, Job] = {}
        self.lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self.lock:
            self.jobs[job.id] = job
            while len(self.jobs) > 8:  # keep memory bounded
                oldest = next(iter(self.jobs))
                if not self.jobs[oldest].done.is_set():
                    break
                del self.jobs[oldest]

    def get(self, jid: str) -> Optional[Job]:
        with self.lock:
            return self.jobs.get(jid)


STATE = State()


def start_run(params: Dict[str, Any]) -> Job:
    cfg = params_to_config(params)
    job = Job("run")

    def work() -> None:
        try:
            res = dp.run(cfg, job.control)
            job.result = res
            payload = {"summary": _clean(res["summary"]), "columns": _decimate(res["columns"]),
                       "n_points": int(len(res["columns"]["t_yr"]))}
            try:
                payload["disk"] = disk_overview(cfg, cfg.a_init_cm)
            except Exception as e:  # noqa: BLE001 - disk panel is optional
                payload["disk_error"] = str(e)
            job.payload = payload
        except Exception as e:  # noqa: BLE001 - reported to the page
            job.error = str(e)
        finally:
            job.done.set()

    threading.Thread(target=work, daemon=True).start()
    STATE.add(job)
    return job


def start_diffusion(params: Dict[str, Any], opts: Dict[str, Any]) -> Job:
    cfg = params_to_config(params)
    r_au = float(opts.get("r_au", 100.0))
    a_cm = float(opts.get("a_cm", 0.01))
    alpha_t = float(opts.get("alpha_t", 1e-3))
    n_paths = int(opts.get("n_paths", 20000))
    if not (r_au > 0 and a_cm > 0 and alpha_t > 0 and 100 <= n_paths <= 1_000_000):
        raise ValueError("перемешивание: r, a, α_t > 0; число траекторий 100 … 10⁶")
    job = Job("diffusion")

    def work() -> None:
        try:
            d = dp.vertical_diffusion(cfg, r_au, a_cm, alpha_t, n_paths)
            H = d["H_au"]
            z = np.linspace(-5 * H, 5 * H, 401)
            ex = dp.stationary_dust_density(cfg, r_au, a_cm, alpha_t, z.tolist())
            zf = np.asarray(d["z_final_au"]) / H
            hist, edges = np.histogram(zf, bins=60, range=(-5, 5), density=True)
            dens = np.asarray(ex["density"])
            dens = dens / np.trapezoid(dens, z / H) if hasattr(np, "trapezoid") else dens / np.sum(
                0.5 * (dens[1:] + dens[:-1]) * np.diff(z / H))
            Hd_dub = d["Hd_dubrulle_au"]
            gauss = np.exp(-0.5 * (z / Hd_dub) ** 2)
            gauss /= np.sum(0.5 * (gauss[1:] + gauss[:-1]) * np.diff(z / H))
            job.payload = _clean({
                "status": d["status"], "message": d["message"], "H_au": H, "St_mid": d["St_mid"],
                "Hd_sim": d["Hd_simulated_au"] / H, "Hd_err": d["Hd_stat_error_au"] / H,
                "Hd_exact": d["Hd_exact_au"] / H, "Hd_dubrulle": Hd_dub / H,
                "t_yr": d["t_yr"], "z_rms_H": np.asarray(d["z_rms_au"]) / H,
                "z_H": z / H, "density_exact": dens, "density_dubrulle": gauss,
                "hist_z": 0.5 * (edges[1:] + edges[:-1]), "hist": hist,
                "wall_time_s": d["wall_time_s"], "n_paths": n_paths, "warnings": d["warnings"],
                "r_au": r_au, "a_cm": a_cm, "alpha_t": alpha_t,
            })
        except Exception as e:  # noqa: BLE001
            job.error = str(e)
        finally:
            job.done.set()

    threading.Thread(target=work, daemon=True).start()
    STATE.add(job)
    return job


def trajectory_csv(job: Job) -> str:
    cols = job.result["columns"]
    names = list(cols)
    buf = io.StringIO()
    np.savetxt(buf, np.column_stack([cols[n] for n in names]), delimiter=",", header=",".join(names),
               comments="", fmt="%.17g")
    return buf.getvalue()


def schema() -> Dict[str, Any]:
    groups = []
    for g, fields in FIELDS:
        groups.append({"title": g, "fields": [
            {"key": a, "label": lab, "unit": u, "kind": k, "choices": c} for a, lab, u, k, c in fields]})
    cfg = dp.reference_config()
    return {"groups": groups, "defaults": config_to_params(cfg),
            "presets": [{"id": k, "title": v[0]} for k, v in PRESETS.items()],
            "version": dp.__version__, "backend": dp.active_backend()}


class Handler(BaseHTTPRequestHandler):
    server_version = "dustppd-gui"

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet
        pass

    # ---------------------------------------------------------------- helpers
    def _send(self, code: int, body: bytes, ctype: str, extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, allow_nan=False).encode(), "application/json")

    def _error(self, msg: str, code: int = 400) -> None:
        self._json({"error": msg}, code)

    def _body(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _job(self, q: Dict[str, List[str]]) -> Optional[Job]:
        job = STATE.get((q.get("id") or [""])[0])
        if job is None:
            self._error("неизвестный расчёт", 404)
        return job

    # ------------------------------------------------------------------ GET
    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            return self._static("index.html")
        if url.path.startswith("/static/"):
            return self._static(url.path[len("/static/"):])
        if url.path == "/api/schema":
            return self._json(schema())
        if url.path == "/api/status":
            job = self._job(q)
            if job is None:
                return
            out: Dict[str, Any] = {"id": job.id, "kind": job.kind, "done": job.done.is_set()}
            if job.control is not None:
                out["progress"] = job.control.progress
            if job.done.is_set():
                if job.error:
                    out["error"] = job.error
                else:
                    out["result"] = job.payload
            return self._json(out)
        if url.path == "/api/csv":
            job = self._job(q)
            if job is None:
                return
            if not job.done.is_set() or job.result is None:
                return self._error("нет готовой траектории")
            return self._send(200, trajectory_csv(job).encode(), "text/csv",
                              {"Content-Disposition": 'attachment; filename="dustppd_trajectory.csv"'})
        self._error("not found", 404)

    def _static(self, name: str) -> None:
        path = os.path.normpath(os.path.join(WEB_DIR, name))
        if not path.startswith(WEB_DIR + os.sep) or not os.path.isfile(path):
            return self._error("not found", 404)
        ctype = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}.get(
            os.path.splitext(path)[1], "application/octet-stream")
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    # ----------------------------------------------------------------- POST
    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        try:
            body = self._body()
        except (ValueError, json.JSONDecodeError):
            return self._error("некорректный JSON")
        try:
            if url.path == "/api/run":
                return self._json({"id": start_run(body.get("params", {})).id})
            if url.path == "/api/diffusion":
                return self._json({"id": start_diffusion(body.get("params", {}), body.get("options", {})).id})
            if url.path == "/api/cancel":
                job = STATE.get(body.get("id", ""))
                if job is not None and job.control is not None:
                    job.control.cancel()
                return self._json({"ok": True})
            if url.path == "/api/preset":
                pid = body.get("id", "reference")
                if pid not in PRESETS:
                    return self._error("неизвестный пресет")
                cfg = dp.reference_config(PRESETS[pid][1])
                return self._json({"params": config_to_params(cfg)})
            if url.path == "/api/parse_toml":
                cfg = dp.parse_config(str(body.get("text", "")))
                return self._json({"params": config_to_params(cfg)})
            if url.path == "/api/to_toml":
                return self._json({"text": params_to_config(body.get("params", {})).to_toml()})
            if url.path == "/api/validate":
                params_to_config(body.get("params", {}))
                return self._json({"ok": True})
        except Exception as e:  # noqa: BLE001 - configuration / input errors go to the page
            return self._error(str(e))
        self._error("not found", 404)


def make_server(port: int = 0) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    return srv


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="dustppd-gui", description="dustppd browser GUI")
    ap.add_argument("--port", type=int, default=0, help="port (default: any free port)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args(argv)
    srv = make_server(args.port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"dustppd GUI: {url}  (Ctrl+C — остановить)", flush=True)
    if not args.no_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
