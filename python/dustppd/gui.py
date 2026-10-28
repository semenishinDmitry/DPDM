"""Tk/matplotlib GUI for configuring, running, inspecting and exporting dust-model runs.

Launch:  python -m dustppd.gui   (or the ``dustppd-gui`` console script)

The simulation runs in a worker thread; the C++ core releases the GIL, so the
GUI stays responsive. Cancellation is cooperative (checked after every step).
"""
from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, Optional

import numpy as np

import dustppd as dp

try:
    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except ImportError as e:  # pragma: no cover
    raise SystemExit("dustppd GUI needs matplotlib: pip install matplotlib") from e

# (label, RunConfig attribute, type)
FIELDS = [
    ("Particle", [
        ("initial radius a_init [cm]", "a_init_cm", float),
        ("material density rho_d [g/cm^3]", "rho_d", float),
        ("initial r [au]", "r_init_au", float),
        ("initial z [H(r_init)]", "z_init_H", float),
        ("initial v_r [cm/s]", "vr_init_cms", float),
        ("initial v_z [cm/s]", "vz_init_cms", float),
    ]),
    ("Physics", [
        ("dust-to-gas ratio Y_d", "dust_to_gas", float),
        ("mean molecular weight mu", "mu", float),
        ("collision cross-section [cm^2]", "sigma_mol_cm2", float),
        ("inner edge r_in [au]", "r_in_au", float),
        ("settling band eps", "settle_eps", float),
        ("t_end [yr]", "t_end_yr", float),
    ]),
    ("Disk", [
        ("alpha", "alpha", float),
        ("Mdot [1e-8 Msun/yr]", "mdot", float),
        ("M_star [Msun]", "mstar", float),
        ("u_phi / v_K (analytic beta)", "vphi2vk", float),
        ("perturbation amplitude A", "bump_amplitude", float),
        ("perturbation radius [au]", "bump_r_au", float),
        ("perturbation width [au]", "bump_width_au", float),
    ]),
    ("Fossil field / charge", [
        ("B0 at 1 au [G]", "B0_G", float),
        ("field exponent", "field_exponent", float),
        ("B_r / B_z", "br_over_bz", float),
        ("ion mass [amu]", "ion_mass_amu", float),
    ]),
    ("Solver", [
        ("rtol", "rtol", float),
        ("max steps", "max_steps", int),
    ]),
]

PLOTS = {
    "r(t)": ("t_yr", "r_au", "log", "log"),
    "z(r)": ("r_au", "z_au", "linear", "linear"),
    "z(t)": ("t_yr", "z_au", "log", "linear"),
    "|v_r|(r)": ("r_au", "vr_cms", "log", "log"),
    "a(r)": ("r_au", "a_cm", "log", "log"),
    "a(t)": ("t_yr", "a_cm", "log", "log"),
    "St(r)": ("r_au", "St", "log", "log"),
    "Re(r)": ("r_au", "Re", "log", "log"),
    "beta(r)": ("r_au", "beta", "log", "linear"),
    "Hall(r)": ("r_au", "hall", "log", "log"),
}


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("dustppd - dust dynamics in protoplanetary disks")
        self.geometry("1280x820")
        self.cfg: dp.RunConfig = dp.reference_config()
        self.result: Optional[Dict[str, Any]] = None
        self.control: Optional[dp.RunControl] = None
        self.worker: Optional[threading.Thread] = None
        self.done_q: "queue.Queue[Any]" = queue.Queue()
        self.vars: Dict[str, tk.Variable] = {}
        self._build()
        self._load_into_widgets()

    # ------------------------------------------------------------- layout --
    def _build(self) -> None:
        left = ttk.Frame(self, padding=6)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(self, padding=6)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        bar = ttk.Frame(left)
        bar.pack(fill=tk.X)
        ttk.Button(bar, text="Reference case", command=self.on_reference).pack(side=tk.LEFT)
        ttk.Button(bar, text="Load...", command=self.on_load).pack(side=tk.LEFT)
        ttk.Button(bar, text="Save...", command=self.on_save).pack(side=tk.LEFT)

        for section, fields in FIELDS:
            box = ttk.LabelFrame(left, text=section, padding=4)
            box.pack(fill=tk.X, pady=3)
            for label, attr, typ in fields:
                row = ttk.Frame(box)
                row.pack(fill=tk.X)
                ttk.Label(row, text=label, width=30).pack(side=tk.LEFT)
                v = tk.StringVar()
                ttk.Entry(row, textvariable=v, width=14).pack(side=tk.RIGHT)
                self.vars[attr] = v
            if section == "Physics":
                row = ttk.Frame(box)
                row.pack(fill=tk.X)
                ttk.Label(row, text="drag law", width=30).pack(side=tk.LEFT)
                self.vars["drag"] = tk.StringVar()
                ttk.Combobox(row, textvariable=self.vars["drag"], values=["both", "epstein", "stokes"], width=11,
                             state="readonly").pack(side=tk.RIGHT)
                self.vars["coagulation"] = tk.BooleanVar()
                self.vars["sedimentation"] = tk.BooleanVar()
                ttk.Checkbutton(box, text="coagulation (Eq. 10)", variable=self.vars["coagulation"]).pack(anchor=tk.W)
                ttk.Checkbutton(box, text="sedimentation (2D)", variable=self.vars["sedimentation"]).pack(anchor=tk.W)
            if section == "Disk":
                for label, attr, values in (("disk model", "disk_model",
                                             ["analytic", "self_consistent", "tabulated_beta", "tabulated"]),
                                            ("perturbation", "perturbation", ["none", "gaussian", "dead_zone_edge"])):
                    row = ttk.Frame(box)
                    row.pack(fill=tk.X)
                    ttk.Label(row, text=label, width=30).pack(side=tk.LEFT)
                    self.vars[attr] = tk.StringVar()
                    ttk.Combobox(row, textvariable=self.vars[attr], values=values, width=14,
                                 state="readonly").pack(side=tk.RIGHT)
            if section == "Fossil field / charge":
                row = ttk.Frame(box)
                row.pack(fill=tk.X)
                ttk.Label(row, text="grain charge", width=30).pack(side=tk.LEFT)
                self.vars["charge"] = tk.StringVar()
                ttk.Combobox(row, textvariable=self.vars["charge"], values=["none", "oml"], width=11,
                             state="readonly").pack(side=tk.RIGHT)
            if section == "Disk":
                row = ttk.Frame(box)
                row.pack(fill=tk.X)
                self.vars["beta_table"] = tk.StringVar()
                ttk.Button(row, text="beta table...", command=self.on_beta).pack(side=tk.LEFT)
                ttk.Button(row, text="clear", command=lambda: self.vars["beta_table"].set("")).pack(side=tk.LEFT)
                ttk.Label(box, textvariable=self.vars["beta_table"], wraplength=300).pack(anchor=tk.W)
            if section == "Solver":
                for label, attr, values in (("method", "method", ["auto", "dopri5", "bdf", "rk4"]),
                                            ("linear solver", "linear_solver", ["dense_lu", "gmres"])):
                    row = ttk.Frame(box)
                    row.pack(fill=tk.X)
                    ttk.Label(row, text=label, width=30).pack(side=tk.LEFT)
                    self.vars[attr] = tk.StringVar()
                    ttk.Combobox(row, textvariable=self.vars[attr], values=values, width=11,
                                 state="readonly").pack(side=tk.RIGHT)

        run_bar = ttk.Frame(left)
        run_bar.pack(fill=tk.X, pady=6)
        self.btn_run = ttk.Button(run_bar, text="Run", command=self.on_run)
        self.btn_run.pack(side=tk.LEFT)
        self.btn_cancel = ttk.Button(run_bar, text="Cancel", command=self.on_cancel, state=tk.DISABLED)
        self.btn_cancel.pack(side=tk.LEFT)
        ttk.Button(run_bar, text="Export CSV...", command=self.on_export).pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(left, maximum=1.0, length=300)
        self.progress.pack(fill=tk.X)
        self.status = tk.StringVar(value="ready")
        ttk.Label(left, textvariable=self.status, wraplength=320).pack(anchor=tk.W)

        nb = ttk.Notebook(right)
        nb.pack(fill=tk.BOTH, expand=True)
        plot_tab = ttk.Frame(nb)
        nb.add(plot_tab, text="Plots")
        self.fig = Figure(figsize=(9, 6.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_tab)
        NavigationToolbar2Tk(self.canvas, plot_tab).update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        sel = ttk.Frame(plot_tab)
        sel.pack(fill=tk.X)
        self.plot_sel = []
        for i, default in enumerate(["r(t)", "z(r)", "|v_r|(r)", "a(r)"]):
            v = tk.StringVar(value=default)
            ttk.Combobox(sel, textvariable=v, values=list(PLOTS), width=10, state="readonly").pack(side=tk.LEFT)
            v.trace_add("write", lambda *_: self.redraw())
            self.plot_sel.append(v)

        diag_tab = ttk.Frame(nb)
        nb.add(diag_tab, text="Summary & diagnostics")
        self.text = tk.Text(diag_tab, wrap=tk.WORD, font=("Menlo", 11))
        self.text.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------- config <-> UI --
    def _load_into_widgets(self) -> None:
        for _, fields in FIELDS:
            for _, attr, _t in fields:
                self.vars[attr].set(repr(getattr(self.cfg, attr)))
        for attr in ("drag", "method", "linear_solver", "beta_table", "disk_model", "perturbation", "charge"):
            self.vars[attr].set(getattr(self.cfg, attr))
        self.vars["coagulation"].set(self.cfg.coagulation)
        self.vars["sedimentation"].set(self.cfg.sedimentation)

    def _read_widgets(self) -> dp.RunConfig:
        cfg = self.cfg.copy()
        for _, fields in FIELDS:
            for label, attr, typ in fields:
                txt = self.vars[attr].get().strip()
                try:
                    val = typ(float(txt)) if typ is int else typ(txt)
                except ValueError:
                    raise ValueError(f"'{label}': cannot parse '{txt}'") from None
                if typ is float and not math.isfinite(val):
                    raise ValueError(f"'{label}' must be finite")
                setattr(cfg, attr, val)
        cfg.drag = self.vars["drag"].get()
        cfg.method = self.vars["method"].get()
        cfg.linear_solver = self.vars["linear_solver"].get()
        cfg.coagulation = bool(self.vars["coagulation"].get())
        cfg.sedimentation = bool(self.vars["sedimentation"].get())
        cfg.charge = self.vars["charge"].get()
        cfg.perturbation = self.vars["perturbation"].get()
        model = self.vars["disk_model"].get()
        if model == "tabulated_beta":
            cfg.beta_table = self.vars["beta_table"].get()
        else:
            cfg.beta_table = ""
            if model != "tabulated":
                cfg.disk_model = model
        # Validate exactly like the CLI/TOML path does.
        return dp.parse_config(cfg.to_toml())

    # ------------------------------------------------------------ actions --
    def on_reference(self) -> None:
        self.cfg = dp.reference_config()
        self._load_into_widgets()
        self.status.set("loaded reference case (Kobozeva 2025, 1 m from 100 au)")

    def on_load(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("TOML", "*.toml"), ("all", "*")])
        if not path:
            return
        try:
            self.cfg = dp.load_config(path)
        except Exception as e:  # noqa: BLE001 - show any configuration error
            messagebox.showerror("Load failed", str(e))
            return
        self._load_into_widgets()
        self.status.set(f"loaded {path}")

    def on_save(self) -> None:
        try:
            cfg = self._read_widgets()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Invalid parameters", str(e))
            return
        path = filedialog.asksaveasfilename(defaultextension=".toml", filetypes=[("TOML", "*.toml")])
        if path:
            dp.save_config(cfg, path)
            self.status.set(f"saved {path}")

    def on_beta(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("all", "*")])
        if path:
            self.vars["beta_table"].set(path)

    def on_run(self) -> None:
        try:
            cfg = self._read_widgets()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Invalid parameters", str(e))
            return
        self.cfg = cfg
        self.control = dp.RunControl()
        self.btn_run.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.NORMAL)
        self.status.set("running...")
        self.progress["value"] = 0

        def work(cfg: dp.RunConfig, ctl: dp.RunControl) -> None:
            try:
                self.done_q.put(dp.run(cfg, ctl))
            except Exception as e:  # noqa: BLE001 - forwarded to the GUI thread
                self.done_q.put(e)

        self.worker = threading.Thread(target=work, args=(cfg, self.control), daemon=True)
        self.worker.start()
        self.after(100, self._poll)

    def on_cancel(self) -> None:
        if self.control is not None:
            self.control.cancel()
            self.status.set("cancelling...")

    def _poll(self) -> None:
        if self.control is not None:
            self.progress["value"] = self.control.progress
        try:
            res = self.done_q.get_nowait()
        except queue.Empty:
            self.after(100, self._poll)
            return
        self.btn_run.config(state=tk.NORMAL)
        self.btn_cancel.config(state=tk.DISABLED)
        if isinstance(res, Exception):
            messagebox.showerror("Run failed", str(res))
            self.status.set("error")
            return
        self.result = res
        s = res["summary"]
        self.status.set(f"{s['status']} in {s['wall_time_s']:.3f} s"
                        + ("" if s["success"] else f" - {s['message']}"))
        if not s["success"] and s["status"] != "cancelled":
            messagebox.showwarning("Integration did not complete", f"{s['status']}: {s['message']}")
        self.show_summary()
        self.redraw()

    def on_export(self) -> None:
        if not self.result:
            messagebox.showinfo("Export", "Run a simulation first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        cols = self.result["columns"]
        names = list(cols)
        data = np.column_stack([cols[n] for n in names])
        np.savetxt(path, data, delimiter=",", header=",".join(names), comments="", fmt="%.17g")
        self.status.set(f"exported {len(data)} rows to {path}")

    # ---------------------------------------------------------- display --
    def show_summary(self) -> None:
        s = self.result["summary"]
        st = s["stats"]
        lines = [
            f"status              {s['status']}  {s['message']}",
            f"reached inner edge  {s['reached_inner_edge']}",
            f"t_drift             {s['t_drift_yr']:.6g} yr",
            f"t_settle (first)    {s['t_settle_first_yr']:.6g} yr",
            f"t_settle (last)     {s['t_settle_last_yr']:.6g} yr",
            f"max |v_r|           {s['vr_max_cms'] / 100:.6g} m/s at r = {s['r_at_vr_max_au']:.4g} au",
            f"a_final / a_max     {s['a_final_cm']:.6g} / {s['a_max_cm']:.6g} cm",
            f"Stokes regime seen  {s['stokes_regime_seen']} (max Re {s['max_Re']:.3g})",
            f"trapped             {s['trapped']}" + (f" at r = {s['r_trap_au']:.6g} au" if s['trapped'] else ""),
            "",
            f"final method {s['final_method']}; steps {st['n_steps']}, rejected "
            f"{st['n_reject_error']} (error) + {st['n_reject_newton']} (Newton), RHS {st['n_rhs']}, "
            f"Jacobians {st['n_jac']}, LU {st['n_lu']}, Newton it. {st['n_newton_iter']}, "
            f"lost increments {st['n_lost_increment']}",
            f"h_min {st['h_min']:.3g}, h_max {st['h_max']:.3g} (code units); wall {s['wall_time_s']:.4f} s",
        ]
        for w in st["switches"]:
            lines.append(f"switch {w['from']} -> {w['to']} at t = {w['t'] * dp.code_units()['t0_yr']:.6g} yr: "
                         f"{w['reason']}")
        for w in s["warnings"]:
            lines.append(f"WARNING: {w}")
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, "\n".join(lines))

    def redraw(self) -> None:
        self.fig.clear()
        if not self.result:
            self.canvas.draw_idle()
            return
        cols = self.result["columns"]
        for i, var in enumerate(self.plot_sel):
            ax = self.fig.add_subplot(2, 2, i + 1)
            xk, yk, xs, ys = PLOTS[var.get()]
            x, y = cols[xk], cols[yk]
            if yk == "vr_cms":
                y = np.abs(y)
            mask = np.isfinite(x) & np.isfinite(y)
            if xs == "log":
                mask &= x > 0
            if ys == "log":
                mask &= y > 0
            ax.plot(x[mask], y[mask], lw=1.2, color="#1f5f8b")
            ax.set_xscale(xs)
            ax.set_yscale(ys)
            ax.set_xlabel(xk)
            ax.set_ylabel(("|v_r| [cm/s]" if yk == "vr_cms" else yk))
            ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
