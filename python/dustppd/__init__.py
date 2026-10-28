"""dustppd: high-performance ODE solvers and the protoplanetary dust-dynamics model
of Kobozeva (2025, SPbU thesis) with a C++ core.

Main entry points
-----------------
- :func:`solve` -- integrate a Python right-hand side ``fun(t, y)``.
- :func:`solve_system` -- integrate a built-in C++ system (GIL released).
- :func:`load_config`, :class:`RunConfig`, :func:`run` -- the dust model.
- :func:`run_ensemble` -- parallel parameter sweeps.
- :func:`reference_config` -- the paper's reference case (1 m body from 100 au).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from ._core import (  # noqa: F401
    OdeSystem,
    RunConfig,
    RunControl,
    active_backend,
    available_backends,
    code_units,
    harmonic_oscillator,
    load_config,
    parse_config,
    reaction_diffusion,
    robertson,
    run as _run,
    run_ensemble,
    save_config,
    set_backend,
    solve,
    solve_system,
    van_der_pol,
    write_csv,
)

__version__ = "0.1.0"

_HERE = os.path.dirname(os.path.abspath(__file__))


class SolverError(RuntimeError):
    """Raised by :func:`run` with ``check=True`` when the integration fails."""

    def __init__(self, summary: Dict[str, Any]):
        super().__init__(f"integration failed: {summary['status']}: {summary['message']}")
        self.summary = summary


def reference_config_path() -> str:
    """Path of the paper's reference configuration (installed with the package)."""
    p = os.path.join(_HERE, "configs", "paper_1m_100au.toml")
    if os.path.exists(p):
        return p
    # source checkout
    return os.path.normpath(os.path.join(_HERE, "..", "..", "configs", "reference", "paper_1m_100au.toml"))


def reference_config(overrides: Optional[Sequence[str]] = None) -> RunConfig:
    """The paper's reference case: a 1 m body from r = 100 au, z = 3H (Sec. 3.3-3.4)."""
    return load_config(reference_config_path(), list(overrides or []))


def run(config: RunConfig, control: Optional[RunControl] = None, keep_trajectory: bool = True,
        check: bool = False) -> Dict[str, Any]:
    """Run one simulation. Returns ``{'summary': dict, 'columns': {name: ndarray}, ...}``.

    With ``check=True`` a :class:`SolverError` is raised on failure (status other
    than success / terminal event); otherwise inspect ``result['summary']['status']``.
    """
    res = _run(config, control, keep_trajectory)
    if check and not res["summary"]["success"]:
        raise SolverError(res["summary"])
    return res


def sweep(config: RunConfig, attribute: str, values: Sequence[float], threads: int = 0) -> List[Dict[str, Any]]:
    """Parameter sweep over one RunConfig attribute, run in parallel. Returns summaries."""
    cfgs = []
    for v in values:
        c = config.copy()
        setattr(c, attribute, v)
        cfgs.append(c)
    return run_ensemble(cfgs, threads)
