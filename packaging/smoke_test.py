"""Checks a packaged application: starts it without a browser, runs the paper's
reference case through the HTTP API, then quits it through the page's Quit call.

    python packaging/smoke_test.py <path to the dustppd executable>
"""
import json
import subprocess
import sys
import time
import urllib.request

PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"


def call(path, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    proc = subprocess.Popen([sys.argv[1], "--no-browser", "--port", str(PORT)])
    try:
        for _ in range(120):  # first start of a frozen app can be slow
            try:
                schema = call("/api/schema")
                break
            except OSError:
                if proc.poll() is not None:
                    print("application exited with code", proc.returncode)
                    return 1
                time.sleep(0.5)
        else:
            print("server did not start")
            return 1
        assert schema["app_mode"], "not started in app mode"
        print("started:", schema["version"], "SIMD", schema["backend"])
        params = call("/api/preset", {"id": "reference"})["params"]
        jid = call("/api/run", {"params": params})["id"]
        for _ in range(600):
            st = call(f"/api/status?id={jid}")
            if st["done"]:
                break
            time.sleep(0.1)
        s = st["result"]["summary"]
        t_kyr = s["t_drift_yr"] / 1e3
        print(f"reference case: {s['status']}, t_drift = {t_kyr:.1f} kyr")
        # Paper: 306 kyr; this code: 304.7 kyr (docs/validation.md).
        assert s["reached_inner_edge"] and abs(t_kyr - 304.7) < 1.0, s
        call("/api/quit", {})
        proc.wait(timeout=20)
        print("quit: exit code", proc.returncode)
        return 0
    finally:
        if proc.poll() is None:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
