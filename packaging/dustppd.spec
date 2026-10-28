# PyInstaller spec for the desktop application: pyinstaller packaging/dustppd.spec
# Produces dist/dustppd/ (Windows, Linux) and dist/dustppd.app (macOS).
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

import dustppd

HERE = Path(SPECPATH)
datas, binaries, hiddenimports = collect_all("dustppd")  # _core extension, web/, configs/

a = Analysis(
    [str(HERE / "dustppd_app.py")],
    datas=datas,
    binaries=binaries,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "scipy", "pytest", "PIL"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="dustppd",
    console=False,  # no terminal window; the GUI lives in the browser
    icon=str(HERE / "icon.png"),
)
coll = COLLECT(exe, a.binaries, a.datas, name="dustppd")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="dustppd.app",
        icon=str(HERE / "icon.png"),
        bundle_identifier="io.github.semenishindmitry.dustppd",
        info_plist={
            "CFBundleDisplayName": "dustppd",
            "CFBundleShortVersionString": dustppd.__version__,
            "CFBundleVersion": dustppd.__version__,
            # No Dock icon: the program has no window of its own, it serves the browser page.
            "LSUIElement": True,
            "NSHumanReadableCopyright": "MIT License",
        },
    )
