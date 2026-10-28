"""Entry point of the packaged desktop application (PyInstaller).

Starts the browser GUI in app mode: a tab opens in the default browser, the
program quits when the page is closed or on the page's "Quit" button, and a
second launch reopens the running instance."""
import sys

from dustppd.gui import main

if __name__ == "__main__":
    # Finder used to pass a process serial number (-psn_...) on old macOS versions.
    main(["--app", *[a for a in sys.argv[1:] if not a.startswith("-psn")]])
