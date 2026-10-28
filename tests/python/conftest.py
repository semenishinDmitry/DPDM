import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
# The installed package must be tested, never the source tree: refuse to run if
# the import would resolve into python/ of the checkout.
import dustppd  # noqa: E402

assert not os.path.abspath(dustppd.__file__).startswith(os.path.join(REPO, "python")), \
    "tests must run against the installed package (pip install .)"
sys.path.insert(0, os.path.join(REPO, "tests", "reference"))
