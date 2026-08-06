import subprocess
import sys


def test_boundary_opt_does_not_load_knitting_or_polyscope() -> None:
    code = """
import sys
import boundary_opt
assert not any(name == 'knitting' or name.startswith('knitting.') for name in sys.modules)
assert not any(name == 'polyscope' or name.startswith('polyscope.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
