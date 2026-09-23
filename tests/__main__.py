"""Run every test, each in its own process:  python -m tests

Several tests point the data root at a temporary folder for their own run, so
they are kept apart rather than imported into one interpreter together.
"""
import subprocess
import sys
from pathlib import Path

failed = [t.stem for t in sorted(Path(__file__).parent.glob("test_*.py"))
          if subprocess.run([sys.executable, "-m", f"tests.{t.stem}"]).returncode]
print("\nall passed" if not failed else "\nfailed: " + ", ".join(failed))
sys.exit(1 if failed else 0)
