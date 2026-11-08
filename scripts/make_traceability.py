"""Rebuild traceability.csv and results/traceability_summary.md from results/ (and the pytest JUnit file if present).

    python -I scripts/make_traceability.py      # exit code 1 if any trace is missing in either direction
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dms import trace  # noqa: E402


def main() -> int:
    if not trace.junit_verdicts():
        print("warning: results/pytest_junit.xml missing or empty; unit-test rows will read 未运行. "
              "Run: python -I -m pytest -q tests --junitxml=results/pytest_junit.xml")
    problems = trace.build()
    print("traceability:", "OK" if not problems else f"{len(problems)} problems")
    for p in problems:
        print("  -", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
