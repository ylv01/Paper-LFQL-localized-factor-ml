from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    build = ROOT / "scripts" / "build_processed_from_raw.py"
    code = subprocess.call([sys.executable, str(build)], cwd=ROOT)
    if code:
        return code
    script = ROOT / "src" / "experiments" / "localized_factor_ml.py"
    return subprocess.call([sys.executable, str(script)], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
