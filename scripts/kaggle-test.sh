#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/configure-env.sh"
credential="$KAGGLE_CONFIG_DIR/kaggle.json"

"$ML_LAB_HOME/.venv-train/bin/python" - "$credential" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("Kaggle credential: file not found")
    raise SystemExit(4)
try:
    value = json.loads(path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError):
    print("Kaggle credential: file found; JSON invalid")
    raise SystemExit(5)
present = [field for field in ("username", "key") if isinstance(value, dict) and field in value]
missing = [field for field in ("username", "key") if field not in present]
print("Kaggle credential: file found; JSON valid")
print("Required fields present: " + (", ".join(present) if present else "none"))
print("Required fields missing: " + (", ".join(missing) if missing else "none"))
if missing:
    raise SystemExit(6)
PY

"$ML_LAB_HOME/.venv-train/bin/kaggle" competitions list --page-size 1 >/dev/null
echo "Kaggle authentication: success"
