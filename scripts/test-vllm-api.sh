#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/configure-env.sh"
port="${1:-8000}"

"$ML_LAB_HOME/.venv-vllm/bin/python" - "$port" <<'PY'
import json
import os
import sys
import urllib.request

port = int(sys.argv[1])
base = f"http://127.0.0.1:{port}"
headers = {"Content-Type": "application/json"}
if os.environ.get("VLLM_API_KEY"):
    headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]

with urllib.request.urlopen(urllib.request.Request(base + "/v1/models", headers=headers), timeout=15) as response:
    model = json.load(response)["data"][0]["id"]
payload = json.dumps({
    "model": model,
    "prompt": "In one sentence, a GPU is",
    "max_tokens": 16,
    "temperature": 0.0,
}).encode()
request = urllib.request.Request(base + "/v1/completions", data=payload, headers=headers)
with urllib.request.urlopen(request, timeout=60) as response:
    body = json.load(response)
text = body["choices"][0]["text"].strip()
if not text:
    raise SystemExit("vLLM returned an empty completion")
print("vLLM API smoke test: success")
print("Generated text:", text)
PY
