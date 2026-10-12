#!/usr/bin/env bash
# Idempotent bootstrap of the benchmark host. Everything goes under $ROOT (shared lab account).
#   usage: host_setup.sh <root> <src dir of the commit being run>
set -euo pipefail
ROOT="$1"; SRC="$2"
TORCH_PIN="2.7.1+cu128"
TRACKEVAL_SHA=9c25232f6f2b56c9f203f1eb55784ff1e97df683
CALIB_SHA=ab38f461bec729fead86b6986839de1bb826f16d
export UV_CACHE_DIR="$ROOT/cache/uv" PIP_CACHE_DIR="$ROOT/cache/pip" HF_HOME="$ROOT/cache/hf" \
  YOLO_CONFIG_DIR="$ROOT/cache/ultralytics" HF_HUB_DISABLE_XET=1
mkdir -p "$ROOT"/{bin,cache,data,runs,src,ext}

# Unmanaged install: no receipt in the shared ~/.config/uv, no PATH edits.
[ -x "$ROOT/bin/uv" ] || curl -LsSf https://astral.sh/uv/0.13.0/install.sh \
  | env UV_UNMANAGED_INSTALL="$ROOT/bin" sh

# Pipeline venv on the system torch. pip, not uv: uv ignores system site-packages and would
# pull a second torch. Code runs from $ROOT/src/<sha> (cwd first on sys.path); the install
# only brings the dependencies.
[ -x "$ROOT/venv/bin/python" ] || "$ROOT/bin/uv" venv -q --seed --system-site-packages \
  --python /usr/bin/python3 "$ROOT/venv"
"$ROOT/venv/bin/pip" install -q "$SRC[dev,calib]"
"$ROOT/venv/bin/python" - "$TORCH_PIN" <<'PY'
import sys, torch
assert torch.__version__ == sys.argv[1], f"torch {torch.__version__}, expected {sys.argv[1]}"
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, torch.cuda.get_device_name(0))
PY
echo "$TORCH_PIN" > "$ROOT/venv/.torch-version"

# Evaluator venv: third-party evaluators run here as subprocesses, never imported.
[ -x "$ROOT/venv-eval/bin/python" ] || "$ROOT/bin/uv" venv -q --python /usr/bin/python3 \
  "$ROOT/venv-eval"
VIRTUAL_ENV="$ROOT/venv-eval" "$ROOT/bin/uv" pip install -q numpy scipy tqdm \
  opencv-python-headless matplotlib
clone() {
  [ -d "$ROOT/ext/$1/.git" ] || git clone -q "https://github.com/SoccerNet/$1" "$ROOT/ext/$1"
  git -C "$ROOT/ext/$1" fetch -q origin && git -C "$ROOT/ext/$1" checkout -q "$2"
}
clone sn-trackeval "$TRACKEVAL_SHA"
clone sn-calibration "$CALIB_SHA"

# remote.env_fingerprint: a change to either file rebuilds the host on the next run
cat "$SRC/pyproject.toml" "$SRC/footlytics/bench/host_setup.sh" | sha256sum | cut -d' ' -f1 \
  > "$ROOT/venv/.deps-sha"
echo "host ready: $ROOT"
