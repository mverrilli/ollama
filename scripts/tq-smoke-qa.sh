#!/usr/bin/env bash
#
# tq-smoke-qa.sh — smoke test that the 6 ship presets activate the
# GPU-native TurboQuant path at runtime and produce coherent output.
#
# The failure mode this guards against: silent fallback to f16 for a
# requested TQ preset produces PPL numbers bit-identical to f16 that
# look like "correct measurements". Every run must hit
# `TQ_ACTIVATION ... gpu_active=true path=gpu-native` for the requested
# preset, AND the model must produce coherent text (e.g. capital-of-France
# returns "Paris"), or this script exits non-zero.
#
# Usage:
#   ./scripts/tq-smoke-qa.sh                # run all 6 ship presets
#   ./scripts/tq-smoke-qa.sh tq3 tq2        # run a subset
#
# Env:
#   OLLAMA_BIN           path to the ollama binary to test (default: repo build)
#   OLLAMA_MODELS        models dir (default: /tmp/bench-ollama/models)
#   OLLAMA_HOST          host:port (default: 127.0.0.1:11435)
#   SMOKE_MODEL          model to load (default: llama3.2:3b)

set -u

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
OLLAMA_BIN=${OLLAMA_BIN:-$ROOT/dist/bin/ollama}
OLLAMA_MODELS=${OLLAMA_MODELS:-/tmp/bench-ollama/models}
HOST=${OLLAMA_HOST:-127.0.0.1:11435}
SMOKE_MODEL=${SMOKE_MODEL:-llama3.2:3b}

# The 6 ship presets exposed via OLLAMA_KV_CACHE_TYPE. Each is
# asymmetric+outliers, no QJL.
ALL_PRESETS=(tq2 tq3 tq4 tq2k tq3k tq4k)
PRESETS=("$@")
if [ ${#PRESETS[@]} -eq 0 ]; then
  PRESETS=("${ALL_PRESETS[@]}")
fi

if [ ! -x "$OLLAMA_BIN" ]; then
  echo "FATAL: OLLAMA_BIN=$OLLAMA_BIN is not executable. Build with: go build -o dist/bin/ollama ." >&2
  exit 2
fi

outdir=$(mktemp -d /tmp/tq-smoke-qa.XXXX)
echo "smoke logs -> $outdir"

fail=0
pass=0

assert_line() {
  local log=$1 preset=$2
  # Find the TQ_ACTIVATION line that mentions our preset.
  local line
  line=$(grep "TQ_ACTIVATION" "$log" | grep "preset=$preset" | tail -1)
  if [ -z "$line" ]; then
    echo "  FAIL: no TQ_ACTIVATION line in serve log for preset=$preset"
    return 1
  fi
  # Required fields.
  if ! grep -q 'gpu_active=true' <<< "$line"; then
    echo "  FAIL: gpu_active != true in: $line"
    return 1
  fi
  if ! grep -q 'path=gpu-native' <<< "$line"; then
    echo "  FAIL: path != gpu-native in: $line"
    return 1
  fi
  echo "  PASS: $line"
  return 0
}

for preset in "${PRESETS[@]}"; do
  echo "=== $preset ==="
  log="$outdir/$preset.log"

  # Cleanly stop any serve on our port.
  pid=$(ss -tlnp 2>/dev/null | awk -v p=":${HOST##*:}" '$4 ~ p {print $0}' | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
  [ -n "$pid" ] && kill "$pid" 2>/dev/null
  sleep 2

  nohup env \
    OLLAMA_NEW_ENGINE=1 \
    OLLAMA_MODELS="$OLLAMA_MODELS" \
    OLLAMA_HOST="$HOST" \
    OLLAMA_FLASH_ATTENTION=1 \
    OLLAMA_KV_CACHE_TYPE="$preset" \
    "$OLLAMA_BIN" serve > "$log" 2>&1 &
  serve_pid=$!
  disown "$serve_pid" 2>/dev/null || true

  # Wait for ready.
  for _ in $(seq 1 60); do
    curl -sf "http://$HOST/api/version" >/dev/null 2>&1 && break
    sleep 1
  done

  # Trigger model load: a single-token generate. This causes StartForward
  # → activateGPUEncode → TQ_ACTIVATION log on first Put.
  curl -sf "http://$HOST/api/generate" \
    -d "{\"model\":\"$SMOKE_MODEL\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1,\"num_ctx\":2048}}" \
    >/dev/null 2>&1 || echo "  warn: warm request failed (model missing?)"

  # Response coherence check: capital-of-France must contain "Paris".
  # Guards against quantisation incoherence where a model produces gibberish
  # while still emitting the TQ_ACTIVATION log line (the prior blind-spot).
  resp_file="$outdir/$preset-resp.json"
  curl -sf --max-time 60 "http://$HOST/api/generate" \
    -d "{\"model\":\"$SMOKE_MODEL\",\"prompt\":\"The capital of France is\",\"stream\":false,\"options\":{\"num_predict\":8,\"temperature\":0,\"num_ctx\":512}}" \
    -o "$resp_file" 2>/dev/null || true

  # Give the log a moment to flush.
  sleep 1
  kill "$serve_pid" 2>/dev/null
  wait "$serve_pid" 2>/dev/null

  ok=1
  if ! assert_line "$log" "$preset"; then
    ok=0
  fi

  RESP=$(python3 -c "import json; print(json.load(open('$resp_file'))['response'])" 2>/dev/null || echo "")
  if [[ "$RESP" == *"Paris"* ]]; then
    echo "  PASS response: $(printf '%q' "$RESP")"
  else
    echo "  FAIL response: $(printf '%q' "$RESP") (expected substring Paris)"
    ok=0
  fi

  if [ "$ok" -eq 1 ]; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    echo "  log: $log"
    grep -E 'level=ERROR|MEASUREMENT PRESET REQUESTED' "$log" | head -3 | sed 's/^/    /'
  fi
done

echo
echo "=== smoke summary: ${pass} PASS / ${fail} FAIL ==="
if [ "$fail" -gt 0 ]; then
  exit 1
fi
exit 0
