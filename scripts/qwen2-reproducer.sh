#!/usr/bin/env bash
#
# qwen2-reproducer.sh — coherence gate for qwen2.5:7b under TurboQuant KV
# compression. Exits non-zero if any capital prompt fails to contain the
# expected city name.
#
# Originally written as a regression gate for the *qa Qwen2 incoherence
# bug (asymmetric encoder race, fixed in ac34ae96/0be9367a). Now used as
# a general coherence smoke against the 6 ship presets.
#
# Usage:
#   ./scripts/qwen2-reproducer.sh [preset]     # default: tq3 tq2
#   ./scripts/qwen2-reproducer.sh tq3 tq2 tq4
#
# Env:
#   OLLAMA_BIN     binary path (default: repo dist/bin/ollama)
#   OLLAMA_HOST    host:port   (default: 127.0.0.1:11448)
#   OLLAMA_MODELS  models dir  (default: /usr/share/ollama/.ollama/models)

set -u

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
OLLAMA_BIN=${OLLAMA_BIN:-$ROOT/dist/bin/ollama}
HOST=${OLLAMA_HOST:-127.0.0.1:11448}
OLLAMA_MODELS=${OLLAMA_MODELS:-/usr/share/ollama/.ollama/models}
MODEL="qwen2.5:7b"
PRESETS=("$@")
if [ ${#PRESETS[@]} -eq 0 ]; then
  PRESETS=(tq3 tq2)
fi

if [ ! -x "$OLLAMA_BIN" ]; then
  echo "FATAL: OLLAMA_BIN=$OLLAMA_BIN not executable. Build with: go build -o dist/bin/ollama ." >&2
  exit 2
fi

total_fail=0

for PRESET in "${PRESETS[@]}"; do
  echo "=== qwen2-reproducer: model=$MODEL preset=$PRESET ==="

  pgrep -f "$OLLAMA_BIN" | xargs -r kill 2>/dev/null
  sleep 2

  SERVE_LOG=$(mktemp /tmp/qwen2-reproducer-XXXX.log)
  OLLAMA_NEW_ENGINE=1 \
  OLLAMA_KV_CACHE_TYPE="$PRESET" \
  OLLAMA_FLASH_ATTENTION=1 \
  OLLAMA_HOST="$HOST" \
  OLLAMA_MODELS="$OLLAMA_MODELS" \
  nohup "$OLLAMA_BIN" serve > "$SERVE_LOG" 2>&1 &
  SERVE_PID=$!

  for _ in $(seq 1 60); do
    curl -sf "http://$HOST/api/version" >/dev/null 2>&1 && break
    sleep 1
  done
  if ! curl -sf "http://$HOST/api/version" >/dev/null 2>&1; then
    echo "  FATAL: server failed to start; log: $SERVE_LOG" >&2
    kill "$SERVE_PID" 2>/dev/null
    total_fail=$((total_fail + 3))
    continue
  fi

  preset_fail=0
  for ENTRY in "France:Paris" "Spain:Madrid" "Germany:Berlin"; do
    COUNTRY="${ENTRY%%:*}"
    EXPECTED="${ENTRY##*:}"
    RESP=$(curl -sf --max-time 60 "http://$HOST/api/generate" \
      -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of $COUNTRY is\",\"stream\":false,\"options\":{\"num_predict\":6,\"temperature\":0,\"num_ctx\":512}}" \
      | python3 -c "import sys,json; print(json.load(sys.stdin)['response'])" 2>/dev/null || echo "")
    if [[ "$RESP" == *"$EXPECTED"* ]]; then
      echo "  PASS $COUNTRY: $(printf '%q' "$RESP")"
    else
      echo "  FAIL $COUNTRY: $(printf '%q' "$RESP") (expected $EXPECTED)"
      preset_fail=$((preset_fail + 1))
    fi
  done

  kill "$SERVE_PID" 2>/dev/null
  wait "$SERVE_PID" 2>/dev/null

  ACT=$(grep "TQ_ACTIVATION" "$SERVE_LOG" | grep "preset=$PRESET" | tail -1)
  echo "  activation: ${ACT:-(none)}"
  echo "  log: $SERVE_LOG"

  if [ "$preset_fail" -gt 0 ]; then
    echo "  PRESET FAILED: $preset_fail/3 incoherent"
    total_fail=$((total_fail + preset_fail))
  else
    echo "  PRESET PASSED"
  fi
  echo
done

if [ "$total_fail" -gt 0 ]; then
  echo "=== FAILED: $total_fail prompt(s) incoherent ==="
  exit 1
fi
echo "=== PASSED: all qwen2.5:7b capital prompts coherent ==="
exit 0
