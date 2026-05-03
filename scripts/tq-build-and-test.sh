#!/usr/bin/env bash
# tq-build-and-test.sh — full build + functional + smoke harness for *qa fix.
#
# Builds CUDA shared lib (required after .cu/.cuh edits), Go binary, then
# kills any prior dev serves, starts a fresh one on a non-system port, and
# runs functional + smoke checks. Captures all output to a timestamped
# /tmp/tq-test-runs/ directory.
#
# Exit code 0 only if smoke passes AND all capital prompts produce coherent
# output (Paris/Madrid/Berlin substring match).

set -u
set -o pipefail

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$ROOT"

PORT=${TQ_TEST_PORT:-11445}
PRESET=${TQ_TEST_PRESET:-tq3}
MODELS=${OLLAMA_MODELS:-/home/mike/.ollama/models}
BINARY="$ROOT/dist/bin/ollama"
TS=$(date +%Y%m%d-%H%M%S)
RUNDIR=/tmp/tq-test-runs/$TS
mkdir -p "$RUNDIR"
SUMMARY="$RUNDIR/summary.txt"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUMMARY"; }
fail() { log "FAIL: $*"; exit 1; }

log "tq-build-and-test starting (port=$PORT preset=$PRESET rundir=$RUNDIR)"

# ── Step 1: Build CUDA shared lib ─────────────────────────────────────────────
log "Step 1: cmake build ggml-cuda"
if ! cmake --build build --target ggml-cuda -j 8 > "$RUNDIR/cuda-build.log" 2>&1; then
  log "CUDA build FAILED:"
  tail -50 "$RUNDIR/cuda-build.log" | tee -a "$SUMMARY"
  fail "cmake ggml-cuda"
fi
log "CUDA build OK"

# ── Step 2: Build Go binary ───────────────────────────────────────────────────
log "Step 2: go build dist/bin/ollama"
if ! go build -o "$BINARY" . > "$RUNDIR/go-build.log" 2>&1; then
  log "Go build FAILED:"
  tail -30 "$RUNDIR/go-build.log" | tee -a "$SUMMARY"
  fail "go build"
fi
log "Go build OK ($(stat -c '%s' "$BINARY") bytes)"

# ── Step 3: Kill prior dev serves on our ports (NOT the system 11434) ────────
log "Step 3: kill prior dev serves"
for OLDPID in $(pgrep -f "$BINARY") $(pgrep -f "/tmp/ollama-tq-fix/dist/bin/ollama"); do
  log "  killing pid=$OLDPID"
  kill "$OLDPID" 2>/dev/null || true
done
sleep 2

# ── Step 4: Start fresh serve ─────────────────────────────────────────────────
log "Step 4: start serve on 127.0.0.1:$PORT (preset=$PRESET, NEW_ENGINE=1)"
SERVE_LOG="$RUNDIR/serve.log"
OLLAMA_NEW_ENGINE=1 \
OLLAMA_KV_CACHE_TYPE="$PRESET" \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_HOST="127.0.0.1:$PORT" \
OLLAMA_MODELS="$MODELS" \
nohup "$BINARY" serve > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!
log "  serve pid=$SERVE_PID, log=$SERVE_LOG"

# Wait up to 60s for server to be ready
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/api/version" > /dev/null 2>&1; then
    log "  server ready in ${i}s"
    break
  fi
  sleep 1
done
if ! curl -sf "http://127.0.0.1:$PORT/api/version" > /dev/null 2>&1; then
  log "  server FAILED to start"
  tail -50 "$SERVE_LOG" | tee -a "$SUMMARY"
  fail "server startup"
fi

# ── Step 5: Trigger model load + check TQ_ACTIVATION ──────────────────────────
log "Step 5: warm-up + TQ_ACTIVATION check"
curl -sf "http://127.0.0.1:$PORT/api/generate" -d \
  "{\"model\":\"llama3.2:3b\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1,\"num_ctx\":512},\"keep_alive\":\"5m\"}" \
  > "$RUNDIR/warmup.json" 2>&1 || true

sleep 1
ACTIVATION_LINE=$(grep "TQ_ACTIVATION" "$SERVE_LOG" | head -1 || true)
if [ -z "$ACTIVATION_LINE" ]; then
  log "  no TQ_ACTIVATION line — preset never engaged"
  tail -30 "$SERVE_LOG" | tee -a "$SUMMARY"
  kill $SERVE_PID 2>/dev/null
  fail "TQ_ACTIVATION missing"
fi
log "  ACT: $ACTIVATION_LINE"
if ! grep -q "gpu_active=true" <<< "$ACTIVATION_LINE"; then
  fail "gpu_active != true"
fi
if ! grep -q "path=gpu-native" <<< "$ACTIVATION_LINE"; then
  fail "path != gpu-native"
fi

# ── Step 6: Functional capital test ───────────────────────────────────────────
log "Step 6: functional capital test"
declare -A EXPECT=( [France]="Paris" [Spain]="Madrid" [Germany]="Berlin" )
PASS_COUNT=0
FAIL_COUNT=0
for COUNTRY in France Spain Germany; do
  RESP_FILE="$RUNDIR/cap-$COUNTRY.json"
  curl -s --max-time 60 -o "$RESP_FILE" "http://127.0.0.1:$PORT/api/generate" -d \
    "{\"model\":\"llama3.2:3b\",\"prompt\":\"The capital of $COUNTRY is\",\"stream\":false,\"options\":{\"num_predict\":8,\"temperature\":0,\"num_ctx\":512}}"
  RESP=$(python3 -c "import json; print(json.load(open('$RESP_FILE'))['response'])" 2>/dev/null || echo "(parse error)")
  EXPECTED="${EXPECT[$COUNTRY]}"
  if [[ "$RESP" == *"$EXPECTED"* ]]; then
    log "  PASS: $COUNTRY -> $(printf '%q' "$RESP")"
    PASS_COUNT=$((PASS_COUNT + 1))
  else
    log "  FAIL: $COUNTRY -> $(printf '%q' "$RESP") (expected $EXPECTED)"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
done

# ── Step 7: Run smoke harness (separate ports) ────────────────────────────────
log "Step 7: tq-smoke-qa.sh (uses its own ports, kills serves between presets)"
kill $SERVE_PID 2>/dev/null
sleep 2
SMOKE_LOG="$RUNDIR/smoke.log"
if OLLAMA_BIN="$BINARY" OLLAMA_MODELS="$MODELS" OLLAMA_HOST="127.0.0.1:11447" SMOKE_MODEL="llama3.2:3b" \
   bash "$ROOT/scripts/tq-smoke-qa.sh" > "$SMOKE_LOG" 2>&1; then
  SMOKE_RESULT="PASS"
else
  SMOKE_RESULT="FAIL"
fi
log "  smoke: $SMOKE_RESULT"
tail -10 "$SMOKE_LOG" | tee -a "$SUMMARY"

# ── Step 8: Final summary ─────────────────────────────────────────────────────
log "================================================================"
log "RESULT: capital=$PASS_COUNT/3 PASS, smoke=$SMOKE_RESULT"
log "rundir: $RUNDIR"
log "================================================================"

if [ "$FAIL_COUNT" -gt 0 ] || [ "$SMOKE_RESULT" != "PASS" ]; then
  exit 1
fi
exit 0
