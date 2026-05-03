#!/usr/bin/env bash
# tq-bench-throughput.sh — measure prefill/decode tok/s for the 6 TurboQuant
# ship presets vs f16 on llama3.2:3b. Reports VRAM at idle vs loaded.
# Single model only (fast).
#
# Run AFTER tq-build-and-test.sh confirms functional correctness.

set -u

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$ROOT"

PORT=${TQ_BENCH_PORT:-11448}
MODELS=${OLLAMA_MODELS:-/home/mike/.ollama/models}
BINARY="$ROOT/dist/bin/ollama"
TS=$(date +%Y%m%d-%H%M%S)
RUNDIR=/tmp/tq-bench-runs/$TS
mkdir -p "$RUNDIR"
RESULTS="$RUNDIR/throughput.csv"

PROMPT='The history of computing is a long and winding road that starts with mechanical calculators and abacuses, then through punch cards and mainframes, then minicomputers and microcomputers, and finally the personal computer revolution. Each step represented a fundamental shift in how humans interacted with machines. The transistor replaced vacuum tubes. Integrated circuits replaced individual transistors. Operating systems abstracted away hardware. Networks connected isolated machines into a global mesh. Each layer built on the last, accumulating capability and complexity in roughly equal measure. The current era of large language models continues this pattern: massive parallelism on commodity GPUs, training on internet-scale corpora, and the gradual reshaping of every desktop application as AI assistants take over routine cognitive work.'

echo "preset,prefill_tok_s,decode_tok_s,total_duration_s,kv_size_mib,compute_graph_mib" > "$RESULTS"

run_preset() {
  local PRESET=$1
  local NEW_ENGINE=${2:-1}
  local SERVE_LOG="$RUNDIR/serve-$PRESET.log"
  local RESP="$RUNDIR/resp-$PRESET.json"
  echo "[bench] preset=$PRESET new_engine=$NEW_ENGINE"

  pgrep -f "$BINARY" | xargs -r kill 2>/dev/null
  sleep 2

  OLLAMA_NEW_ENGINE=$NEW_ENGINE \
  OLLAMA_KV_CACHE_TYPE="$PRESET" \
  OLLAMA_FLASH_ATTENTION=1 \
  OLLAMA_HOST="127.0.0.1:$PORT" \
  OLLAMA_MODELS="$MODELS" \
  nohup "$BINARY" serve > "$SERVE_LOG" 2>&1 &
  local SERVE_PID=$!

  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/api/version" > /dev/null 2>&1 && break
    sleep 1
  done

  # Warm-up — load model
  curl -s --max-time 90 -o /dev/null "http://127.0.0.1:$PORT/api/generate" -d \
    "{\"model\":\"llama3.2:3b\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1,\"num_ctx\":2048},\"keep_alive\":\"10m\"}"

  # Measured run: ~140-token prefill, 64-token decode
  curl -s --max-time 120 -o "$RESP" "http://127.0.0.1:$PORT/api/generate" -d \
    "$(jq -nc --arg p "$PROMPT" '{model:"llama3.2:3b", prompt:$p, stream:false, options:{num_predict:64, temperature:0, num_ctx:2048}, keep_alive:"10m"}')"

  python3 <<EOF
import json, os
r = json.load(open("$RESP"))
prefill_count = r.get("prompt_eval_count", 0)
prefill_dur_ns = r.get("prompt_eval_duration", 1)
decode_count = r.get("eval_count", 0)
decode_dur_ns = r.get("eval_duration", 1)
total_dur_ns = r.get("total_duration", 1)
prefill_tps = prefill_count / (prefill_dur_ns / 1e9) if prefill_dur_ns > 0 else 0
decode_tps = decode_count / (decode_dur_ns / 1e9) if decode_dur_ns > 0 else 0
total_s = total_dur_ns / 1e9

# extract KV size + compute graph from serve log
kv_mib = 0
graph_mib = 0
import re
with open("$SERVE_LOG") as f:
    for line in f:
        if 'msg="kv cache"' in line:
            m = re.search(r'size="([\d.]+) MiB"', line)
            if m: kv_mib = float(m.group(1))
        if 'msg="compute graph"' in line and "CUDA" in line:
            m = re.search(r'size="([\d.]+) MiB"', line)
            if m: graph_mib = float(m.group(1))
print(f"  prefill={prefill_tps:.1f} tok/s  decode={decode_tps:.1f} tok/s  total={total_s:.2f}s  KV={kv_mib} MiB  graph={graph_mib} MiB")
with open("$RESULTS", "a") as out:
    out.write(f"$PRESET,{prefill_tps:.1f},{decode_tps:.1f},{total_s:.2f},{kv_mib},{graph_mib}\n")
EOF

  kill $SERVE_PID 2>/dev/null
  sleep 2
}

echo "[bench] starting throughput sweep -> $RESULTS"
run_preset f16 1
run_preset tq2 1
run_preset tq3 1
run_preset tq4 1
run_preset tq2k 1
run_preset tq3k 1
run_preset tq4k 1

echo
echo "================================================================"
echo "Results: $RESULTS"
column -ts, "$RESULTS"
echo "================================================================"
