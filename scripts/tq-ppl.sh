#!/usr/bin/env bash
# tq-ppl.sh — WikiText-2 perplexity sweep across TurboQuant KV cache presets.
#
# Uses cmd/perplexity (dist/bin/perplexity) which computes proper NLL-based
# perplexity: for each token position i, measures -log P(token_i | token_0..i-1)
# then reports PPL = exp(mean NLL). This is the same methodology used by
# KVQuant, KIVI, TurboQuant paper, and other KV compression papers.
#
# Usage:
#   ./scripts/tq-ppl.sh                          # all presets
#   PRESETS="f16 tq3 tq3k" ./scripts/tq-ppl.sh
#   WIKITEXT_FILE=/path/to/wiki.test.raw ./scripts/tq-ppl.sh
#   CTX=1024 ./scripts/tq-ppl.sh                 # larger context window
#
# Env:
#   ROOT            repo root (auto-detected)
#   PPL_BIN         path to perplexity binary (default: dist/bin/perplexity)
#   OLLAMA_LIBRARY_PATH  directory containing libggml-cuda.so (default: build/lib/ollama)
#   MODEL_FILE      GGUF model path (auto-detected from ollama blobs)
#   WIKITEXT_FILE   path to wiki.test.raw (downloaded to /tmp if missing)
#   PRESETS         space-separated list of presets to benchmark
#   CTX             tokens per chunk / context window (default: 512)
#   MODEL_NAME      ollama model name for auto-detection (default: llama3.2:3b)

set -euo pipefail

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
PPL_BIN=${PPL_BIN:-$ROOT/dist/bin/perplexity}
OLLAMA_LIBRARY_PATH=${OLLAMA_LIBRARY_PATH:-$ROOT/build/lib/ollama}
MODEL_NAME=${MODEL_NAME:-llama3.2:3b}
CTX=${CTX:-512}
MAX_BATCH=${MAX_BATCH:-512}
TS=$(date +%Y%m%d-%H%M%S)
RUNDIR=/tmp/tq-ppl-runs/$TS
mkdir -p "$RUNDIR"
RESULTS="$RUNDIR/ppl.csv"

DEFAULT_PRESETS="f16 tq2 tq3 tq4 tq2k tq3k tq4k"
PRESETS=${PRESETS:-$DEFAULT_PRESETS}

# ── Sanity checks ──────────────────────────────────────────────────────────
if [ ! -x "$PPL_BIN" ]; then
  echo "[ppl] ERROR: $PPL_BIN not found or not executable."
  echo "       Build with: go build -o dist/bin/perplexity ./cmd/perplexity"
  exit 1
fi

if [ ! -f "$OLLAMA_LIBRARY_PATH/libggml-cuda.so" ]; then
  echo "[ppl] ERROR: libggml-cuda.so not found in $OLLAMA_LIBRARY_PATH"
  echo "       Build with: cmake --build build -j 4"
  exit 1
fi

# ── Resolve MODEL_FILE ────────────────────────────────────────────────────
if [ -z "${MODEL_FILE:-}" ]; then
  # Parse ollama manifest to find the GGUF blob
  MODEL_SHORT="${MODEL_NAME%%:*}"          # "llama3.2"
  MODEL_TAG="${MODEL_NAME##*:}"            # "3b"
  MANIFEST="/home/mike/.ollama/models/manifests/registry.ollama.ai/library/$MODEL_SHORT/$MODEL_TAG"
  if [ ! -f "$MANIFEST" ]; then
    echo "[ppl] ERROR: ollama manifest not found: $MANIFEST"
    echo "       Run: ollama pull $MODEL_NAME"
    exit 1
  fi
  # Extract the largest blob (the GGUF model weights)
  BLOB_DIGEST=$(python3 -c "
import json, sys
with open('$MANIFEST') as f:
    d = json.load(f)
# find the largest layer (the model weights GGUF)
best = max(d.get('layers', []), key=lambda l: l.get('size', 0))
# digest is sha256:xxxx; blob path uses sha256-xxxx
print(best['digest'].replace(':', '-'))
")
  MODEL_FILE="/home/mike/.ollama/models/blobs/$BLOB_DIGEST"
  if [ ! -f "$MODEL_FILE" ]; then
    echo "[ppl] ERROR: model blob not found: $MODEL_FILE"
    exit 1
  fi
  echo "[ppl] model=$MODEL_NAME  blob=$MODEL_FILE"
fi

# ── WikiText-2 dataset ────────────────────────────────────────────────────
if [ -z "${WIKITEXT_FILE:-}" ]; then
  WIKITEXT_FILE=/tmp/wikitext-2-raw/wiki.test.raw
fi

if [ ! -f "$WIKITEXT_FILE" ]; then
  echo "[ppl] WikiText-2 test set not found at $WIKITEXT_FILE"
  echo "[ppl] Downloading from HuggingFace..."
  mkdir -p "$(dirname "$WIKITEXT_FILE")"
  # WikiText-2-raw from HuggingFace datasets mirror
  python3 - "$WIKITEXT_FILE" << 'PYEOF'
import sys, urllib.request, zipfile, io, os

out_path = sys.argv[1]
url = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet"
# Try parquet approach via Python pandas, fall back to direct text download
try:
    import pandas as pd
    print(f"Downloading WikiText-2 test split (parquet)...")
    df = pd.read_parquet(url)
    text = "\n".join(df["text"].tolist())
    with open(out_path, "w") as f:
        f.write(text)
    print(f"Downloaded {len(text):,} chars, {len(df)} rows")
except Exception as e:
    print(f"Parquet download failed ({e}), trying raw text...")
    # Fall back to raw text from the WikiText-2 raw dataset
    raw_url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt"
    try:
        with urllib.request.urlopen(raw_url, timeout=60) as resp:
            text = resp.read().decode("utf-8")
        with open(out_path, "w") as f:
            f.write(text)
        print(f"Downloaded {len(text):,} chars")
    except Exception as e2:
        print(f"ERROR: could not download WikiText-2: {e2}")
        print("Please set WIKITEXT_FILE=/path/to/wiki.test.raw manually.")
        sys.exit(1)
PYEOF
fi

WIKITEXT_CHARS=$(wc -c < "$WIKITEXT_FILE")
echo "[ppl] wikitext=$WIKITEXT_FILE  size=${WIKITEXT_CHARS} bytes"
echo "[ppl] ctx=$CTX  presets=$PRESETS"
echo

# ── CSV header ────────────────────────────────────────────────────────────
echo "preset,ppl,eval_tokens,prefill_tps,decode_tps,kv_mib,model_mib,total_mib" > "$RESULTS"

# ── Run perplexity for each preset ───────────────────────────────────────
run_preset() {
  local PRESET=$1
  echo "[ppl] preset=$PRESET ..."

  local OUT
  OUT=$(OLLAMA_LIBRARY_PATH="$OLLAMA_LIBRARY_PATH" \
        "$PPL_BIN" \
          --model "$MODEL_FILE" \
          --preset "$PRESET" \
          --ctx "$CTX" \
          --max-batch "$MAX_BATCH" \
        < "$WIKITEXT_FILE" 2>"$RUNDIR/stderr-${PRESET}.log")

  echo "  $OUT"

  # Parse output line: preset=f16       ppl= 7.3456  eval_tokens=12345  ...
  python3 - "$PRESET" "$OUT" "$RESULTS" << 'PYEOF'
import sys, re, csv

preset = sys.argv[1]
line   = sys.argv[2]
out_f  = sys.argv[3]

def grab(pattern, text, default=""):
    m = re.search(pattern, text)
    return m.group(1) if m else default

ppl        = grab(r'ppl=\s*([\d.]+)', line)
eval_tok   = grab(r'eval_tokens=(\d+)', line)
prefill    = grab(r'prefill=\s*([\d.]+)', line)
decode     = grab(r'decode=\s*([\d.]+)', line)
kv_mib     = grab(r'kv_mib=\s*(-?\d+)', line)
model_mib  = grab(r'model_mib=\s*(-?\d+)', line)
total_mib  = grab(r'total_mib=\s*(-?\d+)', line)

with open(out_f, "a", newline="") as f:
    csv.writer(f).writerow([preset, ppl, eval_tok, prefill, decode,
                             kv_mib, model_mib, total_mib])
PYEOF
}

for PRESET in $PRESETS; do
  run_preset "$PRESET"
done

# ── Summary table ─────────────────────────────────────────────────────────
echo
echo "================================================================"
python3 - "$RESULTS" << 'PYEOF'
import csv, sys, collections

with open(sys.argv[1]) as f:
    rows = list(csv.DictReader(f))

if not rows:
    print("No results.")
    sys.exit(0)

f16_ppl = None
for r in rows:
    if r["preset"] == "f16":
        try: f16_ppl = float(r["ppl"])
        except: pass

print(f"{'preset':<10}  {'ppl':>8}  {'vs_f16':>8}  {'kv_mib':>8}  {'decode_tps':>10}  {'prefill_tps':>11}")
print("-" * 64)
for r in rows:
    try: ppl = float(r["ppl"])
    except: ppl = float("nan")
    delta = f"{ppl - f16_ppl:+.4f}" if (f16_ppl and r["preset"] != "f16") else ("baseline" if r["preset"] == "f16" else "")
    print(f"{r['preset']:<10}  {ppl:>8.4f}  {delta:>8}  "
          f"{r.get('kv_mib','?'):>8}  {r.get('decode_tps','?'):>10}  {r.get('prefill_tps','?'):>11}")
PYEOF
echo
echo "Full results: $RESULTS"
echo "================================================================"
