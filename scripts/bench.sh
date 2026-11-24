#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Reproducible DGX Spark GB10 benchmark suite for Qwen 3.8 27B Dense + Full-Chip APU.
# Must be run on DGX Spark inside an exclusive quiet spark-window:
#   WID=$(spark-window open omega-qwen38-27b-bench --lease 15 --gb10 exclusive --quiet exclusive --wait | tail -n 1)
#   trap 'spark-window close "$WID" >/dev/null 2>&1 || true' EXIT
#   spark-window run spark-receipt omega-qwen38-27b-bench -- sh scripts/bench.sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
BIN="${BIN_DIR:-$ROOT/build/q38}"
MODEL="${Q38_MODEL_DIR:-/home/aienalpha/models/qwen3.8-27b-fp8-017b9c7a}"
PORT="${Q38_BENCH_PORT:-18193}"
OUT="${BENCH_OUT:-$ROOT/build/bench-output.txt}"
mkdir -p "$(dirname "$OUT")"
: > "$OUT"
log() { echo "$1" | tee -a "$OUT"; }

log "=== ENVIRONMENT ==="
{
  date -u +%FT%TZ
  uname -a
  nvidia-smi --query-gpu=name,driver_version,clocks.current.sm,temperature.gpu,power.draw --format=csv
} | tee -a "$OUT"

log "=== 1. FULL-CHIP GB10 APU (20 CPU + 48 SM121 SMs) ==="
"$BIN/apu-fullchip" 2>&1 | tee -a "$OUT"
sleep 5

log "=== 2. LIVE SELF-TEST (q38-serve --self-test) ==="
"$BIN/q38-serve" --model-dir "$MODEL" --self-test 2>&1 | tee -a "$OUT"
sleep 8

log "=== 3. FULL-W4 DECODE (24 tok) ==="
"$BIN/q38-decode" "$MODEL" 24 w4 2>&1 | tee -a "$OUT"
sleep 8

log "=== 4. EXACT FP8 DECODE (24 tok) ==="
"$BIN/q38-decode" "$MODEL" 24 fp8 2>&1 | tee -a "$OUT"
sleep 8

log "=== 5. HMMA PREFILL 1024 TOKENS ==="
"$BIN/q38-decode" "$MODEL" 1024 p 2>&1 | tee -a "$OUT"
sleep 8

log "=== 6. HMMA PREFILL 4096 TOKENS ==="
"$BIN/q38-decode" "$MODEL" 4096 p 2>&1 | tee -a "$OUT"
sleep 8

log "=== 7. HMMA PREFILL 16384 TOKENS ==="
"$BIN/q38-decode" "$MODEL" 16384 p 2>&1 | tee -a "$OUT"
sleep 8

log "=== 8. CONTINUOUS BATCH-4 DECODE (32 steps) ==="
"$BIN/q38-decode" "$MODEL" 32 batch4 2>&1 | tee -a "$OUT"
sleep 8

log "=== 9. CONTINUOUS BATCH-8 DECODE (32 steps) ==="
"$BIN/q38-decode" "$MODEL" 32 batch8 2>&1 | tee -a "$OUT"
sleep 8

log "=== 10. SPECULATIVE VERIFY M=4 & M=8 ==="
"$BIN/q38-decode" "$MODEL" 32 spec4 2>&1 | tee -a "$OUT"
sleep 8
"$BIN/q38-decode" "$MODEL" 32 spec8 2>&1 | tee -a "$OUT"
sleep 8

log "=== 11. LIVE HTTP PROSE DECODE & 3K NEEDLE RECALL (q38-serve :$PORT) ==="
SRV_LOG="$ROOT/build/q38-serve-bench.log"
"$BIN/q38-serve" --model-dir "$MODEL" --port "$PORT" > "$SRV_LOG" 2>&1 &
SRV_PID=$!
cleanup_srv() { kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; }
trap cleanup_srv EXIT

ready=0
for _ in $(seq 1 90); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    log "ERROR: q38-serve exited early"
    cat "$SRV_LOG" | tee -a "$OUT"
    exit 1
  fi
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  log "ERROR: q38-serve not ready"
  cat "$SRV_LOG" | tee -a "$OUT"
  exit 1
fi

log "--- /health ---"
curl -s "http://127.0.0.1:$PORT/health" | tee -a "$OUT"; echo "" | tee -a "$OUT"

log "--- HTTP Short Factual Prose (max_tokens=32) ---"
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b-fp8","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32,"temperature":0}' | tee -a "$OUT"; echo "" | tee -a "$OUT"

log "--- HTTP Long-Form Prose Decode (max_tokens=256) ---"
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b-fp8","messages":[{"role":"user","content":"Write a clear, detailed technical essay explaining how unified-memory superchips couple CPU cores and GPU streaming multiprocessors over coherent interconnects, covering memory bandwidth ceilings, L2 cache locality, and autoregressive prose decoding latency."}],"max_tokens":256,"temperature":0}' | tee -a "$OUT"; echo "" | tee -a "$OUT"

log "--- HTTP 3K Needle-in-a-Haystack Recall ---"
python3 - "$PORT" << "PYEOF" | tee -a "$OUT"
import json, sys, urllib.request
port = sys.argv[1]
filler = ("The archive records routine telemetry from orbital relay stations across sectors one through nine, noting nominal thermal margins, stable optical downlink framing, and periodic solar array calibration sweeps without anomaly. " * 42)
prompt = filler + "CRITICAL RECORD: Captain Elena Vance authorized deployment code Aurora-742 at station Omega-9. " + filler + "Question: Who authorized deployment code Aurora-742, and at which station? Reply concisely."
body = json.dumps({
    "model": "qwen3.8-27b-fp8",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 48,
    "temperature": 0
}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as resp:
    print(resp.read().decode())
PYEOF

cleanup_srv
trap - EXIT
log "=== DONE ==="
