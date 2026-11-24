#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Launch the pure seeded-Omega Zero-CUDA Qwen 3.8 27B OpenAI-compatible HTTP server.
# Defaults: Full-W4 + 128K context window (pass --fp8 for Exact FP8).
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
BIN="${Q38_SERVE_BIN:-$ROOT/build/q38/q38-serve}"
MODEL_DIR="${Q38_MODEL_DIR:-/home/aienalpha/models/qwen3.8-27b-fp8-017b9c7a}"
PORT="${Q38_PORT:-18099}"
FLAGS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --fp8) FLAGS="$FLAGS --fp8"; shift ;;
    --w4) shift ;;
    --self-test)
      # shellcheck disable=SC2086
      exec "$BIN" --model-dir "$MODEL_DIR" $FLAGS --self-test
      ;;
    -h|--help)
      echo "usage: sh scripts/serve.sh [--model-dir PATH] [--port PORT] [--w4|--fp8] [--self-test]"
      exit 0
      ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ ! -x "$BIN" ]; then
  echo "error: $BIN not found; run 'sh scripts/build.sh' first" >&2
  exit 1
fi

# shellcheck disable=SC2086
exec "$BIN" --model-dir "$MODEL_DIR" --port "$PORT" $FLAGS
