#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Build all pure seeded-Omega Zero-CUDA GB10 binaries (q38-decode, q38-serve, apu-fullchip)
# from the omega-mv commit pinned in omega-mv.lock.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
# shellcheck source=../omega-mv.lock
. "$ROOT/omega-mv.lock"

OMEGA_MV_DIR="${OMEGA_MV_DIR:-$ROOT/../omega-mv}"
OUT_DIR="${1:-$ROOT/build/q38}"
ALLOW_DRIFT="${ALLOW_OMEGA_MV_DRIFT:-0}"

if [ ! -d "$OMEGA_MV_DIR/.git" ]; then
  echo "error: omega-mv checkout not found at $OMEGA_MV_DIR" >&2
  echo "hint: git clone $OMEGA_MV_REPO $OMEGA_MV_DIR" >&2
  exit 3
fi

ACTUAL_COMMIT=$(git -C "$OMEGA_MV_DIR" rev-parse HEAD)
if [ "$ALLOW_DRIFT" != "1" ] && [ "$ACTUAL_COMMIT" != "$OMEGA_MV_COMMIT" ]; then
  echo "error: omega-mv commit mismatch" >&2
  echo "  pinned: $OMEGA_MV_COMMIT" >&2
  echo "  actual: $ACTUAL_COMMIT" >&2
  echo "hint: checkout the pinned commit or set ALLOW_OMEGA_MV_DRIFT=1" >&2
  exit 2
fi

OMC="${OMC:-$OMEGA_MV_DIR/seed/out/omc}"
if [ ! -x "$OMC" ]; then
  echo "==> Bootstrapping seeded omc from $OMEGA_MV_DIR/seed/omega-from-seed.sh"
  (cd "$OMEGA_MV_DIR" && sh seed/omega-from-seed.sh)
fi

mkdir -p "$OUT_DIR"
echo "==> Building q38 suite (q38-decode, q38-serve)"
(cd "$OMEGA_MV_DIR" && OMC="$OMC" sh om/gpu/q38/build.sh "$OUT_DIR")

echo "==> Building Full-Chip GB10 APU binary (apu-fullchip)"
(
  cd "$OMEGA_MV_DIR"
  (
    printf 'om2\n'
    cat gpu/nvidia/native/nvrm/context.om \
        gpu/nvidia/native/blackwell/vector.om \
        gpu/nvidia/native/blackwell/launch.om \
        gpu/nvidia/native/blackwell/batch.om \
        om/encoder/bw_enc_sm121.om \
        om/gpu/q36/emit_fast.om \
        om/gpu/apu_fullchip.om
  ) | "$OMC" > "$OUT_DIR/apu-fullchip"
)
chmod 755 "$OUT_DIR/apu-fullchip"

echo "==> Verifying zero-CUDA static AArch64 ELFs"
for bin in q38-serve q38-decode apu-fullchip; do
  path="$OUT_DIR/$bin"
  if [ ! -x "$path" ]; then
    echo "error: missing expected binary $path" >&2
    exit 1
  fi
  if command -v ldd >/dev/null 2>&1; then
    out=$(ldd "$path" 2>&1 || true)
    case "$out" in
      *"not a dynamic executable"*|*"statically linked"*) ;;
      *)
        echo "error: $bin is not statically linked: $out" >&2
        exit 1
        ;;
    esac
  fi
done

echo "==> Built static zero-CUDA binaries in $OUT_DIR (omega-mv @ $ACTUAL_COMMIT):"
ls -lh "$OUT_DIR"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$OUT_DIR"/*
elif command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "$OUT_DIR"/*
fi
