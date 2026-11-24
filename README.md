# omega-qwen38-27b

**Qwen 3.8 27B Dense running on the Pure Seeded-Omega Zero-CUDA GB10 Engine & Full-Chip APU Stack.**

This repository is a standalone, pinned-dependency deployment and benchmark harness (`omega-mv.lock` → `omega-mv@f4f47af4c6ed`) for running **Qwen 3.8 27B Dense** (`Qwen3_5ForConditionalGeneration`: 64 layers, 48 Gated DeltaNet `16K/48V` + 16 Full-Attention `24Q/4KV` `hd=256`, `H=5120`, `I=17408`, `vocab=248,320`, `131,072`-token / `128K` context window) on **NVIDIA DGX Spark (GB10, SM 12.1)** with **zero CUDA userspace** (`ldd: not a dynamic executable` — zero `libcuda.so.1`, zero cuBLAS/cuDNN, zero `libc`, zero C, zero Rust, zero Python on the inference path).

> **Driver & Firmware Disclosure:** "Zero CUDA" means zero NVIDIA userspace libraries or CUDA toolkit. Execution still opens `/dev/nvidiactl`, `/dev/nvidia0`, and `/dev/nvidia-uvm` through Linux kernel open modules (`nvidia` / `nvidia-uvm` `580.173.02`) and NVIDIA's signed GSP firmware.

---

## Full-Chip GB10 APU & Pure-OM2 64-Layer 27B Engine (`om/gpu/q38/`)

Built directly from pinned `omega-mv` (`f4f47af4c6edb873a30acdab835cdc775b244b93`) via the self-hosted seeded compiler (`seed/out/omc`):

- **Full-Chip GB10 APU (`apu-fullchip`, `488 KB` static ELF):** Fires all **20 physical ARM CPU cores** (`10× Cortex-X925 @ 3.900 GHz` + `10× Cortex-A725 @ 2.808 GHz`) and all **48 Blackwell SM121 SMs** (`6,144` GPU cores) simultaneously over coherent NVLink-C2C (`4.422 T ops/s`, `BIT_EXACT=PASS`).
- **25 Custom SM121 SASS Kernels for 27B Dense (`om/gpu/q38/emit_q38.om`):**
  - `q38W4Gemv4R` (4 KiB streaming SMEM `K%1024==0` for `K=5120/6144/17408` + `lm_head`, 100% occupancy).
  - `q38W4GemmM4` (8 KiB streaming SMEM register-tiled W4 GEMM for Continuous Batch-4/8 & Speculative Verify `M=4/8`).
  - `q38GdnStepRealBf16` & zero-barrier warp-resident `q38PreGdnStepBatch` (`48` V heads / `16` K heads).
  - `q38FaAttendSplitF32` (Split-K `S=8` decode attention) & **Split-K `S=16` HMMA Flash-Attention-2** (`q38PreFaAttendSplitHmma`, `24Q/4KV`, `hd=256`).
  - `q38PreBf16GemmBatch` (`K=5120` HMMA BF16 GEMM for `in_proj_a/b`), on-GPU BF16→W4 `lm_head` quantizer & Two-Pass Exact rescorer (`PARITY=EXACT` vs Full BF16), dynamic Blackwell QMD word-36 SMEM carveout up to 100 KiB for `K=17408` Exact FP8.
- **Full-W4 + Exact FP8 Dual Precision (`q38-decode`, `q38-serve`):** On-GPU FP8 → block-32 INT4 quantization (`~13.8 GB` trunk + `0.72 GB` W4 `lm_head` + `16 × 1 GiB` 128K KV cache + `32 MB` RoPE + `384 MB` prefill workspace) with warm load in **`10.35 s`** and **24/24 bit-identical generated tokens** between Full-W4 and Exact FP8.
- **128K Context Window + OpenAI-Compatible HTTP Server (`q38-serve`):** Pure OM-2 `65,536`-bucket ByteLevel BPE tokenizer (`248,044` vocab) serving `GET /health`, `GET /v1/models`, and `POST /v1/chat/completions`.

---

## Live DGX Spark GB10 Silicon Benchmarks

Measured on **NVIDIA DGX Spark (GB10, `sm_121`, driver `580.173.02`, Linux `7.0.0-1019-nvidia` aarch64)** inside an exclusive quiet `spark-window` (`20261124T202524Z-53ee77`, `exit: 0`, `nvrm_errors_delta: 0`) on `2026-11-24`, running real weights from `/home/aienalpha/models/qwen3.8-27b-fp8-017b9c7a` built from `omega-mv@f4f47af4c6ed`:

- **Verified Quiet Window Receipt:** [gb10-q38-full-suite.19fedb86.json](evidence/gb10-q38-full-suite.19fedb864e9111f9b56bb17cd932bbd36cdda137371ab96cd914bc7ec1ead76e.json) · [raw log](evidence/gb10-q38-full-suite-log.bb38d154b45f44378b09613e25e464b89518e4ae6f4b86e41cae41ce112badbf.txt)

### 1. Prose Decode, Continuous Batched Decode & Speculative Verify (Before vs. After)

| Workload | Precision / Mode | Prompt / Gen | Prefill | Decode Throughput | Latency | Speedup vs. Baseline (`8.04 tok/s`) | Parity / Verification |
|---|---|---:|---:|---:|---:|---:|---|
| *Before: Unoptimized `omega-qwen38` (`aien-serve-http`)* | *Exact FP8 (legacy C/Rust)* | *short / 64 tok* | — | *`8.04 tok/s`* | *`124.4 ms/tok`* | *`1.00×`* | *Prior baseline (`PI_DRIVE_27B_SPARK.md`)* |
| **Short Factual Prose (Live HTTP)** | Full-W4 (`q38-serve`) | `19` / `8` tok | `236 ms` | **`13.86 tok/s`** | `72.13 ms/tok` | **`1.72×`** | Exact: `"The capital of France is **Paris**."` |
| **Long-Form Prose Decode (Live HTTP)** | Full-W4 (`q38-serve`) | `58` / `256` tok | `353 ms` | **`13.69 tok/s`** | `73.06 ms/tok` | **`1.70×`** | Coherent 256-tok technical essay (`finish_reason: stop`) |
| **CLI Decode + Parity (`N=1`)** | Full-W4 (`q38-decode w4`) | `17` / `24` tok | — | **`13.61 tok/s`** | `73.46 ms/tok` | **`1.69×`** | `PARITY=EXACT` vs Full BF16, **24/24 bit-identical to FP8** |
| **CLI Exact FP8 Decode (`N=1`)** | Exact FP8 (`q38-decode fp8`) | `17` / `24` tok | — | **`6.93 tok/s`** | `144.29 ms/tok` | `0.86×` | `PARITY=EXACT` vs Full BF16 (`tok=271`, warm load `15.47 s`) |
| **Continuous Batched Decode (`N=4`)** | Full-W4 (`batch4`, `q38W4GemmM4`) | `4 × 32` = `128` tok | — | **`36.49 tok/s`** (`9.12`/stream) | `109.60 ms/step` | **`4.54×`** (`2.68×` vs `N=1`) | 4 concurrent streams, Split-K `S=16` HMMA FA2 |
| **Continuous Batched Decode (`N=8`)** | Full-W4 (`batch8`, `q38W4GemmM4`) | `8 × 32` = `256` tok | — | **`42.00 tok/s`** (`5.25`/stream) | `190.46 ms/step` | **`5.22×`** (`3.09×` vs `N=1`) | 8 concurrent streams, Split-K `S=16` HMMA FA2 |
| **Speculative Verify (`M=4`)** | Full-W4 (`spec4`) | `4`-tok verify | — | **`36.72 tok/s`** | `27.23 ms/tok` | **`4.57×`** (`2.70×` vs `N=1`) | `acc=4.00 tok/step` ceiling, Split-K `S=16` HMMA FA2 |
| **Speculative Verify (`M=8`)** | Full-W4 (`spec8`) | `8`-tok verify | — | **`42.11 tok/s`** | `23.74 ms/tok` | **`5.24×`** (`3.09×` vs `N=1`) | `acc=8.00 tok/step` ceiling, Split-K `S=16` HMMA FA2 |

### 2. HMMA Chunked Prefill Sweep, 3K Needle Recall & 128K Self-Test

| Workload | Mode | Context / Prompt | Prefill Wall | Prefill Throughput | Decode (post-prefill) | Verification |
|---|---|---:|---:|---:|---:|---|
| **1K Chunked Prefill** | Full-W4 HMMA FA2 (`1024 p`) | `1,024` tok | `3,808 ms` | **`268 tok/s`** | — | `nvrm_errors_delta: 0` |
| **4K Chunked Prefill** | Full-W4 HMMA FA2 (`4096 p`) | `4,096` tok | `15,904 ms` | **`257 tok/s`** | — | `nvrm_errors_delta: 0` |
| **16K Chunked Prefill** | Full-W4 HMMA FA2 (`16384 p`) | `16,384` tok | `77,370 ms` | **`211 tok/s`** | — | `nvrm_errors_delta: 0` |
| **3K Needle-in-a-Haystack (Live HTTP)** | Full-W4 (`q38-serve`) | `2,994` tok / `22` gen | `11,519 ms` | **`259.9 tok/s`** | `12.46 tok/s` (`1,766 ms`) | Exact recall: `**Captain Elena Vance** authorized deployment code **Aurora-742** at **Station Omega**.` |
| **128K Context Self-Test** | Full-W4 (`q38-serve --self-test`) | `131,072` cap (`128K`) | — | — | — | `PASS` (Warm load `10.35 s`, Prompts 0 & 1 exact) |

### 3. Comparison vs. Other Stacks (DGX Spark GB10)

All stacks measured on the **same NVIDIA DGX Spark GB10** on Qwen 3.8 27B:
- **Pure Seeded-Omega `q38` (this repo, `2026-11-24`):** Window `20261124T202524Z-53ee77` (`nvrm_errors_delta: 0`).
- **Unoptimized Omega `aien-serve-http` (`PI_DRIVE_27B_SPARK.md`):** Prior C/Rust FP8 baseline (`8.04 tok/s` decode, `210 tok/s` 3.2K prefill, `206 tok/s` 13K prefill, `30.8×` warm prefix reuse).
- **Omega First-Cut Prefill (`~/yardsticks/qwen38-27b/RESULTS.md`, `2026-10-18`):** In-process FP8 prefill only (`197.7 s` @ 32K = `165.7 tok/s`, `415.3 s` @ 64K, `936.7 s` @ 128K).
- **TandemLLM v0.11.4 NVFP4 (`~/yardsticks/qwen38-27b/RESULTS.md`, `2026-10-18`):** CUDA NVFP4 4-bit + drafters (`47.8 s` @ 32K = `685.5 tok/s`, `92.6 s` @ 64K, `206.5 s` @ 128K).
- **CUDA + PyTorch `2.14.0+cu130` / `transformers 5.17.0` (`PI_DRIVE_27B_SPARK.md`):** Stock HuggingFace FP8 stack on GB10.

| Stack / Engine | Userspace Runtime | Load Time | Decode (`N=1`) | Batched (`N=4` / `N=8`) | Cold Prefill Throughput | Status / Notes |
|---|---|---:|---:|---:|---:|---|
| **Omega `q38-serve` Full-W4 (this repo)** | **Zero-CUDA static ELF (`938 KB`)** | **`10.35 s`** (warm) | **`13.69 tok/s`** (`13.61` CLI) | **`36.49` / `42.00 tok/s`** | **`268 tok/s`** (1K) · **`257 tok/s`** (4K) · **`211 tok/s`** (16K) | **`PASS` (`nvrm_errors_delta: 0`, 128K KV, exact 3K needle)** |
| **Omega `q38-decode` Exact FP8 (this repo)** | **Zero-CUDA static ELF (`928 KB`)** | `15.47 s` (warm) | `6.93 tok/s` | — | — | `PASS` (24/24 bit-identical tokens to Full-W4) |
| Unoptimized `omega-qwen38` (`aien-serve-http`) | Legacy C/Rust Zero-CUDA (FP8) | — | `8.04 tok/s` | — | `210 tok/s` (3.2K) · `206 tok/s` (13K) | Superseded by pure-OM2 `q38` (`1.70×` decode, `1.28×` prefill) |
| Omega First-Cut Prefill (`2026-10-18`) | Legacy FP8 prefill-only prototype | — | — | — | `165.7 tok/s` (`197.7 s` @ 32K) | Superseded (`1.27×` faster at 16K) |
| **TandemLLM v0.11.4** | CUDA NVFP4 (4-bit) + drafters | — | — | — | **`685.5 tok/s`** (`47.8 s` @ 32K, `206.5 s` @ 128K) | Faster long-context prefill via CUDA NVFP4 Tensor Cores (adds 812 hidden conductor tokens; not Zero-CUDA) |
| **CUDA + PyTorch `2.14.0+cu130`** | Python + PyTorch + `transformers 5.17.0` | `171.8 s` | **CRASH** | **CRASH** | **CRASH** | **`ImportError: finegrained-fp8 kernel unavailable`** on GB10 (`16.6×` slower load before crashing) |

### 4. Full-Chip GB10 APU Engine (`apu-fullchip`)

| Slice | Compute Units Active | Steps (64-bit LCG+XOR) | Wall Time | Throughput | Verification |
|---|---|---:|---:|---:|---|
| `GPU_48SM_ONLY_2B_SLICE` | 48 Blackwell SM121 SMs (6,144 cores) | `1,989,672,960` | `1,357 µs` | `4.399 T ops/s` | Reference GPU slice |
| `FULLCHIP_APU_20CPU_48SM_2B` | 20 ARM CPU + 48 Blackwell SM121 SMs | `2,000,000,000` | `1,360 µs` | `4.412 T ops/s` | `BIT_EXACT=PASS` |
| **`FULLCHIP_APU_20CPU_48SM_20B`** | **20 ARM CPU + 48 Blackwell SM121 SMs (6,164 cores)** | **`20,000,000,000`** | **`13,569 µs`** | **`4.422 T ops/s`** | **`BIT_EXACT=PASS`** |

### 5. Zero-CUDA Static Binary Manifest (`omega-mv@f4f47af4c6ed`)

All binaries are compiled purely by `seed/out/omc` (`not a dynamic executable` under `ldd`):

| Binary | Size | SHA-256 | Role |
|---|---:|---|---|
| `q38-serve` | `938 KB` | `7b9606f3ff90f312c89900eb4509fe914ddeba57f654240bb40593e3899120fb` | OpenAI HTTP server (Full-W4 / FP8, 128K KV, 64-layer 27B Dense, BPE tokenizer) |
| `q38-decode` | `928 KB` | `47aa2bba666bca5238e44b71958327e7670b75258f64d80f5ae468a43b02f677` | Real-weight CLI decoder & benchmark runner (`w4`, `fp8`, `p`, `batch4/8`, `spec4/8`) |
| `apu-fullchip` | `488 KB` | `e3da4ae889c141e5eb625ec46c2966987144bd84324c644d20c1f795e8484d64` | Full-Chip GB10 APU (20 CPU cores + 48 SM121 SMs over NVLink-C2C) |

---

## Quick Start (DGX Spark AArch64)

### 1. Build from Pinned `omega-mv`
```sh
git clone https://interplane.io/drakestapleton/omega-mv.git ../omega-mv
git -C ../omega-mv checkout f4f47af4c6edb873a30acdab835cdc775b244b93

sh scripts/build.sh
```

### 2. Serve Qwen 3.8 27B Dense (OpenAI-Compatible API)
```sh
# Full-W4 + 128K context on :18092
sh scripts/serve.sh --model-dir /home/aienalpha/models/qwen3.8-27b-fp8-017b9c7a --port 18092

# Exact FP8 mode
sh scripts/serve.sh --fp8 --port 18092

# Self-test (128K boundary)
sh scripts/serve.sh --self-test
```

### 3. Reproduce the Full Benchmark Suite
Run inside an exclusive quiet DGX Spark GPU window:
```sh
WID=$(spark-window open omega-qwen38-27b-bench --lease 15 --gb10 exclusive --quiet exclusive --wait | tail -n 1)
trap 'spark-window close "$WID" >/dev/null 2>&1 || true' EXIT
spark-window run spark-receipt omega-qwen38-27b-bench -- sh scripts/bench.sh
```

---

## License

Apache-2.0. See [LICENSE](LICENSE).
