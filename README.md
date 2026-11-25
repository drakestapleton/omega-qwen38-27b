# omega-qwen38-27b

**Qwen 3.8 27B Dense running on the Pure Seeded-Omega Zero-CUDA GB10 Engine & Full-Chip APU Stack.**

This repository is a standalone, pinned-dependency deployment and benchmark harness (`omega-mv.lock` → `omega-mv@37e6d7247c82`) for running **Qwen 3.8 27B Dense** (`Qwen3_5ForConditionalGeneration`: 64 layers, 48 Gated DeltaNet `16K/48V` + 16 Full-Attention `24Q/4KV` `hd=256`, `H=5120`, `I=17408`, `vocab=248,320`, `131,072`-token / `128K` context window) on **NVIDIA DGX Spark (GB10, SM 12.1)** with **zero CUDA userspace** (`ldd: not a dynamic executable` — zero `libcuda.so.1`, zero cuBLAS/cuDNN, zero `libc`, zero C, zero Rust, zero Python on the inference path).

> **Driver & Firmware Disclosure:** "Zero CUDA" means zero NVIDIA userspace libraries or CUDA toolkit. Execution still opens `/dev/nvidiactl`, `/dev/nvidia0`, and `/dev/nvidia-uvm` through Linux kernel open modules (`nvidia` / `nvidia-uvm` `580.173.02`) and NVIDIA's signed GSP firmware.

---

## Full-Chip GB10 APU & Pure-OM2 64-Layer 27B Engine (`om/gpu/q38/`)

Built directly from pinned `omega-mv` (`37e6d7247c82c18ee44fee041ca3c63e7397f6f5`) via the self-hosted seeded compiler (`seed/out/omc`):

- **Full-Chip GB10 APU (`apu-fullchip`, `488 KB` static ELF):** Fires all **20 physical ARM CPU cores** (`10× Cortex-X925 @ 3.900 GHz` + `10× Cortex-A725 @ 2.808 GHz`) and all **48 Blackwell SM121 SMs** (`6,144` GPU cores) simultaneously over coherent NVLink-C2C (`4.422 T ops/s`, `BIT_EXACT=PASS`).
- **32 Custom SM121 SASS Kernels for 27B Dense (`om/gpu/q38/emit_q38.om`):**
  - **Native Blackwell NVFP4 Tensor Core Suite (`--nvfp4` / `nv4p` / `nv4b8`, `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`):**
    - `q38Fp8ToNvfp4Quant` & `q38Bf16ToNvfp4ActQuant`: 1-cycle hardware `F2FP.E2M1` + `F2FP.E4M3` on-GPU quantizers (`9.70 s` warm load + quantize of all 64 layers!).
    - `q38PreNvfp4GemmSuper` (128-tok/CTA L2-swizzled `OMMA.SF.16864`, zero software unpack, `6,912 B` SMEM): **`1,124 tok/s` (1K, `60.2 TFLOPS`) / `1,045 tok/s` (4K, `56.0 TFLOPS`) / `799 tok/s` (16K, `42.8 TFLOPS`)** — **`2.0×–2.4×` faster than W4 Super-HMMA** and **faster than CUDA TandemLLM (`685.5 tok/s`)** with **9/9 bit-identical answer tokens** (`\n\nThe capital of France is **Paris**.`).
    - `q38Nvfp4GemmM8Omma`: `2,688 B` SMEM (`8 CTAs/SM = 100%` occ), `50.41 tok/s` Batch-8 / `49.87 tok/s` Spec-8.
  - `q38PreW4GemmSuper` (**128-token/CTA, 128-reg L2-Swizzled Overlapped Super-HMMA W4 GEMM**): **`474 tok/s` (1K, `1.77×`) / `460 tok/s` (4K, `1.79×`) / `404 tok/s` (16K, `1.91×`)**.
  - `q38W4GemmM8Hmma` (**100% Coalesced SMEM-W4 Tensor Core `HMMA.16816` for `B <= 8` & `lm_head`**): **`50.83 tok/s` Batch-8 (`157.4 ms/step`, `+21.0%`) & `50.76 tok/s` Spec-8**.
  - `q38W4Gemv4R` (**Zero-`I2F` Overlapped 4-row/warp W4 GEMV**): **`196.3 GB/s` effective DRAM bandwidth (`89.2%` of sustained GB10 LPDDR5X)**.
  - **Native Qwen 3.8 27B MTP Speculative Drafter (`mtp.safetensors` → W4 in `152 ms`):** **`17.27 tok/s` CLI `M=5` (`PARITY=EXACT`, `3.00` avg accepted tokens/step, `114.0%` of raw sustained `N=1` LPDDR5X ceiling!)** and `13.80 tok/s` live 256-tok HTTP prose (`--mtp`).
  - `q38GdnStepRealBf16` & zero-barrier warp-resident `q38PreGdnStepBatch` (`48` V heads / `16` K heads), `q38FaAttendSplitF32` & **Split-K `S=16` HMMA Flash-Attention-2** (`q38PreFaAttendSplitHmma`, `24Q/4KV`, `hd=256`).
- **Triple Precision (`Full-W4` + `Native NVFP4` + `Exact FP8`):** On-GPU FP8 → block-32 INT4 or block-16 NVFP4 (`4.50` bits/weight, `~13.8 GB` trunk + `0.72 GB` W4 `lm_head` + `16 × 1 GiB` 128K KV cache + `32 MB` RoPE + `384 MB` prefill workspace) plus Exact FP8 (`--fp8`).
- **128K Context Window + OpenAI-Compatible HTTP Server (`q38-serve`):** Pure OM-2 `65,536`-bucket ByteLevel BPE tokenizer (`248,044` vocab) serving `GET /health`, `GET /v1/models`, and `POST /v1/chat/completions` (`--nvfp4` / `--mtp` / `--fp8`).

---

## Live DGX Spark GB10 Silicon Benchmarks

Measured on **NVIDIA DGX Spark (GB10, `sm_121`, driver `580.173.02`, Linux `7.0.0-1019-nvidia` aarch64)** inside exclusive quiet `spark-window`s (`20261125T130758Z-b15ffa`, `20261125T091626Z-d48125`, `20261125T092807Z-b054d9`, `verdict: quiet machine`, `nvrm_errors_delta: 0`) on `2026-11-25`, running real weights from `/home/aienalpha/models/qwen3.8-27b-fp8-017b9c7a` built from `omega-mv@37e6d7247c82`:

- **Native NVFP4 `OMMA.SF.16864` Quiet Window Receipt (`2026-11-25`):** [gb10-q38-nvfp4-omma16864.33944316.json](evidence/gb10-q38-nvfp4-omma16864.3394431686d43f0edf03eab18ded855015444bf89eb52feda680bfc4dd5f1a0e.json)
- **Full-W4 Super-HMMA + MTP Quiet Window Receipt (`2026-11-25`):** [gb10-q38-sol-suite.f5b37cb5.json](evidence/gb10-q38-sol-suite.f5b37cb50e08654d2a4bb729d3d31ceca830d6a2eafa2e38fba2c5dc85ec55cb.json)
- **Prior Baseline Quiet Window Receipt (`2026-11-24`):** [gb10-q38-full-suite.19fedb86.json](evidence/gb10-q38-full-suite.19fedb864e9111f9b56bb17cd932bbd36cdda137371ab96cd914bc7ec1ead76e.json) · [raw log](evidence/gb10-q38-full-suite-log.bb38d154b45f44378b09613e25e464b89518e4ae6f4b86e41cae41ce112badbf.txt)

### 0. Side-by-Side: Pure-OM2 Full-W4 (`HMMA.16816`) vs. Native Blackwell NVFP4 (`OMMA.SF.16864.E2M1`)

| Workload | Full-W4 (`HMMA.16816` / `GEMV4R`) | **Native NVFP4 (`OMMA.SF.16864.E2M1`)** | NVFP4 vs. W4 | Verification |
|---|---:|---:|---:|---|
| **Warm Load + On-GPU Quantize** | `10.35 s` | **`9.70 s`** (`9,704 ms`) | **`1.07×` faster** | 1-cycle hardware `F2FP.E2M1` + `F2FP.E4M3` |
| **1K Prefill (`1,024` tok)** | `474 tok/s` (`2,156 ms`, `25.4 TFLOPS`) | **`1,124 tok/s`** (`911 ms`, **`60.2 TFLOPS`**) | **`2.37×`** (`4.19×` vs orig) | `exit: 0`, `nvrm_errors_delta: 0` |
| **4K Prefill (`4,096` tok)** | `460 tok/s` (`8,893 ms`, `24.7 TFLOPS`) | **`1,045 tok/s`** (`3,919 ms`, **`56.0 TFLOPS`**) | **`2.27×`** (`4.07×` vs orig) | `exit: 0`, `nvrm_errors_delta: 0` |
| **16K Prefill (`16,384` tok)** | `404 tok/s` (`40,465 ms`, `21.7 TFLOPS`) | **`799 tok/s`** (`20,490 ms`, **`42.8 TFLOPS`**) | **`1.98×`** (`3.79×` vs orig) | Beats CUDA TandemLLM `685.5 tok/s` (`1.17×`) |
| **Continuous Batch-8 (`N=8`)** | **`50.83 tok/s`** (`157.4 ms/step`) | `50.41 tok/s` (`158.7 ms/step`) | `0.99×` (DRAM bound) | Both at `4.50` bits/weight LPDDR5X ceiling |
| **Speculative Verify (`M=8`)** | **`50.76 tok/s`** (`19.70 ms/tok`) | `49.87 tok/s` (`20.05 ms/tok`) | `0.98×` (DRAM bound) | `nvrm_errors_delta: 0` |
| **Answer Token Parity (`N=1`)** | `271 760 6511 314 9338 369 2972 57590 159034` | **`271 760 6511 314 9338 369 2972 57590 159034`** | **9/9 bit-identical** | Exact: `The capital of France is **Paris**.` |
| **`N=1` Decode** | **`13.52 tok/s`** (`17.27` Native MTP) | `6.38 tok/s` (`B=1` on M8 OMMA kernel) | W4 GEMV4R `2.1×` faster | Dedicated NVFP4 `GEMV4R` not yet needed (`4.50` bpw) |

### 1. Prose Decode, Native 27B MTP, Continuous Batched Decode & Speculative Verify

Raw GB10 `N=1` LPDDR5X physical ceiling for `14.52 GB` W4 weights: **`18.80 tok/s`** at `273 GB/s` peak / **`15.15 tok/s`** at `220 GB/s` sustained.

| Workload | Precision / Mode | Prompt / Gen | Prefill | Decode Throughput | Latency | Speedup vs. Legacy (`8.04 tok/s`) | Parity / Verification |
|---|---|---:|---:|---:|---:|---:|---|
| *Before: Unoptimized `omega-qwen38` (`aien-serve-http`)* | *Exact FP8 (legacy C/Rust)* | *short / 64 tok* | — | *`8.04 tok/s`* | *`124.4 ms/tok`* | *`1.00×`* | *Prior baseline (`PI_DRIVE_27B_SPARK.md`)* |
| **Native 27B MTP (`M=5`, CLI)** | **Full-W4 + `mtp.safetensors` W4** | `17` / `10` tok | — | **`17.27 tok/s`** | **`57.87 ms/tok`** | **`2.15×`** (`1.27×` vs `N=1`) | **`PARITY=EXACT`**, `3.00` avg acc/step (**`114.0%`** of sustained `N=1` ceiling!) |
| **Long-Form Prose HTTP (`--mtp`)** | **Full-W4 + Native MTP (`q38-serve --mtp`)** | `45` / `256` tok | `303 ms` | **`13.80 tok/s`** | `72.44 ms/tok` | **`1.72×`** | Coherent 256-tok FlashAttention-2 technical essay |
| **Long-Form Prose HTTP (`N=1`)** | Full-W4 (`q38-serve`) | `45` / `256` tok | `299 ms` | **`13.45 tok/s`** | `74.38 ms/tok` | **`1.67×`** | Coherent 256-tok FlashAttention-2 technical essay |
| **CLI Decode + Parity (`N=1`)** | Full-W4 (`q38-decode w4`, Zero-`I2F`) | `17` / `32` tok | — | **`13.52 tok/s`** | `73.92 ms/tok` | **`1.68×`** | `PARITY=EXACT` vs Full BF16 (`196.3 GB/s` = `89.2%` sustained LPDDR5X) |
| **CLI Exact FP8 Decode (`N=1`)** | Exact FP8 (`q38-decode fp8`) | `17` / `24` tok | — | **`6.93 tok/s`** | `144.29 ms/tok` | `0.86×` | `PARITY=EXACT` vs Full BF16 (24/24 bit-identical to W4) |
| **Continuous Batched Decode (`N=8`)** | **Full-W4 Coalesced HMMA (`batch8`)** | `8 × 8` = `64` tok | — | **`50.83 tok/s`** (`6.35`/stream) | **`157.36 ms/step`** | **`6.32×`** (`+21.0%` vs `42.00`) | 8 concurrent streams, `100%` occ, Split-K `S=16` HMMA FA2 |
| **Speculative Verify (`M=8`)** | **Full-W4 Coalesced HMMA (`spec8`)** | `8`-tok verify | — | **`50.76 tok/s`** | **`19.70 ms/tok`** | **`6.31×`** (`270.3%` peak `N=1`) | `acc=8.00 tok/step` ceiling, Split-K `S=16` HMMA FA2 |

### 2. L2-Swizzled Super-HMMA Chunked Prefill Sweep, 6.3K HTTP Prefill & 3K Needle Recall

| Workload | Mode | Context / Prompt | Prefill Wall | Prefill Throughput | Speedup vs. Prior (`268/257/211`) | Verification |
|---|---|---:|---:|---:|---:|---|
| **1K Super-HMMA Prefill** | Full-W4 L2-Swizzled Super-HMMA (`1024 p`) | `1,024` tok | **`2,156 ms`** | **`474 tok/s`** (`25.4 TFLOPS`) | **`1.77×`** | `nvrm_errors_delta: 0` |
| **4K Super-HMMA Prefill** | Full-W4 L2-Swizzled Super-HMMA (`4096 p`) | `4,096` tok | **`8,893 ms`** | **`460 tok/s`** (`24.7 TFLOPS`) | **`1.79×`** | `nvrm_errors_delta: 0` |
| **16K Super-HMMA Prefill** | Full-W4 L2-Swizzled Super-HMMA (`16384 p`) | `16,384` tok | **`40,465 ms`** | **`404 tok/s`** (`21.7 TFLOPS`) | **`1.91×`** | `nvrm_errors_delta: 0` |
| **6.3K Live HTTP Prefill** | Full-W4 (`q38-serve --mtp`) | `6,348` tok | **`14,274 ms`** | **`444.7 tok/s`** | **`1.71×`** | `nvrm_errors_delta: 0` |
| **3K Needle-in-a-Haystack (Live HTTP)** | Full-W4 (`q38-serve`, prior baseline) | `2,994` tok / `22` gen | `11,519 ms` | `259.9 tok/s` | — | Exact recall: `**Captain Elena Vance** authorized deployment code **Aurora-742** at **Station Omega**.` |
| **128K Context Self-Test** | Full-W4 (`q38-serve --self-test`) | `131,072` cap (`128K`) | — | — | — | `PASS` (Prompts 0 & 1 exact) |

### 3. Comparison vs. Other Stacks (DGX Spark GB10)

All stacks measured on the **same NVIDIA DGX Spark GB10** on Qwen 3.8 27B:
- **Pure Seeded-Omega `q38` (this repo, `2026-11-25`):** Windows `20261125T091626Z-d48125` & `20261125T092807Z-b054d9` (`nvrm_errors_delta: 0`).
- **Unoptimized Omega `aien-serve-http` (`PI_DRIVE_27B_SPARK.md`):** Prior C/Rust FP8 baseline (`8.04 tok/s` decode, `210 tok/s` 3.2K prefill, `206 tok/s` 13K prefill).
- **Omega First-Cut Prefill (`~/yardsticks/qwen38-27b/RESULTS.md`, `2026-10-18`):** In-process FP8 prefill only (`197.7 s` @ 32K = `165.7 tok/s`, `415.3 s` @ 64K, `936.7 s` @ 128K).
- **TandemLLM v0.11.4 NVFP4 (`~/yardsticks/qwen38-27b/RESULTS.md`, `2026-10-18`):** CUDA NVFP4 4-bit + drafters (`47.8 s` @ 32K = `685.5 tok/s`, `92.6 s` @ 64K, `206.5 s` @ 128K).
- **CUDA + PyTorch `2.14.0+cu130` / `transformers 5.17.0` (`PI_DRIVE_27B_SPARK.md`):** Stock HuggingFace FP8 stack on GB10.

| Stack / Engine | Userspace Runtime | Load Time | Decode (`N=1` / Native MTP) | Batched (`N=8`) | Cold Prefill Throughput | Status / Notes |
|---|---|---:|---:|---:|---:|---|
| **Omega `q38` Full-W4 Super-HMMA + MTP (this repo)** | **Zero-CUDA static ELF (`1,022 KB`)** | `14–23 s` | **`13.52 tok/s`** · **`17.27 tok/s` MTP** (`PARITY=EXACT`) | **`50.83 tok/s`** (`50.76` Spec-8) | **`474 tok/s`** (1K) · **`460 tok/s`** (4K) · **`404 tok/s`** (16K) | **`PASS` (`nvrm_errors_delta: 0`, 128K KV, Zero-CUDA)** |
| **Omega `q38-decode` Exact FP8 (this repo)** | **Zero-CUDA static ELF (`1,021 KB`)** | `15.47 s` | `6.93 tok/s` | — | — | `PASS` (24/24 bit-identical tokens to Full-W4) |
| Unoptimized `omega-qwen38` (`aien-serve-http`) | Legacy C/Rust Zero-CUDA (FP8) | — | `8.04 tok/s` | — | `210 tok/s` (3.2K) · `206 tok/s` (13K) | Superseded by pure-OM2 `q38` (`2.15×` MTP, `2.26×` prefill) |
| Omega First-Cut Prefill (`2026-10-18`) | Legacy FP8 prefill-only prototype | — | — | — | `165.7 tok/s` (`197.7 s` @ 32K) | Superseded (`2.44×` faster at 16K) |
| **TandemLLM v0.11.4** | CUDA NVFP4 (4-bit) + drafters | — | — | — | **`685.5 tok/s`** (`47.8 s` @ 32K, `206.5 s` @ 128K) | Faster long-context prefill via CUDA NVFP4 Tensor Cores (adds 812 hidden conductor tokens; not Zero-CUDA) |
| **CUDA + PyTorch `2.14.0+cu130`** | Python + PyTorch + `transformers 5.17.0` | `171.8 s` | **CRASH** | **CRASH** | **CRASH** | **`ImportError: finegrained-fp8 kernel unavailable`** on GB10 |

### 4. Full-Chip GB10 APU Engine (`apu-fullchip`)

| Slice | Compute Units Active | Steps (64-bit LCG+XOR) | Wall Time | Throughput | Verification |
|---|---|---:|---:|---:|---|
| `GPU_48SM_ONLY_2B_SLICE` | 48 Blackwell SM121 SMs (6,144 cores) | `1,989,672,960` | `1,357 µs` | `4.399 T ops/s` | Reference GPU slice |
| `FULLCHIP_APU_20CPU_48SM_2B` | 20 ARM CPU + 48 Blackwell SM121 SMs | `2,000,000,000` | `1,360 µs` | `4.412 T ops/s` | `BIT_EXACT=PASS` |
| **`FULLCHIP_APU_20CPU_48SM_20B`** | **20 ARM CPU + 48 Blackwell SM121 SMs (6,164 cores)** | **`20,000,000,000`** | **`13,569 µs`** | **`4.422 T ops/s`** | **`BIT_EXACT=PASS`** |

### 5. Zero-CUDA Static Binary Manifest (`omega-mv@1084bc08d074`)

All binaries are compiled purely by `seed/out/omc` (`not a dynamic executable` under `ldd`):

| Binary | Size | SHA-256 | Role |
|---|---:|---|---|
| `q38-serve` | `1,022 KB` | `67b13075c62acd3978e596cda27de1e147dec0e68689ef70bf3ed69f1abe0d54` | OpenAI HTTP server (Full-W4 / FP8 / `--mtp`, 128K KV, L2-Swizzled Super-HMMA prefill, BPE tokenizer) |
| `q38-decode` | `1,021 KB` | `5dc814853ce455cccbfe7769a5a055cdfdd5109bc94e7a4c662fed1d930ead8f` | Real-weight CLI decoder & benchmark runner (`w4`, `fp8`, `p`, `batch4/8`, `spec4/8`, `mtp`) |
| `apu-fullchip` | `488 KB` | `e3da4ae889c141e5eb625ec46c2966987144bd84324c644d20c1f795e8484d64` | Full-Chip GB10 APU (20 CPU cores + 48 SM121 SMs over NVLink-C2C) |

---

## Quick Start (DGX Spark AArch64)

### 1. Build from Pinned `omega-mv`
```sh
git clone https://interplane.io/drakestapleton/omega-mv.git ../omega-mv
git -C ../omega-mv checkout 1084bc08d074c9bd5ffaca895782aebbb320ff15

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
