# omega-qwen38-27b

Standalone pinned-dependency repository for running and benchmarking **Qwen 3.8 27B Dense** on the pure seeded-Omega Zero-CUDA GB10 engine (`om/gpu/q38/`) and Full-Chip APU stack (`omega-mv`).

## Boundaries
- Engine, compiler (`seed/out/omc`), SM121 kernels (`om/gpu/q38/`), and Full-Chip APU (`om/gpu/apu_fullchip.om`) live in `omega-mv` and are pinned via `omega-mv.lock`. Do not vendor copies of `omega-mv` source here.
- `evidence/` is additions-only: never modify, rename, or delete existing receipt files.
- Live GPU runs happen only on DGX Spark GB10 inside an exclusive `spark-window` with a board `CLAIM`/`RELEASE`.
- Zero CUDA userspace (`libcuda.so.1`, cuBLAS, libc, C, Rust, Python) is used on the inference path. NVIDIA open kernel modules (`580.173.02`) and signed GSP firmware are still used by the OS driver layer; always state this explicitly.
