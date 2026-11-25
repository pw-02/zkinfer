# Upstream v23 migration (experimental)

All three repositories use branch ccgrid-upstream-v23. Original ccgrid branches remain available. EZKL is based on upstream main e196b111 and Halo2 on upstream lock revision 01c8884. Use the pinned Halo2 dependency in EZKL rather than the old local Halo2/halo2curves path overrides. Upstream GPU feature is gpu-accelerated, with ICICLE runtime and backend installation required.

Build in a fresh virtual environment with upstream Rust toolchain and maturin: from the EZKL checkout, cargo check --features python-bindings; then maturin develop --release --features python-bindings. GPU build additionally enables gpu-accelerated and requires the upstream ICICLE setup. Check Cargo resolution for duplicate Halo2 types before proceeding. The lockfile source migration has not been verified by Cargo.

Use fresh setup, node cache, profiles and SRS directories; do not reuse old compiled circuits or keys. Start with MNIST and verify cold fused and warm ordinary proofs, then compare CPU/GPU on the same prepared circuit. Exact mode defaults to independent positive scales with deterministic boundary rescaling; check public instances and accuracy. The signed range is now limited to i64. Upstream removed tolerance and the proof_type argument from ordinary prove.

The fused KZG path and phase-transition CSV were ported; no build or proof test has run. Legacy FFT/MSM CSVs, per-phase CPU metrics, extended-domain structure fields, custom benchmark ports and GPU equivalence remain outstanding. Upstream GPU is retained intentionally. Do not use this branch for final paper results until validated.
