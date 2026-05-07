# TODO

The full 8-week plan with weekly gates lives in
[`research/PLAN.md`](research/PLAN.md). This file is the **current sprint**.

**Target:** ES-FoMo IV @ ICML 2026 (Seoul, Jul 10-11, 2026). Expected
submission deadline: late May / early June 2026.

---

## Week 0 — Pivot cleanup (COMPLETED 2026-05-07)

- [x] Write `research/PLAN.md` master plan
- [x] Update `README.md` to reflect new direction
- [x] Update `TODO.md` (this file)
- [x] Delete legacy code: `paretoq/`, `tequila/`, `run_pipeline.py`,
      `notebooks/`, `scripts/colab_ablate.py`, `scripts/colab_final.py`,
      `eval/src/eval/export_bitnet.py`, `exported_phi2.zip`,
      `tests/test_paretoq.py`, `tests/test_tequila.py`,
      `research/ternary_lora_inference_plan.md` (superseded by PLAN.md)
- [x] Remove `paretoq`, `tequila` from `pyproject.toml` workspace members
- [x] Strip `paretoq`/`tequila` imports from `tests/conftest.py`,
      `chat/src/chat/model_loader.py`, `scripts/colab_test.py`
- [x] Verify `pytest` passes after cleanup (41 passed)
- [x] Add `.gitignore` entries: `cache/`, `output/`, `*.zip`, `.venv/`,
      `*.safetensors` (in workdir, not in `tests/fixtures/`)
- [x] Rewrite `configs/pipeline_config.yaml` for current pipeline

## Week 1 — Baselines (CODE READY, needs GPU)

**Gate:** `results/baselines.md` populated; PT²-LLM is the key one.

- [x] Create `scripts/baselines/run_baselines.py` orchestrator
- [x] Create `results/baselines.md` template + evaluation protocol
- [ ] **GPTQModel 2-bit** on Phi-2 → run on Colab T4
- [ ] **HQQ 1-bit binary** group_size=64 on Phi-2 → run on Colab T4
- [ ] **AutoRound int2** with `--enable_alg_ext` on Phi-2 → run on Colab T4
- [ ] **PT²-LLM** clone + run on Phi-2 → run on Colab T4
- [ ] **PTQTP** if code released; if not, document and skip

## Week 2 — Cheap quality wins (IMPLEMENTED, needs GPU validation)

**Gate:** PPL ratio improved by ≥ 0.02× (1.216 → ≤ 1.196).

- [x] **OneBit SVID rank-1 scales** in `pt_bitnet/quantize.py:_symmetric_ternary`
      (added `use_svid_scales` flag to PTBitNetConfig; rank-1 α_row·β_col via
      SVD of |W|). DO NOT adopt OneBit's sign matrix.
- [x] **ApiQ-style LoRA init** in `pt_bitnet/lora.py`
      (`_apiq_initialize_lora` — truncated SVD of error matrix E=W_fp16-W_ternary,
      initializes LoRA from optimal rank-r approximation). Called automatically
      in `finetune_lora` when teacher is available.
- [ ] (optional) Random Hadamard rotation sanity check; drop if regresses
- [ ] Re-run pipeline on T4, validate PPL improvement

## Weeks 3-4 — BitNet Distillation (IMPLEMENTED, needs GPU training)

**Gate:** WikiText-2 PPL ratio ≤ 1.15× on Phi-2.

- [x] New file `pt_bitnet/src/pt_bitnet/subln.py` — `SubLN` RMSNorm,
      `insert_subln` with FQN regex auto-discovery across architectures,
      `remove_subln`, `count_subln`
- [x] Modify `pt_bitnet/lora.py` to add MiniLM Q/K/V attention distillation
      (`minilm_attention_loss`, `_capture_teacher_qkv_relations`,
      `_hook_student_qkv`; config flags: `use_minilm`, `minilm_weight`)
- [x] New script `scripts/warmup_subln.py` — 50M-token C4 streaming warm-up,
      SubLN-only training, checkpoint/resume, gradient checkpointing
- [ ] End-to-end run on T4: Phi-2 + SubLN + warm-up + PT-BitNet + KD w/ MiniLM

## Week 5 — Hybrid runtime spike (IMPLEMENTED, needs llama.cpp/T-MAC)

**Gate:** Coherent generation + ≥ 2× CPU speedup over llama.cpp Q8_0.

- [x] Implement `pt_bitnet/src/pt_bitnet/hybrid_runtime.py` —
      `HybridTernaryLinear` (ternary kernel + sparse outliers + LoRA),
      `PyTorchTernaryKernel` reference, `load_hybrid_model` from export,
      `benchmark_hybrid_layer`, Path A/B/C stubs
- [ ] Verify llama.cpp issue #15193 status in mai/2026 (needs GPU)
- [ ] If TQ2_0 works: wire llama-cpp-python backend
- [ ] If TQ2_0 broken: fall to Path B (T-MAC W2A16 GPTQ format)
- [ ] Benchmark CPU tok/s honestly

## Week 6 — Tooling + scaling curve

**Gate:** 5+ model sizes evaluated.

- [x] FQN-regex auto target-module discovery in `_find_quantizable_linears`
      (architecture-agnostic regex patterns + substring fallback)
- [ ] `TernaryConfig` dataclass + `from_pretrained` integration
- [ ] Run on Pythia-410M, Pythia-1.4B, TinyLlama-1.1B, Qwen2.5-0.5B,
      Phi-2 (existing) → needs GPU
- [ ] Append `results/scaling_curve.md`

## Week 7 — Paper

- [ ] 4-page draft, ICML template, double-blind
- [ ] Anonymized GitHub mirror with reproducibility script
- [ ] Submit to ES-FoMo IV

## Week 8 — Buffer

Rebuttal / camera-ready / ENLSP-VI fallback.

---

## Backlog (post-paper)

- 7B model support — requires CPU-offloaded teacher in calibration; skip in
  v1 paper, add for camera-ready or extended version
- Path C: custom AVX2 ternary kernel with our exact format
  (per-row α + outliers + LoRA, no reformatting). 300-500 LOC C via cffi.
  Reference: `bitnet.cpp/preset_kernels/.../bitnet-lut-kernels-AVX2.h`
- HF Hub auto-upload + GGUF export hook
- Imatrix-style importance file for calibration reuse across runs

---

## Permanently rejected (do not re-add)

See [`research/PLAN.md`](research/PLAN.md) Section 1.2 and Section 9 for the
full list with reasons. Summary:

- Tequila / UltraQuantV3 (re-decomposition incompatible with PT-BitNet denorm)
- ParetoQ / ZeroQAT (`sign()` is binary, not ternary)
- ITF asymmetric + outliers (closed-form denominator destabilizes)
- SSR column reordering (inverse permutation bug)
- AGA grid alignment (no improvement over symmetric)
- OBC row compensation (regresses on small models)
- Re-quantization after LoRA merge (breaks complementary relationship)
- distill_weight ≥ 0.5 (pulls model from correct predictions)
- KD temperature ≥ 3.0 (near-uniform teacher distribution)
- OneBit's sign matrix (incompatible with zero state — adopt only the SVID scale trick)
