# TODO

The full 8-week plan with weekly gates lives in
[`research/PLAN.md`](research/PLAN.md). This file is the **current sprint**.

**Target:** ES-FoMo IV @ ICML 2026 (Seoul, Jul 10-11, 2026). Expected
submission deadline: late May / early June 2026.

---

## Week 0 — Pivot cleanup (in progress, 2026-05-07)

- [x] Write `research/PLAN.md` master plan
- [x] Update `README.md` to reflect new direction
- [x] Update `TODO.md` (this file)
- [ ] Delete legacy code: `paretoq/`, `tequila/`, `run_pipeline.py`,
      `notebooks/`, `scripts/colab_ablate.py`, `scripts/colab_final.py`,
      `eval/src/eval/export_bitnet.py`, `exported_phi2.zip`,
      `tests/test_paretoq.py`, `tests/test_tequila.py`,
      `research/ternary_lora_inference_plan.md` (superseded by PLAN.md)
- [ ] Remove `paretoq`, `tequila` from `pyproject.toml` workspace members
- [ ] Strip `paretoq`/`tequila` imports from `tests/conftest.py`,
      `chat/src/chat/model_loader.py`, `scripts/colab_test.py`
- [ ] Verify `pytest` passes after cleanup
- [ ] Add `.gitignore` entries: `cache/`, `output/`, `*.zip`, `.venv/`,
      `*.safetensors` (in workdir, not in `tests/fixtures/`)

## Week 1 — Baselines

**Gate:** `results/baselines.md` populated; PT²-LLM is the key one.

- [ ] **GPTQModel 2-bit** on Phi-2 / WikiText-2 max_len=128
- [ ] **HQQ 1-bit binary** group_size=64 on Phi-2
- [ ] **AutoRound int2** with `--enable_alg_ext` on Phi-2
- [ ] **PT²-LLM** clone + run on Phi-2 ([repo](https://github.com/XIANGLONGYAN/PT2-LLM))
- [ ] **PTQTP** if code released; if not, document and skip
- [ ] Create `results/baselines.md` with one row per method:
      method, PPL, size, time-to-quantize, T4-VRAM-peak

## Week 2 — Cheap quality wins

**Gate:** PPL ratio improved by ≥ 0.02× (1.216 → ≤ 1.196).

- [ ] **OneBit SVID rank-1 scales** in `pt_bitnet/quantize.py`
      (replace per-row scalar α with α_row · β_col rank-1 outer product;
      compute β_col via SVD of |W|). DO NOT adopt OneBit's sign matrix.
- [ ] **ApiQ-style LoRA init** in `pt_bitnet/lora.py` (replace
      `torch.randn * 0.02` with activation-error-minimizing init from
      32 calibration sequences)
- [ ] (optional) Random Hadamard rotation sanity check; drop if regresses

## Weeks 3-4 — BitNet Distillation

**Gate:** WikiText-2 PPL ratio ≤ 1.15× on Phi-2.

- [ ] New file `pt_bitnet/src/pt_bitnet/subln.py` — insert RMSNorm before
      MHSA `o_proj`/`dense` and FFN `down_proj`/`fc2`
- [ ] Modify `pt_bitnet/lora.py` to add MiniLM Q/K/V attention distillation
      loss (single layer suffices per the paper)
- [ ] New script `scripts/warmup_subln.py` — 50M-token C4 warm-up
- [ ] End-to-end run + update `results/phi2_ternary.md`

## Week 5 — Hybrid runtime spike

**Gate:** Coherent generation + ≥ 2× CPU speedup over llama.cpp Q8_0.

- [ ] Patch `convert_hf_to_gguf.py` to force-emit TQ2_0 for non-BitNet archs
- [ ] Verify llama.cpp issue #15193 status in mai/2026 (1-day spike on Phi-2)
- [ ] If TQ2_0 works: implement `pt_bitnet/src/pt_bitnet/hybrid_runtime.py`
      with `HybridTernaryLinear` (ternary kernel + sparse outliers + LoRA)
- [ ] If TQ2_0 broken: fall to Path B (T-MAC W2A16 GPTQ format)
- [ ] Benchmark CPU tok/s honestly

## Week 6 — Tooling + scaling curve

**Gate:** 5+ model sizes evaluated.

- [ ] `TernaryConfig` dataclass + `from_pretrained` integration
- [ ] FQN-regex auto target-module discovery (replace per-arch hand-coding)
- [ ] Run on Pythia-410M, Pythia-1.4B, TinyLlama-1.1B, Qwen2.5-0.5B,
      Phi-2 (existing)
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
