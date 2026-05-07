# TernaryBoost — Master Plan (2026-05-07 → ES-FoMo IV submission)

> **Audience for this document:** any engineer or AI agent inheriting this
> project without prior context. Everything needed to execute is in this file
> plus the existing code in `pt_bitnet/`, `chat/`, `eval/`, `shared/`. No
> external context required.

---

## 0. One-page summary

TernaryBoost is a post-training ternarization (1.58-bit) pipeline for arbitrary
HuggingFace causal LMs, designed to run on a single Colab T4. Current best
result on Phi-2: **WikiText-2 PPL ratio 1.216×** (FP16 27.39 → ternary 33.31)
with attention+FFN LoRA recovery, ~47 min pipeline, INT2 packed export.

**The problem:** that result is competitive but not novel. PT²-LLM (arXiv
2510.03267) and PTQTP (arXiv 2509.16989) already report comparable or better
ternary PTQ numbers on larger models. The project as currently framed cannot
publish.

**The pivot:** stop competing on a single quality metric. Build the only thing
that doesn't yet exist as a polished, T4-feasible artifact:

> **A universal post-training ternarization pipeline for any HF causal LM, with
> distillation-based quality recovery and a hybrid CPU runtime that combines a
> ternary kernel + sparse FP16 outliers + LoRA adapters at inference time.**

This is workshop-paper-shaped (target: ES-FoMo IV @ ICML 2026, July 10-11,
Seoul). The novelty is the **runtime architecture** (ternary kernel + sparse
outliers + LoRA-at-inference, all stitched together in a Python wrapper) — a
combination that has SpQR and QLoRA precedents individually but never together
for ternary.

**Three axes of contribution:**
1. **Quality** — adopt BitNet Distillation (arXiv 2510.13998) as the recovery
   stage; targets WikiText-2 PPL ratio 1.05–1.10× on Phi-2 / TinyLlama / Pythia.
2. **Tooling** — universal `TernaryConfig`-style API, FQN-regex auto-discovery,
   sequential layer offload, sharded export. Polish the existing pipeline to
   library-grade.
3. **Runtime** — hybrid CPU inference: T-MAC (or llama.cpp TQ2_0 if its bug is
   fixed) for the ternary base × per-row α, `scipy.sparse` for the 1% FP16
   outliers, plain PyTorch for LoRA. Compose at the application layer.

**Hard constraints:** Colab T4 (15.6 GB VRAM, 12.7 GB RAM), no paid cloud, no
local GPU. Solo researcher. ~8-week budget.

---

## 1. Project state as of 2026-05-07

### 1.1 What works (do not break)

| Component | File | Status |
|---|---|---|
| Symmetric ternary quantizer with 256-candidate threshold search | `pt_bitnet/src/pt_bitnet/quantize.py` | Stable |
| 1% outlier retention (SpQR-style, kept in FP16) | same | Stable |
| 30-step Hessian compensation on `lm_head` | same | Stable, marginal gain (2%) |
| LoRA rank-64 KD from FP16 teacher (attention + FFN) | `pt_bitnet/src/pt_bitnet/lora.py` | Stable |
| INT2 packed export (4 weights/byte) with row α + outlier sidecar | `pt_bitnet/src/pt_bitnet/int2_packing.py`, `pt_bitnet/src/pt_bitnet/export.py` | Stable, 0 errors / 629M weights verified |
| `TernaryInferenceLinear` (PyTorch unpack-and-matmul forward) | `pt_bitnet/src/pt_bitnet/ternary_linear.py` | Stable for correctness, no speedup |
| Sharded safetensors save (800 MB chunks) | `shared/src/shared/checkpoint.py` | Stable |
| Checkpoint/resume across pipeline stages | `shared/src/shared/checkpoint.py`, `chat/src/chat/model_loader.py` | Stable |
| `tchat` interactive CLI with auto-compression | `chat/src/chat/cli.py` | Stable |
| WikiText-2 PPL eval at `max_len=128` | `scripts/colab_test.py`, `eval/src/eval/benchmarks.py` | Stable |

### 1.2 What was removed and must NOT be re-added

| Technique | Reason | Reference |
|---|---|---|
| Tequila / UltraQuantV3 | Re-decomposes ternary on PT-BitNet's denormalized weights → destroys sparsity (PPL 20.7) | `results/history.md` Exp 1 |
| ParetoQ / ZeroQAT | `w_bits=1` uses `sign()` (binary), `w_bits=0` uses {-0.667α, 0, +0.667α} — neither is true ternary | `results/history.md` Exp 4 |
| ITF asymmetric + outliers | Outlier zeroing destabilizes the closed-form denominator → 96/96 layer fallback | `results/history.md` Exp 3 |
| SSR column reordering | Inverse permutation bug, garbled output | history Failed Techniques |
| AGA activation-aware grid | Code fixed but unused; does not improve over symmetric | history Failed Techniques |
| OBC row compensation | Regressed PPL 1.239→1.281× on small models (over-compensation) | history Exp 10 |
| Re-quantization after LoRA merge | Breaks the W_ternary + LoRA complementary relationship → PPL 17.86 | history Exp 8 |
| Per-element Lambada | 2.5 GB just for Lambada params on 7B | history Failed Techniques |

### 1.3 What's deleted from the repo as part of this pivot

- `paretoq/` (entire workspace member) — removed technique, no longer needed even as artifact
- `tequila/` (entire workspace member) — same
- `run_pipeline.py` — broken, references removed stages
- `notebooks/colab_demo.ipynb` — describes the legacy 4-stage pipeline
- `notebooks/ablation_study.ipynb` — tied to deleted `colab_ablate.py` flow
- `scripts/colab_ablate.py`, `scripts/colab_final.py` — superseded by `colab_test.py` and `colab_export_test.py`
- `eval/src/eval/export_bitnet.py` — superseded by `pt_bitnet/export.py` + future llama.cpp path
- `exported_phi2.zip` — 3.9 GB artifact, never should have been committed
- `tests/test_paretoq.py`, `tests/test_tequila.py` — tests for removed code

After cleanup the workspace members in `pyproject.toml` are: `pt_bitnet`,
`shared`, `chat`, `eval`.

---

## 2. Architecture: Hybrid Runtime

This is the key novelty. Forward pass for each compressed linear layer is a
sum of three components on three substrates:

```
y = TernaryKernel(x, W_int2, alpha_per_row)   # ~99% of FLOPs, fast CPU kernel
  + SparseMatMul(x, W_outlier_csr_fp16)        # ~1% sparsity, FP16, cheap
  + (alpha_lora / rank) * (B @ (A @ x))        # LoRA correction, rank ≪ dim
```

Precedents for this composition:
- **SpQR** ([Dettmers 2023, arXiv 2306.03078](https://arxiv.org/abs/2306.03078)):
  sparse FP16 outliers + dense low-bit kernel, dispatched separately.
- **QLoRA** ([Dettmers 2023, arXiv 2305.14314](https://arxiv.org/abs/2305.14314)):
  NF4 base + LoRA adapter applied at runtime.

TernaryBoost's runtime is the **union of those two patterns**, with a ternary
kernel in place of NF4. Not previously published as a single deployable
artifact.

### 2.1 Substrate options for the ternary kernel

Three paths, in order of decreasing risk:

**Path A — llama.cpp TQ2_0 + LoRA + outliers in PyTorch** (preferred)
- Convert: `python convert_hf_to_gguf.py --outtype tq2_0 ./quantized_model`
  - Phi-2 (hidden=2560, FFN=10240) qualifies (multiples of 256)
  - TinyLlama-1.1B (hidden=2048) qualifies
  - Pythia-410M (hidden=1024) and Pythia-1.4B (hidden=2048) qualify
  - Qwen2.5-1.5B (hidden=1536) does **not** qualify — skip in scaling curve
- Load via `llama-cpp-python` for the ternary base
- Apply outliers + LoRA in PyTorch on top
- **Risk:** [llama.cpp issue #15193](https://github.com/ggml-org/llama.cpp/issues/15193)
  reports TQ1_0/TQ2_0 producing garbled output on non-BitNet-architecture
  models, closed without fix. **First action of week 5 is to verify this
  status in mai/2026** with a 1-day spike on Phi-2. If still broken, fall to
  Path B.

**Path B — T-MAC W2A16 GPTQ-format**
- Re-encode ternary {-α, 0, +α} as W2 with codebook {0, 1, 2} (index 3 unused)
- Per-row α expressed as group_size = in_features GPTQ scales (one group per row)
- Build T-MAC via TVM, call generated `.so` from Python via `ctypes`/`cffi`
- T-MAC explicitly supports W2A16 from BitDistiller / EfficientQAT, so the
  codebook is accepted; per-row scales need verification
- Outliers + LoRA in PyTorch on top, same as Path A

**Path C — custom AVX2 kernel via cffi**
- ~300-500 lines of C, reference: `bitnet.cpp/preset_kernels/.../bitnet-lut-kernels-AVX2.h`
- Bates exact format: per-row α, outliers preserved as separate sparse, LoRA
  preserved as separate FP16
- Most control, no external bug exposure
- Effort: 1-2 weeks tightly scoped
- Use only if A and B both fail or both reformat the data lossily

### 2.2 Forward-pass code sketch

```python
# pt_bitnet/src/pt_bitnet/hybrid_runtime.py  (NEW FILE in week 5)

import torch
import torch.nn as nn
import scipy.sparse as sp
from llama_cpp import Llama  # for Path A

class HybridTernaryLinear(nn.Module):
    """Hybrid runtime: ternary kernel + sparse outliers + LoRA, all at app level."""

    def __init__(self, ternary_kernel, outliers_csr, lora_A, lora_B, lora_scaling, bias=None):
        super().__init__()
        self.ternary_kernel = ternary_kernel        # callable: x -> y
        # outliers_csr: scipy.sparse.csr_matrix [out_f, in_f], dtype=fp16, ~1% nnz
        self.register_buffer("outliers_dense", torch.from_numpy(outliers_csr.toarray()).half())
        self.lora_A = nn.Parameter(lora_A, requires_grad=False)
        self.lora_B = nn.Parameter(lora_B, requires_grad=False)
        self.lora_scaling = lora_scaling
        self.bias = bias

    def forward(self, x):
        y = self.ternary_kernel(x)                       # bulk
        y = y + x @ self.outliers_dense.T                # sparse correction
        y = y + (x @ self.lora_A.T) @ self.lora_B.T * self.lora_scaling  # LoRA
        if self.bias is not None:
            y = y + self.bias
        return y
```

For Path A specifically, `ternary_kernel` is a thin wrapper that calls
`llama-cpp-python`'s per-layer matmul or, if that's not exposed, runs the
whole TQ2_0 model and we apply outliers+LoRA at a higher granularity (whole
hidden state correction). For Path B/C, `ternary_kernel` is a `ctypes`
function pointer.

### 2.3 Honest expected speedup

| Scheme | CPU speedup vs FP16 baseline (llama.cpp Q8_0) |
|---|---|
| Pure T-MAC W1.58A8 (BitNet-from-scratch) | 4-5× |
| **Hybrid (ours)**: T-MAC base + sparse outliers + LoRA | **2.5-3.5×** |
| llama.cpp TQ2_0 alone (no outliers, no LoRA) | 3-4× |
| Status quo (PyTorch FP16) | 1× |

The 1-1.5× gap vs pure BitNet kernel is the cost of preserving quality
(outliers, LoRA). That's the trade we're making explicit — and that's the
paper's contribution.

---

## 3. Quality plan: BitNet Distillation integration

### 3.1 What BitDistill is

[Wu et al., "BitNet Distillation," arXiv 2510.13998, Oct 2025](https://arxiv.org/abs/2510.13998).
Three-stage recipe to fine-tune an FP16 model to ternary:

1. **SubLN insertion** — add an RMSNorm before the output projection of MHSA
   and before the output projection of FFN. **Architectural change**, must
   happen before quantization.
2. **Continued pre-training warm-up** — paper uses 10B tokens. We'll scale
   down to 50-100M tokens of C4 (T4-feasible).
3. **MiniLM-style attention distillation** — Q/K/V relation matrices from
   the FP16 teacher distilled into the ternary student, alongside logit KD.
   A single layer of attention distillation is sufficient per the paper.

### 3.2 Files to add / modify

```
pt_bitnet/src/pt_bitnet/
├── subln.py            (NEW)  — insert SubLN before MHSA/FFN output projections
├── lora.py             MODIFY — add MiniLM Q/K/V attention distillation loss
└── quantize.py         (no change)
```

`subln.py` skeleton:

```python
"""SubLN insertion for BitNet-Distillation-compatible ternarization.

Adds an RMSNorm immediately before the output projection of each MHSA and FFN
block. Required architectural prep before BitDistill quality recovery.

Reference: Wu et al., "BitNet Distillation," arXiv 2510.13998, Oct 2025.
"""

import torch.nn as nn
from transformers import PreTrainedModel

class SubLN(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight

def insert_subln(model: PreTrainedModel) -> PreTrainedModel:
    """Wrap MHSA.o_proj and FFN.down_proj/fc2 inputs with SubLN.

    Implementation: walk model.named_modules, identify output-projection
    linears by FQN regex (works across Phi-2 'dense', LLaMA 'o_proj',
    'down_proj', GPT-style 'fc2'), replace each with Sequential(SubLN, original).
    Initialize SubLN.weight to 1 so the model is functionally identical at
    insertion time; ternarization-aware fine-tuning then learns the scale.
    """
    ...
```

`lora.py` addition for MiniLM attention distillation:

```python
def minilm_attention_loss(student_attn, teacher_attn):
    """Q/K/V relation-matrix distillation (MiniLM style)."""
    losses = []
    for s, t in zip(student_attn, teacher_attn):  # only the LAST layer is enough per paper
        # s.q, s.k, s.v shape: [batch, heads, seq, dhead]
        for proj in ('q', 'k', 'v'):
            sR = torch.matmul(getattr(s, proj), getattr(s, proj).transpose(-1, -2))
            tR = torch.matmul(getattr(t, proj), getattr(t, proj).transpose(-1, -2))
            losses.append(F.kl_div(F.log_softmax(sR, -1), F.softmax(tR, -1), reduction='batchmean'))
    return sum(losses) / len(losses)
```

### 3.3 BitDistill stage in the pipeline

Inserts between PT-BitNet quantization and LoRA-KD:

```
FP16 model
  → insert_subln(model)                       # NEW
  → 50M-token C4 warm-up (1-2 hours T4)        # NEW
  → PT-BitNet ternarize (existing)
  → LoRA wrap (existing)
  → KD training with logit + MiniLM losses    # MODIFY existing loop
  → INT2 export (existing)
```

### 3.4 Cheaper wins to test before BitDistill (week 2)

These are 1-line code changes that may stack additively. Run on Phi-2 +
WikiText-2 baseline before committing 2 weeks to BitDistill.

1. **Random Hadamard rotation** before threshold search. 1 line in
   `quantize.py`: `W = W @ H` where `H = scipy.linalg.hadamard(in_f) /
   sqrt(in_f)`. Fold inverse into the next layer's input. Sanity-check first
   on a single layer; if PPL regresses, drop rotations entirely (do not pursue
   SpinQuant — see deferred list).

2. **OneBit SVID rank-1 scales**. Replace per-row scalar `α_i` with rank-1
   outer product `α_row[i] · β_col[j]`. Edit
   [`pt_bitnet/src/pt_bitnet/quantize.py:_symmetric_ternary`] (or wherever
   the scale is applied) to compute `β_col` via SVD of |W|. Storage: one
   extra FP16 vector of length `in_features` per layer. **Critical: do NOT
   adopt OneBit's sign matrix** — `sign()` destroys the zero state and
   conflicts with ternary {-1, 0, +1}.

3. **ApiQ-style LoRA initialization**. Replace the current
   `torch.randn * 0.02` init in [`lora.py:LoRALinear.__init__`] with
   activation-error-minimizing init computed layer-wise from 32 calibration
   sequences. Reference: Liao et al., EMNLP 2024, arXiv 2402.05147.

### 3.5 Deferred quality techniques (do not pursue)

- **PTQTP (arXiv 2509.16989)** — uses 2 trit-planes, effective 3.16 bits, not
  1.58. Reframes the project. Cite as related work; do not adopt unless we
  decide to relax the bit budget.
- **SpinQuant / QuaRot learned rotations** — untested at 1.58 bits, may
  regress like OBC did.
- **OmniQuant LWC/LET, BiLLM, PB-LLM, SliM-LLM, FBI-LLM** — incompatible with
  symmetric ternary {-α, 0, +α} or with our outlier policy.
- **QuIP# E8 lattice codebook** — breaks INT2 packed export, not a fit.
- **EfficientQAT, LoftQ, QA-LoRA** — designed for 4-bit, no validated 1.58
  results.

---

## 4. Tooling / UX plan

Goal: make the pipeline feel like a library (`from ternary_boost import
TernaryConfig`), not a script. Borrow patterns from `torchao` and
`bitsandbytes`.

### 4.1 New API surface (week 6)

```python
# Target user-facing API:
from transformers import AutoModelForCausalLM
from ternary_boost import TernaryConfig

config = TernaryConfig(
    outlier_fraction=0.01,
    lora_rank=64,
    lora_steps=1000,
    distill_method="bitdistill",   # or "logit_kd_only"
    calibration_dataset="wikitext-2",
)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    quantization_config=config,
)
# Usage from here is normal HF generate()
```

### 4.2 UX patterns to adopt

| Pattern | Source of inspiration | Effort | File |
|---|---|---|---|
| FQN-regex auto target-module discovery | `torchao` `FqnToConfig` | 2h | `pt_bitnet/quantize.py:_find_quantizable_linears` (rewrite) |
| `TernaryConfig` dataclass + `from_pretrained` integration | `transformers/quantization/torchao.md` | 4h | new `pt_bitnet/integration.py` |
| Sequential layer offload during calibration | `GPTQModel` | 6h | refactor `pt_bitnet/quantize.py:apply_pt_bitnet` |
| Imatrix-style importance file (reusable across runs) | `llama.cpp` | 4h | new `shared/imatrix.py` |
| HF Hub auto-upload + GGUF export hook | `optimum-quanto` | 4h | extend `pt_bitnet/export.py` |

### 4.3 Pip-installable package

End-state: `pip install ternary-boost` should work. Currently we have a uv
workspace with 4 sub-packages. Consolidate to one `pt_bitnet` (rename
package to `ternary_boost`) for distribution; keep `chat`/`eval` as optional
extras.

---

## 5. Sequenced execution plan (8 weeks)

Each week is gated by a concrete deliverable. **If the gate doesn't fire, do
not advance to the next week** — debug or pivot.

### Week 1 — Baselines (non-negotiable)

**Goal:** Anchor the 1.216× number against the field.

Deliverables:
- [ ] Run **GPTQModel 2-bit** on Phi-2, WikiText-2 PPL @ max_len=128, same seed
- [ ] Run **HQQ 1-bit (binary, group-size 64)** on Phi-2, same protocol
- [ ] Run **AutoRound int2 (with `--enable_alg_ext`)** on Phi-2
- [ ] Clone and run **PT²-LLM** ([repo](https://github.com/XIANGLONGYAN/PT2-LLM))
  on Phi-2 — this is the direct ternary competitor
- [ ] Clone and run **PTQTP** if their code is released by now
- [ ] Add results to a new file `results/baselines.md` with one row per method:
  PPL, size, time-to-quantize, T4-VRAM-peak

**Gate:** Table populated. Decision criteria:
- If TernaryBoost is within 5% PPL of PT²-LLM → continue, story is "we match
  PT²-LLM with simpler method + LoRA recovery + universal HF support".
- If TernaryBoost is >10% worse than PT²-LLM → adopt PT²-LLM's asymmetric
  ITF (without outliers) as Stage 1 in our pipeline (already implemented in
  `quantize.py` via `asymmetric=True, outlier_fraction=0`).
- If PT²-LLM crashes on Phi-2 → that's a finding, document it in `baselines.md`.

### Week 2 — Cheap quality wins

**Goal:** Fastest possible PPL improvement, low-risk changes.

Deliverables:
- [ ] Implement OneBit SVID rank-1 scales (Section 3.4 #2). 1 file, ~30 LOC.
- [ ] Implement ApiQ-style LoRA init (Section 3.4 #3). Modify `lora.py`.
- [ ] Optional: Random Hadamard rotation sanity check (Section 3.4 #1). If
  it regresses on Phi-2, drop rotations entirely.
- [ ] Re-run pipeline, append to `results/baselines.md`

**Gate:** PPL ratio improved at least 0.02× (1.216 → ≤1.196). If not, the
cheap wins didn't compound; move on regardless to BitDistill — they're orthogonal.

### Weeks 3-4 — BitNet Distillation

**Goal:** Hit 1.05-1.10× PPL ratio target.

Deliverables:
- [ ] Implement `pt_bitnet/src/pt_bitnet/subln.py` (Section 3.2)
- [ ] Modify `pt_bitnet/src/pt_bitnet/lora.py` to add MiniLM Q/K/V attention
  distillation loss (Section 3.2)
- [ ] Implement 50M-token C4 warm-up loop (new script: `scripts/warmup_subln.py`)
- [ ] Run end-to-end: Phi-2 + SubLN + warm-up + PT-BitNet + KD with attention
  distill + LoRA wrap
- [ ] Update `results/phi2_ternary.md` with the new number

**Gate:** WikiText-2 PPL ratio ≤ 1.15× on Phi-2. If we land between 1.10 and
1.15, accept and move on. If we land worse than 1.20× (regression), stop and
debug — likely a SubLN insertion bug or warm-up data domain mismatch.

### Week 5 — Hybrid runtime spike (Path A first)

**Goal:** Validate that llama.cpp TQ2_0 + LoRA + outliers actually works on
Phi-2 in mai/2026.

Deliverables:
- [ ] **Day 1-2:** Convert ternarized Phi-2 to GGUF TQ2_0 via patched
  `convert_hf_to_gguf.py`. Patch needed because the converter only emits TQ2_0
  for models registered as ternary architecture; Phi-2 isn't. Patch: add
  `--force-ternary` flag that overrides the dispatch.
- [ ] **Day 2-3:** Load via `llama-cpp-python`, generate 100 tokens, check
  output is coherent (not gibberish per [issue #15193](https://github.com/ggml-org/llama.cpp/issues/15193)).
  If gibberish: switch to Path B (T-MAC) immediately, do not debug llama.cpp.
- [ ] **Day 3-5:** Implement `HybridTernaryLinear` (Section 2.2) and wire
  outliers + LoRA on top of llama.cpp generation.
- [ ] Benchmark CPU tokens/sec vs llama.cpp Q8_0 baseline on the same
  machine. Report honestly.

**Gate:** Coherent generation + measurable speedup ≥ 2×. If not, Path B (T-MAC).

### Week 6 — Tooling polish + scale curve

**Goal:** Workshop-paper-grade reproducibility.

Deliverables:
- [ ] Implement `TernaryConfig` API (Section 4.1)
- [ ] FQN-regex auto-discovery (replace per-arch hand-coding)
- [ ] Run pipeline on Pythia-410M, Pythia-1.4B, TinyLlama-1.1B, Qwen2.5-0.5B,
  Qwen2.5-1.5B (skip Qwen2.5-1.5B from runtime story — hidden=1536 doesn't
  fit TQ2_0 256-multiple constraint)
- [ ] Append to `results/scaling_curve.md`: model size vs PPL ratio vs runtime
  speedup

**Gate:** 5+ model sizes evaluated. Curve is monotonic-ish (degradation
shrinks with size, or doesn't — either is publishable as a finding).

### Week 7 — Paper draft

**Goal:** ES-FoMo IV submission-ready.

Deliverables:
- [ ] 4-page paper, ICML template, double-blind
- [ ] Sections: Intro, Related Work (PT²-LLM, PTQTP, BitNet b1.58, BitDistill,
  T-MAC, SpQR, QLoRA), Method (Hybrid Runtime + BitDistill integration),
  Experiments (baselines table + scaling curve + speedup), Limitations
  (bit-budget caveats, multiple-of-256 constraint, T4-only validation), Conclusion
- [ ] Anonymized GitHub repo with reproducibility script

**Gate:** ES-FoMo IV CFP opened; deadline confirmed (expected late May / early
June 2026). Submit.

### Week 8 — Buffer / rebuttal prep / camera-ready

If accepted: camera-ready improvements, additional baselines if reviewers ask.
If rejected: pivot to ENLSP-VI @ NeurIPS 2026 (CFP expected July, deadline
August/September) or LXAI @ NeurIPS 2026.

---

## 6. Decision gates and pivots

| Trigger | Action |
|---|---|
| Week 1 baselines show TernaryBoost competitive with PT²-LLM | Continue as planned |
| Week 1 baselines show >10% gap to PT²-LLM | Switch Stage 1 to PT²-LLM's asymmetric ITF (`asymmetric=True, outlier_fraction=0`) |
| Week 1 baselines show INT4 GPTQ strictly dominates ternary on Phi-2 size+PPL | Pivot framing: ternary as **inference-speed** play (Path A speedup), not size play |
| Weeks 3-4: BitDistill regresses (PPL > 1.20×) | Roll back to logit-only KD; accept 1.18-1.22× and lean harder on tooling/runtime story |
| Week 5: llama.cpp TQ2_0 produces gibberish (issue #15193 unfixed) | Switch to Path B (T-MAC W2A16). Add 1 week to schedule. |
| Week 5: T-MAC also fails (Path B) | Path C: write custom AVX2 kernel. 2 weeks. Push paper to ENLSP-VI deadline. |
| Anytime: PTQTP releases polished pip package | Re-evaluate competitive positioning. The window may have closed. Pivot to "complementary tool: PTQTP base + LoRA + hybrid runtime" rather than competitor. |

---

## 7. Reproducibility and benchmarking discipline

### 7.1 Standard evaluation protocol

Every variant must report on the same protocol or it is not a comparison:

- **Calibration:** 128 sequences × 2048 tokens from C4-en, fixed seed=42
  (GPTQ/AutoRound convention)
- **PPL:** WikiText-2 test set, max_len=128 stride, 500 lines (Phi-2
  convention from `colab_test.py`). Also report C4-validation PPL for
  cross-check.
- **Downstream:** MMLU 5-shot, HellaSwag 10-shot, ARC-Easy/Challenge 25-shot,
  Winogrande zero-shot. Tools: `lm-evaluation-harness` v0.4+.
- **Hardware:** Colab T4 only. Document CPU type if reporting runtime speedup.
- **Metrics reported:** mean ± std over 3 seeds for stochastic stages
  (LoRA training); single run for deterministic stages.

### 7.2 Data hygiene

- Do not commit model weights, datasets, or `.zip` artifacts to git
- Add `cache/`, `output/`, `*.zip`, `*.safetensors`, `.venv/` to `.gitignore`
  (verify before week 1)
- Calibration data lives in `shared/data.py`; do not duplicate

### 7.3 Test discipline

After deletions in Section 1.3, the test suite is:
- `tests/test_pt_bitnet.py` — quantization correctness
- `tests/test_int2_packing.py` — packing roundtrip
- `tests/test_export.py` — export integrity
- `tests/test_ternary_linear.py` — forward equivalence

All four should pass after refactoring (Phase 7). New code added in weeks 3-5
must be accompanied by tests.

---

## 8. Reference appendix

### 8.1 Repos to clone for week 1

```bash
git clone https://github.com/XIANGLONGYAN/PT2-LLM            # PT²-LLM
git clone https://github.com/ModelCloud/GPTQModel            # GPTQModel
git clone https://github.com/mobiusml/hqq                    # HQQ
git clone https://github.com/intel/auto-round                # AutoRound
git clone https://github.com/microsoft/T-MAC                 # T-MAC (week 5)
git clone https://github.com/microsoft/BitNet                # bitnet.cpp (week 5)
git clone https://github.com/ggml-org/llama.cpp              # llama.cpp (week 5)
```

### 8.2 Key papers (in order of relevance to this plan)

1. Wu et al., **BitNet Distillation**, arXiv 2510.13998 (Oct 2025) — core quality recipe
2. Yan et al., **PT²-LLM**, arXiv 2510.03267 (Oct 2025) — direct competitor + ablation baseline
3. Xu et al., **PTQTP**, arXiv 2509.16989 (Sep 2025) — competitor at 3.16-bit effective
4. Ma et al., **BitNet b1.58**, arXiv 2402.17764 (Feb 2024) — foundational
5. Wang et al., **bitnet.cpp**, arXiv 2502.11880 (Feb 2025) — runtime SOTA
6. Wei et al., **T-MAC**, arXiv 2407.00088, EuroSys 2025 — preferred kernel for Path B
7. Dettmers et al., **SpQR**, arXiv 2306.03078 (Jun 2023) — outlier+lowbit precedent
8. Dettmers et al., **QLoRA**, arXiv 2305.14314 (May 2023) — base+LoRA-at-runtime precedent
9. Liao et al., **ApiQ**, arXiv 2402.05147 (Feb 2024) — LoRA init technique
10. Xu et al., **OneBit**, arXiv 2402.11295 (Feb 2024) — SVID rank-1 scales technique

### 8.3 Venue calendar (target 2026)

| Venue | Format | Expected deadline | Notes |
|---|---|---|---|
| **ES-FoMo IV @ ICML 2026** | 4 pages, ICML template, double-blind | Late May / early June 2026 | Primary target. Workshop is Jul 10-11, Seoul. CFP at https://es-fomo.com/call/ |
| ENLSP-VI @ NeurIPS 2026 | 4-8 pages | Aug-Sep 2026 | Backup if ES-FoMo rejects. NeurIPS workshop application deadline 2026-06-06 |
| LXAI @ NeurIPS 2026 | 4-8 pages, OpenReview | Per NeurIPS 2026 calendar | Brazilian/LatinX researchers explicitly welcomed. Lower bar, strong community visibility |
| ACL/EMNLP 2026 Findings (short) | 4 pages | Per ACL/EMNLP rolling deadlines | Stretch target if scaling curve includes 7B+ results (would require collaboration for compute) |

### 8.4 Glossary for new readers

- **Ternary / 1.58-bit:** weights restricted to {-α, 0, +α}; log₂(3) ≈ 1.585 bits per weight
- **PT-BitNet (this repo's term):** synthesis of BitNet b1.58 PTQ + SpQR outliers + GPTQ Hessian compensation. Not a published method on its own. See README.
- **PTQ vs QAT:** PTQ does not retrain weights; QAT does. This project is PTQ-only with a light LoRA-adapter retraining stage.
- **LoRA:** low-rank adapter matrices A [rank, in_f] and B [out_f, rank], typically rank ≪ dim
- **SubLN:** an extra RMSNorm inserted before the output projection of MHSA/FFN; required by BitDistill
- **TQ2_0:** llama.cpp's 2.06-bpw ternary format. Distinct from IQ2_XXS (non-uniform codebook, not ternary).
- **T-MAC:** Microsoft's lookup-table-based mpGEMM kernel for ≤4-bit weights. Accepts W2A16 GPTQ format among others.

---

## 9. Glossary of "do nots" (consolidated)

To prevent rediscovery of dead ends:

1. Do not re-add Tequila — see Section 1.2
2. Do not re-add ParetoQ/QAT — see Section 1.2
3. Do not enable ITF (`asymmetric=True`) **with** outliers (`outlier_fraction>0`)
4. Do not enable OBC on models ≤7B — regression confirmed
5. Do not merge LoRA into ternary base (`merge_and_requantize`) — destroys correction
6. Do not use distill_weight ≥ 0.5 in KD — pulls model away from correct predictions
7. Do not use KD temperature ≥ 3.0 — near-uniform teacher distribution
8. Do not adopt OneBit's sign matrix — incompatible with zero state. Adopt only the SVID scale decomposition.
9. Do not use IQ1_S/IQ1_M/IQ2_XXS in llama.cpp — non-uniform codebooks, not ternary
10. Do not commit model weights, datasets, or zip artifacts to git
