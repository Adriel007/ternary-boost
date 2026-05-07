# TernaryBoost

**Universal Post-Training Ternarization Pipeline for HuggingFace Causal LMs**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

---

## What this is

A reproducible PTQ pipeline that compresses arbitrary HuggingFace causal LMs to
1.58-bit ternary representation and recovers quality via knowledge distillation,
runnable end-to-end on a Colab T4. The pivot toward a **hybrid CPU runtime**
(ternary kernel + sparse outliers + LoRA-at-inference) is in active development —
see [`research/PLAN.md`](research/PLAN.md).

**This is not a new quantization method.** It synthesizes BitNet b1.58 PTQ
(Ma et al., 2024), SpQR outlier retention (Dettmers et al., 2023), and
GPTQ-style Hessian compensation (Frantar et al., 2023), with a LoRA-based
recovery stage adapted from QLoRA. The contribution is the integration:
universal HF-arch coverage, T4-feasible end-to-end, INT2 packed export, and
(in progress) hybrid runtime that composes existing kernels at the
application layer.

---

## Status (2026-05-07)

**Active pipeline:**

```
FP16 model
  → PT-BitNet ternarization (sym + 1% outliers + lm_head Hessian comp)
  → LoRA rank-64 KD from FP16 teacher (attention + FFN, distill_weight=0.1)
  → Sharded INT2 export (packed 2 bits/weight, row-α + outlier sidecars + LoRA)
  → tchat CLI for interactive inference
```

**Best result on Phi-2:**

| Metric | FP16 | TernaryBoost | Degradation |
|---|---|---|---|
| WikiText-2 PPL (max_len=128, 500 lines) | 27.39 | 33.31 | **1.216×** |
| Inference disk (INT2 + LoRA + outliers) | 5.6 GB | ~450 MB | **12.4× smaller** |
| Pipeline time (Colab T4) | — | ~47 min | — |
| INT2 integrity | — | 0 errors / 629M weights | — |

Full history of experiments and failures: [`results/history.md`](results/history.md).

**Not yet:** native CPU speedup. The INT2 export is currently inference-runs
at FP16 speed via PyTorch unpacking. Hybrid runtime is the active work item —
see Phase 5 of `research/PLAN.md`.

---

## What changed (recent)

Recent restructuring to align with the new direction:

- **Removed** the legacy 4-stage pipeline (`run_pipeline.py`, ParetoQ, Tequila
  — all proven incompatible with ternary; see `results/history.md` Exps 1, 4).
- **Added** [`research/PLAN.md`](research/PLAN.md) — the master plan. Any
  contributor (or AI agent) should read this first.
- **Removed** stale notebooks and one-shot scripts (`colab_demo.ipynb`,
  `ablation_study.ipynb`, `colab_ablate.py`, `colab_final.py`).
- **Active scripts:** `scripts/colab_test.py` (full pipeline) and
  `scripts/colab_export_test.py` (export validation).

---

## Active components

### Stage 1 — PT-BitNet (Post-Training Ternary Quantization)

Symmetric ternary quantization {-α, 0, +α} per row via 256-candidate threshold
search; 1% outlier retention in FP16 (SpQR-style); 30-step Hessian
compensation on `lm_head` only.

Implementation: [`pt_bitnet/src/pt_bitnet/quantize.py`](pt_bitnet/src/pt_bitnet/quantize.py).
Targets all linear projections in attention and MLP. Skips `lm_head` and
`embed_tokens`.

> "PT-BitNet" is a label internal to this project. It is not a published
> method. It synthesizes BitNet PTQ + SpQR + partial GPTQ.

### Stage 2 — LoRA Fine-Tuning (Quality Recovery)

Rank-64 LoRA adapters wrap the frozen ternary base, fine-tuned via knowledge
distillation from the FP16 teacher (logit KD with temperature 1.5,
`distill_weight=0.1`, 1000 steps, attention + FFN coverage).

Implementation: [`pt_bitnet/src/pt_bitnet/lora.py`](pt_bitnet/src/pt_bitnet/lora.py).

LoRA weights are **kept separate** from the ternary base (no merging) — this
preserves the INT2 packed structure for export and enables the hybrid runtime
(see PLAN Section 2).

### Stage 3 — INT2 Export

Packed 2-bit ternary weights (4 weights/byte), per-row α as FP16, sparse
outlier indices + values as FP16 sidecar, LoRA adapters as standard
safetensors. Sharded to 800 MB chunks for Colab-safe save.

Implementation: [`pt_bitnet/src/pt_bitnet/int2_packing.py`](pt_bitnet/src/pt_bitnet/int2_packing.py),
[`pt_bitnet/src/pt_bitnet/export.py`](pt_bitnet/src/pt_bitnet/export.py).

### tchat — Interactive CLI

Auto-loads (or auto-compresses) a registered model and provides a chat loop.
Supports thinking-mode toggle, conversation save/load, multi-model registry.

Implementation: [`chat/src/chat/cli.py`](chat/src/chat/cli.py).

---

## Roadmap (high-level)

For the detailed 8-week plan with weekly gates, see
[`research/PLAN.md`](research/PLAN.md).

| Phase | Work | Goal |
|---|---|---|
| Week 1 | Baselines: GPTQModel, HQQ, AutoRound, PT²-LLM, PTQTP | Anchor 1.216× against the field |
| Week 2 | Cheap quality wins: SVID rank-1 scales, ApiQ LoRA init | PPL ratio < 1.20× |
| Weeks 3-4 | BitNet Distillation integration (SubLN + MiniLM attention KD) | PPL ratio ≤ 1.10× |
| Week 5 | Hybrid runtime spike (Path A: llama.cpp TQ2_0 + LoRA + outliers in PyTorch) | Coherent generation + ≥ 2× CPU speedup |
| Week 6 | Tooling polish + scaling curve (Pythia 410M-2.8B, TinyLlama, Qwen2.5) | 5+ models on PPL × size × speed Pareto |
| Week 7 | Paper draft for ES-FoMo IV @ ICML 2026 | 4-page submission |

---

## Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **Python** | 3.10 | 3.12+ |
| **GPU VRAM** (pipeline, ≤3B model) | 8 GB (CPU fallback works) | 16 GB (Colab T4) |
| **RAM** (CPU pipeline) | 8 GB (Phi-2) | 16 GB+ |
| **Disk** | 20 GB | 50 GB |
| **OS** | Linux | Linux (Ubuntu 22.04+) |

CPU-only pipeline supported but slow. A100/L4 not required and not the
target — the project is explicitly T4-only.

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/adriel007/ternary-boost.git
cd ternary-boost
uv sync
```

## Usage

```bash
# Interactive chat (auto-compresses on first run, caches result)
tchat --model phi-2 --device cpu

# Full pipeline programmatically (Phi-2 on Colab T4)
python scripts/colab_test.py
```

### Chat commands

| Command | Action |
|---|---|
| `/help` | Command reference |
| `/clear` | Reset conversation |
| `/system <text>` | Set/view system prompt |
| `/save [path]` | Export conversation JSON |
| `/load <path>` | Import conversation JSON |
| `/config` | Show settings |
| `/model [name]` | List or switch models |
| `/add-model` | Register a HuggingFace model |
| `/thinking` | Toggle chain-of-thought mode |
| `/stats` | Generation statistics |
| `/cache` | Show cached compressed models |
| `/quit`, `/exit` | Exit |

### Registered models

| CLI name | Model | Size |
|---|---|---|
| `phi-2` | Phi-2 | 2.7B |
| `mistral-7b` | Mistral 7B Instruct v0.3 | 7B |
| `olmo-7b` | OLMo 7B Instruct | 7B |
| `falcon-7b` | Falcon 7B Instruct | 7B |
| `qwen2.5-7b` | Qwen2.5 7B Instruct | 7B |
| `phi-3-small` | Phi-3 Small 8K | 7B |
| `phi-3-medium` | Phi-3 Medium 4K | 14B |

7B+ models are registered but **not validated** — the project focuses on
≤3B with the curve from Pythia-410M to Phi-2 + TinyLlama.

---

## Repository layout

```
ternary-boost/
├── pyproject.toml                  uv workspace
├── README.md                       (this file)
├── TODO.md                         current sprint
├── research/
│   ├── PLAN.md                     master plan — read this first
│   └── pipeline_improvements.md    technique-level notes
├── results/
│   ├── phi2_ternary.md             Phi-2 best run details
│   └── history.md                  full experiment log including failures
├── scripts/
│   ├── colab_test.py               full pipeline test
│   ├── colab_export_test.py        export validation
│   └── download_data.py            calibration data
├── shared/                         checkpoint, logging, data loaders
├── pt_bitnet/                      ternarization + LoRA + INT2 export
├── eval/                           benchmarks (lm-eval-harness wrapper)
├── chat/                           tchat CLI
└── tests/                          unit tests
```

Removed in the 2026-05-07 cleanup: `paretoq/`, `tequila/`, `run_pipeline.py`,
`notebooks/`, `scripts/colab_ablate.py`, `scripts/colab_final.py`,
`eval/src/eval/export_bitnet.py`. See `results/history.md` for the technical
reasons.

---

## Citation

```bibtex
@article{ma2024bitnet,
  title   = {The Era of 1-bit {LLM}s: All Large Language Models are in 1.58 Bits},
  author  = {Ma, Shuming and others},
  journal = {arXiv preprint arXiv:2402.17764},
  year    = {2024}
}

@article{wu2025bitdistill,
  title   = {BitNet Distillation},
  author  = {Wu and others},
  journal = {arXiv preprint arXiv:2510.13998},
  year    = {2025}
}

@article{yan2026pt2llm,
  title   = {{PT}$^2$-LLM: Post-Training Ternarization for Large Language Models},
  author  = {Yan, Xianglong and others},
  journal = {arXiv preprint arXiv:2510.03267},
  year    = {2025}
}

@article{xu2025ptqtp,
  title   = {{PTQTP}: Post-Training Quantization to Trit-Planes for Large Language Models},
  author  = {Xu and others},
  journal = {arXiv preprint arXiv:2509.16989},
  year    = {2025}
}

@article{dettmers2023spqr,
  title   = {{SpQR}: A Sparse-Quantized Representation for Near-Lossless {LLM} Weight Compression},
  author  = {Dettmers, Tim and others},
  journal = {arXiv preprint arXiv:2306.03078},
  year    = {2023}
}

@article{dettmers2023qlora,
  title   = {{QLoRA}: Efficient Finetuning of Quantized {LLM}s},
  author  = {Dettmers, Tim and others},
  journal = {arXiv preprint arXiv:2305.14314},
  year    = {2023}
}
```

## License

Apache License 2.0.
