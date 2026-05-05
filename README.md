# TernaryBoost

**A Hybrid Pipeline for Extreme LLM Compression via Ternary Quantization**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-51%2F51%20passed-brightgreen)](tests/)

---

## Abstract

TernaryBoost compresses any HuggingFace causal LM to 1.58-bit ternary representation via **PT-BitNet**: symmetric ternary quantization with 1% outlier retention (SpQR-style) and Hessian compensation on lm_head (GPTQ-style). The pipeline auto-detects GPU/CPU, checkpoints incrementally, and produces a standard HuggingFace-compatible checkpoint. An interactive chat CLI (`tchat`) provides immediate access to compressed models.

---

## Architecture

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│  FP16 Model  │───▶│ PT-BitNet       │───▶│ Save          │
│  (HF Hub)    │    │ Symmetric Tern. │    │ Sharded       │
└──────────────┘    │ + 1% Outliers   │    │ Safetensors   │
                    │ + Hessian Comp. │    └──────┬───────┘
                    └─────────────────┘           ▼
                                            ┌──────────────┐
                                            │ tchat CLI     │
                                            │ Interactive   │
                                            │ Chat Interface│
                                            └──────────────┘
```

## Components

### Stage 1 — PT-BitNet (Post-Training Ternary Quantization)

Synthesizes post-training quantization techniques from the BitNet b1.58 framework (Ma et al., 2024) and PT²-LLM (Yan et al., ICLR 2026). Three sub-steps execute sequentially:

1. **Symmetric Ternary Quantization** — weights compressed to {-α, 0, +α} per row via threshold search over 256 candidates. The asymmetric ITF mode (PT²-LLM Eq. 9-10) is available but disabled by default — it destabilizes when combined with outlier retention.
2. **Outlier Retention** (SpQR-style) — top 1% weights by z-score kept in FP16, preventing large weights from distorting the ternary grid.
3. **Hessian Compensation** (GPTQ-style) — lm_head fine-tuned on calibration texts to absorb quantization error. Memory-efficient: forward hook captures last hidden state, only lm_head receives gradients. 50 steps GPU (~3 min), 10 steps CPU (~14 min).

Targets all linear projections in attention and MLP while preserving `lm_head` and `embed_tokens`.

> "PT-BitNet" is not a published method. It synthesizes BitNet PTQ with PT²-LLM, SpQR, and GPTQ.

### Tequila (Research Artifact — Removed from Pipeline)

Tequila (Huang et al., 2025) addresses the _deadzone trapping_ problem in quantization-aware training. It was removed from the pipeline for a technical incompatibility:

**Why removed:** PT-BitNet normalizes weights, quantizes, then **denormalizes** (w*std + mean). The final weights have non-zero row means. Tequila's UltraQuantV3 recomputes the ternary decomposition from scratch using its own threshold (mean(|w|)/2). When row means are large relative to the ternary scale, the "zero" weights in PT-BitNet's pattern get classified as active by UltraQuantV3, destroying the carefully optimized sparsity structure. Empirically, this caused perplexity to jump from 3.45 (PT-BitNet alone) to 20.7 (PT-BitNet + Tequila).

The Tequila code (`tequila/`) is retained as a research artifact for future quantization-aware training work. The `UltraQuantLinear`, `Lambada` optimizer, and baking infrastructure are functional but not used in the default pipeline.

| Speed Mode | Prompt | Meaning |
|-----------|--------|---------|
| `[baked]` | blue | Standard `nn.Linear` with ternary weights (fast inference) |
| `[FP16]` | red | Standard `nn.Linear` — no ternary optimization active |

### On ParetoQ / ZeroQAT (Removed from Pipeline)

ParetoQ (Zhang et al., NeurIPS 2025) and ZeroQAT were originally included as a QAT stage. They were removed for two reasons:

1. **No native ternary mode:** `QuantizeLinear` with `w_bits=1` uses `sign()` (binary {-1, +1}, destroys sparsity). With `w_bits=0`, StretchedElasticQuant scales to {-0.667a, 0, +0.667a} (incompatible with PT-BitNet's {-a, 0, +a}).
2. **Empirical degradation:** On Phi-2, QAT increased loss from 5.86 to 6.69.

The Tequila stage provides equivalent ternary-aware optimization with proper deadzone handling. The ParetoQ code (`paretoq/`) is retained as a research artifact (LSQ quantization, MeZO-style ZO optimizer, SPSA estimator).

### Evaluation and Export

**Benchmarks** — integration with `lm-evaluation-harness` (MMLU, HellaSwag, ARC). Fallback perplexity on WikiText-2.

**Speedup** — baked ternary models run at standard FP16 speed in PyTorch. The 3-6x speedup claimed in BitNet papers requires a native SIMD kernel (microsoft/BitNet, GGUF I2_S format) that replaces float multiplications with integer additions on packed ternary weights. Integration path documented; Python bindings tracked in TODO.md.

---

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.10 | 3.12+ |
| **GPU VRAM** (pipeline) | 8 GB (2B model) | 24 GB (7B model) |
| **RAM** (CPU pipeline) | 8 GB (2B model) | 32 GB (7B model) |
| **RAM** (CPU inference) | 4 GB (2B model) | 16 GB (7B model) |
| **Disk** | 20 GB | 50 GB |
| **OS** | Linux | Linux (Ubuntu 22.04+) |

CPU-only fully supported. GPU optional but recommended for faster compression.

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/adriel007/ternary-boost.git
cd ternary-boost
uv sync
```

## Usage

```bash
# Interactive chat (auto-compresses on first run)
tchat --model phi-2 --device cpu

# Full pipeline programmatically
python run_pipeline.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --output ./output \
  --eval-tasks mmlu,hellaswag,arc_easy,arc_challenge
```

### Chat Commands

| Command | Action |
|---------|--------|
| `/help` | Show command reference |
| `/clear` | Reset conversation history |
| `/system <text>` | Set/view system prompt |
| `/save [path]` | Export conversation to JSON |
| `/load <path>` | Import conversation from JSON |
| `/config` | Display current settings |
| `/model [name]` | List or switch models |
| `/models` | List registered models |
| `/add-model` | Register any HuggingFace model |
| `/thinking` | Toggle chain-of-thought mode |
| `/stats` | Show generation statistics |
| `/cache` | Show cached compressed models |
| `/quit`, `/exit` | Exit |

### Model Registry (all open-access)

| CLI name | Model | Size |
|----------|-------|------|
| `mistral-7b` | Mistral 7B Instruct v0.3 | 7B |
| `olmo-7b` | OLMo 7B Instruct | 7B |
| `falcon-7b` | Falcon 7B Instruct | 7B |
| `qwen2.5-7b` | Qwen2.5 7B Instruct | 7B |
| `phi-3-small` | Phi-3 Small 8K | 7B |
| `phi-3-medium` | Phi-3 Medium 4K | 14B |
| `phi-2` | Phi-2 | 2.7B |

## Results

### Phi-2 (2.7B) — Colab T4 (15.6 GB VRAM)

| Metric | Baseline FP16 | Quantized (ternary) | Ratio |
|--------|--------------|---------------------|-------|
| **Perplexity** (diverse texts) | 2.61 | 3.45 | 1.32x |
| **Generation quality** (avg 0-100) | 94 | 94 | 1.00x |
| **Repetition ratio** | 0.00 | 0.00 | — |
| **Inference speed** | — | 20.5 tok/s | — |
| **Pipeline time** | — | 10.1 min | — |
| **Disk** | 5.6 GB | 5.6 GB | — |

**Verdict: GOOD** — minor perplexity loss (32%), identical generation quality, no degeneration. Post-training ternary on small models (2.7B) is inherently harder due to less parameter redundancy. PT²-LLM reports 1.15-1.25x PPL ratios on 7-70B models.

Full results and sample outputs: [`results/phi2_ternary.md`](results/phi2_ternary.md)

### 7B models — requires A100/L4 (24+ GB VRAM)

Change the `MODEL` variable in `scripts/colab_test.py`:
```python
MODEL = "mistralai/Mistral-7B-v0.1"
```

The script auto-detects VRAM and warns if the GPU is too small. Pipeline time estimate: ~20-30 min on A100.

### Aspirational (with native kernel)

| Metric | Value |
|--------|-------|
| CPU inference (microsoft/BitNet GGUF I2_S) | 3-6× vs baked |
| Disk (INT2 packed) | ~0.5 GB |
| 7B model pipeline (GPU A100) | ~20 min est. |

> **Status:** Pipeline produces coherent, factually-correct output on Phi-2 (verified 2026-05-05). 7B testing pending hardware availability.

## Repository Structure

```
ternary-boost/
├── pyproject.toml              # Root workspace (uv)
├── run_pipeline.py             # Pipeline orchestrator
├── README.md
├── TODO.md
├── results/
│   └── phi2_ternary.md         # Phi-2 benchmark results
├── notebooks/                  # Colab demo + ablation study
├── scripts/
│   ├── colab_test.py           # Full pipeline test (configurable model)
│   ├── colab_ablate.py         # Ablation: isolates each stage
│   └── colab_final.py          # Minimal pipeline (sym+comp only)
│
├── shared/                     # Checkpoint, logging, data loaders
├── pt_bitnet/                  # PT-BitNet: symmetric ternary + outliers + compensation
├── tequila/                    # Research artifact: deadzone trapping (not in active pipeline)
├── paretoq/                    # Research artifact: LSQ, ZO optimizer (not in active pipeline)
├── eval/                       # Benchmarks + bitnet.cpp export
├── chat/                       # Interactive CLI (tchat)
└── tests/                      # 51 unit tests
```

## Citation

```bibtex
@article{ma2024bitnet,
  title   = {The Era of 1-bit {LLM}s: All Large Language Models are in 1.58 Bits},
  author  = {Ma, Shuming and Wang, Hongyu and Ma, Lingxiao and Wang, Lei and
             Wang, Wenhui and Huang, Shaohan and Dong, Li and Wang, Ruiping and
             Xue, Jilong and Wei, Furu},
  journal = {arXiv preprint arXiv:2402.17764},
  year    = {2024}
}

@article{huang2025tequila,
  title   = {Tequila: Trapping-free Ternary Quantization for Large Language Models},
  author  = {Huang, Hong and Wu, Decheng and Cen, Rui and Yu, Guanghua and
             Li, Zonghang and Liu, Kai and Zhu, Jianchen and Chen, Peng and
             Liu, Xue and Wu, Dapeng},
  journal = {arXiv preprint arXiv:2509.23809},
  year    = {2025}
}

@inproceedings{zhang2025paretoq,
  title     = {{ParetoQ}: Scaling Laws in Extremely Low-bit {LLM} Quantization},
  author    = {Zhang, Beichen and others},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}

@article{malladi2023mezo,
  title   = {Fine-Tuning Language Models with Just Forward Passes},
  author  = {Malladi, Sadhika and Gao, Tianyu and Nichani, Eshaan and
             Damian, Alex and Lee, Jason D. and Chen, Danqi and Arora, Sanjeev},
  journal = {arXiv preprint arXiv:2305.17333},
  year    = {2023}
}

@inproceedings{esser2020lsq,
  title     = {Learned Step Size Quantization},
  author    = {Esser, Steven K. and McKinstry, Jeffrey L. and Bablani, Deepika
               and Appuswamy, Rathinakumar and Modha, Dharmendra S.},
  booktitle = {International Conference on Learning Representations},
  year      = {2020}
}

@article{guo2024zo,
  title   = {Zeroth-Order Fine-Tuning of {LLM}s with Extreme Sparsity},
  author  = {Guo, Mengzhou and others},
  journal = {arXiv preprint arXiv:2406.02913},
  year    = {2024}
}
```

## License

Apache License 2.0. Individual techniques derived from works under their respective licenses: Meta torchao (BSD 3-Clause), Tencent AngelSlim (Apache 2.0), HuggingFace Transformers (Apache 2.0).
