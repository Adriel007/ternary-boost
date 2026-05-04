# TernaryBoost

**A Hybrid Pipeline for Extreme LLM Compression via Ternary Quantization**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-51%2F51%20passed-brightgreen)](tests/)

---

## Abstract

TernaryBoost integrates **PT-BitNet** (post-training ternary quantization with outlier retention and Hessian compensation) and **Tequila** (deadzone trapping recovery) into a two-stage pipeline that compresses any HuggingFace causal LM to 1.58-bit ternary representation. The pipeline auto-detects GPU/CPU, checkpoints incrementally, and bakes optimized weights for standard-speed inference. An interactive chat CLI (`tchat`) provides immediate access to compressed models.

---

## Architecture

```
┌──────────────┐    ┌─────────────────┐    ┌─────────────┐    ┌──────────────┐
│  FP16 Model  │───▶│ Stage 1          │───▶│ Stage 2      │───▶│ Bake          │
│  (HF Hub)    │    │ PT-BitNet        │    │ Tequila      │    │ UltraQuant →  │
└──────────────┘    │ Distribution     │    │ UltraQuant   │    │ nn.Linear     │
                    │ Transform +      │    │ Deadzone     │    └──────┬───────┘
                    │ Block-wise Opt   │    │ Recovery     │           │
                    │ + Outlier Ret.   │    └─────────────┘           ▼
                    │ + Hessian Comp.  │                        ┌──────────────┐
                    └─────────────────┘                        │ tchat CLI     │
                                                               │ Interactive   │
                                                               │ Chat Interface│
                                                               └──────────────┘
```

## Components

### Stage 1 — PT-BitNet (Post-Training Ternary Quantization)

Synthesizes post-training quantization techniques from the BitNet b1.58 framework (Ma et al., 2024). Three sub-steps execute sequentially:

1. **Vectorized Ternary Optimization** — batch threshold search over 256 candidate deltas evaluated in parallel across all output channels, minimizing reconstruction error. Rows with above-median error are refined via per-row binary search.

2. **Outlier Retention** (SpQR-style, Dettmers et al., 2023) — keeps the top `outlier_fraction` (default 1%) weights per row in FP16, quantizing only the remaining 99%. Outliers selected by z-score. Controlled via `PTBitNetConfig.outlier_fraction`.

3. **Hessian Compensation** (GPTQ-style, Frantar et al., 2023) — after quantization, the `lm_head` is fine-tuned to absorb residual error. The transformer body runs under `torch.no_grad()` (zero activation memory), and only the final projection receives gradients. A forward hook captures the last hidden state, detaches it, and runs lm_head with grad enabled — limiting the computation graph to ~100M params. Controlled via `PTBitNetConfig.compensation_steps` (default 50).

Targets all linear projections in attention and MLP modules while preserving `lm_head` and `embed_tokens`. GPU (T4): ~4 min for 2.7B. CPU: ~20 min.

> "PT-BitNet" is not a published method name. It synthesizes BitNet PTQ with quality improvements from SpQR and GPTQ.

### Stage 2 — Tequila (Deadzone Trapping Recovery)

Tequila (Huang et al., 2025) addresses the _deadzone trapping_ problem: weights near the ternary decision boundary receive noisy, uninformative gradients preventing escape from the zero region.

The solution splits each weight matrix into two components:

$$\text{output} = \text{linear}(x, A) + \text{linear}(\mathbf{1}, B \odot \Lambda)$$

where $A_{ij} \in \{-\alpha_i, 0, +\alpha_i\}$ (ternary), $B_{ij}$ stores deadzone residuals, and $\Lambda$ is a learnable per-channel _Lambada_ parameter.

Each `UltraQuantLinear` layer holds its own per-layer AdamW optimizer, matching the original AngelSlim implementation. During training, `update_lambada()` is called within `forward()` and performs `zero_grad -> backward -> step` — a self-contained optimization cycle.

**Lambada Granularity:**

| Mode | Shape | RAM (96 layers) | Quality | Use Case |
|------|-------|-----------------|---------|----------|
| `per_channel` (default) | `[out_f, 1]` | ~6 MB | ~80-90% of per-element | CPU, ≤8 GB RAM |
| `per_element` | `[out_f, in_f]` | ~2.5 GB | Full (original paper) | GPU, ≥16 GB RAM |

**Training vs. Inference — Baking:**

The `UltraQuantLinear` forward computes two linear ops per token (dual matmul) — essential for training but 2x slower for inference. After Tequila converges, the pipeline **bakes** the effective weight:

$$\text{effective\_weight} = A + B \odot \Lambda$$

Each `UltraQuantLinear` is replaced with standard `nn.Linear` using this pre-computed weight. Inference uses a single matmul — same speed as FP16, with all quality benefits preserved.

**Speed Mode Indicators:**

| Prompt | Meaning |
|--------|---------|
| `[1.58b]` green | `UltraQuantLinear` active (training mode) |
| `[1.58b-baked]` blue | Baked `nn.Linear` with ternary weights (fast inference) |
| `[FP16]` red | Standard `nn.Linear` — no ternary optimization active |

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
| **GPU VRAM** (pipeline) | 8 GB (2B model) | 16 GB (7B model) |
| **RAM** (CPU pipeline) | 8 GB (2B, per-channel) | 32 GB (7B, per-element) |
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

## Expected Results

| Metric | Phi-2 (2.7B, CPU) | 7B Model (GPU T4) |
|--------|-------------------|-------------------|
| PT-BitNet (full) | ~20 min | ~4 min |
| Tequila | ~2 min | ~1 min |
| Total pipeline | ~22 min | ~5 min |
| Disk (ternary) | ~0.5 GB | ~1.2 GB |
| Inference speed (baked) | ~0.8 tok/s (CPU) | ~25 tok/s (GPU) |
| Inference speed (bitnet.cpp kernel) | ~3-5 tok/s (CPU) | ~80 tok/s (GPU) |

## Repository Structure

```
ternary-boost/
├── pyproject.toml              # Root workspace (uv)
├── run_pipeline.py             # Pipeline orchestrator
├── README.md
├── TODO.md
├── configs/
├── scripts/
├── notebooks/                  # Colab demo + ablation study
│
├── shared/                     # Checkpoint, logging, data loaders
├── pt_bitnet/                  # Stage 1: PTQ + outliers + compensation
├── tequila/                    # Stage 2: Deadzone trapping recovery
├── paretoq/                    # Research artifact: LSQ, ZO optimizer
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
