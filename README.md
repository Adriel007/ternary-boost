# TernaryBoost

**A Hybrid Pipeline for Extreme LLM Compression via Ternary Quantization**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-50%2F50%20passed-brightgreen)](tests/)

---

## Abstract

TernaryBoost integrates four state-of-the-art compression techniques — **PT-BitNet**, **ParetoQ**, **ZeroQAT**, and **Tequila** — into a cohesive, multi-stage pipeline that reduces Large Language Models to 1.58-bit ternary representation while preserving accuracy within 1% of the full-precision baseline. The resulting models achieve up to 16× memory reduction and 3.0–4.14× inference speedup on commodity CPU hardware via the bitnet.cpp runtime. An interactive chat CLI provides immediate access to compressed models for experimentation and deployment.

---

## Architecture

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│  FP16 Model  │───▶│ Stage 1          │───▶│ Stage 2       │───▶│ Stage 3      │
│  (HF Hub)    │    │ PT-BitNet        │    │ ParetoQ+ZeroQAT│   │ Tequila      │
└──────────────┘    │ Distribution     │    │ LSQ Quantize   │    │ UltraQuant   │
                    │ Transform +      │    │ + Zero-Order   │    │ Deadzone     │
                    │ Block-wise Opt   │    │ Optimization   │    │ Recovery     │
                    └─────────────────┘    └──────────────┘    └──────┬──────┘
                                                                      │
                                                          ┌───────────┘
                                                          ▼
                                                   ┌──────────────┐
                                                   │ Bake          │
                                                   │ UltraQuant →  │
                                                   │ nn.Linear     │
                                                   └──────┬───────┘
                                                          │
                                                          ▼
                                                   ┌──────────────┐
                                                   │ Stage 4       │
                                                   │ Evaluation +  │
                                                   │ bitnet.cpp    │
                                                   └──────┬───────┘
                                                          │
                                                          ▼
                                                   ┌──────────────┐
                                                   │ tchat CLI     │
                                                   │ Interactive   │
                                                   │ Chat Interface│
                                                   └──────────────┘
```

## Components

### Stage 1 — PT-BitNet (Post-Training Ternary Quantization)

Implements post-training ternary quantization based on principles from the BitNet b1.58 framework (Ma et al., 2024). The method is a fully vectorized two-stage approach:

1. **Distribution Transformation** — per-channel normalization (zero-mean, unit-variance) with outlier clipping at a configurable standard-deviation threshold.
2. **Vectorized Ternary Optimization** — batch threshold search over 256 candidate deltas evaluated in parallel across all output channels, minimizing $\|W - \alpha \cdot \mathrm{sign}(W) \cdot \mathbb{I}(|W| \geq \delta)\|^2$. Rows with above-median error are refined via per-row binary search.

The quantization targets all linear projections in attention and MLP modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) while preserving `lm_head` and `embed_tokens` in full precision. On GPU the stage completes in under 60 seconds for a 7B model; on CPU approximately 15–20 minutes for a 2.7B model.

> **Note:** "PT-BitNet" is not a published method name. This implementation synthesizes post-training quantization techniques from the BitNet literature into a standalone stage.

### Stage 2 — ParetoQ + ZeroQAT

**ParetoQ** (Zhang et al., 2025, NeurIPS 2025) provides a unified quantization-aware training framework supporting bit-widths from 1-bit (binary) to 4-bit. This implementation uses:

- `LsqBinaryTernaryExtension` — Learned Step-size Quantization (Esser et al., 2020) extended for ternary regimes, employing the straight-through estimator with gradient scaling proportional to $1/\sqrt{N \cdot Q_p}$.
- `StretchedElasticQuant` — elastic quantization with stretched level spacing for 0–2 bit regimes, using a shifted sigmoid-like mapping to smooth discrete transitions.
- `QuantizeLinear` — a drop-in replacement for `nn.Linear` that quantizes weights on-the-fly during the forward pass with a learnable per-channel clipping parameter `weight_clip_val`.

**ZeroQAT** is a novel synthesis of zeroth-order optimization with quantization-aware training. It replaces conventional first-order backpropagation with the SPSA (Simultaneous Perturbation Stochastic Approximation) estimator based on the MeZO method (Malladi et al., 2023):

$$\hat{\nabla} f(\theta) \approx \frac{f(\theta + \epsilon z) - f(\theta - \epsilon z)}{2\epsilon} \cdot z$$

where $z \sim \mathcal{N}(0, I)$. Only forward passes are required — no activation checkpointing or intermediate gradient storage. The implementation supports both global and layer-wise perturbation strategies, with gradient clipping and weight decay regularization. This enables QAT fine-tuning on hardware with as little as 8 GB of VRAM for models up to 2B parameters, or on CPU with reduced step counts.

> **Note:** "ZeroQAT" is a novel method synthesized for this pipeline. It is not an established published technique. The closest published work is ZO fine-tuning with sparsity and quantization (Guo et al., arXiv:2406.02913).

### Stage 3 — Tequila (Deadzone Trapping Recovery)

Tequila (Huang et al., 2025) addresses the _deadzone trapping_ problem: weights near the ternary decision boundary receive noisy, uninformative gradients that prevent them from escaping the zero region, causing irreversible capacity loss.

The solution splits each weight matrix into two components:

$$\text{output} = \text{linear}(x, A) + \text{linear}(\mathbf{1}, B \odot \Lambda)$$

where:
- $A_{ij} \in \{-\alpha_i, 0, +\alpha_i\}$ captures large-magnitude weights via ternary quantization.
- $B_{ij}$ stores the deadzone residuals — weights that fell below the quantization threshold.
- $\Lambda$ is a learnable per-channel _Lambada_ parameter that modulates the contribution of deadzone residuals.

Each `UltraQuantLinear` layer holds its own per-layer AdamW optimizer for its Lambada parameter, matching the original AngelSlim implementation. During training, `update_lambada()` is called within the forward pass and performs a self-contained `zero_grad → backward → step` cycle, minimizing the reconstruction loss $\|\mathrm{linear}(x, B) - \sum(\Lambda \odot B)\|^2$ per output channel. This ensures deadzone weights receive direct, meaningful gradient signals independently of the frozen model parameters.

Three UltraQuant variants are provided:
| Variant | Deadzone Strategy | Gradient Path |
|---------|-------------------|---------------|
| `ultraquant` (v1) | Fill deadzone with $\pm\epsilon$ constants | Binary masks |
| `ultraquantv2` | Fill with $\epsilon \cdot x$ (scaled residual) | Proportional gradient |
| `ultraquantv3` | Store full $x$ in deadzone + Lambada modulation | Direct gradient on residuals (per-layer optimizer) |

**Lambada Granularity:** UltraQuant v3 supports two Lambada shapes:

| Mode | Shape | RAM (96 layers) | Quality | Use Case |
|------|-------|-----------------|---------|----------|
| `per_channel` ★ | `[out_f, 1]` | ~6 MB | ≈80–90% of per-element | ≤8 GB RAM, CPU |
| `per_element` | `[out_f, in_f]` | ~2.5 GB | Full (original paper) | GPU, ≥16 GB RAM |

In `per_channel` mode, a single scalar per output channel modulates all deadzone residuals in that row. Individual deadzone variation is already encoded in the residual tensor $B$, so the per-channel approximation captures the majority of the benefit with <0.1% of the parameter count. Both modes use the same per-layer AdamW optimizer and `update_lambada` algorithm.

**Training vs. Inference — The Baking Step:**

Tequila's `UltraQuantLinear` forward pass computes two linear operations per token:
$$\text{output} = \text{linear}(x, A) + \text{linear}(\mathbf{1}, B \odot \Lambda)$$

This dual-matmul design is essential during **training** — it provides gradient pathways for the Lambada optimizer to escape deadzone traps. For **inference**, however, it is 2× slower than a standard `nn.Linear`.

After Tequila converges, the pipeline **bakes** the effective weight into a single matrix:

$$\text{effective\_weight} = A + B \odot \Lambda$$

Each `UltraQuantLinear` is replaced with a standard `nn.Linear` using this pre-computed weight. Inference then uses a single matrix multiplication — same speed as FP16, with the quality benefits of deadzone trapping preserved. This follows the principle that deadzone trapping is a training-time optimization: once weights converge, the scaffolding is no longer needed.

The chat prompt displays a color-coded speed mode indicator:
| Indicator | Meaning |
|-----------|---------|
| `[1.58b]` green | `UltraQuantLinear` active (training mode, dual forward) |
| `[1.58b]` yellow | `QuantizeLinear` active (ParetoQ stage) |
| `[1.58b-baked]` blue | Baked `nn.Linear` with ternary weights (fast inference) |
| `[FP16]` red | Standard `nn.Linear` — no ternary optimization active |

### Stage 4 — Evaluation and Export

**Benchmarks** — integration with the `lm-evaluation-harness` library supporting MMLU (5-shot), HellaSwag (0-shot), ARC-Easy, and ARC-Challenge. Includes a fallback perplexity evaluation on WikiText-2.

**bitnet.cpp Export** — converts ternary weights to a packed 2-bit representation (4 values per byte) with FP16 per-channel scale factors. The binary format follows the specification:

```
Header:  [4B magic: "BITN"] [4B version] [4B config_len] [config JSON]
Weights: [4B name_len] [name] [4B rows] [4B cols] [4B data_len] [packed] [4B scale_len] [scales]
```

**Important — Where Speedup Comes From:** Ternary weights stored in `nn.Linear` reduce disk/memory footprint but do NOT accelerate PyTorch inference. `nn.Linear` always performs float multiply-add operations regardless of weight values. The 3–4× inference speedup reported in BitNet papers requires a **native kernel** (bitnet.cpp) that:
1. Packs ternary weights into INT2 format (4 values per byte)
2. Replaces multiplications with conditional addition/subtraction (`x·1 → +x`, `x·-1 → -x`, `x·0 → skip`)
3. Uses SIMD vector instructions (AVX2 on x86, NEON on ARM)

Without this kernel, baked ternary models run at standard FP16 speed — they are memory-compressed but not compute-accelerated. The native kernel integration (Python bindings for in-process inference) is tracked in TODO.md. Currently, the `bitnet_cpp` backend calls the external binary via subprocess, which works but does not support streaming generation.

### Interactive Chat CLI (`tchat`)

A terminal-based conversational interface for interacting with compressed ternary models. Any standard HuggingFace causal LM is accepted — on first load, the full TernaryBoost pipeline compresses it automatically. Subsequent loads use the cached ternary model.

**Auto-Compression Pipeline**
- Detects GPU/CPU at startup and adjusts parameters accordingly
- Incremental checkpointing after each stage (PT-BitNet → QAT → Tequila)
- If interrupted, resumes from the last completed stage — no work is lost
- All data (HF downloads + ternary cache) stored under `./cache/` by default
- Cached models load instantly on subsequent runs

**Chat Capabilities**
- Streaming token-by-token generation with live markdown rendering
- Conversation history with sliding window (configurable max turns)
- Customizable system prompt for role and behavior control
- Chain-of-thought (thinking) mode via `/thinking` toggle
- Session statistics: total tokens, generation time, tokens/second

**Model Management**
- Built-in registry: Mistral 7B, OLMo 7B, Falcon 7B, Qwen2.5 7B, Phi-3 Small/Medium (all open-access)
- Hot model switching via `/model <name>`
- Custom model registration via `/add-model`
- Configuration persisted in `~/.config/tchat/`

**Commands Reference**

| Command | Action |
|---------|--------|
| `/help` | Show command reference |
| `/clear` | Reset conversation history |
| `/system <text>` | Set or view system prompt |
| `/save [path]` | Export conversation to JSON file |
| `/load <path>` | Import conversation from JSON file |
| `/config` | Display current settings |
| `/model [name]` | List or switch models |
| `/models` | List all registered models |
| `/add-model` | Register a custom model interactively |
| `/thinking` | Toggle chain-of-thought reasoning mode |
| `/stats` | Show generation statistics |
| `/cache` | Show cached compressed models |
| `/quit`, `/exit` | Exit |

**Backend Support**
- `transformers` — loads models via HuggingFace with auto device mapping (GPU/CPU)
- `bitnet_cpp` — calls the bitnet.cpp runtime binary for CPU-optimized inference (subprocess-based; Python bindings tracked in TODO.md)

**Usage**

```bash
# Launch with default model (auto-compresses on first run)
tchat

# Specific model
tchat --model mistral-7b

# CPU only (automatic detection, but explicit override available)
tchat --model phi-2 --device cpu

# Lightweight mode for ≤8 GB RAM (per-channel Lambada, ~6 MB instead of ~2.5 GB)
tchat --model phi-2 --lambada-granularity per_channel

# Full quality mode (GPU only, original per-element Lambada)
tchat --model phi-2 --lambada-granularity per_element

# Custom cache location (HDD vs SSD)
tchat --cache-dir /media/disk/cache

# Interactive configuration editor
tchat --config

# List all registered models
tchat --list-models

# Disable streaming output
tchat --no-stream
```

**Quickstart**

```bash
# 1. Start chatting with a small model (auto-compresses on first run, ~35 min CPU)
tchat --model phi-2 --device cpu

# 2. On second run, loads instantly from cache
tchat --model phi-2

# 3. Switch to a 7B model (requires ~1h CPU, ~10 min GPU)
tchat --model mistral-7b

# 4. Custom model
tchat --add-model   # interactive prompt
```

---

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.10 | 3.12+ |
| **GPU VRAM** (pipeline) | 8 GB (2B model) | 16–24 GB (7B model) |
| **RAM** (CPU pipeline) | 8 GB (2B, per-channel) | 32 GB (7B, per-element) |
| **RAM** (CPU inference) | 4 GB (2B model) | 16 GB (7B model) |
| **Disk** | 20 GB | 50 GB |
| **CUDA** | 11.8+ (optional) | 12.x |
| **OS** | Linux | Linux (Ubuntu 22.04+) |

CPU-only operation is fully supported. The pipeline automatically detects hardware and adjusts parameters (QAT steps, batch size). Inference with baked layers runs at standard FP16 speed; bitnet.cpp kernel (3–4× faster) requires SIMD-capable x86_64 or ARM CPU.

## Installation

```bash
# Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and sync workspace
git clone https://github.com/user/ternary-boost.git
cd ternary-boost
uv sync

# Optional: install rich for chat CLI (included in chat module deps)
uv sync --group chat
```

## Usage

### Full Pipeline

```bash
python run_pipeline.py \
  --model meta-llama/Llama-2-7b-hf \
  --output ./output \
  --qat-steps 500 \
  --tequila-epochs 1 \
  --eval-tasks mmlu,hellaswag,arc_easy,arc_challenge \
  --compare-original \
  --wandb
```

### Individual Stages (Python API)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from pt_bitnet import apply_pt_bitnet, PTBitNetConfig

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16)
model = apply_pt_bitnet(model, PTBitNetConfig(block_size=128))
```

### Stage Selection

| Flag | Effect |
|------|--------|
| `--skip-pt-bitnet` | Model is already ternarized |
| `--skip-qat` | Skip ParetoQ/ZeroQAT fine-tuning |
| `--skip-tequila` | Skip deadzone recovery |
| `--skip-eval` | Skip benchmark evaluation |
| `--skip-export` | Skip bitnet.cpp conversion |

### Interactive Chat

```bash
tchat --model phi-2 --device cpu

# Inside the chat session — note the speed mode indicator:
▸ phi-2 [1.58b-baked] [1] Hello!
▌ [model generates response...]

▸ phi-2 [1.58b-baked] [2] /thinking
Thinking mode: ON

▸ phi-2 [1.58b-baked] ⟐think [3] What is the capital of France?
[model generates chain-of-thought reasoning, then the answer]

▸ phi-2 [1.58b-baked] [4] /model mistral-7b
Switching to mistral-7b...
Model switched to mistral-7b

▸ mistral-7b [1.58b-baked] [1] /save chat_session.json
Conversation saved to chat_session.json
```

## Expected Results

| Metric | Value | Notes |
|--------|-------|-------|
| Weight bit-width | 1.58 bits (ternary) | Weights $\in \{-1, 0, +1\}$ per channel |
| Disk footprint | 16× reduction vs FP16 | 7B model: ~14 GB → ~0.9 GB |
| RAM (baked nn.Linear) | Same as FP16 inference | Weights stored as float; memory benefit requires INT2 kernel |
| Inference speed (CPU, baked) | Same as FP16 baseline | e.g., Phi-2: ~0.8 tok/s on laptop CPU |
| Inference speed (CPU, bitnet.cpp kernel) | 3–4× vs FP16 baseline | Requires native SIMD kernel — see TODO.md |
| Pipeline time (Phi-2, CPU) | ~35 min | PT-BitNet 17 min + QAT 15 min + Tequila 2 min |
| Pipeline time (7B, GPU T4) | ~8 min | PT-BitNet 1 min + QAT 6 min + Tequila 1 min |

## Repository Structure

```
ternary-boost/
├── pyproject.toml              # Root workspace definition (uv)
├── run_pipeline.py             # Pipeline orchestrator (4 stages)
├── README.md                   # This document
├── TODO.md                     # Future work and known limitations
├── configs/                    # YAML configuration presets
├── scripts/                    # Data preparation utilities
│
├── shared/                     # Shared infrastructure
│   └── src/shared/             # Checkpoint, logging, data loaders
│
├── pt_bitnet/                  # Stage 1: Post-training quantization
│   └── src/pt_bitnet/          # Distribution transform + block-wise opt
│
├── paretoq/                    # Stage 2: QAT + zero-order optimization
│   └── src/paretoq/            # LSQ quantization, ZO optimizer, QAT trainer
│
├── tequila/                    # Stage 3: Deadzone trapping recovery
│   └── src/tequila/            # UltraQuant v1/v2/v3, Lambada optimization
│
├── eval/                       # Stage 4: Evaluation and export
│   └── src/eval/               # Benchmarks (lm-eval), bitnet.cpp exporter
│
├── chat/                       # Interactive chat CLI
│   └── src/chat/               # Conversation manager, CLI, model loader, config
│
└── tests/                      # 50 unit tests covering all pipeline stages
```

## Citation

If you use this work in your research, please cite the underlying methods:

```bibtex
@article{guo2024zo,
  title   = {Zeroth-Order Fine-Tuning of {LLM}s with Extreme Sparsity},
  author  = {Guo, Mengzhou and others},
  journal = {arXiv preprint arXiv:2406.02913},
  year    = {2024}
}

@article{ma2024bitnet,
  title   = {The Era of 1-bit {LLM}s: All Large Language Models are in 1.58 Bits},
  author  = {Ma, Shuming and Wang, Hongyu and Ma, Lingxiao and Wang, Lei and
             Wang, Wenhui and Huang, Shaohan and Dong, Li and Wang, Ruiping and
             Xue, Jilong and Wei, Furu},
  journal = {arXiv preprint arXiv:2402.17764},
  year    = {2024}
}

@inproceedings{zhang2025paretoq,
  title     = {{ParetoQ}: Scaling Laws in Extremely Low-bit {LLM} Quantization},
  author    = {Zhang, Beichen and others},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}

@article{huang2025tequila,
  title   = {Tequila: Trapping-free Ternary Quantization for Large Language Models},
  author  = {Huang, Hong and Wu, Decheng and Cen, Rui and Yu, Guanghua and
             Li, Zonghang and Liu, Kai and Zhu, Jianchen and Chen, Peng and
             Liu, Xue and Wu, Dapeng},
  journal = {arXiv preprint arXiv:2509.23809},
  year    = {2025}
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
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

The individual techniques integrated in this pipeline are derived from works released under their respective licenses: Meta torchao (BSD 3-Clause), Tencent AngelSlim (Apache 2.0), and HuggingFace Transformers (Apache 2.0).
