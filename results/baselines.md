# Baselines — microsoft/phi-2

Date: (run on Colab T4)

Standard protocol: WikiText-2 test set, max_len=128, 500 lines, seed=42, C4 calibration 128×2048.

## Known reference point

From `results/phi2_ternary.md`:
- FP16 teacher: WikiText-2 PPL = **27.39** (TensorFlow-style w/ stride, max_len=128)
- TernaryBoost (PT-BitNet + LoRA KD): PPL = **33.31** (ratio 1.216×)

## Results table

| Method | WikiText-2 PPL | PPL Ratio | Size (MB) | Quantize Time (s) | VRAM Peak (GB) | Notes |
|---|---|---|---|---|---|---|
| **FP16 teacher** (upper bound) | — | 1.000× | — | — | — | |
| **TernaryBoost** (PT-BitNet + LoRA KD) | — | — | — | — | — | Self-baseline |
| **GPTQModel 2-bit** | — | — | — | — | — | |
| **HQQ 1-bit binary** | — | — | — | — | — | |
| **AutoRound int2** | — | — | — | — | — | |
| **PT²-LLM** | — | — | — | — | — | Direct ternary competitor |
| **PTQTP** | — | — | — | — | — | If code released by mai/2026 |

## Decision (Week 1 gate)

- [ ] TernaryBoost within 5% PPL of PT²-LLM? → Continue as planned
- [ ] TernaryBoost >10% worse than PT²-LLM? → Adopt PT²-LLM's asymmetric ITF (`asymmetric=True, outlier_fraction=0`)
- [ ] PT²-LLM crashed on Phi-2? → Document finding, continue
- [ ] INT4 GPTQ strictly dominates ternary on size+PPL? → Pivot to inference-speed story

## Runner instructions

Run on Colab T4:

```bash
# Install dependencies
!pip install gptqmodel hqq auto-round lm-eval datasets

# Clone PT²-LLM
!git clone https://github.com/XIANGLONGYAN/PT2-LLM /content/PT2-LLM

# Run all baselines
!python scripts/baselines/run_baselines.py --model microsoft/phi-2
```

Or run individual baselines:

```bash
python scripts/baselines/run_baselines.py --model microsoft/phi-2 --baseline gptqmodel
python scripts/baselines/run_baselines.py --model microsoft/phi-2 --baseline hqq
python scripts/baselines/run_baselines.py --model microsoft/phi-2 --baseline autoround
```
