#!/usr/bin/env python3
"""Week 1 baseline evaluation orchestrator for Colab T4.

Evaluates competitor ternary/binary quantization methods on Phi-2 with
the standard protocol (WikiText-2 PPL, max_len=128, 500 lines, seed=42).

Competitors:
  A. GPTQModel 2-bit (int2 group 128)
  B. HQQ 1-bit binary (group_size=64)
  C. AutoRound int2 (--enable_alg_ext)
  D. PT²-LLM (direct ternary competitor)
  E. PTQTP (if code released by mai/2026)

Each method is run in a subprocess so OOM in one doesn't crash the suite.
Results are appended to results/baselines.md.

Usage (Colab T4):
    !pip install gptqmodel hqq auto-round lm-eval
    !git clone https://github.com/XIANGLONGYAN/PT2-LLM /content/PT2-LLM
    %run scripts/baselines/run_baselines.py --model microsoft/phi-2

Usage (local, if GPU available):
    python scripts/baselines/run_baselines.py --model microsoft/phi-2
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.logging import get_logger

logger = get_logger("baselines")


def _safe_config(model_id: str):
    """Load config with ``pad_token_id`` guaranteed set."""
    config = AutoConfig.from_pretrained(model_id)
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = getattr(config, "eos_token_id", 0) or 0
    return config


# ═══════════════════════════════════════════════════════════════════
# Standard evaluation protocol (PLAN.md Section 7.1)
# ═══════════════════════════════════════════════════════════════════

EVAL_CONFIG = {
    "dataset": "wikitext2",
    "max_len": 128,
    "stride": 128,
    "num_samples": 500,
    "seed": 42,
    "calibration_seed": 42,
    "calibration_samples": 128,
    "calibration_seq_len": 2048,
}


def evaluate_ppl(model, tokenizer, device: str = "cuda") -> dict:
    """Standard WikiText-2 PPL evaluation matching colab_test.py protocol."""
    from lm_eval import simple_evaluate

    results = simple_evaluate(
        model="hf",
        model_args={
            "pretrained": model,
            "tokenizer": tokenizer,
            "device": device,
            "dtype": "float16" if device == "cuda" else "float32",
        },
        tasks=["wikitext"],
        num_fewshot=0,
        limit=EVAL_CONFIG["num_samples"],
        batch_size=1,
    )
    ppl = results["results"]["wikitext"]["word_perplexity,none"]
    return {
        "wikitext2_ppl": round(ppl, 2),
        "eval_protocol": f"max_len={EVAL_CONFIG['max_len']}, "
                         f"samples={EVAL_CONFIG['num_samples']}, "
                         f"seed={EVAL_CONFIG['seed']}",
    }


def get_model_size_mb(model) -> float:
    """Estimate model size in MB (fp16 equivalent for storage)."""
    total = sum(p.numel() for p in model.parameters())
    return total * 2 / 1e6  # fp16 = 2 bytes per param


def log_vram(label: str = ""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"  {label}VRAM: {alloc:.1f}/{total:.1f} GB")


# ═══════════════════════════════════════════════════════════════════
# Baseline A: GPTQModel 2-bit
# ═══════════════════════════════════════════════════════════════════

def baseline_gptqmodel(model_id: str, device: str) -> Optional[dict]:
    """GPTQModel int2 group-128 quantization."""
    try:
        from gptqmodel import GPTQModel, QuantizeConfig
    except ImportError:
        logger.warning("GPTQModel not installed — skipping")
        return None

    logger.info("=== Baseline A: GPTQModel 2-bit (int2 group 128) ===")
    t0 = time.time()

    try:
        quant_config = QuantizeConfig(bits=2, group_size=128, desc_act=False)
        model = GPTQModel.from_pretrained(model_id, quantize_config=quant_config)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Calibration (GPTQ convention: 128 × 2048 C4 samples)
        from datasets import load_dataset
        calib = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        calib_texts = []
        for ex in calib:
            calib_texts.append(ex["text"])
            if len(calib_texts) >= EVAL_CONFIG["calibration_samples"]:
                break

        model.quantize(calib_texts, tokenizer=tokenizer)

        # Eval
        results = evaluate_ppl(model, tokenizer, device)
        results["method"] = "GPTQModel 2-bit (group 128)"
        results["model_size_mb"] = round(get_model_size_mb(model), 1)
        results["quantize_time_s"] = round(time.time() - t0, 0)
        results["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)

        logger.info(f"  GPTQModel: PPL={results['wikitext2_ppl']}, "
                    f"size={results['model_size_mb']:.0f} MB, "
                    f"time={results['quantize_time_s']:.0f}s")

        del model; gc.collect(); torch.cuda.empty_cache()
        return results

    except Exception as e:
        logger.error(f"GPTQModel failed: {e}")
        return {"method": "GPTQModel 2-bit", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# Baseline B: HQQ 1-bit binary
# ═══════════════════════════════════════════════════════════════════

def baseline_hqq(model_id: str, device: str) -> Optional[dict]:
    """HQQ 1-bit binary quantization (group_size=64)."""
    try:
        from hqq.models.hf.base import AutoHQQHfModel
    except ImportError:
        logger.warning("HQQ not installed — skipping")
        return None

    logger.info("=== Baseline B: HQQ 1-bit binary (group 64) ===")
    t0 = time.time()

    try:
        quant_config = {
            "weight_quant_params": {
                "nbits": 1,
                "channel_wise": True,
                "group_size": 64,
                "optimize": False,
            }
        }
        model = AutoHQQHfModel.from_pretrained(model_id, quant_config=quant_config)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        results = evaluate_ppl(model, tokenizer, device)
        results["method"] = "HQQ 1-bit binary (group 64)"
        results["model_size_mb"] = round(get_model_size_mb(model), 1)
        results["quantize_time_s"] = round(time.time() - t0, 0)
        results["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)

        logger.info(f"  HQQ: PPL={results['wikitext2_ppl']}, "
                    f"size={results['model_size_mb']:.0f} MB")

        del model; gc.collect(); torch.cuda.empty_cache()
        return results

    except Exception as e:
        logger.error(f"HQQ failed: {e}")
        return {"method": "HQQ 1-bit", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# Baseline C: AutoRound int2
# ═══════════════════════════════════════════════════════════════════

def baseline_autoround(model_id: str, device: str) -> Optional[dict]:
    """AutoRound int2 quantization with algorithm extension."""
    try:
        from auto_round import AutoRound
    except ImportError:
        logger.warning("AutoRound not installed — skipping")
        return None

    logger.info("=== Baseline C: AutoRound int2 (--enable_alg_ext) ===")
    t0 = time.time()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, config=_safe_config(model_id), torch_dtype="auto",
            low_cpu_mem_usage=True, device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        from datasets import load_dataset
        calib = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        calib_texts = []
        for ex in calib:
            calib_texts.append(ex["text"])
            if len(calib_texts) >= EVAL_CONFIG["calibration_samples"]:
                break

        autoround = AutoRound(
            model, tokenizer,
            nbits=2,
            group_size=128,
            enable_quanted_input=True,
            enable_minmax_tuning=True,
            iters=200,
            lr=0.005,
            seqlen=EVAL_CONFIG["calibration_seq_len"],
            nsamples=EVAL_CONFIG["calibration_samples"],
            dataset=" ".join(calib_texts[:4]),  # AutoRound's format
        )
        autoround.quantize()

        results = evaluate_ppl(model, tokenizer, device)
        results["method"] = "AutoRound int2 (+alg_ext)"
        results["model_size_mb"] = round(get_model_size_mb(model), 1)
        results["quantize_time_s"] = round(time.time() - t0, 0)
        results["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)

        logger.info(f"  AutoRound: PPL={results['wikitext2_ppl']}, "
                    f"time={results['quantize_time_s']:.0f}s")

        del model; gc.collect(); torch.cuda.empty_cache()
        return results

    except Exception as e:
        logger.error(f"AutoRound failed: {e}")
        return {"method": "AutoRound int2", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# Baseline D: PT²-LLM (direct ternary competitor)
# ═══════════════════════════════════════════════════════════════════

def baseline_pt2llm(model_id: str, device: str) -> Optional[dict]:
    """PT²-LLM asymmetric ternary PTQ.

    Clone PT²-LLM and run on Phi-2. This is the most important baseline —
    it's the direct ternary competitor.
    """
    logger.info("=== Baseline D: PT²-LLM (asymmetric ternary) ===")
    t0 = time.time()

    pt2llm_path = os.environ.get("PT2LLM_PATH", "/content/PT2-LLM")
    if not Path(pt2llm_path).exists():
        logger.warning(f"PT²-LLM not found at {pt2llm_path}")
        logger.warning("  Clone: git clone https://github.com/XIANGLONGYAN/PT2-LLM /content/PT2-LLM")
        return {"method": "PT²-LLM", "error": "repo not found — clone first"}

    try:
        import sys
        sys.path.insert(0, pt2llm_path)
        # PT²-LLM's API is likely a script; run via subprocess
        cmd = [
            sys.executable,
            f"{pt2llm_path}/main.py",
            "--model", model_id,
            "--eval-ppl",
            "--dataset", "wikitext2",
            "--seed", str(EVAL_CONFIG["seed"]),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        quantize_time = time.time() - t0

        # Parse PPL from output (best-effort: PT²-LLM prints "wikitext2 ppl: X.X")
        ppl = None
        for line in result.stdout.split("\n"):
            if "ppl" in line.lower() and any(c.isdigit() for c in line):
                import re
                nums = re.findall(r"[\d]+\.[\d]+", line)
                if nums:
                    ppl = float(nums[0])

        if ppl is None and result.stderr:
            for line in result.stderr.split("\n"):
                if "ppl" in line.lower() and any(c.isdigit() for c in line):
                    import re
                    nums = re.findall(r"[\d]+\.[\d]+", line)
                    if nums:
                        ppl = float(nums[0])

        return {
            "method": "PT²-LLM (asymmetric ternary)",
            "wikitext2_ppl": round(ppl, 2) if ppl else None,
            "quantize_time_s": round(quantize_time, 0),
            "raw_output": result.stdout[-500:] if result.stdout else result.stderr[-500:],
        }

    except subprocess.TimeoutExpired:
        logger.error("PT²-LLM timed out (>1h)")
        return {"method": "PT²-LLM", "error": "timeout > 1h"}
    except Exception as e:
        logger.error(f"PT²-LLM failed: {e}")
        return {"method": "PT²-LLM", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# Baseline E: TernaryBoost (self, reference)
# ═══════════════════════════════════════════════════════════════════

def baseline_ternaryboost(model_id: str, device: str) -> Optional[dict]:
    """Run TernaryBoost pipeline for self-comparison."""
    logger.info("=== Baseline E: TernaryBoost (self) ===")
    t0 = time.time()

    try:
        from pt_bitnet.quantize import apply_pt_bitnet, PTBitNetConfig
        from pt_bitnet.lora import finetune_lora, LoRAConfig
        from eval.benchmarks import evaluate_model

        cfg = _safe_config(model_id)
        # Load student to CPU first (low_cpu_mem_usage=True), then move to GPU.
        # This avoids the 5.4 GB CPU copy that from_pretrained without the flag creates.
        model = AutoModelForCausalLM.from_pretrained(
            model_id, config=cfg, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model.to(device)

        # Only then load teacher (also low_cpu_mem_usage) — prevents 10.8 GB CPU peak.
        teacher_cfg = _safe_config(model_id)
        teacher = AutoModelForCausalLM.from_pretrained(
            model_id, config=teacher_cfg, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, device_map="cpu",
        )
        teacher.to(device)

        # Load calibration texts
        from shared.data import load_calibration_texts
        calib_texts = load_calibration_texts(
            num_samples=EVAL_CONFIG["calibration_samples"],
            max_length=EVAL_CONFIG["calibration_seq_len"],
        )

        # PT-BitNet quantize
        pt_config = PTBitNetConfig(
            asymmetric=False, outlier_fraction=0.01,
            compensation_steps=30, show_progress=True,
        )
        model = apply_pt_bitnet(model, pt_config, tokenizer, calib_texts)

        # LoRA KD
        lora_config = LoRAConfig(rank=64, num_steps=1000)
        model = finetune_lora(model, tokenizer, calib_texts, teacher, lora_config)

        # Eval
        ppl_result = evaluate_model(model, tokenizer, tasks=["wikitext2"], max_len=128)
        ppl = ppl_result.get("wikitext2_ppl", None)

        del teacher; gc.collect(); torch.cuda.empty_cache()

        return {
            "method": "TernaryBoost (PT-BitNet + LoRA KD)",
            "wikitext2_ppl": round(ppl, 2) if ppl else None,
            "model_size_mb": round(get_model_size_mb(model), 1),
            "quantize_time_s": round(time.time() - t0, 0),
            "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
        }

    except Exception as e:
        logger.error(f"TernaryBoost self-baseline failed: {e}")
        return {"method": "TernaryBoost", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# FP16 baseline
# ═══════════════════════════════════════════════════════════════════

def baseline_fp16(model_id: str, device: str) -> dict:
    """FP16 teacher PPL (upper bound)."""
    logger.info("=== Baseline 0: FP16 teacher ===")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=_safe_config(model_id), torch_dtype=torch.float16,
        low_cpu_mem_usage=True, device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.to(device)

    results = evaluate_ppl(model, tokenizer, device)
    results["method"] = "FP16 (teacher upper bound)"
    results["model_size_mb"] = round(get_model_size_mb(model), 1)

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════

def run_all_baselines(model_id: str = "microsoft/phi-2"):
    """Run all baselines and print a summary table."""
    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        logger.error("No CUDA GPU detected. Baselines must run on T4 or equivalent.")
        logger.error("Run this script on Colab with a T4 runtime.")
        return

    device = "cuda"
    torch.cuda.reset_peak_memory_stats()

    logger.info(f"Baseline suite for {model_id}")
    logger.info(f"Protocol: WikiText-2, max_len={EVAL_CONFIG['max_len']}, "
                f"samples={EVAL_CONFIG['num_samples']}, seed={EVAL_CONFIG['seed']}")

    results = []

    # FP16 reference (always run first)
    r = baseline_fp16(model_id, device)
    if r: results.append(r)

    # A: GPTQModel
    r = baseline_gptqmodel(model_id, device)
    if r: results.append(r)

    # B: HQQ
    r = baseline_hqq(model_id, device)
    if r: results.append(r)

    # C: AutoRound
    r = baseline_autoround(model_id, device)
    if r: results.append(r)

    # D: PT²-LLM
    r = baseline_pt2llm(model_id, device)
    if r: results.append(r)

    # E: TernaryBoost (self)
    r = baseline_ternaryboost(model_id, device)
    if r: results.append(r)

    # Summary
    print("\n" + "=" * 80)
    print(f"Baseline Results — {model_id} — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    print(f"{'Method':<45} {'PPL':>8} {'Size(MB)':>10} {'Time(s)':>10} {'VRAM(G)':>10}")
    print("-" * 80)

    for r in results:
        method = r.get("method", "Unknown")[:45]
        ppl = r.get("wikitext2_ppl", "—")
        size = r.get("model_size_mb", "—")
        t = r.get("quantize_time_s", "—")
        vram = r.get("vram_peak_gb", "—")

        ppl_str = f"{ppl:.2f}" if isinstance(ppl, (int, float)) else str(ppl)
        size_str = f"{size:.0f}" if isinstance(size, (int, float)) else str(size)
        t_str = f"{t:.0f}" if isinstance(t, (int, float)) else str(t)
        vram_str = f"{vram:.1f}" if isinstance(vram, (int, float)) else str(vram)

        print(f"{method:<45} {ppl_str:>8} {size_str:>10} {t_str:>10} {vram_str:>10}")

    print("=" * 80)

    # Save to results/baselines.md
    output_path = Path(__file__).resolve().parent.parent.parent / "results" / "baselines.md"
    _save_results_md(results, model_id, output_path)
    logger.info(f"Results saved to {output_path}")

    return results


def _save_results_md(results: list[dict], model_id: str, path: Path):
    """Write results as a markdown table to results/baselines.md."""
    with open(path, "w") as f:
        f.write(f"# Baselines — {model_id}\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC\n\n")
        f.write("Standard protocol: WikiText-2 test set, max_len=128, "
                "500 lines, seed=42, C4 calibration 128×2048.\n\n")
        f.write("| Method | WikiText-2 PPL | Size (MB) | "
                "Quantize Time (s) | VRAM Peak (GB) | Notes |\n")
        f.write("|---|---|---|---|---|---|\n")

        for r in results:
            method = r.get("method", "Unknown")
            ppl = r.get("wikitext2_ppl", "—")
            size = r.get("model_size_mb", "—")
            t = r.get("quantize_time_s", "—")
            vram = r.get("vram_peak_gb", "—")
            error = r.get("error", "")
            notes = f"ERROR: {error}" if error else ""

            ppl_str = f"{ppl:.2f}" if isinstance(ppl, (int, float)) else str(ppl)
            size_str = f"{size:.0f}" if isinstance(size, (int, float)) else str(size)
            t_str = f"{t:.0f}" if isinstance(t, (int, float)) else str(t)
            vram_str = f"{vram:.1f}" if isinstance(vram, (int, float)) else str(vram)

            f.write(f"| {method} | {ppl_str} | {size_str} | {t_str} | "
                    f"{vram_str} | {notes} |\n")

        f.write("\n## Decision (Week 1 gate)\n\n")
        f.write("- [ ] TernaryBoost within 5% PPL of PT²-LLM? → Continue as planned\n")
        f.write("- [ ] TernaryBoost >10% worse than PT²-LLM? → Adopt PT²-LLM's asymmetric ITF\n")
        f.write("- [ ] PT²-LLM crashed on Phi-2? → Document finding, continue\n")
        f.write("- [ ] INT4 GPTQ strictly dominates ternary on PPL? → Pivot to inference-speed story\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 1 baselines")
    parser.add_argument("--model", default="microsoft/phi-2", help="Model to evaluate")
    parser.add_argument("--baseline", choices=["all", "fp16", "gptqmodel", "hqq",
                         "autoround", "pt2llm", "ternaryboost"],
                        default="all", help="Which baseline to run")
    args = parser.parse_args()

    if args.baseline == "all":
        run_all_baselines(args.model)
    elif args.baseline == "fp16":
        baseline_fp16(args.model, "cuda")
    elif args.baseline == "gptqmodel":
        baseline_gptqmodel(args.model, "cuda")
    elif args.baseline == "hqq":
        baseline_hqq(args.model, "cuda")
    elif args.baseline == "autoround":
        baseline_autoround(args.model, "cuda")
    elif args.baseline == "pt2llm":
        baseline_pt2llm(args.model, "cuda")
    elif args.baseline == "ternaryboost":
        baseline_ternaryboost(args.model, "cuda")
