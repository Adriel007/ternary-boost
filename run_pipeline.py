#!/usr/bin/env python3
"""Hybrid Ternary LLM Compression Pipeline.

Orchestrates four state-of-the-art techniques into a unified compression flow:

  Stage 1 — PT-BitNet: Post-training ternary quantization
  Stage 2 — ParetoQ + ZeroQAT: QAT with zero-order optimization
  Stage 3 — Tequila: Deadzone trapping recovery
  Stage 4 — Evaluation + bitnet.cpp export

Usage:
  python run_pipeline.py \
    --model meta-llama/Llama-2-7b-hf \
    --output ./output \
    --qat-steps 500 \
    --tequila-epochs 1 \
    --eval-tasks mmlu,hellaswag,arc_easy,arc_challenge
"""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

from shared.logging import get_logger, MetricsTracker, log_memory_usage
from shared.checkpoint import save_checkpoint, load_checkpoint

from pt_bitnet import apply_pt_bitnet, PTBitNetConfig
from paretoq import apply_paretoq_qat, ZeroQATConfig
from tequila import apply_tequila, TequilaConfig
from eval import run_benchmarks, export_to_bitnet_cpp, BitNetExportConfig, EvalConfig

logger = get_logger("pipeline")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hybrid Ternary LLM Compression Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory for all artifacts")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype")

    parser.add_argument("--skip-pt-bitnet", action="store_true",
                        help="Skip PT-BitNet stage (model already ternarized)")
    parser.add_argument("--skip-qat", action="store_true",
                        help="Skip ParetoQ/ZeroQAT stage")
    parser.add_argument("--skip-tequila", action="store_true",
                        help="Skip Tequila stage")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation stage")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip bitnet.cpp export")

    pt_group = parser.add_argument_group("PT-BitNet")
    pt_group.add_argument("--block-size", type=int, default=128,
                           help="Block size for block-wise optimization")
    pt_group.add_argument("--clip-threshold", type=float, default=3.0,
                           help="Outlier clipping threshold (std devs)")

    qat_group = parser.add_argument_group("ParetoQ / ZeroQAT")
    qat_group.add_argument("--w-bits", type=int, default=1,
                            help="Weight bit-width (1=ternary)")
    qat_group.add_argument("--qat-steps", type=int, default=500,
                            help="Number of QAT training steps")
    qat_group.add_argument("--qat-lr", type=float, default=1e-5,
                            help="QAT learning rate")
    qat_group.add_argument("--perturbation-scale", type=float, default=1e-3,
                            help="ZO perturbation scale")
    qat_group.add_argument("--qat-batch-size", type=int, default=4,
                            help="QAT batch size (small for ZO)")
    qat_group.add_argument("--grad-clip", type=float, default=1.0,
                            help="Gradient clipping for ZO estimates")

    teq_group = parser.add_argument_group("Tequila")
    teq_group.add_argument("--quant-method", type=str, default="ultraquantv3",
                            choices=["ultraquant", "ultraquantv2", "ultraquantv3"],
                            help="UltraQuant variant for deadzone handling")
    teq_group.add_argument("--tequila-epochs", type=int, default=1,
                            help="Number of Tequila calibration epochs")
    teq_group.add_argument("--range-of-lambada", type=float, default=0.01,
                            help="Initial range for Lambada parameters")

    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument("--eval-tasks", type=str,
                             default="mmlu,hellaswag,arc_easy,arc_challenge",
                             help="Comma-separated eval tasks")
    eval_group.add_argument("--eval-batch-size", type=int, default=16,
                             help="Evaluation batch size")
    eval_group.add_argument("--compare-original", action="store_true",
                             help="Also evaluate original model for comparison")

    wandb_group = parser.add_argument_group("Logging")
    wandb_group.add_argument("--wandb", action="store_true",
                              help="Enable wandb logging")
    wandb_group.add_argument("--wandb-project", type=str, default="ternary-boost",
                              help="wandb project name")
    wandb_group.add_argument("--run-name", type=str, default=None,
                              help="wandb run name")

    return parser


class TernaryBoostPipeline:
    """Orchestrates the full hybrid ternary compression pipeline."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.metrics = MetricsTracker()

        os.makedirs(args.output, exist_ok=True)

        self.pt_bitnet_output = os.path.join(args.output, "stage1_pt_bitnet")
        self.paretoq_output = os.path.join(args.output, "stage2_paretoq")
        self.tequila_output = os.path.join(args.output, "stage3_tequila")
        self.export_output = os.path.join(args.output, "stage4_export")

        self.dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.dtype = self.dtype_map[args.dtype]

    def run(self) -> dict:
        """Execute the full pipeline sequentially."""
        start_time = time.time()

        if self.args.wandb:
            import wandb
            wandb.init(
                project=self.args.wandb_project,
                name=self.args.run_name or f"ternary-boost-{int(start_time)}",
                config=vars(self.args),
            )

        logger.info("=" * 60)
        logger.info("TERNARY-BOOST: Hybrid LLM Compression Pipeline")
        logger.info(f"Model: {self.args.model}")
        logger.info(f"Output: {self.args.output}")
        logger.info("=" * 60)

        logger.info("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        original_model = None
        if self.args.compare_original and not self.args.skip_eval:
            original_model = model
            original_tokenizer = tokenizer

        log_memory_usage(logger, "After model load: ")

        if not self.args.skip_pt_bitnet:
            model = self._run_pt_bitnet(model)
        else:
            logger.info("Skipping PT-BitNet stage")

        if not self.args.skip_qat:
            model = self._run_paretoq_qat(model, tokenizer)
        else:
            logger.info("Skipping ParetoQ/ZeroQAT stage")

        if not self.args.skip_tequila:
            model = self._run_tequila(model, tokenizer)
        else:
            logger.info("Skipping Tequila stage")

        if not self.args.skip_eval:
            self._run_evaluation(model, tokenizer, original_model)

        if not self.args.skip_export:
            self._run_export(model)

        elapsed = time.time() - start_time
        logger.info(f"Pipeline complete in {elapsed:.1f}s ({elapsed/60:.1f}m)")

        self.metrics.record("pipeline", total_time_s=elapsed)

        metadata_path = os.path.join(self.args.output, "pipeline_results.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metrics.to_dict(), f, indent=2)
        logger.info(f"Results saved to {metadata_path}")

        if self.args.wandb:
            import wandb
            wandb.log(self.metrics.to_dict())
            wandb.finish()

        return self.metrics.to_dict()

    def _run_pt_bitnet(self, model):
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: PT-BitNet Post-Training Quantization")
        logger.info("=" * 60)

        pt_config = PTBitNetConfig(
            block_size=self.args.block_size,
            outlier_clip_threshold=self.args.clip_threshold,
        )

        model = apply_pt_bitnet(model, pt_config)
        save_checkpoint(model, self.pt_bitnet_output, metadata={
            "stage": "pt_bitnet",
            "config": asdict(pt_config),
        })
        log_memory_usage(logger, "After PT-BitNet: ")
        return model

    def _run_paretoq_qat(self, model, tokenizer):
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2: ParetoQ + ZeroQAT QAT Fine-Tuning")
        logger.info("=" * 60)

        zo_config = ZeroQATConfig(
            learning_rate=self.args.qat_lr,
            max_steps=self.args.qat_steps,
            perturbation_scale=self.args.perturbation_scale,
            grad_clip=self.args.grad_clip,
        )

        dummy_dataset = [{"input_ids": tokenizer("Hello world", return_tensors="pt")["input_ids"][0]}] * 100
        train_dataloader = DataLoader(
            dummy_dataset,
            batch_size=self.args.qat_batch_size,
            shuffle=True,
            collate_fn=lambda batch: {
                "input_ids": torch.nn.utils.rnn.pad_sequence(
                    [b["input_ids"] for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id or 0
                ),
                "labels": torch.nn.utils.rnn.pad_sequence(
                    [b["input_ids"] for b in batch], batch_first=True, padding_value=-100
                ),
                "attention_mask": torch.nn.utils.rnn.pad_sequence(
                    [torch.ones_like(b["input_ids"]) for b in batch], batch_first=True, padding_value=0
                ),
            },
        )

        model = apply_paretoq_qat(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            output_path=self.paretoq_output,
            w_bits=self.args.w_bits,
            qat_steps=self.args.qat_steps,
            zo_config=zo_config,
        )
        log_memory_usage(logger, "After ParetoQ/ZeroQAT: ")
        return model

    def _run_tequila(self, model, tokenizer):
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 3: Tequila Deadzone Trapping Recovery")
        logger.info("=" * 60)

        tequila_config = TequilaConfig(
            quant_method=self.args.quant_method,
            num_epochs=self.args.tequila_epochs,
            range_of_lambada=self.args.range_of_lambada,
        )

        dummy_dataset = [{"input_ids": tokenizer("Hello world", return_tensors="pt")["input_ids"][0]}] * 50
        calibration_dataloader = DataLoader(
            dummy_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=lambda batch: {
                "input_ids": torch.nn.utils.rnn.pad_sequence(
                    [b["input_ids"] for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id or 0
                ),
            },
        )

        model = apply_tequila(model, calibration_dataloader, tequila_config)
        save_checkpoint(model, self.tequila_output, metadata={
            "stage": "tequila",
            "config": asdict(tequila_config),
        })
        log_memory_usage(logger, "After Tequila: ")
        return model

    def _run_evaluation(self, model, tokenizer, original_model=None):
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 4a: Evaluation")
        logger.info("=" * 60)

        eval_config = EvalConfig(
            tasks=tuple(self.args.eval_tasks.split(",")),
            batch_size=self.args.eval_batch_size,
        )

        results = run_benchmarks(
            model=model,
            tokenizer=tokenizer,
            original_model=original_model,
            config=eval_config,
        )

        for task, acc in results.get("quantized", {}).items():
            self.metrics.record("evaluation", **{task: acc})

        return results

    def _run_export(self, model):
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 4b: bitnet.cpp Export")
        logger.info("=" * 60)

        export_config = BitNetExportConfig()
        export_path = export_to_bitnet_cpp(model, self.export_output, export_config)
        logger.info(f"Model exported to: {export_path}")


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    pipeline = TernaryBoostPipeline(args)
    results = pipeline.run()

    logger.info("\nPipeline results:")
    for key, value in sorted(results.items()):
        logger.info(f"  {key}: {value}")

    return results


if __name__ == "__main__":
    main()
