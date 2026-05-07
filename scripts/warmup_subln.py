#!/usr/bin/env python3
"""C4 warm-up script for SubLN-inserted models (BitDistill step 2).

After inserting SubLN before MHSA/FFN output projections, a short continued
pre-training phase on C4 learns the SubLN scales and stabilizes activations
for downstream ternarization.

Target: 50M tokens (~1-2 hours on T4 for Phi-2).
Reference: Wu et al., "BitNet Distillation," arXiv 2510.13998, Oct 2025.

Usage:
    python scripts/warmup_subln.py \
        --model microsoft/phi-2 \
        --tokens 50000000 \
        --batch-size 2 \
        --output ./output/phi2_subln
"""

import argparse
import gc
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.logging import get_logger
from shared.checkpoint import save_checkpoint, load_checkpoint
from pt_bitnet.subln import insert_subln, remove_subln, count_subln

logger = get_logger("warmup_subln")


def parse_args():
    p = argparse.ArgumentParser(description="C4 warm-up for SubLN-inserted models")
    p.add_argument("--model", default="microsoft/phi-2", help="HF model ID or path")
    p.add_argument("--tokens", type=int, default=50_000_000,
                   help="Total tokens to train on (default: 50M)")
    p.add_argument("--batch-size", type=int, default=2, help="Micro-batch size")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation steps")
    p.add_argument("--seq-length", type=int, default=2048, help="Sequence length")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--warmup-steps", type=int, default=100, help="LR warmup steps")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Max steps (overrides --tokens if set)")
    p.add_argument("--output", default="./output/warmup_subln",
                   help="Output directory for checkpoint")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    p.add_argument("--checkpoint-steps", type=int, default=500,
                   help="Save checkpoint every N steps")
    p.add_argument("--log-steps", type=int, default=50, help="Log every N steps")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--grad-checkpoint", action="store_true", default=True,
                   help="Use gradient checkpointing (saves VRAM)")
    p.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false",
                   help="Disable gradient checkpointing")
    p.add_argument("--c4-split", default="train", help="C4 split to use")
    p.add_argument("--c4-streaming", action="store_true", default=True,
                   help="Stream C4 (no disk cache)")
    return p.parse_args()


def load_c4_streaming(split: str = "train", seq_length: int = 2048):
    """Stream C4-en with minimal RAM overhead."""
    try:
        ds = load_dataset(
            "allenai/c4", "en", split=split, streaming=True,
            trust_remote_code=False,
        )
    except Exception:
        logger.warning("allenai/c4 not accessible — trying HuggingFace fallback")
        ds = load_dataset(
            "c4", "en", split=split, streaming=True,
            trust_remote_code=False,
        )

    def _tokenize_and_chunk(examples):
        """Tokenize and chunk into fixed-length blocks, discarding partials."""
        # Tokenizer is injected via closure
        tokens = _tokenize_and_chunk.tokenizer(
            examples["text"], truncation=False, add_special_tokens=True,
        )["input_ids"]
        chunks = []
        for seq in tokens:
            for i in range(0, len(seq) - seq_length, seq_length):
                chunks.append({"input_ids": seq[i:i + seq_length]})
        return {"input_ids": [c["input_ids"] for c in chunks]} if chunks else {"input_ids": []}

    return ds


def warmup_subln(args):
    """Main warm-up routine."""

    # ── Device setup ──────────────────────────────────────────────
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    if has_cuda:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB)")
    else:
        logger.info("No GPU detected — running on CPU (slow)")

    # ── Load model ────────────────────────────────────────────────
    logger.info(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = getattr(config, "eos_token_id", 0) or 0
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, torch_dtype=dtype, trust_remote_code=False,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model loaded: {total_params:.0f}M params")

    # ── Insert SubLN ──────────────────────────────────────────────
    model = insert_subln(model)
    n_subln = count_subln(model)
    logger.info(f"Inserted {n_subln} SubLN modules")

    subln_params = sum(
        p.numel() for m in model.modules()
        if type(m).__name__ == "SubLN" for p in m.parameters()
    )
    logger.info(f"  SubLN params: {subln_params:,} ({subln_params * 4:.0f} B)")

    # ── Optimizer / scheduler ─────────────────────────────────────
    # Only train SubLN weights + biases; backbone stays frozen
    trainable = []
    for name, param in model.named_parameters():
        if "SubLN" in name or "subln" in name.lower():
            param.requires_grad = True
            trainable.append(param)
        else:
            param.requires_grad = False

    trainable_count = sum(p.numel() for p in trainable)
    logger.info(f"Trainable (SubLN only): {trainable_count:,} params")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    effective_steps = args.max_steps or (args.tokens // (args.batch_size * args.grad_accum * args.seq_length))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=effective_steps,
    )

    # ── Gradient checkpointing ────────────────────────────────────
    if args.grad_checkpoint and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # ── Resume ────────────────────────────────────────────────────
    start_step = 0
    total_tokens_seen = 0
    if args.resume:
        checkpoint = load_checkpoint(args.output)
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_step = checkpoint.get("step", 0)
            total_tokens_seen = checkpoint.get("tokens_seen", 0)
            logger.info(f"Resumed from step {start_step} ({total_tokens_seen:,} tokens)")

    # ── C4 data stream ───────────────────────────────────────────
    logger.info(f"Setting up C4 streaming ({args.c4_split})...")
    ds = load_c4_streaming(args.c4_split, args.seq_length)

    # Attach tokenizer to the helper so _tokenize_and_chunk can access it
    def _tk_chunk(examples):
        tokens = tokenizer(
            examples["text"], truncation=False, add_special_tokens=True,
        )["input_ids"]
        out_ids = []
        for seq in tokens:
            for i in range(0, len(seq) - args.seq_length, args.seq_length):
                out_ids.append(seq[i:i + args.seq_length])
        return {"input_ids": out_ids, "labels": [ids[:] for ids in out_ids]}

    # Create a simple iterable that yields batches
    def _data_iterator():
        buffer = []
        for example in ds:
            tokens = tokenizer(example["text"], truncation=False, add_special_tokens=True)["input_ids"]
            for i in range(0, len(tokens) - args.seq_length, args.seq_length):
                chunk = tokens[i:i + args.seq_length]
                buffer.append(chunk)
                if len(buffer) >= args.batch_size:
                    # Pad to max length in batch
                    max_len = max(len(c) for c in buffer)
                    input_ids = torch.zeros((len(buffer), max_len), dtype=torch.long)
                    labels = torch.full((len(buffer), max_len), -100, dtype=torch.long)
                    for j, c in enumerate(buffer):
                        input_ids[j, :len(c)] = torch.tensor(c, dtype=torch.long)
                        labels[j, :len(c)] = torch.tensor(c, dtype=torch.long)
                    yield input_ids, labels
                    buffer = []

    data_iter = _data_iterator()

    # ── Training loop ─────────────────────────────────────────────
    model.train()
    step = start_step
    total_tokens_seen_local = total_tokens_seen
    tokens_per_step = args.batch_size * args.grad_accum * args.seq_length

    logger.info(f"Starting warm-up: {effective_steps} steps, "
                f"{tokens_per_step:,} tokens/step, "
                f"target {args.tokens:,} tokens")
    t_start = time.time()

    optimizer.zero_grad()
    loss_ema = None

    while step < effective_steps:
        step_loss = 0.0

        for micro_step in range(args.grad_accum):
            try:
                input_ids, labels = next(data_iter)
            except StopIteration:
                logger.info("C4 stream exhausted — restarting")
                data_iter = _data_iterator()
                input_ids, labels = next(data_iter)

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            step_loss += loss.item()
            total_tokens_seen_local += input_ids.numel()

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        step += 1
        loss_ema = step_loss if loss_ema is None else 0.95 * loss_ema + 0.05 * step_loss

        # ── Logging ───────────────────────────────────────────────
        if step % args.log_steps == 0:
            elapsed = time.time() - t_start
            tok_per_sec = total_tokens_seen_local / max(elapsed, 1)
            lr = scheduler.get_last_lr()[0]
            ppl = math.exp(loss_ema) if loss_ema < 20 else float("inf")
            logger.info(
                f"  Step {step}/{effective_steps} | "
                f"loss={loss_ema:.4f} ppl={ppl:.1f} | "
                f"tokens={total_tokens_seen_local:,} | "
                f"{tok_per_sec:.0f} tok/s | lr={lr:.2e}"
            )

        # ── Checkpoint ────────────────────────────────────────────
        if step % args.checkpoint_steps == 0:
            save_checkpoint(
                args.output,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                step=step,
                tokens_seen=total_tokens_seen_local,
            )
            logger.info(f"Checkpoint saved at step {step} ({total_tokens_seen_local:,} tokens)")

    # ── Final save ────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    save_checkpoint(
        args.output,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        step=step,
        tokens_seen=total_tokens_seen_local,
    )

    elapsed_m = (time.time() - t_start) / 60
    logger.info(f"Warm-up complete: {step} steps, "
                f"{total_tokens_seen_local:,} tokens in {elapsed_m:.1f} min")

    # Cleanup before return (free VRAM for downstream pipeline)
    del optimizer
    gc.collect()
    if has_cuda:
        torch.cuda.empty_cache()

    return model, tokenizer


if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = warmup_subln(args)
    logger.info(f"SubLN-warmed model saved to {args.output}")
