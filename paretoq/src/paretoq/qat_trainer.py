"""ParetoQ QAT Trainer with ZeroQAT integration.

Replaces standard nn.Linear layers with QuantizeLinear, initializes
weight clip values heuristically, and runs QAT using the ZeroQAT
zero-order optimizer for memory efficiency.
"""

import copy
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import default_data_collator
from transformers import get_linear_schedule_with_warmup

from shared.checkpoint import save_checkpoint
from shared.logging import get_logger, log_memory_usage

from .utils_quant import QuantizeLinear
from .zo_optim import ZeroOrderOptimizer, ZeroQATConfig

logger = get_logger("paretoq_qat")


def _replace_with_quantized(
    model: nn.Module,
    w_bits: int,
    target_modules: tuple[str, ...],
    skip_modules: tuple[str, ...],
) -> nn.Module:
    """Replace target nn.Linear layers with QuantizeLinear."""
    replacements = 0
    for module_name, module in model.named_modules():
        if any(skip in module_name for skip in skip_modules):
            continue
        if not any(target in module_name for target in target_modules):
            continue
        if not isinstance(module, nn.Linear):
            continue

        parent_name = ".".join(module_name.split(".")[:-1])
        child_name = module_name.split(".")[-1]
        parent = model if not parent_name else model.get_submodule(parent_name)

        quant_linear = QuantizeLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            w_bits=w_bits,
        )
        quant_linear.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            quant_linear.bias.data.copy_(module.bias.data)

        setattr(parent, child_name, quant_linear)
        replacements += 1

    logger.info(f"Replaced {replacements} nn.Linear layers with QuantizeLinear")
    return model


def _init_weight_clip_values(model: nn.Module, w_bits: int) -> None:
    """Initialize weight_clip_val parameters based on weight statistics."""
    for name, param in model.named_parameters():
        if "weight_clip_val" not in name:
            continue
        weight_name = name.replace("weight_clip_val", "weight")
        weight_param = None
        for wname, wparam in model.named_parameters():
            if wname == weight_name:
                weight_param = wparam
                break
        if weight_param is None:
            continue

        with torch.no_grad():
            if w_bits == 1 or w_bits == 0:
                scale = torch.mean(weight_param.abs(), dim=-1, keepdim=True).detach()
            elif w_bits == 2:
                scale, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
            elif w_bits in (3, 4):
                xmax, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                maxq = 2 ** (w_bits - 1) - 1
                scale = xmax / maxq
            else:
                raise NotImplementedError(f"w_bits={w_bits}")
            param.data.copy_(scale)


def apply_paretoq_qat(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataloader: DataLoader,
    output_path: str,
    w_bits: int = 1,
    qat_steps: int = 500,
    zo_config: Optional[ZeroQATConfig] = None,
    val_dataloader: Optional[DataLoader] = None,
    log_interval: int = 50,
) -> PreTrainedModel:
    """Apply ParetoQ QAT with ZeroQAT zero-order optimization.

    Args:
        model: Pre-trained model (already PT-BitNet quantized recommended).
        tokenizer: HuggingFace tokenizer for padding/config.
        train_dataloader: Training data loader.
        output_path: Where to save the QAT-fine-tuned model.
        w_bits: Quantization bit-width (1 for ternary).
        qat_steps: Number of QAT training steps.
        zo_config: ZeroQAT configuration. Uses defaults if None.
        val_dataloader: Optional validation loader for perplexity tracking.
        log_interval: Log metrics every N steps.

    Returns:
        QAT-fine-tuned model.
    """
    if zo_config is None:
        zo_config = ZeroQATConfig(max_steps=qat_steps)

    logger.info(f"Starting ParetoQ QAT (w_bits={w_bits}, steps={qat_steps})")
    log_memory_usage(logger, "Before QAT: ")

    target_modules = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    skip_modules = ("lm_head", "embed_tokens")

    model = _replace_with_quantized(
        model, w_bits, target_modules, skip_modules
    )
    _init_weight_clip_values(model, w_bits)

    trainable_params = []
    for name, param in model.named_parameters():
        if "weight_clip_val" in name:
            param.requires_grad = True
            trainable_params.append((name, param))
    logger.info(f"Trainable parameters: {len(trainable_params)} (weight_clip_val only)")

    zo_optimizer = ZeroOrderOptimizer(model, zo_config, named_params=trainable_params)

    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"

    model.train()
    if has_cuda:
        model.cuda()

    train_iter = iter(train_dataloader)

    def qat_loss_fn(outputs, labels):
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    best_loss = float("inf")
    accum_loss = 0.0

    for step in range(qat_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        grads = zo_optimizer.estimate_gradient_layerwise(batch, qat_loss_fn)
        zo_optimizer.step(grads)

        with torch.no_grad():
            loss = zo_optimizer._forward_and_loss(batch, qat_loss_fn)
            accum_loss += loss.item()

        if (step + 1) % log_interval == 0:
            avg_loss = accum_loss / log_interval
            logger.info(
                f"QAT step {step + 1}/{qat_steps} - loss: {avg_loss:.4f}"
            )
            accum_loss = 0.0

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(
                    model, output_path, tokenizer,
                    metadata={"step": step + 1, "loss": avg_loss},
                )

    save_checkpoint(
        model, output_path, tokenizer,
        metadata={"step": qat_steps, "loss": best_loss, "stage": "paretoq_qat"},
    )

    log_memory_usage(logger, "After QAT: ")
    logger.info(f"ParetoQ QAT complete. Model saved to {output_path}")
    return model
