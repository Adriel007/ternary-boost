"""Block-wise Straight-Through Estimator fine-tuning for ternary models.

Processes the model one transformer block at a time, adjusting ternary
weights via STE gradients to match FP16 teacher hidden states. Only one
block is active on GPU at a time — the rest stay frozen on CPU.

This is the ternary adaptation of GPTQ's block-wise reconstruction:
instead of second-order Hessian compensation, we use first-order STE
gradients which are cheaper and work for ternary's discrete structure.
"""

import gc
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from shared.logging import get_logger

logger = get_logger("pt_bitnet.ste")


@dataclass
class BlockSTEConfig:
    """Configuration for block-wise STE fine-tuning."""

    num_steps: int = 200                # Steps per block
    batch_size: int = 2
    learning_rate: float = 1e-5         # Lower than LoRA — adjusting ternary, not learning from scratch
    max_seq_length: int = 128
    gradient_accumulation: int = 4

    # Loss weights
    hidden_loss_weight: float = 1.0    # MSE on hidden states
    logit_loss_weight: float = 0.1     # CE on logits (lighter — hidden states are primary target)

    # STE: gradient only flows to weights above deadzone threshold
    # (reduces noise from zero-region weights)
    deadzone_grad_scale: float = 0.1   # Scale gradients for deadzone weights (0 = no gradient)
    re_quantize_after: bool = True      # Re-quantize weights after each block

    # Blocks to process (all transformer layers by default)
    skip_modules: tuple = ("lm_head", "embed_tokens")


class TernarySTEForward(torch.autograd.Function):
    """Straight-Through Estimator for ternary quantization.

    Forward: quantizes to {-alpha, 0, +alpha} per row.
    Backward: passes gradient through, optionally scaled for deadzone weights.

    The key insight: deadzone weights (those quantized to 0) have less
    reliable gradient direction. Scaling down their gradients focuses
    the optimization on weights that are already contributing.
    """

    @staticmethod
    def forward(ctx, w_float, alpha, deadzone_grad_scale=0.1):
        """Quantize to ternary {-alpha, 0, +alpha} per row."""
        out_f = w_float.shape[0]
        w_q = torch.zeros_like(w_float)

        # Per-row thresholds: delta = alpha/2
        delta = alpha / 2.0

        mask_pos = w_float > delta
        mask_neg = w_float < -delta
        mask_zero = ~(mask_pos | mask_neg)

        w_q[mask_pos] = alpha.expand_as(w_float)[mask_pos]
        w_q[mask_neg] = -alpha.expand_as(w_float)[mask_neg]
        # Deadzone stays at 0

        ctx.save_for_backward(mask_pos, mask_neg, mask_zero)
        ctx.deadzone_scale = deadzone_grad_scale
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        """STE: pass gradient through, scaled for deadzone."""
        mask_pos, mask_neg, mask_zero = ctx.saved_tensors
        deadzone_scale = ctx.deadzone_scale

        # Gradient flows to all weights, but deadzone weights get scaled down
        grad_input = grad_output.clone()
        grad_input[mask_zero] *= deadzone_scale

        return grad_input, None, None


class TernarySTELinear(nn.Module):
    """Linear layer with STE ternary quantization during forward.

    Wraps an nn.Linear. During forward, the weight is quantized to ternary
    via STE, allowing gradients to flow to the underlying float weights.
    Alpha is computed adaptively per forward pass.
    """

    def __init__(self, base: nn.Linear, config: BlockSTEConfig):
        super().__init__()
        self.base = base
        self.config = config
        self.in_features = base.in_features
        self.out_features = base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.base.weight

        # Compute per-row alpha from current weight values
        # alpha = mean of active weights (weights that cross threshold)
        w_abs = w.abs()
        row_mean = w_abs.mean(dim=-1, keepdim=True)
        delta = row_mean / 2.0
        active = w_abs >= delta
        active_count = active.sum(dim=-1, keepdim=True).clamp_min(1)
        alpha = (w_abs * active.float()).sum(dim=-1, keepdim=True) / active_count

        # Ternary STE forward
        w_q = TernarySTEForward.apply(w, alpha, self.config.deadzone_grad_scale)
        out = F.linear(x, w_q, self.base.bias)
        return out

    @property
    def weight(self):
        """Proxy for HF introspection."""
        return self.base.weight


def _wrap_block_with_ste(
    model: nn.Module, block_idx: int, config: BlockSTEConfig,
) -> list[str]:
    """Replace nn.Linear layers in a specific transformer block with STELinear.

    Returns the list of module names that were wrapped (for later unwrapping).
    """
    wrapped = []
    for module_name, module in list(model.named_modules()):
        # Only process layers in this block
        if f"model.layers.{block_idx}." not in module_name:
            continue
        if any(skip in module_name for skip in config.skip_modules):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if not module.weight.requires_grad:
            # Frozen ternary weight — enable grad for STE
            module.weight.requires_grad = True

        parent_name = ".".join(module_name.split(".")[:-1])
        child_name = module_name.split(".")[-1]
        parent = model if not parent_name else model.get_submodule(parent_name)

        ste_layer = TernarySTELinear(module, config)
        setattr(parent, child_name, ste_layer)
        wrapped.append(module_name)

    return wrapped


def _unwrap_ste_block(model: nn.Module, block_idx: int,
                      wrapped_names: list[str]) -> None:
    """Replace STELinear layers back to standard nn.Linear (frozen)."""
    for module_name in wrapped_names:
        parts = module_name.split(".")[-1]
        parent_name = ".".join(module_name.split(".")[:-1])
        child_name = parts
        parent = model if not parent_name else model.get_submodule(parent_name)
        module = parent.get_submodule(child_name)

        if isinstance(module, TernarySTELinear):
            new_linear = nn.Linear(
                module.in_features, module.out_features,
                bias=module.base.bias is not None,
                dtype=module.base.weight.dtype, device=module.base.weight.device,
            )
            new_linear.weight.data.copy_(module.base.weight.data)
            if module.base.bias is not None:
                new_linear.bias.data.copy_(module.base.bias.data)
            new_linear.weight.requires_grad = False
            setattr(parent, child_name, new_linear)


def _re_quantize_block(model: nn.Module, block_idx: int) -> None:
    """Re-apply symmetric ternary quantization to a block's weights.

    After STE fine-tuning, weights may have drifted from strict ternary.
    Re-quantizing snaps them back to {-alpha, 0, +alpha}.
    """
    from pt_bitnet.quantize import _symmetric_ternary, PTBitNetConfig

    config = PTBitNetConfig(show_progress=False)
    for module_name, module in model.named_modules():
        if f"model.layers.{block_idx}." not in module_name:
            continue
        if not isinstance(module, nn.Linear):
            continue
        with torch.no_grad():
            w = module.weight.data.float()
            ternary_w = _symmetric_ternary(w, config)
            module.weight.data.copy_(ternary_w.to(module.weight.dtype))
            module.weight.requires_grad = False


@torch.no_grad()
def _collect_teacher_hidden_states(
    teacher: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    calibration_texts: list[str],
    num_layers: int,
    device: torch.device,
    max_seq_length: int = 128,
) -> dict[int, list[torch.Tensor]]:
    """Run teacher on calibration data, collect hidden states after each block.

    Returns: {block_idx: [hidden_after_block_0_text_0, ...], ...}
    """
    teacher.eval()
    hidden_states = {i: [] for i in range(num_layers)}

    # Register hooks to capture output of each transformer layer
    hooks = []
    captured = {}

    def _make_hook(idx):
        def _hook(module, input, output):
            # output is a tuple (hidden_states, ...) from transformer layer
            if isinstance(output, tuple):
                captured[idx] = output[0].detach().cpu()
            else:
                captured[idx] = output.detach().cpu()
        return _hook

    for i in range(num_layers):
        layer = teacher.get_submodule(f"model.layers.{i}")
        hooks.append(layer.register_forward_hook(_make_hook(i)))

    # Also capture embedding output (input to block 0)
    embedding_output = []

    def _embed_hook(module, input, output):
        embedding_output.append(output.detach().cpu())

    embed = teacher.get_input_embeddings()
    embed_handle = embed.register_forward_hook(_embed_hook)

    for text in calibration_texts[:16]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=max_seq_length)
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            teacher(input_ids=input_ids)
        for i in range(num_layers):
            if i in captured:
                hidden_states[i].append(captured[i])

    for h in hooks:
        h.remove()
    embed_handle.remove()

    return hidden_states, embedding_output


def finetune_blocks_ste(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    calibration_texts: list[str],
    teacher_model: PreTrainedModel,
    config: Optional[BlockSTEConfig] = None,
) -> PreTrainedModel:
    """Block-wise STE fine-tuning: optimize one transformer block at a time.

    For each block:
      1. Wrap block layers with STELinear (forward ternary, backward STE)
      2. Fine-tune to match teacher hidden states at block output
      3. Re-quantize to strict ternary
      4. Freeze block, move to next

    Only ONE block has requires_grad=True at any time. The teacher is
    kept on CPU and moved to GPU only for hidden state collection.
    """
    if config is None:
        config = BlockSTEConfig()

    has_cuda = torch.cuda.is_available()
    device = next(model.parameters()).device

    # Count transformer layers
    num_layers = 0
    for name, _ in model.named_modules():
        if name.startswith("model.layers."):
            n = int(name.split(".")[2])
            num_layers = max(num_layers, n + 1)
    logger.info(f"Block-wise STE: {num_layers} transformer layers")

    # ── Collect teacher hidden states (one pass, CPU storage) ──────
    logger.info("  Collecting teacher hidden states...")
    teacher_device = next(teacher_model.parameters()).device
    # Move teacher to same device as model for hidden collection
    if has_cuda and teacher_device.type != "cuda":
        teacher_model.cuda()
    teacher_hidden, embedding_output = _collect_teacher_hidden_states(
        teacher_model, tokenizer, calibration_texts, num_layers,
        device if has_cuda else torch.device("cpu"),
        config.max_seq_length,
    )
    # Free teacher VRAM
    teacher_model.cpu()
    gc.collect()
    if has_cuda:
        torch.cuda.empty_cache()
    logger.info(f"  Collected hidden states for {len(teacher_hidden)} layers")

    # ── Process blocks sequentially ────────────────────────────────
    model.eval()
    for block_idx in range(num_layers):
        t0 = time.time()
        if not teacher_hidden.get(block_idx):
            logger.info(f"  Block {block_idx}: no teacher data, skipping")
            continue

        logger.info(f"  Block {block_idx}/{num_layers}: STE fine-tuning...")

        # Wrap block with STE
        wrapped = _wrap_block_with_ste(model, block_idx, config)
        trainable = [p for n, p in model.named_parameters()
                     if p.requires_grad and f"model.layers.{block_idx}." in n]
        logger.info(f"    Wrapped {len(wrapped)} layers, {sum(p.numel() for p in trainable):,} trainable params")

        if not trainable:
            logger.info(f"    No trainable params in block {block_idx}, skipping")
            continue

        optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate,
                                      weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_steps,
            eta_min=config.learning_rate * 0.1,
        )

        # Get teacher targets for this block (hidden state AFTER this block)
        targets = teacher_hidden[block_idx]
        # Input to this block
        if block_idx == 0:
            inputs_list = embedding_output
        else:
            inputs_list = teacher_hidden.get(block_idx - 1, targets)

        total_loss_ema = 0.0
        optimizer.zero_grad()

        for step in range(config.num_steps):
            # Get teacher input/output for this step
            target = targets[step % len(targets)].to(device)
            input_hidden = inputs_list[step % len(inputs_list)].to(device)

            # ── Forward through student block ──────────────────
            # Use the standard HuggingFace transformer layer forward,
            # which handles attention + FFN + residuals correctly.
            layer = model.get_submodule(f"model.layers.{block_idx}")

            # Standard HF layer forward: (hidden_states, attention_mask, ...) → outputs
            # Returns tuple (hidden_states, ...) or just hidden_states
            layer_out = layer(input_hidden)
            if isinstance(layer_out, tuple):
                hidden_states = layer_out[0]
            else:
                hidden_states = layer_out

            # ── Loss: MSE on hidden states ────────────────────
            hidden_loss = F.mse_loss(hidden_states, target)

            # Optional: logit loss (cheap, only on final output)
            logit_loss = torch.tensor(0.0, device=device)
            if config.logit_loss_weight > 0 and block_idx == num_layers - 1:
                # Use lm_head on last block output
                if hasattr(model, "lm_head"):
                    logits = model.lm_head(hidden_states)
                    # Simple: minimize ||logits|| (regularization)
                    logit_loss = logits.abs().mean()

            loss = (config.hidden_loss_weight * hidden_loss +
                    config.logit_loss_weight * logit_loss)
            loss = loss / config.gradient_accumulation
            loss.backward()

            # ── Optimization step ─────────────────────────────
            if (step + 1) % config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss_ema = 0.95 * total_loss_ema + 0.05 * hidden_loss.item()

            if step == 0 or (step + 1) % max(1, config.num_steps // 5) == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"    step {step + 1}/{config.num_steps} "
                    f"hidden_loss={total_loss_ema:.6f} lr={lr:.2e}"
                )

        # ── Re-quantize to strict ternary ─────────────────────────
        if config.re_quantize_after:
            _unwrap_ste_block(model, block_idx, wrapped)
            _re_quantize_block(model, block_idx)
            logger.info(f"    Re-quantized block {block_idx} to strict ternary")
        else:
            _unwrap_ste_block(model, block_idx, wrapped)

        # Free memory
        del optimizer, scheduler, trainable, wrapped
        for p in model.parameters():
            p.requires_grad = False
        gc.collect()
        if has_cuda:
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        logger.info(f"  Block {block_idx} done in {elapsed:.1f}s")

    model.eval()
    logger.info("Block-wise STE fine-tuning complete")
    return model
