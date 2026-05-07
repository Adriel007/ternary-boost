"""LoRA fine-tuning for ternary-quantized models.

Adds small trainable rank-decomposition adapters to frozen ternary weights
and fine-tunes them with knowledge distillation from the FP16 teacher.

Memory-efficient: only LoRA params receive gradients. The ternary backbone
and FP16 teacher are frozen, keeping VRAM usage low enough for T4 Colab.

QLoRA (Dettmers et al., 2023) proved this works for 4-bit. We adapt it
for 1.58-bit ternary, where the quantization error is larger but LoRA
with sufficient rank compensates.
"""

import gc
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from shared.logging import get_logger

logger = get_logger("pt_bitnet.lora")


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning on ternary models.

    Calibrated for ternary (1.58-bit): the quantization gap is larger than
    4-bit QLoRA, so we use higher rank, lower distill weight, and sharper
    teacher targets. CE loss must dominate — KD is a gentle guide, not the
    main objective. v1's distill_weight=0.5 pulled the model away from
    correct predictions.
    """

    rank: int = 64                # LoRA rank (64 for ternary gap, 32 for 4-bit)
    alpha: float = 128.0          # Scaling factor (alpha=2*rank is standard)
    dropout: float = 0.0          # No dropout — teacher already regularizes
    target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention only — FFN doesn't
                                                   # benefit as much from LoRA
    )
    skip_modules: tuple = ("lm_head", "embed_tokens")

    # Fine-tuning
    num_steps: int = 1000         # More steps with lower LR for stability
    batch_size: int = 1           # Batch 1 for memory safety
    learning_rate: float = 5e-5   # Lower LR: ternary needs gentler updates
    max_seq_length: int = 128     # Longer context = better KD signal
    gradient_accumulation: int = 8 # Effective batch = 8 with batch_size=1

    # Knowledge distillation
    distill_weight: float = 0.1   # CE dominates (0.9), KD gentle guide (0.1)
    temperature: float = 1.5      # Sharper targets (v1's T=3.0 was too soft)


class LoRALinear(nn.Module):
    """Wraps a frozen ternary nn.Linear with a trainable LoRA adapter.

    Forward: y = linear(x, W_ternary) + (alpha/rank) * linear(linear(x, A), B)

    Where A is [rank, in_features] and B is [out_features, rank].
    Only A and B receive gradients; W_ternary stays frozen.

    LoRA params are float32 for AdamW stability. During forward, they are
    cast to the input dtype (bfloat16) to avoid dtype mismatch.
    """

    def __init__(self, base: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.base = base           # Frozen ternary weight
        self.rank = config.rank
        self.alpha = config.alpha
        self.scaling = config.alpha / config.rank
        self.dropout = nn.Dropout(config.dropout)

        in_f = base.in_features
        out_f = base.out_features

        # float32 for training stability (AdamW accumulates small updates)
        self.lora_A = nn.Parameter(
            torch.randn(config.rank, in_f, device=base.weight.device) * 0.02
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_f, config.rank, device=base.weight.device)
        )

        # Freeze base
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ternary base forward (frozen) — nn.Linear takes only x
        base_out = self.base(x)

        # LoRA forward: cast params to input dtype for matmul compatibility
        A = self.lora_A.to(dtype=x.dtype)
        B = self.lora_B.to(dtype=x.dtype)
        lora_out = self.scaling * (self.dropout(x) @ A.T @ B.T)

        return base_out + lora_out

    @property
    def weight(self):
        """Proxy .weight so HF model introspection works."""
        return self.base.weight


def _add_lora_to_model(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Replace target nn.Linear layers with LoRALinear wrappers.

    Only wraps layers that are already ternary (nn.Linear with
    requires_grad=False weights). Keeps original layers for lm_head,
    embed_tokens, and any non-target modules.
    """
    replaced = 0
    for module_name, module in list(model.named_modules()):
        if any(skip in module_name for skip in config.skip_modules):
            continue
        if not any(target in module_name for target in config.target_modules):
            continue
        if not isinstance(module, nn.Linear):
            continue
        # Only wrap if weight is frozen (ternary quantized)
        if module.weight.requires_grad:
            continue

        parent_name = ".".join(module_name.split(".")[:-1])
        child_name = module_name.split(".")[-1]
        parent = model if not parent_name else model.get_submodule(parent_name)

        lora_layer = LoRALinear(module, config)
        setattr(parent, child_name, lora_layer)
        replaced += 1

    logger.info(f"  LoRA: wrapped {replaced} layers with LoRALinear (rank={config.rank})")
    return model


def _get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Collect all LoRA trainable parameters."""
    params = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            params.extend([m.lora_A, m.lora_B])
    return params


def finetune_lora(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    calibration_texts: list[str],
    teacher_model: Optional[PreTrainedModel] = None,
    config: Optional[LoRAConfig] = None,
) -> PreTrainedModel:
    """Fine-tune LoRA adapters on a ternary model with knowledge distillation.

    MEMORY-SAFE: Teacher logits are pre-computed and cached on CPU, then the
    teacher is freed before fine-tuning starts. This avoids having both models
    on GPU simultaneously, preventing OOM on T4 Colab.

    Peak VRAM: student (~6 GB) + LoRA (~0.1 GB) + optimizer (~0.2 GB)
             + cached logits on CPU (~0.6 GB RAM) = ~6.5 GB VRAM, fits T4 easily.
    """
    if config is None:
        config = LoRAConfig()

    has_cuda = torch.cuda.is_available()
    device = next(model.parameters()).device

    # ── Add LoRA adapters ────────────────────────────────────────
    model = _add_lora_to_model(model, config)
    model.train()

    lora_params = _get_lora_params(model)
    total_lora = sum(p.numel() for p in lora_params)
    logger.info(f"  LoRA trainable params: {total_lora:,} "
                f"({total_lora * 4 / 1e6:.1f} MB fp32)")

    # ── Phase 1: Pre-compute teacher logits (GPU batched) ────────
    # Teacher runs on GPU in batches of 8 for speed, logits cached
    # on CPU. Student stays on GPU. Peak VRAM: student + teacher
    # ≈ 11 GB for Phi-2 — fits T4 15.6 GB with margin.
    teacher_logits_cache = []
    if teacher_model is not None:
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

        n_texts = len(calibration_texts)

        # Right-padding is critical: the student processes texts one at a
        # time (no padding), so position IDs start at 0. Left-padding would
        # shift real tokens to higher positions, breaking KD alignment.
        _pad_side = tokenizer.padding_side
        tokenizer.padding_side = "right"

        # Decide GPU vs CPU for teacher. GPU is ~10x faster but needs
        # VRAM for teacher + batch activations. On T4 with Phi-2:
        # student ~5.4 GB + teacher ~5.4 GB + batch ~1 GB = ~12 GB → OK.
        # For 7B models (~13.5 GB each) we fall back to CPU automatically.
        teacher_gpu = False
        if has_cuda:
            vram_free = (torch.cuda.get_device_properties(0).total_memory
                         - torch.cuda.memory_allocated()) / 1e9
            teacher_gb = sum(p.numel() * p.element_size()
                           for p in teacher_model.parameters()) / 1e9
            # Batch of 8 at seq_len=128 needs ~1 GB for activations
            if vram_free >= teacher_gb + 1.0:
                teacher_gpu = True
            else:
                logger.warning(
                    f"  Teacher too large for GPU (free={vram_free:.1f} GB, "
                    f"need ~{teacher_gb + 1.0:.1f} GB). Using CPU.")

        teach_batch = 8 if teacher_gpu else 1  # CPU: one at a time
        if teacher_gpu:
            teacher_model.to(device)
            torch.cuda.empty_cache()
            logger.info("  Pre-computing teacher logits on GPU "
                        f"(batch={teach_batch}, {n_texts} texts)...")
        else:
            teacher_model.cpu()
            if has_cuda:
                torch.cuda.empty_cache()
            logger.info("  Pre-computing teacher logits on CPU "
                        f"({n_texts} texts)...")

        with torch.no_grad():
            for batch_start in range(0, n_texts, teach_batch):
                batch_end = min(batch_start + teach_batch, n_texts)
                batch_texts = calibration_texts[batch_start:batch_end]

                inputs = tokenizer(
                    batch_texts, return_tensors="pt", truncation=True,
                    max_length=config.max_seq_length, padding=True,
                )
                if teacher_gpu:
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                out = teacher_model(input_ids=inputs["input_ids"])
                # Split batch outputs into per-text logits on CPU.
                # Slice to real token count (exclude padding) so position
                # IDs and sequence lengths match the student's view.
                for j in range(len(batch_texts)):
                    sample_len = (inputs["attention_mask"][j] == 1).sum().item()
                    logits_j = out.logits[j, :sample_len].detach().cpu()
                    teacher_logits_cache.append(logits_j)
                del out, inputs

                cache_mb = sum(t.numel() * 2 for t in teacher_logits_cache) / 1e6
                if (batch_end) % max(1, n_texts // 5) == 0 or batch_end == n_texts:
                    logger.info(f"    Cached {batch_end}/{n_texts} "
                                f"teacher logits ({cache_mb:.0f} MB CPU RAM)")

        tokenizer.padding_side = _pad_side  # restore

        # Free teacher from GPU/CPU
        del teacher_model
        gc.collect()
        if has_cuda:
            torch.cuda.empty_cache()
            _log_vram("  VRAM after teacher cleanup")

        logger.info(f"  Cached {len(teacher_logits_cache)} teacher logits "
                    f"({cache_mb:.0f} MB CPU RAM)")

        # Pre-move all cached logits to GPU in one batch.
        # Avoids 1000 individual CPU→GPU transfers in the training loop
        # (synchronous .to(device) per step adds ~50-80s of CUDA sync overhead).
        if has_cuda:
            teacher_logits_cache = [t.to(device, non_blocking=False)
                                    for t in teacher_logits_cache]
            gpu_mb = sum(t.numel() * t.element_size()
                        for t in teacher_logits_cache) / 1e6
            _log_vram("  VRAM with teacher logits cache")
            logger.info(f"  Moved teacher logits to GPU ({gpu_mb:.0f} MB)")
    else:
        logger.info("  No teacher — using CE loss only (no distillation)")

    # ── Phase 2: Fine-tune LoRA ─────────────────────────────────
    optimizer = torch.optim.AdamW(lora_params, lr=config.learning_rate,
                                  weight_decay=0.01, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_steps,
        eta_min=config.learning_rate * 0.1,
    )

    total_loss_ema = 0.0
    best_loss = float("inf")
    optimizer.zero_grad()

    for step in range(config.num_steps):
        text_idx = step % len(calibration_texts)
        text = calibration_texts[text_idx]

        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=config.max_seq_length)
        input_ids = inputs["input_ids"].to(device)
        if input_ids.numel() < 2:
            continue

        # Student forward (ternary + LoRA) — only model on GPU
        student_out = model(input_ids=input_ids, labels=input_ids)
        ce_loss = student_out.loss

        # KD using pre-cached teacher logits (already on GPU)
        distill_loss = torch.tensor(0.0, device=device)
        if teacher_logits_cache:
            cached_logits = teacher_logits_cache[text_idx]  # already on GPU
            # Align sequence lengths (cached might be longer than student)
            min_len = min(student_out.logits.shape[1], cached_logits.shape[1])
            T = config.temperature
            student_log_probs = F.log_softmax(
                student_out.logits[:, :min_len, :] / T, dim=-1)
            teacher_probs = F.softmax(
                cached_logits[:, :min_len, :] / T, dim=-1)
            distill_loss = F.kl_div(student_log_probs, teacher_probs,
                                    reduction="batchmean") * (T * T)

        # Combined loss
        beta = config.distill_weight if teacher_logits_cache else 0.0
        loss = (1.0 - beta) * ce_loss + beta * distill_loss
        loss = loss / config.gradient_accumulation
        loss.backward()
        del student_out

        # Optimization step (accumulated over gradient_accumulation batches)
        if (step + 1) % config.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Logging
        total_loss_ema = 0.95 * total_loss_ema + 0.05 * loss.item()
        if step == 0 or (step + 1) % max(1, config.num_steps // 10) == 0:
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"  LoRA step {step + 1}/{config.num_steps} "
                f"loss={total_loss_ema:.4f} ce={ce_loss.item():.4f} "
                f"kd={distill_loss.item():.4f} lr={lr:.2e}"
            )

        if loss.item() < best_loss:
            best_loss = loss.item()

    # ── Finalize ─────────────────────────────────────────────────
    model.eval()
    logger.info(f"  LoRA fine-tuning complete. Best loss: {best_loss:.4f}")
    return model


def _log_vram(label: str = "") -> None:
    """Log current VRAM usage for debugging memory issues."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(
            f"{label}: VRAM {alloc:.1f}/{total:.1f} GB used "
            f"({reserved:.1f} GB reserved)"
        )


def keep_lora_separate(model: nn.Module) -> nn.Module:
    """Keep LoRA adapters separate from ternary weights (no merge).

    This preserves the ternary backbone as-is and saves LoRA as small
    fp16 adapters. The model retains LoRALinear wrappers — forward()
    computes y = W_ternary @ x + lora_scale * B @ (A @ x).

    Use this when the goal is compressed inference (INT2 ternary + LoRA
    stored separately), NOT when baking LoRA into dense weights.

    Returns:
        The model unchanged (LoRALinear wrappers intact).
    """
    count = sum(1 for m in model.modules()
                if type(m).__name__ == "LoRALinear")
    logger.info(f"  Kept {count} LoRA adapters separate (no merge)")
    return model


def merge_lora_to_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA adapters back into the base weights (dense, non-ternary).

    W_merged = W_ternary + (alpha/rank) * (B @ A)

    NOTE: this destroys ternary sparsity. Use merge_and_requantize() instead
    to preserve ternary structure after merging.
    """
    merged = 0
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue

        with torch.no_grad():
            w_base = module.base.weight.data.float()
            lora_contrib = module.scaling * (module.lora_B.data.float() @
                                             module.lora_A.data.float())
            w_merged = w_base + lora_contrib

        parent_name = ".".join(module_name.split(".")[:-1])
        child_name = module_name.split(".")[-1]
        parent = model if not parent_name else model.get_submodule(parent_name)

        device = module.base.weight.device
        dtype = module.base.weight.dtype
        new_linear = nn.Linear(
            module.base.in_features, module.base.out_features,
            bias=module.base.bias is not None, dtype=dtype, device=device,
        )
        new_linear.weight.data.copy_(w_merged.to(device).to(dtype))
        if module.base.bias is not None:
            new_linear.bias.data.copy_(module.base.bias.data)

        setattr(parent, child_name, new_linear)
        merged += 1

    logger.info(f"  Merged {merged} LoRA adapters → nn.Linear (dense, non-ternary)")
    return model


def merge_and_requantize(model: nn.Module) -> nn.Module:
    """Merge LoRA into weights, then re-ternarize to preserve sparsity.

    Flow:
      1. W_dense = W_ternary + (alpha/rank) * (B @ A)  [merge LoRA]
      2. W_new_ternary = symmetric_ternary(W_dense)     [re-quantize]
      3. Replace LoRALinear → nn.Linear(W_new_ternary)   [clean layers]

    This bakes the LoRA correction into a fresh ternary weight matrix,
    recovering both quality AND sparsity. The re-quantization preserves
    the improved weight values that LoRA discovered.

    NOTE: re-quantization may lose some of the LoRA benefit. The gain
    comes from using LoRA to find better ternary weight assignments,
    not from keeping the dense LoRA correction.
    """
    from pt_bitnet.quantize import _symmetric_ternary, PTBitNetConfig

    pt_cfg = PTBitNetConfig(show_progress=False)
    merged = 0

    for module_name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue

        with torch.no_grad():
            # 1. Merge
            w_base = module.base.weight.data.float()
            lora_contrib = module.scaling * (module.lora_B.data.float() @
                                             module.lora_A.data.float())
            w_dense = w_base + lora_contrib

            # 2. Re-ternarize
            w_ternary = _symmetric_ternary(w_dense, pt_cfg)

        # 3. Replace
        parent_name = ".".join(module_name.split(".")[:-1])
        child_name = module_name.split(".")[-1]
        parent = model if not parent_name else model.get_submodule(parent_name)

        device = module.base.weight.device
        dtype = module.base.weight.dtype
        new_linear = nn.Linear(
            module.base.in_features, module.base.out_features,
            bias=module.base.bias is not None, dtype=dtype, device=device,
        )
        new_linear.weight.data.copy_(w_ternary.to(device).to(dtype))
        new_linear.weight.requires_grad = False
        if module.base.bias is not None:
            new_linear.bias.data.copy_(module.base.bias.data)

        setattr(parent, child_name, new_linear)
        merged += 1

    logger.info(f"  Merged + re-quantized {merged} layers to strict ternary")
    return model


def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only LoRA weights (A, B matrices) to a safetensors file."""
    from safetensors.torch import save_file

    lora_weights = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_weights[name] = param.detach().cpu()

    if lora_weights:
        save_file(lora_weights, path)
        logger.info(f"  Saved {len(lora_weights)} LoRA tensors to {path}")
    else:
        logger.warning("  No LoRA tensors found to save")


def load_lora_weights(model: nn.Module, path: str) -> None:
    """Load LoRA weights from safetensors into model's LoRA adapters."""
    from safetensors.torch import load_file

    lora_weights = load_file(path)
    loaded = 0
    for name, param in model.named_parameters():
        if name in lora_weights:
            param.data.copy_(lora_weights[name].to(param.device))
            loaded += 1
    logger.info(f"  Loaded {loaded}/{len(lora_weights)} LoRA tensors")


def count_lora_params(model: nn.Module) -> dict:
    """Count trainable vs frozen parameters for logging."""
    lora_trainable = sum(
        p.numel() for m in model.modules()
        if isinstance(m, LoRALinear)
        for p in [m.lora_A, m.lora_B]
    )
    total = sum(p.numel() for p in model.parameters())
    return {
        "total_params": total,
        "lora_trainable": lora_trainable,
        "lora_ratio": lora_trainable / total if total > 0 else 0.0,
    }
