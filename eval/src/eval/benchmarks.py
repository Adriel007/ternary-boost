"""Evaluation harness for ternary models.

Supports MMLU, HellaSwag, ARC (Easy + Challenge), and WikiText perplexity
using the lm-evaluation-harness library with fallback to manual evaluation.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from shared.logging import get_logger, MetricsTracker

logger = get_logger("eval")


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    tasks: tuple[str, ...] = ("mmlu", "hellaswag", "arc_easy", "arc_challenge")
    batch_size: int = 16
    num_fewshot: int = 0
    mmlu_fewshot: int = 5
    max_samples: Optional[int] = None
    use_lm_eval: bool = True


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: Optional[EvalConfig] = None,
    metrics_tracker: Optional[MetricsTracker] = None,
) -> dict:
    """Evaluate model on standard benchmarks.

    Args:
        model: Quantized model to evaluate.
        tokenizer: Associated tokenizer.
        config: Evaluation configuration.
        metrics_tracker: Optional tracker to record results.

    Returns:
        Dictionary mapping task -> accuracy.
    """
    if config is None:
        config = EvalConfig()

    if metrics_tracker is None:
        metrics_tracker = MetricsTracker()

    results = {}

    if config.use_lm_eval:
        try:
            results = _evaluate_with_lm_eval(model, tokenizer, config, metrics_tracker)
        except ImportError:
            logger.warning("lm-eval not available, using manual evaluation")
            results = _evaluate_manual(model, tokenizer, config, metrics_tracker)
    else:
        results = _evaluate_manual(model, tokenizer, config, metrics_tracker)

    return results


def _evaluate_with_lm_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: EvalConfig,
    metrics_tracker: MetricsTracker,
) -> dict:
    """Evaluate using lm-evaluation-harness."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    results = {}
    model.eval()

    task_list = list(config.tasks)
    logger.info(f"Running lm-eval on tasks: {task_list}")

    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=config.batch_size)

    task_manager = TaskManager()

    for task in task_list:
        fewshot = config.mmlu_fewshot if task == "mmlu" else config.num_fewshot
        try:
            eval_results = lm_eval.simple_evaluate(
                model=lm_eval_model,
                tasks=[task],
                num_fewshot=fewshot,
                task_manager=task_manager,
                limit=config.max_samples,
            )
            task_acc = eval_results["results"][task].get("acc,none")
            if task_acc is not None:
                results[task] = task_acc
                metrics_tracker.record("benchmarks", **{task: task_acc})
                logger.info(f"  {task}: {task_acc:.4f}")
            else:
                logger.warning(f"  {task}: no accuracy metric found")
        except Exception as e:
            logger.error(f"  {task}: evaluation failed - {e}")

    avg_acc = sum(results.values()) / len(results) if results else 0
    metrics_tracker.record("benchmarks", average_accuracy=avg_acc)
    logger.info(f"Average accuracy: {avg_acc:.4f}")
    return results


def _evaluate_manual(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: EvalConfig,
    metrics_tracker: MetricsTracker,
) -> dict:
    """Fallback: manual perplexity evaluation on WikiText."""
    results = {}
    logger.info("Manual evaluation: computing perplexity only")

    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = dataset["text"]
        texts = [t for t in texts if t.strip()]

        total_loss = 0.0
        total_tokens = 0
        model.cuda()
        model.eval()

        with torch.no_grad():
            for text in texts[: config.max_samples or len(texts)]:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                input_ids = enc["input_ids"].cuda()
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                if loss is not None:
                    total_loss += loss.item() * input_ids.numel()
                    total_tokens += input_ids.numel()

        perplexity = torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()
        results["wikitext_ppl"] = perplexity
        metrics_tracker.record("benchmarks", wikitext_perplexity=perplexity)
        logger.info(f"WikiText-2 perplexity: {perplexity:.4f}")
    except Exception as e:
        logger.error(f"Manual evaluation failed: {e}")

    return results


def run_benchmarks(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    original_model: Optional[PreTrainedModel] = None,
    original_tokenizer: Optional[PreTrainedTokenizer] = None,
    config: Optional[EvalConfig] = None,
) -> dict:
    """Run full benchmark suite, optionally comparing against original model.

    Returns dict with results from both models if comparison is provided.
    """
    if config is None:
        config = EvalConfig()

    metrics = MetricsTracker()

    logger.info("=" * 60)
    logger.info("Evaluating quantized model...")
    logger.info("=" * 60)

    quant_results = evaluate_model(model, tokenizer, config, metrics)

    original_results = None
    if original_model is not None:
        logger.info("=" * 60)
        logger.info("Evaluating original (full-precision) model...")
        logger.info("=" * 60)
        original_results = evaluate_model(
            original_model,
            original_tokenizer or tokenizer,
            config,
            metrics,
        )

        logger.info("=" * 60)
        logger.info("Comparison (Quantized vs Original):")
        for task, q_acc in quant_results.items():
            if task in (original_results or {}):
                o_acc = original_results[task]
                delta = (q_acc - o_acc) * 100
                logger.info(f"  {task}: {q_acc:.4f} vs {o_acc:.4f} (delta={delta:+.2f}%)")

    metrics.summary(logger)
    return {"quantized": quant_results, "original": original_results}
