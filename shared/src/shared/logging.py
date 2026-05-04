import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def log_memory_usage(logger: logging.Logger, prefix: str = "") -> None:
    """Log current GPU and CPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(
            f"{prefix}GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )


@dataclass
class MetricsTracker:
    """Accumulates and reports pipeline metrics across stages."""

    metrics: dict = field(default_factory=dict)

    def record(self, stage: str, **kwargs) -> None:
        for key, value in kwargs.items():
            self.metrics[f"{stage}/{key}"] = value

    def to_dict(self) -> dict:
        return dict(self.metrics)

    def summary(self, logger: Optional[logging.Logger] = None) -> str:
        lines = []
        current_stage = None
        for key, value in sorted(self.metrics.items()):
            stage, metric = key.split("/", 1)
            if stage != current_stage:
                current_stage = stage
                lines.append(f"\n{stage}:")
            if isinstance(value, float):
                lines.append(f"  {metric}: {value:.4f}")
            else:
                lines.append(f"  {metric}: {value}")
        summary_str = "\n".join(lines)
        if logger:
            logger.info(f"Pipeline metrics summary:{summary_str}")
        return summary_str
