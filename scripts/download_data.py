#!/usr/bin/env python3
"""Download and prepare calibration/training data for ternary-boost pipeline.

Supports:
  - WikiText-2 (default, ~2M tokens, ideal for calibration)
  - C4 (larger dataset for QAT training)
  - Custom JSONL files

Usage:
  python scripts/download_data.py --dataset wikitext --num-samples 500
  python scripts/download_data.py --dataset c4 --num-samples 1000
  python scripts/download_data.py --dataset path/to/data.jsonl --num-samples 200
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.data import load_calibration_texts


def main():
    parser = argparse.ArgumentParser(
        description="Download/prepare data for ternary-boost pipeline"
    )
    parser.add_argument("--dataset", type=str, default="wikitext",
                        help="Dataset name (wikitext, c4, pile) or path to JSONL file")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="Number of text samples to load")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file path (default: data/<dataset>_samples.jsonl)")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split (default: train)")
    args = parser.parse_args()

    print(f"Loading {args.num_samples} samples from {args.dataset}...")
    texts = load_calibration_texts(
        source=args.dataset,
        split=args.split,
        num_samples=args.num_samples,
    )

    output_path = args.output
    if output_path is None:
        os.makedirs("data", exist_ok=True)
        ds_name = os.path.basename(args.dataset).replace(".jsonl", "")
        output_path = f"data/{ds_name}_{args.num_samples}samples.jsonl"

    with open(output_path, "w") as f:
        for text in texts:
            f.write(json.dumps({"text": text}) + "\n")

    print(f"Saved {len(texts)} samples to {output_path}")

    total_chars = sum(len(t) for t in texts)
    print(f"Total characters: {total_chars:,}")
    print(f"Avg chars/sample: {total_chars // max(len(texts), 1):,}")


if __name__ == "__main__":
    main()
