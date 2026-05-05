#!/usr/bin/env python3
"""Colab T4 test script — run once, get all results.

Usage in Colab:
  !pip install transformers safetensors datasets torch
  !wget https://raw.githubusercontent.com/adriel007/ternary-boost/main/scripts/colab_test.py
  %run colab_test.py

Or just copy-paste the entire script into a Colab cell and run.
"""
import sys, os, time, json

# Use our cache to avoid re-downloading the model for baseline comparison
os.environ.setdefault("HF_HOME", "/content/cache/huggingface")

# ── Setup ──────────────────────────────────────────────────────────
# Script expects to be run from INSIDE the ternary-boost directory.
# The user should have cloned and cd'd before running this.
ROOT = os.getcwd()
if not os.path.exists(os.path.join(ROOT, "pyproject.toml")):
    # Try parent directory (if we're in scripts/)
    ROOT = os.path.dirname(ROOT)
    if not os.path.exists(os.path.join(ROOT, "pyproject.toml")):
        raise RuntimeError("Not in ternary-boost directory. Run: %cd ternary-boost")

for m in ["shared", "pt_bitnet", "tequila", "chat"]:
    sys.path.insert(0, os.path.join(m, "src"))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from chat.model_loader import load_model
from chat.config import ModelEntry

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
print(f"Device: {device} | Dtype: {dtype}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── Pipeline ────────────────────────────────────────────────────────
MODEL = "microsoft/phi-2"
print(f"\n{'='*60}\nPipeline: {MODEL}\n{'='*60}")

entry = ModelEntry(
    name="phi-2", path=MODEL, device=device,
    lambada_granularity="per_channel",
)

t0 = time.time()
model, tokenizer = load_model(entry, cache_root="/content/cache")
elapsed = time.time() - t0

# Verify layers
from collections import Counter
layer_types = Counter(type(m).__name__ for m in model.modules() if "Linear" in type(m).__name__)
has_ultra = any("UltraQuant" in t for t in layer_types)
speed_mode = "[1.58b]" if has_ultra else "[baked]"
print(f"\nPipeline done in {elapsed/60:.1f} min | Layers: {dict(layer_types)} | Mode: {speed_mode}")

# ── Quality Tests ───────────────────────────────────────────────────
print(f"\n{'='*60}\nQuality Tests\n{'='*60}")

tests = [
    ("Factual", "Question: What is the capital of France?\nAnswer:"),
    ("Definition", "Question: What is machine learning?\nAnswer:"),
    ("Math", "Question: If a train travels at 60mph for 2 hours, how far does it go?\nAnswer:"),
    ("Creative", "Write a short poem about the moon:"),
]

results = {"pipeline_time_s": elapsed, "model": MODEL, "device": device,
           "layers": dict(layer_types), "speed_mode": speed_mode, "tests": []}

model.eval()
for category, prompt in tests:
    print(f"\n[{category}] {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(inputs.input_ids, max_new_tokens=30, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    gen_time = time.time() - t0
    gen_tokens = out.shape[1] - inputs.input_ids.shape[1]
    response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # Honest quality check: does response contain relevant keywords?
    is_coherent = len(response.split()) > 2
    is_relevant = any(w in response.lower() for w in ["paris", "france", "learn", "data", "algorithm", "moon", "night", "sky", "mile", "120", "distance", "speed"])
    quality = "OK" if (is_coherent and is_relevant) else "POOR"

    print(f"  Response: '{response.strip()[:200]}'")
    print(f"  Tokens: {gen_tokens} | Time: {gen_time:.1f}s | Speed: {gen_tokens/gen_time:.2f} tok/s | Quality: {quality}")

    results["tests"].append({
        "category": category, "prompt": prompt, "response": response.strip(),
        "tokens": gen_tokens, "time_s": gen_time, "speed_tok_s": gen_tokens/gen_time,
        "quality": quality,
    })

# ── Compare with baseline ───────────────────────────────────────────
print(f"\n{'='*60}\nBaseline FP16 Comparison\n{'='*60}")

del model
torch.cuda.empty_cache()

base = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=dtype, low_cpu_mem_usage=True,
    device_map=device, trust_remote_code=True,
)
base_tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
base_tok.pad_token = base_tok.eos_token

results["baseline_tests"] = []
base.eval()
for category, prompt in tests:
    inputs = base_tok(prompt, return_tensors="pt").to(base.device)
    t0 = time.time()
    with torch.no_grad():
        out = base.generate(inputs.input_ids, max_new_tokens=30, temperature=0.7,
                            do_sample=True, pad_token_id=base_tok.eos_token_id)
    gen_time = time.time() - t0
    gen_tokens = out.shape[1] - inputs.input_ids.shape[1]
    response = base_tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    is_coherent = len(response.split()) > 2
    is_relevant = any(w in response.lower() for w in ["paris", "france", "learn", "data", "algorithm", "moon", "night", "sky", "mile", "120", "distance", "speed"])
    quality = "OK" if (is_coherent and is_relevant) else "POOR"
    print(f"[{category}] '{response.strip()[:150]}' ({gen_tokens} tok, {gen_time:.1f}s)")

    results["baseline_tests"].append({
        "category": category, "response": response.strip(),
        "tokens": gen_tokens, "time_s": gen_time,
    })

# ── Summary ────────────────────────────────────────────────────────
print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
ters = results["tests"]
good = sum(1 for t in ters if t["quality"] == "GOOD")
avg_speed = sum(t["speed_tok_s"] for t in ters) / len(ters) if ters else 0
print(f"Pipeline: {elapsed/60:.1f} min | Quality: {good}/{len(ters)} good | Speed: {avg_speed:.2f} tok/s")

# Save
with open("colab_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to colab_results.json")
print("Copy ALL output above and share it.")
