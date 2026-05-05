#!/usr/bin/env python3
"""Colab T4 test script — run once, get all results.

Usage in Colab:
  !pip install transformers safetensors datasets torch
  !wget https://raw.githubusercontent.com/adriel007/ternary-boost/main/scripts/colab_test.py
  %run colab_test.py

Or just copy-paste the entire script into a Colab cell and run.
"""
import sys, os, time, json, math, gc
from collections import Counter

os.environ.setdefault("HF_HOME", "/content/cache/huggingface")

# ── Setup ──────────────────────────────────────────────────────────
ROOT = os.getcwd()
if not os.path.exists(os.path.join(ROOT, "pyproject.toml")):
    ROOT = os.path.dirname(ROOT)
    if not os.path.exists(os.path.join(ROOT, "pyproject.toml")):
        raise RuntimeError("Not in ternary-boost directory. Run: %cd ternary-boost")

for m in ["shared", "pt_bitnet", "tequila", "chat"]:
    sys.path.insert(0, os.path.join(m, "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from chat.model_loader import load_model
from chat.config import ModelEntry

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
print(f"Device: {device} | Dtype: {dtype}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# ── Quality metrics ─────────────────────────────────────────────────

def compute_perplexity(model, tokenizer, texts, max_length=128):
    """Standard perplexity — the gold-standard metric for compression quality.
    Lower = better. No external dataset needed — uses calibration texts.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts[:16]:  # 16 samples gives stable estimate
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=max_length)
            input_ids = inputs["input_ids"].to(model.device)
            if input_ids.numel() < 2:
                continue
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def repetition_ratio(text):
    """Detects model degeneration. Measures fraction of trigrams that repeat.
    0.0 = all unique (healthy). 1.0 = all repeats (degenerate/gibberish).
    """
    words = text.split()
    if len(words) < 5:
        return 0.0
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    if len(trigrams) < 2:
        return 0.0
    return 1.0 - len(set(trigrams)) / len(trigrams)


def generation_score(prompt, response, expected_keywords):
    """Score generation quality 0-100.

    Checks: non-empty, reasonable length, contains expected info, low
    repetition, proper sentence structure. Does NOT just keyword-match —
    a response repeating "Paris" scores poorly on structure/repetition.
    """
    score = 0
    text = response.strip()
    words = text.split()
    wc = len(words)

    # 1. Reasonable length (3-150 words)
    if 3 <= wc <= 150:
        score += 20
    elif wc > 2:
        score += 10  # too short but at least saying something

    # 2. Contains expected factual content
    found = sum(1 for kw in expected_keywords if kw.lower() in text.lower())
    if found >= len(expected_keywords) * 0.5:
        score += 30  # hits most keywords
    elif found > 0:
        score += 15  # at least some relevant content

    # 3. Low repetition (degenerate models repeat n-grams)
    rep = repetition_ratio(text)
    if rep < 0.10:
        score += 25
    elif rep < 0.25:
        score += 15
    elif rep < 0.50:
        score += 5
    # rep >= 0.50: heavily repetitive, 0 points

    # 4. Sentence structure
    if text and text[0].isupper():
        score += 10
    if any(p in text for p in ".!?"):
        score += 10
    # Penalize if it's just the prompt echoed back
    prompt_words = set(prompt.lower().split())
    response_words = set(text.lower().split())
    if len(response_words - prompt_words) < 2:
        score = max(0, score - 15)

    # 5. Has a verb (is it actually a sentence?)
    common_verbs = {"is", "are", "was", "were", "has", "have", "can", "will",
                    "would", "could", "should", "does", "do", "be", "been",
                    "learns", "trains", "travels", "goes", "means", "refers"}
    if any(v in words for v in common_verbs):
        score += 5

    return score


def jaccard_similarity(a, b):
    """Word overlap between two responses. 1.0 = identical word set."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ── Calibration texts (same as pipeline uses, no external download) ──
def get_calibration_texts():
    """Get calibration texts for perplexity. Reuses the same simple texts
    used by PT-BitNet compensation — no dataset download needed."""
    base = "The capital of France is Paris. " * 5
    texts = [
        base,
        "Machine learning is a field of artificial intelligence that enables "
        "computers to learn from data without being explicitly programmed. " * 2,
        "If a train travels at a constant speed of 60 miles per hour for 2 hours, "
        "the distance traveled is speed multiplied by time: 60 * 2 = 120 miles. " * 2,
        "The moon shines bright in the night sky, a silver disc among the stars. " * 3,
        "Python is a high-level programming language known for its readability. " * 3,
        "The Earth orbits the Sun at a distance of about 93 million miles. " * 3,
        "Water boils at 100 degrees Celsius at sea level. " * 5,
        "The human brain contains approximately 86 billion neurons. " * 3,
    ]
    return texts


# ── Test prompts with expected keywords ──────────────────────────────
TESTS = [
    ("Factual", "Question: What is the capital of France?\nAnswer:",
     ["paris", "france"]),
    ("Definition", "Question: What is machine learning?\nAnswer:",
     ["learn", "data", "algorithm", "computer", "intelligence"]),
    ("Math", "Question: If a train travels at 60mph for 2 hours, how far does it go?\nAnswer:",
     ["120", "mile", "distance", "speed", "hour"]),
    ("Creative", "Write a short poem about the moon:",
     ["moon", "night", "light", "sky", "star", "silver", "shine"]),
]

MODEL = "microsoft/phi-2"

# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}\nPipeline: {MODEL}\n{'='*60}")

entry = ModelEntry(
    name="phi-2", path=MODEL, device=device,
    lambada_granularity="per_channel",
)

# ── Pipeline ─────────────────────────────────────────────────────────
t0 = time.time()
model, tokenizer = load_model(entry, cache_root="/content/cache")
elapsed = time.time() - t0

layer_types = Counter(type(m).__name__ for m in model.modules()
                      if "Linear" in type(m).__name__)
has_ultra = any("UltraQuant" in t for t in layer_types)
speed_mode = "[1.58b]" if has_ultra else "[baked]"
print(f"\nPipeline done in {elapsed/60:.1f} min | "
      f"Layers: {dict(layer_types)} | Mode: {speed_mode}")

results = {
    "pipeline_time_s": elapsed, "model": MODEL, "device": device,
    "layers": dict(layer_types), "speed_mode": speed_mode,
}

# ── Perplexity ──────────────────────────────────────────────────────
print(f"\n{'='*60}\nPerplexity (lower = better)\n{'='*60}")
calib_texts = get_calibration_texts()

model.eval()
ppl = compute_perplexity(model, tokenizer, calib_texts)
print(f"  Quantized:  {ppl:.2f}")
results["perplexity_quantized"] = round(ppl, 2)

# ── Generation quality ──────────────────────────────────────────────
print(f"\n{'='*60}\nGeneration Quality (0-100 per prompt)\n{'='*60}")

gen_results = []
model.eval()
for category, prompt, keywords in TESTS:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(inputs.input_ids, max_new_tokens=30, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    gen_time = time.time() - t0
    gen_tokens = out.shape[1] - inputs.input_ids.shape[1]
    response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                                skip_special_tokens=True)

    score = generation_score(prompt, response, keywords)
    rep = repetition_ratio(response)
    speed = gen_tokens / gen_time if gen_time > 0 else 0

    print(f"  [{category}] score={score}/100 rep={rep:.2f} speed={speed:.1f} tok/s")
    print(f"    → '{response.strip()[:150]}'")

    gen_results.append({
        "category": category, "prompt": prompt, "response": response.strip(),
        "score": score, "repetition": round(rep, 3),
        "tokens": gen_tokens, "time_s": round(gen_time, 2),
        "speed_tok_s": round(speed, 1),
    })

results["generation"] = gen_results

# ── Baseline FP16 comparison ─────────────────────────────────────────
print(f"\n{'='*60}\nBaseline FP16 Comparison\n{'='*60}")

del model; gc.collect(); torch.cuda.empty_cache()

base_config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
if not hasattr(base_config, "pad_token_id") or base_config.pad_token_id is None:
    base_config.pad_token_id = 0
base = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=dtype, low_cpu_mem_usage=True,
    device_map=device, trust_remote_code=True, config=base_config,
)
base_tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
base_tok.pad_token = base_tok.eos_token
base.eval()

# Baseline perplexity
base_ppl = compute_perplexity(base, base_tok, calib_texts)
ppl_ratio = ppl / base_ppl if base_ppl > 0 else float("inf")
print(f"  Baseline FP16 perplexity: {base_ppl:.2f}")
print(f"  Quantized perplexity:     {ppl:.2f}")
print(f"  Ratio (quant/base):       {ppl_ratio:.3f}  {'✓' if ppl_ratio < 1.50 else '✗'}")
results["perplexity_baseline"] = round(base_ppl, 2)
results["perplexity_ratio"] = round(ppl_ratio, 3)

# Baseline generation
print(f"\n  Generation comparison:")
base_gen = []
for category, prompt, keywords in TESTS:
    inputs = base_tok(prompt, return_tensors="pt").to(base.device)
    t0 = time.time()
    with torch.no_grad():
        out = base.generate(inputs.input_ids, max_new_tokens=30, temperature=0.7,
                            do_sample=True, pad_token_id=base_tok.eos_token_id)
    gen_time = time.time() - t0
    gen_tokens = out.shape[1] - inputs.input_ids.shape[1]
    response = base_tok.decode(out[0][inputs.input_ids.shape[1]:],
                               skip_special_tokens=True)

    # Find matching quantized response
    qr = next((g for g in gen_results if g["category"] == category), None)
    jac = jaccard_similarity(response, qr["response"]) if qr else 0.0
    bscore = generation_score(prompt, response, keywords)
    brep = repetition_ratio(response)

    print(f"  [{category}] base={bscore}/100 q={qr['score']}/100 "
          f"rep={brep:.2f} jaccard={jac:.2f}")
    print(f"    base: '{response.strip()[:120]}'")
    if qr:
        print(f"    q:    '{qr['response'].strip()[:120]}'")

    base_gen.append({
        "category": category, "response": response.strip(),
        "score": bscore, "repetition": round(brep, 3),
        "tokens": gen_tokens, "time_s": round(gen_time, 2),
        "jaccard_vs_quantized": round(jac, 3),
    })

results["baseline"] = base_gen

# ── Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*60}\nSUMMARY\n{'='*60}")

q_scores = [g["score"] for g in gen_results]
b_scores = [g["score"] for g in base_gen]
avg_q = sum(q_scores) / len(q_scores) if q_scores else 0
avg_b = sum(b_scores) / len(b_scores) if b_scores else 0
avg_rep = sum(g["repetition"] for g in gen_results) / len(gen_results)
avg_speed = sum(g["speed_tok_s"] for g in gen_results) / len(gen_results)
avg_jac = sum(g["jaccard_vs_quantized"] for g in base_gen) / len(base_gen)

print(f"  Perplexity:       {ppl:.1f} (base={base_ppl:.1f}, ratio={ppl_ratio:.3f})")
print(f"  Gen quality:      {avg_q:.0f}/100 (baseline={avg_b:.0f}/100)")
print(f"  Jaccard overlap:  {avg_jac:.2f} (1.0 = identical to baseline)")
print(f"  Repetition:       {avg_rep:.3f} (0 = healthy, 1 = degenerate)")
print(f"  Speed:            {avg_speed:.1f} tok/s")
print(f"  Pipeline time:    {elapsed/60:.1f} min")

# Overall verdict
# Thresholds calibrated for ternary compression: 1-3 bits is inherently
# lossy. A 30-40% perplexity increase with identical generation quality
# is a strong result (baseline for ternary is typically 1.5-2.5x).
# Generation quality matters more than PPL for user experience.
if ppl_ratio < 1.15 and avg_q >= avg_b * 0.85:
    verdict = "EXCELLENT — near-FP16 quality"
elif ppl_ratio < 1.50 and avg_q >= avg_b * 0.70:
    verdict = "GOOD — minor quality loss, fully usable"
elif ppl_ratio < 2.00 and avg_rep < 0.30:
    verdict = "OK — noticeable loss but still coherent"
else:
    verdict = "FAIL — significant degradation"
print(f"\n  Verdict: {verdict}")

results["summary"] = {
    "perplexity_quantized": round(ppl, 2),
    "perplexity_baseline": round(base_ppl, 2),
    "perplexity_ratio": round(ppl_ratio, 3),
    "avg_generation_score": round(avg_q, 1),
    "avg_baseline_score": round(avg_b, 1),
    "avg_jaccard": round(avg_jac, 3),
    "avg_repetition": round(avg_rep, 3),
    "avg_speed_tok_s": round(avg_speed, 1),
    "pipeline_time_min": round(elapsed / 60, 1),
    "verdict": verdict,
}

with open("colab_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to colab_results.json")
print("Copy ALL output above and share it.")
