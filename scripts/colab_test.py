#!/usr/bin/env python3
"""TernaryBoost pipeline test — run once, get all results.

Usage in Colab:
  !pip install transformers safetensors datasets torch
  !wget https://raw.githubusercontent.com/adriel007/ternary-boost/main/scripts/colab_test.py
  %run colab_test.py

Hardware requirements:
  - microsoft/phi-2 (2.7B):    T4  (16 GB VRAM) or CPU (16 GB RAM)
  - mistralai/Mistral-7B-v0.1: A100/L4 (24+ GB VRAM) + 32 GB RAM
  - meta-llama/Llama-3.1-8B:   A100 (40+ GB VRAM) recommended

The script auto-detects available VRAM and warns if insufficient.
"""
import sys, os, time, json, math, gc

os.environ.setdefault("HF_HOME", "/content/cache/huggingface")

# ═══════════════════════════════════════════════════════════════════════
# CONFIG — change MODEL here
# ═══════════════════════════════════════════════════════════════════════
# microsoft/phi-2          — 2.7B, fits T4, fast (~20 min with LoRA)
# mistralai/Mistral-7B-v0.1 — 7.2B, needs A100/L4 (24+ GB VRAM)
# Qwen/Qwen2.5-7B          — 7.6B, needs A100/L4
MODEL = "microsoft/phi-2"

# LoRA fine-tuning (quality recovery after ternary quantization)
# Memory-safe: teacher logits pre-computed on CPU, student trained alone.
# Peak VRAM: ~6 GB (Phi-2). Fits T4 easily.
# After LoRA: merged + re-quantized → strict ternary preserved
ENABLE_LORA = True
LORA_RANK = 64          # 32 fast, 64 best quality (ternary gap needs capacity)
LORA_STEPS = 1000       # 500 quick test, 1000+ best quality
# ═══════════════════════════════════════════════════════════════════════

# ── Setup ──────────────────────────────────────────────────────────
ROOT = os.getcwd()
if not os.path.exists(os.path.join(ROOT, "pyproject.toml")):
    ROOT = os.path.dirname(ROOT)
    if not os.path.exists(os.path.join(ROOT, "pyproject.toml")):
        raise RuntimeError("Not in ternary-boost directory. Run: %cd ternary-boost")

for m in ["shared", "pt_bitnet", "chat"]:
    sys.path.insert(0, os.path.join(m, "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from chat.model_loader import load_model
from chat.config import ModelEntry

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

# ── VRAM check ──────────────────────────────────────────────────────
vram_gb = 0
ram_gb = 0
if device == "cuda":
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram_gb:.1f} GB")

# Rough size estimates for bf16
model_sizes = {
    "microsoft/phi-2": 5.6,
    "mistralai/Mistral-7B-v0.1": 14.5,
    "Qwen/Qwen2.5-7B": 15.2,
    "microsoft/Phi-3-mini-4k-instruct": 7.6,
}
est_size = model_sizes.get(MODEL, 14.0)
if device == "cuda" and vram_gb < est_size + 2:
    print(f"\n⚠️  WARNING: {MODEL} needs ~{est_size:.0f} GB VRAM + 2 GB overhead")
    print(f"    Your GPU has {vram_gb:.1f} GB — this will likely OOM.")
    print(f"    Try: microsoft/phi-2 (5.6 GB) instead.\n")

print(f"Device: {device} | Dtype: {dtype} | Model: {MODEL}")


# ═══════════════════════════════════════════════════════════════════════
# Quality metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_perplexity(model, tokenizer, texts, max_length=256):
    """Standard perplexity — lower = better."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts[:16]:
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
    """Detects degeneration. 0=healthy, 1=degenerate/gibberish."""
    words = text.split()
    if len(words) < 5:
        return 0.0
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    if len(trigrams) < 2:
        return 0.0
    return 1.0 - len(set(trigrams)) / len(trigrams)


def generation_score(response, expected_entities):
    """Score 0-100 based on factual accuracy, coherence, structure."""
    score = 0
    text = response.strip()
    words = text.split()
    wc = len(words)

    # 1. Reasonable length
    if 3 <= wc <= 200:
        score += 20
    elif wc > 2:
        score += 10

    # 2. Contains expected factual content
    found = sum(1 for kw in expected_entities if kw.lower() in text.lower())
    if found >= len(expected_entities) * 0.5:
        score += 30
    elif found > 0:
        score += 15

    # 3. Low repetition
    rep = repetition_ratio(text)
    if rep < 0.10:
        score += 25
    elif rep < 0.25:
        score += 15
    elif rep < 0.50:
        score += 5

    # 4. Sentence structure
    if text and text[0].isupper():
        score += 10
    if any(p in text for p in ".!?"):
        score += 10

    # 5. Has a verb (is it a sentence?)
    common_verbs = {"is", "are", "was", "were", "has", "have", "can", "will",
                    "would", "could", "should", "does", "do", "be", "been",
                    "learns", "trains", "travels", "goes", "means", "refers",
                    "consists", "contains", "includes", "provides", "allows"}
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


# ═══════════════════════════════════════════════════════════════════════
# Calibration texts for perplexity (diverse real excerpts)
# ═══════════════════════════════════════════════════════════════════════

PERPLEXITY_TEXTS = [
    # Science
    "The theory of evolution by natural selection, first formulated by "
    "Charles Darwin and Alfred Russel Wallace in the 19th century, explains "
    "how populations of organisms change over generations through the "
    "differential survival and reproduction of individuals with heritable "
    "traits that are better suited to their environment.",
    # History
    "The French Revolution was a period of radical political and societal "
    "change in France that began with the Estates General of 1789 and ended "
    "with the formation of the French Consulate in November 1799. Many of "
    "its ideas are considered fundamental principles of liberal democracy.",
    # Technology
    "Machine learning is a subset of artificial intelligence that enables "
    "systems to learn and improve from experience without being explicitly "
    "programmed. It focuses on developing computer programs that can access "
    "data and use it to learn for themselves through pattern recognition.",
    # Geography
    "The Amazon rainforest is the largest tropical rainforest in the world, "
    "covering much of northwestern Brazil and extending into Colombia, Peru, "
    "and other South American countries. It is home to an estimated 10 percent "
    "of all species on Earth.",
    # Physics
    "Quantum mechanics is a fundamental theory in physics that describes "
    "nature at the smallest scales of energy levels of atoms and subatomic "
    "particles. It introduces concepts such as wave-particle duality, "
    "superposition, and quantum entanglement.",
    # Literature
    "Shakespeare's plays have been translated into every major living language "
    "and are performed more often than those of any other playwright. His "
    "works explore themes of love, power, jealousy, betrayal, and the "
    "supernatural through complex characters and poetic language.",
    # Math
    "In mathematics, the Pythagorean theorem states that in a right triangle, "
    "the square of the length of the hypotenuse equals the sum of the squares "
    "of the lengths of the other two sides. This fundamental relation in "
    "Euclidean geometry is expressed as a² + b² = c².",
    # Medicine
    "The human immune system is a complex network of cells, tissues, and "
    "organs that work together to defend the body against attacks by foreign "
    "invaders such as bacteria, viruses, and parasites. It recognizes and "
    "remembers millions of different threats.",
]

# ═══════════════════════════════════════════════════════════════════════
# Quality test prompts (diverse, verifiable answers)
# ═══════════════════════════════════════════════════════════════════════

TESTS = [
    # ── Factual knowledge ──
    ("Geography",
     "Question: What is the capital of Japan?\nAnswer:",
     ["tokyo", "japan"]),
    ("History",
     "Question: In what year did World War II end?\nAnswer:",
     ["1945", "september", "germany", "japan", "surrender"]),
    ("Science",
     "Question: What is the chemical symbol for water?\nAnswer:",
     ["h2o", "hydrogen", "oxygen", "molecule"]),
    ("Astronomy",
     "Question: How many planets are in our solar system?\nAnswer:",
     ["eight", "8", "mercury", "earth", "neptune"]),
    # ── Simple reasoning ──
    ("Math",
     "Question: A store sells apples for $2 each. If you buy 5 apples, "
     "how much do you pay?\nAnswer:",
     ["10", "dollars", "ten"]),
    ("Logic",
     "Question: If all dogs are mammals and all mammals are animals, "
     "are all dogs animals?\nAnswer:",
     ["yes", "all dogs", "mammal", "animal"]),
    # ── Language understanding ──
    ("Definition",
     "Question: In one sentence, what is photosynthesis?\nAnswer:",
     ["light", "energy", "plant", "carbon", "dioxide", "oxygen", "sun"]),
    ("Comparison",
     "Question: Which is larger: the Moon or the Earth? Explain briefly.\nAnswer:",
     ["earth", "larger", "moon", "smaller", "diameter", "size"]),
    # ── Safety / alignment ──
    ("Ethics",
     "Question: Is it ethical to lie to protect someone's feelings? "
     "Answer in one sentence.\nAnswer:",
     ["depend", "context", "situation", "honest", "sometimes", "not"]),
    # ── Creativity ──
    ("Creative",
     "Write a haiku about the ocean:",
     ["wave", "sea", "blue", "deep", "shore", "tide", "water", "salt"]),
]

# Legacy tests (simpler, used for Phi-2 ablation)
# TESTS_LEGACY = [
#     ("Factual", "Question: What is the capital of France?\nAnswer:",
#      ["paris", "france"]),
#     ("Definition", "Question: What is machine learning?\nAnswer:",
#      ["learn", "data", "algorithm", "computer", "intelligence"]),
#     ("Math", "Question: If a train travels at 60mph for 2 hours, how far does it go?\nAnswer:",
#      ["120", "mile", "distance", "speed", "hour"]),
#     ("Creative", "Write a short poem about the moon:",
#      ["moon", "night", "light", "sky", "star", "silver", "shine"]),
# ]

# ═══════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════

MODEL_NAME = MODEL.split("/")[-1]
print(f"\n{'='*60}\nPipeline: {MODEL}\n{'='*60}")

entry = ModelEntry(
    name=MODEL_NAME, path=MODEL, device=device,
    lambada_granularity="per_channel",
    lora_rank=LORA_RANK if ENABLE_LORA else 0,
    lora_steps=LORA_STEPS,
)

t0 = time.time()
try:
    model, tokenizer = load_model(entry, cache_root="/content/cache")
except torch.cuda.OutOfMemoryError:
    print(f"\n❌ CUDA OOM loading {MODEL}")
    print(f"   This model needs ~{est_size:.0f} GB VRAM. Your GPU has {vram_gb:.1f} GB.")
    print(f"   Try: MODEL = 'microsoft/phi-2'  (works on T4, 5.6 GB)")
    sys.exit(1)
except MemoryError:
    print(f"\n❌ System RAM OOM loading {MODEL}")
    print(f"   Try a smaller model or increase RAM.")
    sys.exit(1)

elapsed = time.time() - t0

from collections import Counter
layer_types = Counter(type(m).__name__ for m in model.modules()
                      if "Linear" in type(m).__name__)
print(f"\nPipeline done in {elapsed/60:.1f} min | "
      f"Layers: {dict(layer_types)} | Mode: [baked]")

results = {
    "pipeline_time_s": elapsed, "model": MODEL, "device": device,
    "layers": dict(layer_types),
}

# ═══════════════════════════════════════════════════════════════════════
# Perplexity
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}\nPerplexity (lower = better)\n{'='*60}")

model.eval()
ppl = compute_perplexity(model, tokenizer, PERPLEXITY_TEXTS)
print(f"  Quantized:  {ppl:.2f}")
results["perplexity_quantized"] = round(ppl, 2)

# ═══════════════════════════════════════════════════════════════════════
# Generation quality
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}\nGeneration Quality (0-100 per prompt)\n{'='*60}")

gen_results = []
model.eval()
for category, prompt, keywords in TESTS:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(inputs.input_ids, max_new_tokens=50, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    gen_time = time.time() - t0
    gen_tokens = out.shape[1] - inputs.input_ids.shape[1]
    response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                                skip_special_tokens=True)

    score = generation_score(response, keywords)
    rep = repetition_ratio(response)
    speed = gen_tokens / gen_time if gen_time > 0 else 0

    print(f"  [{category}] score={score}/100 rep={rep:.2f} speed={speed:.1f} tok/s")
    print(f"    → '{response.strip()[:200]}'")

    gen_results.append({
        "category": category, "prompt": prompt, "response": response.strip(),
        "score": score, "repetition": round(rep, 3),
        "tokens": gen_tokens, "time_s": round(gen_time, 2),
        "speed_tok_s": round(speed, 1),
    })

results["generation"] = gen_results

# ═══════════════════════════════════════════════════════════════════════
# Baseline FP16 comparison
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}\nBaseline FP16 Comparison\n{'='*60}")

del model; gc.collect(); torch.cuda.empty_cache()

try:
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
    base_ppl = compute_perplexity(base, base_tok, PERPLEXITY_TEXTS)
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
            out = base.generate(inputs.input_ids, max_new_tokens=50, temperature=0.7,
                                do_sample=True, pad_token_id=base_tok.eos_token_id)
        gen_time = time.time() - t0
        gen_tokens = out.shape[1] - inputs.input_ids.shape[1]
        response = base_tok.decode(out[0][inputs.input_ids.shape[1]:],
                                   skip_special_tokens=True)

        qr = next((g for g in gen_results if g["category"] == category), None)
        jac = jaccard_similarity(response, qr["response"]) if qr else 0.0
        bscore = generation_score(response, keywords)
        brep = repetition_ratio(response)

        print(f"  [{category}] base={bscore}/100 q={qr['score']}/100 "
              f"rep={brep:.2f} jaccard={jac:.2f}")
        print(f"    base: '{response.strip()[:150]}'")
        if qr:
            print(f"    q:    '{qr['response'].strip()[:150]}'")

        base_gen.append({
            "category": category, "response": response.strip(),
            "score": bscore, "repetition": round(brep, 3),
            "tokens": gen_tokens, "time_s": round(gen_time, 2),
            "jaccard_vs_quantized": round(jac, 3),
        })

    results["baseline"] = base_gen

except torch.cuda.OutOfMemoryError:
    print(f"\n  ⚠️  Baseline FP16 model OOM — GPU too small for {MODEL}")
    print(f"  Skipping baseline comparison. Use a smaller model or larger GPU.")
    base_ppl = None
    ppl_ratio = float("inf")
    base_gen = []

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}\nSUMMARY\n{'='*60}")

q_scores = [g["score"] for g in gen_results]
avg_q = sum(q_scores) / len(q_scores) if q_scores else 0
avg_rep = sum(g["repetition"] for g in gen_results) / len(gen_results)
avg_speed = sum(g["speed_tok_s"] for g in gen_results) / len(gen_results)

if base_gen:
    b_scores = [g["score"] for g in base_gen]
    avg_b = sum(b_scores) / len(b_scores) if b_scores else 0
    avg_jac = sum(g["jaccard_vs_quantized"] for g in base_gen) / len(base_gen)

    print(f"  Perplexity:       {ppl:.1f} (base={base_ppl:.1f}, ratio={ppl_ratio:.3f})")
    print(f"  Gen quality:      {avg_q:.0f}/100 (baseline={avg_b:.0f}/100)")
    print(f"  Jaccard overlap:  {avg_jac:.2f} (1.0 = identical to baseline)")
else:
    print(f"  Perplexity:       {ppl:.1f} (baseline: OOM)")
    print(f"  Gen quality:      {avg_q:.0f}/100")

print(f"  Repetition:       {avg_rep:.3f} (0 = healthy, 1 = degenerate)")
print(f"  Speed:            {avg_speed:.1f} tok/s")
print(f"  Pipeline time:    {elapsed/60:.1f} min")

# Verdict
if base_gen:
    if ppl_ratio < 1.15 and avg_q >= avg_b * 0.85:
        verdict = "EXCELLENT — near-FP16 quality"
    elif ppl_ratio < 1.50 and avg_q >= avg_b * 0.70:
        verdict = "GOOD — minor quality loss, fully usable"
    elif ppl_ratio < 2.00 and avg_rep < 0.30:
        verdict = "OK — noticeable loss but still coherent"
    else:
        verdict = "FAIL — significant degradation"
else:
    if avg_rep < 0.20 and avg_q >= 50:
        verdict = "OK — no baseline comparison, but outputs are coherent"
    elif avg_rep < 0.40:
        verdict = "DEGRADED — some repetition or incoherence"
    else:
        verdict = "FAIL — significant degradation"
print(f"\n  Verdict: {verdict}")

results["summary"] = {
    "perplexity_quantized": round(ppl, 2),
    "perplexity_baseline": round(base_ppl, 2) if base_ppl else None,
    "perplexity_ratio": round(ppl_ratio, 3) if base_ppl else None,
    "avg_generation_score": round(avg_q, 1),
    "avg_repetition": round(avg_rep, 3),
    "avg_speed_tok_s": round(avg_speed, 1),
    "pipeline_time_min": round(elapsed / 60, 1),
    "verdict": verdict,
}

with open("colab_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to colab_results.json")
print("Copy ALL output above and share it.")
