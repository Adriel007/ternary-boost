#!/usr/bin/env python3
"""TernaryBoost export pipeline test — valida INT2 packing + TernaryInferenceLinear.

Focado APENAS no novo pipeline de exportação (fases 1-4 do plano).
NÃO repete os testes de PPL/generation que o colab_test.py já cobre.

Runtime estimado no Colab T4:
  - Sem LoRA: ~10 min (PT-BitNet 2min + export 1min + verify 5min + size 1min)
  - Com LoRA: ~50 min (+42min de LoRA training)

Uso:
  %run scripts/colab_export_test.py
"""

import sys, os, time, json, gc, math

os.environ.setdefault("HF_HOME", "/content/cache/huggingface")

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
MODEL = "microsoft/phi-2"
ENABLE_LORA = True  # True = testa export COM LoRA (~50 min)
LORA_RANK = 64
LORA_STEPS = 1000
EXPORT_DIR = "/content/exported_phi2"
CKPT_DIR = f"/content/export_checkpoints/{MODEL.replace('/', '__')}"
# ═══════════════════════════════════════════════════════════════════════

ROOT = os.getcwd()
if not os.path.exists(os.path.join(ROOT, "pyproject.toml")):
    ROOT = os.path.dirname(ROOT)
    if not os.path.exists(os.path.join(ROOT, "pyproject.toml")):
        # Colab: clone if needed
        if not os.path.exists("ternary-boost"):
            os.system("git clone https://github.com/adriel007/ternary-boost.git")
        ROOT = os.path.join(os.getcwd(), "ternary-boost")

for m in ["shared", "pt_bitnet", "chat"]:
    sys.path.insert(0, os.path.join(ROOT, m, "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from pt_bitnet import (
    apply_pt_bitnet,
    PTBitNetConfig,
    export_ternary_lora,
    load_ternary_lora,
    pack_int2,
    unpack_int2,
    TernaryInferenceLinear,
)
from pt_bitnet.int2_packing import verify_roundtrip
from shared.data import load_calibration_texts

torch.manual_seed(42)

# ═══════════════════════════════════════════════════════════════════════
# Checkpoint helpers — allow resume after crash without re-running
# ═══════════════════════════════════════════════════════════════════════
from pathlib import Path as _Path


def _ckpt_path(stage):
    return _Path(CKPT_DIR) / stage


def _has_ckpt(stage):
    return (_ckpt_path(stage) / "done.txt").exists()


def _save_ckpt(model, tokenizer, stage):
    """Save model + tokenizer as sharded safetensors checkpoint."""
    from chat.model_loader import _save_model_state
    p = _ckpt_path(stage)
    p.mkdir(parents=True, exist_ok=True)
    _save_model_state(model, tokenizer, p)
    (p / "done.txt").write_text("ok")
    print(f"  [checkpoint] Saved {stage}")


def _load_ckpt(stage, device, dtype):
    """Load model + tokenizer from checkpoint."""
    p = _ckpt_path(stage)
    config = AutoConfig.from_pretrained(str(p), trust_remote_code=True)
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = 0
    model = AutoModelForCausalLM.from_pretrained(
        str(p), torch_dtype=dtype, device_map=device,
        trust_remote_code=True, config=config,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n = sum(p.numel() for p in model.parameters())
    print(f"  [checkpoint] Loaded {stage} ({n:,} params)")
    return model, tokenizer


def _save_lora_ckpt(model, stage):
    """Save LoRA weights + config (lightweight, reuses stage2 base model)."""
    from pt_bitnet.lora import save_lora_weights
    p = _ckpt_path(stage)
    p.mkdir(parents=True, exist_ok=True)
    save_lora_weights(model, str(p / "lora_weights.safetensors"))
    (p / "lora_config.json").write_text(json.dumps({
        "rank": LORA_RANK, "steps": LORA_STEPS,
    }))
    (p / "done.txt").write_text("ok")
    print(f"  [checkpoint] Saved {stage} (LoRA weights only)")


def _load_lora_ckpt(model, stage):
    """Load LoRA weights onto a model that already has LoRA wrappers."""
    from pt_bitnet.lora import load_lora_weights
    p = _ckpt_path(stage)
    # Validate saved config matches current — warn if mismatch
    cfg_path = p / "lora_config.json"
    if cfg_path.exists():
        saved = json.loads(cfg_path.read_text())
        if saved.get("rank") != LORA_RANK or saved.get("steps") != LORA_STEPS:
            print(f"  [!] LoRA config changed (saved rank={saved.get('rank')} "
                  f"vs current={LORA_RANK}). Delete {p} to re-train.")
    lora_path = str(p / "lora_weights.safetensors")
    if _Path(lora_path).exists():
        load_lora_weights(model, lora_path)
        print(f"  [checkpoint] Loaded LoRA weights from {stage}")
    return model


# ═══════════════════════════════════════════════════════════════════════

resumed_stages = []

print("=" * 60)
print("TernaryBoost Export Pipeline Test")
print(f"Model: {MODEL}  |  LoRA: {'ON' if ENABLE_LORA else 'OFF'}")
print(f"Export dir: {EXPORT_DIR}")
print(f"Checkpoints: {CKPT_DIR}")
if any(_has_ckpt(s) for s in ["stage1_base", "stage2_pt_bitnet", "stage3_lora"]):
    print("Resume: YES — skipping completed stages")
print("=" * 60)

# ── Step 0: INT2 packing sanity check (local, 1s) ────────────────────
print("\n[0/5] INT2 packing verification...")
t0 = time.time()
results = verify_roundtrip()
all_ok = all(r["match"] for r in results.values())
print(
    f"  {'OK' if all_ok else 'FAIL'} — {len(results)} sizes tested in {time.time()-t0:.1f}s"
)
if not all_ok:
    print("  FAILED sizes:", [k for k, v in results.items() if not v["match"]])
    sys.exit(1)

# ── Step 1: Load base model ──────────────────────────────────────────
print("\n[1/5] Loading base model...")
t0 = time.time()

has_cuda = torch.cuda.is_available()
device = "cuda" if has_cuda else "cpu"
dtype = torch.bfloat16 if has_cuda else torch.float32

if _has_ckpt("stage1_base"):
    model, tokenizer = _load_ckpt("stage1_base", "cpu", dtype)
    resumed_stages.append("base")
else:
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
        trust_remote_code=True,
        config=config,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _save_ckpt(model, tokenizer, "stage1_base")

n_params = sum(p.numel() for p in model.parameters())
print(f"  Loaded {n_params:,} params in {time.time()-t0:.1f}s")

# ── Step 2: PT-BitNet quantization ───────────────────────────────────
print("\n[2/5] PT-BitNet quantization...")
t0 = time.time()

texts = load_calibration_texts(source="wikitext", num_samples=200)

if _has_ckpt("stage2_pt_bitnet"):
    del model; gc.collect()
    if has_cuda:
        torch.cuda.empty_cache()
    model, tokenizer = _load_ckpt("stage2_pt_bitnet", device, dtype)
    resumed_stages.append("pt_bitnet")
else:
    pt_config = PTBitNetConfig(
        asymmetric=False,
        outlier_fraction=0.01,
        compensation_steps=30 if has_cuda else 10,
        use_obc=False,
    )

    if has_cuda:
        model.cuda()
        torch.cuda.empty_cache()

    model = apply_pt_bitnet(model, pt_config, tokenizer, texts[:32])
    if has_cuda:
        _save_ckpt(model, tokenizer, "stage2_pt_bitnet")

elapsed_pt = time.time() - t0
print(f"  PT-BitNet done in {elapsed_pt:.1f}s")

# ── Step 3: LoRA (optional) ──────────────────────────────────────────
lora_elapsed = 0
if ENABLE_LORA:
    print(f"\n[3/5] LoRA fine-tuning (rank={LORA_RANK}, steps={LORA_STEPS})...")
    t0 = time.time()

    from pt_bitnet.lora import (
        finetune_lora, keep_lora_separate, LoRAConfig,
        _add_lora_to_model,
    )

    if has_cuda:
        model.cuda()
        torch.cuda.empty_cache()
        import gc as _gc
        _gc.collect()

    if _has_ckpt("stage3_lora"):
        # Resume: base model loaded from stage2, just re-wrap with LoRA + load weights
        lo_cfg = LoRAConfig(rank=LORA_RANK, num_steps=LORA_STEPS)
        model = _add_lora_to_model(model, lo_cfg)
        model = _load_lora_ckpt(model, "stage3_lora")
        model = keep_lora_separate(model)
        resumed_stages.append("lora")
    else:
        teacher_config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
        if not hasattr(teacher_config, "pad_token_id") or teacher_config.pad_token_id is None:
            teacher_config.pad_token_id = 0
        teacher = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="cpu",
            trust_remote_code=True,
            config=teacher_config,
        )

        lo_cfg = LoRAConfig(rank=LORA_RANK, num_steps=LORA_STEPS)
        model = finetune_lora(model, tokenizer, texts[:50], teacher, lo_cfg)

        del teacher
        import gc as _gc2
        _gc2.collect()

        model = keep_lora_separate(model)
        _save_lora_ckpt(model, "stage3_lora")

    lora_elapsed = time.time() - t0
    print(f"  LoRA done in {lora_elapsed:.1f}s")
else:
    print("\n[3/5] LoRA: SKIPPED (ENABLE_LORA=False)")

# ── Step 4: Export to INT2 format ────────────────────────────────────
print(f"\n[4/5] Exporting to {EXPORT_DIR}...")
t0 = time.time()

export_ternary_lora(model, tokenizer, EXPORT_DIR)
export_elapsed = time.time() - t0

# Measure disk size
total_bytes = 0
file_sizes = {}
for root, dirs, files in os.walk(EXPORT_DIR):
    for f in files:
        fp = os.path.join(root, f)
        sz = os.path.getsize(fp)
        total_bytes += sz
        file_sizes[f] = sz

print(f"  Export done in {export_elapsed:.1f}s")
print(f"  Total size: {total_bytes/1e6:.1f} MB ({total_bytes/1e9:.2f} GB)")
for fname, sz in sorted(file_sizes.items(), key=lambda x: -x[1]):
    print(f"    {fname}: {sz/1e6:.1f} MB")

# Compression ratio vs FP16
fp16_size = n_params * 2  # 2 bytes per param
ratio = fp16_size / total_bytes
print(
    f"  Compression: {fp16_size/1e9:.2f} GB (FP16) → {total_bytes/1e9:.2f} GB = {ratio:.1f}x"
)

# ── Step 5: Load back and verify ─────────────────────────────────────
print("\n[5/5] Loading exported model and verifying equivalence...")
t0 = time.time()

del model
gc.collect()
if has_cuda:
    torch.cuda.empty_cache()

model_loaded = load_ternary_lora(EXPORT_DIR, device=device, torch_dtype=dtype)
load_elapsed = time.time() - t0
print(f"  Loaded in {load_elapsed:.1f}s")

# Count TernaryInferenceLinear layers
ternary_count = sum(
    1 for m in model_loaded.modules() if isinstance(m, TernaryInferenceLinear)
)
print(f"  TernaryInferenceLinear layers: {ternary_count}")

# ── Quick PPL test ───────────────────────────────────────────────────
print("\n─ Quick PPL test (8 diverse excerpts) ─")
TEXTS = [
    "Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic scale. The wave function, represented by the Schrödinger equation, provides a mathematical description of the quantum state of a system. Unlike classical physics, quantum mechanics introduces fundamental uncertainty and probabilistic outcomes.",
    "The Renaissance was a period of European cultural, artistic, political and economic rebirth following the Middle Ages. Generally described as taking place from the 14th century to the 17th century, the Renaissance promoted the rediscovery of classical philosophy, literature and art.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Deep learning uses neural networks with many layers to progressively extract higher-level features from raw input.",
    "The Amazon rainforest, covering much of northwestern Brazil and extending into Colombia, Peru and other South American countries, is the world's largest tropical rainforest. It's famous for its biodiversity, with millions of species of plants, animals, and insects.",
    "Newton's laws of motion are three physical laws that together laid the foundation for classical mechanics. The first law states that an object at rest stays at rest and an object in motion stays in motion unless acted upon by an external force.",
    "In linguistics, syntax is the set of rules, principles and processes that govern the structure of sentences in a given language. Every language has its own syntactic rules that determine how words combine to form phrases and sentences.",
    "The human circulatory system consists of the heart, blood vessels, and blood. The heart pumps blood through arteries, veins, and capillaries, delivering oxygen and nutrients to tissues and removing carbon dioxide and other wastes.",
    "Game theory is the study of mathematical models of strategic interaction among rational decision-makers. It has applications in economics, political science, psychology, computer science and biology. The Nash equilibrium is a fundamental concept in non-cooperative game theory.",
]


def compute_ppl(model, tokenizer, texts, max_len=256):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, text in enumerate(texts[:8]):
            print(f"  Text {i+1}/8...", end=" ", flush=True)
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_len
            )
            input_ids = inputs["input_ids"]
            if has_cuda:
                input_ids = input_ids.to(device)
            if input_ids.numel() < 2:
                continue
            out = model(input_ids=input_ids, labels=input_ids)
            total_loss += out.loss.item() * (input_ids.numel() - 1)
            total_tokens += input_ids.numel() - 1
    return (
        math.exp(total_loss / max(total_tokens, 1))
        if total_tokens > 0
        else float("inf")
    )


ppl = compute_ppl(model_loaded, tokenizer, TEXTS)
print(f"  PPL (diverse texts): {ppl:.2f}")

# ── Quick generation test ────────────────────────────────────────────
print("\n─ Quick generation test ─")
PROMPTS = [
    "The capital of France is",
    "Water freezes at",
    "The largest planet in the solar system is",
]
for pi, prompt in enumerate(PROMPTS):
    inputs = tokenizer(prompt, return_tensors="pt")
    if has_cuda:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out_ids = model_loaded.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    print(f"  [{prompt}] → {response_text[len(prompt):].strip()[:80]}")

# ── INT2 weight integrity check ──────────────────────────────────────
print("\n─ INT2 roundtrip integrity ─")
# Verify that weights reconstructed from INT2 match the originals
# by checking a sample of ternary layers
total_weights = 0
max_error = 0.0
checked = 0
total_ternary = sum(1 for m in model_loaded.modules()
                    if isinstance(m, TernaryInferenceLinear))
for name, module in model_loaded.named_modules():
    if isinstance(module, TernaryInferenceLinear):
        checked += 1
        if checked % 20 == 0 or checked == total_ternary:
            print(f"  INT2 check {checked}/{total_ternary} layers...", flush=True)
        T = unpack_int2(module.int2_packed, module.in_features)
        # All values must be in {-1, 0, 1}
        invalid = (T != -1) & (T != 0) & (T != 1)
        if invalid.any():
            print(f"  FAIL: {name} has {invalid.sum().item()} invalid T values")
        total_weights += T.numel()
        # Check roundtrip
        repacked = pack_int2(T)
        error = (module.int2_packed - repacked).abs().max().item()
        max_error = max(max_error, error)

print(f"  Total ternary weights checked: {total_weights:,}")
print(f"  Max repack error: {max_error}")
print(f"  Integrity: {'OK' if max_error == 0 else 'FAIL'}")

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  PT-BitNet time:     {elapsed_pt/60:.1f} min")
if ENABLE_LORA:
    print(f"  LoRA time:          {lora_elapsed/60:.1f} min")
print(f"  Export time:        {export_elapsed:.1f}s")
print(f"  Load time:          {load_elapsed:.1f}s")
print(f"  PPL (diverse):      {ppl:.2f}")
print(f"  Export size:        {total_bytes/1e6:.1f} MB")
print(f"  Compression:        {ratio:.1f}x")
print(f"  Ternary layers:     {ternary_count}")
print(f"  INT2 integrity:     {'OK' if max_error == 0 else 'FAIL'}")
print(f"  LoRA separate:      {'yes' if ENABLE_LORA else 'n/a'}")

total_time = (elapsed_pt + lora_elapsed + export_elapsed + load_elapsed) / 60
print(f"\n  Total runtime: {total_time:.1f} min")
if resumed_stages:
    print(f"  Resumed stages: {', '.join(resumed_stages)}")
print("=" * 60)
