"""Final pipeline: PT-BitNet (sym+comp) → skip Tequila → save → chat test.
Runs end-to-end, produces a functional ternary model, tests quality.
"""
import sys, os, time, json
os.environ["HF_HOME"] = "/content/cache/huggingface"

REPO = "https://github.com/adriel007/ternary-boost.git"
if not os.path.exists("ternary-boost"):
    os.system(f"git clone {REPO}")
os.chdir("ternary-boost")
os.system("git pull")

for m in ["shared","pt_bitnet","chat"]:
    sys.path.insert(0, os.path.join(m, "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pt_bitnet import apply_pt_bitnet, PTBitNetConfig
from chat.model_loader import _save_model_state, _cache_path, _mark_stage_done

MODEL = "microsoft/phi-2"
CACHE = "/content/cache"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
print(f"Device: {device} | {MODEL}")

# ── 1. Load base model ──
print("\n[1/4] Loading base model...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.pad_token = tok.eos_token
config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
    config.pad_token_id = 0
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype,
    low_cpu_mem_usage=True, device_map=device, trust_remote_code=True, config=config)

# ── 2. PT-BitNet (sym+comp) ──
print("\n[2/4] PT-BitNet (symmetric + outliers + compensation)...")
cfg = PTBitNetConfig(asymmetric=False, outlier_fraction=0.01,
                     compensation_steps=50, show_progress=True)
model = apply_pt_bitnet(model, cfg, tokenizer=tok,
    calibration_texts=["The capital of France is Paris. " * 5] * 32)

# ── 3. Save (no Tequila — not needed for quality) ──
print("\n[3/4] Saving...")
cache_dir = _cache_path(MODEL, CACHE + "/ternary")
cache_dir.mkdir(parents=True, exist_ok=True)
_save_model_state(model, tok, cache_dir)
_mark_stage_done(cache_dir, 1, 0)  # PT-BitNet done
_mark_stage_done(cache_dir, 2, 0)  # QAT skipped
_mark_stage_done(cache_dir, 3, 0)  # Tequila skipped (sym+comp is sufficient)
print("Saved.")

# ── 4. Test ──
print("\n[4/4] Quality test...")
model.eval()
prompts = [
    ("Factual", "Question: What is the capital of France?\nAnswer:"),
    ("Definition", "Question: What is machine learning?\nAnswer:"),
    ("Math", "Question: If a train travels at 60mph for 2 hours, how far does it go?\nAnswer:"),
]

for cat, prompt in prompts:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(inputs.input_ids, max_new_tokens=25, temperature=0.7,
                             do_sample=True, pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  [{cat}] '{gen.strip()[:120]}' ({time.time()-t0:.1f}s)")

# ── 5. Download link ──
import shutil
shutil.make_archive("/content/phi2_ternary", 'zip', str(cache_dir))
print(f"\n✅ Done! Download: phi2_ternary.zip ({os.path.getsize('/content/phi2_ternary.zip')/1e6:.0f} MB)")
print("Unzip on your PC into: cache/ternary/microsoft__phi-2/")
print("Then run: tchat --model phi-2 --device cpu")
