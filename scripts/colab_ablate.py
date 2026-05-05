"""Colab ablation: which step degrades quality? Runs end-to-end, no cache."""
import sys, os, time, json, gc, math
os.environ["HF_HOME"] = "/content/cache/huggingface"

REPO = "https://github.com/adriel007/ternary-boost.git"
if not os.path.exists("ternary-boost"):
    os.system(f"git clone {REPO}")
os.chdir("ternary-boost")
os.system("git pull")

for m in ["shared","pt_bitnet","tequila"]:
    sys.path.insert(0, os.path.join(m, "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pt_bitnet import apply_pt_bitnet, PTBitNetConfig

MODEL = "microsoft/phi-2"
PROMPT = "Question: What is the capital of France?\nAnswer:"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32


def repetition_ratio(text):
    """Detects model degeneration. 0=healthy, 1=degenerate/gibberish."""
    words = text.split()
    if len(words) < 5:
        return 0.0
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    if len(trigrams) < 2:
        return 0.0
    return 1.0 - len(set(trigrams)) / len(trigrams)


def compute_perplexity(model, tokenizer, texts, max_length=64):
    """Standard perplexity on calibration texts. Lower = better."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts[:8]:
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


def load_base():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = 0
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype,
        low_cpu_mem_usage=True, device_map=device, trust_remote_code=True,
        config=config)
    return model, tok


def test(model, tok, label):
    model.eval()
    inputs = tok(PROMPT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(inputs.input_ids, max_new_tokens=20, temperature=0.7,
                             do_sample=True, pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    rep = repetition_ratio(gen)
    has_paris = "paris" in gen.lower() and "france" in gen.lower()
    w = next(p for n,p in model.named_parameters() if "q_proj.weight" in n)
    uniq = w[0].unique().numel()
    status = "OK" if has_paris else ("POOR" if rep > 0.5 else "WEAK")
    print(f"  [{label}] rep={rep:.2f} rows~{uniq}uniq {status}")
    print(f"    '{gen.strip()[:120]}'")
    return gen, rep, has_paris


calib_texts = ["The capital of France is Paris. " * 5] * 8

print(f"Device: {device} | Dtype: {dtype}\n")
results = {}

# ── 1. BASE ──
print("="*50 + "\n1. BASELINE FP16\n" + "="*50)
m, t = load_base()
ppl_base = compute_perplexity(m, t, calib_texts)
gen, rep, ok = test(m, t, "base")
results["base"] = {"gen": gen.strip(), "rep": rep, "ok": ok, "ppl": ppl_base}
print(f"  PPL={ppl_base:.1f}")
del m; gc.collect(); torch.cuda.empty_cache()

# ── 2. SYMMETRIC TERNARY ONLY ──
print("\n" + "="*50 + "\n2. Symmetric ternary (no ITF, no outliers, no comp)\n" + "="*50)
m, t = load_base()
cfg2 = PTBitNetConfig(asymmetric=False, outlier_fraction=0.0,
                       compensation_steps=0, show_progress=False)
m = apply_pt_bitnet(m, cfg2)
ppl = compute_perplexity(m, t, calib_texts)
gen, rep, ok = test(m, t, "sym")
results["symmetric"] = {"gen": gen.strip(), "rep": rep, "ok": ok, "ppl": ppl}
print(f"  PPL={ppl:.1f} (ratio={ppl/ppl_base:.2f}x)")
del m; gc.collect(); torch.cuda.empty_cache()

# ── 3. SYMMETRIC + OUTLIERS + COMPENSATION ──
print("\n" + "="*50 + "\n3. Symmetric + outliers + lm_head compensation\n" + "="*50)
m, t = load_base()
cfg3 = PTBitNetConfig(asymmetric=False, outlier_fraction=0.01,
                       compensation_steps=50, show_progress=False)
m = apply_pt_bitnet(m, cfg3, tokenizer=t, calibration_texts=calib_texts)
ppl = compute_perplexity(m, t, calib_texts)
gen, rep, ok = test(m, t, "sym+comp")
results["sym+comp"] = {"gen": gen.strip(), "rep": rep, "ok": ok, "ppl": ppl}
print(f"  PPL={ppl:.1f} (ratio={ppl/ppl_base:.2f}x)")
del m; gc.collect(); torch.cuda.empty_cache()

# ── 4. ITF TERNARY ONLY ──
print("\n" + "="*50 + "\n4. ITF asymmetric ternary (no outliers, no comp)\n" + "="*50)
m, t = load_base()
cfg4 = PTBitNetConfig(asymmetric=True, outlier_fraction=0.0,
                       compensation_steps=0, show_progress=False)
m = apply_pt_bitnet(m, cfg4)
ppl = compute_perplexity(m, t, calib_texts)
gen, rep, ok = test(m, t, "itf")
results["itf"] = {"gen": gen.strip(), "rep": rep, "ok": ok, "ppl": ppl}
print(f"  PPL={ppl:.1f} (ratio={ppl/ppl_base:.2f}x)")
del m; gc.collect(); torch.cuda.empty_cache()

# ── 5. ITF + OUTLIERS + COMPENSATION ──
print("\n" + "="*50 + "\n5. ITF + outliers + lm_head compensation\n" + "="*50)
m, t = load_base()
cfg5 = PTBitNetConfig(asymmetric=True, outlier_fraction=0.01,
                       compensation_steps=50, show_progress=False)
m = apply_pt_bitnet(m, cfg5, tokenizer=t, calibration_texts=calib_texts)
ppl = compute_perplexity(m, t, calib_texts)
gen, rep, ok = test(m, t, "itf+comp")
results["itf+comp"] = {"gen": gen.strip(), "rep": rep, "ok": ok, "ppl": ppl}
print(f"  PPL={ppl:.1f} (ratio={ppl/ppl_base:.2f}x)")
del m; gc.collect(); torch.cuda.empty_cache()

# ── 6. SYMMETRIC + TEQUILA + BAKE (full pipeline, no save/load) ──
print("\n" + "="*50 + "\n6. Symmetric + Tequila + Bake (no save/load)\n" + "="*50)
m, t = load_base()
cfg6 = PTBitNetConfig(asymmetric=False, outlier_fraction=0.01,
                       compensation_steps=0, show_progress=False)
m = apply_pt_bitnet(m, cfg6)
from tequila.ultraquant import apply_tequila, TequilaConfig
from torch.utils.data import DataLoader

class DS(torch.utils.data.Dataset):
    def __init__(self, texts, tok):
        self.d = [tok(x, truncation=True, max_length=64, return_tensors="pt") for x in texts]
    def __len__(self): return len(self.d)
    def __getitem__(self, i): return self.d[i]

texts = ["The capital of France is Paris. " * 5] * 30
dl = DataLoader(DS(texts, t), batch_size=1, shuffle=True,
    collate_fn=lambda b: {"input_ids": torch.cat([x["input_ids"] for x in b])})
m = apply_tequila(m, dl, TequilaConfig(quant_method="ultraquantv3",
    num_epochs=1, lambada_granularity="per_channel"))
from chat.model_loader import _bake_ultraquant_to_linear
_bake_ultraquant_to_linear(m)
ppl = compute_perplexity(m, t, calib_texts)
gen, rep, ok = test(m, t, "sym+teq+bake")
results["sym+teq+bake"] = {"gen": gen.strip(), "rep": rep, "ok": ok, "ppl": ppl}
print(f"  PPL={ppl:.1f} (ratio={ppl/ppl_base:.2f}x)")
del m; gc.collect(); torch.cuda.empty_cache()

# ── SUMMARY ──
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"{'Variant':15s} {'PPL':>7s} {'Ratio':>7s} {'Rep':>5s} {'OK':>5s}  Response")
print("-" * 65)
for k, v in results.items():
    ratio = v["ppl"] / ppl_base
    print(f"  {k:13s} {v['ppl']:7.1f} {ratio:6.2f}x {v['rep']:5.2f} {'YES' if v['ok'] else 'NO':>5s}  {v['gen'][:70]}")
