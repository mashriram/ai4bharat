#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════╗
║   AI4BHARAT TUNE-ATHON  —  UPLOAD FULL MODEL                        ║
║                                                                       ║
║   Merges LoRA adapter into full FP16 model and uploads to Hub.       ║
║   Works from: adapters/{STATE}/  OR  checkpoints/{STATE}/adapters/   ║
║                                                                       ║
║   USAGE:                                                              ║
║     # Auto-detect adapter, merge, upload:                            ║
║     python upload_model.py                                           ║
║                                                                       ║
║     # Use a specific adapter directory:                              ║
║     python upload_model.py --adapter-path adapters/Tamil_Nadu        ║
║                                                                       ║
║     # Use a specific numbered checkpoint (not the rolling latest):   ║
║     python upload_model.py --iter 3400                               ║
║                                                                       ║
║     # Merge only, don't upload (check it first):                     ║
║     python upload_model.py --no-upload                               ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

import os, sys, json, shutil, argparse, subprocess
from pathlib import Path

# ── Load .env ────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from huggingface_hub import HfApi, login

# ── Config — must match finetune.py exactly ───────────────────────────
def _require(k):
    v = os.getenv(k, "").strip()
    if not v:
        print(f"\n❌  Missing .env variable: {k}\n")
        sys.exit(1)
    return v

def _opt(k, d):
    return os.getenv(k, d).strip() or d

HF_TOKEN     = _require("HF_TOKEN")
HF_USERNAME  = _require("HF_USERNAME")
STATE        = _require("STATE")
PROJECT_NAME = _opt("PROJECT_NAME", "AI4Bharat-State-Expert")

MODEL_ID     = "mlx-community/Qwen3-1.7B-4bit"
MAX_SEQ      = 2048
RANK         = 64
ALPHA        = 128
LORA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

# ── Paths ─────────────────────────────────────────────────────────────
CKPT_ADAPT = Path(f"checkpoints/{STATE}/adapters")   # mid-run checkpoints
ADAPT_DIR  = Path(f"adapters/{STATE}")               # clean saved adapter
MERGE_DIR  = Path(f"merged/{STATE}")                 # fuse output

# ── adapter_config.json ───────────────────────────────────────────────
# unsloth-mlx only writes this at end of full training.
# Mid-run checkpoint dirs only have raw .safetensors files.
# mlx_lm.fuse needs this file — we generate it from our known params.
ADAPTER_CONFIG = {
    "alpha_pattern":           {},
    "auto_mapping":            None,
    "base_model_name_or_path": MODEL_ID,
    "bias":                    "none",
    "fan_in_fan_out":          False,
    "inference_mode":          True,
    "init_lora_weights":       True,
    "layer_replication":       None,
    "layers_pattern":          None,
    "layers_to_transform":     None,
    "loftq_config":            {},
    "lora_alpha":              ALPHA,
    "lora_dropout":            0.0,
    "megatron_config":         None,
    "megatron_core":           "megatron.core",
    "modules_to_save":         None,
    "peft_type":               "LORA",
    "r":                       RANK,
    "rank_pattern":            {},
    "revision":                None,
    "target_modules":          LORA_MODULES,
    "task_type":               "CAUSAL_LM",
    "use_dora":                False,
    "use_rslora":              False,
}

# ── README template ───────────────────────────────────────────────────
def make_readme(repo_id, iter_num):
    trained_on = f"iter {iter_num:,}" if isinstance(iter_num, int) else "full training"
    return f"""---
language:
- ta
base_model: Qwen/Qwen3-1.7B
tags:
- fine-tuned
- indian-languages
- {STATE.lower().replace("_", "-")}
- lora
- ai4bharat
- tune-athon
license: apache-2.0
---

# AI4Bharat State Expert — {STATE.replace("_", " ")}

Fine-tuned from [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}) on the
[AI4Bharat Indic Languages and Cultures](https://huggingface.co/datasets/mashriram/AI4Bharat-Indic-Languages-and-Cultures)
dataset for **{STATE.replace("_", " ")}**, as part of the
AI4Bharat Tune-Athon event at SRMIST Vadapalani (Feb 26, 2026).

Checkpoint: {trained_on}.

## Training details
| Parameter | Value |
|---|---|
| Base model | Qwen3-1.7B (4-bit quantized) |
| LoRA rank | {RANK} |
| LoRA alpha | {ALPHA} |
| Target modules | {", ".join(LORA_MODULES)} |
| Max sequence length | {MAX_SEQ} |
| Dataset splits | indic + conv + cult |
| Framework | unsloth-mlx |
| Hardware | Apple M3 iMac 16GB |

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tok   = AutoTokenizer.from_pretrained("{repo_id}")

prompt = "<|im_start|>user\\n{STATE.replace('_',' ')} பற்றி சொல்லுங்கள்<|im_end|>\\n<|im_start|>assistant\\n"
inputs = tok(prompt, return_tensors="pt")
out    = model.generate(**inputs, max_new_tokens=200)
print(tok.decode(out[0], skip_special_tokens=True))
```

This is the **merged FP16 model** — no PEFT library required.
"""

# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def ensure_adapter_config(adapter_dir: Path):
    """Write adapter_config.json if missing. Required by mlx_lm.fuse."""
    cfg = adapter_dir / "adapter_config.json"
    if not cfg.exists():
        with open(cfg, "w") as f:
            json.dump(ADAPTER_CONFIG, f, indent=2)
        print(f"    📝  Generated adapter_config.json")
    else:
        print(f"    ✅  adapter_config.json present")


def find_adapter_dir(iter_override: int | None) -> tuple[Path, int | str]:
    """
    Locate the adapter directory to use.
    Returns (path, iter_num_or_string).

    Priority:
      1. --iter N  → use checkpoints/{STATE}/adapters/00N_adapters.safetensors
                     copied to a temp dir so fuse gets a clean single file
      2. adapters/{STATE}/  (clean saved adapter from previous save_checkpoint run)
      3. checkpoints/{STATE}/adapters/  (live rolling latest from training)
    """
    if iter_override is not None:
        # Find the specific numbered checkpoint file
        pattern = f"{iter_override:07d}_adapters.safetensors"
        target  = CKPT_ADAPT / pattern
        if not target.exists():
            # Try finding closest match
            all_numbered = sorted(
                CKPT_ADAPT.glob("*_adapters.safetensors"),
                key=lambda f: int(f.stem.split("_")[0])
                              if f.stem.split("_")[0].isdigit() else 0
            )
            if not all_numbered:
                print(f"❌  No numbered checkpoints found in {CKPT_ADAPT}/")
                sys.exit(1)
            # Pick closest
            target = min(all_numbered,
                         key=lambda f: abs(int(f.stem.split("_")[0]) - iter_override))
            actual = int(target.stem.split("_")[0])
            print(f"⚠️   iter {iter_override} not found — using closest: iter {actual:,}")
            iter_override = actual

        # Copy the numbered file as adapters.safetensors into a temp dir
        # alongside adapter_config.json so fuse can load it
        tmp_dir = Path(f"checkpoints/{STATE}/_tmp_iter_{iter_override}")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, tmp_dir / "adapters.safetensors")
        print(f"📂  Using checkpoint iter {iter_override:,}")
        print(f"    {target}")
        return tmp_dir, iter_override

    if ADAPT_DIR.exists() and any(ADAPT_DIR.glob("*.safetensors")):
        print(f"📂  Using saved adapter: {ADAPT_DIR}/")
        return ADAPT_DIR, "saved adapter"

    if CKPT_ADAPT.exists() and any(CKPT_ADAPT.glob("*.safetensors")):
        numbered = sorted(
            CKPT_ADAPT.glob("*_adapters.safetensors"),
            key=lambda f: int(f.stem.split("_")[0])
                          if f.stem.split("_")[0].isdigit() else 0
        )
        iter_num = int(numbered[-1].stem.split("_")[0]) if numbered else "?"
        print(f"📂  Using latest checkpoint: iter {iter_num:,}")
        print(f"    {CKPT_ADAPT}/")
        return CKPT_ADAPT, iter_num

    print(f"\n❌  No adapter found.")
    print(f"    Looked in:")
    print(f"      {ADAPT_DIR}/")
    print(f"      {CKPT_ADAPT}/")
    sys.exit(1)


def fuse_adapter(adapter_dir: Path) -> bool:
    """
    Run mlx_lm.fuse to merge LoRA weights into base model → MERGE_DIR.
    Uses --dequantize (NOT --de-quantize) for FP16 output.
    Returns True on success.
    """
    MERGE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n🔀  Merging LoRA → FP16 via mlx_lm fuse")
    print(f"    adapter : {adapter_dir}/")
    print(f"    output  : {MERGE_DIR}/")
    print(f"    (takes 2-5 min...)\n")

    # mlx_lm.fuse is NOT a callable function — it's a CLI module.
    # The correct invocation is: python -m mlx_lm fuse  (space, not dot)
    # Note: --dequantize (no hyphen between de and quantize)
    result = subprocess.run(
        [sys.executable, "-m", "mlx_lm", "fuse",
         "--model",        MODEL_ID,
         "--adapter-path", str(adapter_dir),
         "--save-path",    str(MERGE_DIR),
         "--dequantize"],
        capture_output=False,
    )

    if result.returncode == 0:
        print(f"\n    ✅  Merge complete")
        return True

    print(f"\n    ❌  Merge failed.")
    print(f"    Run manually:")
    print(f"      python -m mlx_lm fuse \\")
    print(f"        --model {MODEL_ID} \\")
    print(f"        --adapter-path {adapter_dir} \\")
    print(f"        --save-path {MERGE_DIR} \\")
    print(f"        --dequantize")
    return False


def upload_merged(repo_id: str, iter_num) -> bool:
    """Upload MERGE_DIR contents to Hub. Returns True on success."""
    print(f"\n📤  Uploading merged model → {repo_id}")
    print(f"    (uploading ~1GB, takes a few minutes...)")
    try:
        api = HfApi()

        # Always create repo first — harmless if it already exists.
        # Required for org repos (srmsit/...) which don't auto-create.
        print(f"    Creating repo if needed...")
        api.create_repo(repo_id=repo_id, repo_type="model",
                        token=HF_TOKEN, exist_ok=True)
        print(f"    ✅  Repo ready")

        # Write README into MERGE_DIR before upload
        readme = MERGE_DIR / "README.md"
        readme.write_text(make_readme(repo_id, iter_num))

        # Write adapter_config.json into MERGE_DIR too
        cfg = MERGE_DIR / "adapter_config.json"
        with open(cfg, "w") as f:
            json.dump(ADAPTER_CONFIG, f, indent=2)

        api.upload_folder(
            folder_path=str(MERGE_DIR),
            repo_id=repo_id,
            repo_type="model",
            token=HF_TOKEN,
        )
        print(f"    🎉  https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"    ❌  Upload failed: {e}")
        print(f"    Merged model is at {MERGE_DIR}/ — retry when network is available.")
        return False


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter and upload full model to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python upload_model.py                        # auto-detect latest, merge, upload
  python upload_model.py --no-upload            # merge only, check before uploading  
  python upload_model.py --iter 3400            # use iter 3400 checkpoint specifically
  python upload_model.py --adapter-path adapters/Tamil_Nadu  # use explicit path
        """,
    )
    parser.add_argument("--no-upload",    action="store_true",
                        help="Merge only — don't upload to Hub")
    parser.add_argument("--iter",         type=int, default=None,
                        help="Use a specific numbered checkpoint (e.g. --iter 3400)")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Explicit adapter directory path")
    parser.add_argument("--keep-merged",  action="store_true",
                        help="Don't delete local merged dir after upload")
    args = parser.parse_args()

    repo_id = f"{HF_USERNAME}/{PROJECT_NAME}-{STATE}"

    print(f"\n{'═'*64}")
    print(f"  🚀  Upload Full Model  —  {STATE}")
    print(f"{'─'*64}")
    print(f"  Base      : {MODEL_ID}")
    print(f"  Output    : {MERGE_DIR}/")
    if not args.no_upload:
        print(f"  Hub       : https://huggingface.co/{repo_id}")
    print(f"{'═'*64}\n")

    # ── 1. Find adapter dir ───────────────────────────────────
    if args.adapter_path:
        adapter_dir = Path(args.adapter_path)
        if not adapter_dir.exists():
            print(f"❌  Path not found: {adapter_dir}")
            sys.exit(1)
        print(f"📂  Using: {adapter_dir}/")
        iter_num = "custom path"
    else:
        adapter_dir, iter_num = find_adapter_dir(args.iter)

    # ── 2. Ensure adapter_config.json ────────────────────────
    print(f"\n🔧  Checking adapter_config.json...")
    ensure_adapter_config(adapter_dir)

    # ── 3. Merge ──────────────────────────────────────────────
    if not fuse_adapter(adapter_dir):
        sys.exit(1)

    # ── 4. Clean tmp dir if we created one ───────────────────
    tmp_dir = Path(f"checkpoints/{STATE}/_tmp_iter_{args.iter}")
    if args.iter and tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── 5. Upload ─────────────────────────────────────────────
    if args.no_upload:
        print(f"\n{'═'*64}")
        print(f"  ✅  Merge done  (--no-upload: skipped Hub upload)")
        print(f"  📁  {MERGE_DIR}/")
        print(f"\n  To upload:")
        print(f"    python upload_model.py --adapter-path {adapter_dir}")
        print(f"{'═'*64}\n")
        return

    login(token=HF_TOKEN)
    ok = upload_merged(repo_id, iter_num)

    # ── 6. Clean up merged dir ────────────────────────────────
    if ok and not args.keep_merged:
        shutil.rmtree(str(MERGE_DIR), ignore_errors=True)
        print(f"    🧹  Local merged dir cleaned")

    print(f"\n{'═'*64}")
    if ok:
        print(f"  🏁  Done!")
        print(f"  🔗  https://huggingface.co/{repo_id}")
    else:
        print(f"  ⚠️   Upload failed — merged model is at {MERGE_DIR}/")
        print(f"  Retry: python upload_model.py --adapter-path {MERGE_DIR}")
    print(f"{'═'*64}\n")


if __name__ == "__main__":
    main()
