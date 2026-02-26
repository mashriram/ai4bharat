#!/usr/bin/env python3
"""
AI4BHARAT TUNE-ATHON — Upload Full Model

Merges LoRA adapter into full FP16 model and uploads to Hub.
Patches mlx_lm on disk to fix the Qwen3 num_layers bug (one-time, persists).

USAGE:
  python upload_model.py                        # auto-detect, merge, upload
  python upload_model.py --no-upload            # merge locally only
  python upload_model.py --iter 3400            # use specific checkpoint iter
  python upload_model.py --adapter-path PATH    # explicit adapter dir
  python upload_model.py --keep-merged          # don't delete merged/ after upload
"""

import os, sys, json, shutil, argparse, subprocess, site
from pathlib import Path

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

from huggingface_hub import HfApi, login

# ── Config ────────────────────────────────────────────────────────────
def _require(k):
    v = os.getenv(k, "").strip()
    if not v: print(f"\n❌  Missing .env: {k}\n"); sys.exit(1)
    return v
def _opt(k, d): return os.getenv(k, d).strip() or d

HF_TOKEN     = _require("HF_TOKEN")
HF_USERNAME  = _require("HF_USERNAME")
STATE        = _require("STATE")
PROJECT_NAME = _opt("PROJECT_NAME", "AI4Bharat-State-Expert")

MODEL_ID     = "mlx-community/Qwen3-1.7B-4bit"
MAX_SEQ      = 2048
RANK         = 64
ALPHA        = 128
LORA_MODULES = ["q_proj","k_proj","v_proj","o_proj",
                "gate_proj","up_proj","down_proj"]

CKPT_ADAPT = Path(f"checkpoints/{STATE}/adapters")
ADAPT_DIR  = Path(f"adapters/{STATE}")
MERGE_DIR  = Path(f"merged/{STATE}")

ADAPTER_CONFIG = {
    "alpha_pattern": {}, "auto_mapping": None,
    "base_model_name_or_path": MODEL_ID, "bias": "none",
    "fan_in_fan_out": False, "inference_mode": True, "init_lora_weights": True,
    "layer_replication": None, "layers_pattern": None, "layers_to_transform": None,
    "loftq_config": {}, "lora_alpha": ALPHA, "lora_dropout": 0.0,
    "megatron_config": None, "megatron_core": "megatron.core",
    "modules_to_save": None, "peft_type": "LORA", "r": RANK,
    "rank_pattern": {}, "revision": None, "target_modules": LORA_MODULES,
    "task_type": "CAUSAL_LM", "use_dora": False, "use_rslora": False,
}

# ── README ────────────────────────────────────────────────────────────
def make_readme(repo_id, iter_num):
    trained = f"iter {iter_num:,}" if isinstance(iter_num, int) else str(iter_num)
    return f"""---
language: [ta]
base_model: Qwen/Qwen3-1.7B
tags: [fine-tuned, indian-languages, {STATE.lower().replace('_','-')}, lora, ai4bharat]
license: apache-2.0
---
# AI4Bharat State Expert — {STATE.replace('_',' ')}
Fine-tuned Qwen3-1.7B on AI4Bharat Indic Languages dataset for {STATE.replace('_',' ')}.
Trained at SRMIST Vadapalani AI4Bharat Tune-Athon (Feb 26, 2026). Checkpoint: {trained}.

## Training
| Param | Value |
|---|---|
| Base | Qwen3-1.7B 4-bit | LoRA rank | {RANK} | Alpha | {ALPHA} |
| Modules | {', '.join(LORA_MODULES)} | Seq len | {MAX_SEQ} |
| Splits | indic + conv + cult | HW | Apple M3 iMac 16GB |

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tok = AutoTokenizer.from_pretrained("{repo_id}")
```
Merged FP16 model — no PEFT required.
"""

# ─────────────────────────────────────────────────────────────────────
# PATCH mlx_lm ON DISK
#
# mlx_lm/tuner/utils.py does: config.num_layers
# Qwen3's config.json uses "num_hidden_layers" — mlx_lm crashes.
# We patch the installed file once. It persists for the whole session
# and also fixes the subprocess call (which our in-process patches can't).
# ─────────────────────────────────────────────────────────────────────
def patch_mlx_lm_on_disk():
    """
    Find mlx_lm/tuner/utils.py in the active venv and patch the one line
    that crashes on Qwen3: `config.num_layers` → safe getattr fallback.
    Safe to run multiple times (checks if already patched).
    """
    # Find the file across all possible site-packages locations
    candidates = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        p = Path(sp) / "mlx_lm" / "tuner" / "utils.py"
        if p.exists():
            candidates.append(p)
    # Also check relative to the running python
    py = Path(sys.executable)
    for parent in [py.parent, py.parent.parent]:
        p = parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "mlx_lm" / "tuner" / "utils.py"
        if p.exists():
            candidates.append(p)
        # venv style
        for lib in parent.glob("lib/python*/site-packages/mlx_lm/tuner/utils.py"):
            candidates.append(lib)

    if not candidates:
        print("    ⚠️   Could not locate mlx_lm/tuner/utils.py — skipping disk patch")
        return False

    utils_file = candidates[0]
    src = utils_file.read_text()

    # Check if already patched
    if "num_hidden_layers" in src and "AI4BHARAT_PATCH" in src:
        print(f"    ✅  mlx_lm already patched")
        return True

    # The buggy line is: config.num_layers,
    # (inside _load_adapters / load_adapters)
    # We replace it with a safe getattr
    OLD = "config.num_layers,"
    NEW = "getattr(config,'num_layers',getattr(config,'num_hidden_layers',28)),  # AI4BHARAT_PATCH"

    if OLD not in src:
        # Already fixed upstream, or different version — check if it works
        print(f"    ℹ️   '{OLD}' not found in utils.py — may be already fixed or different version")
        return True

    patched = src.replace(OLD, NEW, 1)
    utils_file.write_text(patched)
    print(f"    ✅  Patched {utils_file}")
    return True


# ── Helpers ───────────────────────────────────────────────────────────
def ensure_adapter_config(adapter_dir: Path):
    cfg = adapter_dir / "adapter_config.json"
    if not cfg.exists():
        with open(cfg, "w") as f:
            json.dump(ADAPTER_CONFIG, f, indent=2)
        print(f"    📝  Generated adapter_config.json")
    else:
        print(f"    ✅  adapter_config.json present")


def find_adapter_dir(iter_override):
    if iter_override is not None:
        numbered = sorted(
            CKPT_ADAPT.glob("*_adapters.safetensors"),
            key=lambda f: int(f.stem.split("_")[0]) if f.stem.split("_")[0].isdigit() else 0
        )
        if not numbered:
            print(f"❌  No numbered checkpoints in {CKPT_ADAPT}/"); sys.exit(1)
        target = min(numbered, key=lambda f: abs(int(f.stem.split("_")[0]) - iter_override))
        actual = int(target.stem.split("_")[0])
        tmp = Path(f"checkpoints/{STATE}/_tmp_{actual}")
        tmp.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, tmp / "adapters.safetensors")
        print(f"📂  Checkpoint iter {actual:,}  →  {target.name}")
        return tmp, actual

    if ADAPT_DIR.exists() and any(ADAPT_DIR.glob("*.safetensors")):
        print(f"📂  Saved adapter: {ADAPT_DIR}/")
        return ADAPT_DIR, "saved"

    if CKPT_ADAPT.exists() and any(CKPT_ADAPT.glob("*.safetensors")):
        numbered = sorted(
            CKPT_ADAPT.glob("*_adapters.safetensors"),
            key=lambda f: int(f.stem.split("_")[0]) if f.stem.split("_")[0].isdigit() else 0
        )
        n = int(numbered[-1].stem.split("_")[0]) if numbered else "?"
        print(f"📂  Latest checkpoint: iter {n}")
        print(f"    {CKPT_ADAPT}/")
        return CKPT_ADAPT, n

    print(f"❌  No adapter found in {ADAPT_DIR}/ or {CKPT_ADAPT}/"); sys.exit(1)


def fuse_adapter(adapter_dir: Path) -> bool:
    MERGE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n🔀  Merging LoRA → FP16")
    print(f"    adapter : {adapter_dir}/")
    print(f"    output  : {MERGE_DIR}/")
    print(f"    (takes 2-5 min...)\n")

    # Patch the installed mlx_lm file on disk first
    # This fixes both the Python API and subprocess calls
    print(f"🔧  Patching mlx_lm for Qwen3 compatibility...")
    patch_mlx_lm_on_disk()

    # Run fuse via subprocess (clean process picks up the disk patch)
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

    print(f"\n    ❌  Merge failed. Run manually:")
    print(f"      python -m mlx_lm fuse \\")
    print(f"        --model {MODEL_ID} \\")
    print(f"        --adapter-path {adapter_dir} \\")
    print(f"        --save-path {MERGE_DIR} \\")
    print(f"        --dequantize")
    return False


def upload_merged(repo_id: str, iter_num) -> bool:
    print(f"\n📤  Uploading → {repo_id}  (~1GB, few minutes...)")
    try:
        api = HfApi()
        print(f"    Creating repo...")
        api.create_repo(repo_id=repo_id, repo_type="model",
                        token=HF_TOKEN, exist_ok=True)
        (MERGE_DIR / "README.md").write_text(make_readme(repo_id, iter_num))
        with open(MERGE_DIR / "adapter_config.json", "w") as f:
            json.dump(ADAPTER_CONFIG, f, indent=2)
        api.upload_folder(folder_path=str(MERGE_DIR), repo_id=repo_id,
                          repo_type="model", token=HF_TOKEN)
        print(f"    🎉  https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"    ❌  Upload failed: {e}")
        print(f"    Model saved at {MERGE_DIR}/ — retry: python upload_model.py --adapter-path {MERGE_DIR}")
        return False


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-upload",    action="store_true")
    parser.add_argument("--iter",         type=int,  default=None)
    parser.add_argument("--adapter-path", type=str,  default=None)
    parser.add_argument("--keep-merged",  action="store_true")
    args = parser.parse_args()

    repo_id = f"{HF_USERNAME}/{PROJECT_NAME}-{STATE}"

    print(f"\n{'═'*64}")
    print(f"  🚀  Upload Full Model  —  {STATE}")
    print(f"{'─'*64}")
    print(f"  Base   : {MODEL_ID}")
    print(f"  Output : {MERGE_DIR}/")
    if not args.no_upload:
        print(f"  Hub    : https://huggingface.co/{repo_id}")
    print(f"{'═'*64}\n")

    if args.adapter_path:
        adapter_dir = Path(args.adapter_path)
        if not adapter_dir.exists():
            print(f"❌  Not found: {adapter_dir}"); sys.exit(1)
        print(f"📂  Using: {adapter_dir}/")
        iter_num = "custom"
    else:
        adapter_dir, iter_num = find_adapter_dir(args.iter)

    print(f"\n🔧  Checking adapter_config.json...")
    ensure_adapter_config(adapter_dir)

    if not fuse_adapter(adapter_dir):
        sys.exit(1)

    # Clean temp dir if created for --iter
    if args.iter:
        tmp = Path(f"checkpoints/{STATE}/_tmp_{args.iter}")
        shutil.rmtree(tmp, ignore_errors=True)

    if args.no_upload:
        print(f"\n✅  Merge done. Skipped upload (--no-upload).")
        print(f"📁  {MERGE_DIR}/")
        return

    login(token=HF_TOKEN)
    ok = upload_merged(repo_id, iter_num)

    if ok and not args.keep_merged:
        shutil.rmtree(str(MERGE_DIR), ignore_errors=True)
        print(f"    🧹  Local merged dir cleaned")

    print(f"\n{'═'*64}")
    print(f"  🏁  {'Done!' if ok else 'Upload failed — see above'}")
    if ok: print(f"  🔗  https://huggingface.co/{repo_id}")
    print(f"{'═'*64}\n")


if __name__ == "__main__":
    main()
