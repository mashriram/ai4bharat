#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════╗
║       AI4BHARAT TUNE-ATHON  —  UPLOAD TO HUB  v1.0                  ║
║                                                                       ║
║  Run this AFTER finetune.py has finished training.                   ║
║  Team Member 2 sets their own credentials in upload.env              ║
║  and uploads the trained adapter to their own HuggingFace Hub.       ║
║                                                                       ║
║  SETUP:                                                               ║
║    cp upload.env.example upload.env                                  ║
║    # Edit upload.env with YOUR credentials                           ║
║                                                                       ║
║  USAGE:                                                               ║
║                                                                       ║
║  Option A — Push adapter only (fast, ~1 min):                        ║
║    python upload_to_hub.py --adapter-only                             ║
║                                                                       ║
║  Option B — Merge LoRA into full model, then push (recommended):     ║
║    python upload_to_hub.py                                            ║
║                                                                       ║
║  Option C — Push from a custom adapter path:                         ║
║    python upload_to_hub.py --adapter-path /path/to/adapters/Kerala   ║
╚═══════════════════════════════════════════════════════════════════════╝

WHAT THIS DOES:
  1. Reads YOUR credentials from upload.env (separate from finetune.env)
  2. Loads the trained LoRA adapter from adapters/{STATE}/
  3. Merges it into the Qwen3-1.7B base model (full FP16)
  4. Uploads the merged model to YOUR HuggingFace Hub repo
  5. If merge fails: falls back to pushing the adapter only

WHAT YOU NEED ON THIS MACHINE:
  - adapters/{STATE}/  folder (copied from the training machine, or
    directly available if running on the same machine after training)
  - unsloth-mlx installed  (same setup as finetune.py)
  - YOUR HuggingFace WRITE token in upload.env
"""

import os, sys, shutil, argparse
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# LOAD upload.env  (separate from the training .env)
# ──────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    # Prefer upload.env — falls back to .env if upload.env doesn't exist
    if Path("upload.env").exists():
        load_dotenv("upload.env")
        print("📄  Loaded credentials from upload.env")
    else:
        load_dotenv()
        print("📄  upload.env not found — loaded from .env")
except ImportError:
    pass  # env vars must be set manually

from huggingface_hub import login, HfApi

# ──────────────────────────────────────────────────────────────────────
# CONFIG  (all from upload.env)
# ──────────────────────────────────────────────────────────────────────
def _require(key):
    v = os.getenv(key, "").strip()
    if not v:
        print(f"\n❌  Missing required variable: {key}")
        print(f"    Add it to upload.env:  {key}=<value>\n")
        sys.exit(1)
    return v

def _optional(key, default):
    return os.getenv(key, default).strip() or default

HF_TOKEN     = _require("HF_TOKEN")
HF_USERNAME  = _require("HF_USERNAME")
STATE        = _require("STATE")
PROJECT_NAME = _optional("PROJECT_NAME", "AI4Bharat-State-Expert")

# The base model that was used for training — must match finetune.py
BASE_MODEL   = "mlx-community/Qwen3-1.7B-4bit"
MAX_SEQ      = 2048

# ──────────────────────────────────────────────────────────────────────
# PATHS  (must match finetune.py layout)
# ──────────────────────────────────────────────────────────────────────
DEFAULT_ADAPT_DIR = Path(f"adapters/{STATE}")
MERGE_DIR         = Path(f"merged_upload/{STATE}")   # separate from training merge dir

# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────
def verify_adapter(adapter_path: Path):
    """Check that the adapter directory has the expected files."""
    required = ["adapters.safetensors", "adapter_config.json"]
    missing  = [f for f in required if not (adapter_path / f).exists()]

    # Some versions save directly in the dir, others in an 'adapters/' subdir
    subdir = adapter_path / "adapters"
    if missing and subdir.exists():
        missing = [f for f in required if not (subdir / f).exists()]
        if not missing:
            return subdir   # return the actual location

    if missing:
        print(f"\n❌  Adapter directory incomplete: {adapter_path}")
        print(f"    Missing files: {missing}")
        print(f"    Make sure training completed successfully and")
        print(f"    the adapters/ folder is on this machine.")
        sys.exit(1)

    return adapter_path

def push_adapter_only(adapter_path: Path, repo_id: str):
    """Push raw LoRA adapter files to Hub. Fast (~1 min)."""
    print(f"\n📤  Pushing LoRA adapter files → {repo_id}")
    print(f"    From: {adapter_path}/")
    try:
        api = HfApi()
        api.upload_folder(
            folder_path = str(adapter_path),
            repo_id     = repo_id,
            repo_type   = "model",
            token       = HF_TOKEN,
        )
        print(f"    ✅  Adapter live: https://huggingface.co/{repo_id}")
        print(f"\n    ℹ️   To use this adapter:")
        print(f"         from peft import PeftModel")
        print(f"         from transformers import AutoModelForCausalLM")
        print(f"         base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B')")
        print(f"         model = PeftModel.from_pretrained(base, '{repo_id}')")
        return True
    except Exception as e:
        print(f"    ❌  Upload failed: {e}")
        return False

def merge_and_push(adapter_path: Path, repo_id: str):
    """
    Load base model + adapter, merge weights into full FP16 model, push.
    This produces a standalone model — no PEFT library needed to run it.
    Takes ~10-20 min depending on upload speed.
    """
    from unsloth_mlx import FastLanguageModel

    print(f"\n📥  Loading base model: {BASE_MODEL}")
    model, tok = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL,
        max_seq_length = MAX_SEQ,
        load_in_4bit   = True,
    )

    print(f"🔌  Loading LoRA adapter from: {adapter_path}/")
    model.load_adapter(str(adapter_path))

    # Merge LoRA weights permanently into base model
    MERGE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n🔀  Merging LoRA → FP16 (this takes a few minutes)...")
    try:
        model.save_pretrained_merged(
            str(MERGE_DIR), tok, save_method="merged_16bit"
        )
        print(f"    ✅  Merged model saved → {MERGE_DIR}/")
    except Exception as e:
        print(f"    ❌  Merge failed: {e}")
        print(f"    Falling back to adapter-only upload...")
        shutil.rmtree(str(MERGE_DIR), ignore_errors=True)
        return False

    # Upload merged model
    print(f"\n📤  Uploading merged model → {repo_id}")
    try:
        api = HfApi()
        api.upload_folder(
            folder_path = str(MERGE_DIR),
            repo_id     = repo_id,
            repo_type   = "model",
            token       = HF_TOKEN,
        )
        print(f"    🎉  Full model live: https://huggingface.co/{repo_id}")
        # Clean up local merged dir to save disk space
        shutil.rmtree(str(MERGE_DIR), ignore_errors=True)
        print(f"    🧹  Cleaned up local merged dir")
        return True
    except Exception as e:
        print(f"    ❌  Upload failed: {e}")
        print(f"    Merged model is still saved at {MERGE_DIR}/")
        print(f"    You can retry the upload manually:")
        print(f"      huggingface-cli upload {repo_id} {MERGE_DIR}/ --repo-type model")
        return False

# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="AI4Bharat Tune-Athon: Upload trained model to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python upload_to_hub.py                             # merge + push full model
  python upload_to_hub.py --adapter-only              # push adapter only (fast)
  python upload_to_hub.py --adapter-path adapters/Kerala  # custom adapter path
        """,
    )
    parser.add_argument(
        "--adapter-only",
        action="store_true",
        help="Push raw LoRA adapter files only (fast, ~1 min). "
             "Recipient needs PEFT to use it.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help=f"Path to adapter directory (default: adapters/{STATE})",
    )
    args = parser.parse_args()

    # Resolve adapter path
    adapter_path = Path(args.adapter_path) if args.adapter_path else DEFAULT_ADAPT_DIR

    # Banner
    print(f"\n{'═'*64}")
    print(f"  📤  AI4Bharat Tune-Athon  —  Upload to Hub")
    print(f"{'─'*64}")
    print(f"  State        : {STATE}")
    print(f"  Adapter path : {adapter_path}/")
    repo_id = f"{HF_USERNAME}/{PROJECT_NAME}-{STATE}"
    print(f"  Destination  : https://huggingface.co/{repo_id}")
    print(f"  Mode         : {'Adapter only' if args.adapter_only else 'Merge + full model'}")
    print(f"{'═'*64}")

    # Verify adapter exists and is complete
    adapter_path = verify_adapter(adapter_path)

    # Authenticate
    print(f"\n🔐  Authenticating as {HF_USERNAME}...")
    login(token=HF_TOKEN)
    print(f"    ✅  Authenticated")

    # Create repo if it doesn't exist
    api = HfApi()
    try:
        api.create_repo(
            repo_id   = repo_id,
            repo_type = "model",
            token     = HF_TOKEN,
            exist_ok  = True,   # no error if already exists
        )
    except Exception as e:
        print(f"    ⚠️   Could not create repo (may already exist): {e}")

    # Upload
    if args.adapter_only:
        success = push_adapter_only(adapter_path, repo_id)
    else:
        success = merge_and_push(adapter_path, repo_id)
        if not success:
            print(f"\n↩️   Retrying with adapter-only upload...")
            success = push_adapter_only(adapter_path, repo_id)

    # Final status
    print(f"\n{'═'*64}")
    if success:
        print(f"  🏁  Upload complete!")
        print(f"  🔗  https://huggingface.co/{repo_id}")
    else:
        print(f"  ❌  Upload failed. Adapter is safe locally at {adapter_path}/")
        print(f"      Check your HF_TOKEN has WRITE access and try again.")
    print(f"{'═'*64}\n")

if __name__ == "__main__":
    main()
