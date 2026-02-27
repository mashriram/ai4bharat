#!/usr/bin/env python3
import unsloth
"""
╔══════════════════════════════════════════════════════════════════════════╗
║     AI4BHARAT TUNE-ATHON  —  GPU FINETUNE SCRIPT  v1.0                 ║
║     Auto-selects model based on detected VRAM                           ║
║                                                                          ║
║     VRAM  →  Model                                                       ║
║     ≥ 8GB  →  unsloth/Qwen3-1.7B-bnb-4bit     (best quality)           ║
║     ≥ 6GB  →  unsloth/Qwen3-1.7B-bnb-4bit     (tight, batch=1)         ║
║     ≥ 4GB  →  unsloth/Qwen3-0.6B-bnb-4bit     (safe)                   ║
║     < 4GB  →  unsloth/Llama-3.2-1B-bnb-4bit   (fallback)               ║
║                                                                          ║
║  SETUP (run once):                                                       ║
║    pip install unsloth datasets huggingface_hub python-dotenv trl       ║
║                                                                          ║
║  CONFIGURE:  cp .env.example .env  then edit .env                       ║
║                                                                          ║
║  STEP 1 — Trial run (~5 min, verify + measure real speed):              ║
║    python finetune_gpu.py --trial                                        ║
║                                                                          ║
║  STEP 2 — Full training:                                                 ║
║    python finetune_gpu.py                                                ║
║                                                                          ║
║  STEP 3 — Resume after crash:                                            ║
║    python finetune_gpu.py --resume                                       ║
║                                                                          ║
║  Override model manually (ignores VRAM detection):                      ║
║    MODEL_OVERRIDE=unsloth/Qwen3-1.7B-bnb-4bit python finetune_gpu.py   ║
║                                                                          ║
║  Ctrl+C at ANY point → saves adapter + merges + uploads to Hub          ║
╚══════════════════════════════════════════════════════════════════════════╝

VRAM BUDGET GUIDE (observed peaks with QLoRA, rank=64, seq=2048):
  Qwen3-1.7B  4bit  batch=2  →  ~7.5GB VRAM   (comfortable on 8GB)
  Qwen3-1.7B  4bit  batch=1  →  ~5.5GB VRAM   (workable on 6GB)
  Qwen3-0.6B  4bit  batch=2  →  ~3.5GB VRAM   (safe on 4GB)
  Llama-3.2-1B 4bit batch=1  →  ~3.0GB VRAM   (fallback for <4GB)

KEY DIFFERENCES from the MLX (Apple Silicon) version:
  - Uses standard `unsloth` (pip) instead of `unsloth-mlx`
  - SFTTrainer + SFTConfig come from `trl` (not unsloth_mlx)
  - Optimizer: adamw_8bit  (saves ~1GB vs standard AdamW)
  - use_gradient_checkpointing="unsloth"  (saves ~30% more VRAM)
  - VRAM auto-detection picks model + batch size at startup
  - fp16=True on Ampere+, fp16=True otherwise (bf16 > fp16 on A100/3090+)
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# LOAD .env
# ──────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import torch
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi, login


# ──────────────────────────────────────────────────────────────────────
# CONFIG FROM .env
# ──────────────────────────────────────────────────────────────────────
def _require(key):
    v = os.getenv(key, "").strip()
    if not v:
        print(f"\n❌  Missing required .env variable: {key}")
        print(f"    Open .env and add:  {key}=<value>\n")
        sys.exit(1)
    return v


def _optional(key, default):
    return os.getenv(key, default).strip() or default


HF_TOKEN = _require("HF_TOKEN")
HF_USERNAME = _require("HF_USERNAME")
STATE = _require("STATE")
DATASET_REPO = _optional(
    "DATASET_REPO", "mashriram/AI4Bharat-Indic-Languages-and-Cultures"
)
PROJECT_NAME = _optional("PROJECT_NAME", "AI4Bharat-State-Expert")
_EPOCHS_ENV = int(_optional("EPOCHS", "0"))  # 0 = auto-calculate
_MODEL_OVERRIDE = os.getenv("MODEL_OVERRIDE", "").strip()


# ──────────────────────────────────────────────────────────────────────
# VRAM DETECTION  →  model + batch config
# ──────────────────────────────────────────────────────────────────────
def detect_vram_gb():
    """Returns usable VRAM in GB (0 if no CUDA GPU found)."""
    if not torch.cuda.is_available():
        print("⚠️   No CUDA GPU detected. Cannot run GPU training.")
        sys.exit(1)
    props = torch.cuda.get_device_properties(0)
    total_mb = props.total_memory / (1024**2)
    # Reserve ~500MB for CUDA overhead
    usable_gb = (total_mb - 500) / 1024
    return usable_gb, props.name


def is_ampere_or_newer():
    """True for RTX 30xx / A100 / H100 (compute capability ≥ 8.0)."""
    if not torch.cuda.is_available():
        return False
    cc = torch.cuda.get_device_capability(0)
    return cc[0] >= 8


# Model catalogue: (id, min_vram_gb, batch_size, rank, note)
# Ordered best-first. First entry whose min_vram fits wins.
MODEL_CATALOGUE = [
    {
        "id": "unsloth/Qwen3-1.7B-bnb-4bit",
        "min_vram": 7.0,
        "batch": 2,
        "rank": 64,
        "note": "Qwen3-1.7B  4-bit  batch=2  (best quality, needs ≥8GB)",
    },
    {
        "id": "unsloth/Qwen3-1.7B-bnb-4bit",
        "min_vram": 5.0,
        "batch": 1,
        "rank": 32,
        "note": "Qwen3-1.7B  4-bit  batch=1  (tight on 6GB, rank=32)",
    },
    {
        "id": "unsloth/Qwen3-0.6B-bnb-4bit",
        "min_vram": 3.0,
        "batch": 2,
        "rank": 64,
        "note": "Qwen3-0.6B  4-bit  batch=2  (safe on 4GB)",
    },
    {
        "id": "unsloth/Llama-3.2-1B-bnb-4bit",
        "min_vram": 0.0,
        "batch": 1,
        "rank": 32,
        "note": "Llama-3.2-1B  4-bit  batch=1  (emergency fallback)",
    },
]


def select_model(vram_gb):
    """Pick the best model/config that fits available VRAM."""
    if _MODEL_OVERRIDE:
        # Manual override: use override id but still pick safe batch/rank
        for cfg in MODEL_CATALOGUE:
            if vram_gb >= cfg["min_vram"]:
                return {
                    **cfg,
                    "id": _MODEL_OVERRIDE,
                    "note": f"MANUAL OVERRIDE: {_MODEL_OVERRIDE}  (batch={cfg['batch']})",
                }
    for cfg in MODEL_CATALOGUE:
        if vram_gb >= cfg["min_vram"]:
            return cfg
    # Should never reach here because last entry has min_vram=0
    return MODEL_CATALOGUE[-1]


# ──────────────────────────────────────────────────────────────────────
# FIXED HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────────────
MAX_SEQ = 2048
ALPHA_MULT = 2  # lora_alpha = rank * ALPHA_MULT
GRAD_ACCUM = 8  # effective batch = BATCH × GRAD_ACCUM
LR = 2e-4
WARMUP = 50
MAX_ROWS = 3500  # per split before pre-splitting
TRIAL_ROWS = 50
SAVE_STEPS = 200
LOG_STEPS = 5
BUDGET_H = 6.5
SAFETY = 0.88

LORA_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

NO_CULT = {"Arunachal_Pradesh", "Meghalaya", "Mizoram"}

# ──────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────
CKPT_DIR = Path(f"checkpoints/{STATE}")
ADAPT_DIR = Path(f"adapters/{STATE}")
MERGE_DIR = Path(f"merged/{STATE}")
SPEED_FILE = CKPT_DIR / "speed.json"


# ──────────────────────────────────────────────────────────────────────
# SPEED CACHE
# ──────────────────────────────────────────────────────────────────────
def save_speed(spi):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPEED_FILE, "w") as f:
        json.dump({"secs_per_iter": spi, "at": datetime.now().isoformat()}, f)


def load_speed():
    if SPEED_FILE.exists():
        with open(SPEED_FILE) as f:
            d = json.load(f)
            return d.get("secs_per_iter") or d.get("secs_per_step")
    return None


def auto_epochs(spi, n_rows, batch):
    """
    Compute max epochs that fit inside BUDGET_H * SAFETY.
    Clamped to [1, 3].
    """
    budget = BUDGET_H * SAFETY * 3600
    iters_per_epoch = max(1, n_rows // batch)
    ep = int(budget / (iters_per_epoch * spi))
    return max(1, min(3, ep))


# ──────────────────────────────────────────────────────────────────────
# CHECKPOINT RECOVERY
# ──────────────────────────────────────────────────────────────────────
def latest_checkpoint():
    if not CKPT_DIR.exists():
        return None
    ckpts = sorted(
        [
            d
            for d in CKPT_DIR.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ],
        key=lambda d: int(d.name.split("-")[1]),
    )
    return str(ckpts[-1]) if ckpts else None


# ──────────────────────────────────────────────────────────────────────
# STATS NORMALISER
# trl's SFTTrainer returns a TrainOutput named-tuple; handle both
# named-tuple and plain dict (some versions differ).
# ──────────────────────────────────────────────────────────────────────
def parse_stats(stats, elapsed):
    if isinstance(stats, dict):
        step = stats.get("global_step") or stats.get("step") or 0
        loss = stats.get("training_loss") or stats.get("train_loss") or float("nan")
    else:
        step = getattr(stats, "global_step", 0)
        loss = getattr(stats, "training_loss", float("nan"))
    if step == 0:
        step = max(1, int(elapsed))
    return int(step), float(loss)


# ──────────────────────────────────────────────────────────────────────
# PRE-SPLITTING  (identical logic to MLX version)
# Fixes "sequences longer than 2048" truncation warning.
# ──────────────────────────────────────────────────────────────────────
STRIDE = 1536
SPLIT_AT = MAX_SEQ - 8  # 2-token buffer for BOS/EOS overhead


def split_long_rows(examples, tokenizer):
    out_texts = []
    for text in examples["text"]:
        ids = tokenizer.encode(text)
        if len(ids) <= SPLIT_AT:
            out_texts.append(text)
        else:
            start = 0
            while start < len(ids):
                chunk_ids = ids[start : start + SPLIT_AT]
                out_texts.append(tokenizer.decode(chunk_ids))
                start += STRIDE
                if start + 64 >= len(ids):
                    break
    return {"text": out_texts}


# ──────────────────────────────────────────────────────────────────────
# CHAT TEMPLATE  (Qwen3 / generic ChatML)
# ──────────────────────────────────────────────────────────────────────
def apply_template(ex):
    instr = (ex.get("instruction") or "").strip()
    ctx = (ex.get("input") or "").strip()
    response = (ex.get("response") or "").strip()
    user_msg = f"{instr}\n{ctx}" if ctx else instr
    return {
        "text": (
            "<|im_start|>user\n" + user_msg + "<|im_end|>\n"
            "<|im_start|>assistant\n" + response + "<|im_end|>"
        )
    }


def is_valid(ex):
    return bool((ex.get("instruction") or "").strip()) and bool(
        (ex.get("response") or "").strip()
    )


# ──────────────────────────────────────────────────────────────────────
# DATASET LOADER
# ──────────────────────────────────────────────────────────────────────
def load_data(trial=False, tokenizer=None):
    cap = TRIAL_ROWS if trial else MAX_ROWS
    mode_str = f"TRIAL ({cap} rows/split)" if trial else f"FULL ({cap} rows/split)"

    print(f"\n{'─' * 64}")
    print(f"📥  {DATASET_REPO}  [{STATE}]  —  {mode_str}")
    print(f"{'─' * 64}")

    try:
        ds_dict = load_dataset(DATASET_REPO, name=STATE)
    except Exception as e:
        print(f"\n❌  Cannot load '{STATE}' from {DATASET_REPO}: {e}")
        sys.exit(1)

    splits = ["indic", "conv"]
    if STATE not in NO_CULT:
        if "cult" in ds_dict:
            splits.append("cult")
        else:
            print(f"    ⚠️   cult split missing for {STATE} — using indic+conv")
    else:
        print(f"    ℹ️   {STATE}: no cult split (known) — using indic+conv")

    blended = []
    for sname in splits:
        if sname not in ds_dict:
            print(f"    ⚠️   '{sname}' not found — skipping")
            continue

        ds = ds_dict[sname]
        n_orig = len(ds)
        ds = ds.filter(is_valid)

        if len(ds) == 0:
            print(f"    ⚠️   '{sname}' has 0 valid rows — skipping")
            continue

        if len(ds) > cap:
            ds = ds.shuffle(seed=3407).select(range(cap))

        ds = ds.map(
            apply_template,
            remove_columns=[c for c in ds.column_names if c != "text"],
            desc=f"Formatting {sname}",
        )
        ds = ds.filter(lambda x: len(x["text"]) > 30)

        n_before = len(ds)

        if tokenizer is not None:
            ds = ds.map(
                lambda batch: split_long_rows(batch, tokenizer),
                batched=True,
                batch_size=32,
                remove_columns=["text"],
                desc=f"Pre-splitting {sname}",
            )
            if "text" not in ds.column_names:
                print(f"    ⚠️   Pre-split produced no 'text' column for {sname}")
                continue

            n_after = len(ds)
            if n_after != n_before:
                print(
                    f"    ✅  {sname:<6}  {n_after:>6,} rows after split  "
                    f"(orig {n_orig:,} → capped {n_before:,} → split {n_after:,})"
                )
            else:
                print(
                    f"    ✅  {sname:<6}  {n_after:>6,} rows  (orig {n_orig:,}, no splitting needed)"
                )
        else:
            print(f"    ✅  {sname:<6}  {n_before:>6,} rows  (orig {n_orig:,})")

        blended.append(ds)

    if not blended:
        print(f"\n❌  No usable data for {STATE}.")
        sys.exit(1)

    out = concatenate_datasets(blended).shuffle(seed=3407)
    print(
        f"\n    📊  Final rows : {len(out):,}  (all ≤{MAX_SEQ} tokens, zero truncation)"
    )
    return out


# ──────────────────────────────────────────────────────────────────────
# LOAD MODEL  (unsloth GPU path)
# ──────────────────────────────────────────────────────────────────────
def load_model(cfg):
    from unsloth import FastLanguageModel

    use_bf16 = is_ampere_or_newer()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    dtype_str = "bf16" if use_bf16 else "fp16"
    rank = cfg["rank"]
    alpha = rank * ALPHA_MULT

    print(f"\n📥  Loading {cfg['id']}  ({dtype_str})...")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=cfg["id"],
        max_seq_length=MAX_SEQ,
        dtype=dtype,
        load_in_4bit=True,
    )

    print(f"🔧  LoRA  r={rank}  alpha={alpha}  modules={len(LORA_MODULES)}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=LORA_MODULES,
        lora_alpha=alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # saves ~30% extra VRAM
        random_state=3407,
        max_seq_length=MAX_SEQ,
        use_rslora=False,
    )
    return model, tok, rank, alpha


# ──────────────────────────────────────────────────────────────────────
# TRIAL RUN
# ──────────────────────────────────────────────────────────────────────
def trial_run():
    from trl import SFTConfig, SFTTrainer

    login(token=HF_TOKEN)

    vram_gb, gpu_name = detect_vram_gb()
    cfg = select_model(vram_gb)
    batch = cfg["batch"]

    print("\n" + "═" * 64)
    print("  🧪  TRIAL MODE  —  ~5 minutes")
    print(f"  GPU  : {gpu_name}  ({vram_gb:.1f}GB usable)")
    print(f"  Model: {cfg['note']}")
    print("═" * 64)

    model, tok, rank, alpha = load_model(cfg)
    ds = load_data(trial=True, tokenizer=tok)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        args=SFTConfig(
            output_dir=str(CKPT_DIR),
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=1,
            learning_rate=LR,
            lr_scheduler_type="cosine",
            warmup_steps=min(WARMUP, 5),
            optim="adamw_8bit",
            dataset_text_field="text",
            max_seq_length=MAX_SEQ,
            logging_steps=1,
            save_steps=9999,
            bf16=is_ampere_or_newer(),
            fp16=not is_ampere_or_newer(),
            seed=3407,
            report_to="none",
        ),
    )

    t0 = time.time()
    stats = trainer.train()
    elapsed = time.time() - t0
    step, _ = parse_stats(stats, elapsed)
    expected_iters = max(1, len(ds) // batch)
    actual_iters = step if step > 0 else expected_iters
    spi = elapsed / actual_iters
    save_speed(spi)

    n_splits = 2 if STATE in NO_CULT else 3
    full_rows = int(MAX_ROWS * n_splits * 1.25)
    rec_ep = auto_epochs(spi, full_rows, batch)
    full_iters = (full_rows // batch) * rec_ep
    full_h = full_iters * spi / 3600

    print(f"\n{'═' * 64}")
    print("  🧪  TRIAL RESULT")
    print(f"{'─' * 64}")
    print(f"  GPU                   : {gpu_name}  ({vram_gb:.1f}GB usable)")
    print(f"  Model                 : {cfg['id']}")
    print(f"  Batch                 : {batch}  ×  grad_accum={GRAD_ACCUM}")
    print(f"  Speed (this GPU)      : {spi:.2f}s per iteration")
    print(f"  Full dataset estimate : ~{full_rows:,} rows (after pre-splitting)")
    print(f"  Recommended           : {rec_ep} epoch(s)  ({full_iters:,} iterations)")
    print(f"  Est. training time    : {full_h:.1f}h")
    ok = full_h <= BUDGET_H
    print(
        f"  Budget status         : {'✅ fits in 6.5h window' if ok else '⚠️  check EPOCHS in .env'}"
    )

    if _EPOCHS_ENV:
        ov_iters = (full_rows // batch) * _EPOCHS_ENV
        ov_h = ov_iters * spi / 3600
        flag = "✅ safe" if ov_h <= BUDGET_H else "⚠️  RISKY — may overrun"
        print(f"\n  .env EPOCHS={_EPOCHS_ENV} override → ~{ov_h:.1f}h  [{flag}]")

    print("\n  Speed saved. Run: python finetune_gpu.py")
    print(f"{'═' * 64}\n")


# ──────────────────────────────────────────────────────────────────────
# SAVE + MERGE + UPLOAD
# Called on both clean finish AND Ctrl+C.
# ──────────────────────────────────────────────────────────────────────
def save_and_upload(model, tok, interrupted=False):
    tag = "⚡ INTERRUPTED" if interrupted else "✅ COMPLETE"
    repo_id = f"{HF_USERNAME}/{PROJECT_NAME}-{STATE}"

    print(f"\n{'═' * 64}")
    print(f"  {tag}  —  Saving and uploading {STATE}")
    print(f"{'═' * 64}")

    # 1. Save adapter locally
    ADAPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n💾  Saving adapter → {ADAPT_DIR}/")
    try:
        model.save_pretrained(str(ADAPT_DIR))
        tok.save_pretrained(str(ADAPT_DIR))
        print("    ✅  Adapter saved")
    except Exception as e:
        print(f"    ⚠️   save_pretrained failed: {e}")
        ckpt_adapter = CKPT_DIR / "adapter_model.safetensors"
        if ckpt_adapter.exists():
            print(f"    ↩️   Falling back to checkpoint adapter: {ckpt_adapter}")
        else:
            print("    ❌  No adapter found. Cannot upload.")
            return

    # 2. Push adapter to Hub
    print(f"\n📤  Pushing adapter → {repo_id}")
    adapter_ok = False
    try:
        api = HfApi()
        api.create_repo(
            repo_id=repo_id, repo_type="model", token=HF_TOKEN, exist_ok=True
        )
        model.push_to_hub(repo_id, token=HF_TOKEN)
        tok.push_to_hub(repo_id, token=HF_TOKEN)
        print(f"    ✅  https://huggingface.co/{repo_id}")
        adapter_ok = True
    except Exception as e:
        print(f"    ⚠️   Adapter push failed: {e}")
        print(f"    Adapter safe locally at {ADAPT_DIR}/")

    # 3. Merge LoRA → full 16-bit model
    MERGE_DIR.mkdir(parents=True, exist_ok=True)
    print("\n🔀  Merging LoRA into 16-bit model...")
    merge_ok = False
    try:
        model.save_pretrained_merged(str(MERGE_DIR), tok, save_method="merged_16bit")
        print(f"    ✅  Merged → {MERGE_DIR}/")
        merge_ok = True
    except Exception as e:
        print(f"    ⚠️   Merge failed: {e}")
        if adapter_ok:
            print("    Adapter is on Hub — loadable via PEFT.")

    # 4. Upload merged model
    if merge_ok:
        print(f"\n📤  Uploading merged model → {repo_id}")
        try:
            HfApi().upload_folder(
                folder_path=str(MERGE_DIR),
                repo_id=repo_id,
                repo_type="model",
                token=HF_TOKEN,
            )
            print(f"    🎉  Full model live: https://huggingface.co/{repo_id}")
            shutil.rmtree(str(MERGE_DIR), ignore_errors=True)
        except Exception as e:
            print(f"    ⚠️   Merged upload failed: {e}")
            print(f"    Merged model at {MERGE_DIR}/ — retry manually.")

    print(f"\n  🔗  https://huggingface.co/{repo_id}")
    print(f"{'═' * 64}\n")


# ──────────────────────────────────────────────────────────────────────
# FULL TRAINING
# ──────────────────────────────────────────────────────────────────────
def full_run(resume=False):
    from trl import SFTConfig, SFTTrainer

    login(token=HF_TOKEN)

    vram_gb, gpu_name = detect_vram_gb()
    cfg = select_model(vram_gb)
    batch = cfg["batch"]

    # Epoch count
    if _EPOCHS_ENV:
        epochs = _EPOCHS_ENV
        epoch_note = f".env override (EPOCHS={epochs})"
    else:
        spi = load_speed()
        if spi:
            n_splits = 2 if STATE in NO_CULT else 3
            est_rows = int(MAX_ROWS * n_splits * 1.25)
            epochs = auto_epochs(spi, est_rows, batch)
            epoch_note = f"auto ({spi:.2f}s/iter measured)"
        else:
            epochs = 2
            epoch_note = "default=2  (run --trial first for better estimate)"

    resume_from = None
    if resume:
        resume_from = latest_checkpoint()
        msg = resume_from if resume_from else "none found — starting fresh"
        print(f"♻️   Checkpoint: {msg}")

    use_bf16 = is_ampere_or_newer()
    rank = cfg["rank"]
    alpha = rank * ALPHA_MULT

    print(f"\n{'═' * 64}")
    print(f"  🌟  {STATE}")
    print(f"  🖥️   {gpu_name}  ({vram_gb:.1f}GB usable)")
    print(f"  🤖  {cfg['id']}  —  {cfg['note']}")
    print(f"  📐  seq={MAX_SEQ}  rank={rank}  alpha={alpha}")
    print(f"  📦  batch={batch} × accum={GRAD_ACCUM}  optim=adamw_8bit")
    print(f"  🔁  epochs={epochs}  ({epoch_note})")
    print(f"  📈  lr={LR}  warmup={WARMUP}  cosine")
    print(f"  🎯  precision={'bf16' if use_bf16 else 'fp16'}  grad_ckpt=unsloth")
    print(f"  ✂️   pre-splitting rows>{MAX_SEQ} tokens  →  zero truncation")
    print(f"{'═' * 64}")

    model, tok, rank, alpha = load_model(cfg)

    ds = load_data(trial=False, tokenizer=tok)
    n_rows = len(ds)
    iters_ep = max(1, n_rows // batch)
    total_it = iters_ep * epochs

    spi = load_speed()
    if spi:
        est_h = total_it * spi / 3600
        print(f"\n⏱️   {total_it:,} iters × {spi:.2f}s/iter = ~{est_h:.1f}h")
    else:
        print(
            f"\n⏱️   {total_it:,} iterations total  (no speed data — run --trial first)"
        )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ADAPT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        args=SFTConfig(
            output_dir=str(CKPT_DIR),
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=epochs,
            learning_rate=LR,
            lr_scheduler_type="cosine",
            warmup_steps=WARMUP,
            optim="adamw_8bit",
            logging_steps=LOG_STEPS,
            save_steps=SAVE_STEPS,
            save_total_limit=3,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ,
            resume_from_checkpoint=resume_from,
            bf16=use_bf16,
            fp16=not use_bf16,
            seed=3407,
            report_to="none",
            # packing=True speeds up training by ~20% if rows vary in length
            # packing=False is safer when rows are already split to uniform size
            packing=False,
        ),
    )

    print(f"\n🏋️   Training  {STATE}  —  {epochs} epoch(s)  {total_it:,} iters")
    print(f"    Checkpoint every {SAVE_STEPS} iters  ♻️  crash-safe")
    print("    Press Ctrl+C at any time → saves adapter + uploads immediately")
    t0 = time.time()

    interrupted = False
    try:
        stats = trainer.train()
        elapsed = time.time() - t0
        step, loss = parse_stats(stats, elapsed)
        actual_spi = elapsed / max(1, step)
        save_speed(actual_spi)
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        print(
            f"\n✅  Training done!  {h}h {m}m {s}s  |  loss {loss:.4f}  |  {actual_spi:.2f}s/iter"
        )

    except KeyboardInterrupt:
        elapsed = time.time() - t0
        interrupted = True
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        print(f"\n\n⚡  Ctrl+C received after {h}h {m}m {s}s")
        print(f"    Latest checkpoint is safe in {CKPT_DIR}/")
        print("    Proceeding to save + upload...")

    save_and_upload(model, tok, interrupted=interrupted)

    if not interrupted:
        for d in CKPT_DIR.iterdir():
            if d.is_dir() and d.name.startswith("checkpoint-"):
                shutil.rmtree(d, ignore_errors=True)
        print("    🧹  Checkpoints cleaned")

    status = "INTERRUPTED (adapter saved)" if interrupted else "COMPLETE"
    print(f"\n  🏁  {STATE}  —  {status}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI4Bharat Tune-Athon: GPU finetune with auto VRAM detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
workflow:
  python finetune_gpu.py --trial     # ~5 min — verify + measure speed
  python finetune_gpu.py             # full training
  python finetune_gpu.py --resume    # resume after crash

VRAM → model mapping:
  ≥ 8GB  →  Qwen3-1.7B  batch=2  rank=64   (best)
  ≥ 6GB  →  Qwen3-1.7B  batch=1  rank=32
  ≥ 4GB  →  Qwen3-0.6B  batch=2  rank=64
  < 4GB  →  Llama-3.2-1B batch=1 rank=32

Override model:
  MODEL_OVERRIDE=unsloth/gemma-3-1b-bnb-4bit python finetune_gpu.py
        """,
    )
    parser.add_argument(
        "--trial",
        action="store_true",
        help="Trial: 50 rows/split, 1 epoch. Measures speed.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume full training from latest checkpoint.",
    )
    args = parser.parse_args()

    if args.trial and args.resume:
        print("❌  --trial and --resume cannot be used together.")
        sys.exit(1)

    if args.trial:
        trial_run()
    else:
        full_run(resume=args.resume)
