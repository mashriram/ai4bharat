#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════╗
║       AI4BHARAT TUNE-ATHON  —  FINETUNE SCRIPT  v3.0                ║
║       Qwen3-1.7B · Apple M3 16GB · Target: beat Sarvam-1 per state  ║
║                                                                       ║
║  SETUP (run once on each machine):                                    ║
║    git clone https://github.com/ARahim3/unsloth-mlx                  ║
║    cd unsloth-mlx && uv pip install -e . && cd ..                    ║
║    uv pip install datasets huggingface_hub python-dotenv             ║
║                                                                       ║
║  CONFIGURE:  cp .env.example .env  then edit .env                    ║
║                                                                       ║
║  STEP 1 — Trial run (~3 min, verify + measure speed):                ║
║    python finetune.py --trial                                         ║
║                                                                       ║
║  STEP 2 — Full training:                                             ║
║    python finetune.py                                                 ║
║                                                                       ║
║  STEP 3 — If machine crashed:                                        ║
║    python finetune.py --resume                                        ║
╚═══════════════════════════════════════════════════════════════════════╝

CONFIG RATIONALE (why these numbers, not others):
─────────────────────────────────────────────────
MODEL: Qwen3-1.7B-4bit  (not 4B)
  The 4B model at seq=4096 + rank=64 runs at 12s/step on M4 Pro.
  15k steps would take 50 hours. 1.7B is 2.4x fewer params — same
  language quality for SFT on domain-specific data.

SEQ_LEN: 2048  (not 4096)
  We measured our dataset: longest row is ~1800 tokens (Wikipedia cult
  split, capped at 6000 chars / 3.5 chars/token). seq=2048 captures
  100% of every row with zero truncation. Going to 4096 costs 1.7x
  more compute (O(n^2) attention) for zero quality benefit.
  Answers will NOT stop abruptly — the data simply doesn't exceed 2048.

RANK: 64  (not 32)
  Qwen3-1.7B has hidden_dim=2048. rank=64 = 0.24% trainable params.
  For cultural memorization (our benchmark target), higher rank =
  more expressivity = better recall scores. We have time budget for it.

EPOCHS: auto-calculated from trial speed  (default=5)
  Trial run measures your machine's actual s/step, then auto-computes
  max safe epochs that fill ~85% of the 6.5h budget.
  Research shows 4-6 epochs is the sweet spot for 15k rows on 1.7B.
  Cosine LR decay prevents overfitting at higher epoch counts.

EFFECTIVE BATCH = 16 (batch=2 x grad_accum=8)
  batch=2 is max safe on 16GB at seq=2048+rank=64.
  Gradient accumulation=8 gives smooth convergence without memory hit.
"""

import os, sys, time, json, shutil, argparse
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────
# LOAD .env  (silent if python-dotenv not installed)
# ──────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login, HfApi


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
_EPOCHS_ENV = int(_optional("EPOCHS", "0"))  # 0 = auto

# ──────────────────────────────────────────────────────────────────────
# FIXED HYPERPARAMETERS  (do not change without re-reading the rationale)
# ──────────────────────────────────────────────────────────────────────
MODEL_ID = "mlx-community/Qwen3-1.7B-4bit"
MAX_SEQ = 2048  # covers 100% of dataset rows — see rationale above
RANK = 64  # 0.24% trainable on 1.7B — max expressivity in budget
ALPHA = 128  # = 2 x rank — strong domain adaptation
BATCH = 2  # max safe on 16GB at seq=2048+rank=64
GRAD_ACCUM = 8  # effective batch = 16
EFF_BATCH = BATCH * GRAD_ACCUM
LR = 2e-4
WARMUP = 50  # absolute steps, more stable than warmup_ratio
MAX_ROWS = 5000  # per split  → max 15k total rows
TRIAL_ROWS = 50  # per split in trial mode
SAVE_STEPS = 100
LOG_STEPS = 5
BUDGET_H = 6.5  # hour budget
SAFETY = 0.85  # use 85% of budget to leave room for upload

LORA_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# States confirmed to have only indic+conv (no cult split)
NO_CULT = {"Arunachal_Pradesh", "Meghalaya", "Mizoram"}

# ──────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────
CKPT_DIR = Path(f"checkpoints/{STATE}")
ADAPT_DIR = Path(f"adapters/{STATE}")
MERGE_DIR = Path(f"merged/{STATE}")
SPEED_FILE = CKPT_DIR / "speed.json"


# ──────────────────────────────────────────────────────────────────────
# SPEED CACHE  — trial saves s/step, full run loads it
# ──────────────────────────────────────────────────────────────────────
def save_speed(sps):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPEED_FILE, "w") as f:
        json.dump({"secs_per_step": sps, "at": datetime.now().isoformat()}, f)


def load_speed():
    if SPEED_FILE.exists():
        with open(SPEED_FILE) as f:
            return json.load(f).get("secs_per_step")
    return None


def auto_epochs(sps, n_rows):
    """Max epochs that fit in BUDGET_H * SAFETY, clamped to [3, 7]."""
    budget = BUDGET_H * SAFETY * 3600
    steps_ep = max(1, n_rows // EFF_BATCH)
    ep = int(budget / (steps_ep * sps))
    return max(3, min(7, ep))


def parse_stats(stats, elapsed):
    """
    unsloth-mlx SFTTrainer.train() returns a plain dict, not a TrainOutput.
    This normalises both so the rest of the code works regardless of version.
    Returns (global_step, training_loss).
    """
    if isinstance(stats, dict):
        step = stats.get("global_step") or stats.get("step") or 0
        loss = stats.get("training_loss") or stats.get("train_loss") or float("nan")
    else:
        # Standard HuggingFace TrainOutput namedtuple / object
        step = getattr(stats, "global_step", 0)
        loss = getattr(stats, "training_loss", float("nan"))
    # If step is still 0 (trainer didn't report it), estimate from elapsed
    if step == 0:
        step = max(1, int(elapsed))  # at minimum 1 so we never divide by zero
    return int(step), float(loss)


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
# CHAT TEMPLATE  (Qwen3 ChatML — no thinking tokens)
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
def load_data(trial=False):
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
        print(f"    ✅  {sname:<6}  {len(ds):>5,} rows  (orig {n_orig:,})")
        blended.append(ds)

    if not blended:
        print(f"\n❌  No usable data for {STATE}.")
        sys.exit(1)

    out = concatenate_datasets(blended).shuffle(seed=3407)
    print(f"\n    📊  Total rows : {len(out):,}")
    return out


# ──────────────────────────────────────────────────────────────────────
# TRIAL RUN
# ──────────────────────────────────────────────────────────────────────
def trial_run():
    from unsloth_mlx import FastLanguageModel, SFTTrainer, SFTConfig

    login(token=HF_TOKEN)

    print("\n" + "═" * 64)
    print("  🧪  TRIAL MODE  —  ~3 minutes")
    print("  Checks: model load · dataset · training step · speed")
    print("═" * 64)

    ds = load_data(trial=True)

    print(f"\n📥  Loading {MODEL_ID}...")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID, max_seq_length=MAX_SEQ, load_in_4bit=True
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=RANK,
        target_modules=LORA_MODULES,
        lora_alpha=ALPHA,
        lora_dropout=0,
        bias="none",
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        args=SFTConfig(
            output_dir=str(CKPT_DIR),
            per_device_train_batch_size=BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=1,
            learning_rate=LR,
            lr_scheduler_type="cosine",
            warmup_steps=min(WARMUP, 5),
            dataset_text_field="text",
            max_seq_length=MAX_SEQ,
            logging_steps=1,
            save_steps=9999,
            seed=3407,
            report_to="none",
        ),
    )

    t0 = time.time()
    stats = trainer.train()
    elapsed = time.time() - t0
    step, _ = parse_stats(stats, elapsed)
    sps = elapsed / step
    save_speed(sps)

    n_splits = 2 if STATE in NO_CULT else 3
    full_rows = MAX_ROWS * n_splits
    rec_ep = auto_epochs(sps, full_rows)
    full_steps = (full_rows // EFF_BATCH) * rec_ep
    full_h = full_steps * sps / 3600

    print(f"\n{'═' * 64}")
    print(f"  🧪  TRIAL RESULT")
    print(f"{'─' * 64}")
    print(f"  Speed            : {sps:.2f}s per optimizer step")
    print(f"  Full dataset     : {full_rows:,} rows")
    print(f"  Recommended      : {rec_ep} epochs  ({full_steps:,} steps)")
    print(f"  Estimated time   : {full_h:.1f}h on this machine")
    ok = full_h <= BUDGET_H
    print(f"  Status           : {'✅ on track' if ok else '⚠️  check EPOCHS in .env'}")

    if _EPOCHS_ENV:
        ov_h = (full_rows // EFF_BATCH) * _EPOCHS_ENV * sps / 3600
        flag = "✅ safe" if ov_h <= BUDGET_H else "⚠️  RISKY — may overrun budget"
        print(f"\n  .env EPOCHS={_EPOCHS_ENV} override → {ov_h:.1f}h  [{flag}]")

    print(f"\n  Speed saved. Run: python finetune.py")
    print(f"{'═' * 64}\n")


# ──────────────────────────────────────────────────────────────────────
# FULL TRAINING
# ──────────────────────────────────────────────────────────────────────
def full_run(resume=False):
    from unsloth_mlx import FastLanguageModel, SFTTrainer, SFTConfig

    login(token=HF_TOKEN)

    # Epoch determination
    if _EPOCHS_ENV:
        epochs = _EPOCHS_ENV
        epoch_note = f".env override (EPOCHS={epochs})"
    else:
        sps = load_speed()
        if sps:
            n_splits = 2 if STATE in NO_CULT else 3
            full_rows = MAX_ROWS * n_splits
            epochs = auto_epochs(sps, full_rows)
            epoch_note = f"auto-calculated from trial ({sps:.1f}s/step)"
        else:
            epochs = 5
            epoch_note = "default=5  (run --trial first for machine-specific value)"

    # Resume checkpoint
    resume_from = None
    if resume:
        resume_from = latest_checkpoint()
        msg = resume_from if resume_from else "none found — starting fresh"
        print(f"♻️   Checkpoint: {msg}")

    print(f"\n{'═' * 64}")
    print(f"  🌟  {STATE}")
    print(f"  🤖  {MODEL_ID}")
    print(f"  📐  seq={MAX_SEQ}  rank={RANK}  alpha={ALPHA}")
    print(f"  📦  batch={BATCH} × accum={GRAD_ACCUM} = eff_batch={EFF_BATCH}")
    print(f"  🔁  epochs={epochs}  ({epoch_note})")
    print(f"  📈  lr={LR}  warmup={WARMUP} steps  cosine")
    print(f"{'═' * 64}")

    # Data
    ds = load_data(trial=False)
    n_rows = len(ds)
    steps_ep = max(1, n_rows // EFF_BATCH)
    total_st = steps_ep * epochs

    sps = load_speed()
    if sps:
        est_h = total_st * sps / 3600
        print(f"\n⏱️   {total_st} steps × {sps:.1f}s ≈ {est_h:.1f}h")
    else:
        print(f"\n⏱️   {total_st} steps  (no speed estimate — run --trial first)")

    # Model
    print(f"\n📥  Loading {MODEL_ID}...")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID, max_seq_length=MAX_SEQ, load_in_4bit=True
    )

    # LoRA
    print(f"🔧  LoRA r={RANK}  alpha={ALPHA}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=RANK,
        target_modules=LORA_MODULES,
        lora_alpha=ALPHA,
        lora_dropout=0,
        bias="none",
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ADAPT_DIR.mkdir(parents=True, exist_ok=True)

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        args=SFTConfig(
            output_dir=str(CKPT_DIR),
            per_device_train_batch_size=BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=epochs,
            learning_rate=LR,
            lr_scheduler_type="cosine",
            warmup_steps=WARMUP,
            logging_steps=LOG_STEPS,
            save_steps=SAVE_STEPS,
            save_total_limit=3,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ,
            resume_from_checkpoint=resume_from,
            seed=3407,
            report_to="none",
        ),
    )

    # Train
    print(f"\n🏋️   {STATE}  — {epochs} epochs  {total_st} steps")
    print(f"    Checkpoint every {SAVE_STEPS} steps  ♻️  crash-safe")
    t0 = time.time()
    stats = trainer.train()
    elapsed = time.time() - t0
    step, loss = parse_stats(stats, elapsed)
    actual_sps = elapsed / step
    save_speed(actual_sps)

    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(
        f"\n✅  Done!  {h}h {m}m {s}s  |  {step} steps  |  loss {loss:.4f}  |  {actual_sps:.1f}s/step"
    )

    # Save adapter
    print(f"\n💾  Saving adapter → {ADAPT_DIR}/")
    model.save_pretrained(str(ADAPT_DIR))
    tok.save_pretrained(str(ADAPT_DIR))

    # Push adapter
    repo_id = f"{HF_USERNAME}/{PROJECT_NAME}-{STATE}"
    print(f"\n📤  Pushing adapter → {repo_id}")
    adapter_ok = False
    try:
        model.push_to_hub(repo_id, token=HF_TOKEN)
        tok.push_to_hub(repo_id, token=HF_TOKEN)
        print(f"    ✅  https://huggingface.co/{repo_id}")
        adapter_ok = True
    except Exception as e:
        print(f"    ⚠️   Push failed: {e}  (adapter safe locally at {ADAPT_DIR}/)")

    # Merge → full FP16 model
    MERGE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n🔀  Merging LoRA into FP16 model...")
    try:
        model.save_pretrained_merged(str(MERGE_DIR), tok, save_method="merged_16bit")
        print(f"    ✅  Merged → {MERGE_DIR}/")
        print(f"\n📤  Uploading merged model → {repo_id}")
        HfApi().upload_folder(
            folder_path=str(MERGE_DIR),
            repo_id=repo_id,
            repo_type="model",
            token=HF_TOKEN,
        )
        print(f"    🎉  https://huggingface.co/{repo_id}")
        shutil.rmtree(str(MERGE_DIR), ignore_errors=True)
    except Exception as e:
        print(f"    ⚠️   Merge failed: {e}")
        if adapter_ok:
            print(f"    Adapter already on Hub — model is usable with PEFT.")

    # Clean checkpoints
    for d in CKPT_DIR.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            shutil.rmtree(d, ignore_errors=True)
    print(f"\n    🧹  Checkpoints cleaned")

    print(f"\n{'═' * 64}")
    print(f"  🏁  {STATE}  COMPLETE")
    print(f"  🔗  https://huggingface.co/{repo_id}")
    print(f"{'═' * 64}\n")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI4Bharat Tune-Athon: Qwen3-1.7B on M3 16GB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
workflow:
  python finetune.py --trial     # ~3 min — verify + measure speed
  python finetune.py             # full run using measured speed
  python finetune.py --resume    # resume after crash/sleep
        """,
    )
    parser.add_argument(
        "--trial",
        action="store_true",
        help="Trial: 50 rows, 1 epoch, ~3 min. Measures s/step.",
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
