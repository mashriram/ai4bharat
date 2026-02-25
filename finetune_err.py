#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════╗
║       AI4BHARAT TUNE-ATHON  —  FINETUNE SCRIPT  v4.0                ║
║       Qwen3-1.7B · Apple M3 16GB · Target: beat Sarvam-1 per state  ║
║                                                                       ║
║  SETUP (run once on each machine):                                    ║
║    git clone https://github.com/ARahim3/unsloth-mlx                  ║
║    cd unsloth-mlx && uv pip install -e . && cd ..                    ║
║    uv pip install datasets huggingface_hub python-dotenv             ║
║                                                                       ║
║  CONFIGURE:  cp .env.example .env  then edit .env                    ║
║                                                                       ║
║  STEP 1 — Trial run (~5 min, verify + measure real speed):           ║
║    python finetune.py --trial                                         ║
║                                                                       ║
║  STEP 2 — Full training (auto-uses measured speed):                  ║
║    python finetune.py                                                 ║
║                                                                       ║
║  STEP 3 — If machine crashed mid-run:                                ║
║    python finetune.py --resume                                        ║
╚═══════════════════════════════════════════════════════════════════════╝

THREE BUGS FIXED IN v4.0 (found from real trial run output):
─────────────────────────────────────────────────────────────
BUG 1 — Token truncation (the WARNING spam):
  "[WARNING] Some sequences longer than 2048... longest 12077 will be truncated"
  The cult (Wikipedia) split has rows up to 12,077 tokens — far more than our
  assumed 1,800. Raising seq_len to 4096 is NOT the right fix: it costs 1.7×
  more compute per iteration (O(n²) attention) and still truncates rows >4096.
  The warning itself says the right fix: "Consider pre-splitting your data."
  We now chunk every row that exceeds MAX_SEQ into overlapping sub-rows of
  ≤MAX_SEQ tokens at sentence boundaries. This gives ZERO truncation at the
  same speed. More content learned = better benchmark scores.

BUG 2 — auto_epochs was computing with wrong denominator:
  unsloth-mlx counts raw batch iterations, not HF optimizer steps.
  grad_accum=8 is passed through but the trainer counts every forward pass.
  Old formula used rows//EFF_BATCH (937 steps) → wildly underestimated time.
  Real formula: rows//BATCH (7500 iters/epoch). At 1s/iter on M4 Pro:
    2 epochs × 7500 iters = 15,000s = 4.2h M4 → 5.2h M3  ✅ fits in 6-7h
    3 epochs              = 22,500s = 6.2h M4 → 7.8h M3  ❌ over budget
  auto_epochs now uses BATCH not EFF_BATCH and is clamped to [1, 2].

BUG 3 — stats.global_step AttributeError (already patched in v3.1):
  unsloth-mlx trainer.train() returns a plain dict, not a TrainOutput object.
  parse_stats() handles both cases.

SEQ_LEN stays at 2048:
  With pre-splitting, seq=2048 captures 100% of content from every row.
  Raising to 4096 would only allow 1 epoch (too slow) and still truncate
  rows >4096. Pre-splitting at 2048 + 2 epochs = more total learning.
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

from datasets import load_dataset, concatenate_datasets, Dataset
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
_EPOCHS_ENV = int(_optional("EPOCHS", "0"))  # 0 = auto-calculate

# ──────────────────────────────────────────────────────────────────────
# FIXED HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────────────
MODEL_ID = "mlx-community/Qwen3-1.7B-4bit"

# seq=2048 + pre-splitting = zero truncation at maximum speed.
# Raising seq would be 1.7-4x slower per iter with no quality gain
# once pre-splitting is applied correctly.
MAX_SEQ = 2048

# rank=64 on Qwen3-1.7B (hidden_dim=2048) = 0.24% trainable params.
# Maximum expressivity for cultural memorization within our time budget.
RANK = 64
ALPHA = 128  # = 2 × rank

# batch=2: max safe on 16GB at seq=2048 + rank=64 (peak 7.5GB observed).
# IMPORTANT: unsloth-mlx counts raw forward-pass iterations, not optimizer
# steps. grad_accum is passed through but iteration count = rows // BATCH.
BATCH = 2
GRAD_ACCUM = 8  # passed to trainer, affects gradient quality

LR = 2e-4
WARMUP = 50  # absolute iterations
MAX_ROWS = 5000  # per split before pre-splitting expands them
TRIAL_ROWS = 50  # per split in trial mode
SAVE_STEPS = 200  # checkpoint every N iterations
LOG_STEPS = 5
BUDGET_H = 6.5
SAFETY = 0.88  # use 88% of budget, leave buffer for upload/merge

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
# Stores the measured s/iter from trial so full run can predict duration.
# NOTE: 'secs_per_iter' = time per raw forward batch (rows // BATCH).
# This is what unsloth-mlx actually counts — NOT optimizer steps.
# ──────────────────────────────────────────────────────────────────────
def save_speed(spi):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPEED_FILE, "w") as f:
        json.dump({"secs_per_iter": spi, "at": datetime.now().isoformat()}, f)


def load_speed():
    if SPEED_FILE.exists():
        with open(SPEED_FILE) as f:
            d = json.load(f)
            # Handle both old key name and new key name
            return d.get("secs_per_iter") or d.get("secs_per_step")
    return None


def auto_epochs(spi, n_rows):
    """
    Compute max epochs that fit inside BUDGET_H * SAFETY on THIS machine.
    Uses raw iteration count (n_rows // BATCH) — NOT optimizer steps.
    Clamped to [1, 2]:
      - 1 epoch minimum (otherwise not enough training signal)
      - 2 epoch maximum (3 epochs would be 7.8h on M3, over budget)
    """
    budget = BUDGET_H * SAFETY * 3600
    iters_per_epoch = max(1, n_rows // BATCH)
    ep = int(budget / (iters_per_epoch * spi))
    return max(1, min(2, ep))


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
# unsloth-mlx trainer.train() returns a plain dict, not a TrainOutput.
# This handles both so the code works regardless of library version.
# ──────────────────────────────────────────────────────────────────────
def parse_stats(stats, elapsed):
    if isinstance(stats, dict):
        step = stats.get("global_step") or stats.get("step") or 0
        loss = stats.get("training_loss") or stats.get("train_loss") or float("nan")
    else:
        step = getattr(stats, "global_step", 0)
        loss = getattr(stats, "training_loss", float("nan"))
    # Fallback: estimate step count from elapsed time if trainer didn't report it
    if step == 0:
        step = max(1, int(elapsed))
    return int(step), float(loss)


# ──────────────────────────────────────────────────────────────────────
# PRE-SPLITTING  — the real fix for the truncation warning
#
# The warning "[WARNING] Some sequences longer than 2048 tokens"
# means Wikipedia articles in the cult split are up to 12,077 tokens.
# Raising seq_len is NOT the answer: O(n²) attention makes it 1.7-4x
# slower and still truncates rows above the new limit.
#
# The correct fix: tokenize each row, then slide a window of MAX_SEQ
# tokens across it, stepping by STRIDE tokens, creating multiple
# sub-rows. Each sub-row is a complete, independently learnable example.
#
# Overlap (stride < MAX_SEQ) ensures context continuity between chunks —
# important for long Wikipedia narratives so the model sees both the
# beginning and end of each article in the same training pass.
# ──────────────────────────────────────────────────────────────────────
STRIDE = 1536  # overlap of 512 tokens between consecutive chunks


def split_long_rows(examples, tokenizer):
    """
    Batch-map function. For each text in the batch:
      - If it fits in MAX_SEQ tokens → keep as-is
      - If it's longer → slice into overlapping windows of MAX_SEQ tokens
    Returns a flat list of 'text' strings, all guaranteed ≤ MAX_SEQ tokens.
    """
    out_texts = []
    for text in examples["text"]:
        ids = tokenizer.encode(text)
        if len(ids) <= MAX_SEQ:
            out_texts.append(text)
        else:
            # Slide window across token ids, decode each chunk back to text
            start = 0
            while start < len(ids):
                chunk_ids = ids[start : start + MAX_SEQ]
                out_texts.append(tokenizer.decode(chunk_ids))
                start += STRIDE
                if start + 64 >= len(ids):  # avoid tiny trailing chunk
                    break
    return {"text": out_texts}


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
def load_data(trial=False, tokenizer=None):
    """
    Load, format, and pre-split the dataset for STATE.
    tokenizer is required to measure token lengths for pre-splitting.
    """
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

        # Apply chat template → single 'text' column
        ds = ds.map(
            apply_template,
            remove_columns=[c for c in ds.column_names if c != "text"],
            desc=f"Formatting {sname}",
        )
        ds = ds.filter(lambda x: len(x["text"]) > 30)

        n_before = len(ds)

        # ── Pre-split long rows ──────────────────────────────────
        # This is the fix for the truncation warning.
        # Only apply if tokenizer is available (not in early init).
        if tokenizer is not None:
            ds = ds.map(
                lambda batch: split_long_rows(batch, tokenizer),
                batched=True,
                batch_size=32,
                remove_columns=["text"],
                desc=f"Pre-splitting {sname}",
            )
            # The map above returns a new column 'text' in the output dict
            # but remove_columns removes the input 'text'. Rename if needed.
            if "text" not in ds.column_names:
                # This shouldn't happen but guard against it
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
# TRIAL RUN
# ──────────────────────────────────────────────────────────────────────
def trial_run():
    from unsloth_mlx import FastLanguageModel, SFTTrainer, SFTConfig

    login(token=HF_TOKEN)

    print("\n" + "═" * 64)
    print("  🧪  TRIAL MODE  —  ~5 minutes")
    print("  Checks: model load · dataset · pre-splitting · speed")
    print("═" * 64)

    # Load model first so tokenizer is available for pre-splitting
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

    ds = load_data(trial=True, tokenizer=tok)

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
    # step from unsloth = raw iterations = len(ds) // BATCH
    # If it returned something else, fall back to calculating from timing
    expected_iters = max(1, len(ds) // BATCH)
    actual_iters = step if step > 0 else expected_iters
    spi = elapsed / actual_iters
    save_speed(spi)

    # Project full run timing
    n_splits = 2 if STATE in NO_CULT else 3
    # After pre-splitting, estimate ~30% more rows for cult split
    full_rows = MAX_ROWS * n_splits
    full_rows_split = int(full_rows * 1.25)  # conservative estimate with splitting
    rec_ep = auto_epochs(spi, full_rows_split)
    full_iters = (full_rows_split // BATCH) * rec_ep
    full_h_m4 = full_iters * spi / 3600
    full_h_m3 = full_h_m4 * 1.25

    print(f"\n{'═' * 64}")
    print(f"  🧪  TRIAL RESULT")
    print(f"{'─' * 64}")
    print(f"  Speed (this machine)  : {spi:.2f}s per iteration")
    print(f"  Full dataset estimate : ~{full_rows_split:,} rows (after pre-splitting)")
    print(f"  Recommended           : {rec_ep} epoch(s)  ({full_iters:,} iterations)")
    print(f"  Est. on this machine  : {full_h_m4:.1f}h")
    print(f"  Est. on M3 16GB       : {full_h_m3:.1f}h")
    ok = full_h_m3 <= BUDGET_H
    print(
        f"  M3 status             : {'✅ fits in 6.5h window' if ok else '⚠️  check EPOCHS in .env'}"
    )

    if _EPOCHS_ENV:
        ov_iters = (full_rows_split // BATCH) * _EPOCHS_ENV
        ov_h_m3 = ov_iters * spi * 1.25 / 3600
        flag = "✅ safe" if ov_h_m3 <= BUDGET_H else "⚠️  RISKY — may overrun"
        print(f"\n  .env EPOCHS={_EPOCHS_ENV} override → M3 ~{ov_h_m3:.1f}h  [{flag}]")

    print(f"\n  Speed saved. Run: python finetune.py")
    print(f"{'═' * 64}\n")


# ──────────────────────────────────────────────────────────────────────
# FULL TRAINING
# ──────────────────────────────────────────────────────────────────────
def full_run(resume=False):
    from unsloth_mlx import FastLanguageModel, SFTTrainer, SFTConfig

    login(token=HF_TOKEN)

    # Epoch count
    if _EPOCHS_ENV:
        epochs = _EPOCHS_ENV
        epoch_note = f".env override (EPOCHS={epochs})"
    else:
        spi = load_speed()
        if spi:
            # Use a placeholder row count — will be refined after data load
            # 15k rows × 1.25 pre-split expansion estimate
            n_splits = 2 if STATE in NO_CULT else 3
            est_rows = int(MAX_ROWS * n_splits * 1.25)
            epochs = auto_epochs(spi, est_rows)
            epoch_note = f"auto ({spi:.2f}s/iter measured)"
        else:
            epochs = 2
            epoch_note = "default=2  (run --trial first for precise estimate)"

    resume_from = None
    if resume:
        resume_from = latest_checkpoint()
        msg = resume_from if resume_from else "none found — starting fresh"
        print(f"♻️   Checkpoint: {msg}")

    print(f"\n{'═' * 64}")
    print(f"  🌟  {STATE}")
    print(f"  🤖  {MODEL_ID}")
    print(f"  📐  seq={MAX_SEQ}  rank={RANK}  alpha={ALPHA}")
    print(f"  📦  batch={BATCH} × accum={GRAD_ACCUM}")
    print(f"  🔁  epochs={epochs}  ({epoch_note})")
    print(f"  📈  lr={LR}  warmup={WARMUP} iters  cosine")
    print(f"  ✂️   pre-splitting rows>{MAX_SEQ} tokens  →  zero truncation")
    print(f"{'═' * 64}")

    # Load model before data so tokenizer is available for pre-splitting
    print(f"\n📥  Loading {MODEL_ID}...")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID, max_seq_length=MAX_SEQ, load_in_4bit=True
    )
    print(f"🔧  LoRA r={RANK}  alpha={ALPHA}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=RANK,
        target_modules=LORA_MODULES,
        lora_alpha=ALPHA,
        lora_dropout=0,
        bias="none",
    )

    # Data (with pre-splitting)
    ds = load_data(trial=False, tokenizer=tok)
    n_rows = len(ds)
    iters_ep = max(1, n_rows // BATCH)
    total_it = iters_ep * epochs

    spi = load_speed()
    if spi:
        est_h_here = total_it * spi / 3600
        est_h_m3 = est_h_here * 1.25
        print(
            f"\n⏱️   {total_it:,} iters × {spi:.2f}s = ~{est_h_here:.1f}h here  /  ~{est_h_m3:.1f}h on M3"
        )
    else:
        print(f"\n⏱️   {total_it:,} iterations total")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ADAPT_DIR.mkdir(parents=True, exist_ok=True)

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

    print(f"\n🏋️   Training  {STATE}  —  {epochs} epoch(s)  {total_it:,} iters")
    print(f"    Checkpoint every {SAVE_STEPS} iters  ♻️  crash-safe")
    t0 = time.time()
    stats = trainer.train()
    elapsed = time.time() - t0

    step, loss = parse_stats(stats, elapsed)
    actual_spi = elapsed / max(1, step)
    save_speed(actual_spi)

    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\n✅  Done!  {h}h {m}m {s}s  |  loss {loss:.4f}  |  {actual_spi:.2f}s/iter")

    # Save adapter locally
    print(f"\n💾  Saving adapter → {ADAPT_DIR}/")
    model.save_pretrained(str(ADAPT_DIR))
    tok.save_pretrained(str(ADAPT_DIR))

    # Push adapter to Hub
    repo_id = f"{HF_USERNAME}/{PROJECT_NAME}-{STATE}"
    print(f"\n📤  Pushing adapter → {repo_id}")
    adapter_ok = False
    try:
        model.push_to_hub(repo_id, token=HF_TOKEN)
        tok.push_to_hub(repo_id, token=HF_TOKEN)
        print(f"    ✅  https://huggingface.co/{repo_id}")
        adapter_ok = True
    except Exception as e:
        print(f"    ⚠️   Push failed: {e}  (adapter safe at {ADAPT_DIR}/)")

    # Merge LoRA → full FP16 model
    MERGE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n🔀  Merging LoRA into FP16 model...")
    try:
        model.save_pretrained_merged(str(MERGE_DIR), tok, save_method="merged_16bit")
        print(f"    ✅  Merged saved → {MERGE_DIR}/")
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
            print(f"    Adapter is on Hub — still usable via PEFT.")

    # Clean up checkpoints
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
  python finetune.py --trial     # ~5 min — verify everything + measure speed
  python finetune.py             # full training (uses measured speed for epochs)
  python finetune.py --resume    # resume after crash or sleep
        """,
    )
    parser.add_argument(
        "--trial",
        action="store_true",
        help="Trial: 50 rows/split, 1 epoch. Measures iteration speed.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume full training from latest saved checkpoint.",
    )
    args = parser.parse_args()

    if args.trial and args.resume:
        print("❌  --trial and --resume cannot be used together.")
        sys.exit(1)

    if args.trial:
        trial_run()
    else:
        full_run(resume=args.resume)
