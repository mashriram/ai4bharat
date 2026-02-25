import argparse
import os
import sys
import time
from getpass import getpass
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login, HfApi

# ==========================================
# 🤖 HARDWARE-MAXIMA CONFIG
# ==========================================
MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MAX_SEQ_LENGTH = 4096    # Max context for deep stories
LORA_RANK = 64           # High rank to memorize state facts
LORA_ALPHA = 128         # 2x Rank
BATCH_SIZE = 1           # Essential for 4096 length on 16GB
GRAD_ACCUM = 32          # Effective Batch Size = 32 for smooth Eval
LEARNING_RATE = 1e-4     # Stable learning for high rank
EPOCHS = 2               # 2 Epochs over 15k rows ≈ 8 hours
MAX_ROWS_PER_SPLIT = 5000 # Total 15,000 rows to fit 9-hour window

# ==========================================
# 🔐 CREDENTIALS & CLI
# ==========================================
def setup_env():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, required=True, help="State name (e.g. Tamil_Nadu)")
    parser.add_argument("--dataset_repo", type=str, required=True, help="Your Hub repo with the 30 states")
    args = parser.parse_args()

    print("\n" + "═"*60 + "\n🔐 HUGGING FACE AUTHENTICATION\n" + "═"*60)
    hf_token = os.getenv("HF_TOKEN") or getpass("Enter HF WRITE Token: ")
    login(token=hf_token)
    
    hf_user = input("Enter HF Username: ").strip()
    project = input("Project Name [AI4Bharat-State-Expert]: ").strip() or "AI4Bharat-State-Expert"
    
    return args, hf_token, hf_user, project

# ==========================================
# 📥 DATASET PREP
# ==========================================
def get_dataset(repo, state):
    print(f"\n📥 Loading and Slicing data for {state}...")
    ds = load_dataset(repo, name=state)
    
    blended = []
    for split in ['indic', 'conv', 'cult']:
        if split in ds:
            # Slice to ensure we finish in 9 hours
            slice_size = min(MAX_ROWS_PER_SPLIT, len(ds[split]))
            blended.append(ds[split].select(range(slice_size)))
            print(f"   ✅ Added {slice_size} rows from {split}")
    
    final_ds = concatenate_datasets(blended).shuffle(seed=3407)
    print(f"📊 Total Training Samples: {len(final_ds)}")
    return final_ds

# ==========================================
# 🚀 TRAINING EXECUTION
# ==========================================
def run_training():
    args, hf_token, hf_user, project = setup_env()
    
    from unsloth_mlx import FastLanguageModel, SFTTrainer, SFTConfig

    # 1. Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_ID,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = True,
    )

    # 2. Add LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_RANK,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = LORA_ALPHA,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = True, # Critical for 16GB RAM
    )

    # 3. Load Data
    train_ds = get_dataset(args.dataset_repo, args.state)

    # 4. Trainer Config
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_ds,
        args = SFTConfig(
            output_dir = f"checkpoints/{args.state}",
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            num_train_epochs = EPOCHS,
            learning_rate = LEARNING_RATE,
            lr_scheduler_type = "cosine",
            warmup_ratio = 0.1,
            max_seq_length = MAX_SEQ_LENGTH,
            dataset_text_field = "text",
            logging_steps = 1,
            save_total_limit = 1,
            seed = 3407,
        ),
    )

    # 5. Train
    print(f"\n🏋️ Training {args.state}... Estimated time: 7-8 Hours")
    trainer.train()

    # 6. Save & Merge
    repo_id = f"{hf_user}/{project}-{args.state}"
    print(f"\n💾 Training Finished. Finalizing {args.state}...")

    # Upload Adapters
    print("   -> Uploading LoRA Adapters...")
    model.push_to_hub(repo_id, token=hf_token)
    tokenizer.push_to_hub(repo_id, token=hf_token)

    # Merge to FP16 (Final step for 16GB RAM)
    print("   -> Attempting FP16 Merge (Final Hour)...")
    try:
        local_path = f"merged_model_{args.state}"
        model.save_pretrained_merged(local_path, tokenizer, save_method="merged_16bit")
        
        api = HfApi()
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token
        )
        print(f"🎉 SUCCESS! Full model live: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"⚠️ Merge failed: {e}. Adapters are safe on Hub.")

if __name__ == "__main__":
    run_training()