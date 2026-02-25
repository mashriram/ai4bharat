import os
import ast
from datasets import load_dataset, DatasetDict, Dataset
from huggingface_hub import login

# ==========================================
# 🛑 ORGANIZER SETTINGS
# ==========================================
HF_TOKEN = ""
login(token=HF_TOKEN)

HF_REPO_NAME = "mashriram/AI4Bharat-Languages-and-Cultures"

# ==========================================
# 📦 OUTPUT FORMAT — Model Agnostic
#
# Instead of ChatML/Qwen blobs, every row is stored as plain columns:
#   instruction : str  — the question / prompt
#   input       : str  — optional context (empty string if none)
#   response    : str  — the answer / completion
#
# This lets anyone apply their own chat template:
#   Qwen3   → <|im_start|>user\n{instruction}\n{input}<|im_end|>...
#   LLaMA3  → <|user|>\n{instruction}\n{input}<|assistant|>...
#   Mistral → [INST]{instruction}[/INST]{response}
#   etc.
# ==========================================

# ==========================================
# 🗺️ 30-STATE MASTER CONFIGURATION (FULLY VERIFIED)
# ==========================================
STATE_CONFIG = {
    # --- ZONE 1: SOUTH INDIA ---
    "Tamil_Nadu": {
        "indic": "tam_Taml",
        "indic_latn": "tam_Latn",
        "conv": "Tensoic/GPTeacher-Tamil",
        "cult_type": "wiki",
        "cult_val": "20231101.ta",
    },
    "Kerala": {
        "indic": "mal_Mlym",
        "indic_latn": "mal_Latn",
        "conv": "Tensoic/GPTeacher-Malayalam",
        "cult_type": "wiki",
        "cult_val": "20231101.ml",
    },
    "Karnataka": {
        "indic": "kan_Knda",
        "indic_latn": "kan_Latn",
        "conv": "Tensoic/GPTeacher-Kannada",
        "cult_type": "wiki",
        "cult_val": "20231101.kn",
    },
    "Andhra_Pradesh": {
        "indic": "tel_Telu",
        "indic_latn": "tel_Latn",
        "conv": "Telugu-LLM-Labs/telugu_alpaca_yahma_cleaned_filtered_romanized",
        "cult_type": "wiki",
        "cult_val": "20231101.te",
    },
    "Telangana": {
        "indic": "tel_Telu",
        "indic_latn": "tel_Latn",
        "conv": "Tensoic/GPTeacher-Telugu",
        "cult_type": "wikiqa",
        "cult_val": ["Telangana", "Hyderabad", "Charminar"],
    },
    # --- ZONE 2: WEST INDIA ---
    "Maharashtra": {
        "indic": "mar_Deva",
        "indic_latn": "mar_Latn",
        "conv": "Tensoic/GPTeacher-Marathi",
        "cult_type": "wiki",
        "cult_val": "20231101.mr",
    },
    "Gujarat": {
        "indic": "guj_Gujr",
        "indic_latn": "guj_Latn",
        "conv": "Tensoic/GPTeacher-Gujarati",
        "cult_type": "wiki",
        "cult_val": "20231101.gu",
    },
    "Goa": {
        "indic": "eng_Latn",
        "indic_latn": "eng_Latn",
        "conv": "Tensoic/GPTeacher-Konkani",
        "cult_type": "wiki",
        "cult_val": "20231101.gom",
    },
    "Rajasthan": {
        "indic": "hin_Deva",
        "indic_latn": "hin_Latn",
        "conv": "Tensoic/GPTeacher-Hindi",
        "cult_type": "wikiqa",
        "cult_val": ["Rajasthan", "Jaipur", "Thar"],
    },
    # --- ZONE 3: NORTH INDIA ---
    "Punjab": {
        "indic": "pan_Guru",
        "indic_latn": "pan_Latn",
        "conv": "Tensoic/GPTeacher-Punjabi",
        "cult_type": "wiki",
        "cult_val": "20231101.pa",
    },
    "Haryana": {
        "indic": "hin_Deva",
        "indic_latn": "hin_Latn",
        "conv": "Tensoic/GPTeacher-Hindi",
        "cult_type": "wikiqa",
        "cult_val": ["Haryana", "Kurukshetra"],
    },
    "Himachal_Pradesh": {
        "indic": "hin_Deva",
        "indic_latn": "hin_Latn",
        "conv": "Tensoic/GPTeacher-Hindi",
        "cult_type": "wikiqa",
        "cult_val": ["Himachal Pradesh", "Shimla"],
    },
    "Uttarakhand": {
        "indic": "hin_Deva",
        "indic_latn": "hin_Latn",
        "conv": "Tensoic/GPTeacher-Hindi",
        "cult_type": "wiki",
        "cult_val": "20231101.sa",
    },
    "Uttar_Pradesh": {
        "indic": "hin_Deva",
        "indic_latn": "hin_Latn",
        "conv": "Tensoic/GPTeacher-Hindi",
        "cult_type": "wikiqa",
        "cult_val": ["Uttar Pradesh", "Varanasi", "Awadh"],
    },
    # --- ZONE 4: CENTRAL & EAST INDIA ---
    "Madhya_Pradesh": {
        "indic": "hin_Deva",
        "indic_latn": "hin_Latn",
        "conv": "Tensoic/GPTeacher-Hindi",
        "cult_type": "wikiqa",
        "cult_val": ["Madhya Pradesh", "Bhopal"],
    },
    "Chhattisgarh": {
        "indic": "hin_Deva",
        "indic_latn": "hin_Latn",
        "conv": "Tensoic/GPTeacher-Hindi",
        "cult_type": "wikiqa",
        "cult_val": ["Chhattisgarh", "Raipur", "Bastar"],
    },
    "Bihar": {
        "indic": "hin_Deva",
        "indic_latn": "hin_Latn",
        "conv": "Tensoic/GPTeacher-Hindi",
        "cult_type": "wiki",
        "cult_val": "20231101.mai",
    },
    "Jharkhand": {
        "indic": "hin_Deva",
        "indic_latn": "hin_Latn",
        "conv": "Tensoic/GPTeacher-Hindi",
        "cult_type": "wiki",
        "cult_val": "20231101.sat",
    },
    "Odisha": {
        "indic": "ory_Orya",
        "indic_latn": "ory_Latn",
        "conv": "OdiaGenAI/Odia_Alpaca_instructions_52k",
        "cult_type": "wiki",
        "cult_val": "20231101.or",
    },
    "West_Bengal": {
        "indic": "ben_Beng",
        "indic_latn": "ben_Latn",
        "conv": "Tensoic/GPTeacher-Bangla",
        "cult_type": "wiki",
        "cult_val": "20231101.bn",
    },
    # --- ZONE 5: NORTH-EAST INDIA ---
    "Assam": {
        "indic": "asm_Beng",
        "indic_latn": "asm_Latn",
        "conv": "Tensoic/GPTeacher-Assamese",
        "cult_type": "wiki",
        "cult_val": "20231101.as",
    },
    "Arunachal_Pradesh": {
        "indic": "eng_Latn",
        "indic_latn": "eng_Latn",
        "conv": "yahma/alpaca-cleaned",
        "cult_type": "wikiqa",
        "cult_val": ["Arunachal Pradesh", "Tawang"],
    },
    "Manipur": {
        "indic": "eng_Latn",
        "indic_latn": "eng_Latn",
        "conv": "yahma/alpaca-cleaned",
        "cult_type": "wiki",
        "cult_val": "20231101.mni",
    },
    "Meghalaya": {
        "indic": "eng_Latn",
        "indic_latn": "eng_Latn",
        "conv": "yahma/alpaca-cleaned",
        "cult_type": "wikiqa",
        "cult_val": ["Meghalaya", "Shillong", "Khasi"],
    },
    "Mizoram": {
        "indic": "eng_Latn",
        "indic_latn": "eng_Latn",
        "conv": "yahma/alpaca-cleaned",
        "cult_type": "wikiqa",
        "cult_val": ["Mizoram", "Aizawl", "Mizo"],
    },
    "Nagaland": {
        "indic": "eng_Latn",
        "indic_latn": "eng_Latn",
        "conv": "yahma/alpaca-cleaned",
        "cult_type": "wikiqa",
        "cult_val": ["Nagaland", "Kohima", "Hornbill"],
    },
    "Tripura": {
        "indic": "ben_Beng",
        "indic_latn": "ben_Latn",
        "conv": "Tensoic/GPTeacher-Bangla",
        "cult_type": "wikiqa",
        "cult_val": ["Tripura", "Agartala", "Kokborok"],
    },
    "Sikkim": {
        "indic": "eng_Latn",
        "indic_latn": "eng_Latn",
        "conv": "Telugu-LLM-Labs/nepali_alpaca_yahma_cleaned_filtered",
        "cult_type": "wiki",
        "cult_val": "20231101.ne",
    },
    # --- ZONE 6: UNION TERRITORIES ---
    "Jammu_and_Kashmir": {
        "indic": "urd_Arab",
        "indic_latn": "urd_Latn",
        "conv": "Tensoic/GPTeacher-Urdu",
        "cult_type": "wiki",
        "cult_val": "20231101.ks",
    },
    "Delhi": {
        "indic": "hin_Deva",
        "indic_latn": "hin_Latn",
        "conv": "Tensoic/GPTeacher-Hindi",
        "cult_type": "wikiqa",
        "cult_val": ["New Delhi", "Mughal", "Red Fort"],
    },
}

# ==========================================
# ⚙️ HELPER FUNCTIONS
# ==========================================


def make_row(instruction, response, input_ctx=""):
    """
    Returns a plain dict with consistent columns.
    No chat template applied — caller can do that themselves.
    """
    return {
        "instruction": instruction.strip(),
        "input": input_ctx.strip(),  # empty string if no context
        "response": response.strip(),
    }


def extract_pair_from_cell(cell):
    """
    Extracts list of (q, a) tuples from a WikiHow/Dolly_T cell.

    Confirmed schema:
        cell = [['question', 'answer']]          ← single-turn
        cell = [['q1','a1'], ['q2','a2'], ...]   ← multi-turn
    """
    if isinstance(cell, str):
        try:
            cell = ast.literal_eval(cell)
        except Exception:
            return []

    if not isinstance(cell, list) or len(cell) == 0:
        return []

    pairs = []
    for item in cell:
        if isinstance(item, list) and len(item) >= 2:
            q, a = str(item[0]).strip(), str(item[1]).strip()
            if q and a:
                pairs.append((q, a))
    return pairs


def safe_extract_conv(example):
    """
    Extracts instruction / input / response from Alpaca-format datasets.
    Returns a dict with those 3 columns, or all-empty strings if unparseable.
    """
    instr = next(
        (
            example[k]
            for k in ["instruction", "prompt", "text"]
            if k in example and example[k]
        ),
        "",
    )
    resp = next(
        (
            example[k]
            for k in ["output", "response", "target"]
            if k in example and example[k]
        ),
        "",
    )
    # Optional context field
    inp = ""
    if "input" in example and example["input"] and str(example["input"]).strip():
        inp = str(example["input"]).strip()

    return {
        "instruction": str(instr).strip(),
        "input": inp,
        "response": str(resp).strip(),
    }


def is_valid_row(example):
    return bool(example["instruction"]) and bool(example["response"])


# ==========================================
# 🚀 PRE-LOAD GLOBAL DATASETS (once, reused for all 30 states)
# ==========================================
print("⏳ Pre-loading global datasets...")

indic_wikihow = load_dataset("ai4bharat/indic-align", "WikiHow", split="train")
indic_dolly = load_dataset("ai4bharat/indic-align", "Dolly_T", split="train")
indic_anudesh = load_dataset("ai4bharat/indic-align", "Anudesh", split="train")
# Anudesh = English only (interactions column), useful only for eng_Latn states

global_wikiqa = load_dataset("microsoft/wiki_qa", split="train").filter(
    lambda x: x["label"] == 1
)

print(f"✅ WikiHow : {len(indic_wikihow)} rows")
print(f"✅ Dolly_T : {len(indic_dolly)} rows")
print(f"✅ Anudesh : {len(indic_anudesh)} rows (English only)")
print(f"✅ WikiQA  : {len(global_wikiqa)} answered QA pairs\n")

# ==========================================
# 🔁 MAIN PIPELINE
# ==========================================
for state, config in STATE_CONFIG.items():
    print(f"\n{'=' * 60}")
    print(f"🌟 PROCESSING: {state}")
    print(f"{'=' * 60}")
    splits = {}

    # ----------------------------------------------------------
    # SPLIT 1 — INDIC (Linguistic Foundation Layer)
    #
    # Schema A — WikiHow/Dolly_T:
    #   row['tam_Taml'] = [['question', 'answer']]
    #   → extract with extract_pair_from_cell()
    #
    # Schema B — Anudesh:
    #   row['interactions'] = [['english q', 'english a'], ...]
    #   → English only, used only for eng_Latn states
    # ----------------------------------------------------------
    print(f"  [1/3] Indic split: {config['indic']} + {config['indic_latn']}")
    try:
        rows = []  # list of dicts: {instruction, input, response}
        seen = set()
        target_keys = set([config["indic"], config["indic_latn"]])
        is_english = config["indic"] == "eng_Latn"

        # Schema A — WikiHow + Dolly_T
        for dataset in [indic_wikihow, indic_dolly]:
            for script_key in target_keys:
                if script_key in dataset.column_names:
                    for row in dataset:
                        for q, a in extract_pair_from_cell(row[script_key]):
                            key = (q, a)
                            if key not in seen:
                                seen.add(key)
                                rows.append(make_row(q, a))

        # Schema B — Anudesh (English only)
        if is_english:
            for row in indic_anudesh:
                # Extract interactions based on row type
                interactions = []
                if isinstance(row, dict):
                    interactions = row.get("interactions", [])
                elif hasattr(row, "interactions"):
                    interactions = getattr(row, "interactions", [])

                # Ensure interactions is a list before iterating
                if not isinstance(interactions, list):
                    interactions = []

                for turn in interactions:
                    if isinstance(turn, list) and len(turn) >= 2:
                        q, a = str(turn[0]).strip(), str(turn[1]).strip()
                        if q and a:
                            key = (q, a)
                            if key not in seen:
                                seen.add(key)
                                rows.append(make_row(q, a))

        splits["indic"] = Dataset.from_dict(
            {
                "instruction": [r["instruction"] for r in rows],
                "input": [r["input"] for r in rows],
                "response": [r["response"] for r in rows],
            }
        )
        print(f"  ✅ Indic: {len(rows)} unique rows.")

    except Exception as e:
        print(f"  ⚠️ Indic split error: {e}")
        splits["indic"] = Dataset.from_dict(
            {"instruction": [], "input": [], "response": []}
        )

    # ----------------------------------------------------------
    # SPLIT 2 — CONV (Conversational Fluency Layer)
    # ----------------------------------------------------------
    print(f"  [2/3] Conv split from: {config['conv']}")
    try:
        conv_ds = load_dataset(config["conv"], split="train")
        mapped_conv = conv_ds.map(
            safe_extract_conv, remove_columns=conv_ds.column_names
        )
        splits["conv"] = mapped_conv.filter(is_valid_row)
        print(f"  ✅ Conv: {len(splits['conv'])} rows.")
    except Exception as e:
        print(f"  ⚠️ Conv split error: {e}")

    # ----------------------------------------------------------
    # SPLIT 3 — CULT (Cultural Knowledge Layer)
    # ----------------------------------------------------------
    print(f"  [3/3] Cult split ({config['cult_type'].upper()})")
    try:
        if config["cult_type"] == "wiki":
            print(f"        Wikipedia dump: {config['cult_val']}")
            wiki_ds = load_dataset(
                "wikimedia/wikipedia", config["cult_val"], split="train"
            )

            def map_wiki(example):
                return make_row(
                    instruction=f"Provide a detailed explanation about {example['title']}.",
                    response=str(example["text"])[:6000],
                )

            splits["cult"] = wiki_ds.map(map_wiki, remove_columns=wiki_ds.column_names)
            print(f"  ✅ Cult (wiki): {len(splits['cult'])} articles.")

        elif config["cult_type"] == "wikiqa":
            keywords = config["cult_val"]
            state_wikiqa = global_wikiqa.filter(
                lambda x, kw=keywords: any(k.lower() in str(x).lower() for k in kw)
            )

            def map_wikiqa(example, s=state):
                return make_row(
                    instruction=(
                        f"Explain this aspect of {s.replace('_', ' ')} in detail: "
                        f"{example['question']}"
                    ),
                    response=example["answer"],
                    input_ctx=example[
                        "document_title"
                    ],  # context goes in 'input' column
                )

            splits["cult"] = state_wikiqa.map(
                map_wikiqa, remove_columns=state_wikiqa.column_names
            )
            print(f"  ✅ Cult (wikiqa): {len(splits['cult'])} QA rows.")

    except Exception as e:
        print(f"  ⚠️ Cult split error: {e}")

    # ----------------------------------------------------------
    # PUSH TO HUB
    # ----------------------------------------------------------
    print(f"\n  📤 Uploading {state}...")
    valid_splits = {k: v for k, v in splits.items() if len(v) > 0}

    if valid_splits:
        state_dataset = DatasetDict(valid_splits)
        state_dataset.push_to_hub(HF_REPO_NAME, config_name=state)
        total = sum(len(v) for v in valid_splits.values())
        print(
            f"  🎉 {state} done! ({total} rows | splits: {list(valid_splits.keys())})"
        )
    else:
        print(f"  ❌ Skipped {state} — all splits empty.")

print("\n" + "=" * 60)
print("🚀 ALL 30 STATES PUSHED SUCCESSFULLY!")
print("=" * 60)
