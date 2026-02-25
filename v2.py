import os
import ast
from datasets import load_dataset, DatasetDict, Dataset
from huggingface_hub import login

# ==========================================
# 🛑 ORGANIZER SETTINGS
# ==========================================
HF_TOKEN = ""  # Replace with your token if needed
login(token=HF_TOKEN)

# ⚠️ IMPORTANT: Use a NEW repo name to ensure a clean slate
HF_REPO_NAME = "mashriram/AI4Bharat-Indic-Languages-and-Cultures"

# ==========================================
# 🗺️ 30-STATE MASTER CONFIGURATION (MERGED & FIXED)
# ==========================================
# This config integrates the "Patch" logic directly.
# Types of Culture Data:
#   "wiki"          -> Full Native Wikipedia Dump (Good for strong regional langs like Tamil, Malayalam)
#   "wiki_filtered" -> Native Wiki Dump filtered by specific keywords (Good for Hindi belt)
#   "wikiqa"        -> Microsoft WikiQA filtered by English keywords (Fallback for NE states)

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
        # ✅ FIX APPLIED: Using the correct Romanized dataset
        "conv": "Telugu-LLM-Labs/telugu_alpaca_yahma_cleaned_filtered_romanized",
        "cult_type": "wiki",
        "cult_val": "20231101.te",
    },
    "Telangana": {
        "indic": "tel_Telu",
        "indic_latn": "tel_Latn",
        "conv": "Tensoic/GPTeacher-Telugu",
        # ✅ FIX APPLIED: Using Native Wiki Filtered
        "cult_type": "wiki_filtered",
        "cult_val": "20231101.te",
        "cult_keywords": ["తెలంగాణ", "హైదరాబాద్", "చార్మినార్", "నిజాం"],
        "cult_fallback": ["Telangana", "Hyderabad", "Charminar"],
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
        # ✅ FIX APPLIED: Using Native Wiki Filtered
        "cult_type": "wiki_filtered",
        "cult_val": "20231101.hi",
        "cult_keywords": ["राजस्थान", "जयपुर", "थार", "राजपूत", "उदयपुर"],
        "cult_fallback": ["Rajasthan", "Jaipur", "Thar"],
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
        # ✅ FIX APPLIED: Using Native Wiki Filtered
        "cult_type": "wiki_filtered",
        "cult_val": "20231101.hi",
        "cult_keywords": ["हरियाणा", "कुरुक्षेत्र", "गुरुग्राम", "पानीपत"],
        "cult_fallback": ["Haryana", "Kurukshetra"],
    },
    "Himachal_Pradesh": {
        "indic": "hin_Deva",
        "indic_latn": "hin_Latn",
        "conv": "Tensoic/GPTeacher-Hindi",
        # ✅ FIX APPLIED: Using Native Wiki Filtered
        "cult_type": "wiki_filtered",
        "cult_val": "20231101.hi",
        "cult_keywords": ["हिमाचल प्रदेश", "शिमला", "मनाली", "धर्मशाला"],
        "cult_fallback": ["Himachal Pradesh", "Shimla", "Manali"],
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
        # ✅ FIX APPLIED: Using Native Wiki Filtered
        "cult_type": "wiki_filtered",
        "cult_val": "20231101.hi",
        "cult_keywords": ["उत्तर प्रदेश", "वाराणसी", "अवध", "लखनऊ", "आगरा"],
        "cult_fallback": ["Uttar Pradesh", "Varanasi", "Lucknow", "Agra"],
    },
    # --- ZONE 4: CENTRAL & EAST INDIA ---
    "Madhya_Pradesh": {
        "indic": "hin_Deva",
        "indic_latn": "hin_Latn",
        "conv": "Tensoic/GPTeacher-Hindi",
        # ✅ FIX APPLIED: Using Native Wiki Filtered
        "cult_type": "wiki_filtered",
        "cult_val": "20231101.hi",
        "cult_keywords": ["मध्य प्रदेश", "भोपाल", "इंदौर", "ग्वालियर"],
        "cult_fallback": ["Madhya Pradesh", "Bhopal", "Indore"],
    },
    "Chhattisgarh": {
        "indic": "hin_Deva",
        "indic_latn": "hin_Latn",
        "conv": "Tensoic/GPTeacher-Hindi",
        # ✅ FIX APPLIED: Using Native Wiki Filtered
        "cult_type": "wiki_filtered",
        "cult_val": "20231101.hi",
        "cult_keywords": ["छत्तीसगढ़", "रायपुर", "बस्तर", "बिलासपुर"],
        "cult_fallback": ["Chhattisgarh", "Raipur", "Bastar"],
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
        "cult_val": ["Arunachal Pradesh", "Tawang", "monastery"],
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
        "cult_val": ["Meghalaya", "Shillong", "Khasi", "Cherrapunji"],
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
        "cult_val": ["Nagaland", "Kohima", "Naga", "Hornbill"],
    },
    "Tripura": {
        "indic": "ben_Beng",
        "indic_latn": "ben_Latn",
        "conv": "Tensoic/GPTeacher-Bangla",
        # ✅ FIX APPLIED: Using Native Wiki Filtered (Bengali)
        "cult_type": "wiki_filtered",
        "cult_val": "20231101.bn",
        "cult_keywords": ["ত্রিপুরা", "আগরতলা", "কোকবরক"],
        "cult_fallback": ["Tripura", "Agartala"],
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
        # ✅ FIX APPLIED: Using Native Wiki Filtered
        "cult_type": "wiki_filtered",
        "cult_val": "20231101.hi",
        "cult_keywords": ["दिल्ली", "नई दिल्ली", "लाल किला", "मुगल"],
        "cult_fallback": ["Delhi", "New Delhi", "Red Fort", "Mughal"],
    },
}

# ==========================================
# ⚙️ HELPER FUNCTIONS
# ==========================================


def make_row(instruction, response, input_ctx=""):
    """Returns a plain dict with consistent columns."""
    return {
        "instruction": instruction.strip(),
        "input": input_ctx.strip(),
        "response": response.strip(),
    }


def extract_pair_from_cell(cell):
    """Extracts list of (q, a) tuples from a WikiHow/Dolly_T cell."""
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
    """Extracts info from Alpaca/Instruction style datasets."""
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
    inp = ""
    if "input" in example and example["input"] and str(example["input"]).strip():
        inp = str(example["input"]).strip()
    return make_row(str(instr), str(resp), inp)


def is_valid_row(example):
    return bool(example["instruction"]) and bool(example["response"])


# ==========================================
# 🚀 PRE-LOAD GLOBAL DATASETS
# ==========================================
print("⏳ Pre-loading global datasets...")

indic_wikihow = load_dataset("ai4bharat/indic-align", "WikiHow", split="train")
indic_dolly = load_dataset("ai4bharat/indic-align", "Dolly_T", split="train")
indic_anudesh = load_dataset("ai4bharat/indic-align", "Anudesh", split="train")
# Pre-load WikiQA only once
global_wikiqa = load_dataset("microsoft/wiki_qa", split="train").filter(
    lambda x: x["label"] == 1
)

print(f"✅ WikiHow : {len(indic_wikihow)} rows")
print(f"✅ Dolly_T : {len(indic_dolly)} rows")
print(f"✅ WikiQA  : {len(global_wikiqa)} rows")

# ==========================================
# 🔁 MAIN PROCESSING LOOP
# ==========================================
for state, config in STATE_CONFIG.items():
    print(f"\n{'=' * 60}")
    print(f"🌟 PROCESSING: {state}")
    print(f"{'=' * 60}")
    splits = {}

    # ----------------------------------------------------------
    # 1. INDIC SPLIT (Linguistic Foundation)
    # ----------------------------------------------------------
    print(f"  [1/3] Indic split: {config['indic']} + {config['indic_latn']}")
    try:
        rows = []
        seen = set()
        target_keys = set([config["indic"], config["indic_latn"]])
        is_english = config["indic"] == "eng_Latn"

        # Schema A: WikiHow + Dolly_T
        for dataset in [indic_wikihow, indic_dolly]:
            for script_key in target_keys:
                if script_key in dataset.column_names:
                    for row in dataset:
                        for q, a in extract_pair_from_cell(row[script_key]):
                            key = (q, a)
                            if key not in seen:
                                seen.add(key)
                                rows.append(make_row(q, a))

        # Schema B: Anudesh (English only)
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
    # 2. CONV SPLIT (Conversational)
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
        splits["conv"] = Dataset.from_dict(
            {"instruction": [], "input": [], "response": []}
        )

    # ----------------------------------------------------------
    # 3. CULT SPLIT (Cultural Knowledge - The Complex Part)
    # ----------------------------------------------------------
    ctype = config["cult_type"]
    print(f"  [3/3] Cult split ({ctype})")
    try:
        cult_ds = None

        # Logic A: FULL WIKI DUMP
        if ctype == "wiki":
            print(f"        Loading full Wikipedia: {config['cult_val']}")
            wiki_ds = load_dataset(
                "wikimedia/wikipedia", config["cult_val"], split="train"
            )

            def map_wiki(example):
                return make_row(
                    f"Provide a detailed explanation about {example['title']}.",
                    str(example["text"])[:6000],
                )

            cult_ds = wiki_ds.map(map_wiki, remove_columns=wiki_ds.column_names)

        # Logic B: FILTERED WIKI DUMP (Native keywords)
        elif ctype == "wiki_filtered":
            print(f"        Loading Wikipedia {config['cult_val']} and filtering...")
            wiki_ds = load_dataset(
                "wikimedia/wikipedia", config["cult_val"], split="train"
            )

            # 1. Try Native keywords
            filtered = wiki_ds.filter(
                lambda x, kw=config["cult_keywords"]: any(k in x["title"] for k in kw)
            )
            print(f"        Native matches: {len(filtered)}")

            # 2. Fallback to Romanized if needed
            if len(filtered) < 10:
                print("        ⚠️ Low match count, trying fallback keywords...")
                filtered = wiki_ds.filter(
                    lambda x, kw=config["cult_fallback"]: any(
                        k.lower() in x["title"].lower() for k in kw
                    )
                )
                print(f"        Fallback matches: {len(filtered)}")

            if len(filtered) > 0:

                def map_wiki(example):
                    return make_row(
                        f"Provide a detailed explanation about {example['title']}.",
                        str(example["text"])[:6000],
                    )

                cult_ds = filtered.map(map_wiki, remove_columns=filtered.column_names)

        # Logic C: WIKIQA (English Fallback for NE states)
        elif ctype == "wikiqa":
            print("        Filtering WikiQA...")
            # Filter specifically on Question or Document Title column
            filtered_qa = global_wikiqa.filter(
                lambda x, kw=config["cult_val"]: any(
                    k.lower() in (x["question"] + " " + x["document_title"]).lower()
                    for k in kw
                )
            )
            if len(filtered_qa) > 0:

                def map_wikiqa(example, s=state):
                    return make_row(
                        f"Explain this aspect of {s.replace('_', ' ')} in detail: {example['question']}",
                        example["answer"],
                        example["document_title"],
                    )

                cult_ds = filtered_qa.map(
                    map_wikiqa, remove_columns=filtered_qa.column_names
                )

        # Final Assignment
        if cult_ds and len(cult_ds) > 0:
            splits["cult"] = cult_ds
            print(f"  ✅ Cult: {len(cult_ds)} rows.")
        else:
            print("  ⚠️ Cult split resulted in 0 rows.")
            splits["cult"] = Dataset.from_dict(
                {"instruction": [], "input": [], "response": []}
            )

    except Exception as e:
        print(f"  ⚠️ Cult split error: {e}")
        splits["cult"] = Dataset.from_dict(
            {"instruction": [], "input": [], "response": []}
        )

    # ----------------------------------------------------------
    # PUSH TO HUB
    # ----------------------------------------------------------
    print(f"\n  📤 Uploading {state} to {HF_REPO_NAME}...")

    # Only push non-empty splits to avoid errors, but create an empty dict if all fail
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
print(f"🚀 ALL STATES PROCESSED & UPLOADED TO {HF_REPO_NAME}!")
print("=" * 60)
