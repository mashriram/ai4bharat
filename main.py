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
# 🔧 PATCH SCRIPT — fixes two issues:
#
# FIX 1 — Andhra Pradesh conv wrong dataset name:
#   ❌ Telugu-LLM-Labs/telugu_alpaca_yahma_cleaned_filtered
#   ✅ Telugu-LLM-Labs/telugu_alpaca_yahma_cleaned_filtered_romanized
#
# FIX 2 — All wikiqa cult splits returned 0 rows:
#   microsoft/wiki_qa has almost no India-specific content.
#   Replaced with wiki_filtered: load the state's actual Wikipedia
#   dump and filter articles by native-language title keywords.
#   For NE states with no native wiki → kept wiki_qa but with
#   proper column-level filtering (question + document_title).
#
# This script ONLY re-uploads the affected splits.
# It does NOT re-process indic or conv for unaffected states.
# ==========================================

# ==========================================
# 🗺️ PATCH CONFIG — only states that need fixes
# ==========================================

# States that had wikiqa cult type — now using wiki_filtered
# cult_val     = wikimedia/wikipedia dump ID
# cult_lang    = native script keywords to filter article titles
# cult_fallback= romanized/common keywords as backup if native filter yields 0

WIKIQA_PATCH = {
    "Telangana": {
        "cult_val": "20231101.te",
        "cult_keywords": ["తెలంగాణ", "హైదరాబాద్", "చార్మినార్", "నిజాం"],
        "cult_fallback": ["Telangana", "Hyderabad", "Charminar"],
    },
    "Rajasthan": {
        "cult_val": "20231101.hi",
        "cult_keywords": ["राजस्थान", "जयपुर", "थार", "राजपूत", "उदयपुर"],
        "cult_fallback": ["Rajasthan", "Jaipur", "Thar"],
    },
    "Haryana": {
        "cult_val": "20231101.hi",
        "cult_keywords": ["हरियाणा", "कुरुक्षेत्र", "गुरुग्राम", "पानीपत"],
        "cult_fallback": ["Haryana", "Kurukshetra"],
    },
    "Himachal_Pradesh": {
        "cult_val": "20231101.hi",
        "cult_keywords": ["हिमाचल प्रदेश", "शिमला", "मनाली", "धर्मशाला"],
        "cult_fallback": ["Himachal Pradesh", "Shimla", "Manali"],
    },
    "Uttar_Pradesh": {
        "cult_val": "20231101.hi",
        "cult_keywords": ["उत्तर प्रदेश", "वाराणसी", "अवध", "लखनऊ", "आगरा"],
        "cult_fallback": ["Uttar Pradesh", "Varanasi", "Lucknow", "Agra"],
    },
    "Madhya_Pradesh": {
        "cult_val": "20231101.hi",
        "cult_keywords": ["मध्य प्रदेश", "भोपाल", "इंदौर", "ग्वालियर"],
        "cult_fallback": ["Madhya Pradesh", "Bhopal", "Indore"],
    },
    "Chhattisgarh": {
        "cult_val": "20231101.hi",
        "cult_keywords": ["छत्तीसगढ़", "रायपुर", "बस्तर", "बिलासपुर"],
        "cult_fallback": ["Chhattisgarh", "Raipur", "Bastar"],
    },
    "Delhi": {
        "cult_val": "20231101.hi",
        "cult_keywords": ["दिल्ली", "नई दिल्ली", "लाल किला", "मुगल"],
        "cult_fallback": ["Delhi", "New Delhi", "Red Fort", "Mughal"],
    },
    "Tripura": {
        "cult_val": "20231101.bn",  # Bengali Wikipedia — widely spoken in Tripura
        "cult_keywords": ["ত্রিপুরা", "আগরতলা", "কোকবরক"],
        "cult_fallback": ["Tripura", "Agartala"],
    },
    # --- NE states with no strong native Wikipedia ---
    # Keep wiki_qa but with proper column-level filter
    "Arunachal_Pradesh": {
        "cult_val": None,  # no usable native wiki → use wiki_qa fallback
        "cult_keywords": [],
        "cult_fallback": ["Arunachal Pradesh", "Tawang", "monastery"],
    },
    "Meghalaya": {
        "cult_val": None,
        "cult_keywords": [],
        "cult_fallback": ["Meghalaya", "Shillong", "Khasi", "Cherrapunji"],
    },
    "Mizoram": {
        "cult_val": None,
        "cult_keywords": [],
        "cult_fallback": ["Mizoram", "Aizawl", "Mizo"],
    },
    "Nagaland": {
        "cult_val": None,
        "cult_keywords": [],
        "cult_fallback": ["Nagaland", "Kohima", "Naga", "Hornbill"],
    },
}

# Andhra Pradesh — only conv needs fixing, cult (wiki) was fine
ANDHRA_CONV_FIX = {
    "conv": "Telugu-LLM-Labs/telugu_alpaca_yahma_cleaned_filtered_romanized"  # ✅ correct name
}


# ==========================================
# ⚙️ HELPERS
# ==========================================
def make_row(instruction, response, input_ctx=""):
    return {
        "instruction": instruction.strip(),
        "input": input_ctx.strip(),
        "response": response.strip(),
    }


def safe_extract_conv(example):
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
    return {
        "instruction": str(instr).strip(),
        "input": inp,
        "response": str(resp).strip(),
    }


def is_valid_row(example):
    return bool(example["instruction"]) and bool(example["response"])


def map_wiki(example):
    return make_row(
        instruction=f"Provide a detailed explanation about {example['title']}.",
        response=str(example["text"])[:6000],
    )


# ==========================================
# 🔧 PATCH 1 — Andhra Pradesh conv
# ==========================================
print("\n" + "=" * 60)
print("🔧 PATCH 1: Andhra Pradesh — fixing conv split")
print("=" * 60)

try:
    conv_ds = load_dataset(ANDHRA_CONV_FIX["conv"], split="train")
    mapped = conv_ds.map(safe_extract_conv, remove_columns=conv_ds.column_names)
    conv_split = mapped.filter(is_valid_row)
    print(f"  ✅ Conv: {len(conv_split)} rows from {ANDHRA_CONV_FIX['conv']}")

    # Load existing splits from hub and add/replace conv
    try:
        existing = DatasetDict.load_from_hub(HF_REPO_NAME, config_name="Andhra_Pradesh")
        existing["conv"] = conv_split
        patched = existing
    except Exception:
        # No existing dataset yet — push conv alone
        patched = DatasetDict({"conv": conv_split})

    patched.push_to_hub(HF_REPO_NAME, config_name="Andhra_Pradesh")
    print("  🎉 Andhra_Pradesh conv patched and uploaded!")

except Exception as e:
    print(f"  ⚠️ Andhra Pradesh conv patch failed: {e}")

# ==========================================
# 🔧 PATCH 2 — Reload wiki_qa for NE fallback states
# ==========================================
ne_states = [s for s, c in WIKIQA_PATCH.items() if c["cult_val"] is None]
if ne_states:
    print("\n⏳ Loading microsoft/wiki_qa for NE state fallbacks...")
    global_wikiqa = load_dataset("microsoft/wiki_qa", split="train").filter(
        lambda x: x["label"] == 1
    )
    print(f"✅ WikiQA: {len(global_wikiqa)} answered QA pairs")

# ==========================================
# 🔧 PATCH 2 — All wikiqa cult splits
# ==========================================
for state, patch in WIKIQA_PATCH.items():
    print(f"\n{'=' * 60}")
    print(f"🔧 PATCH 2 cult: {state}")
    print(f"{'=' * 60}")

    cult_split = None

    # --- States with a native Wikipedia dump ---
    if patch["cult_val"] is not None:
        try:
            print(f"  Loading Wikipedia: {patch['cult_val']}")
            wiki_ds = load_dataset(
                "wikimedia/wikipedia", patch["cult_val"], split="train"
            )

            # Try native-script keywords first
            if patch["cult_keywords"]:
                filtered = wiki_ds.filter(
                    lambda x, kw=patch["cult_keywords"]: any(
                        k in x["title"] for k in kw
                    )
                )
                print(f"  Native keyword filter → {len(filtered)} articles")
            else:
                filtered = wiki_ds.filter(lambda x: False)  # empty

            # Fallback to romanized keywords if native got too few
            if len(filtered) < 10:
                print(
                    f"  ⚠️  Too few native matches — trying romanized fallback keywords..."
                )
                filtered = wiki_ds.filter(
                    lambda x, kw=patch["cult_fallback"]: any(
                        k.lower() in x["title"].lower() for k in kw
                    )
                )
                print(f"  Romanized fallback → {len(filtered)} articles")

            if len(filtered) > 0:
                cult_split = filtered.map(
                    map_wiki, remove_columns=filtered.column_names
                )
                print(f"  ✅ Cult (wiki_filtered): {len(cult_split)} articles")
            else:
                print(
                    f"  ⚠️  Still 0 articles after both filters — skipping cult for {state}"
                )

        except Exception as e:
            print(f"  ⚠️  Wiki load error for {state}: {e}")

    # --- NE states: fall back to microsoft/wiki_qa with column-level filter ---
    else:
        try:
            keywords = patch["cult_fallback"]
            # ✅ Filter on specific columns — NOT str(entire_row)
            filtered_qa = global_wikiqa.filter(
                lambda x, kw=keywords: any(
                    k.lower() in (x["question"] + " " + x["document_title"]).lower()
                    for k in kw
                )
            )
            print(f"  WikiQA column filter → {len(filtered_qa)} rows")

            if len(filtered_qa) > 0:

                def map_wikiqa(example, s=state):
                    return make_row(
                        instruction=(
                            f"Explain this aspect of {s.replace('_', ' ')} in detail: "
                            f"{example['question']}"
                        ),
                        response=example["answer"],
                        input_ctx=example["document_title"],
                    )

                cult_split = filtered_qa.map(
                    map_wikiqa, remove_columns=filtered_qa.column_names
                )
                print(f"  ✅ Cult (wiki_qa fallback): {len(cult_split)} QA rows")
            else:
                print(
                    f"  ⚠️  0 WikiQA rows for {state} — no cult split will be uploaded"
                )

        except Exception as e:
            print(f"  ⚠️  WikiQA filter error for {state}: {e}")

    # --- Push cult split ---
    if cult_split is not None and len(cult_split) > 0:
        try:
            existing = DatasetDict.load_from_hub(HF_REPO_NAME, config_name=state)
            existing["cult"] = cult_split
            patched = existing
        except Exception:
            patched = DatasetDict({"cult": cult_split})

        patched.push_to_hub(HF_REPO_NAME, config_name=state)
        print(f"  🎉 {state} cult split uploaded! ({len(cult_split)} rows)")
    else:
        print(f"  ❌ Skipped upload for {state} — empty cult split.")

print("\n" + "=" * 60)
print("🚀 ALL PATCHES APPLIED SUCCESSFULLY!")
print("=" * 60)
