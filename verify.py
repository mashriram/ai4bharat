import argparse
from datasets import load_dataset_builder, get_dataset_split_names
from huggingface_hub import list_repo_files, HfApi
import sys

# ==========================================
# 🛑 CONFIGURATION
# ==========================================
# Replace with your actual repo name
DEFAULT_REPO = "mashriram/AI4Bharat-Languages-and-Cultures" 

# The list of 30 states we expect to see
EXPECTED_STATES = [
    "Tamil_Nadu", "Kerala", "Karnataka", "Andhra_Pradesh", "Telangana",
    "Maharashtra", "Gujarat", "Goa", "Rajasthan",
    "Punjab", "Haryana", "Himachal_Pradesh", "Uttarakhand", "Uttar_Pradesh",
    "Madhya_Pradesh", "Chhattisgarh", "Bihar", "Jharkhand", "Odisha", "West_Bengal",
    "Assam", "Arunachal_Pradesh", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Tripura", "Sikkim",
    "Jammu_and_Kashmir", "Delhi"
]

def verify_repo(repo_id):
    print(f"\n🔍 INSPECTING REPO: {repo_id}")
    print("=" * 60)
    
    api = HfApi()
    
    # 1. Check if Repo Exists
    try:
        repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"✅ Repo exists! (Private: {repo_info.private}, Downloads: {repo_info.downloads})")
    except Exception as e:
        print(f"❌ CRITICAL: Repo {repo_id} not found or not accessible.")
        print(f"   Error: {e}")
        sys.exit(1)

    # 2. Check for Configs (States)
    print(f"\n📋 Checking State Configurations...")
    try:
        # HACK: HF Datasets stores configs in a specific way. 
        # We try to load the builder to see available configs.
        ds_builder = load_dataset_builder(repo_id)
        available_configs = ds_builder.config.name if ds_builder.config.name != "default" else "Unknown"
        
        # If the builder method doesn't list all, we check files directly for named folders/parquets
        files = list_repo_files(repo_id, repo_type="dataset")
        found_states = []
        for state in EXPECTED_STATES:
            # Look for file patterns like "Tamil_Nadu/train..." or "Tamil_Nadu/indic..."
            if any(state in f for f in files):
                found_states.append(state)
        
        missing = set(EXPECTED_STATES) - set(found_states)
        
        if len(missing) == 0:
            print(f"✅ ALL 30 STATES FOUND.")
        else:
            print(f"⚠️  WARNING: Missing {len(missing)} states:")
            print(f"   {', '.join(missing)}")
            
    except Exception as e:
        print(f"⚠️  Could not list configs automatically: {e}")

    # 3. Deep Dive: Stream Check Each State
    print(f"\n🔬 Deep Stream Verification (Checking Splits)...")
    print(f"{'STATE':<20} | {'INDIC':<8} | {'CONV':<8} | {'CULT':<8} | {'STATUS'}")
    print("-" * 65)

    error_count = 0

    for state in EXPECTED_STATES:
        status_msg = "✅ OK"
        split_status = {"indic": "❌", "conv": "❌", "cult": "❌"}
        
        try:
            # Get available splits for this config
            # We use 'get_dataset_split_names' which is fast and doesn't download data
            available_splits = get_dataset_split_names(repo_id, config_name=state)
            
            # Since you pushed splits as separate datasets in the build script, 
            # they often appear as 'train' if loaded individually, OR
            # if you pushed them as a single dataset with multiple columns/splits.
            # However, your script pushed them as a DatasetDict.
            # So looking for 'indic', 'conv', 'cult' keys is correct.
            print(available_configs)
            # To verify strictly without downloading, we try to peek at the first row of each expected split
            from datasets import load_dataset
            
            # We assume your script used push_to_hub on a DatasetDict, 
            # so the splits should be named 'indic', 'conv', 'cult'.
            
            # Streaming load (Instant)
            ds = load_dataset(repo_id, name=state, streaming=True)
            
            for split in ["indic", "conv", "cult"]:
                if split in ds:
                    try:
                        # Try to take 1 row to ensure it's not empty
                        next(iter(ds[split]))
                        split_status[split] = "✅"
                    except:
                        split_status[split] = "⚠️ Empty"
                else:
                     split_status[split] = "MISSING"

            # Logic check
            if "MISSING" in split_status.values():
                status_msg = "⚠️ Partial"
                error_count += 1
            
        except Exception as e:
            status_msg = "❌ FAIL"
            error_count += 1
            # print(e) # Uncomment for debug

        print(f"{state:<20} | {split_status['indic']:<8} | {split_status['conv']:<8} | {split_status['cult']:<8} | {status_msg}")

    print("=" * 60)
    if error_count == 0:
        print(f"🎉 SUCCESS! The dataset {repo_id} is 100% ready for the summit.")
    else:
        print(f"⚠️  WARNING: {error_count} states have issues. Check the table above.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO, help="HF Repo ID to verify")
    args = parser.parse_args()
    
    verify_repo(args.repo)