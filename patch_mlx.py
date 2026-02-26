#!/usr/bin/env python3
"""
Run once: uv run patch_mlx.py
Patches mlx_lm/tuner/utils.py to fix all missing Qwen3 config attributes.
"""
import sys, subprocess

# Find the exact utils.py this venv uses
result = subprocess.run(
    [sys.executable, "-c",
     "import mlx_lm.tuner.utils as m; print(m.__file__)"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print(f"❌  Cannot import mlx_lm: {result.stderr}")
    sys.exit(1)

utils_path = result.stdout.strip()
print(f"📄  {utils_path}")

src = open(utils_path).read()
original = src

# ── Patch 1: config.num_layers ────────────────────────────────────────
src = src.replace(
    "config.num_layers,",
    "getattr(config, 'num_layers', getattr(config, 'num_hidden_layers', 28)),  # P1",
)

# ── Patch 2: config.lora_parameters ──────────────────────────────────
# lora_parameters is built from the adapter_config fields we know
src = src.replace(
    "config.lora_parameters,",
    "getattr(config, 'lora_parameters', {'rank': getattr(config, 'r', 64), 'alpha': getattr(config, 'lora_alpha', 128), 'dropout': getattr(config, 'lora_dropout', 0.0), 'scale': getattr(config, 'lora_alpha', 128) / getattr(config, 'r', 64)}),  # P2",
)

if src == original:
    if "P1" in src and "P2" in src:
        print("✅  Already fully patched — nothing to do")
    else:
        print("⚠️   Could not find patch targets. Lines with 'config.' in load_adapters:")
        in_fn = False
        for i, line in enumerate(src.splitlines(), 1):
            if "def load_adapters" in line or "def _load_adapters" in line:
                in_fn = True
            if in_fn and "config." in line:
                print(f"  line {i}: {line.rstrip()}")
            if in_fn and i > 160:
                break
    sys.exit(0)

open(utils_path, "w").write(src)

p1 = "P1" in src
p2 = "P2" in src
print(f"  {'✅' if p1 else '❌'}  num_layers patch")
print(f"  {'✅' if p2 else '❌'}  lora_parameters patch")
print()
print("✅  Done. Now run: uv run upload.py")
