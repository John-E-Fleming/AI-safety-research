"""
Test script for verifying the mech_interp environment setup
Based on Week 1, Day 1-2 requirements from mats_project_guide.md
"""

print("Testing environment setup...")
print("=" * 50)

# Test 1: Import transformer_lens
print("\n1. Testing transformer_lens import...")
try:
    import transformer_lens as tl
    print(f"   [OK] transformer_lens imported successfully")
except Exception as e:
    print(f"   [FAIL] Failed: {e}")

# Test 2: Import torch and check CUDA
print("\n2. Testing PyTorch and CUDA...")
try:
    import torch
    print(f"   [OK] PyTorch version: {torch.__version__}")
    print(f"   [OK] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   [OK] CUDA version: {torch.version.cuda}")
        print(f"   [OK] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("   [INFO] No GPU detected (CPU mode)")
except Exception as e:
    print(f"   [FAIL] Failed: {e}")

# Test 3: Import other core packages
print("\n3. Testing other core packages...")
packages = [
    ('numpy', 'np'),
    ('pandas', 'pd'),
    ('matplotlib.pyplot', 'plt'),
    ('sklearn', 'sklearn'),
]

for package, alias in packages:
    try:
        exec(f"import {package} as {alias}")
        version = eval(f"{alias}.__version__")
        print(f"   [OK] {package}: {version}")
    except Exception as e:
        print(f"   [FAIL] {package} failed: {e}")

print("\n" + "=" * 50)
print("Environment test complete!")
print("\nNext steps (from mats_project_guide.md Day 1-2):")
print("  - First Model Interaction (2-3 hours)")
print("  - ARENA Tutorial sections 1.2.1-1.2.3 (4-5 hours)")
