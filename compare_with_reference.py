
import numpy as np
import sys
import os
sys.path.append('references/RWKV-LM/RWKV-v7')

# Set up RWKV reference
os.environ["RWKV_V7_ON"] = "1"
from rwkv.utils import PIPELINE
from rwkv.model import RWKV as referenceRWKV
import time

# Load our implementation results
import struct
import json
import torch

def load_rust_logits():
    with open('rust_final_logits.bin', 'rb') as f:
        data = f.read()
    num_floats = len(data) // 4
    return np.array(struct.unpack(f'<{num_floats}f', data))

def load_rust_probs():
    with open('rust_final_probs.bin', 'rb') as f:
        data = f.read()
    num_floats = len(data) // 4
    return np.array(struct.unpack(f'<{num_floats}f', data))

def load_rust_metadata():
    with open('rust_final_logits.json', 'r') as f:
        return json.load(f)

print("Loading reference RWKV implementation...")
MODEL_FILE = 'modeldir/rwkv7-g1-0.1b-20250307-ctx4096.pth'
model = referenceRWKV(model=MODEL_FILE[:-4], strategy='cpu fp32')
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

# Get the exact same context and tokens
metadata = load_rust_metadata()
context = metadata['context']
rust_tokens = metadata['tokens']

print(f"Context: {context}")
print(f"Tokens: {rust_tokens} (length: {len(rust_tokens)})")

# Encode with reference tokenizer
ref_tokens = pipeline.encode(context)
print(f"Reference tokens: {ref_tokens} (length: {len(ref_tokens)})")

if rust_tokens != ref_tokens:
    print("WARNING: Token mismatch between Rust and Python!")
    print(f"Rust: {rust_tokens}")
    print(f"Python: {ref_tokens}")

# Forward pass with reference
print("Running reference forward pass...")
py_start = time.time()
reference_logits, state = model.forward(ref_tokens, None)
py_duration = time.time() - py_start
py_tokens_per_sec = len(ref_tokens) / py_duration if py_duration > 0 else float('inf')
reference_logits = reference_logits.numpy()

# Load our results
rust_logits = load_rust_logits()
rust_probs = load_rust_probs()
metadata = load_rust_metadata()
rust_tps = metadata.get('rust_tokens_per_sec', 0.0)

print(f"\nComparison Results:")
print(f"Reference logits shape: {reference_logits.shape}")
print(f"Rust logits shape: {rust_logits.shape}")
print(f"Rust probs shape: {rust_probs.shape}")

if reference_logits.shape != rust_logits.shape:
    print("ERROR: Shape mismatch!")
    sys.exit(1)

# Calculate deviation exactly like rwkv_v7_numpy.py
deviation = np.max(np.abs(rust_logits - reference_logits)) / reference_logits.std()
print(f"Deviation from official RWKV: {deviation:.6e}")

# Additional statistics
correlation = np.corrcoef(rust_logits, reference_logits)[0, 1]
mse = np.mean((rust_logits - reference_logits) ** 2)
mae = np.mean(np.abs(rust_logits - reference_logits))

print(f"Correlation coefficient: {correlation:.6f}")
print(f"Mean Squared Error: {mse:.6e}")
print(f"Mean Absolute Error: {mae:.6e}")

# Probability comparison (using stable softmax in torch for reference)
ref_probs = torch.softmax(torch.tensor(reference_logits, dtype=torch.float32), dim=0).numpy()

print("\nProbability checks:")
print(f"Rust probs sum: {rust_probs.sum():.9f}  min: {rust_probs.min():.9f}  max: {rust_probs.max():.9f}")
print(f"Ref  probs sum: {ref_probs.sum():.9f}  min: {ref_probs.min():.9f}  max: {ref_probs.max():.9f}")

# Compare probabilities
prob_linf = np.max(np.abs(rust_probs - ref_probs))
prob_l1 = np.sum(np.abs(rust_probs - ref_probs))
prob_l2 = np.sqrt(np.sum((rust_probs - ref_probs) ** 2))

# KL divergences (add tiny epsilon for stability)
eps = 1e-12
rp = np.clip(rust_probs, eps, 1.0)
pp = np.clip(ref_probs, eps, 1.0)
kl_rp_pp = np.sum(rp * (np.log(rp) - np.log(pp)))
kl_pp_rp = np.sum(pp * (np.log(pp) - np.log(rp)))

print("\nProbability deltas:")
print(f"L_inf: {prob_linf:.9e}")
print(f"L1:    {prob_l1:.9e}")
print(f"L2:    {prob_l2:.9e}")
print(f"KL(rust||ref): {kl_rp_pp:.9e}")
print(f"KL(ref||rust): {kl_pp_rp:.9e}")

# Top predictions comparison
rust_top_5 = np.argsort(rust_logits)[-5:][::-1]
ref_top_5 = np.argsort(reference_logits)[-5:][::-1]

print(f"\nTop 5 predictions:")
print(f"Rust:      {rust_top_5}")
print(f"Reference: {ref_top_5}")
print(f"Top-1 match: {rust_top_5[0] == ref_top_5[0]}")
print(f"Top-5 overlap: {len(set(rust_top_5) & set(ref_top_5))}/5")

# Success criteria (based on rwkv_v7_numpy.py which shows 6.995911e-06)
# Allow tiny probability sum drift due to FP32 accumulation across 65k dims
if deviation < 1e-5 and abs(rust_probs.sum() - 1.0) < 5e-4 and abs(ref_probs.sum() - 1.0) < 5e-4 and rust_probs.min() >= 0.0:
    print(f"\n✅ SUCCESS: Deviation {deviation:.6e} < 1e-5 (Reference: 6.995911e-06)")
    print("Rust implementation is mathematically equivalent to official RWKV!")
else:
    print("\n❌ FAILED verification")
    print(f"Deviation: {deviation:.6e}")
    print(f"Rust probs sum delta: {abs(rust_probs.sum()-1.0):.6e}")
    print(f"Ref  probs sum delta: {abs(ref_probs.sum()-1.0):.6e}")
    print(f"Rust min prob: {rust_probs.min():.6e}")

print("\nPerformance:")
print(f"Python: {py_tokens_per_sec:.2f} tok/s  (T={len(ref_tokens)})")
print(f"Rust:   {rust_tps:.2f} tok/s  (T={len(ref_tokens)})")
if rust_tps > 0 and py_tokens_per_sec > 0:
    speedup = rust_tps / py_tokens_per_sec
    print(f"Speedup: {speedup:.2f}x")
