#!/usr/bin/env python3
import os
import torch

os.makedirs("vectors", exist_ok=True)
vec_path = "vectors/tech_vector.pt"

# Create a normalized random vector (size inferred from model later; 2048 is a safe default)
v = torch.randn(2048, dtype=torch.float32)
v = v / (v.norm() + 1e-12)
torch.save(v, vec_path)
print(f"Wrote placeholder vector to {vec_path}")
