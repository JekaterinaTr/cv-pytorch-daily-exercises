"""
PHASE 4 — PyTorch Fundamentals
Day 1: Tensor Creation & Basics

Concepts Covered:
- torch.tensor
- torch.zeros
- torch.ones
- torch.rand
- torch.randn
- shape inspection
- datatype control
"""

import torch

print("=== PHASE 4 - DAY 1: TENSOR BASICS ===\n")

# --------------------------------------------------
# 1. Create Tensor from Python List
# --------------------------------------------------

print("1. Tensor from list")

t1 = torch.tensor([1, 2, 3, 4])

print("Tensor:", t1)
print("Shape:", t1.shape)
print("Datatype:", t1.dtype)
print()

# --------------------------------------------------
# 2. Create 2D Tensor (Matrix)
# --------------------------------------------------

print("2. 2D Tensor (Matrix)")

matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print(matrix)
print("Shape:", matrix.shape)
print()

# --------------------------------------------------
# 3. Create Zeros Tensor
# --------------------------------------------------

print("3. Zeros Tensor")

zeros = torch.zeros((3, 4))

print(zeros)
print("Shape:", zeros.shape)
print()

# --------------------------------------------------
# 4. Create Ones Tensor
# --------------------------------------------------

print("4. Ones Tensor")

ones = torch.ones((2, 5))

print(ones)
print("Shape:", ones.shape)
print()

# --------------------------------------------------
# 5. Random Uniform Tensor (0 to 1)
# --------------------------------------------------

print("5. Random Uniform Tensor")

rand_tensor = torch.rand((3, 3))

print(rand_tensor)
print("Shape:", rand_tensor.shape)
print()

# --------------------------------------------------
# 6. Random Normal Tensor (mean=0, std=1)
# --------------------------------------------------

print("6. Random Normal Tensor")

randn_tensor = torch.randn((3, 3))

print(randn_tensor)
print("Shape:", randn_tensor.shape)
print()

# --------------------------------------------------
# 7. Specify Datatype
# --------------------------------------------------

print("7. Tensor with Specific Datatype")

float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)

print(float_tensor)
print("Datatype:", float_tensor.dtype)
print()

# --------------------------------------------------
# 8. Multi-Dimensional Tensors
# --------------------------------------------------

print("8. Different Dimensions")

a = torch.rand((5,))
b = torch.rand((5, 3))
c = torch.rand((2, 3, 4))

print("a shape:", a.shape)  # 1D
print("b shape:", b.shape)  # 2D
print("c shape:", c.shape)  # 3D
print()

print("Day 1 Completed Successfully ✅")