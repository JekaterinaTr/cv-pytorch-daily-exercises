"""
PHASE 4 — PyTorch Fundamentals
Day 2: Tensor Indexing & Slicing

Concepts:
- 1D indexing
- 2D indexing
- slicing
- modifying values
- higher-dimensional tensors
"""

import torch

print("=== PHASE 4 - DAY 2: INDEXING & SLICING ===\n")

# --------------------------------------------------
# 1. 1D Tensor Indexing
# --------------------------------------------------

print("1. 1D Tensor")

t = torch.tensor([10, 20, 30, 40, 50])

print("Tensor:", t)
print("First element:", t[0])
print("Last element:", t[-1])
print()

# --------------------------------------------------
# 2. 1D Tensor Slicing
# --------------------------------------------------

print("2. Slicing")

print("First three:", t[0:3])
print("From index 2:", t[2:])
print("Every second:", t[::2])
print()

# --------------------------------------------------
# 3. 2D Tensor Indexing
# --------------------------------------------------

print("3. 2D Tensor")

matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(matrix)

# Row access
print("First row:", matrix[0])
print("Second row:", matrix[1])

# Single element
print("Element (row1,col2):", matrix[1, 2])
print()

# --------------------------------------------------
# 4. Column Selection
# --------------------------------------------------

print("4. Column Selection")

print("First column:", matrix[:, 0])
print("Third column:", matrix[:, 2])
print()

# --------------------------------------------------
# 5. Modify Tensor Values
# --------------------------------------------------

print("5. Modify Values")

t2 = torch.tensor([1, 2, 3, 4])
print("Before:", t2)

t2[0] = 99
print("After:", t2)

# Modify slice
t2[1:3] = torch.tensor([55, 66])
print("After slice modify:", t2)
print()

# --------------------------------------------------
# 6. 3D Tensor Example (Image-like)
# --------------------------------------------------

print("6. 3D Tensor Example")

# (batch, channels, height, width)
images = torch.rand((4, 3, 64, 64))

print("Shape:", images.shape)

# First image
first_image = images[0]
print("First image shape:", first_image.shape)

# Red channel of first image
red_channel = images[0, 0]
print("Red channel shape:", red_channel.shape)
print()

print("Day 2 Completed Successfully ✅")