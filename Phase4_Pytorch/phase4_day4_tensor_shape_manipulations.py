"""
PHASE 4 - DAY 4
Tensor Reshaping and Shape Manipulation
"""

import torch

print("PHASE 4 - DAY 4")
print("Tensor Reshape and Shape Manipulation")
print("-" * 50)


# --------------------------------------------------
# Exercise 1 — Inspect Tensor Shape
# --------------------------------------------------

print("\nExercise 1 — Tensor Shape")

t = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print("Tensor:")
print(t)

print("Shape:", t.shape)


# --------------------------------------------------
# Exercise 2 — Reshape Tensor
# --------------------------------------------------

print("\nExercise 2 — Reshape Tensor")

t2 = torch.tensor([1, 2, 3, 4, 5, 6])

reshaped = t2.reshape(2, 3)

print("Original:", t2)
print("Reshaped:")
print(reshaped)

print("New shape:", reshaped.shape)


# --------------------------------------------------
# Exercise 3 — View Tensor
# --------------------------------------------------

print("\nExercise 3 — View Tensor")

viewed = t2.view(3, 2)

print("Viewed tensor:")
print(viewed)

print("Shape:", viewed.shape)


# --------------------------------------------------
# Exercise 4 — Flatten Tensor
# --------------------------------------------------

print("\nExercise 4 — Flatten Tensor")

matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])

flat = matrix.flatten()

print("Matrix:")
print(matrix)

print("Flattened:")
print(flat)

print("Shape:", flat.shape)


# --------------------------------------------------
# Exercise 5 — Tensor Dimension Reordering
# --------------------------------------------------

print("\nExercise 5 — Permute Dimensions")

# Fake batch of images
images = torch.rand((4, 64, 64, 3))  # batch, height, width, channels

print("Original shape:", images.shape)

# Change to PyTorch CNN format
images_permuted = images.permute(0, 3, 1, 2)

print("After permute:", images_permuted.shape)


# --------------------------------------------------
# Exercise 6 — Revert Permute
# --------------------------------------------------

print("\nExercise 6 — Reverse Permute")

images_back = images_permuted.permute(0, 2, 3, 1)

print("Restored shape:", images_back.shape)


# --------------------------------------------------
# Exercise 7 — Add Batch Dimension
# --------------------------------------------------

print("\nExercise 7 — Unsqueeze (Add Dimension)")

image = torch.rand((64, 64, 3))

print("Original shape:", image.shape)

image_batch = image.unsqueeze(0)

print("With batch dimension:", image_batch.shape)


# --------------------------------------------------
# Exercise 8 — Remove Dimension
# --------------------------------------------------

print("\nExercise 8 — Squeeze (Remove Dimension)")

no_batch = image_batch.squeeze(0)

print("After squeeze:", no_batch.shape)


print("\nAll exercises completed.")