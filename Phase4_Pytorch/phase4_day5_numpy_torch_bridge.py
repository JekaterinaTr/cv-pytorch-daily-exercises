"""
PHASE 4 — PyTorch Fundamentals
Day 5: NumPy ↔ PyTorch Conversion

Concepts:
- torch.from_numpy()
- tensor.numpy()
- shared memory
- clone() and copy()
- image-style pipelines
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

print("PHASE 4 - DAY 5")
print("NumPy ↔ PyTorch Conversion")
print("-" * 50)


# --------------------------------------------------
# 1. NumPy → PyTorch Tensor
# --------------------------------------------------

print("\nExercise 1 — NumPy to Tensor")

np_array = np.array([1, 2, 3, 4])

tensor = torch.from_numpy(np_array)

print("NumPy array:", np_array)
print("Tensor:", tensor)

print("NumPy type:", type(np_array))
print("Tensor type:", type(tensor))


# --------------------------------------------------
# 2. Shared Memory Demonstration
# --------------------------------------------------

print("\nExercise 2 — Shared Memory")

np_array[0] = 99

print("Modified NumPy:", np_array)
print("Tensor reflects change:", tensor)


# --------------------------------------------------
# 3. PyTorch → NumPy
# --------------------------------------------------

print("\nExercise 3 — Tensor to NumPy")

tensor2 = torch.tensor([5, 6, 7, 8])

np_array2 = tensor2.numpy()

print("Tensor:", tensor2)
print("Converted NumPy:", np_array2)


# --------------------------------------------------
# 4. Safe Copy (Independent Memory)
# --------------------------------------------------

print("\nExercise 4 — Safe Copy")

np_copy = np_array2.copy()
tensor_copy = tensor2.clone()

np_array2[0] = 100

print("Original NumPy:", np_array2)
print("Copied NumPy:", np_copy)

tensor2[1] = 200

print("Original Tensor:", tensor2)
print("Cloned Tensor:", tensor_copy)


# --------------------------------------------------
# 5. Image Example (NumPy → Tensor)
# --------------------------------------------------

print("\nExercise 5 — Image Conversion")

# Fake image (height, width, channels)
img_np = np.random.rand(64, 64, 3)

img_tensor = torch.from_numpy(img_np)

print("NumPy image shape:", img_np.shape)
print("Tensor image shape:", img_tensor.shape)


# --------------------------------------------------
# 6. Tensor Processing Example
# --------------------------------------------------

print("\nExercise 6 — Processing Tensor")

processed_tensor = img_tensor * 2

print("Tensor multiplied by 2")


# --------------------------------------------------
# 7. Convert Back to NumPy
# --------------------------------------------------

print("\nExercise 7 — Tensor Back to NumPy")

img_back = processed_tensor.numpy()

print("Converted back shape:", img_back.shape)


# --------------------------------------------------
# 8. Visualize Image
# --------------------------------------------------

print("\nExercise 8 — Visualization")

plt.imshow(img_back)
plt.title("Processed Image")
plt.axis("off")
plt.show()


print("\nDay 5 Completed Successfully.")