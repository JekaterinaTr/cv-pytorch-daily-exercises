"""
Computer Vision Daily Practice (OpenCV + PyTorch)
PHASE 1 — Python, NumPy & Image Foundations

Day 6: Image Stacking & Batching

This script demonstrates:
- Horizontal and vertical image stacking
- Shape constraints for stacking
- Padding images to enable stacking
- Combining processed images for visualization
- Creating image batches for ML pipelines

Key Concepts:
1. Images must match dimensions to be stacked.
2. hstack → same height, vstack → same width.
3. Padding is required for mismatched sizes.
4. np.stack creates batch dimensions for ML models.
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Create sample grayscale images
# -----------------------------

"""
Theory:
- Images are NumPy arrays of shape (H, W).
- uint8 is standard for grayscale images (0–255).
"""

img1 = np.random.randint(80, 180, (64, 64), dtype=np.uint8)
img2 = np.random.randint(50, 200, (64, 64), dtype=np.uint8)

print("Image 1 shape:", img1.shape)
print("Image 2 shape:", img2.shape)
print("Image dtype:", img1.dtype)
print("-" * 40)

# -----------------------------
# 2. Horizontal stacking
# -----------------------------

"""
Theory:
- np.hstack combines arrays along columns.
- All images must have the same height.
"""

h_stack = np.hstack([img1, img2])
print("Horizontal stack shape:", h_stack.shape)

# -----------------------------
# 3. Vertical stacking
# -----------------------------

"""
Theory:
- np.vstack combines arrays along rows.
- All images must have the same width.
"""

v_stack = np.vstack([img1, img2])
print("Vertical stack shape:", v_stack.shape)
print("-" * 40)

# -----------------------------
# 4. Stack processed versions of an image
# -----------------------------

"""
Theory:
- Image stacking is often used to compare preprocessing effects.
- Here we show darker, original, and brighter images side by side.
"""

brighter = np.clip(img1 + 40, 0, 255)
darker = np.clip(img1 - 40, 0, 255)

processed_stack = np.hstack([darker, img1, brighter])

# -----------------------------
# 5. Padding for mismatched sizes
# -----------------------------

"""
Theory:
- Stacking fails if dimensions do not match.
- Padding adds extra pixels to match dimensions.
"""

img_small = np.random.randint(0, 256, (32, 64), dtype=np.uint8)
pad_height = img1.shape[0] - img_small.shape[0]

img_small_padded = np.pad(
    img_small,
    ((0, pad_height), (0, 0)),
    mode="constant",
    constant_values=0
)

v_stack_padded = np.vstack([img1, img_small_padded])

print("Small image shape:", img_small.shape)
print("Padded image shape:", img_small_padded.shape)
print("Vertical stack with padding shape:", v_stack_padded.shape)
print("-" * 40)

# -----------------------------
# 6. Creating a batch of images
# -----------------------------

"""
Theory:
- np.stack adds a new dimension.
- Common batch shape: (N, H, W)
- Used directly in ML pipelines.
"""

batch = np.stack([img1, img2, img_small_padded])
print("Batch shape (N, H, W):", batch.shape)

# -----------------------------
# 7. Visualization
# -----------------------------

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.title("Horizontal Stack")
plt.imshow(h_stack, cmap="gray", vmin=0, vmax=255)
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Vertical Stack")
plt.imshow(v_stack, cmap="gray", vmin=0, vmax=255)
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Darker | Original | Brighter")
plt.imshow(processed_stack, cmap="gray", vmin=0, vmax=255)
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Vertical Stack with Padding")
plt.imshow(v_stack_padded, cmap="gray", vmin=0, vmax=255)
plt.axis("off")

plt.tight_layout()
plt.show()

"""
Summary:
- Image stacking is essential for visualization and batching.
- hstack → same height, vstack → same width.
- Padding solves dimension mismatches.
- np.stack creates batch dimensions for ML pipelines.
- These operations are used heavily in real-world CV preprocessing.
"""
