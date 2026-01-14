"""
Computer Vision Daily Practice (OpenCV + PyTorch)
PHASE 1 - Python, NumPy & Image Foundations

Day 5: Normalization & Flattening (Preparing Images for ML)

This script demonstrates:
- Why raw uint8 images are unsuitable for ML models.
- How to normalize images to [0,1] and [-1,1].
- How to flatten images into feature vectors.
- The difference between ravel() and flatten().
- Common preprocessing mistakes.

Key Concepts:
1. ML models expect floating-point inputs.
2. Normalization stabilizes training and improves convergence.
3. Images are flattened before feeding into classical ML models.
4. ravel() is memory-efficient; flatten() creates a copy.
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Create a grayscale image
# -----------------------------

"""
Theory:
- Grayscale images are 2D arrays with values in [0,255].
- dtype uint8 is common for raw images but not ideal for ML.
"""

img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

print("Original Image Info:")
print("Shape:", img.shape)
print("Dtype:", img.dtype)
print("Min:", img.min())
print("Max:", img.max())
print("-" * 40)

# -----------------------------
# 2. Normalize to [0, 1]
# -----------------------------

"""
Theory:
- Convert to float32 BEFORE division.
- Normalization is standard for neural networks.
"""

img_norm = img.astype(np.float32) / 255.0

print("Normalized Image Info [0,1]:")
print("Min:", img_norm.min())
print("Max:", img_norm.max())
print("Dtype:", img_norm.dtype)
print("-" * 40)

# -----------------------------
# 3. Visualize before vs after
# -----------------------------

plt.figure(figsize=(6, 3))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.title("Original uint8")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_norm, cmap="gray", vmin=0, vmax=1)
plt.title("Normalized [0,1]")
plt.axis("off")

plt.tight_layout()
plt.show()

# -----------------------------
# 4. Flattening the image
# -----------------------------

"""
Theory:
- ML models operate on vectors, not 2D grids.
- ravel() returns a view when possible.
- flatten() always returns a copy.
"""

flat_ravel = img_norm.ravel()
flat_flatten = img_norm.flatten()

print("Flattening:")
print("Original shape:", img_norm.shape)
print("Ravel shape:", flat_ravel.shape)
print("Flatten shape:", flat_flatten.shape)
print("-" * 40)

# -----------------------------
# 5. Inspect flattened data
# -----------------------------

print("First 10 flattened pixel values:")
print(flat_ravel[:10])
print("-" * 40)

# -----------------------------
# 6. Normalize to [-1, 1]
# -----------------------------

"""
Theory:
- Some DL architectures expect inputs in [-1,1].
"""

img_norm_neg1 = img_norm * 2.0 - 1.0

print("Normalized Image Info [-1,1]:")
print("Min:", img_norm_neg1.min())
print("Max:", img_norm_neg1.max())
print("-" * 40)

# -----------------------------
# 7. Common normalization bug
# -----------------------------

"""
Theory:
- Dividing uint8 by 255 without casting can lead to subtle bugs.
"""

bad_norm = img / 255
print("Bad normalization dtype:", bad_norm.dtype)

good_norm = img.astype(np.float32) / 255.0
print("Good normalization dtype:", good_norm.dtype)
print("-" * 40)

# -----------------------------
# 8. Preprocessing function
# -----------------------------

def preprocess_image(img):
    """
    Returns:
    - normalized image in [0,1]
    - flattened vector
    """
    img_norm = img.astype(np.float32) / 255.0
    img_flat = img_norm.ravel()
    return img_norm, img_flat

norm_img, flat_img = preprocess_image(img)

print("Preprocess function output:")
print("Normalized shape:", norm_img.shape)
print("Flattened shape:", flat_img.shape)
print("-" * 40)

"""
Summary:
- Raw images must be normalized before ML processing.
- Flattening converts images into feature vectors.
- ravel() is preferred for performance when possible.
- Proper preprocessing prevents silent training failures.
"""
