"""
Computer Vision Daily Practice (OpenCV + PyTorch)
PHASE 1 - Python, NumPy & Image Foundations

Day 3: Image Math – Brightness, Contrast, and Statistics

This script demonstrates:
- How to create a fake grayscale image using NumPy.
- How to adjust brightness using addition/subtraction.
- How to change contrast using multiplication.
- How to normalize pixel values to [0, 1].
- How to compute global image statistics.
- How to calculate differences between two images.
- How to visualize original and modified images.

Key Concepts:
1. Images are numerical arrays; arithmetic operations manipulate them.
2. Brightness = addition/subtraction, contrast = scaling.
3. Clipping keeps pixel values in the valid range (0–255).
4. Normalization prepares data for ML models.
5. Absolute difference highlights pixel-level changes.
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Create a fake grayscale image
# -----------------------------

"""
Theory:
- Images are just 2D arrays with pixel values 0–255 for grayscale.
- Using a mid-range (80–180) ensures brightness/darkness changes are visible.
- dtype=np.uint8 ensures standard 8-bit pixels.
"""

img = np.random.randint(80, 180, (64, 64), dtype=np.uint8)

print("Original Image Info:")
print("Shape:", img.shape)
print("Dtype:", img.dtype)
print("Min pixel value:", img.min())
print("Max pixel value:", img.max())
print("-"*40)

# -----------------------------
# 2. Brightness adjustment
# -----------------------------

"""
Theory:
- Brightness can be increased/decreased by adding/subtracting a constant.
- np.clip ensures values stay within [0, 255].
"""

brighter = np.clip(img + 40, 0, 255)
darker = np.clip(img - 40, 0, 255)

# -----------------------------
# 3. Contrast adjustment
# -----------------------------

"""
Theory:
- Multiplying pixel values changes contrast.
- Values may exceed 255, so np.clip is used.
"""

high_contrast = np.clip(img * 1.5, 0, 255)

print("Original Mean:", np.mean(img))
print("High Contrast Mean:", np.mean(high_contrast))
print("-"*40)

# -----------------------------
# 4. Normalization
# -----------------------------

"""
Theory:
- Divide by 255.0 to convert pixel values to [0,1] float range.
- This is standard preprocessing for ML pipelines.
"""

img_norm = img.astype(np.float32) / 255.0

print("Normalized Image Info:")
print("Min:", img_norm.min())
print("Max:", img_norm.max())
print("Dtype:", img_norm.dtype)
print("-"*40)

# -----------------------------
# 5. Global image statistics
# -----------------------------

mean_val = np.mean(img)
sum_val = np.sum(img)
mean_norm = np.mean(img_norm)

print("Image Statistics:")
print("Mean pixel value:", mean_val)
print("Sum of pixels:", sum_val)
print("Mean normalized value:", mean_norm)
print("-"*40)

# -----------------------------
# 6. Image difference
# -----------------------------

"""
Theory:
- Subtracting two images highlights differences (motion/change detection).
- np.abs ensures all values are positive.
"""

img2 = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
diff = np.abs(img.astype(int) - img2.astype(int))

print("Difference Image Mean:", diff.mean())
print("-"*40)

# -----------------------------
# 7. Function to compute image statistics
# -----------------------------

def image_stats(img):
    """
    Returns mean, min, max, sum of an image
    """
    return {
        "mean": np.mean(img),
        "min": np.min(img),
        "max": np.max(img),
        "sum": np.sum(img)
    }

stats = image_stats(img)
print("Image stats function output:", stats)
print("-"*40)

# -----------------------------
# 8. Visualization
# -----------------------------

plt.figure(figsize=(12, 10))

# Original Image
plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.axis("off")

# Brighter Image
plt.subplot(2, 3, 2)
plt.title("Brighter (+40)")
plt.imshow(brighter, cmap="gray", vmin=0, vmax=255)
plt.axis("off")

# Darker Image
plt.subplot(2, 3, 3)
plt.title("Darker (-40)")
plt.imshow(darker, cmap="gray", vmin=0, vmax=255)
plt.axis("off")

# High Contrast
plt.subplot(2, 3, 4)
plt.title("High Contrast (*1.5)")
plt.imshow(high_contrast, cmap="gray", vmin=0, vmax=255)
plt.axis("off")

# Normalized Image (float in [0,1])
plt.subplot(2, 3, 5)
plt.title("Normalized [0,1]")
plt.imshow(img_norm, cmap="gray", vmin=0, vmax=1)
plt.axis("off")

# Absolute Difference
plt.subplot(2, 3, 6)
plt.title("Absolute Difference")
plt.imshow(diff, cmap="gray", vmin=0, vmax=255)
plt.axis("off")

plt.tight_layout()
plt.show()

"""
Summary:
- Brightness and contrast adjustments are basic arithmetic operations on images.
- Normalization prepares images for machine learning models.
- Global statistics help quantify image intensity and variation.
- Image differences reveal pixel-level changes, foundational for CV tasks.
"""
