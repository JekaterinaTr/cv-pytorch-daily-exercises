"""
Computer Vision Daily Practice (OpenCV + PyTorch)
PHASE 1 — Python, NumPy & Image Foundations

Day 9: Color Spaces (RGB ↔ Grayscale)

This script demonstrates:
- RGB images as 3-channel NumPy arrays
- Splitting RGB channels
- Visualizing individual color channels
- Converting RGB images to grayscale
- Manual grayscale conversion using luminance formula
- Converting grayscale back to 3-channel format

Key Concepts:
1. RGB images have shape (H, W, 3)
2. Each channel stores intensity of one color
3. Grayscale images reduce dimensionality
4. Luminance weights reflect human perception
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load RGB image
# -----------------------------

"""
Theory:
- RGB images contain Red, Green, Blue channels
- PIL stores images as Image objects
"""

# Change filename to an existing image
img = Image.open("sample.jpg").convert("RGB")
img_np = np.array(img)

print("RGB image shape:", img_np.shape)
print("Dtype:", img_np.dtype)
print("-" * 40)

# -----------------------------
# 2. Split RGB channels
# -----------------------------

"""
Theory:
- Each channel is a 2D array (H, W)
"""

R = img_np[:, :, 0]
G = img_np[:, :, 1]
B = img_np[:, :, 2]

print("Red channel shape:", R.shape)
print("Green channel shape:", G.shape)
print("Blue channel shape:", B.shape)
print("-" * 40)

# -----------------------------
# 3. Visualize RGB channels
# -----------------------------

plt.figure(figsize=(9, 3))

plt.subplot(1, 3, 1)
plt.title("Red Channel")
plt.imshow(R, cmap="gray", vmin=0, vmax=255)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Green Channel")
plt.imshow(G, cmap="gray", vmin=0, vmax=255)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Blue Channel")
plt.imshow(B, cmap="gray", vmin=0, vmax=255)
plt.axis("off")

plt.tight_layout()
plt.show()

# -----------------------------
# 4. Convert RGB → Grayscale (PIL)
# -----------------------------

"""
Theory:
- Grayscale reduces image to one channel
- Uses weighted sum of RGB values
"""

img_gray = img.convert("L")
img_gray_np = np.array(img_gray)

print("Grayscale image shape:", img_gray_np.shape)
print("-" * 40)

# -----------------------------
# 5. Visualize RGB vs Grayscale
# -----------------------------

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("RGB Image")
plt.imshow(img_np)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Grayscale Image")
plt.imshow(img_gray_np, cmap="gray", vmin=0, vmax=255)
plt.axis("off")

plt.tight_layout()
plt.show()

# -----------------------------
# 6. Manual grayscale conversion
# -----------------------------

"""
Theory:
- Luminance formula (ITU-R BT.601):
  Gray = 0.299R + 0.587G + 0.114B
- Green contributes most to brightness perception
"""

gray_manual = (
    0.299 * R +
    0.587 * G +
    0.114 * B
).astype(np.uint8)

plt.imshow(gray_manual, cmap="gray", vmin=0, vmax=255)
plt.title("Manual Grayscale Conversion")
plt.axis("off")
plt.show()

# -----------------------------
# 7. Convert grayscale to 3-channel image
# -----------------------------

"""
Theory:
- Some models expect 3-channel input
- Duplicate grayscale channel across RGB
"""

gray_3ch = np.dstack([img_gray_np] * 3)
print("Grayscale 3-channel shape:", gray_3ch.shape)

plt.imshow(gray_3ch)
plt.title("Grayscale as 3-Channel RGB")
plt.axis("off")
plt.show()

"""
Summary:
- RGB images contain three color channels
- Grayscale reduces complexity and computation
- Manual conversion reveals luminance weighting
- Channel manipulation is core to CV pipelines
"""
