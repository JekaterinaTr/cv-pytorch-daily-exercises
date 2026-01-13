"""
Computer Vision Daily Practice (OpenCV + PyTorch)
PHASE 1 - Python, NumPy & Image Foundations

Day 4: Image Visualization & Histograms

This script demonstrates:
- How to visualize grayscale images correctly.
- How to plot pixel intensity histograms.
- How brightness and contrast affect histograms.
- How to connect image math with visual perception.

Key Concepts:
1. plt.imshow renders NumPy arrays as images.
2. vmin/vmax control intensity mapping.
3. Histograms reveal brightness and contrast numerically.
4. Brightness shifts histograms; contrast spreads them.
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Create a controlled grayscale image
# -----------------------------

"""
Theory:
- Use a mid-range image (80–180) so brightness changes are visible.
- dtype uint8 represents standard grayscale images.
"""

img = np.random.randint(80, 180, (64, 64), dtype=np.uint8)

print("Original Image Info:")
print("Shape:", img.shape)
print("Min:", img.min())
print("Max:", img.max())
print("-" * 40)

# -----------------------------
# 2. Display the image
# -----------------------------

plt.figure()
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.title("Original Image")
plt.axis("off")
plt.show()

# -----------------------------
# 3. Pixel histogram
# -----------------------------

"""
Theory:
- Histogram shows how pixel intensities are distributed.
- ravel() flattens the image to 1D.
"""

plt.figure()
plt.hist(img.ravel(), bins=256, range=(0, 255))
plt.title("Original Image Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# 4. Brightness adjustment
# -----------------------------

brighter = np.clip(img + 40, 0, 255)
darker = np.clip(img - 40, 0, 255)

# -----------------------------
# 5. Visualize brightness changes
# -----------------------------

plt.figure(figsize=(9, 3))

plt.subplot(1, 3, 1)
plt.imshow(darker, cmap="gray", vmin=0, vmax=255)
plt.title("Darker")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(brighter, cmap="gray", vmin=0, vmax=255)
plt.title("Brighter")
plt.axis("off")

plt.tight_layout()
plt.show()

# -----------------------------
# 6. Compare histograms
# -----------------------------

plt.figure(figsize=(9, 3))

plt.subplot(1, 3, 1)
plt.hist(darker.ravel(), bins=256, range=(0, 255))
plt.title("Darker Histogram")

plt.subplot(1, 3, 2)
plt.hist(img.ravel(), bins=256, range=(0, 255))
plt.title("Original Histogram")

plt.subplot(1, 3, 3)
plt.hist(brighter.ravel(), bins=256, range=(0, 255))
plt.title("Brighter Histogram")

plt.tight_layout()
plt.show()

# -----------------------------
# 7. Contrast adjustment
# -----------------------------

"""
Theory:
- Multiplication increases contrast.
- Histogram spreads wider.
"""

high_contrast = np.clip(img * 1.5, 0, 255)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(high_contrast, cmap="gray", vmin=0, vmax=255)
plt.title("High Contrast")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.hist(high_contrast.ravel(), bins=256, range=(0, 255))
plt.title("High Contrast Histogram")

plt.tight_layout()
plt.show()

"""
Summary:
- Visualization connects numeric image operations to perception.
- Brightness shifts histograms left/right.
- Contrast spreads or compresses histograms.
- These ideas are fundamental for CV preprocessing and ML pipelines.
"""
