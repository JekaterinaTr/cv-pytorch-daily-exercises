"""
PHASE 2 — OpenCV Image Processing Core
Day 7: Thresholding (Binary & Adaptive)

Concepts:
- Global binary thresholding
- Inverse thresholding
- Adaptive thresholding
- Effect of lighting conditions
- Thresholding as preprocessing for contours
"""

import cv2
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load image
# -----------------------------

img = cv2.imread("sample.jpg")

if img is None:
    raise FileNotFoundError("Image not found")

# -----------------------------
# 2. Convert to grayscale
# -----------------------------

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 3. Apply Gaussian blur
# -----------------------------

blur = cv2.GaussianBlur(gray, (5, 5), 0)

# -----------------------------
# 4. Global binary threshold
# -----------------------------

_, thresh_binary = cv2.threshold(
    blur, 127, 255, cv2.THRESH_BINARY
)

# -----------------------------
# 5. Inverse binary threshold
# -----------------------------

_, thresh_binary_inv = cv2.threshold(
    blur, 127, 255, cv2.THRESH_BINARY_INV
)

# -----------------------------
# 6. Adaptive mean threshold
# -----------------------------

thresh_adaptive_mean = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

# -----------------------------
# 7. Adaptive Gaussian threshold
# -----------------------------

thresh_adaptive_gaussian = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

# -----------------------------
# 8. Visualization
# -----------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Grayscale")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Blurred")
plt.imshow(blur, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Binary Threshold")
plt.imshow(thresh_binary, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.title("Binary Inverse")
plt.imshow(thresh_binary_inv, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("Adaptive Mean")
plt.imshow(thresh_adaptive_mean, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.title("Adaptive Gaussian")
plt.imshow(thresh_adaptive_gaussian, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

"""
Summary:
- Thresholding separates foreground from background
- Global threshold uses one fixed value
- Adaptive threshold adjusts locally
- Adaptive methods handle uneven lighting
- Thresholding is key before contour detection
"""
