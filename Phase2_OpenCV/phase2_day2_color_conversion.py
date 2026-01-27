"""
Computer Vision Daily Practice (OpenCV + PyTorch)
PHASE 2 — OpenCV Image Processing Core

Day 2: Color Conversion (BGR ↔ RGB ↔ Grayscale)

This script demonstrates:
- OpenCV BGR image loading
- BGR → RGB conversion
- RGB → Grayscale conversion
- Grayscale → 3-channel conversion
- Correct visualization using matplotlib

Key Concepts:
1. OpenCV uses BGR by default
2. Most libraries expect RGB
3. Grayscale reduces channel dimensionality
4. cv2.cvtColor is the standard conversion method
"""

import cv2
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load image (BGR)
# -----------------------------

img = cv2.imread("sample.jpg")

if img is None:
    raise FileNotFoundError("Image not found. Check file path.")

print("Original Image (BGR):")
print("Shape:", img.shape)
print("Dtype:", img.dtype)
print("Min:", img.min())
print("Max:", img.max())
print("-" * 40)

# -----------------------------
# 2. Convert BGR → RGB
# -----------------------------

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("Converted RGB Image:")
print("Shape:", img_rgb.shape)
print("Dtype:", img_rgb.dtype)
print("-" * 40)

# -----------------------------
# 3. Display BGR vs RGB
# -----------------------------

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("BGR Image (Wrong Colors)")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("RGB Image (Correct Colors)")
plt.imshow(img_rgb)
plt.axis("off")

plt.tight_layout()
plt.show()

# -----------------------------
# 4. Convert RGB → Grayscale
# -----------------------------

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

print("Grayscale Image:")
print("Shape:", img_gray.shape)
print("Dtype:", img_gray.dtype)
print("Min:", img_gray.min())
print("Max:", img_gray.max())
print("-" * 40)

# -----------------------------
# 5. Display Grayscale Image
# -----------------------------

plt.imshow(img_gray, cmap="gray", vmin=0, vmax=255)
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

# -----------------------------
# 6. Convert Grayscale → 3 Channels
# -----------------------------

img_gray_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

print("Grayscale 3-Channel Image:")
print("Shape:", img_gray_3ch.shape)
print("Dtype:", img_gray_3ch.dtype)
print("-" * 40)

# -----------------------------
# 7. Final Summary
# -----------------------------

"""
Summary:
- OpenCV loads images in BGR format
- cv2.cvtColor is used for all color conversions
- RGB is required for correct matplotlib display
- Grayscale images have shape (H, W)
- Grayscale can be expanded to 3 channels if needed
"""
