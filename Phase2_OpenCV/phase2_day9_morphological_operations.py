"""
PHASE 2 — OpenCV Image Processing Core
Day 9: Morphological Operations (Erosion & Dilation)

Concepts:
- Binary images and structuring elements
- Erosion and dilation
- Opening and closing
- Noise removal and gap filling
- Morphology as preprocessing for contours
"""

import cv2
import numpy as np
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
# 4. Binary thresholding
# -----------------------------

_, thresh = cv2.threshold(
    blur, 127, 255, cv2.THRESH_BINARY
)

# -----------------------------
# 5. Create structuring element
# -----------------------------

kernel = np.ones((5, 5), np.uint8)

print("Kernel shape:", kernel.shape)

# -----------------------------
# 6. Erosion
# -----------------------------

eroded = cv2.erode(
    thresh,
    kernel,
    iterations=1
)

# -----------------------------
# 7. Dilation
# -----------------------------

dilated = cv2.dilate(
    thresh,
    kernel,
    iterations=1
)

# -----------------------------
# 8. Opening (erosion → dilation)
# -----------------------------

opening = cv2.morphologyEx(
    thresh,
    cv2.MORPH_OPEN,
    kernel
)

# -----------------------------
# 9. Closing (dilation → erosion)
# -----------------------------

closing = cv2.morphologyEx(
    thresh,
    cv2.MORPH_CLOSE,
    kernel
)

# -----------------------------
# 10. Contour comparison
# -----------------------------

contours_raw, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

contours_open, _ = cv2.findContours(
    opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

print("Contours before morphology:", len(contours_raw))
print("Contours after opening:", len(contours_open))

# -----------------------------
# 11. Visualization
# -----------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Grayscale")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Binary")
plt.imshow(thresh, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Erosion")
plt.imshow(eroded, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.title("Dilation")
plt.imshow(dilated, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("Opening")
plt.imshow(opening, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.title("Closing")
plt.imshow(closing, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

"""
Summary:
- Morphological operations modify object shapes
- Erosion shrinks objects and removes noise
- Dilation expands objects and fills gaps
- Opening cleans noise
- Closing repairs broken shapes
- Morphology improves contour detection reliability
"""
