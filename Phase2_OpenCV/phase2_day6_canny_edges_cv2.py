"""
PHASE 2 — OpenCV Image Processing Core
Day 6: Edge Detection with Canny

Concepts:
- Grayscale conversion
- Noise reduction using Gaussian blur
- Canny edge detection
- Effect of threshold values
- Classic CV preprocessing pipeline
"""

import cv2
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load image
# -----------------------------

img = cv2.imread("sample.jpg")

if img is None:
    raise FileNotFoundError("Image not found")

print("Original Image Info")
print("Shape:", img.shape)
print("Dtype:", img.dtype)
print("-" * 40)

# -----------------------------
# 2. Convert to grayscale
# -----------------------------

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 3. Edge detection without blur
# -----------------------------

edges_no_blur = cv2.Canny(gray, 100, 200)

# -----------------------------
# 4. Apply Gaussian blur
# -----------------------------

blur = cv2.GaussianBlur(gray, (5, 5), 0)

# -----------------------------
# 5. Edge detection after blur
# -----------------------------

edges_blur = cv2.Canny(blur, 100, 200)

# -----------------------------
# 6. Edge detection with different thresholds
# -----------------------------

edges_low_thresh = cv2.Canny(blur, 50, 150)
edges_high_thresh = cv2.Canny(blur, 150, 300)

# -----------------------------
# 7. Visualization
# -----------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Grayscale")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Edges (No Blur)")
plt.imshow(edges_no_blur, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Edges (After Blur)")
plt.imshow(edges_blur, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.title("Low Thresholds (50,150)")
plt.imshow(edges_low_thresh, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("High Thresholds (150,300)")
plt.imshow(edges_high_thresh, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.title("Final Pipeline Output")
plt.imshow(edges_blur, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

"""
Summary:
- Edge detection finds intensity discontinuities
- Grayscale images are required for Canny
- Gaussian blur reduces noise before edge detection
- Lower thresholds detect more edges (including noise)
- Higher thresholds detect fewer but stronger edges
"""
