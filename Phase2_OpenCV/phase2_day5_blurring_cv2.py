"""
PHASE 2 — OpenCV Image Processing Core
Day 5: Image Blurring & Smoothing

Concepts:
- Noise reduction using blurring
- Average blur vs Gaussian blur
- Effect of kernel size
- Blurring as preprocessing for edge detection
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

# Convert BGR → RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -----------------------------
# 2. Average Blur
# -----------------------------

blur_avg = cv2.blur(img, (5, 5))

# -----------------------------
# 3. Gaussian Blur
# -----------------------------

blur_gaussian_5 = cv2.GaussianBlur(img, (5, 5), sigmaX=0)
blur_gaussian_15 = cv2.GaussianBlur(img, (15, 15), sigmaX=0)

# -----------------------------
# 4. Edge detection comparison
# -----------------------------

edges_no_blur = cv2.Canny(img, 100, 200)
edges_blur = cv2.Canny(blur_gaussian_5, 100, 200)

# -----------------------------
# 5. Visualization
# -----------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Original")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Average Blur (5x5)")
plt.imshow(cv2.cvtColor(blur_avg, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Gaussian Blur (5x5)")
plt.imshow(cv2.cvtColor(blur_gaussian_5, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2, 3, 4)
plt.title("Gaussian Blur (15x15)")
plt.imshow(cv2.cvtColor(blur_gaussian_15, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("Edges (No Blur)")
plt.imshow(edges_no_blur, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.title("Edges (After Blur)")
plt.imshow(edges_blur, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

"""
Summary:
- Blurring smooths images and reduces noise
- Average blur treats all neighbors equally
- Gaussian blur weights center pixels more
- Larger kernels increase smoothing but lose detail
- Blurring improves edge detection quality
"""
