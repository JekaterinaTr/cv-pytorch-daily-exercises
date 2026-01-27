"""
Computer Vision Daily Practice (OpenCV + PyTorch)
PHASE 2 — OpenCV Image Processing Core

Day 1: Load & Show Images with OpenCV

This script demonstrates:
- Loading images using OpenCV
- Understanding OpenCV image representation
- Displaying images using OpenCV windows
- Comparing OpenCV and Matplotlib visualization
- Loading grayscale images

Key Concepts:
1. OpenCV images are NumPy arrays
2. OpenCV loads images in BGR order
3. cv2.imshow uses event-based windows
4. Grayscale images reduce channel dimensionality
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load image with OpenCV
# -----------------------------

"""
Theory:
- cv2.imread loads images as NumPy arrays
- Default color format is BGR (not RGB!)
- Pixel values range from 0–255 (uint8)
"""

# Replace with a valid image path
img = cv2.imread("sample.jpg")

if img is None:
    raise FileNotFoundError("Image not found. Check file path.")

print("OpenCV Image Info:")
print("Shape:", img.shape)
print("Dtype:", img.dtype)
print("Min pixel value:", img.min())
print("Max pixel value:", img.max())
print("-" * 40)

# -----------------------------
# 2. Display image using OpenCV
# -----------------------------

"""
Theory:
- cv2.imshow opens a window
- cv2.waitKey waits for keyboard input
- cv2.destroyAllWindows closes all windows
"""

cv2.imshow("OpenCV Image (BGR)", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------------
# 3. Display image using Matplotlib
# -----------------------------

"""
Theory:
- Matplotlib expects RGB format
- OpenCV loads images in BGR → colors appear incorrect
"""

plt.figure(figsize=(4, 4))
plt.title("Matplotlib Display (Wrong Colors)")
plt.imshow(img)
plt.axis("off")
plt.show()

# -----------------------------
# 4. Inspect color channels
# -----------------------------

"""
Theory:
- Channel order in OpenCV:
  Channel 0 → Blue
  Channel 1 → Green
  Channel 2 → Red
"""

blue = img[:, :, 0]
green = img[:, :, 1]
red = img[:, :, 2]

print("Channel Shapes:")
print("Blue:", blue.shape)
print("Green:", green.shape)
print("Red:", red.shape)
print("-" * 40)

# -----------------------------
# 5. Load grayscale image
# -----------------------------

"""
Theory:
- cv2.IMREAD_GRAYSCALE loads image as single channel
- Shape becomes (H, W)
"""

img_gray = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)

print("Grayscale Image Info:")
print("Shape:", img_gray.shape)
print("Dtype:", img_gray.dtype)
print("-" * 40)

cv2.imshow("Grayscale Image", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------------
# 6. Compare RGB and Grayscale
# -----------------------------

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("Original (BGR shown as RGB - wrong)")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Grayscale")
plt.imshow(img_gray, cmap="gray", vmin=0, vmax=255)
plt.axis("off")

plt.tight_layout()
plt.show()

# -----------------------------
# 7. Summary (Important)
# -----------------------------

"""
Key Takeaways:
- OpenCV images are NumPy arrays
- OpenCV uses BGR color ordering
- Grayscale images have no channel dimension
- cv2.imshow requires waitKey
- Matplotlib auto-assumes RGB format
"""
