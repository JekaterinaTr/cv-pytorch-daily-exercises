"""
Computer Vision Daily Practice (OpenCV + PyTorch)
PHASE 1 - Python, NumPy & Image Foundations

Day 2: 2D Gradient, ROI, and Flips

This script demonstrates:
- How to create a smooth 2D gradient image as a NumPy array.
- How to extract a Region of Interest (ROI) using slicing.
- How to perform horizontal, vertical, and 180° flips.
- How to visualize multiple images in a single figure.

Key Concepts:
1. Images are 2D arrays of pixel intensities.
2. ROI extraction via array slicing.
3. Flipping images using NumPy slicing.
4. Matplotlib subplotting and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Create a 2D gradient image
# -----------------------------

"""
Theory:
- Use np.arange to create 1D indices for rows and columns.
- np.meshgrid generates 2D coordinate arrays from 1D indices.
- Combine coordinates to form a gradient: ((xx + yy) // 2).
- Convert to np.uint8 to represent standard 8-bit grayscale pixels (0–255).
"""

x = np.arange(256)
y = np.arange(256)
xx, yy = np.meshgrid(x, y)
img = ((xx + yy) // 2).astype(np.uint8)

print("Gradient Image Info:")
print("Shape:", img.shape)
print("Dtype:", img.dtype)
print("Min value:", img.min())
print("Max value:", img.max())
print("-"*40)


# -----------------------------
# 2. Crop a Region of Interest (ROI)
# -----------------------------

"""
Theory:
- Select a subarray using slicing: img[y1:y2, x1:x2].
- ROI contains only a small portion of the original image.
- Slicing creates a view; changes affect the original array unless copied.
"""

y1, y2 = 0, 30
x1, x2 = 0, 30
roi = img[y1:y2, x1:x2]

print("ROI Info:")
print("Shape:", roi.shape)
print("Dtype:", roi.dtype)
print("Min value:", roi.min())
print("Max value:", roi.max())
print("-"*40)


# -----------------------------
# 3. Flip operations
# -----------------------------

"""
Theory:
- Horizontal flip: img[:, ::-1] → mirrors left-right.
- Vertical flip: img[::-1, :] → mirrors top-bottom.
- Both flips: img[::-1, ::-1] → 180° rotation.
- Flipping is done efficiently using NumPy slicing.
"""

horizontal_flip = img[:, ::-1]
vertical_flip = img[::-1, :]
both_flips = img[::-1, ::-1]


# -----------------------------
# 4. Visualization
# -----------------------------

"""
Theory:
- plt.figure(figsize) → set figure size.
- plt.subplot(rows, cols, index) → select subplot.
- plt.imshow() → render array as image (cmap="gray" for grayscale).
- plt.axis("off") → remove axes for clarity.
- plt.tight_layout() → adjust spacing automatically.
"""

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.title("Original Gradient")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Cropped ROI")
plt.imshow(roi, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Horizontal Flip")
plt.imshow(horizontal_flip, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Vertical Flip")
plt.imshow(vertical_flip, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()


"""
Summary:
- 2D gradient image illustrates intensity variation across pixels.
- ROI extraction allows focusing on a subset of the image.
- Horizontal and vertical flips are basic image augmentation techniques.
- These operations are foundational for computer vision preprocessing.

"""
