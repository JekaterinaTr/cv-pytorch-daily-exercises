"""
Computer Vision Daily Practice (OpenCV + PyTorch)
PHASE 1 - Python, NumPy & Image Foundations

Day 1: Arrays & Fake Images

This script demonstrates:
- How digital images can be represented as numerical arrays.
- How to generate synthetic grayscale and RGB images.
- How to visualize images using Matplotlib.

Key Concepts:
1. Grayscale image: 2D array, each value represents pixel intensity (0=black, 255=white).
2. RGB image: 3D array, each pixel has three color channels (Red, Green, Blue).
3. plt.imshow: renders an array as an image.
4. cmap: colormap used to map numerical values to colors for single-channel images.

"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Create a fake grayscale image
# -----------------------------

"""
Theory:
- Grayscale images only store intensity (brightness) information.
- Each pixel is a value between 0 (black) and 255 (white).
- np.random.randint generates random integers in the specified range:
    low=0, high=256 → pixel intensity range
    size=(256, 256) → image dimensions
    dtype=np.uint8 → 8-bit unsigned integer, standard for images
"""

gray_img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

print("Grayscale Image Info:")
print("Shape:", gray_img.shape)
print("Dtype:", gray_img.dtype)
print("Min value:", gray_img.min())
print("Max value:", gray_img.max())
print("-"*40)


# -----------------------------
# 2. Create a fake RGB image
# -----------------------------

"""
Theory:
- RGB images have 3 channels: Red, Green, Blue.
- Each channel is an 8-bit integer from 0 to 255.
- size=(256, 256, 3) → 3D array: height x width x channels
- Random values produce a noisy color image.
"""

rgb_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

print("RGB Image Info:")
print("Shape:", rgb_img.shape)
print("Dtype:", rgb_img.dtype)
print("Min value:", rgb_img.min())
print("Max value:", rgb_img.max())
print("-"*40)


# -----------------------------
# 3. Visualization
# -----------------------------

"""
Theory:
- plt.figure(figsize=(width, height)) → sets figure size in inches
- plt.subplot(rows, cols, index) → selects subplot in a grid
- plt.imshow() → renders array as an image
    - cmap="gray" → for grayscale images only
    - RGB arrays are automatically displayed in color
- plt.axis("off") → hides axes for cleaner visualization
"""

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Grayscale Image")
plt.imshow(gray_img, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("RGB Image")
plt.imshow(rgb_img)
plt.axis("off")

plt.show()

"""
Summary:
- Grayscale images: 2D arrays, one intensity per pixel.
- RGB images: 3D arrays, three color values per pixel.
- Synthetic images help in understanding image representation and testing processing algorithms.

"""
