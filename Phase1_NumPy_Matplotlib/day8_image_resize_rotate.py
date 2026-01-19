"""
Computer Vision Daily Practice (OpenCV + PyTorch)
PHASE 1 — Python, NumPy & Image Foundations

Day 8: Image Resize & Rotate (PIL)

This script demonstrates:
- Loading images from disk using PIL
- Resizing images to fixed dimensions
- Resizing images while preserving aspect ratio
- Rotating images (90° and arbitrary angles)
- Combining resize + rotation
- Saving processed images

Key Concepts:
1. resize() expects (width, height)
2. Rotation is counterclockwise by default
3. expand=True prevents cropping during rotation
4. Resizing standardizes inputs for ML pipelines
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load image
# -----------------------------

"""
Theory:
- PIL loads images as Image objects
- img.size → (width, height)
- img.mode → color format (RGB, L)
"""

# Change this filename to an image in your directory
img = Image.open("sample.jpg")

print("Original size (W,H):", img.size)
print("Image mode:", img.mode)
print("-" * 40)

# -----------------------------
# 2. Resize to fixed dimensions
# -----------------------------

"""
Theory:
- resize() requires (width, height)
- Common preprocessing step for ML models
"""

img_resized = img.resize((128, 128))
print("Resized (fixed) size:", img_resized.size)

# -----------------------------
# 3. Resize while keeping aspect ratio
# -----------------------------

"""
Theory:
- Maintain proportions to avoid distortion
"""

orig_w, orig_h = img.size
new_w = 128
new_h = int((new_w / orig_w) * orig_h)

img_resized_aspect = img.resize((new_w, new_h))
print("Resized (aspect ratio) size:", img_resized_aspect.size)
print("-" * 40)

# -----------------------------
# 4. Rotate images
# -----------------------------

"""
Theory:
- rotate(angle) → counterclockwise
- expand=True fits entire rotated image
"""

img_rotated_90 = img.rotate(90)
img_rotated_45 = img.rotate(45, expand=True)

# -----------------------------
# 5. Combine resize + rotate
# -----------------------------

"""
Theory:
- Typical augmentation pipeline
"""

img_processed = img.resize((128, 128)).rotate(30, expand=True)

# -----------------------------
# 6. Save processed images
# -----------------------------

img_resized.save("resized_128.png")
img_rotated_90.save("rotated_90.png")
img_rotated_45.save("rotated_45.png")
img_processed.save("resized_rotated.png")

print("Saved resized and rotated images")
print("-" * 40)

# -----------------------------
# 7. Visualization
# -----------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Original")
plt.imshow(np.array(img))
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Resized 128x128")
plt.imshow(np.array(img_resized))
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Rotated 45°")
plt.imshow(np.array(img_rotated_45))
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Resized + 30° Rotated")
plt.imshow(np.array(img_processed))
plt.axis("off")

plt.tight_layout()
plt.show()

"""
Summary:
- resize() standardizes image dimensions
- Preserving aspect ratio prevents distortion
- rotate() is counterclockwise by default
- expand=True avoids cropping
- Resize + rotate form the basis of image augmentation
"""
