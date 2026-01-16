"""
Computer Vision Daily Practice (OpenCV + PyTorch)
PHASE 1 - Python, NumPy & Image Foundations

Day 7: Image I/O with PIL (Load, Save, Convert)

This script demonstrates:
- How to load images from disk using PIL
- Convert images between PIL and NumPy
- Convert RGB → Grayscale
- Process images (brightness/darkness)
- Save processed images back to disk
- Stack images and save combined results
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load an image
# -----------------------------

"""
Theory:
- PIL.Image.open reads image from disk.
- img.mode shows color format (RGB, L, etc.)
- img.size gives (width, height)
"""

# Change the filename to a valid image in your folder
img = Image.open("sample.jpg")

print("PIL Image type:", type(img))
print("Image mode:", img.mode)
print("Image size (W,H):", img.size)
print("-" * 40)

# -----------------------------
# 2. Convert PIL to NumPy array
# -----------------------------

img_np = np.array(img)
print("NumPy array shape:", img_np.shape)
print("Dtype:", img_np.dtype)
print("-" * 40)

# -----------------------------
# 3. Convert to Grayscale
# -----------------------------

img_gray = img.convert("L")
img_gray_np = np.array(img_gray)

print("Grayscale image shape:", img_gray_np.shape)
print("Dtype:", img_gray_np.dtype)
print("-" * 40)

# -----------------------------
# 4. Brightness/Darkness adjustment
# -----------------------------

brighter = np.clip(img_gray_np + 40, 0, 255).astype(np.uint8)
darker = np.clip(img_gray_np - 40, 0, 255).astype(np.uint8)

# -----------------------------
# 5. Save processed images
# -----------------------------

Image.fromarray(img_gray_np).save("gray.png")
Image.fromarray(brighter).save("bright.png")
Image.fromarray(darker).save("dark.png")

print("Saved gray.png, bright.png, dark.png")
print("-" * 40)

# -----------------------------
# 6. Visualize original vs processed
# -----------------------------

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img_gray_np, cmap="gray", vmin=0, vmax=255)
plt.title("Original Grayscale")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(brighter, cmap="gray", vmin=0, vmax=255)
plt.title("Brighter")
plt.axis("off")

plt.tight_layout()
plt.show()

# -----------------------------
# 7. Stack multiple images
# -----------------------------

stacked = np.hstack([darker, img_gray_np, brighter])
Image.fromarray(stacked).save("comparison_stack.png")
print("Saved comparison_stack.png")

# -----------------------------
# Summary
# -----------------------------

"""
- PIL loads and saves images easily.
- Convert to NumPy arrays for CV processing.
- Brightness/darkness adjustments can be applied with np.clip.
- Stacking images helps in visualization or dataset preparation.
"""
