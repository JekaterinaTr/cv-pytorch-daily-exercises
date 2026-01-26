"""
Computer Vision Daily Practice (NumPy & PIL)
Day 10: NumPy ↔ Image Conversion with Modifications

This script demonstrates:
- Loading an image and converting to NumPy
- Modifying pixels using NumPy
- Converting images to float and normalized float
- Grayscale conversion using PIL and NumPy
- Creating an image from scratch with gradients


"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load image and convert to NumPy
# -----------------------------
img_path = r"F:\14_pollen_dataset\3_Quercus\images\AT6_GRE06_lat_15052025-112901.jpg"
img = Image.open(img_path).convert("RGB")  # Ensure image is in RGB
img_np = np.array(img)

print("Image shape:", img_np.shape)         # (H, W, 3)
print("Image dtype:", img_np.dtype)         # uint8
print("Min pixel value:", img_np.min())     # 0–255
print("Max pixel value:", img_np.max())     # 0–255
print("-" * 50)

# -----------------------------
# 2. Modify pixels using NumPy
# -----------------------------
# Create a copy to avoid changing the original image
img_modified = img_np.copy()

# Set top-left 100x100 region to pure red
img_modified[0:100, 0:100, 0] = 255  # Red channel
img_modified[0:100, 0:100, 1] = 0    # Green channel
img_modified[0:100, 0:100, 2] = 0    # Blue channel

# Convert back to PIL image for display
img_modified_pil = Image.fromarray(img_modified)

plt.imshow(img_modified_pil)
plt.title("Modified Image (Red Patch)")
plt.axis("off")
plt.show()

# Explanation:
# - We use .copy() to avoid modifying the original image.
# - Direct pixel access via NumPy makes it fast and flexible.

# -----------------------------
# 3. Convert image to float and normalize
# -----------------------------
# Convert to float without scaling
img_float = img_np.astype(np.float64)

# Normalize to range [0,1]
img_float_norm = img_np / 255.0

print("img_float dtype:", img_float.dtype, "max value:", img_float.max())
print("img_float_norm dtype:", img_float_norm.dtype, "max value:", img_float_norm.max())
print("-" * 50)

# Explanation:
# - astype(float64) preserves original pixel range (0–255)
# - Division by 255 normalizes values for ML models or float computations

# -----------------------------
# 4. Grayscale conversion
# -----------------------------
# Using PIL's weighted conversion (human perception)
img_gray = img.convert("L")

# Using NumPy mean of channels (simple average)
gray_np = img_np.mean(axis=2).astype(np.uint8)
gray_pil = Image.fromarray(gray_np)

plt.figure(figsize=(8, 12))

plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(img_gray, cmap="gray", vmin=0, vmax=255)
plt.title("Grayscale (PIL Weighted)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(gray_pil, cmap="gray", vmin=0, vmax=255)
plt.title("Grayscale (NumPy Mean)")
plt.axis("off")

plt.show()

# Explanation:
# - PIL uses weighted sum: 0.299*R + 0.587*G + 0.114*B (perceptually accurate)
# - NumPy mean averages all channels equally, simpler but less accurate

# -----------------------------
# 5. Create image from scratch using NumPy
# -----------------------------
h, w = 256, 256
generated = np.zeros((h, w, 3), dtype=np.uint8)

# Horizontal red gradient
for x in range(w):
    generated[:, x, 0] = x

# Vertical green gradient
for y in range(h):
    generated[y, :, 1] = y

generated_pil = Image.fromarray(generated)

plt.imshow(generated_pil)
plt.title("Generated Image (NumPy Gradients)")
plt.axis("off")
plt.show()

# Explanation:
# - Red gradient changes horizontally, green gradient changes vertically
# - Blue remains 0, producing a combined RGB gradient
# - Demonstrates how images can be fully generated with NumPy
