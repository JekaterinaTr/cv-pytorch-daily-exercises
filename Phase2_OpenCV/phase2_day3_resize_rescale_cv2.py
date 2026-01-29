"""
Phase 2 — OpenCV Image Processing Core
Day 3: Resize & Crop Images

Concepts:
- Resizing images (fixed size & scale factors)
- Cropping regions of interest (ROI) using NumPy slicing
- Visual comparison of original, resized, and cropped images
"""

import cv2
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Image
# -----------------------------
img = cv2.imread("sample.jpg")

if img is None:
    raise FileNotFoundError("Image not found. Check your file path.")

print("Original Image")
print("Shape:", img.shape)
print("Dtype:", img.dtype)
print("-" * 40)

# -----------------------------
# 2. Resize to fixed size (256x256)
# -----------------------------
img_resized = cv2.resize(img, (256, 256))
print("Resized Image (256x256)")
print("Shape:", img_resized.shape)
print("-" * 40)

# -----------------------------
# 3. Resize using scale factors (50%)
# -----------------------------
img_half = cv2.resize(
    img,
    None,
    fx=0.5,
    fy=0.5,
    interpolation=cv2.INTER_LINEAR
)
print("Resized Image (50%)")
print("Shape:", img_half.shape)
print("-" * 40)

# -----------------------------
# 4. Crop center region (200x200)
# -----------------------------
h, w, _ = img.shape
center_y, center_x = h // 2, w // 2
crop_size = 200

y1 = center_y - crop_size // 2
y2 = center_y + crop_size // 2
x1 = center_x - crop_size // 2
x2 = center_x + crop_size // 2

img_crop = img[y1:y2, x1:x2]
print("Cropped Center 200x200")
print("Shape:", img_crop.shape)
print("-" * 40)

# -----------------------------
# 5. Display Original, Resized, Cropped
# -----------------------------
# Convert BGR -> RGB for matplotlib display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_crop_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Resized 256x256")
plt.imshow(img_resized_rgb)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Cropped 200x200")
plt.imshow(img_crop_rgb)
plt.axis("off")

plt.tight_layout()
plt.show()
