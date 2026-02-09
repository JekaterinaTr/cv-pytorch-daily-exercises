"""
PHASE 2 — OpenCV Image Processing Core
Day 10: Image Blending (Weighted Combination)

Concepts:
- Weighted pixel-wise image combination
- Alpha and beta weights
- Blending overlays, watermarks, and transitions
- Grayscale blending
- Crossfade effect for videos
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load two images
# -----------------------------
img1 = cv2.imread("image1.jpg")
img2 = cv2.imread("image2.jpg")

if img1 is None or img2 is None:
    raise FileNotFoundError("One of the images was not found")

print("Image1 shape:", img1.shape)
print("Image2 shape:", img2.shape)

# -----------------------------
# 2. Resize second image to match first
# -----------------------------
img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
print("Resized img2 shape:", img2_resized.shape)

# -----------------------------
# 3. Simple image addition (unsafe)
# -----------------------------
added = img1 + img2_resized  # May overflow
plt.imshow(cv2.cvtColor(added, cv2.COLOR_BGR2RGB))
plt.title("Naive addition (may overflow)")
plt.axis("off")
plt.show()

# -----------------------------
# 4. Weighted blending
# -----------------------------
alpha = 0.5
beta = 0.5
gamma = 0
blended = cv2.addWeighted(img1, alpha, img2_resized, beta, gamma)

plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
plt.title("Blended Image (α=0.5, β=0.5)")
plt.axis("off")
plt.show()

# -----------------------------
# 5. Vary blending weights
# -----------------------------
weights = [(0.7, 0.3), (0.3, 0.7)]
plt.figure(figsize=(10, 5))
for i, (a, b) in enumerate(weights):
    result = cv2.addWeighted(img1, a, img2_resized, b, 0)
    plt.subplot(1, 2, i+1)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f"α={a}, β={b}")
    plt.axis("off")
plt.show()

# -----------------------------
# 6. Overlay effect (e.g., watermark)
# -----------------------------
overlay = cv2.addWeighted(img1, 0.8, img2_resized, 0.2, 0)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Overlay (α=0.8, β=0.2)")
plt.axis("off")
plt.show()

# -----------------------------
# 7. Grayscale blending
# -----------------------------
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
blended_gray = cv2.addWeighted(gray1, 0.5, gray2, 0.5, 0)

plt.imshow(blended_gray, cmap="gray")
plt.title("Blended Grayscale")
plt.axis("off")
plt.show()

# -----------------------------
# 8. Blending as transition (crossfade)
# -----------------------------
import numpy as np

for alpha in np.linspace(0, 1, 5):
    beta = 1 - alpha
    transition = cv2.addWeighted(img1, alpha, img2_resized, beta, 0)
    plt.imshow(cv2.cvtColor(transition, cv2.COLOR_BGR2RGB))
    plt.title(f"α={alpha:.2f}, β={beta:.2f}")
    plt.axis("off")
    plt.show()

"""
Summary:
- addWeighted safely blends images using alpha and beta weights
- Changing alpha and beta allows control over which image dominates
- Useful for overlays, watermarks, and smooth transitions
- Works with both color and grayscale images
"""
