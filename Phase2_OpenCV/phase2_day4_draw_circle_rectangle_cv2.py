"""
PHASE 2 — OpenCV Image Processing Core
Day 4: Drawing Shapes & Text

This script demonstrates:
- Drawing rectangles, circles, and text on images
- Understanding OpenCV coordinate system
- Using BGR colors
- Visualizing annotations with matplotlib
"""

import cv2
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load image
# -----------------------------
img = cv2.imread("sample.jpg")

if img is None:
    raise FileNotFoundError("Image not found. Check file path.")

print("Image Info:")
print("Shape:", img.shape)
print("Dtype:", img.dtype)
print("-" * 40)

# Make a copy so original stays unchanged
draw_img = img.copy()

# -----------------------------
# 2. Draw Rectangle
# -----------------------------
"""
cv2.rectangle(img, pt1, pt2, color, thickness)
- pt1: top-left corner (x, y)
- pt2: bottom-right corner (x, y)
- color: BGR
"""

cv2.rectangle(
    draw_img,
    (50, 50),
    (250, 200),
    color=(0, 255, 0),   # Green
    thickness=2
)

# -----------------------------
# 3. Draw Circle
# -----------------------------
"""
cv2.circle(img, center, radius, color, thickness)
"""

cv2.circle(
    draw_img,
    center=(150, 125),
    radius=40,
    color=(255, 0, 0),   # Blue
    thickness=3
)

# -----------------------------
# 4. Draw Filled Circle
# -----------------------------
cv2.circle(
    draw_img,
    center=(350, 125),
    radius=30,
    color=(0, 0, 255),   # Red
    thickness=-1         # Filled
)

# -----------------------------
# 5. Put Text
# -----------------------------
"""
cv2.putText(img, text, org, font, fontScale, color, thickness)
"""

cv2.putText(
    draw_img,
    text="Object A",
    org=(50, 40),  # bottom-left corner of text
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=0.9,
    color=(0, 255, 0),
    thickness=2
)

cv2.putText(
    draw_img,
    text="Center Point",
    org=(110, 180),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=0.6,
    color=(255, 0, 0),
    thickness=2,
    lineType=cv2.LINE_AA
)

# -----------------------------
# 6. Display result
# -----------------------------
# Convert BGR → RGB for matplotlib
draw_img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 5))
plt.imshow(draw_img_rgb)
plt.title("Day 4 — Drawing Shapes & Text")
plt.axis("off")
plt.show()
