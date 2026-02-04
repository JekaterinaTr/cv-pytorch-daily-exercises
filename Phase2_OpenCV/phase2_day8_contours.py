"""
PHASE 2 — OpenCV Image Processing Core
Day 8: Contours (Finding & Drawing Objects)

Concepts:
- Binary images as input for contours
- Contour detection and hierarchy
- Drawing contours
- Bounding boxes
- Filtering contours by area
"""

import cv2
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load image
# -----------------------------

img = cv2.imread("sample.jpg")

if img is None:
    raise FileNotFoundError("Image not found")

# -----------------------------
# 2. Convert to grayscale
# -----------------------------

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 3. Apply Gaussian blur
# -----------------------------

blur = cv2.GaussianBlur(gray, (5, 5), 0)

# -----------------------------
# 4. Binary thresholding
# -----------------------------

_, thresh = cv2.threshold(
    blur, 127, 255, cv2.THRESH_BINARY
)

# -----------------------------
# 5. Find contours
# -----------------------------

contours, hierarchy = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("Contours found:", len(contours))

# -----------------------------
# 6. Draw all contours
# -----------------------------

img_contours = img.copy()

cv2.drawContours(
    img_contours,
    contours,
    -1,
    (0, 255, 0),
    2
)

# -----------------------------
# 7. Draw bounding boxes
# -----------------------------

img_boxes = img.copy()

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(
        img_boxes,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

# -----------------------------
# 8. Filter contours by area
# -----------------------------

img_filtered = img.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(
            img_filtered,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

# -----------------------------
# 9. Visualization
# -----------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Grayscale")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Blurred")
plt.imshow(blur, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Binary Threshold")
plt.imshow(thresh, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.title("All Contours")
plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("Bounding Boxes")
plt.imshow(cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2, 3, 6)
plt.title("Filtered Contours (Area > 500)")
plt.imshow(cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()

"""
Summary:
- Contours represent object boundaries
- Binary images are required for contour detection
- findContours returns a list of point arrays
- drawContours visualizes object outlines
- Bounding boxes localize objects
- Area filtering removes noise
"""
