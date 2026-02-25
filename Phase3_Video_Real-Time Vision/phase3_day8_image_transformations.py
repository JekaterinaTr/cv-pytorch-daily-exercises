"""
PHASE 3 — Video & Real-Time Vision
Day 8: Image Transformations

Concepts:
- Resize images
- Rotate images
- Translate (shift)
- Affine transform
- Perspective transform
- Transformation pipelines
"""

import cv2
import numpy as np

# ----------------------------------
# 1. Load Image
# ----------------------------------

img = cv2.imread("sample.jpg")

if img is None:
    raise FileNotFoundError("Image not found")

rows, cols = img.shape[:2]

# ----------------------------------
# 2. Resize Image
# ----------------------------------

# Half size
img_small = cv2.resize(img, (cols // 2, rows // 2))

# Scale factor (1.5x)
img_large = cv2.resize(img, None, fx=1.5, fy=1.5)

cv2.imshow("Original", img)
cv2.imshow("Small", img_small)
cv2.imshow("Large", img_large)
cv2.waitKey(0)

# ----------------------------------
# 3. Rotate Image
# ----------------------------------

center = (cols // 2, rows // 2)
angle = 45
scale = 1.0

# Rotation matrix
M_rot = cv2.getRotationMatrix2D(center, angle, scale)

# Apply rotation
rotated = cv2.warpAffine(img, M_rot, (cols, rows))

cv2.imshow("Rotated 45 Degrees", rotated)
cv2.waitKey(0)

# ----------------------------------
# 4. Translate (Shift) Image
# ----------------------------------

tx, ty = 100, 50  # shift right and down

M_trans = np.float32([[1, 0, tx],
                     [0, 1, ty]])

shifted = cv2.warpAffine(img, M_trans, (cols, rows))

cv2.imshow("Translated", shifted)
cv2.waitKey(0)

# ----------------------------------
# 5. Affine Transformation
# ----------------------------------

# Three source points
pts1 = np.float32([[50, 50],
                  [200, 50],
                  [50, 200]])

# Three destination points
pts2 = np.float32([[10, 100],
                  [200, 50],
                  [100, 250]])

# Get affine matrix
M_affine = cv2.getAffineTransform(pts1, pts2)

# Apply affine transform
affine_img = cv2.warpAffine(img, M_affine, (cols, rows))

cv2.imshow("Affine Transform", affine_img)
cv2.waitKey(0)

# ----------------------------------
# 6. Perspective Transformation
# ----------------------------------

# Four source points (corners)
pts1 = np.float32([[50, 50],
                  [cols - 50, 50],
                  [50, rows - 50],
                  [cols - 50, rows - 50]])

# Four destination points (rectangle)
pts2 = np.float32([[0, 0],
                  [400, 0],
                  [0, 400],
                  [400, 400]])

# Perspective matrix
M_persp = cv2.getPerspectiveTransform(pts1, pts2)

# Warp perspective
perspective_img = cv2.warpPerspective(img, M_persp, (400, 400))

cv2.imshow("Perspective Transform", perspective_img)
cv2.waitKey(0)

# ----------------------------------
# 7. Transformation Pipeline Example
# ----------------------------------

# Resize
resized = cv2.resize(img, (cols // 2, rows // 2))

# Rotate resized image
center = (cols // 4, rows // 4)
M_rot2 = cv2.getRotationMatrix2D(center, 30, 1.0)
rotated2 = cv2.warpAffine(resized, M_rot2, (cols // 2, rows // 2))

# Translate rotated image
M_trans2 = np.float32([[1, 0, 50],
                      [0, 1, 30]])

final = cv2.warpAffine(rotated2, M_trans2, (cols // 2, rows // 2))

cv2.imshow("Pipeline Result", final)
cv2.waitKey(0)

# ----------------------------------
# 8. Cleanup
# ----------------------------------

cv2.destroyAllWindows()