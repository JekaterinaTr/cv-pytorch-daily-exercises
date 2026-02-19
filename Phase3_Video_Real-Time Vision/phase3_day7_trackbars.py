"""
PHASE 3 — Video & Real-Time Vision
Day 7: Trackbars & Real-Time Parameter Control

Concepts:
- Creating trackbars (sliders)
- Reading slider values in real-time
- Adjusting brightness & contrast
- Applying Gaussian blur dynamically
- Building interactive CV pipelines
"""

import cv2

# ----------------------------------
# 1. Callback function (required)
# ----------------------------------

def nothing(x):
    pass

# ----------------------------------
# 2. Open Webcam
# ----------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

# ----------------------------------
# 3. Create Window & Trackbars
# ----------------------------------

cv2.namedWindow("Live Controls")

# Brightness: 0–100
cv2.createTrackbar("Brightness", "Live Controls", 50, 100, nothing)

# Contrast: 0–100
cv2.createTrackbar("Contrast", "Live Controls", 50, 100, nothing)

# Blur intensity: 0–20
cv2.createTrackbar("Blur", "Live Controls", 0, 20, nothing)

print("Trackbars ready. Press 'q' to exit.")

# ----------------------------------
# 4. Main Loop
# ----------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Get trackbar positions
    brightness = cv2.getTrackbarPos("Brightness", "Live Controls")
    contrast = cv2.getTrackbarPos("Contrast", "Live Controls")
    blur_value = cv2.getTrackbarPos("Blur", "Live Controls")

    # ----------------------------------
    # 5. Apply Brightness & Contrast
    # ----------------------------------

    # alpha controls contrast (1.0 = normal)
    alpha = contrast / 50  # scale around 1
    beta = brightness - 50  # shift brightness

    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # ----------------------------------
    # 6. Apply Gaussian Blur
    # ----------------------------------

    if blur_value > 0:
        # Kernel size must be odd
        k = blur_value if blur_value % 2 == 1 else blur_value + 1
        adjusted = cv2.GaussianBlur(adjusted, (k, k), 0)

    # ----------------------------------
    # 7. Display Instructions
    # ----------------------------------

    cv2.putText(adjusted,
                "Day 7 - Live Controls",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    # ----------------------------------
    # 8. Show Frame
    # ----------------------------------

    cv2.imshow("Live Controls", adjusted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break


# ----------------------------------
# 9. Release Resources
# ----------------------------------

cap.release()
cv2.destroyAllWindows()

print("Resources released successfully.")
