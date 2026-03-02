"""
PHASE 3 — Video & Real-Time Vision
Day 9: Motion Tracking (Contours & Bounding Boxes)

Concepts:
- Frame differencing
- Thresholding
- Contours
- Bounding boxes
- Motion detection pipeline
"""

import cv2

# ----------------------------------
# 1. Open Webcam
# ----------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

# ----------------------------------
# 2. Capture First Frame (Reference)
# ----------------------------------

ret, prev_frame = cap.read()

if not ret:
    raise RuntimeError("Cannot read first frame")

# Convert to grayscale and blur (noise reduction)
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

print("Motion tracking started. Press 'q' to exit.")

# ----------------------------------
# 3. Main Loop
# ----------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # ----------------------------------
    # 4. Frame Difference
    # ----------------------------------

    diff = cv2.absdiff(prev_gray, gray)

    # Threshold to binary image
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # ----------------------------------
    # 5. Find Contours
    # ----------------------------------

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # ----------------------------------
    # 6. Draw Bounding Boxes
    # ----------------------------------

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Ignore small movements (noise)
        if area < 1000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # ----------------------------------
    # 7. Display Result
    # ----------------------------------

    cv2.imshow("Motion Tracking", frame)

    # Update reference frame
    prev_gray = gray

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# ----------------------------------
# 8. Cleanup
# ----------------------------------

cap.release()
cv2.destroyAllWindows()

print("Resources released.")