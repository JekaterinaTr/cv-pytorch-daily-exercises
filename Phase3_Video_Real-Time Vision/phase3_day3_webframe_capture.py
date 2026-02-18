"""
PHASE 3 — Video & Real-Time Vision
Day 3: Webcam Capture (Live Camera Feed)

Concepts:
- Accessing webcam using OpenCV
- Reading live frames
- Real-time display
- Applying transformations (grayscale, flip)
- Handling exit keys
- Releasing hardware resources properly
"""

import cv2

# ----------------------------------
# 1. Open Webcam
# ----------------------------------

# 0 = default webcam (change to 1 if you have multiple cameras)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Webcam opened successfully!")
print("Press 'q' to exit.")

# ----------------------------------
# 2. Live Webcam Loop
# ----------------------------------

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # ----------------------------------
    # 3. Apply Transformations
    # ----------------------------------

    # Mirror effect (horizontal flip)
    flipped = cv2.flip(frame, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)

    # ----------------------------------
    # 4. Display Frame
    # ----------------------------------

    cv2.imshow("Live Webcam - Mirror + Gray", gray)

    # ----------------------------------
    # 5. Exit Condition
    # ----------------------------------

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting webcam...")
        break


# ----------------------------------
# 6. Release Resources
# ----------------------------------

cap.release()
cv2.destroyAllWindows()

print("Webcam released successfully.")
