"""
PHASE 3 — Video & Real-Time Vision
Day 5: Drawing on Live Video (HUD System)

Concepts:
- Drawing shapes (rectangle, circle, line)
- Adding text overlays
- Calculating FPS
- Creating a simple HUD (Head-Up Display)
- Real-time frame manipulation
"""

import cv2
import time

# ----------------------------------
# 1. Open Webcam
# ----------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Webcam opened successfully!")
print("Press 'q' to exit.")

# ----------------------------------
# 2. Initialize FPS Variables
# ----------------------------------

prev_time = 0

# ----------------------------------
# 3. Live Webcam Loop
# ----------------------------------

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Mirror effect (like selfie camera)
    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    # ----------------------------------
    # 4. Draw Shapes
    # ----------------------------------

    # Rectangle
    cv2.rectangle(frame, (50, 50), (250, 250), (0, 255, 0), 2)

    # Circle in center
    cv2.circle(frame, (width // 2, height // 2), 50, (255, 0, 0), 2)

    # Diagonal line
    cv2.line(frame, (0, 0), (width, height), (0, 0, 255), 2)

    # ----------------------------------
    # 5. FPS Calculation
    # ----------------------------------

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # ----------------------------------
    # 6. Add Text Overlays
    # ----------------------------------

    cv2.putText(frame,
                "Phase 3 - Day 5 HUD",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2)

    cv2.putText(frame,
                f"FPS: {int(fps)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    # ----------------------------------
    # 7. Display Frame
    # ----------------------------------

    cv2.imshow("Live Vision HUD", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break


# ----------------------------------
# 8. Release Resources
# ----------------------------------

cap.release()
cv2.destroyAllWindows()

print("Resources released successfully.")
