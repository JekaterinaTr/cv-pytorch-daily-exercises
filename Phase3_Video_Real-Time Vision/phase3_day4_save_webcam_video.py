"""
PHASE 3 — Video & Real-Time Vision
Day 4: Save Webcam Video Using VideoWriter

Concepts:
- Capturing webcam frames
- Creating VideoWriter object
- Choosing codec
- Writing frames to file
- Recording processed frames
- Proper resource cleanup
"""

import cv2

# ----------------------------------
# 1. Open Webcam
# ----------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Webcam opened successfully!")

# ----------------------------------
# 2. Get Webcam Properties
# ----------------------------------

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# FPS can be unreliable from webcam, so set manually
fps = 20

print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")

# ----------------------------------
# 3. Create VideoWriter Object
# ----------------------------------

# Choose codec (common options: 'XVID', 'MJPG', 'mp4v')
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter(
    "recorded_output.avi",  # Output file name
    fourcc,                 # Codec
    fps,                    # Frames per second
    (width, height)         # Frame size (must match!)
)

print("VideoWriter initialized successfully!")
print("Recording... Press 'q' to stop.")

# ----------------------------------
# 4. Record Webcam Frames
# ----------------------------------

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Optional processing:
    # Mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to grayscale (optional)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale back to BGR (VideoWriter expects 3 channels)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Write processed frame to file
    out.write(gray_bgr)

    # Display what is being recorded
    cv2.imshow("Recording (Mirror + Gray)", gray_bgr)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Recording stopped by user.")
        break


# ----------------------------------
# 5. Release Resources
# ----------------------------------

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved successfully as 'recorded_output.avi'")
