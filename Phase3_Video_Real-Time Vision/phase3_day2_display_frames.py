"""
PHASE 3 — Video & Real-Time Vision
Day 2: Display Video Frames

Concepts:
- Reading frames in a loop
- Displaying video in real-time using OpenCV
- Controlling playback speed
- Handling keyboard input to stop video
- Proper cleanup of resources
"""

import cv2
import matplotlib.pyplot as plt

# ----------------------------------
# 1. Load Video
# ----------------------------------
video_path = "sample_video.mp4"  # Replace with your file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

print("Video loaded successfully!")

# ----------------------------------
# 2. Display Video in Real-Time (OpenCV)
# ----------------------------------

print("Press 'q' to stop playback.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video reached.")
        break

    # Show frame
    cv2.imshow("Video Playback", frame)

    # Wait 30 ms between frames (controls playback speed)
    # If 'q' is pressed, exit loop
    if cv2.waitKey(30) & 0xFF == ord('q'):
        print("Playback stopped by user.")
        break


# ----------------------------------
# 3. Optional: Display First Frame with Matplotlib
# ----------------------------------

# Reset video to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

ret, frame = cap.read()
if ret:
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("First Frame (Matplotlib)")
    plt.axis("off")
    plt.show()


# ----------------------------------
# 4. Release Resources
# ----------------------------------

cap.release()
cv2.destroyAllWindows()
print("Resources released successfully.")
