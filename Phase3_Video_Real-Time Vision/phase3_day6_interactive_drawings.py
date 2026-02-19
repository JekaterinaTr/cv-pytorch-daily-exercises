"""
PHASE 3 — Video & Real-Time Vision
Day 6: Mouse Events & Interactive Drawing

Concepts:
- Detect mouse clicks
- Track mouse movement
- Draw interactively on a window
- Build mini drawing tools
- Keyboard interaction for clearing and exiting
"""

import cv2
import numpy as np

# ----------------------------------
# 1. Create a blank canvas
# ----------------------------------

img = np.zeros((500, 700, 3), dtype=np.uint8)

# Global variables for drawing state
drawing = False
start_x, start_y = -1, -1

# ----------------------------------
# 2. Mouse Callback Function
# ----------------------------------

def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y

    # Left button pressed: start drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    # Mouse moving while button pressed: draw rectangle dynamically
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = img.copy()
            cv2.rectangle(temp_img, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow("Interactive Draw", temp_img)

    # Left button released: finalize rectangle
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (start_x, start_y), (x, y), (0, 255, 0), 2)

# ----------------------------------
# 3. Set up window and callback
# ----------------------------------

cv2.namedWindow("Interactive Draw")
cv2.setMouseCallback("Interactive Draw", mouse_callback)

print("Interactive drawing window open. Press 'c' to clear, 'q' to quit.")

# ----------------------------------
# 4. Main loop
# ----------------------------------

while True:
    cv2.imshow("Interactive Draw", img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        img[:] = 0  # Clear canvas
        print("Canvas cleared.")

    if key == ord('q'):
        print("Exiting...")
        break

# ----------------------------------
# 5. Release resources
# ----------------------------------

cv2.destroyAllWindows()
print("Resources released successfully.")
