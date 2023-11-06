import cv2
from openni import openni2

# 최대 10개의 카메라를 확인

# for i in range(10):
#     cap = cv2.VideoCapture(i)
#     if cap.isOpened():
#         print(f"Camera found at index {i}")
#         cap.release()
#     else:
#         print(f"No camera at index {i}")

# Initialize the camera
# Using DirectShow to capture the video
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Set the resolution (e.g., 1920x1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Check if the camera was successfully initialized
if not cap.isOpened():
    print("Error: Could not open the camera.")
else:
    print("Camera successfully initialized.")

# Capture frames from the camera
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Display the resulting frame
    cv2.imshow('Camera', frame)

    # Press 'q' to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

print("Camera released and windows closed.")
