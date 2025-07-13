import cv2

# Start webcam capture (0 = default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Show the frame in a window
    cv2.imshow("Webcam Feed", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
