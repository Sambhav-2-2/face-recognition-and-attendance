import cv2

cap = cv2.VideoCapture(0)  # 0 means default webcam

while True:
    ret, frame = cap.read()
    cv2.imshow("Webcam", frame)  # Show frame in window

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # press 'q' to quit

cap.release()
cv2.destroyAllWindows()
