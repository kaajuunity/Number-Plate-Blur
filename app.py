import cv2

# Load Haar cascade for license plates
# Make sure 'model.xml' (like 'haarcascade_russian_plate_number.xml') is in the same folder
plate_cascade = cv2.CascadeClassifier('model.xml')

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Cannot access webcam.")
    exit()

print("✅ Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect license plates
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in plates:
        # Extract the plate area and blur it
        roi = frame[y:y+h, x:x+w]
        blur = cv2.GaussianBlur(roi, (51, 51), 0)
        frame[y:y+h, x:x+w] = blur

        # Optional: Draw rectangle (you can comment this)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the live result
    cv2.imshow('Real-Time Plate Blur', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
