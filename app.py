import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("./runs/detect/train/weights/last.pt")
# model = YOLO("yolov8n.pt")

reader = easyocr.Reader(['en'], gpu=True)
video_path = "basketball_game.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define output video settings
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            confidence = float(box.conf[0])  # Confidence score
            class_id = int(box.cls[0])  # Class ID

            if confidence > 0.7:  # Confidence threshold
                # Draw bounding box around detected player
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Extract **only the jersey area** (upper 40% of the bounding box)
                jersey_y1 = y1  # Keep top boundary
                jersey_y2 = y1 + int(1 * (y2 - y1))  # Crop only the upper 40% 

                roi = frame[jersey_y1:jersey_y2, x1:x2]

                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray_roi = cv2.equalizeHist(gray_roi)
                gray_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, 11, 2)

                # Resize the image to improve OCR readability
                gray_roi = cv2.resize(gray_roi, (gray_roi.shape[1] * 2, gray_roi.shape[0] * 2))

                # Debugging: Display extracted jersey area
                # cv2.imshow("Jersey Region", gray_roi)
                # cv2.waitKey(1)

                # Use EasyOCR to detect jersey numbers
                ocr_result = reader.readtext(gray_roi)

                for (bbox, text, conf) in ocr_result:
                    # if conf > 0.5:  # Only use high-confidence results
                    print(f"Detected Jersey Number: {text} (Confidence: {conf})")

                    # Draw the detected number above the player
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)

    cv2.imshow("Player Detection & Jersey Number Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
