import cv2
import mediapipe as mp
import numpy as np
import argparse

# Mediapipe initialization
mp_face_detection = mp.solutions.face_detection

# Function to detect and highlight faces using Mediapipe with NMS
def highlightFacesWithNMS(frame, face_detections, nms_threshold=0.3):
    frameHeight, frameWidth = frame.shape[:2]
    boxes = []
    confidences = []

    # Extract bounding boxes and confidences
    for detection in face_detections:
        bboxC = detection.location_data.relative_bounding_box
        confidence = detection.score[0]
        x = int(bboxC.xmin * frameWidth)
        y = int(bboxC.ymin * frameHeight)
        w = int(bboxC.width * frameWidth)
        h = int(bboxC.height * frameHeight)
        boxes.append([x, y, x + w, y + h])
        confidences.append(confidence)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, nms_threshold)

    faceBoxes = []
    for i in indices.flatten():
        x1, y1, x2, y2 = boxes[i]
        faceBoxes.append([x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame, faceBoxes


# Argument parser for image or webcam feed
parser = argparse.ArgumentParser()
parser.add_argument('--image', help='Path to input image (optional). If not provided, webcam feed will be used.')
args = parser.parse_args()

# Paths to age and gender models
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Mean values and labels for age and gender
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load models
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Initialize Mediapipe face detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize video capture
video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20

# Real-time loop
while True:
    hasFrame, frame = video.read()
    if not hasFrame:
        print("No frame captured. Exiting...")
        break

    # Convert the image to RGB (Mediapipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    # Check for face detections
    faceBoxes = []
    if results.detections:
        frame, faceBoxes = highlightFacesWithNMS(frame, results.detections)

    if not faceBoxes:
        print("No face detected")
        cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for faceBox in faceBoxes:
        # Extract face region
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                     max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

        # Skip small or invalid regions
        if face.shape[0] < 10 or face.shape[1] < 10:
            print("Face region too small, skipping...")
            continue

        # Predict Gender
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        genderConfidence = max(genderPreds[0])

        # Predict Age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        agePreds = np.exp(agePreds) / np.sum(np.exp(agePreds))  # Softmax normalization
        age = ageList[agePreds[0].argmax()]
        ageConfidence = max(agePreds[0])

        # Annotate the frame
        cv2.putText(frame, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        print(f"Gender: {gender} (Confidence: {genderConfidence:.2f}), Age: {age} (Confidence: {ageConfidence:.2f})")

    # Display the real-time video feed
    cv2.imshow("Real-Time Age and Gender Detection with Mediapipe", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
video.release()
cv2.destroyAllWindows()