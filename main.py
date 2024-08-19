import cv2
import numpy as np
import tensorflow as tf
import kagglehub

# Download the latest model version
path = kagglehub.model_download("tensorflow/ssd-mobilenet-v2/tensorFlow2/ssd-mobilenet-v2")
print("Path to model files:", path)

# Load the pre-trained model from the downloaded path
model = tf.saved_model.load(path)
model_inference = model.signatures['serving_default']

# Load the labels (e.g., COCO dataset labels)
try:
    with open('coco-labels-paper.txt', 'r') as f:
        labels = f.read().splitlines()
except FileNotFoundError:
    print("Label file not found.")
    labels = []

def detect_objects(frame):
    # Preprocess the frame: resize to 300x300 and convert to uint8
    input_frame = cv2.resize(frame, (300, 300))
    input_frame = input_frame.astype(np.uint8)  # Convert to uint8
    input_tensor = tf.convert_to_tensor(input_frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Perform detection
    detections = model_inference(input_tensor)

    # Extract detection data
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_scores = detections['detection_scores'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(int)

    # Draw bounding boxes and labels
    height, width, _ = frame.shape
    for i in range(len(detection_scores)):
        if detection_scores[i] > 0.5:  # Confidence threshold
            box = detection_boxes[i] * np.array([height, width, height, width])
            (ymin, xmin, ymax, xmax) = box.astype(int)

            # Draw the bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Draw the label
            label = f"{labels[detection_classes[i]]}: {detection_scores[i]:.2f}" if labels else f"Class {detection_classes[i]}: {detection_scores[i]:.2f}"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Access the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the frame
        frame = detect_objects(frame)

        # Display the resulting frame
        cv2.imshow('Object Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
