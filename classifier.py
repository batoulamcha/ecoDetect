import sys
import cv2
import numpy as np
import tensorflow as tf
import os

# Set the log level to '2' to suppress informational messages and only display warnings and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Check if a model name is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python classifier.py <model_name>")
    sys.exit(1)

model_name = sys.argv[1]  # Get model name from command line argument

# Define the path to the model based on the provided model name
model_paths = {
    'MobileNetV2': '/Users/dev_lap01/Downloads/ecoDetect---thesis/models/MobileNetV2-model/MobileNetV2-model.keras',
    'VGG16': '/Users/dev_lap01/Downloads/ecoDetect---thesis/models/VGG16-model/VGG16-model.keras',
    'CustomModel': '/Users/dev_lap01/Downloads/ecoDetect---thesis/models/custom-model/custom-model.keras'
}

model_path = model_paths.get(model_name)
if not model_path:
    print(f"Model {model_name} is not supported.")
    sys.exit(1)

# Load the model
print("Attempting to load model from:", model_path)
model = tf.keras.models.load_model(model_path)

# Check if the model was loaded successfully
if not model:
    print("Failed to load model.")
    exit(1)

print("Model loaded successfully.")

# Label mapping
label_mapping = {
    0: 'battery',
    1: 'biological',
    2: 'brown-glass',
    3: 'cardboard',
    4: 'clothes',
    5: 'green-glass',
    6: 'metal',
    7: 'paper',
    8: 'plastic',
    9: 'shoes',
    10: 'trash',
    11: 'white-glass'
}

# Open the camera for real-time detection
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Define the region of interest (ROI) where we will show the color image
    height, width, _ = frame.shape
    top_left = (int(width / 4), int(height / 4))
    bottom_right = (int(width * 3 / 4), int(height * 3 / 4))

    # Convert the background to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Place the original frame in the ROI
    colored[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # Resize ROI for model prediction
    roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    resized_frame = cv2.resize(roi, (224, 224))
    normalized_frame = resized_frame / 255.0  # Normalize the frame
    input_frame = np.expand_dims(normalized_frame, axis=0)  # Reshape for the model

    # Predict
    prediction = model.predict(input_frame)
    class_id = np.argmax(prediction)
    label = label_mapping.get(class_id, "Unknown")
    confidence = np.max(prediction) * 100  # Convert to percentage

    # Display the label and confidence on the frame
    text = f"{label}: {confidence:.2f}%"
    cv2.putText(colored, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with ROI
    cv2.imshow('Real-time Detection', colored)

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
