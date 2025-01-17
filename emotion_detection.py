import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Load model
with open('facialemotionmodel.json', 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights('facialemotionmodel.h5')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Emotion labels
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

# Initialize the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If faces are detected, process each face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the face from the frame
        face_region = gray[y:y + h, x:x + w]

        # Preprocess the face for prediction
        resized_image = cv2.resize(face_region, (48, 48))  # Resize as per model's requirement
        resized_image = resized_image.astype('float32') / 255.0  # Normalize pixel values
        resized_image = np.reshape(resized_image, (1, 48, 48, 1))  # Add batch dimension

        # Predict the emotion
        predictions = model.predict(resized_image)
        predicted_class = np.argmax(predictions)

        # Map the predicted class to emotion name
        predicted_emotion = emotion_labels[predicted_class]

        # Display the predicted emotion on the frame
        cv2.putText(frame, f"Emotion: {predicted_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame with detected faces and emotions
    cv2.imshow('Emotion Detector', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()