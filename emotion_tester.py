import cv2
import torch
from emotion_model import EmotionModel
from utils import preprocess_image, get_emotion_labels

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionModel()
model.load_state_dict(torch.load('emotion_model.pth'))
model.to(device)
model.eval()

# Define emotion labels
emotion_labels = get_emotion_labels()

# Load the pre-trained model and start emotion detection
def detect_emotion(image_path):
    image = cv2.imread(image_path)
    processed_image = preprocess_image(image)
    image_tensor = torch.tensor(processed_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1)
        emotion = emotion_labels[predicted_class.item()]
    
    return emotion

def live_detection():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotion = detect_emotion(frame)

        cv2.putText(frame, f'Emotion: {emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    live_detection()
