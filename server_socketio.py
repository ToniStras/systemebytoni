from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import threading

app = Flask(__name__)
socketio = SocketIO(app)

# Initialisation de Mediapipe pour la détection des mains
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Ouvre la caméra
cap = cv2.VideoCapture(0)

def video_stream():
    while True:
        success, frame = cap.read()
        if not success:
            continue

        # Convertir l'image en RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Détection des mains
        results = hands.process(rgb_frame)

        # Dessiner les contours des mains détectées
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
                x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
                y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])
                y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        # Encodez le frame en JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        socketio.emit('video_frame', {'image': frame.hex()}, broadcast=True)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    threading.Thread(target=video_stream).start()
    socketio.run(app, host='0.0.0.0', port=5000)
