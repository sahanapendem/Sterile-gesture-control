from flask import Flask, render_template, Response, jsonify
import cv2
import time
import os
import numpy as np

app = Flask(__name__)

# Detect if running on Render
IS_RENDER = os.environ.get("RENDER") == "true"

# ---------------- SAFE MEDIAPIPE SETUP ----------------
mp_hands = None
hands = None
mp_draw = None

if not IS_RENDER:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7,
                           min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

# ---------------- CAMERA ----------------
cap = None
if not IS_RENDER:
    cap = cv2.VideoCapture(0)

# ---------------- DEVICE STATE ----------------
device_state = {
    "light": "OFF ❌",
    "fan": "OFF ❌",
    "ac": "OFF ❌",
    "brightness": 50,
    "gesture": "No Gesture"
}

prev_x = 0
last_action_time = 0


# ---------------- FINGER DETECTION ----------------
def get_finger_states(hand_landmarks):
    fingers = []

    fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
    fingers.append(1 if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y else 0)
    fingers.append(1 if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y else 0)
    fingers.append(1 if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y else 0)
    fingers.append(1 if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y else 0)

    return fingers


# ---------------- SWIPE DETECTION ----------------
def detect_swipe(hand_landmarks):
    global prev_x
    current_x = hand_landmarks.landmark[0].x
    direction = ""

    if prev_x != 0:
        if current_x - prev_x > 0.15:
            direction = "RIGHT"
        elif prev_x - current_x > 0.15:
            direction = "LEFT"

    prev_x = current_x
    return direction


# ---------------- FRAME GENERATOR ----------------
def generate_frames():
    global last_action_time

    # If running on Render → show dummy frame
    if IS_RENDER or cap is None or not cap.isOpened():
        while True:
            frame = 255 * np.ones((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera not available on server",
                        (50, 240), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        current_time = time.time()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                finger_state = get_finger_states(hand_landmarks)
                swipe = detect_swipe(hand_landmarks)

                gesture = tuple(finger_state)

                if current_time - last_action_time > 1:

                    if gesture == (0, 1, 0, 0, 0):
                        device_state["brightness"] = max(0, device_state["brightness"] - 10)
                        device_state["gesture"] = "Brightness DOWN"

                    elif gesture == (0, 1, 1, 0, 0):
                        device_state["brightness"] = min(100, device_state["brightness"] + 10)
                        device_state["gesture"] = "Brightness UP"

                    elif gesture == (0, 1, 1, 1, 0):
                        device_state["ac"] = "ON ❄️"
                        device_state["gesture"] = "AC ON ❄️"

                    elif gesture == (1, 1, 1, 1, 1):
                        device_state["light"] = "ON 💡"
                        device_state["gesture"] = "Light ON"

                    elif gesture == (0, 0, 0, 0, 0):
                        device_state["light"] = "OFF ❌"
                        device_state["gesture"] = "Light OFF"

                    else:
                        device_state["gesture"] = "❌ Invalid Gesture"

                    last_action_time = current_time

        else:
            device_state["gesture"] = "No Hand Detected"

        cv2.putText(frame, device_state["gesture"], (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ---------------- ROUTES ----------------
@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/dashboard')
def dashboard():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    return jsonify(device_state)


if __name__ == "__main__":
    app.run(debug=True)