from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import time

app = Flask(__name__)

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Camera
cap = cv2.VideoCapture(0)

# Device state
device_state = {
    "light": "OFF ❌",
    "fan": "OFF ❌",
    "ac": "OFF ❌",
    "brightness": 50,
    "gesture": ""
}

prev_x = 0
last_action_time = 0

def count_fingers(hand_landmarks):
    tips = [8, 12, 16, 20]
    fingers = 0
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers += 1
    return fingers

def detect_thumb(hand_landmarks):
    return hand_landmarks.landmark[4].y < hand_landmarks.landmark[2].y

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

def generate_frames():
    global last_action_time

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

                fingers = count_fingers(hand_landmarks)
                swipe = detect_swipe(hand_landmarks)

                if current_time - last_action_time > 1:

                    if fingers >= 4:
                        device_state["light"] = "ON"
                        device_state["gesture"] = "Light ON"

                    elif fingers == 0:
                        device_state["light"] = "OFF"
                        device_state["gesture"] = "Light OFF"

                    elif fingers == 2:
                        device_state["brightness"] = min(100, device_state["brightness"] + 10)
                        device_state["gesture"] = "Brightness UP"

                    elif fingers == 1:
                        device_state["brightness"] = max(0, device_state["brightness"] - 10)
                        device_state["gesture"] = "Brightness DOWN"

                    elif fingers == 3:
                        device_state["ac"] = "ON"
                        device_state["gesture"] = "AC ON"

                    # Fan control
                    if detect_thumb(hand_landmarks):
                        device_state["fan"] = "ON"
                    else:
                        device_state["fan"] = "OFF"

                    # Swipe
                    if swipe == "RIGHT":
                        device_state["gesture"] = "Next Mode ➡️"
                    elif swipe == "LEFT":
                        device_state["gesture"] = "Previous Mode ⬅️"

                    last_action_time = current_time

        cv2.putText(frame, device_state["gesture"], (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ROUTES
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
