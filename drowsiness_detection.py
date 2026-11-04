import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
def enhance_night_frame(frame):
    # Step 1: Gamma correction to increase brightness
    gamma = 0.6
    lookup_table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).astype("uint8")
    bright_frame = cv2.LUT(frame, lookup_table)
    # Step 2: Convert BGR to LAB
    lab = cv2.cvtColor(bright_frame, cv2.COLOR_BGR2LAB)
    # Step 3: Apply CLAHE to improve local contrast
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    # Step 4: Merge channels
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    # Step 5: Convert back to BGR
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_frame


# --- LOAD BOTH MODELS ---
try:
    eye_model = tf.keras.models.load_model("eye_state_cnn.keras")
    drowsy_model = tf.keras.models.load_model("drowsiness_cnn.keras")
    print("Both models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# --- MEDIAPIPE SETUP ---
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

# Eye landmark indices
left_eye = [33, 133, 159, 145, 160, 144, 158, 153]
right_eye = [362, 263, 386, 374, 385, 380, 387, 373]
EAR_LEFT = [33, 160, 158, 133, 153, 144]
EAR_RIGHT = [362, 385, 387, 263, 373, 380]

# History for smooth predictions
left_history = deque(maxlen=6)
right_history = deque(maxlen=6)
drowsy_history = deque(maxlen=6)

def calculate_ear(landmarks, indices, width, height):
    pts = [(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in indices]
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    h = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if h == 0: return 0.3
    return (v1 + v2) / (2.0 * h)

def preprocess_eye(frame, landmarks, indices):
    h, w = frame.shape[:2]
    pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices])
    x_min = max(0, np.min(pts[:, 0]) - 3)
    x_max = min(w, np.max(pts[:, 0]) + 3)
    y_min = max(0, np.min(pts[:, 1]) - 3)
    y_max = min(h, np.max(pts[:, 1]) + 3)
    eye = frame[y_min:y_max, x_min:x_max]
    if eye.size == 0:
        return None, None
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    eye = cv2.resize(eye, (64, 64))
    eye = eye.astype("float32") / 255.0
    eye = np.expand_dims(eye, axis=-1)
    eye = np.expand_dims(eye, axis=0)
    return eye, (x_min, y_min, x_max, y_max)

def preprocess_face(frame, landmarks):
    h, w = frame.shape[:2]
    pts = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks])
    x_min = max(0, np.min(pts[:, 0]) - 20)
    x_max = min(w, np.max(pts[:, 0]) + 20)
    y_min = max(0, np.min(pts[:, 1]) - 20)
    y_max = min(h, np.max(pts[:, 1]) + 20)
    face = frame[y_min:y_max, x_min:x_max]
    if face.size == 0:
        return None
    face = cv2.resize(face, (96, 96))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera could not open!")
    exit()

timestamp_ms = 0

try:
    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            frame = enhance_night_frame(frame)

            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms += 33
            result = landmarker.detect_for_video(mp_img, timestamp_ms)

            eye_status = "Open"
            face_status = "Awake"
            ear_avg = 0.3

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                h, w = frame.shape[:2]

                # 1. Eye EAR & CNN
                ear_left = calculate_ear(lm, EAR_LEFT, w, h)
                ear_right = calculate_ear(lm, EAR_RIGHT, w, h)
                ear_avg = (ear_left + ear_right) / 2.0

                eye_labels = []
                for name, idx, ear_val in [("L", left_eye, ear_left), ("R", right_eye, ear_right)]:
                    eye_img, box = preprocess_eye(frame, lm, idx)
                    if eye_img is None:
                        continue
                    pred = eye_model.predict(eye_img, verbose=0)[0][0]
                    
                    # Hybrid Logic (EAR + CNN)
                    if ear_val < 0.18:
                        label = "Closed"
                    elif ear_val > 0.25:
                        label = "Open"
                    else:
                        label = "Open" if pred >= 0.6 else "Closed"
                        
                    eye_labels.append(label)
                    if box:
                        x1, y1, x2, y2 = box
                        c = (0, 255, 0) if label == "Open" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)

                eye_status = "Closed" if all(l == "Closed" for l in eye_labels) else "Open"

                # 2. Face-level CNN (Drowsiness)
                face_img = preprocess_face(frame, lm)
                if face_img is not None:
                    d_pred = drowsy_model.predict(face_img, verbose=0)[0][0]
                    drowsy_history.append(1 if d_pred >= 0.5 else 0)
                    smoothed = np.mean(drowsy_history)
                    face_status = "Sleepy" if smoothed > 0.5 else "Awake"

            # 3. Final Combined Alert Logic
            if eye_status == "Closed" or face_status == "Sleepy":
                alert = "DROWSINESS ALERT"
                color = (0, 0, 255)
            else:
                alert = "Driver Active"
                color = (0, 255, 0)

            # Display info on frame
            cv2.putText(frame, f"Eyes: {eye_status}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Face: {face_status}", (30, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"EAR: {ear_avg:.2f}", (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, alert, (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

            cv2.imshow("Driver Safety Monitoring System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    cap.release()
    cv2.destroyAllWindows()