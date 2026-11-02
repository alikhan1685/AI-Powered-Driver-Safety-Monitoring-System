import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
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
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced_l = clahe.apply(l_channel)
    # Step 4: Merge channels
    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )
    # Step 5: Convert back to BGR
    enhanced_frame = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )
    return enhanced_frame
# Load trained CNN model
model = tf.keras.models.load_model("eye_state_cnn.keras")
print("CNN Model Loaded:", model.input_shape)
# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)
# Eye landmark indices
left_eye = [33, 133, 159, 145, 160, 144, 158, 153]
right_eye = [362, 263, 386, 374, 385, 380, 387, 373]
# Extract and preprocess eye
def preprocess_eye(frame, landmarks, indices):
    height, width = frame.shape[:2]
    points = []
    for index in indices:
        landmark = landmarks[index]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))
    points = np.array(points)
    x_min = max(0, np.min(points[:, 0]) - 5)
    x_max = min(width, np.max(points[:, 0]) + 5)
    y_min = max(0, np.min(points[:, 1]) - 5)
    y_max = min(height, np.max(points[:, 1]) + 5)
    eye = frame[y_min:y_max, x_min:x_max]
    if eye.size == 0:
        return None, None
    # Convert to grayscale
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    # Resize to CNN input size
    eye = cv2.resize(eye, (64, 64))
    # Normalize pixel values
    eye = eye.astype("float32") / 255.0
    # Prepare CNN input
    eye = np.expand_dims(eye, axis=-1)
    eye = np.expand_dims(eye, axis=0)
    return eye, (x_min, y_min, x_max, y_max)
# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera could not open!")
    exit()
timestamp_ms = 0
try:
    with FaceLandmarker.create_from_options(options) as landmarker:
        mode = "DAY"
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read camera frame")
                break
            # Apply Night Mode enhancement
                if mode == "NIGHT":
                    frame = enhance_night_frame(frame)
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )
            timestamp_ms += 33
            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )
            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]
                eye_results = []
                for eye_name, indices in [
                    ("Left", left_eye),
                    ("Right", right_eye)
                ]:
                    eye_image, box = preprocess_eye(
                        frame,
                        face_landmarks,
                        indices
                    )
                    if eye_image is None:
                        continue
                    # Display actual CNN input
                        eye_display = (eye_image[0, :, :, 0] * 255).astype("uint8")
                        cv2.imshow(
                         f"{eye_name} CNN Input",
                         cv2.resize(eye_display, (256, 256))
                        )
                    # CNN prediction
                    prediction = model.predict(
                        eye_image,
                        verbose=0
                    )[0][0]
                    print(f"{eye_name} Prediction: {prediction:.4f}")
                    if prediction >= 0.5:
                        label = "Open"
                    else:
                        label = "Closed"
                    eye_results.append(label)
                    x1, y1, x2, y2 = box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"{eye_name}: {label}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                if eye_results:
                    if all(label == "Closed" for label in eye_results):
                        status = "Eyes Closed"
                    else:
                        status = "Eyes Open"
                    cv2.putText(
                        frame,
                        status,
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255) if status == "Eyes Closed"
                        else (0, 255, 0),
                        2
                    )
            else:
                cv2.putText(
                    frame,
                    "Face Not Detected",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
          # Display current mode
            cv2.putText(
            frame,
            f"Mode: {mode} | D: Day | N: Night | Q: Quit",
             (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
            )
            cv2.imshow(
                "Driver Drowsiness Detection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("d"):
                mode = "DAY"
                print("Day Mode Activated")

            elif key == ord("n"):
                mode = "NIGHT"
                print("Night Mode Activated")

            elif key == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()