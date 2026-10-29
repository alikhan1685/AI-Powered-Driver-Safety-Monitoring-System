import cv2
import mediapipe as mp
model_path = "face_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera could not open!")
    exit()
timestamp_ms = 0
# Eye landmark indices
eye_indices = [
    33, 133, 159, 145, 160, 144, 158, 153,
    362, 263, 386, 374, 385, 380, 387, 373
]
try:
    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read camera frame")
                break
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )
            timestamp_ms += 33
            result = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )
            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]
                for index in eye_indices:
                    landmark = face_landmarks[index]
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(
                        frame, (x, y), 2, (0, 255, 0), -1
                    )
            cv2.imshow("Eye Landmarks Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    cap.release()
    cv2.destroyAllWindows()
