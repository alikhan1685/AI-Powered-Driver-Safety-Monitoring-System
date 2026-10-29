import cv2
import mediapipe as mp
timestamp_ms = 0
model_path = "blaze_face_short_range.tflite"
options = mp.tasks.vision.FaceDetectorOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO
)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera could not open!")
    exit()
print("Camera opened successfully!")
with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
    while True:
        success, frame = cap.read()
        if not success:
            print("Camera could not open!")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )
        timestamp_ms += 33
        result = detector.detect_for_video(mp_image, timestamp_ms)
        for detection in result.detections:
            bbox = detection.bounding_box
            x = bbox.origin_x
            y = bbox.origin_y
            w = bbox.width
            h = bbox.height
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )
        cv2.imshow("Driver Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()
