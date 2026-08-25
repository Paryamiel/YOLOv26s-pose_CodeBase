import time
import cv2
from ultralytics import YOLO

# Load TensorRT GPU model (.engine)
model = YOLO("yolo26s-pose.engine")

# Initialize webcam with 1920x1080 resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("TensorRT webcam feed running at 1920x1080. Press 'q' or 'ESC' to exit.")

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Unable to fetch camera frame.")
        break

    # Calculate real-time FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
    prev_time = curr_time

    # Predict with TensorRT model on GPU
    results = model(frame, verbose=False)

    # Access keypoints from predictions
    for result in results:
        keypoints = result.keypoints
        if keypoints is not None:
            xy = keypoints.xy      # x and y coordinates
            xyn = keypoints.xyn    # normalized coordinates
            kpts = keypoints.data  # x, y, visibility

    # Draw annotated keypoints & skeleton on 1920x1080 frame
    annotated_frame = results[0].plot()

    # Overlay FPS counter on the video window
    cv2.putText(
        annotated_frame,
        f"FPS (TensorRT GPU): {fps:.1f}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # Display 1920x1080 stream with TensorRT GPU overlay
    cv2.imshow("YOLO Pose Detection (TensorRT GPU)", annotated_frame)

    # Exit on 'q' or ESC
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
