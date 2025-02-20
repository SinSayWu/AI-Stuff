import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Set up webcam capture
cap = cv2.VideoCapture(0)

# Define VideoWriter to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 files
output_filename = input("File name without extension: ") + ".mp4"
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Convert the frame to RGB (MediaPipe uses RGB format)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        # Create a blank image (black background)
        blank_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        # Draw pose landmarks on the blank image
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                blank_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

        # Display the blank image with the stick figure
        cv2.imshow('Stick Figure - Live', blank_image)

        # Write the frame to the output video file
        out.write(blank_image)

        # Exit when 'Esc' key is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved as {output_filename}")
