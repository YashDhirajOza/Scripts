from ultralytics import YOLO
import torch
import cv2
import numpy as np
from sort import Sort

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # 'yolov8n.pt' is the small version of the model

# Initialize SORT tracker
tracker = Sort()

# Define a function to draw bounding boxes and label the cars
def draw_boxes_and_label(frame, tracked_objects):
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        label = f'Car {int(obj_id)}'
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return frame

# Load the video
video_path = 'D:\\2103099-uhd_3840_2160_30fps.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error.")
        break

    # Perform inference on the frame
    try:
        # Convert frame to RGB as YOLO expects RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame, device='cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        print(f"Error during model inference: {e}")
        break

    # Prepare detections for the tracker
    detections = []
    for result in results:
        for box in result.boxes:
            cls = box.cls.cpu().numpy()[0]
            if int(cls) == 2:  # YOLO class '2' corresponds to 'car' in COCO dataset
                xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
                detections.append([xmin, ymin, xmax, ymax, box.conf.cpu().numpy()[0]])

    # Update tracker with new detections
    detections = np.array(detections)
    tracked_objects = tracker.update(detections[:, :4])  # Use only bbox coordinates for tracking

    # Draw boxes and label the cars
    labeled_frame = draw_boxes_and_label(frame.copy(), tracked_objects)

    # Write the frame to the output video
    out.write(labeled_frame)

    # Display the frame (optional)
    resized_frame = cv2.resize(labeled_frame, (960, 540))  # Resize frame for display
    cv2.imshow('Frame', resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output video saved as 'output_video.mp4'.")
