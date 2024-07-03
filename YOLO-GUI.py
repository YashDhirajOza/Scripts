import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import tempfile
import torch

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

# Streamlit UI
st.title("Car Detection and Tracking with YOLOv8")
st.write("Upload a video file to detect and track cars.")

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Load the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error: Could not open video.")
    else:
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = 'output_video.mp4'
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        # Processing video frames
        progress_bar = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference on the frame
            try:
                # Convert frame to RGB as YOLO expects RGB images
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(rgb_frame, device='cuda' if torch.cuda.is_available() else 'cpu')
            except Exception as e:
                st.error(f"Error during model inference: {e}")
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
            st.image(cv2.cvtColor(labeled_frame, cv2.COLOR_BGR2RGB), channels="RGB")

            current_frame += 1
            progress_bar.progress(current_frame / frame_count)

        # Release everything if job is finished
        cap.release()
        out.release()
        
        st.success("Processing complete. Output video saved.")
        st.video(out_path)
