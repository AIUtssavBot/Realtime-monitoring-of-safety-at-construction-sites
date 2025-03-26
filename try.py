import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
import threading
import time

# Load the pretrained model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Global state for recording
recording = False

def detect_faces(frame):
    """
    Detect faces in a frame using Hugging Face DETR model and draw bounding boxes.
    """
    image = Image.fromarray(frame)  # Convert to PIL image
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Process results
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    boxes = results["boxes"].detach().numpy()

    for box in boxes:
        x, y, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
    return frame

def video_stream(source, placeholders, output_files):
    """
    Capture video from a source, apply face detection, and display it.
    Saves recordings for each placeholder.
    """
    global recording

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error(f"Unable to open camera source: {source}")
        return

    # Define video writers
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    outs = [cv2.VideoWriter(file, fourcc, 20.0, (frame_width, frame_height)) for file in output_files]

    while recording:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to retrieve video from the camera.")
            break

        # Detect faces
        frame_with_boxes = detect_faces(frame)
        frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)

        # Save and display the same frame in all placeholders
        for i, placeholder in enumerate(placeholders):
            outs[i].write(frame_with_boxes)  # Record video
            placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()
    for out in outs:
        out.release()

def main():
    global recording

    st.title("Single Camera with Face Detection and Recording")
    st.text("Click the button below to start/stop recording.")

    # Camera source
    laptop_camera = 0  # Default laptop camera

    # Create placeholders for the four screens
    placeholders = [st.empty() for _ in range(4)]

    # Define output files for recording
    output_files = ["output1.avi", "output2.avi", "output3.avi", "output4.avi"]

    # Buttons to control recording
    if st.button("Start Recording") and not recording:
        recording = True

        # Start video stream thread
        threading.Thread(target=video_stream, args=(laptop_camera, placeholders, output_files)).start()

    if st.button("Stop Recording") and recording:
        recording = False
        time.sleep(1)  # Allow thread to terminate gracefully

if __name__ == "__main__":
    main()
