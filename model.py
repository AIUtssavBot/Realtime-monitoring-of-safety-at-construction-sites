import cv2
import numpy as np
from scipy.spatial import distance

# Define the IP camera URLs (Replace these with your actual IP addresses)
url1 = "http://192.0.0.4:8080/video"
url2 = "http://192.168.0.104:8080/video"
url3 = "http://192.168.70.67:8080/video"
url4 = "http://192.0.0.4:8080/video"

# Capture video from all four devices
cap1 = cv2.VideoCapture(url1)
cap2 = cv2.VideoCapture(0)
cap3 = cv2.VideoCapture(0)
cap4 = cv2.VideoCapture(0)

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()

    if not (ret1 and ret2 and ret3 and ret4):
        print("Failed to retrieve one or more frames")
        break

    # Resize frames to make them fit properly in a grid
    frame1 = cv2.resize(frame1, (320, 240))
    frame2 = cv2.resize(frame2, (320, 240))
    frame3 = cv2.resize(frame3, (320, 240))
    frame4 = cv2.resize(frame4, (320, 240))

    # Arrange frames in a 2x2 grid
    top_row = cv2.hconcat([frame1, frame2])
    bottom_row = cv2.hconcat([frame3, frame4])
    grid = cv2.vconcat([top_row, bottom_row])

    # Show the multi-camera stream
    cv2.imshow("Multi-Camera Stream (2x2 Grid)", grid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap1.release()
cap2.release()
cap3.release()
cap4.release()
# Constants for proximity detection
MIN_DISTANCE = 50  # minimum distance in pixels
CONFIDENCE_THRESHOLD = 0.5

# Load YOLO model for person detection
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

def detect_people(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    boxes = []
    confidences = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > CONFIDENCE_THRESHOLD and class_id == 0:  # class 0 is person
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
    
    return boxes, confidences

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()

    if not (ret1 and ret2 and ret3 and ret4):
        print("Failed to retrieve one or more frames")
        break

    # Resize frames
    frame1 = cv2.resize(frame1, (320, 240))
    frame2 = cv2.resize(frame2, (320, 240))
    frame3 = cv2.resize(frame3, (320, 240))
    frame4 = cv2.resize(frame4, (320, 240))

    # Detect people in each frame
    for frame in [frame1, frame2, frame3, frame4]:
        boxes, confidences = detect_people(frame)
        
        # Check proximity between detected people
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                dist = distance.euclidean(
                    (boxes[i][0] + boxes[i][2]/2, boxes[i][1] + boxes[i][3]/2),
                    (boxes[j][0] + boxes[j][2]/2, boxes[j][1] + boxes[j][3]/2)
                )
                
                # Draw boxes and lines
                if dist < MIN_DISTANCE:
                    cv2.rectangle(frame, (boxes[i][0], boxes[i][1]), 
                                (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]), (0, 0, 255), 2)
                    cv2.rectangle(frame, (boxes[j][0], boxes[j][1]), 
                                (boxes[j][0] + boxes[j][2], boxes[j][1] + boxes[j][3]), (0, 0, 255), 2)
                    cv2.line(frame, 
                            (int(boxes[i][0] + boxes[i][2]/2), int(boxes[i][1] + boxes[i][3]/2)),
                            (int(boxes[j][0] + boxes[j][2]/2), int(boxes[j][1] + boxes[j][3]/2)),
                            (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (boxes[i][0], boxes[i][1]), 
                                (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]), (0, 255, 0), 2)

    # Arrange frames in grid
    top_row = cv2.hconcat([frame1, frame2])
    bottom_row = cv2.hconcat([frame3, frame4])
    grid = cv2.vconcat([top_row, bottom_row])

    cv2.imshow("Multi-Camera Stream with Proximity Detection", grid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()