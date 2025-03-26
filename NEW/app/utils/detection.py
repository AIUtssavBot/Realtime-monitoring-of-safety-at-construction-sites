import cv2
import numpy as np
import time
import os
from collections import defaultdict

class SafetyGearDetector:
    def __init__(self, model_path=None):
        """
        Initialize the safety gear detector.
        Can be adapted to use pretrained models for PPE detection.
        """
        self.model_path = model_path
        self.model = None
        self.classes = ["helmet", "vest", "mask", "gloves", "boots"]
        self.violation_time = defaultdict(dict)  # Track violation time {camera_id: {worker_id: start_time}}
        self.alert_threshold = 5  # Alert after 5 seconds of violation
        self.load_model()
        
    def load_model(self):
        """
        Load the safety gear detection model.
        For now, we'll use a placeholder function.
        In production, you would load a trained model here.
        """
        # Placeholder for model loading
        print("Safety gear detection model loaded (placeholder)")
        # In a real implementation, you would load a TensorFlow/PyTorch model:
        # if os.path.exists(self.model_path):
        #     self.model = load_model_here(self.model_path)
        
    def detect(self, frame, camera_id):
        """
        Detect safety gear violations in the frame.
        Returns the processed frame and a list of violations.
        """
        # For demonstration, we'll use a simple person detection with OpenCV's HOG detector
        # and randomly simulate safety gear violations
        # In a real implementation, you would use a proper PPE detection model
        
        # Convert to grayscale for HOG detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect people using HOG
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(4, 4), scale=1.05)
        
        current_time = time.time()
        violations = []
        
        # Process each detected person
        for i, (x, y, w, h) in enumerate(boxes):
            worker_id = f"worker_{camera_id}_{i}"
            
            # Simulate random safety gear detection
            # In a real implementation, this would be the result of your model prediction
            missing_gear = []
            if np.random.random() < 0.3:  # 30% chance of missing helmet
                missing_gear.append("helmet")
            if np.random.random() < 0.2:  # 20% chance of missing vest
                missing_gear.append("vest")
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for compliant
            
            if missing_gear:
                color = (0, 0, 255)  # Red for violation
                
                # Track violation time
                if worker_id not in self.violation_time[camera_id]:
                    self.violation_time[camera_id][worker_id] = current_time
                
                # Check if violation has persisted past threshold
                violation_duration = current_time - self.violation_time[camera_id][worker_id]
                if violation_duration >= self.alert_threshold:
                    violations.append({
                        "worker_id": worker_id,
                        "missing_gear": missing_gear,
                        "duration": round(violation_duration, 1)
                    })
                    
                    # Add violation duration text
                    cv2.putText(frame, f"Violation: {round(violation_duration, 1)}s", 
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                # Reset violation time if compliant
                if worker_id in self.violation_time[camera_id]:
                    del self.violation_time[camera_id][worker_id]
            
            # Draw box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"Worker {i+1}", (x, y + h + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw missing equipment text
            if missing_gear:
                cv2.putText(frame, f"Missing: {', '.join(missing_gear)}", 
                            (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame, violations


class ProximityDetector:
    def __init__(self):
        """
        Initialize the proximity detector for workers and heavy machinery.
        """
        self.proximity_threshold = 100  # Distance threshold in pixels
        self.machinery_positions = {}  # Track machinery positions {camera_id: [(x, y, w, h), ...]}
        self.alert_history = defaultdict(dict)  # Track alerts {camera_id: {worker_id: last_alert_time}}
        self.alert_cooldown = 10  # Cooldown period between alerts (seconds)
        
    def detect(self, frame, camera_id):
        """
        Detect proximity between workers and machinery.
        Returns the processed frame and a list of proximity alerts.
        """
        # For demonstration, we'll simulate machinery with random boxes
        # In a real implementation, you would use object detection models to identify machinery
        
        height, width = frame.shape[:2]
        alerts = []
        
        # Simulate machinery detection (in reality, you'd use an object detection model)
        # For demo, we'll place 1-2 "machines" at semi-fixed positions
        machinery = []
        
        # Create/update machinery for this camera if needed
        if camera_id not in self.machinery_positions:
            # Generate random positions for machinery
            num_machines = np.random.randint(1, 3)
            self.machinery_positions[camera_id] = []
            
            for _ in range(num_machines):
                mx = np.random.randint(0, width - 100)
                my = np.random.randint(0, height - 100)
                mw = np.random.randint(80, 150)
                mh = np.random.randint(80, 150)
                self.machinery_positions[camera_id].append((mx, my, mw, mh))
        
        # Get the machinery positions for this camera
        machinery = self.machinery_positions[camera_id]
        
        # Draw machinery boxes
        for i, (mx, my, mw, mh) in enumerate(machinery):
            cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (255, 165, 0), 2)
            cv2.putText(frame, f"Machinery {i+1}", (mx, my - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        # Detect people using HOG detector
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(4, 4), scale=1.05)
        
        current_time = time.time()
        
        # Check proximity for each person against each machinery
        for i, (x, y, w, h) in enumerate(boxes):
            worker_id = f"worker_{camera_id}_{i}"
            worker_center = (x + w//2, y + h//2)
            
            # Draw worker box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Worker {i+1}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Check proximity to each machinery
            for j, (mx, my, mw, mh) in enumerate(machinery):
                machine_center = (mx + mw//2, my + mh//2)
                
                # Calculate Euclidean distance between centers
                distance = np.sqrt((worker_center[0] - machine_center[0])**2 + 
                                  (worker_center[1] - machine_center[1])**2)
                
                # Check if distance is below threshold
                if distance < self.proximity_threshold:
                    # Check cooldown
                    key = f"{worker_id}_machine_{j}"
                    last_alert = self.alert_history[camera_id].get(key, 0)
                    
                    if current_time - last_alert >= self.alert_cooldown:
                        # Draw proximity warning line
                        cv2.line(frame, worker_center, machine_center, (0, 0, 255), 2)
                        
                        # Add warning text
                        mid_point = ((worker_center[0] + machine_center[0])//2, 
                                     (worker_center[1] + machine_center[1])//2)
                        cv2.putText(frame, f"PROXIMITY WARNING!", mid_point, 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Create alert
                        alerts.append({
                            "worker_id": worker_id,
                            "machine_id": f"machine_{camera_id}_{j}",
                            "distance": round(distance, 2)
                        })
                        
                        # Update alert history
                        self.alert_history[camera_id][key] = current_time
        
        return frame, alerts 