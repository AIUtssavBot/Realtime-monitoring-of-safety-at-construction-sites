from flask import Blueprint, render_template, Response, jsonify
from app.utils.camera import CameraManager
from app.utils.detection import SafetyGearDetector, ProximityDetector
from app import socketio
import json
import cv2
import time

main = Blueprint('main', __name__)
camera_manager = CameraManager()
safety_detector = SafetyGearDetector()
proximity_detector = ProximityDetector()

# Store alerts
alerts = {
    'safety_gear': [],
    'proximity': []
}

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames(camera_id):
    camera = camera_manager.get_camera(camera_id)
    if not camera:
        return None
        
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
            
        # Run safety gear detection
        frame, safety_violations = safety_detector.detect(frame, camera_id)
        
        # Run proximity detection
        frame, proximity_alerts = proximity_detector.detect(frame, camera_id)
        
        # If violations detected, emit alerts via socket
        if safety_violations:
            alerts['safety_gear'].append({
                'camera_id': camera_id,
                'violations': safety_violations,
                'timestamp': time.time()
            })
            socketio.emit('safety_alert', json.dumps({
                'camera_id': camera_id,
                'violations': safety_violations
            }))
            
        if proximity_alerts:
            alerts['proximity'].append({
                'camera_id': camera_id,
                'alerts': proximity_alerts,
                'timestamp': time.time()
            })
            socketio.emit('proximity_alert', json.dumps({
                'camera_id': camera_id,
                'alerts': proximity_alerts
            }))
        
        # Encode the processed frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@main.route('/alerts')
def get_alerts():
    return jsonify(alerts) 