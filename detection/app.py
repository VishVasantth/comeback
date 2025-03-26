from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
import os
import requests
import threading
import time
from ultralytics import YOLO
import json
import traceback

app = Flask(__name__)
CORS(app)

# Configuration
BACKEND_URL = os.environ.get('BACKEND_URL', 'http://localhost:5000')
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/yolov11n.pt')
RTSP_URL = os.environ.get('RTSP_URL', '')
DETECTION_THRESHOLD = float(os.environ.get('DETECTION_THRESHOLD', '0.3'))  # Lower threshold for better detection
# Include more classes as obstacles - COCO dataset classes
OBSTACLE_CLASSES = os.environ.get('OBSTACLE_CLASSES', '0,1,2,3,5,7,9,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28').split(',')
OBSTACLE_CLASSES = [int(x) for x in OBSTACLE_CLASSES]  # person, bicycle, car, motorcycle, bus, truck, etc.

# Global variables
model = None
video_capture = None
detection_thread = None
keep_running = False
current_frame = None
detected_objects = []
last_obstacle_time = 0
current_location = (0, 0)  # Default location
obstacle_reported = False  # Flag to track if an obstacle has been reported
blocking_threshold = 0.3  # How much of width/height is considered blocking

# COCO class names for display
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def load_model():
    global model
    try:
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            print(f"Model file not found at {MODEL_PATH}")
            print(f"Current working directory: {os.getcwd()}")
            return False
        
        print(f"Loading model from {MODEL_PATH}")
        # Load YOLOv11n model
        model = YOLO(MODEL_PATH)
        model.to(device)
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        print(traceback.format_exc())
        return False

def connect_to_rtsp():
    global video_capture
    try:
        if not RTSP_URL:
            print("RTSP URL not provided. Using webcam instead.")
            video_capture = cv2.VideoCapture(0)
        else:
            print(f"Connecting to RTSP stream: {RTSP_URL}")
            # Check if it's a path to a test image
            if os.path.exists(RTSP_URL) and (RTSP_URL.lower().endswith('.jpg') or 
                                              RTSP_URL.lower().endswith('.jpeg') or 
                                              RTSP_URL.lower().endswith('.png')):
                print(f"Using test image: {RTSP_URL}")
                # Load the image directly
                img = cv2.imread(RTSP_URL)
                if img is not None:
                    # Save a temporary video file from the image for compatibility
                    temp_video = "temp_video.mp4"
                    height, width = img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(temp_video, fourcc, 1, (width, height))
                    # Write the same image 30 times (30 frames)
                    for _ in range(30):
                        writer.write(img)
                    writer.release()
                    video_capture = cv2.VideoCapture(temp_video)
                    return video_capture.isOpened()
            
            # Otherwise treat as RTSP URL or video file
            video_capture = cv2.VideoCapture(RTSP_URL)
        
        if not video_capture.isOpened():
            print("Failed to open video stream. Trying to use test video or image...")
            # Try to use a test image from the test_data directory
            test_paths = [
                os.path.join("detection", "test_data", "synthetic_obstacle.jpg"),
                os.path.join("detection", "test_data", "people_walking.jpg"),
                os.path.join("test_data", "synthetic_obstacle.jpg"),
                os.path.join("test_data", "people_walking.jpg"),
                "detection/test_video.mp4", "test_video.mp4", 
                "detection/test_image.jpg", "test_image.jpg"
            ]
            
            for path in test_paths:
                if os.path.exists(path):
                    print(f"Using test file: {path}")
                    # If it's an image, convert to a looping video
                    if path.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img = cv2.imread(path)
                        if img is not None:
                            # Save a temporary video file
                            temp_video = "temp_video.mp4"
                            height, width = img.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            writer = cv2.VideoWriter(temp_video, fourcc, 1, (width, height))
                            # Write the same image 30 times (30 frames)
                            for _ in range(30):
                                writer.write(img)
                            writer.release()
                            video_capture = cv2.VideoCapture(temp_video)
                            if video_capture.isOpened():
                                return True
                    else:
                        video_capture = cv2.VideoCapture(path)
                        if video_capture.isOpened():
                            return True
            
            # If we still can't open a video, create a test pattern
            print("Creating test pattern for object detection")
            height, width = 480, 640
            test_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add a road-like gray area
            cv2.rectangle(test_image, (0, 180), (width, 300), (80, 80, 80), -1)
            
            # Add some shapes that might be detected as objects
            cv2.rectangle(test_image, (100, 100), (200, 300), (0, 255, 0), -1)  # Green rectangle
            cv2.circle(test_image, (400, 200), 80, (0, 0, 255), -1)  # Red circle
            
            # Add a person-like obstacle in the center
            cv2.rectangle(test_image, (width//2 - 15, 200), (width//2 + 15, 280), (255, 0, 0), -1)
            
            # Add labels
            cv2.putText(test_image, "Test Pattern with Obstacles", (180, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save the test image
            cv2.imwrite("test_pattern.jpg", test_image)
            
            # Create a video from the test image
            temp_video = "test_pattern.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(temp_video, fourcc, 1, (width, height))
            # Write the same image 30 times (30 frames)
            for _ in range(30):
                writer.write(test_image)
            writer.release()
            
            # Load as video
            video_capture = cv2.VideoCapture(temp_video)
            return video_capture.isOpened()
        
        print("Connected to video stream successfully")
        return True
    except Exception as e:
        print(f"Error connecting to video stream: {e}")
        print(traceback.format_exc())
        return False

def is_obstacle_blocking_path(frame_shape, box, path_width_ratio=0.3):
    """Determine if an object is blocking the path based on its position in the frame"""
    x1, y1, x2, y2 = box
    frame_height, frame_width = frame_shape[:2]
    
    box_width = x2 - x1
    box_height = y2 - y1
    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2
    
    # Consider the object blocking if it's in the center portion of the frame
    # and takes up significant space
    path_width = frame_width * path_width_ratio
    path_left = (frame_width - path_width) / 2
    path_right = path_left + path_width
    
    # Check if object is in the path
    in_path = path_left <= box_center_x <= path_right
    
    # Check if object is large enough to be considered an obstacle
    # Objects higher in the frame (further away) should be treated as obstacles
    # even if they appear smaller
    height_factor = 1.0 - (box_center_y / frame_height)  # 1.0 at top, 0.0 at bottom
    size_threshold = 0.05 + (height_factor * 0.1)  # Dynamic threshold based on position
    
    relative_size = (box_width * box_height) / (frame_width * frame_height)
    is_large_enough = relative_size > size_threshold
    
    # Check if the object is in the lower half of the frame (closer to the camera)
    is_close = box_center_y > frame_height * 0.4
    
    return in_path and (is_large_enough or is_close)

def detect_objects():
    global current_frame, detected_objects, last_obstacle_time, keep_running, obstacle_reported
    
    while keep_running:
        if video_capture is None or not video_capture.isOpened():
            print("Video capture not available")
            time.sleep(1)
            continue
        
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            # If it's a test image or we're at the end of a video, reset to first frame
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.1)
            continue
        
        # Store current frame for display
        current_frame = frame.copy()
        
        # Run detection
        results = model(frame, conf=DETECTION_THRESHOLD)
        
        # Process results
        current_detected = []
        obstacle_detected = False
        obstacle_box = None
        obstacle_class = None
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0].item())
                confidence = box.conf[0].item()
                
                if confidence >= DETECTION_THRESHOLD:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Determine if object is in the path
                    is_obstacle = False
                    class_name = COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else f"Class {cls}"
                    
                    # Check if this is a class we consider an obstacle
                    if cls in OBSTACLE_CLASSES:
                        is_obstacle = is_obstacle_blocking_path(frame.shape, [x1, y1, x2, y2])
                        if is_obstacle:
                            obstacle_detected = True
                            obstacle_box = [x1, y1, x2, y2]
                            obstacle_class = cls
                    
                    label = f"{class_name}: {confidence:.2f}"
                    current_detected.append({
                        "class": cls,
                        "class_name": class_name,
                        "confidence": confidence,
                        "box": [x1, y1, x2, y2],
                        "is_obstacle": is_obstacle
                    })
                    
                    # Draw rectangle around object
                    color = (0, 0, 255) if is_obstacle else (0, 255, 0)
                    cv2.rectangle(current_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(current_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        detected_objects = current_detected
        
        # Draw path region in the center
        frame_height, frame_width = frame.shape[:2]
        path_width = frame_width * blocking_threshold
        path_left = int((frame_width - path_width) / 2)
        path_right = int(path_left + path_width)
        
        # Draw semi-transparent path overlay
        overlay = current_frame.copy()
        cv2.rectangle(overlay, (path_left, 0), (path_right, frame_height), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.2, current_frame, 0.8, 0, current_frame)
        
        # Draw path borders
        cv2.line(current_frame, (path_left, 0), (path_left, frame_height), (0, 255, 255), 2)
        cv2.line(current_frame, (path_right, 0), (path_right, frame_height), (0, 255, 255), 2)
        
        # Notify backend if obstacle detected and not recently reported
        current_time = time.time()
        if obstacle_detected and not obstacle_reported and (current_time - last_obstacle_time > 5):
            last_obstacle_time = current_time
            obstacle_reported = True
            
            # Add a clear visual indication of obstacle detection
            cv2.putText(current_frame, "OBSTACLE DETECTED! REROUTING...", 
                       (frame_width//2 - 200, frame_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            try:
                # Get center of obstacle for reporting
                if obstacle_box:
                    obstacle_center_x = (obstacle_box[0] + obstacle_box[2]) / 2 / frame_width
                    obstacle_center_y = (obstacle_box[1] + obstacle_box[3]) / 2 / frame_height
                    
                    # Convert relative position to map coordinates (simplified)
                    # This is just a placeholder - in a real system, you'd use proper coordinate transformation
                    lat_offset = (0.5 - obstacle_center_y) * 0.001  # Small offset based on position
                    lon_offset = (obstacle_center_x - 0.5) * 0.001
                    
                    obstacle_lat = current_location[0] + lat_offset
                    obstacle_lon = current_location[1] + lon_offset
                    
                    print(f"Reporting obstacle: Class={COCO_CLASSES[obstacle_class] if obstacle_class < len(COCO_CLASSES) else f'Class {obstacle_class}'}")
                    print(f"At position: ({obstacle_lat}, {obstacle_lon})")
                    
                    response = requests.post(
                        f"{BACKEND_URL}/obstacle",
                        json={"lat": obstacle_lat, "lon": obstacle_lon},
                        timeout=2
                    )
                    print(f"Reported obstacle to backend: {response.status_code}")
                    print(f"Response: {response.text}")
                else:
                    # If we don't have box info, just report at current location
                    response = requests.post(
                        f"{BACKEND_URL}/obstacle",
                        json={"lat": current_location[0], "lon": current_location[1]},
                        timeout=2
                    )
                    print(f"Reported obstacle at current location: {response.status_code}")
            except Exception as e:
                print(f"Failed to report obstacle: {e}")
        elif not obstacle_detected and obstacle_reported:
            # Reset the obstacle reported flag if no obstacle is detected
            obstacle_reported = False
        
        time.sleep(0.05)  # Limit processing rate

def start_detection():
    global detection_thread, keep_running
    
    if detection_thread is not None and detection_thread.is_alive():
        return {"status": "error", "message": "Detection already running"}
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        model_path = os.path.abspath(MODEL_PATH)
        print(f"Model file not found at {model_path}")
        return {
            "status": "error", 
            "message": f"Model file not found. Please run 'python download_model.py' to download the model to {model_path}"
        }
    
    if model is None and not load_model():
        return {"status": "error", "message": "Failed to load model. Check console for details."}
    
    if (video_capture is None or not video_capture.isOpened()) and not connect_to_rtsp():
        return {"status": "error", "message": "Failed to connect to video stream"}
    
    keep_running = True
    detection_thread = threading.Thread(target=detect_objects)
    detection_thread.daemon = True
    detection_thread.start()
    
    return {"status": "success", "message": "Detection started"}

def stop_detection():
    global keep_running, detection_thread, video_capture
    
    keep_running = False
    
    if detection_thread is not None:
        detection_thread.join(timeout=1.0)
        detection_thread = None
    
    if video_capture is not None:
        video_capture.release()
        video_capture = None
    
    return {"status": "success", "message": "Detection stopped"}

@app.route('/start', methods=['POST'])
def api_start_detection():
    result = start_detection()
    return jsonify(result)

@app.route('/stop', methods=['POST'])
def api_stop_detection():
    result = stop_detection()
    return jsonify(result)

@app.route('/status', methods=['GET'])
def api_status():
    return jsonify({
        "running": detection_thread is not None and detection_thread.is_alive(),
        "model_loaded": model is not None,
        "video_connected": video_capture is not None and video_capture.isOpened(),
        "detected_objects": len(detected_objects),
        "obstacle_detected": any(obj.get("is_obstacle", False) for obj in detected_objects),
        "detection_threshold": DETECTION_THRESHOLD,
        "obstacle_classes": [COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else f"Class {cls}" for cls in OBSTACLE_CLASSES]
    })

@app.route('/objects', methods=['GET'])
def get_objects():
    return jsonify({"objects": detected_objects})

@app.route('/frame', methods=['GET'])
def get_frame():
    if current_frame is None:
        return jsonify({"status": "error", "message": "No frame available"}), 404
    
    _, buffer = cv2.imencode('.jpg', current_frame)
    frame_bytes = buffer.tobytes()
    
    response = app.response_class(
        response=frame_bytes,
        status=200,
        mimetype='image/jpeg'
    )
    return response

@app.route('/location', methods=['POST'])
def update_location():
    global current_location, obstacle_reported
    data = request.json
    lat = data.get('lat')
    lon = data.get('lon')
    
    if not all([lat, lon]):
        return jsonify({"status": "error", "message": "Missing coordinates"}), 400
    
    # When location changes significantly, reset obstacle reported flag
    if current_location != (0, 0):  # Not first update
        old_lat, old_lon = current_location
        # Calculate approximate distance
        dist = ((float(lat) - old_lat)**2 + (float(lon) - old_lon)**2)**0.5
        if dist > 0.0001:  # If moved more than ~10 meters
            obstacle_reported = False
    
    current_location = (lat, lon)
    return jsonify({"status": "success", "message": "Location updated"})

@app.route('/config', methods=['POST'])
def update_config():
    global DETECTION_THRESHOLD, OBSTACLE_CLASSES, blocking_threshold
    
    data = request.json
    if 'threshold' in data:
        DETECTION_THRESHOLD = float(data.get('threshold'))
    
    if 'blocking_threshold' in data:
        blocking_threshold = float(data.get('blocking_threshold'))
    
    if 'obstacle_classes' in data:
        OBSTACLE_CLASSES = [int(x) for x in data.get('obstacle_classes').split(',')]
    
    return jsonify({
        "status": "success", 
        "message": "Configuration updated",
        "config": {
            "detection_threshold": DETECTION_THRESHOLD,
            "blocking_threshold": blocking_threshold,
            "obstacle_classes": OBSTACLE_CLASSES
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    # Try to load model at startup
    load_model()
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True) 