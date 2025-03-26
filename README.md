# Navigation App with Obstacle Detection

This application provides real-time path planning using OpenStreetMap (OSM) with obstacle detection capabilities using YOLOv11n.

## Features

- Path planning from point A to point B using OpenStreetMap data
- Real-time object detection using YOLOv11n with CUDA acceleration
- Automatic rerouting when obstacles are detected
- RTSP video feed support from smartphones
- Interactive web interface showing map, path, and video feed

## System Architecture

The system consists of three main components:

1. **Backend Server**: Handles OSM path planning and rerouting
2. **Detection Service**: Processes video feed for obstacle detection using YOLOv11n
3. **Frontend**: Web interface for user interaction and visualization

All components can run independently, communicating through RESTful APIs.

## Prerequisites

- Python 3.8+
- Node.js 14+
- CUDA-compatible GPU (for YOLOv11n acceleration)
- YOLOv11n model (download instructions below)

## Installation

### 1. Backend Server

```bash
cd backend
pip install -r requirements.txt
```

### 2. Detection Service

```bash
cd detection
pip install -r requirements.txt

# Create directory for model
mkdir -p models

# Download YOLOv11n model to models directory
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov11n.pt -o models/yolov11n.pt
```

### 3. Frontend

```bash
cd frontend
npm install
```

## Configuration

### Environment Variables

Create `.env` files in each directory to configure the services:

#### Backend (.env)
```
PORT=5000
```

#### Detection (.env)
```
PORT=5001
BACKEND_URL=http://localhost:5000
MODEL_PATH=models/yolov11n.pt
RTSP_URL=your_rtsp_url_here
DETECTION_THRESHOLD=0.5
OBSTACLE_CLASSES=0,1,2,3,5,7
```

#### Frontend (.env)
```
REACT_APP_BACKEND_URL=http://localhost:5000
REACT_APP_DETECTION_URL=http://localhost:5001
```

## Running the Application

### Start Backend Server

```bash
cd backend
python app.py
```

### Start Detection Service

```bash
cd detection
python app.py
```

### Start Frontend

```bash
cd frontend
npm start
```

## Usage

1. Open the web interface at `http://localhost:3000`
2. Select start and destination locations from the dropdown or enter coordinates
3. Click "Find Path" to calculate a route
4. Click "Start Detection" to begin obstacle detection
5. Use "Simulate Movement" to test obstacle detection and rerouting

## Notes

- For actual deployment, consider using production-ready servers like Gunicorn for Python services
- The RTSP URL should be provided by your smartphone camera app (e.g., IP Webcam)
- Make sure your GPU drivers are properly configured for CUDA support
- If you encounter "Model not found" errors, verify that the model file exists in the models directory 