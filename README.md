# Face Detector & Tracker

A high-performance real-time face detection, tracking, and landmark analysis suite featuring:
- **Flutter Mobile Application** (`gesture_detector/`): Built with Flutter, supporting on-device Google ML Kit face detection and high-speed USB-NCM camera processing via OpenCV YuNet (`dartcv4`).
- **Python Real-Time Detector** (`python_face_detector/`): Standalone Python pipeline using OpenCV YuNet ONNX, MediaPipe, Haar cascades, Kalman tracking stabilizers, and USB-NCM board streaming.

---

## Repository Structure

```text
face-detector/
├── gesture_detector/        # Flutter mobile application
│   ├── lib/                 # UI screens, camera services, ML Kit & YuNet engines
│   ├── android/             # Android native configuration
│   ├── ios/                 # iOS native configuration
│   └── test/                # Unit, widget, and service tests
├── python_face_detector/    # Python live face detector and stream pipeline
│   ├── live_face_detector.py
│   ├── robust_face_detector.py
│   ├── ncm_camera.py
│   └── requirements.txt
├── run_face_detection.bat   # Windows camera selection menu (Webcam / USB-NCM)
├── run_ncm_face_detection.bat # Windows batch runner for Python detector (USB-NCM)
└── .gitignore
```

---

## Quick Start

### 1. Python Live Detector
Run the batch script:
```cmd
run_face_detection.bat
```
Choose **1** for a standard webcam or **2** for the USB-NCM camera. With explicit
arguments (such as `--ncm`), the launcher skips the menu.

Or for USB-NCM camera input:
```cmd
run_ncm_face_detection.bat
```

### 2. Flutter Mobile App
Navigate into `gesture_detector/`:
```bash
cd gesture_detector
flutter pub get
flutter run
```

---

## Features
- **Real-time Face Detection**: High precision with low latency.
- **Facial Landmark & Mesh Tracking**: Eyes, nose, lips, contours, and orientations.
- **Tracking Stabilization**: Consistent tracking IDs across continuous frames.
- **Multiple Camera Backends**: Front/rear phone cameras, standard webcams, and USB-NCM video streams.
