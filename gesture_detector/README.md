# Face Detector

A Flutter app with two camera paths: USB-NCM board camera with the PC's YuNet face pipeline running on-device, and built-in phone cameras with Google ML Kit.

Choose **USB-NCM camera** on the start screen for the external board. No PC or Python server is needed. See [NCM mobile setup and validation](docs/NCM_MOBILE.md) for Android/iOS build requirements, USB network setup, and hardware acceptance tests.

## Overview

Face Detector processes live camera feeds from front and back cameras, detecting faces in real-time with:
- **Persistent Face Tracking**: Unique tracking IDs across consecutive video frames (`Face #1`, `Face #2`, etc.).
- **Facial Contours (Mesh)**: Real-time polygonal contours tracing the face oval, left/right eyebrows, eye outlines, lips, and nose bridge.
- **Facial Landmarks**: Precise landmark coordinates for eyes, ears, nose base, and mouth corners.
- **Classification & Metrics**:
  - Smiling probability percentage (`😊 85%`)
  - Left and right eye open status (`👁️ L:95% R:92%`)
  - Head rotation and orientation (Euler roll/tilt degrees)
- **Camera Controls**:
  - Front and back camera switching
  - Flashlight / Torch toggle for back camera
  - High-resolution live preview with aspect ratio preservation
  - Detection latency and real-time face count badge
- **Interactive Configuration**:
  - Modal bottom sheet to toggle bounding boxes, mesh lines, landmarks, and classification metrics
  - Switchable performance modes: **Fast** (optimized for maximum frame rates) vs. **Accurate** (high-precision detection)

## Tech Stack

| Layer | Library / Tool |
| --- | --- |
| Framework | Flutter (Dart) |
| Face Detection Engine | `google_mlkit_face_detection: ^0.14.0` |
| USB-NCM Face Engine | `dartcv4: 2.3.1`, OpenCV YuNet, same ONNX model as PC |
| USB-NCM Transport | Direct Dart TCP/JLIP + UDP discovery, CRC32 and heartbeat handling |
| Camera Stream | `camera: ^0.12.0+2` |
| Permissions | `permission_handler: ^12.0.3` |
| Image Processing | Background Isolate NV21 frame encoder (Android) / BGRA8888 plane extraction (iOS) |

## Project Structure

```text
lib/
├── main.dart                                  # Application entry point
├── app/
│   └── face_detector_app.dart                 # MaterialApp setup, dark camera theme
├── models/
│   ├── detected_face.dart                     # Model representing detected face attributes & coordinates
│   └── face_detector_options_model.dart       # Configuration options for detection engine and UI
├── services/
│   └── face_detector_service.dart             # ML Kit FaceDetector lifecycle and frame processing
├── utils/
│   ├── camera_image_converter.dart            # CameraImage to InputImage conversion & coordinate mapping
│   ├── face_geometry_utils.dart               # Preview normalization, canvas projection & aspect ratio
│   └── app_snack_bar.dart                     # Reusable floating snackbar feedback
└── presentation/
    ├── screens/
    │   └── face_detector_screen.dart          # Full-screen live camera view, overlay, and action controls
    ├── painters/
    │   └── face_detection_painter.dart        # CustomPainter for bounding boxes, mesh, landmarks & stats
    └── widgets/
        ├── camera_loading_view.dart           # Loading indicator, permission request / error prompt view
        ├── face_count_badge.dart              # Floating status badge with face count and latency
        └── face_options_bottom_sheet.dart     # Settings sheet to toggle detector options
```

## Running the App

Ensure Flutter is installed on your machine and run:

```bash
# Get dependencies
flutter pub get

# Run tests
flutter test

# Run the app on your connected device
flutter run
```

## Distance, backlighting, and false detections

The mobile camera now requests the 1080p preset, uses automatic exposure/focus
where supported, and lifts recoverable shadows before ML Kit processing. Actual
stream resolution depends on the device. Accurate mode defaults to a 3% minimum
face-width search setting; this does not guarantee detection at that size.
Face detection and overlay updates are limited to 10 FPS; busy or early camera
frames are skipped so processing always uses a recent image.

Landmark geometry and three consecutive observations are required before a face
appears, including when smoothing is disabled. Detection continues for all faces;
the optional contour detector runs separately for faces at least 200 pixels wide
and tall. To keep face detection responsive, contours refresh at most once every
300 ms and only when primary processing finishes within 50 ms. Recent contour
points follow the current face box and expire after 400 ms. The camera converter
uses a persistent background worker and copies contiguous luminance rows in bulk,
preserving full-resolution input and avoiding an isolate startup per frame.
Lost tracks are marked as held for at most 200 ms, excluded from the
face count, and do not show stale landmarks or facial metrics. Busy camera frames
are skipped without clearing the previous result.

Very small, blurred, or fully silhouetted faces can still lack enough detail to
detect safely. Zero false positives cannot be guaranteed. Test the intended phone
and camera in real lighting; unit tests cannot measure native ML Kit accuracy.
See [ML Kit input image guidance](https://developers.google.com/ml-kit/vision/face-detection/android#input-image-guidelines).

The Windows webcam version is in [`python_face_detector`](python_face_detector/README.md).
