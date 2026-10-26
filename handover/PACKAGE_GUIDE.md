# NCM Face Detection Handover Guide

This guide explains the delivered Flutter packages, what each part does, what the app team must include, and how to integrate the detector.

## 1. Detection technology

The PC, Android, and iOS pipelines use the same OpenCV YuNet model:

```text
face_detection_yunet_2023mar.onnx
```

The pipeline also includes multi-scale detection, shadow/backlight recovery, landmark and pose validation, confidence filtering, temporal confirmation, tracking stabilization, and single-face selection.

The previous PC detector used MediaPipe BlazeFace short-range. It was less reliable for distant faces, low or tilted camera positions, side angles, small faces, shadows, and strong backlighting. YuNet was selected to provide one consistent detector across all platforms.

## 2. Detection behavior

- Confidence threshold: `0.85`
- Detection confirmation: three consecutive hits
- Processing/display limit: 10 FPS
- Camera stream: may provide more than 10 FPS; extra frames are dropped or skipped
- Public output: zero or one selected face
- Selection: locks to the current confirmed face
- Replacement: selects another confirmed face when the current target disappears
- Dropout hold: up to approximately 200 ms for short missed frames
- `trackingId`: temporary tracking identifier, not biometric identity recognition

The detector may inspect multiple internal candidates so that replacement is fast, but the application never receives multiple selected faces at once.

## 3. Package structure

### `sdk/` — required

The complete reusable Flutter SDK. Copy this entire folder into the existing Flutter application, for example:

```text
vendor/ncm_face_sdk/
```

Do not copy only selected files from this folder.

### `sdk/pubspec.yaml` — required

Defines the package name, Dart/Flutter compatibility, `dartcv4` dependency, model asset, and native build configuration.

### `sdk/lib/` — required

All public and internal runtime code. It must remain complete and intact.

Important parts include:

- `ncm_face_sdk.dart` — public `NcmFaceDetector` controller and result objects.
- NCM TCP/UDP connection and reconnection handling.
- JLIP packet parsing and CRC validation.
- JPEG decoding and latest-frame processing.
- YuNet model loading and OpenCV DNN inference.
- Multi-scale and backlight/shadow recovery.
- Landmark/pose and geometric validation.
- Confidence filtering and three-hit confirmation.
- Face tracking and stabilization.
- Single-face locking and automatic replacement.
- FPS, frame age, dropped-frame, and transport-error metrics.
- App lifecycle pause/resume handling.

The app should use the public API from `ncm_face_sdk.dart`. It should not import or modify `lib/src/` directly.

### `sdk/assets/models/face_detection_yunet_2023mar.onnx` — required

The trained YuNet model used by the SDK. It must remain at the same path and filename. It cannot be used by itself; it requires the complete SDK code and native OpenCV configuration.

### `example/` — optional

Standalone Flutter acceptance/demo application. Use it to test the NCM connection and detector before integrating into the product app. It is not a replacement for the existing company application and does not need to ship in production.

### `docs/INTEGRATION.md` — required reading

General path-dependency setup, native hooks, controller lifecycle, result handling, and coordinate/preview rules.

### `docs/ANDROID_README.md` — required Android reading

Android SDK/JDK/NDK requirements, network permissions, USB-NCM behavior, and physical-device testing instructions.

### `docs/IOS_README.md` — required iOS reading

iOS deployment target, Local Network permission, USB-NCM requirements, lifecycle behavior, and physical-device testing instructions.

### `docs/PROTOCOL.md` — optional but useful

Low-level NCM/JLIP packet format, ports, discovery, framing, CRC, and troubleshooting information. Normally only needed when debugging camera communication.

### `docs/VALIDATION.md` — recommended for QA

Device acceptance plan covering detection, one-face behavior, replacement, rotation, reconnect, lighting, FPS, and error handling.

### `POLICY.json` — reference

Machine-readable detection and performance policy, including the confidence threshold, confirmation count, hold time, and FPS limit.

### `test/` — optional QA code

Automated SDK tests. They cover:

- Face confirmation and temporal filtering.
- Tracking stabilization and dropout behavior.
- Latest-frame and dropped-frame handling.
- NCM connection and reconnection.
- JLIP packet encoding, decoding, and CRC.
- Rotation and coordinate transforms.
- Public API behavior.
- Single-face locking and replacement.
- YuNet confidence/landmark validation.
- Worker/model lifecycle and cleanup.
- `fixtures/face.jpg` and `fixtures/non_face.jpg` test images.

These tests are not called by the production app at runtime.

### `tools/` — optional release QA

- `verify_package.py` — validates archive paths, files, and checksums.
- `verify_release.py` — checks Android/iOS package parity, model/configuration consistency, and required settings.

These are not imported into Flutter runtime code.

### PC Python source — not required in the mobile app

The PC `python_face_detector/` code and batch launchers are the reference PC implementation and diagnostic tools. They are not dependencies of the Android or iOS Flutter SDK.

## 4. Flutter integration steps

Add the complete SDK as a path dependency in the host app:

```yaml
dependencies:
  ncm_face_sdk:
    path: vendor/ncm_face_sdk
```

Merge the native `dartcv4` settings into the host app's root `pubspec.yaml`:

```yaml
hooks:
  user_defines:
    dartcv4:
      parallel_jobs: 2
      include_modules:
        - imgcodecs
        - imgproc
        - objdetect
        - dnn
      ios:
        deployment_target: '15.5'
```

Create one controller for the screen/service lifetime:

```dart
final detector = NcmFaceDetector();
```

Start the NCM camera connection:

```dart
await detector.start(const NcmCameraConfig(
  hostAddress: '192.168.50.1',
  deviceAddress: '192.168.50.2',
  tcpPort: 5000,
));
```

Read the selected result:

```dart
final frame = detector.frame;
final face = frame?.freshFace; // null or one confirmed fresh face
```

Use `face.displayBox` and `face.displayLandmarks` over the JPEG from the same `frame`. Avoid constructing a model/worker for every frame. Remove the listener and call `dispose()` when the owning screen/service is permanently removed.

## 5. Platform requirements

### Android

- Android SDK 36, minimum SDK 24, JDK 17, and the documented NDK/CMake toolchain.
- Add `INTERNET` and `ACCESS_NETWORK_STATE` permissions.
- The SDK uses the phone's NCM network interface, not Android's UVC camera API.
- No phone camera permission is required for this NCM path.
- Test on the real supported arm64 device and USB/OTG adapter.

### iOS

- iOS deployment target 15.5 or higher.
- Add `NSLocalNetworkUsageDescription` to the host app's `Info.plist`.
- The SDK uses raw TCP/unicast UDP over the NCM network interface.
- No phone camera permission, Bonjour, or arbitrary ATS bypass is required for this protocol.
- Test on a physical iPhone/iPad with the actual adapter and camera board.

## 6. NCM network requirements

The phone must expose the camera board as a USB-NCM/Ethernet network interface. The usual configuration is:

```text
Phone/host:  192.168.50.1/30
Camera:      192.168.50.2/30
TCP:         5000
UDP:         5001
```

A USB cable or tethering connection alone does not guarantee that the operating system has created the required network interface or route. The app cannot provide a missing OS driver or unsupported adapter configuration.

## 7. What the app team must use

They must use:

1. The complete `sdk/` folder.
2. The YuNet model inside `sdk/assets/models/` without renaming or moving it.
3. The SDK path dependency and native build hooks.
4. The relevant Android or iOS platform configuration.
5. The public `NcmFaceDetector` API.

They must not integrate only the `.onnx` file or selected files from `lib/src/`.

## 8. What they may ignore

They may omit these from the production app if they do not need them:

- `example/` demo application.
- `docs/PROTOCOL.md` after successful integration.
- `docs/VALIDATION.md` after QA is complete, although it should be retained by QA.
- `POLICY.json` at runtime.
- `tools/verify_package.py` and `tools/verify_release.py` at runtime.
- PC Python source and batch launchers.
- SDK test files at runtime.

Keep the documentation and tests in the handover archive for maintenance and release verification even if they are not copied into the production application.

## 9. Important limitations and acceptance notes

YuNet improves robustness but cannot guarantee detection in every image. Extremely small faces, severe overexposure, complete shadow, heavy motion blur, occlusion, or unsupported NCM/OS networking can still prevent detection. Android/iOS native builds and physical-device FPS must be validated by the app team on their target hardware. The delivered source package is not a prebuilt APK, IPA, AAR, or XCFramework.

