# USB-NCM face detection on Android and iOS

The app receives the board's JPEG stream directly and runs the PC YuNet pipeline on the phone. It does not depend on a PC, Python backend, cloud service, UVC webcam API, or the gesture classifier from the reference project.

```
Board sensor → USB-NCM network → TCP/JLIP → validated JPEG
  → latest-frame runner (10 inference starts/s maximum)
  → persistent isolate: JPEG decode → YuNet → fresh crop/backlight refinement
  → geometry/score rejection → three-hit confirmation → synchronized preview
```

The connection format follows [the reference NCM client](https://github.com/mdmuhaiminulislamjamee/Custom-Hand-Gesture-Recognition/blob/main/gesture-dashboard-onnx-ncm-camera/backend/ncm_camera.py). This is a new Dart transport implementation; no hand model is used.

The PC test script also accepts this same board stream: run
`python_face_detector/live_face_detector.py --ncm`, or use
`run_ncm_face_detection.bat`. See the [PC NCM setup and profiling guide](../python_face_detector/README.md#test-the-ncm-board-with-live_face_detectorpy).

## What matches the PC

The app returns at most one selected face. It locks onto the largest confirmed
target initially, preserves that lock while it is visible, and switches to a
fresh confirmed alternative after the current track's 200ms dropout hold expires.
The selected count remains zero or one even when internal scanning finds several
candidates. Selection resets with the session and tracker. The app-team delivery
builder and platform package instructions are in `../../handover/README.md`.

| Component | PC webcam | Phone with USB-NCM |
| --- | --- | --- |
| Detector/model | OpenCV YuNet, `face_detection_yunet_2023mar.onnx` | Same model, bundled in the app |
| Model SHA-256 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | Identical |
| Search | 640px full-frame each processed frame; 1280px detail every fifth | Same |
| Search proposals / accepted score | 0.60 / 0.85 | Same |
| Native detail | Up to four fresh crop refinements, 192/320px, up to 4x | Same |
| Backlight recovery | LAB, CLAHE 2.0 (8×8), gamma 0.72, 65/35 blend | Same |
| Pose recovery | Fresh eye/mouth-aligned inference; bounded rotated/stretch search | Same algorithm |
| False-positive checks | Finite geometry, 18px minimum, five landmarks, IoU suppression | Same |
| Confirmation / dropout | Three hits; 200ms hold, held boxes excluded from counts | Same policy |
| Movement response | Alpha at least 0.92 during deliberate motion | Enabled for NCM |
| Scheduling | Latest camera frame; maximum 10 starts/s | One in-flight inference and one replaceable pending frame; maximum 10 starts/s |
| Source | Webcam/MJPEG | JPEG packets from the USB board |

This matches the model and detection policy, not a measured device benchmark. The board's resolution, optics, JPEG compression, exposure, motion blur, and the phone CPU/thermal state still affect results. The app does not lower confidence or remove validation to reach its FPS target.

Pose handling uses a 0.94 gate for directly accepted foreshortened geometry, with
extra mouth topology checks. Weak proposals instead need a fresh inference in an
aligned view at the normal 0.85 floor. Box corners and all five landmarks map back
through the exact inverse transform. At most two full-frame alternatives are
started per recovery pass, with a 25ms scheduling budget after the first start;
recovery is skipped when earlier work has already exceeded 70ms. Native calls
cannot be interrupted, so this is not a hard latency guarantee. Cached pose
parameters are only search hints and need a fresh hit on each current image.
Real 3-D occlusion at extreme pitch/yaw is still a model limitation; rotating a
picture or checking synthetic perspective is not proof of every-angle detection.

The NCM screen shows face boxes, five real YuNet landmarks, confidence, roll, and track IDs. YuNet does not output ML Kit's dense contours, smile probability or eye-open probability, so those values are not fabricated. The separate **Phone camera** option retains the ML Kit features and settings.

## USB network requirement

The phone OS must expose the board as a working USB-NCM network interface. A USB cable alone does not guarantee that: support depends on the exact phone/adapter, board USB descriptors, power supply, OS driver and board firmware. The app cannot install a USB Ethernet driver or assign a static address to a system interface.

Use the same network as the reference firmware:

| Setting | Value |
| --- | --- |
| Phone USB interface | `192.168.50.1/30` (mask `255.255.255.252`) |
| Board | `192.168.50.2` |
| JPEG stream | TCP `5000` |
| Discovery | UDP `5001`, ASCII `JL-CAMERA-DISCOVER` |

Arrange host IP configuration using the device's available USB Ethernet settings or a firmware DHCP server. If a phone cannot expose/configure this interface, that must be resolved in the OS/board integration before application inference can receive frames. The app includes **Discover / check USB network**, which lists the actual IPv4 interfaces and attempts unicast discovery. It never changes Wi-Fi/cellular settings.

The default phone IP explicitly binds traffic to the USB address, which avoids choosing Wi-Fi/cellular. A blank phone IP permits automatic routing if the OS already has a correct route. The board IP and TCP port are editable. Discovery is diagnostic; connecting does not require a discovery reply.

### Android

- Use a USB host/OTG connection and enough power for the board.
- The OS must provide the NCM Ethernet interface; USB tethering is not automatically equivalent to this board connection.
- Release manifest includes `INTERNET` and `ACCESS_NETWORK_STATE`. Camera hardware is optional for the NCM path.
- No phone camera permission is requested for the NCM path.

### iOS / iPadOS

- Use a compatible USB connection/adapter with sufficient power and an OS-supported NCM interface.
- Allow the app's **Local Network** request. If denied, enable it in Settings → Privacy & Security → Local Network.
- `NSLocalNetworkUsageDescription` is included. Discovery is direct unicast; this implementation does not require Bonjour or a multicast entitlement.
- Background/inactive transitions stop capture and clear detections. Returning to the app reconnects only if the user had requested a connection.

## Build

Use Flutter **3.41.7** / Dart **3.11.5** (the repository's `.fvmrc`), or a compatible newer version. Run `flutter pub get`.

`dartcv4` builds OpenCV 4.13 native assets from source with `imgcodecs`, `imgproc`, `objdetect`, and `dnn` enabled. The first build downloads native sources and takes longer than subsequent builds. CMake must be available. Native OpenCV builds remain enabled in the checked-in pubspec.

Android requires JDK 17, Android SDK 36, and NDK `28.2.13676358`; install CMake through the SDK Manager or on PATH. Make sure `android/local.properties` points to the SDK on the build machine. For performance testing on a physical phone:

```sh
flutter run --profile
```

iOS requires a Mac, Xcode, CocoaPods, CMake, and an iOS deployment target of **15.5** or newer. Open `ios/Runner.xcworkspace`, select a signing team for a physical device, then:

```sh
flutter run --profile
```

For unsigned iOS compilation on a Mac:

```sh
flutter build ios --release --no-codesign
```

The included manual GitHub Actions workflow (`.github/workflows/ncm-mobile.yml`) builds Android and unsigned iOS, runs the Flutter tests, and runs the native host detector checks. It has not been run as part of this local change.

## Use and verify

1. Connect and power the board; verify the phone USB network address.
2. Open **USB-NCM camera**. Keep the default IP/port unless firmware differs.
3. Use **Discover / check USB network** to confirm the interface and board response.
4. Tap **Connect camera**. A TCP connection can succeed before the first image arrives; the screen waits for a fresh JPEG before displaying detections.
5. Enable the upside-down mounting option only if needed. Neither preview nor inference is mirrored.

The displayed **detection FPS** measures actual inference start intervals over a rolling three-second window. **Camera FPS** measures received JPEGs separately. The hard cap is 10 starts/s, including failed attempts; a cap cannot force a slow camera or processor to produce 10 usable results/s. **Detection ms** includes native JPEG decoding and the entire YuNet/refinement path. **Frame age** is measured since local receipt, not an unsynchronized board timestamp, so it does not include sensor/firmware latency.

No pending-frame queue can grow: each new frame replaces the pending one. Frames waiting more than 500ms are discarded before inference; results older than 750ms are suppressed. Disconnects and stream stalls clear overlays. Previous face boxes are only fresh-crop search hints and never count as detector hits.

### Automated checks

The PC and mobile YuNet paths also search overlapping enlarged tiles in 640x480
streams when a small face has no full-frame proposal. Strong exposure recovery
and enlarged landmark-aligned views retain the 0.85 acceptance floor and existing
geometry/temporal checks. Small pose-search regions can follow local pixel motion;
these hints always need a fresh detector result and cannot keep a departed face
alive. The native verifier's `--backlight` cases exercise this path on JPEGs.

```sh
flutter analyze
flutter test
dart run tool/verify_yunet.dart --poses --backlight
```

The first two require the Flutter SDK; normal tests also invoke OpenCV's native build hook, requiring the host C++ toolchain. The verifier calls the real OpenCV bindings and checks near/distant/backlit faces, moving JPEGs, disappearance, multiple faces, and non-face scenes. It requires CMake plus a host compiler (Visual Studio C++ on Windows, Xcode on macOS, or GCC/Clang on Linux).

Local verification during this implementation: 77 Flutter tests passed with `dartcv4.skip_build` temporarily enabled for the test run, including real loopback TCP/UDP tests, pose geometry/inverse-coordinate tests, worker loading/disposal tests, and a fake detector for pipeline/lifecycle checks. The option was then removed. Native mobile inference and Android/iOS compilation could not be verified on this Windows machine because its CMake/C++/mobile toolchains are absent. These test results are not evidence of camera accuracy or phone FPS. The Python pose/NCM tests exercise the real PC YuNet engine separately.

### Physical acceptance still required

On each target iPhone/Android phone, use profile/release mode and the real NCM board. Test for at least five minutes with near and distant faces, movement, a bright window behind the subject, multiple faces, empty backgrounds and non-face objects. Compare against the PC using the same board frames whenever possible, rather than comparing different sensors. Record received resolution, camera FPS, detection FPS, detection latency, acquisition delay, misses, and false detections. Repeat after the phone is warm.

Verify that unplugging USB, stopping board video, switching apps and reconnecting never leave a stale face counted. CRC errors, invalid JPEGs and sequence gaps should remain zero on a reliable link. A three-hit confirmation typically needs about 200–300ms at 10 FPS. No fixed distance, guaranteed 10 FPS, or zero-false-positive rate is claimed without this camera/phone validation.
