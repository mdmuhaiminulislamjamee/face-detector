# Integrate into an existing Flutter app

Copy `sdk/` into your repository, for example `vendor/ncm_face_sdk/`. Add a path
dependency to **your app's** pubspec:

```yaml
dependencies:
  ncm_face_sdk:
    path: vendor/ncm_face_sdk
```

Also merge these native-build settings into your **app's root pubspec**, not just
the SDK dependency. Keep any modules your app already uses:

```yaml
hooks:
  user_defines:
    dartcv4:
      parallel_jobs: 2
      include_modules: [imgcodecs, imgproc, objdetect, dnn]
      ios:
        deployment_target: '15.5'
```

Use a compatible Flutter SDK with Dart >=3.11.5. The reference project pins
Flutter 3.41.7; local source analysis/tests were run on Flutter 3.47.8/Dart 3.13.7.
`dartcv4` is pinned to 2.3.1 (OpenCV 4.13.0). Do not ship `skip_build: true`: the
native OpenCV libraries are required. First native builds need network access,
CMake, a host C++ compiler, and the platform toolchain. If the host app already
uses another OpenCV binding, resolve native-library version conflicts first.
Flutter's [package integration guide](https://docs.flutter.dev/packages-and-plugins/using-packages)
and the [dartcv4 source](https://github.com/rainyl/opencv_dart)
provide dependency/build context.

The model is bundled by the package and loaded from
`packages/ncm_face_sdk/assets/models/face_detection_yunet_2023mar.onnx`.
Do not add a second differently named model or use a PC file path on phones.
The SDK has no ML Kit, camera-plugin, Python, or network-service dependency.

## Minimal integration

The fragment below shows controller usage; place the listener/controller in your
State/service and the `await start`/cleanup calls in the relevant methods. A
complete, runnable widget is provided in `example/lib/main.dart`.

```dart
import 'package:flutter/widgets.dart';
import 'package:ncm_face_sdk/ncm_face_sdk.dart';

// Construct from an initialized Flutter app, e.g. State.initState().
final detector = NcmFaceDetector();
int? lastProcessedFrame;

void onUpdate() {
  final current = detector.frame;
  if (current == null) {
    // Clear your preview overlay and application face state.
    return;
  }
  // Status notifications can repeat the SAME processed frame. Do not count
  // them as new detector observations or run business actions twice.
  if (current.frameId == lastProcessedFrame) return;
  lastProcessedFrame = current.frameId;

  final face = current.freshFace; // null or ONE confirmed, fresh face
  if (face != null) {
    final normalizedBox = face.displayBox;
    final score = face.detectionConfidence;
    final trackId = face.trackingId;
    // Use this face for your application's behavior.
    // Paint on current.jpeg using current.width/current.height.
  }
}

// In your screen initialization/connection action:
detector.addListener(onUpdate);
await detector.start(const NcmCameraConfig(
  hostAddress: '192.168.50.1',
  deviceAddress: '192.168.50.2',
  tcpPort: 5000,
));

// On manual disconnect:
detector.stop();
// When the owning screen/service is permanently removed:
detector.removeListener(onUpdate);
detector.dispose();
```

`start()` initializes the model and starts connection attempts; it does not mean
the first image has arrived. Check `status`, `error`, `initializing`, and `frame`.
Do not instantiate a worker/model per camera frame. Keep one controller alive
for the screen's lifetime and dispose it once. The default controller stops on
inactive/background lifecycle states and reconnects on resume only if the user
had requested capture. Use `manageAppLifecycle: false` only when the host app
implements this stop/resume behavior itself.

## Result and selection contract

`frame` is null while waiting/stopped/disconnected or after stale-result clearing.
`frame.faces` has **zero or one** selected track, including an optional held
overlay. `frame.freshFace` excludes held overlays and is the correct input for
business actions and the active face counter. A track ID associates boxes over
time; it is not biometric identity or a recognition result.

The first target is the largest confirmed face, with confidence and ID as
deterministic tie breakers. The lock stays on it when a larger face enters or
detector output order changes. The existing dropout hold lasts at most 200ms.
When the target expires, a different freshly visible, confirmed face can take
over on the next processed result. If the replacement is new, it needs its own
three consecutive hits. Internal candidate scanning/confirmation can inspect
multiple faces so that replacement is prompt; the public SDK never exposes more
than one selected target. A held alternative cannot acquire a lock. Disconnect,
resolution change, and session reset clear both tracker and selection.

## Preview coordinates

Use the JPEG from the **same** `NcmFaceFrame`, never a newer raw camera image.
`boundingBox` uses processed-image pixels. `displayBox` and `displayLandmarks`
are normalized coordinates. Multiply x by the painted image width and y by its
painted image height; account for any `BoxFit.contain` letterboxing offset.
The example paints inside `AspectRatio(width / height)` to avoid that mismatch.
No mirror is applied. If `rotate180` is true, rotate the original preview JPEG
180 degrees too; inference and reported coordinates already use that rotation.
Only five actual YuNet landmarks and roll are supplied. Pitch/yaw, smile,
eye-open probabilities, and dense contours are not fabricated.

The example explicitly evicts old `MemoryImage` objects. Do not retain every JPEG
or inference result in an ever-growing list, image cache, or UI event queue.
The controller exposes actual detection FPS, camera FPS, processing duration,
frame age, dropped frames, and transport error counters for measurement.
