# Validation and performance acceptance

This delivery matches model bytes, confidence policy, recovery searches, selected
face behavior, and the processing-rate cap. It does **not** claim identical phone
latency/accuracy/FPS to a PC without testing the actual device and camera.
The PC uses its installed OpenCV build (recorded in verification output); the
mobile dependency builds OpenCV 4.13.0. CPU architecture, thermal throttling,
exposure, JPEG compression, camera angle, distance, and available pixels matter.

`POLICY.json` records the delivered threshold snapshot. The user's PC is currently
configured for 0.85 accepted score, 0.60 proposals, and 0.94 for directly accepted
foreshortened geometry. Geometry and three-hit confirmation remain. These model
scores are not measured accuracy percentages; zero false detections and
every-angle/silhouette detection cannot be guaranteed by these thresholds.

## Automated checks

From `sdk/` after installing Flutter and the host C++/CMake toolchain:

```sh
flutter pub get
flutter analyze
flutter test
dart run tool/verify_yunet.dart --poses --backlight
```

The native verifier checks the raw engine, which can find multiple internal
candidates. The service and single-face selector tests check the exclusive
zero/one output and replacement behavior. Native calls require compiled OpenCV
assets; unit tests using a fake processor are not a substitute for that check.
Build hooks remain enabled in the delivered pubspecs. See `VALIDATION_STATUS.json`
for exactly which local checks passed and which need the target toolchain.

## Physical phone acceptance

Run `example/` or the integrated host app with `flutter run --profile` on a real
Android phone or iPhone. Do not compare debug-build speed with the PC release
runtime. Verify all of these cases:

| Scenario | Expected result |
| --- | --- |
| Two people initially visible | At most one selected face; largest confirmed target wins |
| Another larger face enters | Original visible target keeps its lock |
| Detection ordering changes | Target does not switch just because list order changes |
| Selected person leaves | Brief hold <=200ms, then another fresh confirmed target takes over |
| New unconfirmed alternative | Needs its own three consecutive fresh observations |
| Everyone leaves | Fresh count zero; no stale active face; held overlay expires |
| USB unplug/stall/reconnect | Clear results; never publish a late old-session face |
| Resolution/orientation changes | Reset history; boxes/landmarks align with the processed JPEG |
| Near/far, low/high/left/right, backlight | Record misses/acquisition delay; do not invent invisible detail |
| Windows, shelves, clothes, animal images | No confirmed face in the tested negatives |
| Five minutes warm operation | Record thermal effects, memory growth, and sustained processing rate |

Record device model/OS, build mode, received resolution, camera FPS, detection
FPS, processing p50/p95, frame age p50/p95, acquisition and target-switch delay,
confirmed false detections, miss rate, CRC/header/JPEG errors, sequence gaps, and
memory after warm-up. The example emits one `NCM_RESULT` log line per NEW
processed frame. Repeated status notifications are not new frames.

The 10 FPS limit caps **inference starts**, not camera delivery. A board sending
17 or 20 JPEGs/s is normal; extra frames are replaced rather than queued. Slow
inference may reduce achieved FPS below 10; never fake this metric. Frames waiting
over 500ms are dropped and results older than 750ms are suppressed. Native calls
cannot be interrupted, so a recovery pass can occasionally exceed 100ms.

## PC reference

The bundled `pc_reference/` is the same source snapshot and model, not a PC
service required by the phone. On Windows, run its `run_face_detection.bat` and
choose 2. For a headless measurement from `pc_reference/python_face_detector/`:

```sh
python -m venv .venv
# Activate the environment using the command appropriate to your shell.
python -m pip install -r requirements-lock.txt
python live_face_detector.py --ncm --profile-camera --frames 100
```

The report includes `max_selected_faces` (must be <=1) and measured processing
rate. Compare equivalent recorded/captured board images where possible, one client
at a time. Do not treat synthetic perspective warps as proof of real 3-D views.

`PC_RUNTIME.json` and `requirements-lock.txt` record the PC versions actually
tested for this snapshot. Keep the host Flutter app's resolved dependency lock
file as well when recording phone benchmarks.
