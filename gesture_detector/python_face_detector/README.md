# Windows webcam and USB-NCM detector

Run `..\run_face_detection.bat` or `run.bat` without arguments to choose:

- **1: Webcam** — built-in camera or standard USB webcam, camera index `0`.
- **2: USB-NCM camera** — the board's TCP/JLIP JPEG stream (`--ncm`).
- **Q: Quit**.

Explicit command-line arguments bypass the menu, for example
`..\run_face_detection.bat --source webcam 1` or
`..\run_face_detection.bat --ncm --ncm-host-ip auto`. The dedicated
`..\run_ncm_face_detection.bat` shortcut also opens NCM directly.
The launcher uses the bundled YuNet ONNX model and requires OpenCV and NumPy;
MediaPipe is no longer required.

## Test the NCM board with live_face_detector.py

From `D:\face_detector`, launch the dedicated shortcut:

```powershell
.\run_ncm_face_detection.bat
```

Or use the existing launcher / Python script:

```powershell
.\run_face_detection.bat --ncm
cd python_face_detector
.venv\Scripts\python.exe live_face_detector.py --ncm
```

`--source ncm` is equivalent to `--ncm`. When calling `live_face_detector.py`
directly with no source argument, it opens the normal webcam index (default `0`);
the batch launcher's no-argument menu selects the appropriate source for you.
The NCM source uses the
same `RobustFaceDetector`, model, multi-scale/backlight processing, strict
confidence and landmark gates, three-hit confirmation, and 10 FPS cap.

The board must expose the reference firmware's USB-NCM network protocol:

| Setting | Default |
| --- | --- |
| Windows USB-NCM adapter | `192.168.50.1/30` (mask `255.255.255.252`) |
| Camera board | `192.168.50.2` |
| JPEG stream | TCP `5000` |
| Discovery | UDP `5001`, `JL-CAMERA-DISCOVER` |

Assign/check that host address on the USB-NCM adapter, using Windows network
settings or the firmware's supported network configuration. The app does not
change any adapter settings. The camera is a USB network device; an OpenCV webcam
index is not used in NCM mode. Close other apps connected to the same board if
the firmware accepts only one streaming client.

Check the adapter and discover the board without starting inference:

```powershell
Get-NetIPAddress -AddressFamily IPv4 | Where-Object IPAddress -eq '192.168.50.1'
.venv\Scripts\python.exe live_face_detector.py --ncm-discover
```

Discovery prints JSON with the board reply or the bind/UDP error and exits. A
missing discovery reply does not prevent TCP streaming if the board's stream
service works. For alternate firmware settings:

```powershell
.venv\Scripts\python.exe live_face_detector.py --ncm --ncm-host-ip 192.168.50.1 --ncm-device-ip 192.168.50.2 --ncm-port 5000
```

Use `--ncm-host-ip auto` only if Windows already has the correct route. The
default explicitly binds the socket to the USB address to avoid selecting Wi-Fi.
`--ncm-discovery-port` changes the UDP port if necessary. `--ncm-rotate-180`
handles an upside-down mount. NCM frames are unmirrored by default, matching the
phone NCM path. `--mirror` enables a horizontal flip; `--no-mirror` disables the
usual mirror when using a normal webcam.
The NCM preview fits the original aspect ratio into a 1280×720 canvas so the HUD
also remains readable with small board images; inference keeps the native pixels.

Profile the NCM input through the real detector:

```powershell
.venv\Scripts\python.exe live_face_detector.py --ncm --profile-camera --preview --frames 60
```

Omit `--preview` for a headless profile. Diagnostics report the measured processing
FPS, received camera FPS, JPEG decoding time (`capture_read_ms` for this source),
detection time, age since local receipt, CRC/header errors, invalid JPEGs, decode
errors, sequence gaps, and reconnect count. Sensor/firmware latency is not
included in local frame age. A 10 FPS cap cannot force a slow board to deliver
10 frames/s.

The receiver verifies the 24-byte big-endian JLIP v1 header, the 200 KiB payload
limit, CRC32, JPEG markers and dimensions. It reassembles fragmented/coalesced
TCP reads and replies to heartbeat `0x03` with `0x03`, as the reference host does.
Only the latest validated JPEG is retained; JPEG decoding happens at consumption,
so discarded frames do not waste decoding time.

Connections automatically retry. **C** requests an immediate reconnect in NCM
mode; all other existing controls still work. Disconnects, resolution changes,
and new TCP sessions reset detection/tracking history. Frames older than 500 ms
are not selected; results older than 750 ms are suppressed, and a stalled preview
clears its face boxes. The initial frame wait and stalled headless profiles time
out after 10 seconds; change this with `--ncm-timeout 30` if necessary. After the
first frame, an interactive preview keeps reconnecting until you press Q/Esc.

`tests/test_ncm_camera.py` contains a loopback simulated board, including actual
JPEG decoding and a headless run through YuNet. It does not require physical
camera hardware. Real board/USB performance still needs a hardware run.

## Camera angles and pose recovery

640x480 USB-NCM streams now get an enlarged search in overlapping image tiles
when the initial scan misses a face. One tile alternates with pose recovery
while searching, rather than requiring an existing face proposal first. Stronger
shadow lifting and enlarged landmark-aligned inference address small, shadowed
faces against bright windows. Landmark coordinates retain subpixel precision
until drawing. A verified alignment remains steady; local template matching can
move a small search region with the face, but never supplies a detection itself.
All accepted results still require fresh YuNet confidence of at least **0.85**,
the existing geometry gates, and three consecutive hits before display.

`tests/benchmark_backlight.py` reports 640x480 JPEG recovery and latency, including
an extremely underexposed case that may remain undetected. The related regression
tests cover fresh consecutive hits, motion, disappearance, non-face/window
patterns, and the NCM receive/detect/display path at its 10 FPS cap. These are
synthetic tests; the exact physical low-camera/backlit view still needs a rerun.

The detector rechecks uncertain face proposals after aligning their eye/mouth
landmarks into an upright 192×224 view. This compensates moderate roll, pitch and
perspective foreshortening, including low-camera views. A candidate must pass a
new YuNet inference and geometry validation in that view before it can count as
a hit. A weak proposal itself never counts toward three-frame confirmation.

If no usable proposal exists, a bounded recovery search tries at most two rotated
or stretched views in one frame, cycling alternatives across frames. It starts
only when the earlier processing is under 70ms, and does not begin a second
alternative after 25ms of recovery work. A native inference cannot be interrupted;
these are scheduling limits rather than a guarantee every frame finishes in 100ms.
Ordinary accepted faces retain the fast path, and existing distant/backlit crop
checks remain active. Pose hints retain only an alignment/search region and are
verified on new pixels every frame; they are never returned as cached hits.

The normal score floor stays **0.85**. Direct detections with unusually
foreshortened landmark geometry require **0.94** plus additional mouth direction
and layout checks. Rotated/rectified detections are validated in their upright
view and mapped back using the exact inverse transform, preserving box/landmark
alignment in the preview. The 10 FPS cap and three-hit confirmation remain.

`tests/test_pose_detection.py` checks roll/inverted views, synthetic low/high
perspective and left/right distortions, tilted backlit JPEGs over NCM, fresh-hit
confirmation, stale-pose expiry, and non-face rejection over a full recovery
cycle. `tests/benchmark_pose.py` reports acquisition delay and latency. Synthetic
warps do not recreate real 3-D self-occlusion: an extreme profile with hidden
eyes/nose/mouth, the back of the head, motion blur, or a full silhouette may still
be missed. No every-angle or zero-false-positive guarantee is made; retest the
actual camera angle after restarting the process.

## Standard capture and display

The app exposes at most one selected face. It initially selects the largest
confirmed face, keeps that track while it remains visible, and automatically
selects another freshly confirmed face after the current target's 200ms dropout
hold expires. Held alternative faces cannot take over. Internal scanning still
finds other candidates for confirmation, but the overlay/count show zero or one.
Camera, resolution and session resets clear the lock. The profile JSON reports
`max_selected_faces` so this behavior can be checked without a preview window.

Native thread pools are bounded before importing OpenCV/NumPy: BLAS/OpenMP/MKL
use one thread and OpenCV uses two. Both the batch launcher and direct Python
entry points apply these process-local limits; no Windows settings are changed.
If startup still reports an OpenBLAS allocation failure or a DLL paging-file
error, Windows has insufficient committed-memory capacity. Close unnecessary
apps or enable a system-managed Windows paging file and restart as required.
The OpenCV 5 graph-engine target warning is separate from these startup failures.
Use Q/Esc in the preview to quit; Ctrl+C in a batch console can display Windows'
`Terminate batch job (Y/N)?` prompt.

The camera requests 1920 x 1080 at 30 FPS (actual rates depend on the camera).
MJPEG is requested **after** setting resolution, frame rate, and focus: DirectShow
can silently revert to slow uncompressed YUY2 if those settings are applied later.
Startup prints the negotiated resolution and codec. A dedicated capture reader
keeps only the latest frame, so a slow detection pass cannot create a video backlog.
Detection and preview updates are capped at **10 FPS**. The camera reader keeps
running at its native rate and discards old frames during the pacing wait, so the
next detection uses the newest available image. A slow frame never causes a
catch-up burst. The HUD shows the measured processing FPS and the 10 FPS limit.
The preview is prepared at up to 1280 pixels wide while detection retains the
original camera pixels. Deliberate motion uses faster tracking updates; small
bounding-box jitter remains smoothed.
The detector runs a 640-pixel image scan on every frame and a detailed scan up to
1280 pixels every fifth frame. Previously found faces are checked again using
fresh inference on small crops from the original camera pixels. New distant faces
may take up to five processed frames to acquire. Weak proposals are checked again using enlarged image crops;
they never appear unless the final score reaches 0.85. Dark regions receive a
bounded shadow lift and local contrast recovery, including against bright windows.
Five-point landmark geometry and three consecutive observations are required.
Turning off the tracking stabilizer disables smoothing, while confirmation stays on.

Temporarily lost faces have an amber box for at most 200 ms. They do not count as
current detections and do not display stale landmarks. The printed confidence is
a model score, not a measured accuracy percentage. Extremely small, blurred, or
fully silhouetted faces can still be missed; zero false positives is not guaranteed.

Verify dependencies/model inference with:

```powershell
.venv\Scripts\python.exe live_face_detector.py --check
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe tests\benchmark_detection.py --frames 60
.venv\Scripts\python.exe live_face_detector.py --profile-camera --preview --frames 60
```

The integration tests cover smaller face samples, simulated backlighting,
multiple faces, blank/noise/animal images, and temporal false-positive rejection.
They do not replace testing on the intended camera under real lighting.
The benchmark uses synthetic 1080p scenes and reports mean inference FPS and
95th-percentile latency. It excludes camera capture, display, and initial model
warm-up. Detailed searches and shadow recovery are scheduled; strict scores,
landmark checks, and three-frame confirmation remain active.

`--profile-camera --preview` measures the actual camera-to-preview loop, including
the camera reader, detector, overlays, and real window. It saves no camera images.
The on-screen FPS is measured from processing start intervals. Raw camera FPS is
reported separately in the diagnostic output; it can exceed the 10 FPS processing
limit. Inference time and the age since a frame was received remain visible.

Model: [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet),
`face_detection_yunet_2023mar.onnx`, SHA-256
`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
The model's MIT license is included in `YUNET_LICENSE`.
Test images come from OpenCV's sample data:
[lena.jpg](https://github.com/opencv/opencv/blob/master/samples/data/lena.jpg) and
[baboon.jpg](https://github.com/opencv/opencv/blob/master/samples/data/baboon.jpg).
