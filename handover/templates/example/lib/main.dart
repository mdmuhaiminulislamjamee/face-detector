import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:ncm_face_sdk/ncm_face_sdk.dart';

void main() => runApp(const MaterialApp(home: NcmAcceptanceScreen()));

class NcmAcceptanceScreen extends StatefulWidget {
  const NcmAcceptanceScreen({super.key});

  @override
  State<NcmAcceptanceScreen> createState() => _NcmAcceptanceScreenState();
}

class _NcmAcceptanceScreenState extends State<NcmAcceptanceScreen> {
  late final NcmFaceDetector _detector;
  final _host = TextEditingController(text: '192.168.50.1');
  final _device = TextEditingController(text: '192.168.50.2');
  String? _message;
  bool _rotate180 = false;
  int? _lastFrameId;

  @override
  void initState() {
    super.initState();
    _detector = NcmFaceDetector()..addListener(_refresh);
  }

  void _refresh() {
    final frame = _detector.frame;
    if (frame != null && frame.frameId != _lastFrameId) {
      _lastFrameId = frame.frameId;
      // One machine-readable line per NEW processed frame, for acceptance logs.
      debugPrint(
        'NCM_RESULT ${frame.frameId} '
        'fresh=${frame.freshFace == null ? 0 : 1} '
        'selected=${frame.faces.length} '
        'track=${frame.faces.isEmpty ? "none" : frame.faces.single.trackingId} '
        'held=${frame.faces.isNotEmpty && frame.faces.single.isTemporarilyLost} '
        'detection_fps=${_detector.detectionFps.toStringAsFixed(2)} '
        'camera_fps=${_detector.status.cameraFps.toStringAsFixed(2)} '
        'processing_ms=${frame.processingDuration.inMilliseconds} '
        'age_ms=${_detector.frameAge.inMilliseconds}',
      );
    }
    if (mounted) setState(() {});
  }

  NcmCameraConfig get _config => NcmCameraConfig(
    hostAddress: _host.text.trim(),
    deviceAddress: _device.text.trim(),
    rotate180: _rotate180,
  );

  Future<void> _connect() async {
    try {
      await _detector.start(_config);
    } catch (error) {
      if (mounted) setState(() => _message = '$error');
    }
  }

  Future<void> _discover() async {
    try {
      final addresses = await NcmFaceDetector.localAddresses();
      final reply = await NcmFaceDetector.discover(_config);
      if (mounted) setState(() => _message = '${addresses.join("\n")}\n$reply');
    } catch (error) {
      if (mounted) setState(() => _message = '$error');
    }
  }

  @override
  void dispose() {
    _detector.removeListener(_refresh);
    _detector.dispose();
    _host.dispose();
    _device.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final frame = _detector.frame;
    return Scaffold(
      appBar: AppBar(title: const Text('NCM single-face acceptance')),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: Center(
                child: frame == null
                    ? const Text('Connect the USB-NCM camera to begin')
                    : AspectRatio(
                        aspectRatio: frame.width / frame.height,
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            _FrameImage(
                              jpeg: frame.jpeg,
                              rotate180: frame.rotate180,
                            ),
                            CustomPaint(painter: _FacePainter(frame)),
                          ],
                        ),
                      ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                children: [
                  Text(
                    'State: ${_detector.status.state} | '
                    'Fresh faces: ${frame?.freshFace == null ? 0 : 1} | limit 10 FPS',
                  ),
                  Text(
                    'Detection ${_detector.detectionFps.toStringAsFixed(1)} FPS | '
                    'Camera ${_detector.status.cameraFps.toStringAsFixed(1)} FPS | '
                    'Age ${_detector.frameAge.inMilliseconds} ms',
                  ),
                  if (_detector.error != null) Text(_detector.error!),
                  if (_message != null) SelectableText(_message!),
                  Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: _host,
                          decoration: const InputDecoration(
                            labelText: 'Phone USB IPv4 (blank = OS route)',
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: TextField(
                          controller: _device,
                          decoration: const InputDecoration(
                            labelText: 'Camera IPv4',
                          ),
                        ),
                      ),
                    ],
                  ),
                  SwitchListTile(
                    title: const Text('Upside-down camera mount'),
                    value: _rotate180,
                    onChanged: (value) => setState(() => _rotate180 = value),
                  ),
                  Wrap(
                    spacing: 12,
                    children: [
                      FilledButton(
                        onPressed: _detector.initializing
                            ? null
                            : () => unawaited(_connect()),
                        child: const Text('Connect'),
                      ),
                      OutlinedButton(
                        onPressed: _detector.stop,
                        child: const Text('Disconnect'),
                      ),
                      TextButton(
                        onPressed: () => unawaited(_discover()),
                        child: const Text('Discover'),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _FrameImage extends StatefulWidget {
  const _FrameImage({required this.jpeg, required this.rotate180});
  final Uint8List jpeg;
  final bool rotate180;
  @override
  State<_FrameImage> createState() => _FrameImageState();
}

class _FrameImageState extends State<_FrameImage> {
  late MemoryImage _image;
  @override
  void initState() {
    super.initState();
    _image = MemoryImage(widget.jpeg);
  }

  @override
  void didUpdateWidget(_FrameImage oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (!identical(widget.jpeg, oldWidget.jpeg)) {
      _image.evict();
      _image = MemoryImage(widget.jpeg);
    }
  }

  @override
  void dispose() {
    _image.evict();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => RotatedBox(
    quarterTurns: widget.rotate180 ? 2 : 0,
    child: Image(
      image: _image,
      fit: BoxFit.fill,
      gaplessPlayback: false,
      errorBuilder: (_, error, _) =>
          const Center(child: Text('Invalid preview JPEG')),
    ),
  );
}

class _FacePainter extends CustomPainter {
  _FacePainter(this.frame);
  final NcmFaceFrame frame;
  @override
  void paint(Canvas canvas, Size size) {
    for (final face in frame.faces) {
      final paint = Paint()
        ..color = face.isTemporarilyLost ? Colors.amber : Colors.cyan
        ..strokeWidth = 2
        ..style = PaintingStyle.stroke;
      final box = face.displayBox;
      canvas.drawRect(
        Rect.fromLTRB(
          box.left * size.width,
          box.top * size.height,
          box.right * size.width,
          box.bottom * size.height,
        ),
        paint,
      );
      if (!face.isTemporarilyLost) {
        paint.style = PaintingStyle.fill;
        for (final point in face.displayLandmarks.values) {
          canvas.drawCircle(
            Offset(point.dx * size.width, point.dy * size.height),
            2,
            paint,
          );
        }
      }
    }
  }

  @override
  bool shouldRepaint(_FacePainter oldDelegate) => oldDelegate.frame != frame;
}
