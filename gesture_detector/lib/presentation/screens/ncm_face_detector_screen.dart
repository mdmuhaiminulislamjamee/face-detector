import 'dart:async';

import 'package:flutter/material.dart';

import '../../models/face_detector_options_model.dart';
import '../../ncm/ncm_camera_client.dart';
import '../../services/ncm_face_detection_service.dart';
import '../painters/face_detection_painter.dart';
import '../widgets/ncm_jpeg_preview.dart';

class NcmFaceDetectorScreen extends StatefulWidget {
  const NcmFaceDetectorScreen({super.key});

  @override
  State<NcmFaceDetectorScreen> createState() => _NcmFaceDetectorScreenState();
}

class _NcmFaceDetectorScreenState extends State<NcmFaceDetectorScreen>
    with WidgetsBindingObserver {
  final _service = NcmFaceDetectionService();
  final _host = TextEditingController(text: '192.168.50.1');
  final _device = TextEditingController(text: '192.168.50.2');
  final _port = TextEditingController(text: '5000');
  final _form = GlobalKey<FormState>();
  bool _rotate180 = false, _showLandmarks = true;
  bool _requestedConnection = false, _discovering = false;
  String? _diagnostic;
  NcmCameraConfig? _activeConfig;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _service.addListener(_refresh);
  }

  void _refresh() {
    if (mounted) setState(() {});
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused ||
        state == AppLifecycleState.detached) {
      _service.stop();
    } else if (state == AppLifecycleState.resumed &&
        _requestedConnection &&
        _activeConfig != null) {
      unawaited(_service.start(_activeConfig!));
    }
  }

  NcmCameraConfig _config() => NcmCameraConfig(
    hostAddress: _host.text.trim(),
    deviceAddress: _device.text.trim(),
    tcpPort: int.parse(_port.text),
    rotate180: _rotate180,
  );

  void _connect() {
    if (!_form.currentState!.validate()) return;
    FocusScope.of(context).unfocus();
    _activeConfig = _config();
    _requestedConnection = true;
    unawaited(_service.start(_activeConfig!));
  }

  void _disconnect() {
    _requestedConnection = false;
    _service.stop();
  }

  Future<void> _discover() async {
    if (!_form.currentState!.validate()) return;
    setState(() => _discovering = true);
    try {
      final addresses = await NcmCameraClient.localAddresses();
      String discovery;
      try {
        discovery =
            'Camera reply: ${await NcmCameraClient.discover(_config())}';
      } catch (error) {
        discovery = 'Discovery failed: $error';
      }
      if (mounted) {
        setState(
          () => _diagnostic =
              '${addresses.isEmpty ? "No IPv4 interfaces found" : addresses.join("\n")}\n$discovery',
        );
      }
    } catch (error) {
      if (mounted) setState(() => _diagnostic = error.toString());
    } finally {
      if (mounted) setState(() => _discovering = false);
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _service.removeListener(_refresh);
    _service.dispose();
    _host.dispose();
    _device.dispose();
    _port.dispose();
    super.dispose();
  }

  String? _validateIp(String? text, {bool optional = false}) {
    final value = text?.trim() ?? '';
    if (optional && value.isEmpty) return null;
    final parts = value.split('.');
    if (parts.length != 4 ||
        parts.any(
          (p) =>
              int.tryParse(p) == null || int.parse(p) < 0 || int.parse(p) > 255,
        )) {
      return 'Enter an IPv4 address';
    }
    return null;
  }

  @override
  Widget build(BuildContext context) {
    final current = _service.frame;
    return Scaffold(
      appBar: AppBar(
        title: const Text('USB-NCM Face Detector'),
        actions: [
          if (_requestedConnection)
            IconButton(
              onPressed: _disconnect,
              tooltip: 'Disconnect camera',
              icon: const Icon(Icons.link_off),
            ),
        ],
      ),
      body: SafeArea(
        child: current == null
            ? _buildConnection()
            : Column(
                children: [
                  Expanded(
                    child: Center(
                      child: AspectRatio(
                        aspectRatio:
                            current.result.width / current.result.height,
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            NcmJpegPreview(
                              jpeg: current.source.jpeg,
                              rotate180: current.rotate180,
                            ),
                            CustomPaint(
                              painter: FaceDetectionPainter(
                                faces: current.faces,
                                config: FaceDetectorConfig(
                                  showMesh: false,
                                  showLandmarks: _showLandmarks,
                                ),
                              ),
                            ),
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
                          '${current.faces.where((face) => !face.isTemporarilyLost).length} faces  ·  '
                          '${_service.detectionFps.toStringAsFixed(1)} detection FPS  ·  limit 10',
                          textAlign: TextAlign.center,
                        ),
                        const SizedBox(height: 6),
                        Text(
                          'Camera ${_service.status.cameraFps.toStringAsFixed(1)} FPS  ·  '
                          'Detection ${current.result.duration.inMilliseconds} ms  ·  '
                          'Frame age ${_service.frameAge.inMilliseconds} ms',
                          textAlign: TextAlign.center,
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                        TextButton.icon(
                          onPressed: () =>
                              setState(() => _showLandmarks = !_showLandmarks),
                          icon: const Icon(Icons.scatter_plot),
                          label: Text(
                            _showLandmarks
                                ? 'Hide landmarks'
                                : 'Show landmarks',
                          ),
                        ),
                        _transportStats(),
                      ],
                    ),
                  ),
                ],
              ),
      ),
    );
  }

  Widget _transportStats() => Text(
    'CRC errors ${_service.status.crcErrors}  ·  Invalid JPEGs ${_service.status.invalidJpegs}'
    '  ·  Sequence gaps ${_service.status.sequenceGaps}',
    textAlign: TextAlign.center,
    style: Theme.of(context).textTheme.bodySmall,
  );

  Widget _buildConnection() => Center(
    child: SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 480),
        child: Form(
          key: _form,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Text(
                'Connect the camera through USB',
                style: TextStyle(fontSize: 22),
              ),
              const SizedBox(height: 12),
              const Text(
                'The phone must expose the board as a USB-NCM network interface. '
                'Use host 192.168.50.1/30 and board 192.168.50.2. Allow local-network access when prompted.',
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _host,
                enabled: !_requestedConnection,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: 'Phone USB IP',
                  helperText: 'Leave empty only to use automatic routing',
                ),
                validator: (v) => _validateIp(v, optional: true),
              ),
              TextFormField(
                controller: _device,
                enabled: !_requestedConnection,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(labelText: 'Camera IP'),
                validator: _validateIp,
              ),
              TextFormField(
                controller: _port,
                enabled: !_requestedConnection,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(labelText: 'TCP port'),
                validator: (v) {
                  final port = int.tryParse(v ?? '');
                  return port == null || port < 1 || port > 65535
                      ? 'Enter port 1–65535'
                      : null;
                },
              ),
              SwitchListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text('Camera mounted upside down'),
                value: _rotate180,
                onChanged: _requestedConnection
                    ? null
                    : (v) => setState(() => _rotate180 = v),
              ),
              const SizedBox(height: 12),
              FilledButton.icon(
                onPressed: _service.initializing
                    ? null
                    : _requestedConnection
                    ? _disconnect
                    : _connect,
                icon: Icon(_requestedConnection ? Icons.link_off : Icons.cable),
                label: Text(
                  _service.initializing
                      ? 'Loading detector…'
                      : _requestedConnection
                      ? 'Disconnect'
                      : 'Connect camera',
                ),
              ),
              OutlinedButton(
                onPressed: _discovering ? null : _discover,
                child: Text(
                  _discovering
                      ? 'Checking connection…'
                      : 'Discover / check USB network',
                ),
              ),
              const SizedBox(height: 12),
              Text(
                _service.initializing
                    ? 'Preparing YuNet on this phone'
                    : 'Camera: ${_service.status.state}',
                textAlign: TextAlign.center,
              ),
              if (_requestedConnection && _service.status.connected)
                const Text(
                  'Waiting for a fresh, validated JPEG frame…',
                  textAlign: TextAlign.center,
                ),
              if (_service.error != null)
                Padding(
                  padding: const EdgeInsets.only(top: 12),
                  child: Text(
                    _service.error!,
                    style: const TextStyle(color: Colors.orangeAccent),
                  ),
                ),
              if (_diagnostic != null)
                Padding(
                  padding: const EdgeInsets.only(top: 12),
                  child: SelectableText(_diagnostic!),
                ),
              const SizedBox(height: 12),
              _transportStats(),
            ],
          ),
        ),
      ),
    ),
  );
}
