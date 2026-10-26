import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import '../../models/detected_face.dart';
import '../../models/face_detector_options_model.dart';
import '../../services/face_detector_service.dart';
import '../../utils/face_geometry_utils.dart';
import '../painters/face_detection_painter.dart';
import '../widgets/camera_loading_view.dart';
import '../widgets/face_count_badge.dart';
import '../widgets/face_options_bottom_sheet.dart';

/// Full-screen live camera feed running real-time face detection with overlays.
class FaceDetectorScreen extends StatefulWidget {
  const FaceDetectorScreen({
    super.key,
    this.initialLensDirection = CameraLensDirection.front,
    this.autoStartCamera = true,
  });

  final CameraLensDirection initialLensDirection;
  final bool autoStartCamera;

  @override
  State<FaceDetectorScreen> createState() => _FaceDetectorScreenState();
}

class _FaceDetectorScreenState extends State<FaceDetectorScreen>
    with WidgetsBindingObserver {
  late final FaceDetectorService _detectorService;
  FaceDetectorConfig _config = const FaceDetectorConfig();

  CameraController? _cameraController;
  List<CameraDescription> _availableCameras = const [];
  int _selectedCameraIndex = 0;

  bool _isCameraSetupInProgress = false;
  bool _isCameraInitialized = false;
  bool _isStreaming = false;
  bool _isTorchOn = false;

  String _statusTitle = 'Initializing camera...';
  String _statusMessage = 'Preparing real-time face detection.';
  String? _actionLabel;
  VoidCallback? _onAction;

  List<DetectedFace> _detectedFaces = const [];
  DateTime? _lastFrameProcessedAt;
  static const _minFrameInterval = Duration(milliseconds: 100); // 10 FPS cap

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _detectorService = FaceDetectorService(config: _config);

    if (widget.autoStartCamera) {
      unawaited(_setupCamera());
    } else {
      _statusTitle = 'Face Detector';
      _statusMessage = 'Camera startup is ready.';
      _actionLabel = 'Start Camera';
      _onAction = () => unawaited(_setupCamera());
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) return;

    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      unawaited(_cleanupCamera());
    } else if (state == AppLifecycleState.resumed) {
      unawaited(_setupCamera());
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    unawaited(_cleanupCamera());
    unawaited(_detectorService.dispose());
    super.dispose();
  }

  Future<void> _setupCamera() async {
    if (_isCameraSetupInProgress) return;
    _isCameraSetupInProgress = true;

    _setLoading(
      title: 'Initializing camera...',
      message: 'Requesting camera access for face detection.',
    );

    try {
      var status = await Permission.camera.status;
      if (status.isDenied) {
        status = await Permission.camera.request();
      }

      if (!status.isGranted) {
        final permanentlyDenied = status.isPermanentlyDenied;
        _setError(
          title: permanentlyDenied
              ? 'Camera Access Needed'
              : 'Permission Denied',
          message: permanentlyDenied
              ? 'Enable camera permission in device settings to detect faces.'
              : 'Allow camera access to use live face detection.',
          actionLabel: permanentlyDenied ? 'Open Settings' : 'Try Again',
          onAction: permanentlyDenied
              ? () => openAppSettings()
              : () => unawaited(_setupCamera()),
        );
        return;
      }

      _availableCameras = await availableCameras();
      if (_availableCameras.isEmpty) {
        _setError(
          title: 'No Camera Found',
          message: 'This device does not report any available cameras.',
          actionLabel: 'Retry',
          onAction: () => unawaited(_setupCamera()),
        );
        return;
      }

      // Pick matching lens direction or first camera
      final index = _availableCameras.indexWhere(
        (cam) => cam.lensDirection == widget.initialLensDirection,
      );
      _selectedCameraIndex = index >= 0 ? index : 0;

      await _initController(_availableCameras[_selectedCameraIndex]);
    } catch (error, stackTrace) {
      debugPrint('Camera setup failed: $error\n$stackTrace');
      _setError(
        title: 'Camera Error',
        message: 'Could not open the camera: $error',
        actionLabel: 'Retry',
        onAction: () => unawaited(_setupCamera()),
      );
    } finally {
      _isCameraSetupInProgress = false;
      if (mounted) setState(() {});
    }
  }

  Future<void> _initController(CameraDescription camera) async {
    await _cleanupCamera();

    final controller = CameraController(
      camera,
      ResolutionPreset.veryHigh,
      enableAudio: false,
      imageFormatGroup: Platform.isIOS
          ? ImageFormatGroup.bgra8888
          : Platform.isAndroid
          ? ImageFormatGroup.yuv420
          : ImageFormatGroup.bgra8888,
    );

    _cameraController = controller;
    await controller.initialize();

    // Use the best exposure/focus supported by this camera before streaming.
    try {
      await controller.setExposureMode(ExposureMode.auto);
    } on CameraException catch (error) {
      debugPrint('Auto exposure unavailable: $error');
    }
    try {
      await controller.setFocusMode(FocusMode.auto);
    } on CameraException catch (error) {
      debugPrint('Auto focus unavailable: $error');
    }

    if (!mounted) {
      await controller.dispose();
      return;
    }

    setState(() {
      _isCameraInitialized = true;
      _isTorchOn = false;
      _detectedFaces = const [];
    });

    await _startImageStream();
  }

  Future<void> _startImageStream() async {
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized || _isStreaming) {
      return;
    }

    try {
      await controller.startImageStream(_handleCameraFrame);
      if (mounted) setState(() => _isStreaming = true);
    } catch (error) {
      debugPrint('Start stream failed: $error');
    }
  }

  Future<void> _stopImageStream() async {
    final controller = _cameraController;
    if (controller == null ||
        !controller.value.isInitialized ||
        !_isStreaming) {
      return;
    }

    try {
      await controller.stopImageStream();
    } catch (error) {
      debugPrint('Stop stream failed: $error');
    } finally {
      _isStreaming = false;
    }
  }

  Future<void> _cleanupCamera() async {
    _detectorService.clearTracking();
    _lastFrameProcessedAt = null;
    await _stopImageStream();
    final controller = _cameraController;
    _cameraController = null;
    if (controller != null) {
      try {
        await controller.dispose();
      } catch (error) {
        debugPrint('Controller dispose failed: $error');
      }
    }

    if (mounted) {
      setState(() {
        _isCameraInitialized = false;
        _detectedFaces = const [];
      });
    }
  }

  void _handleCameraFrame(CameraImage image) {
    final controller = _cameraController;
    if (controller == null ||
        !controller.value.isInitialized ||
        !mounted ||
        _detectorService.isProcessing) {
      return;
    }

    final now = DateTime.now();
    final lastTime = _lastFrameProcessedAt;
    if (lastTime != null && now.difference(lastTime) < _minFrameInterval) {
      return;
    }
    _lastFrameProcessedAt = now;

    unawaited(
      _detectorService
          .processFrame(
            image: image,
            camera: controller.description,
            deviceOrientation: controller.value.deviceOrientation,
          )
          .then((faces) {
            if (!mounted || !identical(controller, _cameraController)) return;
            setState(() {
              _detectedFaces = faces;
            });
          }),
    );
  }

  Future<void> _switchCamera() async {
    if (_availableCameras.length < 2 || _isCameraSetupInProgress) return;
    _selectedCameraIndex =
        (_selectedCameraIndex + 1) % _availableCameras.length;
    _detectorService.clearTracking();
    await _initController(_availableCameras[_selectedCameraIndex]);
  }

  Future<void> _toggleTorch() async {
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) return;

    try {
      if (_isTorchOn) {
        await controller.setFlashMode(FlashMode.off);
      } else {
        await controller.setFlashMode(FlashMode.torch);
      }
      if (mounted) {
        setState(() => _isTorchOn = !_isTorchOn);
      }
    } catch (error) {
      debugPrint('Torch toggle failed: $error');
    }
  }

  void _onConfigUpdated(FaceDetectorConfig newConfig) {
    setState(() => _config = newConfig);
    _detectorService.updateConfig(newConfig);
  }

  void _setLoading({required String title, required String message}) {
    if (!mounted) return;
    setState(() {
      _statusTitle = title;
      _statusMessage = message;
      _actionLabel = null;
      _onAction = null;
    });
  }

  void _setError({
    required String title,
    required String message,
    required String actionLabel,
    required VoidCallback onAction,
  }) {
    if (!mounted) return;
    setState(() {
      _statusTitle = title;
      _statusMessage = message;
      _actionLabel = actionLabel;
      _onAction = onAction;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (!_isCameraInitialized || _cameraController == null) {
      return Scaffold(
        backgroundColor: const Color(0xFF121212),
        body: CameraLoadingView(
          title: _statusTitle,
          message: _statusMessage,
          actionLabel: _actionLabel,
          onAction: _onAction,
          isLoading: _isCameraSetupInProgress,
        ),
      );
    }

    final controller = _cameraController!;
    final isFrontCamera =
        controller.description.lensDirection == CameraLensDirection.front;
    final isLandscape =
        MediaQuery.orientationOf(context) == Orientation.landscape;
    final previewDisplaySize = orientedPreviewSize(
      rawPreviewSize: controller.value.previewSize,
      isLandscape: isLandscape,
    );

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // 1. Camera live preview with aspect ratio preservation
          Center(
            child: AspectRatio(
              aspectRatio: previewDisplaySize.width / previewDisplaySize.height,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  CameraPreview(controller),
                  // 2. Real-time Face Detection overlay painter
                  CustomPaint(
                    painter: FaceDetectionPainter(
                      faces: _detectedFaces,
                      config: _config,
                    ),
                  ),
                ],
              ),
            ),
          ),

          // 3. Top Action Bar
          SafeArea(
            child: Align(
              alignment: Alignment.topCenter,
              child: Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 8,
                ),
                child: Row(
                  children: [
                    // App Logo / Title
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 6,
                      ),
                      decoration: BoxDecoration(
                        color: const Color(0xCC000000),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(color: Colors.white24, width: 1),
                      ),
                      child: const Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            Icons.face_retouching_natural,
                            color: Color(0xFF00E5FF),
                            size: 18,
                          ),
                          SizedBox(width: 8),
                          Text(
                            'Face Detector',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 14,
                              fontWeight: FontWeight.bold,
                              letterSpacing: 0.5,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const Spacer(),

                    // Flashlight toggle (only for back camera)
                    if (!isFrontCamera)
                      _buildHeaderButton(
                        icon: _isTorchOn ? Icons.flash_on : Icons.flash_off,
                        color: _isTorchOn
                            ? const Color(0xFFFFD700)
                            : Colors.white,
                        onPressed: _toggleTorch,
                      ),
                    const SizedBox(width: 8),

                    // Camera flip (front / back)
                    if (_availableCameras.length > 1)
                      _buildHeaderButton(
                        icon: Icons.flip_camera_ios,
                        onPressed: _switchCamera,
                      ),
                    const SizedBox(width: 8),

                    // Settings bottom sheet
                    _buildHeaderButton(
                      icon: Icons.tune,
                      onPressed: () => FaceOptionsBottomSheet.show(
                        context,
                        config: _config,
                        onConfigChanged: _onConfigUpdated,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),

          // 4. Bottom status chip and quick toggles
          SafeArea(
            child: Align(
              alignment: Alignment.bottomCenter,
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    // Live Face Count & Latency Badge
                    FaceCountBadge(
                      faceCount: _detectedFaces
                          .where((face) => !face.isTemporarilyLost)
                          .length,
                      latency: _detectorService.lastProcessingDuration,
                      isFrontCamera: isFrontCamera,
                    ),
                    const SizedBox(height: 16),

                    // Quick Toggle Bar
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 8,
                      ),
                      decoration: BoxDecoration(
                        color: const Color(0xCC000000),
                        borderRadius: BorderRadius.circular(30),
                        border: Border.all(color: Colors.white24, width: 1),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          // Toggle Mesh
                          _buildQuickToggle(
                            icon: Icons.grain,
                            label: 'Mesh',
                            isActive: _config.showMesh,
                            onTap: () => _onConfigUpdated(
                              _config.copyWith(showMesh: !_config.showMesh),
                            ),
                          ),
                          const SizedBox(width: 16),

                          // Toggle Landmarks
                          _buildQuickToggle(
                            icon: Icons.scatter_plot,
                            label: 'Landmarks',
                            isActive: _config.showLandmarks,
                            onTap: () => _onConfigUpdated(
                              _config.copyWith(
                                showLandmarks: !_config.showLandmarks,
                              ),
                            ),
                          ),
                          const SizedBox(width: 16),

                          // Toggle Metrics (Smile/Eyes)
                          _buildQuickToggle(
                            icon: Icons.sentiment_satisfied_alt,
                            label: 'Metrics',
                            isActive: _config.showMetrics,
                            onTap: () => _onConfigUpdated(
                              _config.copyWith(
                                showMetrics: !_config.showMetrics,
                              ),
                            ),
                          ),
                          if (_availableCameras.length > 1) ...[
                            const SizedBox(width: 16),
                            Container(
                              width: 1,
                              height: 24,
                              color: Colors.white24,
                            ),
                            const SizedBox(width: 16),
                            // Quick Switch Camera
                            _buildQuickToggle(
                              icon: Icons.cameraswitch_outlined,
                              label: isFrontCamera ? 'Front' : 'Back',
                              isActive: true,
                              onTap: _switchCamera,
                            ),
                          ],
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeaderButton({
    required IconData icon,
    required VoidCallback onPressed,
    Color color = Colors.white,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xCC000000),
        shape: BoxShape.circle,
        border: Border.all(color: Colors.white24, width: 1),
      ),
      child: IconButton(
        icon: Icon(icon, color: color, size: 20),
        onPressed: onPressed,
        constraints: const BoxConstraints(minWidth: 40, minHeight: 40),
        padding: EdgeInsets.zero,
      ),
    );
  }

  Widget _buildQuickToggle({
    required IconData icon,
    required String label,
    required bool isActive,
    required VoidCallback onTap,
  }) {
    final activeColor = const Color(0xFF00E5FF);
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(16),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 4),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              size: 20,
              color: isActive ? activeColor : Colors.white54,
            ),
            const SizedBox(height: 2),
            Text(
              label,
              style: TextStyle(
                fontSize: 11,
                color: isActive ? activeColor : Colors.white54,
                fontWeight: isActive ? FontWeight.bold : FontWeight.normal,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
