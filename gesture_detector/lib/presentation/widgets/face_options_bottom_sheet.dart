import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

import '../../models/face_detector_options_model.dart';

/// Modal bottom sheet allowing users to toggle face detector features in real-time.
class FaceOptionsBottomSheet extends StatelessWidget {
  const FaceOptionsBottomSheet({
    super.key,
    required this.config,
    required this.onConfigChanged,
  });

  final FaceDetectorConfig config;
  final ValueChanged<FaceDetectorConfig> onConfigChanged;

  static Future<void> show(
    BuildContext context, {
    required FaceDetectorConfig config,
    required ValueChanged<FaceDetectorConfig> onConfigChanged,
  }) {
    return showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: const Color(0xFF1E1E1E),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) => DraggableScrollableSheet(
        initialChildSize: 0.65,
        minChildSize: 0.4,
        maxChildSize: 0.9,
        expand: false,
        builder: (_, scrollController) => SingleChildScrollView(
          controller: scrollController,
          child: FaceOptionsBottomSheet(
            config: config,
            onConfigChanged: onConfigChanged,
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Container(
                width: 40,
                height: 4,
                margin: const EdgeInsets.only(bottom: 16),
                decoration: BoxDecoration(
                  color: Colors.white24,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
            const Text(
              'Face Detector Settings',
              style: TextStyle(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),

            // Performance Mode Toggle (Accurate by default)
            ListTile(
              contentPadding: EdgeInsets.zero,
              leading: const Icon(Icons.speed, color: Color(0xFF00E5FF)),
              title: const Text(
                'High Accuracy Mode',
                style: TextStyle(color: Colors.white),
              ),
              subtitle: Text(
                config.performanceMode == FaceDetectorMode.accurate
                    ? 'Accurate (Prioritizes detection accuracy)'
                    : 'Fast (Lower CPU usage, basic detection)',
                style: const TextStyle(color: Colors.white60, fontSize: 12),
              ),
              trailing: Switch(
                value: config.performanceMode == FaceDetectorMode.accurate,
                activeThumbColor: const Color(0xFF00E5FF),
                onChanged: (isAccurate) {
                  onConfigChanged(
                    config.copyWith(
                      performanceMode: isAccurate
                          ? FaceDetectorMode.accurate
                          : FaceDetectorMode.fast,
                    ),
                  );
                },
              ),
            ),
            const Divider(color: Colors.white12),

            // Tracking Stabilizer Toggle
            _buildSwitchTile(
              icon: Icons.auto_awesome,
              title: 'Tracking Stabilizer',
              subtitle:
                  'Smooths jitter & holds target lock during fast motion or blinks',
              value: config.enableStabilizer,
              onChanged: (val) =>
                  onConfigChanged(config.copyWith(enableStabilizer: val)),
            ),

            // Far Face Distance Sensitivity
            ListTile(
              contentPadding: EdgeInsets.zero,
              leading: const Icon(Icons.zoom_out_map, color: Color(0xFF00E5FF)),
              title: const Text(
                'Distance Sensitivity',
                style: TextStyle(color: Colors.white),
              ),
              subtitle: Text(
                config.minFaceSize <= 0.03
                    ? 'Search smaller faces (requires sufficient detail)'
                    : (config.minFaceSize <= 0.06
                          ? 'Balanced search for nearby and distant faces'
                          : 'Prioritize close-up faces'),
                style: const TextStyle(color: Colors.white60, fontSize: 12),
              ),
              trailing: DropdownButton<double>(
                value: config.minFaceSize <= 0.03
                    ? 0.03
                    : (config.minFaceSize <= 0.06 ? 0.05 : 0.15),
                dropdownColor: const Color(0xFF2A2A2A),
                style: const TextStyle(color: Color(0xFF00E5FF), fontSize: 13),
                underline: const SizedBox(),
                items: const [
                  DropdownMenuItem(
                    value: 0.03,
                    child: Text('Small faces (3%)'),
                  ),
                  DropdownMenuItem(
                    value: 0.05,
                    child: Text('High/Balanced (5%)'),
                  ),
                  DropdownMenuItem(value: 0.15, child: Text('Close-up (15%)')),
                ],
                onChanged: (val) {
                  if (val != null) {
                    onConfigChanged(config.copyWith(minFaceSize: val));
                  }
                },
              ),
            ),
            const Divider(color: Colors.white12),

            // Bounding Box Toggle
            _buildSwitchTile(
              icon: Icons.crop_free,
              title: 'Bounding Boxes',
              subtitle: 'Show glowing border and Face ID badge',
              value: config.showBoundingBoxes,
              onChanged: (val) =>
                  onConfigChanged(config.copyWith(showBoundingBoxes: val)),
            ),

            // Contours / Mesh Toggle
            _buildSwitchTile(
              icon: Icons.grain,
              title: 'Facial Contours (Mesh)',
              subtitle: 'Draw lines tracing eyes, eyebrows, nose, and lips',
              value: config.showMesh,
              onChanged: (val) =>
                  onConfigChanged(config.copyWith(showMesh: val)),
            ),

            // Landmarks Toggle
            _buildSwitchTile(
              icon: Icons.scatter_plot,
              title: 'Facial Landmarks',
              subtitle:
                  'Draw key landmark points (eyes, ears, nose base, mouth)',
              value: config.showLandmarks,
              onChanged: (val) =>
                  onConfigChanged(config.copyWith(showLandmarks: val)),
            ),

            // Metrics Toggle
            _buildSwitchTile(
              icon: Icons.sentiment_satisfied_alt,
              title: 'Facial Metrics',
              subtitle:
                  'Show smile probability, eye open status, and head tilt',
              value: config.showMetrics,
              onChanged: (val) =>
                  onConfigChanged(config.copyWith(showMetrics: val)),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSwitchTile({
    required IconData icon,
    required String title,
    required String subtitle,
    required bool value,
    required ValueChanged<bool> onChanged,
  }) {
    return ListTile(
      contentPadding: EdgeInsets.zero,
      leading: Icon(icon, color: const Color(0xFF00E5FF)),
      title: Text(title, style: const TextStyle(color: Colors.white)),
      subtitle: Text(
        subtitle,
        style: const TextStyle(color: Colors.white60, fontSize: 12),
      ),
      trailing: Switch(
        value: value,
        activeThumbColor: const Color(0xFF00E5FF),
        onChanged: onChanged,
      ),
    );
  }
}
