import 'package:flutter/material.dart';

import 'face_detector_screen.dart';
import 'ncm_face_detector_screen.dart';

class CameraSourceScreen extends StatelessWidget {
  const CameraSourceScreen({super.key});

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('Face Detector')),
    body: Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 480),
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Icon(
                Icons.face_retouching_natural,
                size: 64,
                color: Color(0xFF00E5FF),
              ),
              const SizedBox(height: 24),
              const Text('Choose your camera', style: TextStyle(fontSize: 24)),
              const SizedBox(height: 12),
              const Text(
                'USB-NCM uses the same YuNet face model and detection pipeline as the PC version. Processing runs on this phone.',
              ),
              const SizedBox(height: 24),
              FilledButton.icon(
                icon: const Icon(Icons.cable),
                label: const Text('USB-NCM camera'),
                onPressed: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (_) => const NcmFaceDetectorScreen(),
                  ),
                ),
              ),
              const SizedBox(height: 12),
              OutlinedButton.icon(
                icon: const Icon(Icons.camera_alt_outlined),
                label: const Text('Phone camera'),
                onPressed: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (_) => const FaceDetectorScreen(),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    ),
  );
}
