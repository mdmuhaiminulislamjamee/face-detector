import 'package:flutter/material.dart';

/// Floating chip displaying the current number of detected faces and frame processing latency.
class FaceCountBadge extends StatelessWidget {
  const FaceCountBadge({
    super.key,
    required this.faceCount,
    required this.latency,
    this.isFrontCamera = true,
  });

  final int faceCount;
  final Duration latency;
  final bool isFrontCamera;

  @override
  Widget build(BuildContext context) {
    final hasFaces = faceCount > 0;
    final accentColor = hasFaces ? const Color(0xFF00E5FF) : Colors.white60;
    final latencyMs = latency.inMilliseconds;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0xCC1A1A1A),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: hasFaces ? const Color(0x6600E5FF) : Colors.white24,
          width: 1,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.3),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              color: hasFaces ? const Color(0xFF00E5FF) : Colors.grey,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 8),
          Text(
            hasFaces
                ? (faceCount == 1 ? '1 Face Detected' : '$faceCount Faces Detected')
                : 'Searching for faces...',
            style: TextStyle(
              color: accentColor,
              fontWeight: FontWeight.w600,
              fontSize: 13,
            ),
          ),
          if (latencyMs > 0) ...[
            const SizedBox(width: 10),
            Container(width: 1, height: 12, color: Colors.white24),
            const SizedBox(width: 10),
            Text(
              '${latencyMs}ms',
              style: const TextStyle(
                color: Colors.white54,
                fontSize: 12,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ],
      ),
    );
  }
}

