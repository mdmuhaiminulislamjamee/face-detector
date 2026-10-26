import 'dart:typed_data';

import 'package:flutter/material.dart';

/// Evict old frames explicitly; a live stream must not fill the global image cache.
class NcmJpegPreview extends StatefulWidget {
  const NcmJpegPreview({
    super.key,
    required this.jpeg,
    required this.rotate180,
  });
  final Uint8List jpeg;
  final bool rotate180;

  @override
  State<NcmJpegPreview> createState() => _NcmJpegPreviewState();
}

class _NcmJpegPreviewState extends State<NcmJpegPreview> {
  late MemoryImage _image = MemoryImage(widget.jpeg);

  @override
  void didUpdateWidget(NcmJpegPreview oldWidget) {
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
      filterQuality: FilterQuality.low,
      errorBuilder: (_, error, _) =>
          const Center(child: Text('JPEG preview unavailable')),
    ),
  );
}
