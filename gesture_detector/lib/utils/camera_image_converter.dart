import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart'
    as ml_face;

import 'shadow_recovery.dart';

/// Transferable plane data extracted from a [CameraImage] for background isolate encoding.
class CameraPixelPlaneData {
  const CameraPixelPlaneData({
    required this.bytes,
    required this.bytesPerRow,
    required this.bytesPerPixel,
  });

  final Uint8List bytes;
  final int bytesPerRow;
  final int bytesPerPixel;
}

/// Standalone container of camera frame data that can be safely passed to an isolate.
class CameraPixelFrameData {
  const CameraPixelFrameData({
    required this.width,
    required this.height,
    required this.planes,
  });

  final int width;
  final int height;
  final List<CameraPixelPlaneData> planes;

  factory CameraPixelFrameData.fromCameraImage(CameraImage image) {
    return CameraPixelFrameData(
      width: image.width,
      height: image.height,
      planes: [
        for (final plane in image.planes)
          CameraPixelPlaneData(
            bytes: plane.bytes,
            bytesPerRow: plane.bytesPerRow,
            bytesPerPixel: plane.bytesPerPixel ?? 1,
          ),
      ],
    );
  }
}

/// One reusable worker per camera service, rather than spawning an isolate for
/// every camera frame. Full-resolution pixels are preserved for distant faces.
class CameraFrameProcessor {
  Future<(Isolate, SendPort)>? _worker;
  bool _disposed = false;
  int _active = 0;
  Completer<void>? _idle;

  Future<Uint8List?> process(
    CameraPixelFrameData frame, {
    bool bgra = false,
  }) async {
    if (_disposed) throw StateError('Camera frame processor is disposed');
    if (_active++ == 0) _idle = Completer<void>();
    final reply = ReceivePort();
    try {
      final (_, port) = await (_worker ??= _start());
      port.send((reply.sendPort, frame, bgra));
      final result = await reply.first as (TransferableTypedData?, String?);
      if (result.$2 != null) throw StateError(result.$2!);
      return result.$1?.materialize().asUint8List();
    } finally {
      reply.close();
      if (--_active == 0) _idle?.complete();
    }
  }

  Future<(Isolate, SendPort)> _start() async {
    final ready = ReceivePort();
    try {
      final isolate = await Isolate.spawn(_cameraFrameWorker, ready.sendPort);
      final port = await ready.first as SendPort;
      return (isolate, port);
    } finally {
      ready.close();
    }
  }

  Future<void> dispose() async {
    _disposed = true;
    if (_active > 0) await _idle?.future;
    final worker = await _worker;
    worker?.$1.kill(priority: Isolate.immediate);
    _worker = null;
  }
}

void _cameraFrameWorker(SendPort ready) {
  final requests = ReceivePort();
  ready.send(requests.sendPort);
  requests.listen((dynamic request) {
    final (reply, frame, bgra) =
        request as (SendPort, CameraPixelFrameData, bool);
    try {
      final Uint8List? bytes;
      if (bgra) {
        final plane = frame.planes.single;
        bytes = recoverBgraShadows(
          plane.bytes,
          frame.width,
          frame.height,
          plane.bytesPerRow,
        );
      } else {
        bytes = encodeNv21(frame, recoverShadows: true);
      }
      reply.send((
        bytes == null ? null : TransferableTypedData.fromList([bytes]),
        null,
      ));
    } catch (error) {
      reply.send((null, error.toString()));
    }
  });
}

/// Packs an Android YUV420 camera frame into the NV21 byte layout expected by ML Kit.
Uint8List? encodeNv21(
  CameraPixelFrameData frame, {
  bool recoverShadows = false,
}) {
  if (frame.planes.length < 3 ||
      frame.width <= 0 ||
      frame.height <= 0 ||
      frame.width.isOdd ||
      frame.height.isOdd) {
    return null;
  }

  final width = frame.width;
  final height = frame.height;
  final yPlane = frame.planes[0];
  final uPlane = frame.planes[1];
  final vPlane = frame.planes[2];
  if (!_validPlane(yPlane, width, height) ||
      !_validPlane(uPlane, width ~/ 2, height ~/ 2) ||
      !_validPlane(vPlane, width ~/ 2, height ~/ 2)) {
    return null;
  }
  final ySize = width * height;
  final bytes = Uint8List(ySize + (width * height ~/ 2));

  if (yPlane.bytesPerPixel == 1) {
    if (yPlane.bytesPerRow == width) {
      bytes.setRange(0, ySize, yPlane.bytes);
    } else {
      for (var row = 0; row < height; row++) {
        bytes.setRange(
          row * width,
          (row + 1) * width,
          yPlane.bytes,
          row * yPlane.bytesPerRow,
        );
      }
    }
  } else {
    for (var row = 0; row < height; row++) {
      var source = row * yPlane.bytesPerRow;
      final end = (row + 1) * width;
      for (var out = row * width; out < end; out++) {
        bytes[out] = yPlane.bytes[source];
        source += yPlane.bytesPerPixel;
      }
    }
  }

  for (var row = 0; row < height ~/ 2; row++) {
    var uIndex = row * uPlane.bytesPerRow;
    var vIndex = row * vPlane.bytesPerRow;
    var outIndex = ySize + row * width;
    for (var col = 0; col < width ~/ 2; col++) {
      bytes[outIndex++] = vPlane.bytes[vIndex];
      bytes[outIndex++] = uPlane.bytes[uIndex];
      vIndex += vPlane.bytesPerPixel;
      uIndex += uPlane.bytesPerPixel;
    }
  }

  if (recoverShadows) recoverNv21Shadows(bytes, width, height);
  return bytes;
}

bool _validPlane(CameraPixelPlaneData plane, int columns, int rows) =>
    plane.bytesPerPixel > 0 &&
    plane.bytesPerRow >= (columns - 1) * plane.bytesPerPixel + 1 &&
    (rows - 1) * plane.bytesPerRow + (columns - 1) * plane.bytesPerPixel <
        plane.bytes.length;

/// Computes the proper [ml_face.InputImageRotation] from sensor orientation and device orientation.
ml_face.InputImageRotation computeMlKitRotation({
  required int sensorOrientation,
  required DeviceOrientation deviceOrientation,
  required bool isFrontCamera,
}) {
  var rotationCompensation = switch (deviceOrientation) {
    DeviceOrientation.portraitUp => 0,
    DeviceOrientation.landscapeLeft => 90,
    DeviceOrientation.portraitDown => 180,
    DeviceOrientation.landscapeRight => 270,
  };

  final int rotationDegrees;
  if (isFrontCamera) {
    rotationDegrees = (sensorOrientation + rotationCompensation) % 360;
  } else {
    rotationDegrees = (sensorOrientation - rotationCompensation + 360) % 360;
  }

  return switch (rotationDegrees) {
    90 => ml_face.InputImageRotation.rotation90deg,
    180 => ml_face.InputImageRotation.rotation180deg,
    270 => ml_face.InputImageRotation.rotation270deg,
    _ => ml_face.InputImageRotation.rotation0deg,
  };
}

/// Converts a [CameraImage] into an [ml_face.InputImage] ready for ML Kit face detection.
Future<ml_face.InputImage?> convertCameraImageToMlKitInput({
  required CameraImage image,
  required ml_face.InputImageRotation rotation,
  CameraFrameProcessor? processor,
}) async {
  if (Platform.isAndroid) {
    final frameData = CameraPixelFrameData.fromCameraImage(image);
    final bytes = processor != null
        ? await processor.process(frameData)
        : await Isolate.run(
            () => encodeNv21(frameData, recoverShadows: true),
            debugName: 'face-detector-nv21-encoder',
          );
    if (bytes == null) return null;

    return ml_face.InputImage.fromBytes(
      bytes: bytes,
      metadata: ml_face.InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: ml_face.InputImageFormat.nv21,
        bytesPerRow: image.width,
      ),
    );
  }

  if (Platform.isIOS) {
    final format = ml_face.InputImageFormatValue.fromRawValue(image.format.raw);
    if (format != ml_face.InputImageFormat.bgra8888 ||
        image.planes.length != 1) {
      return null;
    }
    final plane = image.planes.first;
    final source = plane.bytes;
    final width = image.width;
    final height = image.height;
    final stride = plane.bytesPerRow;
    final bytes = processor != null
        ? await processor.process(
            CameraPixelFrameData.fromCameraImage(image),
            bgra: true,
          )
        : await Isolate.run(
            () => recoverBgraShadows(source, width, height, stride),
            debugName: 'face-detector-shadow-recovery',
          );
    if (bytes == null) return null;
    return ml_face.InputImage.fromBytes(
      bytes: bytes,
      metadata: ml_face.InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: format!,
        bytesPerRow: plane.bytesPerRow,
      ),
    );
  }

  return null;
}

/// Projects a point from sensor image space into normalized preview coordinates [0..1].
Offset projectMlKitPointToDisplay(
  Offset point, {
  required Size imageSize,
  required ml_face.InputImageRotation rotation,
  required bool isIOS,
  required bool mirrorHorizontally,
}) {
  final normalizedX = switch (rotation) {
    ml_face.InputImageRotation.rotation90deg =>
      point.dx / (isIOS ? imageSize.width : imageSize.height),
    ml_face.InputImageRotation.rotation270deg =>
      1 - point.dx / (isIOS ? imageSize.width : imageSize.height),
    ml_face.InputImageRotation.rotation0deg ||
    ml_face.InputImageRotation.rotation180deg =>
      mirrorHorizontally
          ? 1 - (point.dx / imageSize.width)
          : point.dx / imageSize.width,
  };

  final normalizedY = switch (rotation) {
    ml_face.InputImageRotation.rotation90deg ||
    ml_face.InputImageRotation.rotation270deg =>
      point.dy / (isIOS ? imageSize.height : imageSize.width),
    ml_face.InputImageRotation.rotation0deg ||
    ml_face.InputImageRotation.rotation180deg => point.dy / imageSize.height,
  };

  return Offset(normalizedX.clamp(0.0, 1.0), normalizedY.clamp(0.0, 1.0));
}

/// Projects an ML Kit bounding box into normalized preview coordinates [0..1].
Rect projectMlKitRectToDisplay(
  Rect rect, {
  required Size imageSize,
  required ml_face.InputImageRotation rotation,
  required bool isIOS,
  required bool mirrorHorizontally,
}) {
  final topLeft = projectMlKitPointToDisplay(
    Offset(rect.left, rect.top),
    imageSize: imageSize,
    rotation: rotation,
    isIOS: isIOS,
    mirrorHorizontally: mirrorHorizontally,
  );
  final bottomRight = projectMlKitPointToDisplay(
    Offset(rect.right, rect.bottom),
    imageSize: imageSize,
    rotation: rotation,
    isIOS: isIOS,
    mirrorHorizontally: mirrorHorizontally,
  );

  final left = topLeft.dx < bottomRight.dx ? topLeft.dx : bottomRight.dx;
  final right = topLeft.dx > bottomRight.dx ? topLeft.dx : bottomRight.dx;
  final top = topLeft.dy < bottomRight.dy ? topLeft.dy : bottomRight.dy;
  final bottom = topLeft.dy > bottomRight.dy ? topLeft.dy : bottomRight.dy;

  return Rect.fromLTRB(
    left.clamp(0.0, 1.0),
    top.clamp(0.0, 1.0),
    right.clamp(0.0, 1.0),
    bottom.clamp(0.0, 1.0),
  );
}
