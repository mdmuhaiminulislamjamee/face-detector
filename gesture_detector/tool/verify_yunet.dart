// Executes the actual mobile OpenCV detector on the host. No Flutter UI needed.
import 'dart:convert';
import 'dart:io';

import 'package:dartcv4/dartcv.dart' as cv;

import 'package:face_detector/detection/robust_yunet.dart';
import 'package:face_detector/detection/yunet_candidate.dart';

void require(bool condition, String message) {
  if (!condition) throw StateError(message);
}

cv.Mat background([int value = 215]) {
  final colour = cv.Scalar.all(value.toDouble());
  try {
    return cv.Mat.fromScalar(720, 1280, cv.MatType.CV_8UC3, colour);
  } finally {
    colour.dispose();
  }
}

void insertPortrait(
  cv.Mat scene,
  cv.Mat portrait,
  int side, {
  int x = 500,
  int y = 180,
  double brightness = 1,
}) {
  final scaled = cv.resize(portrait, (side, side));
  final rect = cv.Rect(x, y, side, side);
  final target = scene.region(rect);
  try {
    // Truncate uint8 brightness like the Python regression scenes.
    final data = scaled.data;
    if (brightness != 1) {
      for (var i = 0; i < data.length; i++) {
        data[i] = (data[i] * brightness).truncate();
      }
    }
    scaled.copyTo(target);
  } finally {
    target.dispose();
    rect.dispose();
    scaled.dispose();
  }
}

void main() {
  cv.setNumThreads(2);
  final detector = RobustYuNet(
    File('assets/models/face_detection_yunet_2023mar.onnx').readAsBytesSync(),
  );
  final portrait = cv.imread('python_face_detector/tests/fixtures/face.jpg');
  final results = <String, Object>{};
  try {
    for (final side in [512, 256, 128]) {
      for (final brightness in [1.0, .23]) {
        detector.reset();
        final scene = background();
        try {
          insertPortrait(scene, portrait, side, brightness: brightness);
          final watch = Stopwatch()..start();
          final faces = detector.detect(scene);
          require(
            faces.length == 1,
            'Scene $side/$brightness: ${faces.length} faces',
          );
          final face = faces.single;
          require(
            face.score >= .88 &&
                face.box.x >= 500 &&
                face.box.x < 500 + side &&
                face.box.y >= 180 &&
                face.box.y < 180 + side,
            'Scene $side/$brightness: wrong target or weak score',
          );
          results['$side/$brightness'] = {
            'rows': [face.toRow()],
            'milliseconds': watch.elapsedMilliseconds,
          };
        } finally {
          scene.dispose();
        }
      }
    }
    for (final side in [256, 128]) {
      for (final brightness in [1.0, .23]) {
        detector.reset();
        var hits = 0;
        for (var index = 0; index < 12; index++) {
          final scene = background();
          cv.Mat? decoded;
          final params = cv.VecI32.fromList([cv.IMWRITE_JPEG_QUALITY, 90]);
          try {
            insertPortrait(
              scene,
              portrait,
              side,
              x: 450 + index * 6,
              brightness: brightness,
            );
            final (ok, jpeg) = cv.imencode('.jpg', scene, params: params);
            require(ok, 'JPEG encoding failed');
            decoded = cv.imdecode(jpeg, cv.IMREAD_COLOR);
            final faces = detector.detect(decoded);
            require(
              faces.length <= 1 && faces.every(validYuNetCandidate),
              'Motion scene produced extra/invalid faces',
            );
            if (faces.isNotEmpty) hits++;
          } finally {
            params.dispose();
            decoded?.dispose();
            scene.dispose();
          }
        }
        require(
          hits >= 11,
          'Moving JPEG $side/$brightness: only $hits/12 hits',
        );
        results['motion_$side/$brightness'] = hits;
        final blank = background();
        try {
          for (var index = 0; index < 4; index++) {
            require(
              detector.detect(blank).isEmpty,
              'Cached region became a stale hit',
            );
          }
        } finally {
          blank.dispose();
        }
      }
    }
    detector.reset();
    final multiple = background();
    try {
      insertPortrait(multiple, portrait, 256);
      insertPortrait(multiple, portrait, 128, x: 900, y: 400);
      require(
        detector.detect(multiple).length == 2,
        'Multiple faces were merged/missed',
      );
      results['multiple_faces'] = 2;
    } finally {
      multiple.dispose();
    }
    for (final value in [0, 215, 255]) {
      final scene = background(value);
      try {
        require(
          detector.detect(scene).isEmpty,
          'False detection on solid $value',
        );
      } finally {
        scene.dispose();
      }
    }
    final animal = cv.imread(
      'python_face_detector/tests/fixtures/non_face.jpg',
    );
    try {
      require(
        detector.detect(animal).isEmpty,
        'False detection on non-face fixture',
      );
      results['non_face'] = 0;
    } finally {
      animal.dispose();
    }
    stdout.writeln(jsonEncode(results));
  } finally {
    portrait.dispose();
    detector.dispose();
  }
}
