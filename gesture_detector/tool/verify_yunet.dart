// Executes the actual mobile OpenCV detector on the host. No Flutter UI needed.
import 'dart:convert';
import 'dart:io';

import 'package:dartcv4/dartcv.dart' as cv;

import 'package:face_detector/detection/robust_yunet.dart';
import 'package:face_detector/detection/yunet_candidate.dart';
import 'package:face_detector/detection/pose_affine.dart';

void require(bool condition, String message) {
  if (!condition) throw StateError(message);
}

cv.Mat background([int value = 215, int width = 1280, int height = 720]) {
  final colour = cv.Scalar.all(value.toDouble());
  try {
    return cv.Mat.fromScalar(height, width, cv.MatType.CV_8UC3, colour);
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

Map<String, Object> verifyPoses(RobustYuNet detector, cv.Mat portrait) {
  final results = <String, Object>{};
  for (final pose in <(double, double, double, double)>[
    (-40, 1, 1, 1),
    (40, 1, 1, 1),
    (-90, 1, 1, 1),
    (90, 1, 1, 1),
    (180, 1, 1, 1),
    (0, 1, .65, 1),
    (-25, 1, 1.35, .3),
  ]) {
    detector.reset();
    final scaled = cv.resize(portrait, (384, 384));
    final data = scaled.data;
    if (pose.$4 != 1) {
      for (var i = 0; i < data.length; i++) {
        data[i] = (data[i] * pose.$4).truncate();
      }
    }
    final affine = PoseAffine.rotated(384, 384, (pose.$1, pose.$2, pose.$3));
    final matrix = cv.Mat.fromList(2, 3, cv.MatType.CV_64FC1, affine.values);
    final border = cv.Scalar.all(215);
    final warped = cv.warpAffine(scaled, matrix, (
      affine.width,
      affine.height,
    ), borderValue: border);
    final scene = background();
    final rect = cv.Rect(
      350,
      (720 - affine.height) ~/ 2,
      affine.width,
      affine.height,
    );
    final target = scene.region(rect);
    try {
      warped.copyTo(target);
      var hits = 0, consecutive = 0;
      int? firstHit;
      for (var index = 0; index < 30; index++) {
        final found = detector.detect(scene);
        require(
          found.length <= 1 &&
              found.every((face) => face.score >= yuNetScoreThreshold),
          'Invalid pose detection for $pose',
        );
        if (found.isNotEmpty) {
          firstHit ??= index;
          hits++;
          consecutive++;
        } else {
          consecutive = 0;
        }
        if (consecutive >= 6) break;
      }
      require(
        consecutive >= 3,
        'Pose $pose could not produce fresh consecutive hits',
      );
      results['$pose'] = {'first_hit': firstHit!, 'hits': hits};
    } finally {
      target.dispose();
      rect.dispose();
      scene.dispose();
      warped.dispose();
      border.dispose();
      matrix.dispose();
      scaled.dispose();
    }
  }
  return results;
}

cv.Mat backlitJpeg(cv.Mat portrait, int side, double brightness, double pitch) {
  final scaled = cv.resize(portrait, (side, side));
  final values = scaled.data;
  for (var index = 0; index < values.length; index++) {
    values[index] = (values[index] * brightness).truncate();
  }
  final a = 1 - 2 * pitch;
  final matrix = cv.Mat.fromList(3, 3, cv.MatType.CV_64FC1, [
    a,
    -pitch,
    pitch * side,
    0,
    a,
    0,
    0,
    -2 * pitch / side,
    1,
  ]);
  final border = cv.Scalar.all(220);
  final warped = cv.warpPerspective(scaled, matrix, (
    side,
    side,
  ), borderValue: border);
  final scene = background(220, 640, 480);
  final rect = cv.Rect(245, 100, side, side);
  final target = scene.region(rect);
  final params = cv.VecI32.fromList([cv.IMWRITE_JPEG_QUALITY, 80]);
  try {
    warped.copyTo(target);
    final (ok, bytes) = cv.imencode('.jpg', scene, params: params);
    require(ok, 'Could not encode the small backlight scene');
    return cv.imdecode(bytes, cv.IMREAD_COLOR);
  } finally {
    params.dispose();
    target.dispose();
    rect.dispose();
    scene.dispose();
    warped.dispose();
    border.dispose();
    matrix.dispose();
    scaled.dispose();
  }
}

Map<String, Object> verifyBacklight(RobustYuNet detector, cv.Mat portrait) {
  final results = <String, Object>{};
  for (final scenario in <(int, double, double)>[
    (112, .23, .2),
    (160, .18, .3),
    (128, .15, 0),
  ]) {
    detector.reset();
    final scene = backlitJpeg(portrait, scenario.$1, scenario.$2, scenario.$3);
    final blank = background(220, 640, 480);
    try {
      var hits = 0, consecutive = 0, bestRun = 0;
      for (var index = 0; index < 18; index++) {
        final found = detector.detect(scene);
        require(
          found.length <= 1 &&
              found.every((face) => face.score >= yuNetScoreThreshold),
          'Invalid recovered backlight candidate: $scenario',
        );
        if (found.isNotEmpty) {
          hits++;
          consecutive++;
          if (consecutive > bestRun) bestRun = consecutive;
        } else {
          consecutive = 0;
        }
      }
      require(
        bestRun >= 3,
        'Backlight $scenario could not produce consecutive fresh hits',
      );
      for (var index = 0; index < 5; index++) {
        require(
          detector.detect(blank).isEmpty,
          'Backlight pose hint became a stale hit',
        );
      }
      results['$scenario'] = {'hits': hits, 'best_consecutive_hits': bestRun};
    } finally {
      blank.dispose();
      scene.dispose();
    }
  }
  return results;
}

void main(List<String> arguments) {
  cv.setNumThreads(2);
  final detector = RobustYuNet(
    File('assets/models/face_detection_yunet_2023mar.onnx').readAsBytesSync(),
  );
  final portrait = cv.imread('python_face_detector/tests/fixtures/face.jpg');
  final results = <String, Object>{};
  try {
    if (arguments.contains('--poses')) {
      results['pose_checks'] = verifyPoses(detector, portrait);
    }
    if (arguments.contains('--backlight')) {
      results['small_backlight_checks'] = verifyBacklight(detector, portrait);
    }
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
            face.score >= yuNetScoreThreshold &&
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
