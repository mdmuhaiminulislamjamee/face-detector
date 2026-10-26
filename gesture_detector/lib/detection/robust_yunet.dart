import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dartcv4/dartcv.dart' as cv;

import 'yunet_candidate.dart';

/// Port of python_face_detector/robust_face_detector.py. All native work runs
/// in a persistent isolate; full-resolution pixels survive for crop refinement.
class RobustYuNet {
  RobustYuNet(Uint8List model)
    : _detector = cv.FaceDetectorYN.fromBuffer(
        'onnx',
        model,
        Uint8List(0),
        (640, 640),
        scoreThreshold: .60,
        nmsThreshold: .3,
        topK: 5000,
      ),
      _clahe = cv.CLAHE(2.0, (8, 8)),
      _gamma = cv.Mat.fromList(
        1,
        256,
        cv.MatType.CV_8UC1,
        List.generate(256, (i) => (255 * math.pow(i / 255, .72)).round()),
      );

  final cv.FaceDetectorYN _detector;
  final cv.CLAHE _clahe;
  final cv.Mat _gamma;
  int _frameIndex = 0, _missedRegionFrames = 0;
  (int, int)? _frameSize;
  List<FaceBox> _previousBoxes = [];

  void reset() {
    _frameIndex = _missedRegionFrames = 0;
    _previousBoxes = [];
    _frameSize = null;
  }

  List<YuNetCandidate> detect(cv.Mat frame) {
    if (frame.isEmpty) return [];
    final width = frame.cols, height = frame.rows;
    if (_frameSize != (width, height)) {
      reset();
      _frameSize = (width, height);
    }
    final longest = math.max(width, height);
    final results = _infer(
      frame,
      math.min(1.0, 640 / longest),
      proposals: true,
    );
    final detailScan = _frameIndex++ % 5 == 0;
    if (detailScan && longest > 640) {
      results.addAll(
        _infer(frame, math.min(1.0, 1280 / longest), proposals: true),
      );
    }
    final proposals = deduplicateYuNet(results);
    var accepted = proposals.where((face) => face.score >= .88).toList();
    final regions = deduplicateYuNet([
      for (final box in _previousBoxes) YuNetCandidate(box, 1, const []),
      ...proposals.where((face) => face.score < .88),
    ]);
    for (final region in regions.take(4)) {
      if (accepted.any((face) => region.box.overlap(face.box) > .3)) continue;
      accepted.addAll(_refineRegion(frame, region.box));
    }
    if (detailScan && accepted.isEmpty) {
      final small = _resize(frame, math.min(1.0, 960 / longest));
      cv.Mat? enhanced;
      try {
        enhanced = enhanceShadows(small);
        if (enhanced != null) {
          final candidates = _infer(enhanced, 1, proposals: true);
          final sx = width / small.cols, sy = height / small.rows;
          for (final face in candidates.take(4)) {
            final box = face.box;
            accepted.addAll(
              _refineRegion(
                frame,
                FaceBox(
                  (box.x * sx).roundToDouble(),
                  (box.y * sy).roundToDouble(),
                  (box.width * sx).roundToDouble(),
                  (box.height * sy).roundToDouble(),
                ),
              ),
            );
          }
        }
      } finally {
        enhanced?.dispose();
        small.dispose();
      }
    }
    accepted = deduplicateYuNet(accepted);
    if (accepted.isNotEmpty) {
      _previousBoxes = [
        for (final face in accepted) _stableRegion(face.box, regions),
      ];
      _missedRegionFrames = 0;
    } else if (++_missedRegionFrames > 3) {
      _previousBoxes = [];
    }
    return accepted;
  }

  FaceBox _stableRegion(FaceBox box, List<YuNetCandidate> regions) {
    for (final region in regions) {
      final old = region.box;
      final wr = box.width / old.width, hr = box.height / old.height;
      if (box.overlap(old) > .3 &&
          wr >= .7 &&
          wr <= 1.4 &&
          hr >= .7 &&
          hr <= 1.4 &&
          (box.x + box.width / 2 - old.x - old.width / 2).abs() <
              old.width * .25 &&
          (box.y + box.height / 2 - old.y - old.height / 2).abs() <
              old.height * .25) {
        return old;
      }
    }
    return box;
  }

  List<YuNetCandidate> _refineRegion(cv.Mat frame, FaceBox box) {
    final margin = math.max(box.width, box.height) * .65;
    final left = math.max(0, (box.x - margin).truncate());
    final top = math.max(0, (box.y - margin).truncate());
    final right = math.min(frame.cols, (box.x + box.width + margin).truncate());
    final bottom = math.min(
      frame.rows,
      (box.y + box.height + margin).truncate(),
    );
    if (right <= left || bottom <= top) return [];
    final rect = cv.Rect(left, top, right - left, bottom - top);
    final crop = frame.region(rect);
    rect.dispose();
    cv.Mat? enhanced;
    try {
      final longest = math.max(crop.cols, crop.rows);
      final scale = math.min(4.0, 192 / longest);
      var results = _infer(crop, scale);
      if (results.isEmpty) {
        enhanced = enhanceShadows(crop);
        if (enhanced != null) results = _infer(enhanced, scale);
      }
      if (results.isEmpty) results = _infer(crop, math.min(4.0, 320 / longest));
      if (results.isEmpty && enhanced != null) {
        results = _infer(enhanced, math.min(4.0, 320 / longest));
      }
      return [
        for (final face in results)
          face.shifted(left.toDouble(), top.toDouble()),
      ];
    } finally {
      enhanced?.dispose();
      crop.dispose();
    }
  }

  cv.Mat _resize(cv.Mat frame, double scale) => cv.resize(frame, (
    math.max(32, (frame.cols * scale).round()),
    math.max(32, (frame.rows * scale).round()),
  ));

  List<YuNetCandidate> _infer(
    cv.Mat frame,
    double scale, {
    bool proposals = false,
  }) {
    final resized = _resize(frame, scale);
    cv.Mat? detections;
    try {
      final sx = resized.cols / frame.cols, sy = resized.rows / frame.rows;
      _detector.setInputSize((resized.cols, resized.rows));
      detections = _detector.detect(resized);
      final results = <YuNetCandidate>[];
      for (var i = 0; i < detections.rows; i++) {
        final row = List.generate(15, (j) => detections!.atF32(i, i1: j));
        for (var j = 0; j < 14; j += 2) {
          row[j] /= sx;
          row[j + 1] /= sy;
        }
        final candidate = YuNetCandidate.fromRow(row);
        if (!validYuNetCandidate(candidate, proposal: proposals)) continue;
        final box = candidate.box;
        final left = math.max(0, box.x.round()),
            top = math.max(0, box.y.round());
        final right = math.min(frame.cols, (box.x + box.width).round());
        final bottom = math.min(frame.rows, (box.y + box.height).round());
        if (math.min(right - left, bottom - top) < 18) continue;
        results.add(
          YuNetCandidate(
            FaceBox(
              left.toDouble(),
              top.toDouble(),
              (right - left).toDouble(),
              (bottom - top).toDouble(),
            ),
            candidate.score,
            [
              for (final p in candidate.landmarks)
                (p.$1.roundToDouble(), p.$2.roundToDouble()),
            ],
          ),
        );
      }
      return results;
    } finally {
      detections?.dispose();
      resized.dispose();
    }
  }

  /// Identical LAB/CLAHE/gamma blending to the PC shadow recovery path.
  cv.Mat? enhanceShadows(cv.Mat frame) {
    final lab = cv.cvtColor(frame, cv.COLOR_BGR2Lab);
    final light = cv.extractChannel(lab, 0);
    cv.Mat? local, lifted, blend;
    try {
      final data = light.data;
      var shadows = 0, samples = 0;
      for (var y = 0; y < light.rows; y += 8) {
        for (var x = 0; x < light.cols; x += 8) {
          final value = data[y * light.cols + x];
          samples++;
          if (value >= 8 && value < 85) shadows++;
        }
      }
      if (shadows / samples < .01) return null;
      local = _clahe.apply(light);
      lifted = cv.LUT(light, _gamma);
      blend = cv.addWeighted(lifted, .65, local, .35, 0);
      cv.insertChannel(blend, lab, 0);
      return cv.cvtColor(lab, cv.COLOR_Lab2BGR);
    } finally {
      blend?.dispose();
      lifted?.dispose();
      local?.dispose();
      light.dispose();
      lab.dispose();
    }
  }

  void dispose() {
    _detector.dispose();
    _clahe.dispose();
    _gamma.dispose();
  }
}
