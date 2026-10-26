import 'package:flutter_test/flutter_test.dart';
import 'package:ncm_face_sdk/ncm_face_sdk.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test(
    'public controller starts empty, validates configuration, and disposes',
    () {
      final detector = NcmFaceDetector(manageAppLifecycle: false);
      expect(detector.frame, isNull);
      expect(detector.status.connected, isFalse);
      expect(
        () => detector.start(const NcmCameraConfig(tcpPort: 0)),
        throwsArgumentError,
      );
      expect(
        () => detector.start(const NcmCameraConfig(deviceAddress: 'not-an-ip')),
        throwsArgumentError,
      );
      detector.stop();
      expect(detector.frame, isNull);
      detector.dispose();
      expect(() => detector.start(), throwsStateError);
    },
  );

  test(
    'normal construction supports automatic Flutter lifecycle management',
    () {
      final detector = NcmFaceDetector();
      expect(detector.frame, isNull);
      detector.dispose();
    },
  );
}
