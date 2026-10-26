"""Audit both delivery archives for parity, configuration, and accidental caches."""
import hashlib
import json
from pathlib import Path
import plistlib
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

from verify_package import verify


def audit(directory):
    paths = [directory / f'NCM-Face-{platform}-1.0.0.zip'
             for platform in ('android', 'ios')]
    for path in paths:
        verify(path)
    with zipfile.ZipFile(paths[0]) as android, zipfile.ZipFile(paths[1]) as ios:
        for archive in (android, ios):
            names = archive.namelist()
            prohibited = [name for name in names if re.search(
                r'(^|/)(\.dart_tool|\.venv|\.git|\.gradle|Pods|build|xcuserdata)(/|$)'
                r'|(^|/)(local\.properties|key\.properties|Generated\.xcconfig)$'
                r'|\.(jks|keystore|p12|mobileprovision|pyc|log)$', name)]
            assert not prohibited, prohibited
            for filename in ('sdk/pubspec.yaml', 'example/pubspec.yaml'):
                assert not re.search(r'skip_build:\s*true', archive.read(filename).decode())
            policy = json.loads(archive.read('POLICY.json'))
            assert policy['maximum_selected_faces'] == 1
            assert policy['maximum_detection_starts_per_second'] == 10
            assert policy['accepted_score'] == policy['recovery_score'] == .85
            assert policy['confirmation_hits'] == 3 and policy['dropout_hold_ms'] == 200
            model = archive.read('sdk/assets/models/face_detection_yunet_2023mar.onnx')
            assert hashlib.sha256(model).hexdigest() == policy['model_sha256']
            state = json.loads(archive.read('VALIDATION_STATUS.json'))
            assert 'pending' not in json.dumps(state).lower()
        shared = [name for name in android.namelist()
                  if name.startswith(('sdk/lib/', 'sdk/assets/', 'sdk/test/'))
                  or name in ('sdk/pubspec.yaml', 'POLICY.json')]
        for name in shared:
            assert android.read(name) == ios.read(name), f'Platform SDKs differ: {name}'
        permissions = ET.fromstring(android.read('example/android/app/src/main/AndroidManifest.xml'))
        permission_names = {node.attrib['{http://schemas.android.com/apk/res/android}name']
                            for node in permissions.findall('uses-permission')}
        assert {'android.permission.INTERNET', 'android.permission.ACCESS_NETWORK_STATE'} <= permission_names
        assert 'android.permission.CAMERA' not in permission_names
        info = plistlib.loads(ios.read('example/ios/Runner/Info.plist'))
        assert info['NSLocalNetworkUsageDescription']
        assert 'NSCameraUsageDescription' not in info
        assert b'IPHONEOS_DEPLOYMENT_TARGET = 15.5;' in ios.read(
            'example/ios/Runner.xcodeproj/project.pbxproj')
    print(f'Both platform handovers passed parity/configuration audit ({len(shared)} shared SDK files).')


if __name__ == '__main__':
    audit(Path(sys.argv[1]))
