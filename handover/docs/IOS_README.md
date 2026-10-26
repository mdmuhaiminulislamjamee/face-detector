# iOS app-team handover: Flutter USB-NCM single-face detector

This source package plugs into an **existing Flutter iOS app**. `sdk/` is the same
detector/model as the Android package; `example/` is an independent acceptance
harness. It is not a Swift-only framework or a precompiled IPA/XCFramework. No PC
backend, ML Kit, phone camera permission, or gesture model is required.

Read `INTEGRATION.md`, `PROTOCOL.md`, `POLICY.json`, and `VALIDATION.md`.

## Quick test on a Mac

Install compatible Flutter/Dart (Dart >=3.11.5), Xcode, CMake, the iOS SDK, and the
dependency tooling required by your Flutter version. The minimum deployment
target is **iOS 15.5**. First native builds download and compile OpenCV.

```sh
cd example
flutter pub get
flutter run --profile
```

Select a physical iPhone/iPad in Xcode, set your development signing team, and
accept the Local Network prompt. Use a compatible USB/adapter with sufficient
power. Confirm the OS actually exposes/configures the board's NCM network at
192.168.50.1/30; app code cannot add an unsupported driver or static adapter IP.

## Existing iOS app

Follow `INTEGRATION.md` for the SDK path dependency and native hooks in your
app's root pubspec. Match the Runner deployment target and any Podfile/native
build deployment settings to iOS 15.5 or higher. Do not mix a second Flutter
engine or add-to-app framework into an app that is already Flutter.

Merge this into the **host app's Info.plist**:

```xml
<key>NSLocalNetworkUsageDescription</key>
<string>Connect to the USB-NCM camera for on-device face detection.</string>
```

This camera protocol uses raw TCP and unicast UDP. It does not use Bonjour,
multicast, HTTP, or the phone camera; do not add arbitrary ATS bypasses or camera
permissions for it. If Local Network access was denied, enable the app's access
in iOS Settings before retrying.

Keep your host's existing CocoaPods or Swift Package Manager setup consistent
with its Flutter version. The current [Flutter iOS dependency guidance](https://docs.flutter.dev/packages-and-plugins/swift-package-manager/for-app-developers)
describes supported integration; older Flutter projects can retain their
supported CocoaPods setup. The standalone example uses the SDK's generated host
files and does not ship signing material or machine-specific generated configs.

The controller stops capture/clears overlays on background/inactive transitions
and resumes only when capture had been requested. Do not bypass that lifecycle
handling. Test USB unplug/reconnect, Local Network denial, rotation, target
replacement, and warm sustained speed on each supported physical device.

This delivery was prepared on Windows. Xcode compilation, signing, USB interface
compatibility, and real iOS inference/FPS remain target-device acceptance work.
