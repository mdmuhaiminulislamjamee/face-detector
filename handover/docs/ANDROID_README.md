# Android app-team handover: Flutter USB-NCM single-face detector

This source package plugs into an **existing Flutter Android app**. `sdk/` is the
reusable package; `example/` is a separate acceptance harness, not a replacement
for your product. No PC backend, phone camera permission, ML Kit, or gesture model
is needed for this NCM path. Read `INTEGRATION.md`, `PROTOCOL.md`, `POLICY.json`,
and `VALIDATION.md` before comparing devices.

## Quick test

Install a compatible Flutter SDK (Dart >=3.11.5), Android Studio/SDK 36, JDK 17,
NDK 28.2.13676358, CMake, and Ninja. The example was generated from the Flutter
SDK recorded in `VALIDATION_STATUS.json`. First native builds download OpenCV.

```sh
cd example
flutter pub get
flutter run --profile
```

Connect/power the board using a USB host/OTG connection. Confirm that the phone
actually exposes the board as an Ethernet/NCM network interface and has
192.168.50.1/30. Then tap Connect. A cable or USB tethering alone does not ensure
that interface/route exists. The app cannot supply a missing OS driver.

## Existing Android app

Follow `INTEGRATION.md` for the SDK path dependency and root pubspec native hooks.
The reference Android configuration is:

```kotlin
android {
    compileSdk = 36
    ndkVersion = "28.2.13676358"
    defaultConfig {
        minSdk = 24
        targetSdk = 36
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
```

Merge these permissions in **the main/release manifest**, not only debug:

```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
<uses-feature android:name="android.hardware.usb.host" android:required="false" />
```

The SDK talks to an OS network interface using TCP/UDP, not Android's UVC/USB
camera API. It does not request CAMERA permission. Leave any permissions your
existing app independently needs. Use arm64-v8a physical devices for acceptance.
If your app bundles another OpenCV build, resolve dependency/ABI conflicts rather
than hiding a version mismatch with arbitrary library overrides.

Never copy machine-specific `local.properties`, signing keys, or this example's
application ID into your production app. The package includes source, model,
tests, and build instructions; it is not an APK/AAR or a verified Android binary.
Perform the device tests in `VALIDATION.md` and keep your app's release signing.
