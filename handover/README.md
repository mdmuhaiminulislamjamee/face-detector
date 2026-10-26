# NCM face detector handover builder

This creates two self-contained **Flutter** source handovers for the existing
Android and iOS app teams. Both contain the same detector SDK/model and exclusive
single-face selection policy, with platform-specific example hosts and setup.
They do not require a PC backend or a hand-gesture model.

From the repository root on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File handover/Build-Handover.ps1 `
  -FlutterSdk C:/src/flutter -OutputDirectory handover/output/release-1.0.0
```

The output directory must be new. Nothing outside it is deleted or overwritten.
`-PrepareOnly` creates source trees for validation before ZIP creation;
`-ArchiveOnly` creates clean archives from those prepared trees after testing.
`-RefreshSources -PrepareOnly` refreshes generated source snapshots after editing
the canonical implementation/templates; it does not recreate the platform hosts.
The archive builder excludes build outputs, SDK paths, signing files, credentials,
and tool caches. `MANIFEST.sha256` in each package verifies its payload; the outer
`SHA256SUMS.txt` verifies the two ZIPs. Run the supplied `verify_package.py` after
extracting a handover.

The `sdk/lib/src` algorithms are copied from an explicit allowlist of the working
Flutter detector, not reimplemented. A provider-independent face DTO removes the
ML Kit dependency from the NCM-only SDK. The public facade sets the package model
asset path and exposes only the selected face. Canonical source hashes, the
actual PC/mobile thresholds, and validation limits travel with each release.

This is source delivery. Android/iOS compilation and physical-phone camera
performance require the target toolchains and devices; the package README gives
the exact acceptance procedure rather than promising identical phone FPS.
