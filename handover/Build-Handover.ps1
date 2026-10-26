param(
    [Parameter(Mandatory=$true)][string]$FlutterSdk,
    [string]$OutputDirectory = 'handover/output/release-1.0.0',
    [switch]$PrepareOnly,
    [switch]$ArchiveOnly,
    [switch]$RefreshSources
)

$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$outputRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDirectory))
if (-not $outputRoot.StartsWith($repoRoot + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase)) {
    throw 'OutputDirectory must be a new directory inside this repository.'
}
$utf8 = New-Object Text.UTF8Encoding($false)
function Write-Utf8([string]$Path, [string]$Content) {
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Path)) | Out-Null
    [IO.File]::WriteAllText($Path, $Content, $utf8)
}
function Copy-Source([string]$Relative, [string]$Destination) {
    $source = Join-Path $repoRoot $Relative
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing source: $Relative" }
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Destination)) | Out-Null
    Copy-Item -LiteralPath $source -Destination $Destination
}
function Read-Source([string]$Relative) {
    return [IO.File]::ReadAllText((Join-Path $repoRoot $Relative), $utf8)
}
function Copy-TemplateTree([string]$Name, [string]$Destination) {
    $templateRoot = Join-Path $PSScriptRoot "templates/$Name"
    foreach ($file in (Get-ChildItem -LiteralPath $templateRoot -File -Recurse -Force)) {
        $relative = $file.FullName.Substring($templateRoot.Length + 1)
        $target = Join-Path $Destination $relative
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target)) | Out-Null
        Copy-Item -LiteralPath $file.FullName -Destination $target
    }
}
function Read-Number([string]$Text, [string]$Pattern) {
    $match = [regex]::Match($Text, $Pattern)
    if (-not $match.Success) { throw "Missing policy constant: $Pattern" }
    return [double]::Parse($match.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
}

$core = @(
    'detection/pose_affine.dart', 'detection/robust_yunet.dart', 'detection/yunet_candidate.dart',
    'ncm/jlip_protocol.dart', 'ncm/latest_frame_runner.dart', 'ncm/ncm_camera_client.dart',
    'services/face_tracking_stabilizer.dart', 'services/single_face_selector.dart',
    'services/ncm_face_detection_service.dart', 'services/yunet_worker.dart'
)
$tests = @(
    'face_confirmation_test.dart', 'face_tracking_stabilizer_test.dart',
    'single_face_selector_test.dart', 'latest_frame_runner_test.dart',
    'ncm_client_test.dart', 'ncm_detection_service_test.dart', 'ncm_protocol_test.dart',
    'pose_affine_test.dart', 'yunet_candidate_test.dart', 'yunet_worker_lifecycle_test.dart'
)
$pcFiles = @('live_face_detector.py', 'robust_face_detector.py', 'single_face_selector.py', 'run.bat',
    'camera_capture.py', 'ncm_camera.py', 'native_runtime.py', 'requirements.txt', 'README.md',
    'face_detection_yunet_2023mar.onnx', 'YUNET_LICENSE')
$modelHash = '8F2383E4DD3CFBB4553EA8718107FC0423210DC964F9F4280604804ED2552FA4'
$pcPolicy = Read-Source 'python_face_detector/robust_face_detector.py'
$dartPolicy = Read-Source 'gesture_detector/lib/detection/yunet_candidate.dart'
$score = Read-Number $pcPolicy 'SCORE_THRESHOLD\s*=\s*([.0-9]+)'
$mobileScore = Read-Number $dartPolicy 'yuNetScoreThreshold\s*=\s*([.0-9]+)'
if ($score -ne $mobileScore) { throw 'PC/mobile confidence floors differ; resolve before release.' }
foreach ($model in @('python_face_detector/face_detection_yunet_2023mar.onnx',
        'gesture_detector/assets/models/face_detection_yunet_2023mar.onnx')) {
    if ((Get-FileHash -LiteralPath (Join-Path $repoRoot $model)).Hash -ne $modelHash) {
        throw "Unexpected model bytes: $model"
    }
}
$policy = [ordered]@{
    version = '1.0.0'; detector = 'OpenCV YuNet 2023mar'; model_sha256 = $modelHash.ToLowerInvariant()
    accepted_score = $score; recovery_score = $score; proposal_score = 0.60
    direct_foreshortened_score = (Read-Number $pcPolicy 'POSE_SCORE_THRESHOLD\s*=\s*([.0-9]+)')
    nms_iou = 0.30; minimum_face_pixels = 18; confirmation_hits = 3
    dropout_hold_ms = 200; maximum_selected_faces = 1; maximum_detection_starts_per_second = 10
    pending_frame_max_age_ms = 500; result_max_age_ms = 750; opencv_runtime_threads = 2
    native_dependency = 'dartcv4 2.3.1 / OpenCV 4.13.0'; mirror = $false
    selection = 'largest confirmed initially; keep lock; replace after dropout hold expires'
}

if (-not $ArchiveOnly) {
    if ((Test-Path -LiteralPath $outputRoot) -and -not $RefreshSources) {
        throw 'Output directory exists. Choose a new output path.'
    }
    [IO.Directory]::CreateDirectory($outputRoot) | Out-Null
    foreach ($platform in @('android', 'ios')) {
        $packageRoot = Join-Path $outputRoot $platform
        $sdkRoot = Join-Path $packageRoot 'sdk'
        if ($RefreshSources -and -not (Test-Path -LiteralPath (Join-Path $packageRoot 'POLICY.json'))) {
            throw 'RefreshSources requires an existing generated handover, identified by POLICY.json.'
        }
        [IO.Directory]::CreateDirectory($packageRoot) | Out-Null
        Copy-TemplateTree 'sdk' $sdkRoot
        Copy-TemplateTree 'example' (Join-Path $packageRoot 'example')
        foreach ($relative in $core) {
            $destination = Join-Path $sdkRoot "lib/src/$relative"
            Copy-Source "gesture_detector/lib/$relative" $destination
            if ($relative -eq 'services/ncm_face_detection_service.dart') {
                $content = [IO.File]::ReadAllText($destination, $utf8)
                $content = $content.Replace("import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';", '')
                Write-Utf8 $destination $content
            }
        }
        foreach ($file in @('face_detection_yunet_2023mar.onnx', 'YUNET_LICENSE')) {
            Copy-Source "gesture_detector/assets/models/$file" (Join-Path $sdkRoot "assets/models/$file")
        }
        foreach ($test in $tests) {
            $content = Read-Source "gesture_detector/test/$test"
            $content = $content.Replace('package:face_detector/', 'package:ncm_face_sdk/src/')
            $content = $content.Replace('python_face_detector/tests/fixtures/face.jpg', 'test/fixtures/face.jpg')
            Write-Utf8 (Join-Path $sdkRoot "test/$test") $content
        }
        foreach ($fixture in @('face.jpg', 'non_face.jpg')) {
            Copy-Source "python_face_detector/tests/fixtures/$fixture" (Join-Path $sdkRoot "test/fixtures/$fixture")
        }
        $verifier = (Read-Source 'gesture_detector/tool/verify_yunet.dart').Replace(
            'package:face_detector/', 'package:ncm_face_sdk/src/').Replace(
            'python_face_detector/tests/fixtures/', 'test/fixtures/')
        Write-Utf8 (Join-Path $sdkRoot 'tool/verify_yunet.dart') $verifier
        Copy-Source 'handover/docs/INTEGRATION.md' (Join-Path $sdkRoot 'README.md')
        foreach ($doc in @('INTEGRATION.md', 'PROTOCOL.md', 'VALIDATION.md', 'THIRD_PARTY_NOTICES.md')) {
            Copy-Source "handover/docs/$doc" (Join-Path $packageRoot $doc)
        }
        Copy-Source ("handover/docs/" + $platform.ToUpperInvariant() + '_README.md') (Join-Path $packageRoot 'README.md')
        Copy-Source 'handover/tools/verify_package.py' (Join-Path $packageRoot 'verify_package.py')
        Copy-Source 'gesture_detector/assets/models/YUNET_LICENSE' (Join-Path $packageRoot 'licenses/YUNET_LICENSE')
        Copy-Source 'run_face_detection.bat' (Join-Path $packageRoot 'pc_reference/run_face_detection.bat')
        Copy-Source 'run_ncm_face_detection.bat' (Join-Path $packageRoot 'pc_reference/run_ncm_face_detection.bat')
        foreach ($relative in $pcFiles) {
            Copy-Source "python_face_detector/$relative" (Join-Path $packageRoot "pc_reference/python_face_detector/$relative")
        }
        foreach ($testFile in (Get-ChildItem -LiteralPath (Join-Path $repoRoot 'python_face_detector/tests') -File -Recurse)) {
            $relative = $testFile.FullName.Substring((Join-Path $repoRoot 'python_face_detector/').Length)
            if ($relative -match '__pycache__' -or $relative.EndsWith('.pyc')) { continue }
            Copy-Source ("python_face_detector/" + $relative) (Join-Path $packageRoot "pc_reference/python_face_detector/$relative")
        }
        Write-Utf8 (Join-Path $packageRoot 'POLICY.json') ($policy | ConvertTo-Json -Depth 6)
        Write-Utf8 (Join-Path $sdkRoot 'POLICY.json') ($policy | ConvertTo-Json -Depth 6)
        $hashes = [ordered]@{}
        foreach ($relative in $core) {
            $hashes["gesture_detector/lib/$relative"] = (Get-FileHash -LiteralPath (Join-Path $repoRoot "gesture_detector/lib/$relative")).Hash.ToLowerInvariant()
        }
        $sourceManifest = [ordered]@{
            source = 'explicit working-tree snapshot, including current user threshold'
            algorithm_sources_sha256 = $hashes
            adapter = 'NCM-only DTO replaces ML Kit types; public facade selects one face and supplies package model path'
        }
        Write-Utf8 (Join-Path $packageRoot 'SOURCES.json') ($sourceManifest | ConvertTo-Json -Depth 6)
        # Generate a fresh platform host; do not copy an existing app's signing/SDK files.
        if (-not $RefreshSources) {
        $dartExe = Join-Path $FlutterSdk 'bin/cache/dart-sdk/bin/dart.exe'
        $snapshot = Join-Path $FlutterSdk 'bin/cache/flutter_tools.snapshot'
        & $dartExe $snapshot create --empty --no-pub "--platforms=$platform" --project-name ncm_face_example --org com.ncm.handover (Join-Path $packageRoot 'example')
        if ($LASTEXITCODE -ne 0) { throw "Flutter example generation failed: $platform" }
        if ($platform -eq 'android') {
            $gradlePath = Join-Path $packageRoot 'example/android/app/build.gradle.kts'
            $gradle = [IO.File]::ReadAllText($gradlePath, $utf8)
            $gradle = $gradle.Replace('compileSdk = flutter.compileSdkVersion', 'compileSdk = 36')
            $gradle = $gradle.Replace('ndkVersion = flutter.ndkVersion', 'ndkVersion = "28.2.13676358"')
            $gradle = $gradle.Replace('minSdk = flutter.minSdkVersion', 'minSdk = 24')
            $gradle = $gradle.Replace('targetSdk = flutter.targetSdkVersion', 'targetSdk = 36')
            Write-Utf8 $gradlePath $gradle
            $manifestPath = Join-Path $packageRoot 'example/android/app/src/main/AndroidManifest.xml'
            $manifest = [IO.File]::ReadAllText($manifestPath, $utf8)
            $permissions = '<uses-permission android:name="android.permission.INTERNET"/>' + "`n" +
                '<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE"/>' + "`n" +
                '<uses-feature android:name="android.hardware.usb.host" android:required="false"/>' + "`n"
            $manifest = $manifest.Replace('    <application', $permissions + '    <application')
            Write-Utf8 $manifestPath $manifest
        } else {
            $plistPath = Join-Path $packageRoot 'example/ios/Runner/Info.plist'
            $plist = [IO.File]::ReadAllText($plistPath, $utf8)
            $plist = [regex]::new('<dict>').Replace($plist, "<dict>`n<key>NSLocalNetworkUsageDescription</key><string>Connect to the USB-NCM camera for on-device face detection.</string>", 1)
            Write-Utf8 $plistPath $plist
            $projectPath = Join-Path $packageRoot 'example/ios/Runner.xcodeproj/project.pbxproj'
            $project = [IO.File]::ReadAllText($projectPath, $utf8)
            $project = [regex]::Replace($project, 'IPHONEOS_DEPLOYMENT_TARGET = [0-9.]+;', 'IPHONEOS_DEPLOYMENT_TARGET = 15.5;')
            Write-Utf8 $projectPath $project
        }
        }
        $flutterLicense = Join-Path $FlutterSdk 'LICENSE'
        if (Test-Path -LiteralPath $flutterLicense) {
            Copy-Item -LiteralPath $flutterLicense -Destination (Join-Path $packageRoot 'licenses/FLUTTER_LICENSE')
        }
        $config = (Read-Source 'gesture_detector/.dart_tool/package_config.json') | ConvertFrom-Json
        $dartcv = $config.packages | Where-Object name -eq 'dartcv4'
        $dartcvLicense = Join-Path ([Uri]$dartcv.rootUri).LocalPath 'LICENSE'
        if (Test-Path -LiteralPath $dartcvLicense) {
            Copy-Item -LiteralPath $dartcvLicense -Destination (Join-Path $packageRoot 'licenses/DARTCV4_LICENSE')
        }
    }
}

foreach ($platform in @('android', 'ios')) {
    $packageRoot = Join-Path $outputRoot $platform
    if (-not (Test-Path -LiteralPath (Join-Path $packageRoot 'POLICY.json'))) { throw 'No prepared package to archive.' }
    Copy-Source 'handover/VALIDATION_STATUS.json' (Join-Path $packageRoot 'VALIDATION_STATUS.json')
    Copy-Source 'handover/PC_RUNTIME.json' (Join-Path $packageRoot 'PC_RUNTIME.json')
    Copy-Source 'python_face_detector/run.bat' (Join-Path $packageRoot 'pc_reference/python_face_detector/run.bat')
    Copy-Source 'handover/docs/PC_REQUIREMENTS_LOCK.txt' (Join-Path $packageRoot 'pc_reference/python_face_detector/requirements-lock.txt')
    Copy-Source 'handover/tools/verify_package.py' (Join-Path $packageRoot 'verify_package.py')
    foreach ($doc in @('INTEGRATION.md', 'PROTOCOL.md', 'VALIDATION.md', 'THIRD_PARTY_NOTICES.md')) {
        Copy-Source "handover/docs/$doc" (Join-Path $packageRoot $doc)
    }
    Copy-Source ("handover/docs/" + $platform.ToUpperInvariant() + '_README.md') (Join-Path $packageRoot 'README.md')
    $frozenSources = Get-Content -LiteralPath (Join-Path $packageRoot 'SOURCES.json') -Raw | ConvertFrom-Json
    foreach ($property in $frozenSources.algorithm_sources_sha256.PSObject.Properties) {
        if ((Get-FileHash -LiteralPath (Join-Path $repoRoot $property.Name)).Hash.ToLowerInvariant() -ne $property.Value) {
            throw 'Canonical detector changed after preparation; refresh and revalidate before archiving.'
        }
    }
    foreach ($pubspec in @('sdk/pubspec.yaml', 'example/pubspec.yaml')) {
        if ([IO.File]::ReadAllText((Join-Path $packageRoot $pubspec)) -match 'skip_build:\s*true') {
            throw 'Native builds are disabled; restore the pubspec before delivery.'
        }
    }
}
if ($PrepareOnly) { Write-Output "Prepared handovers: $outputRoot"; exit 0 }

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipChecksums = @()
foreach ($platform in @('android', 'ios')) {
    $packageRoot = Join-Path $outputRoot $platform
    $payload = @()
    foreach ($file in (Get-ChildItem -LiteralPath $packageRoot -Recurse -File -Force)) {
        $relative = $file.FullName.Substring($packageRoot.Length + 1).Replace('\', '/')
        if ($relative -match '(^|/)(\.dart_tool|\.git|\.gradle|\.venv|Pods|build|\.symlinks|ephemeral|DerivedData|xcuserdata|\.idea)(/|$)' -or
            $relative -match '(^|/)(local\.properties|key\.properties|Generated\.xcconfig|flutter_export_environment\.sh|\.flutter-plugins-dependencies|MANIFEST\.sha256)$' -or
            $relative -match '\.(jks|keystore|p12|mobileprovision|pyc|log|iml)$') { continue }
        $payload += [pscustomobject]@{File=$file.FullName; Relative=$relative}
    }
    $payload = @($payload | Sort-Object Relative)
    $manifestLines = foreach ($entry in $payload) {
        (Get-FileHash -LiteralPath $entry.File).Hash.ToLowerInvariant() + '  ' + $entry.Relative
    }
    $manifestPath = Join-Path $packageRoot 'MANIFEST.sha256'
    Write-Utf8 $manifestPath (($manifestLines -join "`n") + "`n")
    $zipPath = Join-Path $outputRoot ("NCM-Face-" + $platform + '-1.0.0.zip')
    if (Test-Path -LiteralPath $zipPath) { throw 'Archive exists; use a new release output directory.' }
    $stream = [IO.File]::Open($zipPath, [IO.FileMode]::CreateNew)
    $archive = [IO.Compression.ZipArchive]::new($stream, [IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($entry in ($payload + [pscustomobject]@{File=$manifestPath; Relative='MANIFEST.sha256'})) {
            [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $entry.File,
                $entry.Relative, [IO.Compression.CompressionLevel]::Optimal) | Out-Null
        }
    } finally {
        $archive.Dispose()
        $stream.Dispose()
    }
    $zipChecksums += (Get-FileHash -LiteralPath $zipPath).Hash.ToLowerInvariant() + '  ' + [IO.Path]::GetFileName($zipPath)
    Write-Output "Created $zipPath"
}
Write-Utf8 (Join-Path $outputRoot 'SHA256SUMS.txt') (($zipChecksums -join "`n") + "`n")
