# Source and third-party notices

The face detection application source is supplied for the requested app-team
handover. This package does not invent a new open-source license for that source;
confirm your project's distribution/license terms with its owner.

The model is OpenCV Zoo YuNet `face_detection_yunet_2023mar.onnx`, SHA-256
`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
Its supplied MIT license is included beside the model and under `licenses/`.

The SDK depends on dartcv4 2.3.1/OpenCV 4.13.0 and Flutter. Their license texts,
when available in the builder's SDK/package cache, are copied to `licenses/`.
The final host application must retain its generated dependency license notices
and any notices emitted by the native OpenCV build. The PC reference additionally
uses installed OpenCV-Python and NumPy; these are not bundled Python runtimes.
