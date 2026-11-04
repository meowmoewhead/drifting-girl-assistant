# ALAS framework reuse

## MaaFramework execution engine

Official project: https://github.com/MaaXYZ/MaaFramework, pinned Python package
`maafw==5.14.3` from https://pypi.org/project/maafw/5.14.3/.
Corresponding source: https://github.com/MaaXYZ/MaaFramework/tree/v5.14.3.
MaaFramework is LGPL-3.0; its license is included as
`licenses/MaaFramework-LGPL-3.0.txt` (the base GPL-3.0 text is this project's
`LICENSE`). The official wheel's unmodified Python bindings and Windows native
libraries are collected under `maa/` in the executable. Libraries can be replaced
using the upstream `MAAFW_BINARY_PATH` environment variable, or by rebuilding
from the supplied source and pinned requirements. No executable or game-specific
pipeline from a third-party Drifting Girl script is included.

`drift/core/maa_engine.py` integrates the official Tasker, Resource,
CustomController and CustomAction APIs. Local Python actions retain their strict
game guards and MuMu input adapter. The ALAS-derived matching helpers and local
NextRun scheduler described below continue to be used alongside MaaFramework.

Upstream: https://github.com/LmeSzinc/AzurLaneAutoScript

Pinned commit: `77f4d01fcd2b0acab05a4d89260e8a0a9cd03d21` (retrieved 2026-09-30).

The `alas_core` package contains adapted upstream code:

- `Button`, image utilities and `Timer`: copied from `module/base`, with imports changed.
- Template match scores are exposed as `Button.last_similarity`; algorithms are unchanged.
- `Resource` and cached-property helpers: extracted from upstream. The resource default is `default` instead of an Azur Lane server.
- `ModuleBase.loop`, `appear`, `appear_then_click`: extracted from upstream. Initialization accepts the adapter's config/device; game dependencies and hierarchy detection are omitted.
- Logging uses Python's standard logger instead of ALAS's global logging modifications.

`tools/vendor_alas.py` reproduces extraction from `vendor/alas`. Original checksums are in `alas_core/UPSTREAM.json`.

`app.py`, its ADB adapter, JSON rules and browser panel provide the standalone integration. This application is distributed under GPL-3.0; the upstream license is included as `LICENSE`. This is an independent adaptation, not an official ALAS distribution. ALAS's Azur Lane tasks, game assets and OCR models are not bundled.

`native_capture.py` adapts the MuMu SDK function signatures and bottom-up RGBA conversion documented by the pinned upstream `module/device/method/nemu_ipc.py`. Portrait input uses its original x/y, matching the portrait condition in [BAAS's SDK implementation](https://github.com/pur1fying/BAAS_Cpp/blob/main/src/device/BAASNemu.cpp); the ALAS landscape rotation is not applied. Unverified landscape input is rejected. The SDK DLL remains part of the user's MuMu installation and is not redistributed. Sea God input requires a new verified native screenshot and releases the contact even on stop; other inputs use ADB. The age guard, no-repeat handling for uncertain input, timeout and capture fallback are local code.

The added `CroppedTemplateButton` adapter reuses ALAS `Button.match` on resized variants of a stable icon fragment. Dynamic text and badges are excluded by cropping. The first game-specific icon was supplied by the user, not copied from ALAS.

Optional companion fragments also use upstream `Button.match`, in small search windows anchored to the main fragment at the same scale. The two relief-item rules require their distinct hanging symbols alongside the canopy; this composition is implemented by the local adapter.

`reward_list.py` is local integration code. It uses an ALAS-backed page anchor and OpenCV HSV color checks to select green reward rows; row coordinates follow the anchor's scale and translation. Reward values and task progress are not template content.

## Offline numeric recognition

`digit_ocr.py` runs the English PP-OCRv4 mobile recognition model through ONNX Runtime. Image normalization and CTC decoding follow the model format documented by [RapidOCR](https://github.com/RapidAI/RapidOCR/tree/main/python/rapidocr/ch_ppocr_rec). The numeric-only output filter, confidence gate and application workflow are local integration code.

The model was trained by PaddleOCR/Baidu and converted to ONNX by RapidAI. Its source URL and pinned SHA-256 are recorded in `models/digit-model.json`; `tools/fetch_digit_model.py` downloads that exact model for source builds. Runtime inference is local and has no network requests. This is separate from ALAS's game-specific MXNet OCR models, which are not bundled.

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR): Apache-2.0, included as `licenses/PaddleOCR-APACHE-2.0.txt`.
- [RapidOCR](https://github.com/RapidAI/RapidOCR): Apache-2.0, included as `licenses/RapidOCR-APACHE-2.0.txt`.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime): MIT, with its license and third-party notices under `licenses/`.

## Windows distribution notices

The public Windows build also contains the Python runtime, NumPy, Pillow,
OpenCV, ONNX Runtime, MaaFramework and the PyInstaller bootloader. Notices
copied from the installed distributions are preserved under `licenses/bundled/`
and `licenses/Python-LICENSE.txt`; `licenses/bundled/index.json` records package
versions, relative notice paths and SHA-256 checksums. The PyInstaller license
text includes its bootloader exception. Build-only packages are listed in
`requirements-build.txt`.

Game interface screenshots and icons remain the property of their respective
rights holders. They serve as recognition templates, regression examples and
feature illustrations; this project's code license does not purport to relicense
the game's artwork. No game APK or MuMu SDK DLL is redistributed.
