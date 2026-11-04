"""Fetch the pinned offline recognition model; does not run inference."""
from pathlib import Path
import hashlib
import json
import urllib.request

root = Path(__file__).resolve().parents[1]
url = 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile.onnx'
expected = 'e8770c967605983d1570cdf5352041dfb68fa0c21664f49f47b155abd3e0e318'
target = root / 'models' / 'en_PP-OCRv4_rec_mobile.onnx'
target.parent.mkdir(exist_ok=True)
if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
    data = target.read_bytes()
else:
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
if hashlib.sha256(data).hexdigest() != expected:
    raise RuntimeError('OCR model checksum differs from the upstream manifest')
target.write_bytes(data)
manifest = {'file': target.name, 'source': url, 'sha256': expected,
            'upstream_manifest': 'https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml',
            'description': 'PaddleOCR English PP-OCRv4 mobile recognizer converted to ONNX by RapidAI; used only on numeric crops.'}
(target.parent / 'digit-model.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
licenses = root / 'licenses'
licenses.mkdir(exist_ok=True)
for name, source in {
    'PaddleOCR-APACHE-2.0.txt': 'https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/LICENSE',
    'RapidOCR-APACHE-2.0.txt': 'https://raw.githubusercontent.com/RapidAI/RapidOCR/main/LICENSE',
    'ONNXRuntime-MIT.txt': 'https://raw.githubusercontent.com/microsoft/onnxruntime/main/LICENSE',
}.items():
    if (licenses / name).is_file():
        continue
    for attempt in range(3):
        try:
            with urllib.request.urlopen(source, timeout=15) as response:
                (licenses / name).write_bytes(response.read())
            break
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
print(f'Offline digit model saved: {target.name} ({len(data)} bytes)')
