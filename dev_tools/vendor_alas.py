"""Reproducibly extract a small core from the pinned ALAS checkout."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor" / "alas"
DEST = ROOT / "alas_core"
COMMIT = "77f4d01fcd2b0acab05a4d89260e8a0a9cd03d21"
HEADER = f"# Adapted from LmeSzinc/AzurLaneAutoScript @ {COMMIT}\n# Copyright the ALAS contributors. SPDX-License-Identifier: GPL-3.0-only\n"


def extract(file, names):
    source = (SOURCE / file).read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    return "\n\n".join("".join(lines[n.lineno - 1:n.end_lineno]) for n in ast.parse(source).body
                        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)


def main():
    DEST.mkdir(exist_ok=True)
    (DEST / "__init__.py").write_text(HEADER, encoding="utf-8")
    hashes = {}
    for filename in ("button.py", "utils.py", "timer.py"):
        path = SOURCE / "module" / "base" / filename
        hashes[path.relative_to(SOURCE).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        content = path.read_text(encoding="utf-8")
        content = content.replace("from module.base.", "from alas_core.")
        content = content.replace("from module.config.server import VALID_SERVER", "from alas_core.resource import VALID_SERVER")
        content = content.replace("from module.logger import logger", "from alas_core.log import logger")
        if filename == "button.py":
            # Keep matching intact, expose its score for the dashboard.
            output = []
            for line in content.splitlines():
                output.append(line)
                if "_, sim, _, point = cv2.minMaxLoc(res)" in line:
                    indent = line[:len(line) - len(line.lstrip())]
                    output.append(indent + "self.last_similarity = float(sim)")
            content = "\n".join(output) + "\n"
        (DEST / filename).write_text(HEADER + content, encoding="utf-8")

    decorators = extract("module/base/decorator.py", {"cached_property", "del_cached_property"})
    (DEST / "decorator.py").write_text(HEADER + "from typing import Callable, Generic, TypeVar\nT = TypeVar('T')\n\n" + decorators, encoding="utf-8")
    resource = extract("module/base/resource.py", {"Resource"})
    resource = resource.replace("s = server.server", "s = 'default'")
    resource = resource.replace("from module.logger import logger", "from alas_core.log import logger")
    (DEST / "resource.py").write_text(HEADER + "from .decorator import del_cached_property\nVALID_SERVER = ['default']\n\n" + resource, encoding="utf-8")
    (DEST / "log.py").write_text(HEADER + "import logging\nlogger = logging.getLogger('mumu.alas')\n", encoding="utf-8")

    source = (SOURCE / "module/base/base.py").read_text(encoding="utf-8")
    klass = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "ModuleBase")
    lines = source.splitlines(keepends=True)
    body = "\n\n".join("".join(lines[n.lineno - 1:n.end_lineno]) for n in klass.body
                       if isinstance(n, ast.FunctionDef) and n.name in {"loop", "appear", "appear_then_click"})
    body = body.replace("        if isinstance(button, HierarchyButton):\n            appear = bool(button)\n        elif offset:", "        if offset:")
    init = """from .button import Button
from .timer import Timer

class ModuleBase:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.interval_timer = {}

    def ensure_button(self, button):
        if not isinstance(button, Button):
            raise TypeError('This adapter accepts ALAS image Buttons only')
        return button

"""
    (DEST / "base.py").write_text(HEADER + init + body, encoding="utf-8")
    (ROOT / "LICENSE").write_bytes((SOURCE / "LICENSE").read_bytes())
    (DEST / "UPSTREAM.json").write_text(json.dumps({"repository": "https://github.com/LmeSzinc/AzurLaneAutoScript", "commit": COMMIT, "original_sha256": hashes}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
