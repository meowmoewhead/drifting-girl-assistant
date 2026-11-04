"""Build the portable Windows release, or the legacy source/evidence zip (--full).

Manifest counts include every regular archive member, including version.json
and the empty logs/.gitkeep, but exclude directory entries. Only release
artifacts are written; the public config.example.json is always used for config.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drift.core.paths import UPSTREAM, read_version

PORTABLE_NAME = "mumu-clicker"
MAX_PORTABLE_BYTES = 300_000_000
PORTABLE_FILES = (
    "启动.cmd", "start-panel.ps1", "config.json", "config.example.json", "VERSION", "USER-GUIDE.md",
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
)
PORTABLE_DIRS = ("assets", "models", "web", "licenses", "docs")
EXCLUDED_DIRS = frozenset((
    "tests", "tools", "captures", "diagnostics", "build", "build_tools",
    ".venv", ".git", "release", "__pycache__",
))
EXCLUDED_PATTERNS = (
    "*.before-*", "*.bak-*", "codex_*.md", "review*_codex_*.md",
    "app.py.before-refactor", "*.pyc", "*.pyo",
    "*.pre-*",
)

# --full preserves the old flat layout, source trees, fixtures and evidence set.
FULL_FILES = (
    "app.py", "frame_runtime.py", "features.py", "task_schedule.py",
    "meter_calibration.py", "parallel_world.py", "parallel_skills.py",
    "boss_training.py", "boss_defense.py", "boss_evidence.py", "native_capture.py",
    "fishing_meter.py", "training_sailing.py", "backpack_sale.py",
    "equipment_upgrade.py", "game_recovery.py", "daily_schedule.py",
    "digit_ocr.py", "reward_list.py", "purchase_checks.py", "config.json",
    "README.md", "VALIDATION.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
    "requirements.txt", "start-panel.cmd", "start-panel.ps1", "config.example.json",
    "VERSION", "USER-GUIDE.md", "启动.cmd",
)
FULL_DIRS = ("alas_core", "drift", "web", "tests", "tools", "assets", "models", "licenses")
FULL_SUFFIXES = {".py", ".html", ".css", ".js", ".json", ".ps1", ".png", ".onnx", ".txt"}
FULL_EVIDENCE = {
    "captures": (
        "other-functions-start.png", "mailbox-before-claim.png", "mailbox-after-claim.png",
        "mailbox-return.png", "mailbox-replay-result.json", "achievements-list-before.png",
        "achievements-cycle-result.png", "achievements-complete.json", "achievements-return-final.png",
        "mana-complete.json", "mana-cycle-003-main.png", "mana-cycle-004-mana_potion.png", "mana-cycle-008-mana_close.png",
        "mana-after-cycle.png", "relief-final-window.json", "relief-water-before-tap.png",
        "relief-water-after.png", "relief-star-live.json", "relief-star-live-before.png",
        "relief-star-live-after.png", "relief-water-immediate.json", "relief-water-immediate-before.png",
        "relief-water-immediate-after.png", "relief-star-immediate.json",
        "relief-star-immediate-before.png", "relief-star-immediate-after.png",
    ),
    "diagnostics/daily-reward": (
        "cycle-result.json", "cycle-000-main.png", "cycle-008-daily_reward.png",
        "cycle-012-daily_after_claim.png", "cycle-021-daily_return.png", "cycle-after.png",
    ),
    "diagnostics/event-potion": ("cycle-result.json", "before.png", "after.png", "final-status.json"),
    "diagnostics/2026-10-03/current-event": ("before-live.png", "live-after.png", "live-result.json"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path_allowed(relative: Path) -> bool:
    parts = tuple(part.lower() for part in relative.parts)
    return (not any(part in EXCLUDED_DIRS for part in parts)
            and not any(fnmatchcase(part, pattern)
                        for part in parts for pattern in EXCLUDED_PATTERNS))


def collect_files(root: Path, *, full: bool = False) -> dict[str, Path]:
    """Map archive-relative names to inputs without scanning excluded root trees."""
    files = {"mumu-panel.exe": root / "release" / "mumu-panel.exe"}
    for name in FULL_FILES if full else PORTABLE_FILES:
        files[name] = root / name
    # A developer's enabled tasks, paths and purchases must never become release defaults.
    files['config.json'] = root / 'config.example.json'
    for directory in FULL_DIRS if full else PORTABLE_DIRS:
        base = root / directory
        if not full and directory != "web" and not base.is_dir():
            raise FileNotFoundError(f"Missing release directory: {base}")
        for path in sorted(base.rglob("*")):
            relative = path.relative_to(root)
            if not path.is_file():
                continue
            if full:
                if path.suffix not in FULL_SUFFIXES or "__pycache__" in relative.parts:
                    continue
            elif not portable_path_allowed(relative):
                continue
            files[relative.as_posix()] = path
    if full:
        for directory, names in FULL_EVIDENCE.items():
            for name in names:
                relative = f"{directory}/{name}"
                if (root / relative).is_file():
                    files[relative] = root / relative
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing release file: {path}")
    return files


def build_manifest(root: Path, files: dict[str, Path]) -> dict:
    version = read_version(root)
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?", version):
        raise ValueError(f"Invalid VERSION: {version!r}")
    exe = files["mumu-panel.exe"]
    return {
        "version": version,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exe_sha256": sha256_file(exe),
        "exe_bytes": exe.stat().st_size,
        "upstream": UPSTREAM,
        "files": len(files) + 2,  # version.json and logs/.gitkeep
        "asset_files": sum(name.startswith("assets/") for name in files),
        "model_files": sum(name.startswith("models/") for name in files),
        "source_sha256": source_sha256(root),
    }


def source_sha256(root: Path) -> str:
    """Bind the executable build to runtime code, templates, model and public policy."""
    paths=list(Path(root).glob('*.py'))
    for folder in ('drift','alas_core','web','assets','models','licenses'):
        paths.extend(p for p in (Path(root)/folder).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo'))
    paths.extend(Path(root)/n for n in ('config.example.json','VERSION','requirements.txt','requirements-build.txt'))
    digest=hashlib.sha256()
    for path in sorted(set(paths)):
        if not path.is_file(): continue
        digest.update(path.relative_to(root).as_posix().encode('utf-8')+b'\0')
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def package_release(root: Path = ROOT, *, full: bool = False) -> dict:
    root = Path(root).resolve()
    files = collect_files(root, full=full)
    config=json.loads(files['config.json'].read_text(encoding='utf-8-sig'))
    if config.get('return_parallel_enabled') is True or any(r.get('enabled') is not False for r in config['rules']):
        raise ValueError('Public config.example.json must have all automation features disabled')
    manifest = build_manifest(root, files)
    prefix = "" if full else f"{PORTABLE_NAME}/"
    name = "mumu-framework.zip" if full else f"mumu-clicker-{manifest['version']}-win64.zip"
    archive = root / "release" / name
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as out:
            if prefix:
                out.writestr(prefix, b"")
            for name, path in sorted(files.items()):
                out.write(path, prefix + name)
            out.writestr(prefix + "logs/.gitkeep", b"")
            out.writestr(prefix + "version.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        size = temporary.stat().st_size
        if not full and size > MAX_PORTABLE_BYTES:
            raise ValueError(f"Portable archive is {size:,} bytes; limit is {MAX_PORTABLE_BYTES:,} bytes")
        temporary.replace(archive)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "archive": str(archive), "bytes": size,
        "size_mb": round(size / 1_000_000, 2), "size_mib": round(size / (1024 * 1024), 2),
        "sha256": sha256_file(archive),
        "files": manifest["files"], "asset_files": manifest["asset_files"],
        "model_files": manifest["model_files"],
        "top_level": sorted({name.split("/")[0] for name in (*files, "logs", "version.json")})
                     if full else [PORTABLE_NAME + "/"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Build the legacy mumu-framework.zip with source, tests and evidence")
    args = parser.parse_args()
    print(json.dumps(package_release(full=args.full), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
