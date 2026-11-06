"""Portable archive contract; all generated fixtures live in temporary folders."""
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from deploy import package_release as release


class PackageReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mumu-portable-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in release.PORTABLE_FILES:
            self.write(name, b"fixture\n")
        self.write("VERSION", b"0.1.0\n")
        self.write("config.json", b'\xef\xbb\xbf{"rules": [], "serial": "fixture"}\r\n')
        self.write("config/default.json", b'\xef\xbb\xbf{"rules": [], "serial": "fixture"}\r\n')
        self.write("release/mumu-panel.exe", b"MZ\x00portable-test-executable\xff")
        self.write("assets/sub/template.png", b"template")
        self.write("assets/sub/metadata.bin", b"include all resource extensions")
        self.write("models/recognizer.onnx", b"model")
        self.write("models/model.json", b"{}")
        self.write("web/index.html", b"<html></html>")
        self.write("licenses/dependency.LICENSE", b"license")
        self.write("docs/INTRO.md", b"# Illustrated introduction\n")
        self.write("docs/screenshots/console.png", b"public screenshot")

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def archive(self, *, full=False):
        result = release.package_release(self.root, full=full)
        return zipfile.ZipFile(result["archive"])

    def test_single_top_level_and_expected_layout(self):
        with self.archive() as archive:
            self.assertEqual({name.split("/")[0] for name in archive.namelist()}, {"mumu-clicker"})
            self.assertIn("mumu-clicker/", archive.namelist())
            self.assertEqual(Path(archive.filename).name, "mumu-clicker-0.1.0-win64.zip")
            for name in (*release.PORTABLE_FILES, "mumu-panel.exe", "version.json", "logs/.gitkeep"):
                self.assertIn("mumu-clicker/" + name, archive.namelist())

    def test_excludes_development_evidence_backups_and_private_reports(self):
        for directory in release.EXCLUDED_DIRS:
            self.write(f"{directory}/unwanted.png", b"exclude")
            self.write(f"assets/{directory}/unwanted.png", b"exclude nested")
        for name in ("template.png.before-edit", "template.png.bak-old", "CODEX_PRIVATE.md",
                     "REVIEW2_CODEX_PRIVATE.md", "app.py.before-refactor", "cached.pyc"):
            self.write(name, b"exclude")
            self.write("assets/sub/" + name, b"exclude nested")
        self.write("release/old.zip", b"not an input")
        self.write("logs/clicker.log", b"private user log")
        with self.archive() as archive:
            names = archive.namelist()
            for name in names:
                relative = Path(name).relative_to("mumu-clicker")
                self.assertTrue(release.portable_path_allowed(relative), name)
            self.assertFalse(any("tests/" in name or "diagnostics/" in name for name in names))
            self.assertEqual([name for name in names if "/logs/" in name], ["mumu-clicker/logs/.gitkeep"])
            self.assertEqual(archive.read("mumu-clicker/logs/.gitkeep"), b"")

    def test_resources_and_config_are_included_verbatim(self):
        with self.archive() as archive:
            for name in ("config.json", "assets/sub/template.png", "assets/sub/metadata.bin",
                         "models/recognizer.onnx", "models/model.json", "licenses/dependency.LICENSE",
                         "docs/INTRO.md", "docs/screenshots/console.png"):
                self.assertEqual(archive.read("mumu-clicker/" + name), (self.root / name).read_bytes())

    def test_manifest_fields_hashes_and_counts_match_archive(self):
        with self.archive() as archive:
            manifest = json.loads(archive.read("mumu-clicker/version.json"))
            self.assertEqual(set(manifest), {"version", "built_at", "exe_sha256", "exe_bytes",
                                             "upstream", "files", "asset_files", "model_files", "source_sha256", "update_schema", "payload_sha256"})
            exe = archive.read("mumu-clicker/mumu-panel.exe")
            self.assertEqual(manifest["exe_sha256"], hashlib.sha256(exe).hexdigest())
            self.assertEqual(manifest["exe_bytes"], len(exe))
            self.assertEqual(manifest["version"], "0.1.0")
            self.assertEqual(manifest["upstream"], release.UPSTREAM)
            self.assertIsNotNone(datetime.fromisoformat(manifest["built_at"]).tzinfo)
            self.assertEqual(manifest["files"], sum(not item.is_dir() for item in archive.infolist()))
            self.assertEqual(manifest["asset_files"], 2)
            self.assertEqual(manifest["model_files"], 2)

    def test_manifest_hash_matches_real_release_executable(self):
        # Read the real binary without running it or rebuilding it during unit tests.
        exe = release.ROOT / "release" / "mumu-panel.exe"
        if not exe.is_file():
            self.skipTest('Build artifact not created yet; synthetic executable coverage remains active')
        manifest = release.build_manifest(self.root, {"mumu-panel.exe": exe})
        self.assertEqual(manifest["exe_sha256"], hashlib.sha256(exe.read_bytes()).hexdigest())
        self.assertEqual(manifest["exe_bytes"], exe.stat().st_size)

    def test_missing_required_input_does_not_replace_existing_archive(self):
        archive = self.root / "release/mumu-clicker-0.1.0-win64.zip"
        archive.write_bytes(b"previous release")
        (self.root / "config/default.json").unlink()
        with self.assertRaises(FileNotFoundError):
            release.package_release(self.root)
        self.assertEqual(archive.read_bytes(), b"previous release")

    def test_size_limit_does_not_replace_existing_archive(self):
        archive = self.root / "release/mumu-clicker-0.1.0-win64.zip"
        archive.write_bytes(b"previous release")
        with patch.object(release, "MAX_PORTABLE_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "limit"):
                release.package_release(self.root)
        self.assertEqual(archive.read_bytes(), b"previous release")
        self.assertFalse(archive.with_suffix(".zip.tmp").exists())

    def test_invalid_version_cannot_escape_release_directory(self):
        self.write("VERSION", b"../../outside\n")
        with self.assertRaisesRegex(ValueError, "Invalid VERSION"):
            release.package_release(self.root)

    def test_full_retains_legacy_name_layout_source_and_evidence(self):
        for name in release.FULL_FILES:
            if not (self.root / name).exists():
                self.write(name, b"legacy source")
        for name in ("tests/test_fixture.py", "drift/runtime.py", "dev_tools/example.py",
                     "captures/mailbox-before-claim.png", "diagnostics/daily-reward/cycle-021-daily_return.png"):
            self.write(name, b"legacy fixture")
        with self.archive(full=True) as archive:
            self.assertEqual(Path(archive.filename).name, "mumu-framework.zip")
            self.assertIn("mumu-panel.exe", archive.namelist())
            self.assertIn("tests/test_fixture.py", archive.namelist())
            self.assertIn("drift/runtime.py", archive.namelist())
            self.assertIn("dev_tools/example.py", archive.namelist())
            self.assertIn("captures/mailbox-before-claim.png", archive.namelist())
            self.assertIn("diagnostics/daily-reward/cycle-021-daily_return.png", archive.namelist())
            self.assertFalse(any(name.startswith("mumu-clicker/") for name in archive.namelist()))

    def test_cli_full_switch_and_printed_summary(self):
        with patch("sys.argv", ["package_release.py", "--full"]), \
                patch.object(release, "package_release", return_value={"bytes": 123}) as package, \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            release.main()
        package.assert_called_once_with(full=True)
        self.assertEqual(json.loads(output.getvalue()), {"bytes": 123})


if __name__ == "__main__":
    unittest.main()
