from contextlib import contextmanager
from copy import deepcopy
import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import struct
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import zipfile

from drift.update.package import Version, unpack, sha256, checksum_for, Transaction, program_path
from drift.update.manager import pick_release, UpdateManager, Network
from drift.desktop.client import Client, validate_address
from drift.desktop.service import extend_handler


def fake_executable():
    value=bytearray(512)
    value[:2]=b'MZ';struct.pack_into('<I',value,0x3c,64)
    value[64:68]=b'PE\0\0';struct.pack_into('<H',value,68,0x8664);struct.pack_into('<H',value,88,0x20b)
    return bytes(value)


def package(folder, version='0.3.0', extra=None):
    content = {'mumu-panel.exe': fake_executable(), 'VERSION': version.encode(),
               'assets/icon.png': b'new icon', 'web/app.js': b'new UI',
               'config.json': b'NEVER replace saved settings', 'logs/.gitkeep': b''}
    content.update(extra or {})
    manifest = {'version': version, 'update_schema': 1,
                'exe_sha256': hashlib.sha256(content['mumu-panel.exe']).hexdigest(),
                'payload_sha256': {k: hashlib.sha256(v).hexdigest() for k, v in content.items() if program_path(k)}}
    content['version.json'] = json.dumps(manifest).encode()
    archive = Path(folder) / 'update.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for name, value in content.items():
            z.writestr('mumu-clicker/' + name, value)
    return archive


class PackageTests(unittest.TestCase):
    def test_version_order_and_preview(self):
        values = ['0.2.0-alpha.2', '0.2.0-alpha.10', '0.2.0-beta', '0.2.0-rc.1', '0.2.0', '0.10.0', '1.0.0']
        self.assertEqual(sorted(values, key=Version), values)
        self.assertEqual(Version('v1.0.0+build'), Version('1.0.0'))
        for bad in ['1', '../../x', '01.0.0', '1.0.0-01', '1.0.0-rc..2']:
            with self.assertRaises(ValueError):
                Version(bad)

    def test_only_new_releases_and_matching_windows_package(self):
        def release(v, preview=False):
            return {'tag_name': 'v'+v, 'prerelease': preview, 'assets': [
                {'name': f'mumu-clicker-{v}-win64.zip', 'size': 20, 'browser_download_url': 'zip'},
                {'name': 'SHA256SUMS.txt', 'browser_download_url': 'sums'}]}
        releases = [release('0.2.0'), release('0.3.0-rc.1', True), release('0.1.0')]
        self.assertEqual(pick_release(releases, '0.1.5', False)['version'], '0.2.0')
        self.assertEqual(pick_release(releases, '0.1.5', True)['version'], '0.3.0-rc.1')
        self.assertIsNone(pick_release(releases, '0.3.0', True))
        releases[0]['assets'].pop()
        self.assertIsNone(pick_release(releases, '0.1.5', False))

    def test_checksum_requires_exact_unique_filename(self):
        digest = 'a' * 64
        self.assertEqual(checksum_for(digest + '  package.zip', 'package.zip'), digest)
        for text in [digest + '  other.zip', (digest + '  package.zip\n') * 2, 'bad package.zip']:
            with self.assertRaises(ValueError):
                checksum_for(text, 'package.zip')

    def test_hash_failure_leaves_no_payload(self):
        with tempfile.TemporaryDirectory() as d:
            archive = package(d)
            with self.assertRaises(ValueError):
                unpack(archive, Path(d)/'payload', '0'*64, '0.2.0')
            self.assertFalse((Path(d)/'payload').exists())

    def test_config_and_records_are_never_installed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'installed'
            root.mkdir()
            originals = {'config.json': b'custom enabled switches', 'training_state.json': b'paid inputs',
                         'parallel_skill_state.json': b'expiry', 'daily_state.json': b'daily', 'task_schedule_state.json': b'next run',
                         'updater_settings.json': b'proxy', 'desktop_settings.json': b'local connection'}
            for name, value in originals.items():
                (root/name).write_bytes(value)
            (root/'mumu-panel.exe').write_bytes(b'old executable')
            archive = package(d)
            details = unpack(archive, Path(d)/'payload', sha256(archive), '0.2.0')
            self.assertNotIn('config.json', details['files'])
            txn = Transaction(root, Path(d)/'payload', Path(d)/'backup', details['files'])
            txn.apply()
            self.assertEqual((root/'mumu-panel.exe').read_bytes(), fake_executable())
            for name, value in originals.items():
                self.assertEqual((root/name).read_bytes(), value)
            txn.rollback()
            self.assertEqual((root/'mumu-panel.exe').read_bytes(), b'old executable')
            self.assertFalse((root/'assets/icon.png').exists())
            for name, value in originals.items():
                self.assertEqual((root/name).read_bytes(), value)

    def test_malicious_members_are_rejected(self):
        for name in ['../escape', 'C:/escape', 'assets/../../config.json', 'assets/CON.png',
                     'assets/foo:stream', 'assets/foo.', 'training_state.json', 'assets/custom/user.png', 'drift/runtime.py']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                archive = package(d, extra={name: b'bad'})
                with self.assertRaises(ValueError):
                    unpack(archive, Path(d)/'payload', sha256(archive), '0.2.0')

    def test_no_same_version_or_downgrade(self):
        for version in ['0.1.0', '0.2.0']:
            with tempfile.TemporaryDirectory() as d:
                archive = package(d, version)
                with self.assertRaises(ValueError):
                    unpack(archive, Path(d)/'payload', sha256(archive), '0.2.0')

    def test_non_windows_image_is_rejected_before_install(self):
        with tempfile.TemporaryDirectory() as d:
            archive=package(d,extra={'mumu-panel.exe':b'damaged executable'})
            with self.assertRaisesRegex(ValueError,'Windows'):
                unpack(archive,Path(d)/'payload',sha256(archive),'0.2.0')

    def test_changed_staged_file_rejected_before_install(self):
        with tempfile.TemporaryDirectory() as d:
            archive = package(d)
            details = unpack(archive, Path(d)/'payload', sha256(archive), '0.2.0')
            (Path(d)/'payload/mumu-panel.exe').write_bytes(b'tampered')
            root = Path(d)/'installed'
            root.mkdir()
            (root/'mumu-panel.exe').write_bytes(b'old')
            with self.assertRaises(ValueError):
                Transaction(root, Path(d)/'payload', Path(d)/'backup', details['files']).apply()
            self.assertEqual((root/'mumu-panel.exe').read_bytes(), b'old')
            self.assertFalse((Path(d)/'backup').exists())

    def test_partially_failed_install_rolls_back(self):
        with tempfile.TemporaryDirectory() as d:
            archive = package(d)
            details = unpack(archive, Path(d)/'payload', sha256(archive), '0.2.0')
            root = Path(d)/'installed'
            root.mkdir()
            (root/'mumu-panel.exe').write_bytes(b'old')
            from drift.update import package as module
            original = module.os.replace
            def replace(a, b):
                if Path(b) == root/'VERSION':
                    raise OSError('simulated full disk')
                return original(a, b)
            txn = Transaction(root, Path(d)/'payload', Path(d)/'backup', details['files'])
            with patch.object(module.os, 'replace', side_effect=replace), self.assertRaises(OSError):
                txn.apply()
            self.assertEqual((root/'mumu-panel.exe').read_bytes(), b'old')
            self.assertEqual(txn.journal['phase'], 'rolled_back')

    def test_install_never_stops_running_worker(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'VERSION').write_text('0.2.0')
            manager = UpdateManager(d)
            app = SimpleNamespace(running=lambda: True, stop=Mock())
            with self.assertRaisesRegex(ValueError, '停止'):
                manager.install(app, None)
            app.stop.assert_not_called()

    def test_proxy_and_download_origin_are_scoped(self):
        for bad in ['file:///x', 'http://user:pass@host', 'http://host/path']:
            with self.assertRaises(ValueError):
                Network(bad)
        network = Network()
        for bad in ['http://github.com/x', 'https://evil.invalid/file', 'https://github.com/other/repo/releases/download/x/a.zip', 'https://api.github.com/user']:
            with self.assertRaises(ValueError):
                network.open(bad)


class DesktopAPITests(unittest.TestCase):
    @contextmanager
    def service(self):
        from drift.web.server import _desktop_base_factory
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'VERSION').write_text('0.2.0')
            app = SimpleNamespace(root=root, thread=None, running=lambda: False,
                                  status=lambda: {'features': [], 'rules': [], 'running': False},
                                  change_feature=Mock(return_value={'saved': True}))
            server = ThreadingHTTPServer(('127.0.0.1', 0), extend_handler(_desktop_base_factory, app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                yield Client(f'http://127.0.0.1:{server.server_port}'), app
            finally:
                server.shutdown()
                server.server_close()
                thread.join(3)

    def test_native_session_and_existing_mutation_no_html(self):
        with self.service() as (client, app):
            session = client.connect()
            self.assertEqual(session['protocol'], 1)
            self.assertEqual(session['version'], '0.2.0')
            client.request('/api/features/change', {'id': 'fixture', 'enabled': True})
            app.change_feature.assert_called_once_with({'id': 'fixture', 'enabled': True})
            self.assertEqual(client.request('/api/desktop/update')['phase'], 'idle')

    def test_cross_origin_native_session_rejected(self):
        with self.service() as (client, app):
            for headers in [{}, {'X-Desktop-Client': 'drift-native-v1', 'Origin': 'https://evil.invalid'}]:
                with self.assertRaises(HTTPError) as result:
                    urlopen(Request(client.address+'/api/desktop/session', headers=headers))
                self.assertEqual(result.exception.code, 400)

    def test_invalid_token_cannot_mutate_and_is_not_replayed(self):
        with self.service() as (client, app):
            client.connect()
            client.session['token'] = 'stale'
            with self.assertRaises(ValueError):
                client.request('/api/features/change', {'id': 'fixture', 'enabled': True})
            app.change_feature.assert_not_called()
            self.assertIsNone(client.session)

    def test_settings_only_write_their_own_file(self):
        with self.service() as (client, app):
            client.connect()
            (app.root/'config.json').write_bytes(b'untouched')
            client.request('/api/desktop/update/settings', {'preview': True, 'proxy': ''})
            self.assertEqual((app.root/'config.json').read_bytes(), b'untouched')
            self.assertTrue(json.loads((app.root/'updater_settings.json').read_text())['preview'])

    def test_desktop_addresses_are_loopback_only(self):
        self.assertEqual(validate_address('http://127.0.0.1:18765/'), 'http://127.0.0.1:18765')
        for bad in ['https://example.com', 'http://192.0.2.10:8765', 'http://user@127.0.0.1:8765', 'file:///a', 'http://127.0.0.1:8765/x']:
            with self.assertRaises(ValueError):
                validate_address(bad)


if __name__ == '__main__':
    unittest.main()
