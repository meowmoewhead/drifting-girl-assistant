import json
from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import Mock, patch

from drift.desktop import theme
from drift.desktop.panel import Panel


class ThemePreferencesTests(unittest.TestCase):
    def test_local_preference_merge_preserves_connection_and_runtime_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'config.json').write_bytes(b'{"runtime":true}')
            (root/'workflow_context_state.json').write_bytes(b'{"spent":3}')
            theme.save_preferences(root, address='http://127.0.0.1:19865', future='keep')
            theme.save_preferences(root, theme='dark')
            theme.save_preferences(root, address='http://127.0.0.1:8765')
            self.assertEqual(theme.read_preferences(root), dict(theme='dark', address='http://127.0.0.1:8765', future='keep'))
            self.assertEqual((root/'config.json').read_bytes(), b'{"runtime":true}')
            self.assertEqual((root/'workflow_context_state.json').read_bytes(), b'{"spent":3}')

    def test_bad_preferences_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'desktop_settings.json'
            for raw in (b'{broken', b'[]'):
                path.write_bytes(raw)
                self.assertEqual(theme.read_preferences(directory), {})
                with self.assertRaises(ValueError):
                    theme.save_preferences(directory, theme='dark')
                self.assertEqual(path.read_bytes(), raw)

    def test_palette_text_and_status_contrast(self):
        def luminance(color):
            channels = [int(color[i:i+2], 16)/255 for i in (1, 3, 5)]
            channels = [c/12.92 if c<=.04045 else ((c+.055)/1.055)**2.4 for c in channels]
            return sum(c*w for c,w in zip(channels,(.2126,.7152,.0722)))
        for mode, palette in theme.PALETTES.items():
            pairs = [(palette[k], palette['WHITE']) for k in ('INK','TEXT','MUTED','TEAL','STATUS','DANGER')]
            pairs += [(palette['MUTED'],palette['BG']), (palette['ON_ACCENT'],palette['ACCENT_BG'])]
            pairs += list(palette['BADGES'].values())
            for a,b in pairs:
                low,high = sorted((luminance(a),luminance(b)))
                self.assertGreaterEqual((high+.05)/(low+.05),4.5,(mode,a,b))


class ThemeSwitchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        theme.save_preferences(self.root, theme='light')
        self.window = tk.Tk()
        self.window.withdraw()
        self.client = Mock(address='http://127.0.0.1:19865')
        with patch.object(Panel, 'poll'):
            self.panel = Panel(self.window, self.root, self.client)
        self.panel.state = {'settings': {'poll_interval': .6, 'max_consecutive_clicks': 20}}
        self.panel.build_nav()
        self.panel.page = 'settings'
        self.panel.render_page()
        self.window.update_idletasks()

    def tearDown(self):
        self.panel.close()
        self.directory.cleanup()

    def choose(self, label):
        self.panel.theme_choice.set(label)
        self.panel.change_theme()
        self.window.update_idletasks()

    def test_repeated_live_switch_keeps_edits_widgets_and_does_not_post(self):
        variable = self.panel.fields['poll_interval'][0]
        variable.set('1.25')
        original_fields = self.panel.fields
        widgets = tuple(self.panel.content.winfo_children())
        original_images = None
        for i in range(10):
            self.choose('深色' if i%2==0 else '浅色')
            self.assertIs(self.panel.fields, original_fields)
            self.assertIs(self.panel.fields['poll_interval'][0], variable)
            self.assertEqual(variable.get(), '1.25')
            self.assertTrue(self.panel.dirty())
            self.assertEqual(tuple(self.panel.content.winfo_children()), widgets)
            self.assertEqual(self.panel.page, 'settings')
            assets = {k:tuple(str(p) for p in v) for k,v in self.window._theme_assets.items()}
            if original_images is None: original_images = assets
            self.assertEqual(assets, original_images)
            for names in assets.values():
                for image in names:
                    self.assertGreater(self.window.tk.call('image','width',image),0)
        self.client.request.assert_not_called()
        self.assertEqual(theme.read_preferences(self.root)['theme'], 'light')

    def test_opened_combobox_and_text_surfaces_recolor(self):
        combo = self.panel.theme_selector
        popup = self.window.tk.call('ttk::combobox::PopdownWindow', str(combo))
        text = theme.text_colors(tk.Text(self.panel.content))
        text.insert('end','test selection')
        text.tag_add('sel','1.0','1.4')
        text.tag_configure('time', foreground=theme.MUTED)
        self.choose('深色')
        self.assertEqual(text.get('1.0','end-1c'), 'test selection')
        self.assertEqual(tuple(map(str,text.tag_ranges('sel'))), ('1.0','1.4'))
        self.assertEqual(text.cget('background'),theme.PALETTES['dark']['WHITE'])
        self.assertEqual(self.window.tk.call(str(popup)+'.f.l','cget','-background'),theme.PALETTES['dark']['FIELD'])
        self.choose('浅色')
        self.assertEqual(text.cget('foreground'),theme.PALETTES['light']['TEXT'])

    def test_saved_theme_restored_and_system_change_is_local(self):
        self.choose('深色')
        self.panel.close()
        self.window=tk.Tk(); self.window.withdraw()
        with patch.object(Panel,'poll'):
            self.panel=Panel(self.window,self.root,self.client)
        self.assertEqual(self.window._drift_theme,'dark')
        self.assertEqual(self.panel.theme_choice.get(),'深色')
        with patch.object(theme,'system_theme',return_value='dark'):
            self.choose('跟随系统')
        self.window.after_cancel(self.panel.theme_timer)
        with patch.object(theme,'system_theme',return_value='light'):
            self.panel.watch_system_theme()
        self.assertEqual(self.window._drift_theme,'light')
        self.assertEqual(theme.read_preferences(self.root)['theme'],'system')
        self.client.request.assert_not_called()

    def test_save_failure_keeps_current_theme(self):
        with patch.object(theme,'save_preferences',side_effect=OSError('read only')):
            self.choose('深色')
        self.assertEqual(self.window._drift_theme,'light')
        self.assertEqual(self.panel.theme_choice.get(),'浅色')
        self.assertIn('主题保存失败',self.panel.notice.get())


if __name__ == '__main__':
    unittest.main()
