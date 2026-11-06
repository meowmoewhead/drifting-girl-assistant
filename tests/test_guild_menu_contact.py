"""One 80 ms contact for guild selection/close, with no purchase or battle changes."""
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock
from drift.device.adb import AdbDevice

class GuildMenuContactTests(unittest.TestCase):
    def device(self):
        result=AdbDevice(dict(resolution=[720,1280]),threading.Event())
        result.command=Mock()
        return result

    def test_each_scoped_menu_control_sends_one_stationary_contact(self):
        for name in ('driftinggirl_guild_potion_select','driftinggirl_guild_material_select','driftinggirl_guild_close'):
            device=self.device();device.click(SimpleNamespace(name=name,button=(633,244,700,304)))
            device.command.assert_called_once_with('shell','input','swipe','666','274','666','274','80')

    def test_purchase_confirm_and_unrelated_clicks_keep_original_transport(self):
        for name in (None,'driftinggirl_guild_material_buy','driftinggirl_guild_material_confirm','other'):
            device=self.device();device.click(SimpleNamespace(name=name,button=(10,20,26,38)))
            device.command.assert_called_once_with('shell','input','tap','18','29')

    def test_stopped_control_cannot_send_down(self):
        device=self.device();device.stop.set()
        with self.assertRaises(InterruptedError):
            device.click(SimpleNamespace(name='driftinggirl_guild_close',button=(633,244,700,304)))
        device.command.assert_not_called()

    def test_failed_contact_is_not_retried(self):
        device=self.device();device.command.side_effect=RuntimeError('input failed')
        with self.assertRaises(RuntimeError):
            device.click(SimpleNamespace(name='driftinggirl_guild_close',button=(633,244,700,304)))
        self.assertEqual(device.command.call_count,1)

if __name__=='__main__':unittest.main()
