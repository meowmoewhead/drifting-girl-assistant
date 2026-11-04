"""guild_travel 的出口兜底与"中止不拖垮主循环"的回归。

2026-10-26 18:12 事故:`teleport-confirm` 之后 600 秒内没有任何阶段推进,失败瞬间
那一帧的出口判据(normal_idle + g-icon + guild-location)其实已经成立,只是三帧确认
还差两帧就被阶段截止判死;随后这个 RuntimeError 一路冒到 runner 主循环,整个自动化
停摆 3 h 31 min。这里把两条兜底钉住:临近截止允许单帧放行、中止后不再反复抛错。
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drift.tasks.guild_daily import GuildDaily  # noqa: E402


class GuildTravelFallbackTests(unittest.TestCase):
    def controller(self, directory, *, now=500.):
        c = GuildDaily.__new__(GuildDaily)
        c.clock = lambda: now
        c.path = Path(directory) / 'guild_daily_state.json'
        c.journal = dict(pending=True, phase='guild_travel', deadline=1000.,
                         phase_deadline=520., attempts={})
        c.phase = 'guild_travel'
        c.event = Mock()
        c.done = None
        c.hold = None
        c.next_at = 0
        c.seen = 0
        c.since = None
        c.normal_idle = lambda frame: True
        c.visual = SimpleNamespace(point=lambda frame, name: (679, 1013))
        c._confirmed = lambda key, frame: False
        c.phases = []
        c._set = lambda phase, seconds=120: c.phases.append(phase)
        return c

    def test_single_frame_confirmation_near_the_deadline_advances(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.controller(d)
            frame = object()
            c.step(frame, due=False, period='2026-10-26')
            self.assertEqual(c.phases, ['guild_at'])

    def test_single_frame_is_not_enough_while_budget_remains(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.controller(d, now=400.)  # 距 phase_deadline 还有 120 秒
            c.step(object(), due=False, period='2026-10-26')
            self.assertEqual(c.phases, [])

    def test_waiting_logs_a_diagnostic_line_at_most_every_thirty_seconds(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.controller(d, now=400.)
            c.step(object(), due=False, period='2026-10-26')
            first = c.event.call_count
            self.assertEqual(first, 1)
            c.step(object(), due=False, period='2026-10-26')
            self.assertEqual(c.event.call_count, first)  # 30 秒内不再刷日志
            c.clock = lambda: 431.
            c.step(object(), due=False, period='2026-10-26')
            self.assertEqual(c.event.call_count, first + 1)

    def test_abandon_clears_pending_and_keeps_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.controller(d)
            c.journal.update(attempts={'teleport-confirm': 1}, submitted=0)
            self.assertTrue(c.abandon())
            saved = json.loads(c.path.read_text(encoding='utf-8'))
            self.assertFalse(saved['pending'])
            self.assertEqual(saved['phase'], 'idle')
            self.assertEqual(saved['abandoned_phase'], 'guild_travel')
            self.assertEqual(saved['attempts'], {'teleport-confirm': 1})
            self.assertFalse(c.active)


if __name__ == '__main__':
    unittest.main()
