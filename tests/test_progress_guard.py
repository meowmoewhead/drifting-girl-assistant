"""进展检测与异常分类的单元测试。"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drift.core.exceptions import (DriftError, GameStuck, RetryableStage,
                                   Stopped, TaskError)
from drift.core.progress import ProgressGuard


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class ExceptionTaxonomyTests(unittest.TestCase):
    def test_drift_errors_are_runtime_errors(self):
        # 历史代码与测试用 except RuntimeError 捕获,新类型必须保持兼容。
        for kind in (Stopped, RetryableStage, GameStuck, TaskError):
            self.assertTrue(issubclass(kind, RuntimeError))
            self.assertTrue(issubclass(kind, DriftError))

    def test_retryable_and_stuck_are_distinct(self):
        self.assertFalse(issubclass(RetryableStage, GameStuck))
        self.assertFalse(issubclass(GameStuck, RetryableStage))


class ProgressGuardTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.guard = ProgressGuard(self.clock, window=60, limit=3)

    def test_staying_in_one_stage_is_never_stuck(self):
        for _ in range(50):
            self.assertFalse(self.guard.note('main'))
            self.clock.advance(1)

    def test_first_transition_needs_a_previous_stage(self):
        self.assertFalse(self.guard.note('main'))
        self.assertFalse(self.guard.note('main'))
        self.assertFalse(self.guard.note('mailbox'))

    def test_ping_pong_raises_after_limit(self):
        self.guard.note('main')
        results = []
        for _ in range(6):
            results.append(self.guard.note('mailbox'))
            results.append(self.guard.note('main'))
            self.clock.advance(1)
        self.assertEqual(results[-1], True)
        self.assertFalse(any(results[:4]))

    def test_counts_expire_outside_the_window(self):
        self.guard.note('main')
        for _ in range(3):
            self.assertFalse(self.guard.note('mailbox'))
            self.assertFalse(self.guard.note('main'))
        # 等待超过窗口后,旧的计数必须过期,不能把正常慢循环判成卡住。
        self.clock.advance(120)
        self.assertFalse(self.guard.note('mailbox'))
        self.assertEqual(self.guard.status()['edges'].get('main->mailbox'), 1)

    def test_linear_progress_is_not_stuck(self):
        self.guard.note('main')
        for stage in ('mailbox', 'mailbox_done', 'mailbox_return', 'main',
                      'achievements_tab', 'achievements_claim', 'main'):
            self.assertFalse(self.guard.note(stage))
            self.clock.advance(2)

    def test_status_reports_current_stage(self):
        self.guard.note('main')
        self.guard.note('mailbox')
        status = self.guard.status()
        self.assertEqual(status['stage'], 'mailbox')
        self.assertEqual(status['edges'], {'main->mailbox': 1})


if __name__ == '__main__':
    unittest.main()
