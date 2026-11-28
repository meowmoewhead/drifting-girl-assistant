"""Daily delivery quotas block recognition but still let a delivery finish."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from drift.tasks.daily import BEIJING
from drift.tasks.meal_delivery import MealDelivery, saved_meal_day_status
from drift.tasks.restaurant_upgrade import RestaurantUpgrade
from drift.tasks.schedule import TaskSchedule


class MealDailyResetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reset = datetime(2026, 11, 28, 8, tzinfo=BEIJING)
        self.now = self.reset - timedelta(seconds=1)
        self.task = task = MealDelivery.__new__(MealDelivery)
        task.phase = 'idle'
        task.clock = lambda: self.now.timestamp()
        task.serial = 'test-emulator'
        task.root = self.root
        task.path = self.root/'meal_delivery_state.json'
        task.journal = dict(version=1, serial=task.serial, pending=False, phase='idle',
                            delivered=[dict(period='2026-11-27', input_id=str(i))
                                       for i in range(3)])
        task.meal = Mock()
        task.meal.order.return_value = None
        task.enabled = lambda: True
        task._check_startup = False
        task._reconcile_on_start = True
        task._reconcile_recent_submissions = Mock(return_value=False)
        task._main = Mock(return_value=False)
        task._confirmed = Mock(return_value=False)
        task.event = Mock()

    def test_saved_quota_controls_next_run_and_releases_at_eight(self):
        task = self.task
        task.path.write_text(json.dumps(task.journal), encoding='utf-8')
        saved = task.path.read_bytes()
        schedule = TaskSchedule(self.root, task.serial, lambda: self.now)
        daily = saved_meal_day_status(self.root, task.serial, self.now)
        self.assertEqual(daily['delivered_today'], 3)
        schedule.set_next('meal_delivery', None, daily)
        self.assertEqual(schedule.next_run('meal_delivery', daily), self.reset)
        self.assertFalse(schedule.ready('meal_delivery', daily))
        self.now = self.reset
        daily = saved_meal_day_status(self.root, task.serial, self.now)
        self.assertEqual(daily['delivered_today'], 0)
        self.assertTrue(schedule.ready('meal_delivery', daily))
        self.assertEqual(task.path.read_bytes(), saved)

    def test_capped_quota_skips_order_and_archive_recognition(self):
        task = self.task
        frame = object()
        self.assertFalse(task.offered(frame))
        self.assertFalse(task.step(frame))
        task.meal.order.assert_not_called()
        task._reconcile_recent_submissions.assert_not_called()
        self.now = self.reset
        self.assertFalse(task.offered(frame))
        task.meal.order.assert_called_once_with(frame)
        self.assertFalse(task.step(frame))
        task._reconcile_recent_submissions.assert_called_once_with()

    def test_third_deliveries_restore_remains_active(self):
        task = self.task
        task.phase = 'restore'
        task.journal.update(pending=True, settled=True)
        task.allowed = lambda: True
        frame = object()
        with patch.object(RestaurantUpgrade, 'step', return_value=True) as restore:
            self.assertTrue(task.step(frame, due=False))
        restore.assert_called_once_with(frame, due=False)

    def test_confirmation_after_eight_counts_on_the_submission_day(self):
        task = self.task
        task.journal['delivered'].pop()
        task.journal['last_input'] = dict(input_id='before-reset', at=self.now.timestamp())
        self.now = self.reset
        self.assertTrue(task._record_delivery())
        self.assertFalse(task._record_delivery())
        self.assertEqual(task.journal['delivered'][-1]['period'], '2026-11-27')
        self.assertEqual(task.daily_status()['delivered_today'], 0)
        task.journal['last_input'] = dict(input_id='after-reset', at=self.now.timestamp())
        self.assertTrue(task._record_delivery())
        self.assertEqual(task.daily_status()['delivered_today'], 1)
