from datetime import datetime,timedelta,timezone
import tempfile
from types import SimpleNamespace as NS
import unittest

from drift.core.outcome import Outcome,controller_result,result
from drift.tasks.schedule import TaskSchedule


class TaskOutcomeTests(unittest.TestCase):
    def test_no_items_is_checked_skip_and_expired_failure_is_blocked(self):
        done=NS(journal={'submitted':0,'categories':[0,1]},hold=None)
        self.assertEqual(controller_result(done,True).kind,Outcome.SKIPPED)
        self.assertEqual(controller_result(done,True).delay(86400),86400)
        failed=NS(journal={'phase':'blocked','failure':'原窗口到期'},hold=None)
        self.assertEqual(controller_result(failed,False).kind,Outcome.BLOCKED)
        self.assertEqual(controller_result(failed,False).delay(86400),1800)

    def test_timeout_skip_uses_failure_cooldown(self):
        self.assertEqual(result(False,outcome='skipped').delay(172800),1800)
        self.assertEqual(result(True,outcome='skipped').delay(172800),172800)

    def test_all_four_outcomes_are_persisted_and_manual_date_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            now=datetime(2026,11,6,tzinfo=timezone.utc);s=TaskSchedule(d,'serial',lambda:now)
            for kind in ('complete','retry','skipped','blocked'):
                s.finish('restaurant_upgrade',172800,success=kind in ('complete','skipped'),outcome=kind)
                self.assertEqual(s.entries['restaurant_upgrade']['outcome'],kind)
            date=now+timedelta(days=5);s.set_next('restaurant_upgrade',date.isoformat())
            s.finish('restaurant_upgrade',172800,success=False,outcome='blocked')
            self.assertEqual(s.next_run('restaurant_upgrade'),date)
            self.assertEqual(s.entries['restaurant_upgrade']['reason'],'manual')


if __name__=='__main__':unittest.main()
