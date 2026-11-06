"""邮箱是每日任务:本周期领取过就等到下一个 08:00,不再按新邮件提示反复触发。

用户要求:"邮箱奖励也是每天触发一次就行"。这里守住三件事:
1. 邮箱在功能注册表里属于每日任务(daily),不再是实时触发;
2. 它的入口与收尾规则挂进了每日调度表,每日周期翻转时会被重新放行;
3. 每日完成判定的规则集合包含"全部领取/关闭/超上限关闭/确认返回"。
"""
import inspect
import unittest

from drift.tasks.daily import DAILY_ENTRIES, DAILY_RETURNS, DailySchedule
from drift.tasks.features import DAILY_FEATURES, FEATURE_BY_ID, execution_mode
from drift.tasks.registry import TASK_SPECS


class MailboxDailyTests(unittest.TestCase):
    def test_mailbox_is_a_daily_feature(self):
        self.assertIn("mailbox", DAILY_FEATURES)
        self.assertEqual(execution_mode("mailbox"), "daily")
        self.assertEqual(TASK_SPECS["mailbox"].mode, "daily")

    def test_mailbox_entry_and_return_are_scheduled(self):
        self.assertEqual(DAILY_ENTRIES["driftinggirl_mailbox_open"], "mailbox")
        self.assertEqual(DAILY_RETURNS["mailbox"], "driftinggirl_mailbox_return")
        # 入口与收尾规则必须真的属于邮箱功能,否则调度会挂到别的流程上
        self.assertIn("driftinggirl_mailbox_open", FEATURE_BY_ID["mailbox"]["rules"])
        self.assertIn("driftinggirl_mailbox_return", FEATURE_BY_ID["mailbox"]["rules"])

    def test_daily_completion_rules_belong_to_the_mailbox_feature(self):
        # runtime 的每日完成判定必须覆盖邮箱的收尾路径
        from drift.runtime import Clicker

        source = inspect.getsource(Clicker.daily_completion_visible)
        for rule in ("driftinggirl_mailbox_claim_all", "driftinggirl_mailbox_close",
                     "driftinggirl_mailbox_limit_close", "driftinggirl_mailbox_return"):
            self.assertIn(rule, FEATURE_BY_ID["mailbox"]["rules"])
            self.assertIn(rule, source)

    def test_completion_covers_every_daily_feature(self):
        # PASS and 公会任务提交 use dedicated controllers (no DAILY_ENTRIES 入口规则);
        # the other daily features are opened by their entry rule and use rule entries.
        for feature in DAILY_FEATURES:
            self.assertIn(feature, DAILY_RETURNS, feature)
            if feature == 'pass_daily':
                from drift.tasks.pass_daily import PASS_ID, PassDaily
                self.assertEqual(DAILY_RETURNS[feature], PASS_ID)
                self.assertEqual(FEATURE_BY_ID[feature]['rules'], (PASS_ID,))
                self.assertTrue(callable(PassDaily.step))
            elif feature == 'guild_daily':
                from drift.tasks.guild_daily import GUILD_ID, GuildDaily
                self.assertEqual(DAILY_RETURNS[feature], GUILD_ID)
                self.assertEqual(FEATURE_BY_ID[feature]['rules'], (GUILD_ID,))
                self.assertTrue(callable(GuildDaily.step))
            else:
                self.assertIn(feature, set(DAILY_ENTRIES.values()), feature)
            self.assertTrue(DAILY_RETURNS[feature] in FEATURE_BY_ID[feature]["rules"], feature)

    def test_daily_features_report_a_condition_in_the_execution_payload(self):
        # 面板的任务行用 execution.condition 显示触发说明;定时/每日任务由 TaskSchedule
        # 生成该字段,容易漏(原先只填实时条件,每日功能会显示空白)。
        import json
        import tempfile
        from pathlib import Path

        from drift.tasks.daily import DailySchedule
        from drift.tasks.features import feature_states
        from drift.tasks.schedule import TaskSchedule

        config = json.loads(Path("config.json").read_text(encoding="utf-8-sig"))
        with tempfile.TemporaryDirectory() as directory:
            daily = DailySchedule(Path(directory), config["serial"])
            schedule = TaskSchedule(Path(directory), config["serial"])
            states = {f["id"]: f for f in
                      feature_states(config, daily.status(), schedule.status(config, daily.status(), False))}
        for feature in DAILY_FEATURES:
            self.assertTrue(states[feature]["execution"]["condition"], feature)
        self.assertIn("08:00", states["mailbox"]["execution"]["condition"])

    def test_marked_complete_blocks_the_same_period_and_reopens_after_reset(self):
        from datetime import datetime
        import tempfile
        from pathlib import Path

        from drift.tasks.daily import BEIJING

        with tempfile.TemporaryDirectory() as directory:
            now = [datetime(2026, 10, 10, 9, tzinfo=BEIJING)]
            schedule = DailySchedule(Path(directory), "test", lambda: now[0])
            self.assertFalse(schedule.is_complete("mailbox"))
            self.assertTrue(schedule.mark_complete("mailbox", schedule.period()))
            self.assertTrue(schedule.is_complete("mailbox"))
            # 同一天稍后再检查仍然是完成状态
            now[0] = datetime(2026, 10, 10, 20, tzinfo=BEIJING)
            self.assertTrue(schedule.is_complete("mailbox"))
            # 跨过 08:00 后重新变成待办
            now[0] = datetime(2026, 10, 11, 8, tzinfo=BEIJING)
            self.assertFalse(schedule.is_complete("mailbox"))
            # 重启后完成记录仍在
            restarted = DailySchedule(Path(directory), "test", lambda: now[0])
            self.assertFalse(restarted.is_complete("mailbox"))
            now[0] = datetime(2026, 10, 10, 23, tzinfo=BEIJING)
            self.assertTrue(restarted.is_complete("mailbox"))


if __name__ == "__main__":
    unittest.main()
