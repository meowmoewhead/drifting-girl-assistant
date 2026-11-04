"""Task controls must preserve complete workflows and persisted choices."""
from copy import deepcopy
from http.server import ThreadingHTTPServer
import json
import re
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from tests import test_clicker as helpers
from app import make_handler
from features import FEATURES, FEATURE_BY_ID, RULE_FEATURE, feature_states


class FeatureTests(unittest.TestCase):
    setUp = helpers.FrameworkTests.setUp
    tearDown = helpers.FrameworkTests.tearDown
    rule = helpers.FrameworkTests.rule
    wait_for = helpers.FrameworkTests.wait_for

    def feature(self, key):
        rules = []
        for identifier in FEATURE_BY_ID[key]["rules"]:
            rule = self.rule()
            rule.update(id=identifier, name=identifier)
            rules.append(rule)
        self.app.persist()
        return rules

    def test_all_presets_are_mapped_exactly_once(self):
        # 84 = 原79条 + 每日PASS、公会任务提交、餐品强化 + 更新提示（识别/返回）。
        identifiers = [rule for f in FEATURES for rule in f["rules"]]
        self.assertEqual(len(identifiers), 84)
        self.assertEqual(len(set(identifiers)), 84)
        self.assertEqual(set(RULE_FEATURE), set(identifiers))

    def test_feature_switch_is_atomic_persistent_and_isolated(self):
        self.feature("mailbox")
        unrelated = self.rule()
        unrelated_before = deepcopy(unrelated)
        self.app.change_feature({"id": "mailbox", "enabled": False})
        saved = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        self.assertFalse(any(r["enabled"] for r in saved["rules"] if r["id"] != unrelated["id"]))
        self.assertEqual(saved["rules"][-1], unrelated_before)
        self.assertFalse(next(f for f in feature_states(saved) if f["id"] == "mailbox")["enabled"])
        self.app.change_feature({"id": "mailbox", "enabled": True})
        self.assertTrue(all(r["enabled"] for r in self.app.config["rules"]))

    def test_restaurant_three_day_interval_is_validated_and_saved(self):
        rules=self.feature('restaurant_upgrade')
        rules[0]['stage']='restaurant_upgrade_owned'
        self.app.change_feature({'id':'restaurant_upgrade','interval':259200})
        saved=json.loads((self.root/'config.json').read_text(encoding='utf-8'))
        rule=next(r for r in saved['rules'] if r['id']=='driftinggirl_restaurant_upgrade')
        self.assertEqual(rule['interval'],259200)
        self.app.load_rule(rule)
        with self.assertRaises(ValueError):
            self.app.change_feature({'id':'restaurant_upgrade','interval':259201})

    def test_legacy_rule_switch_controls_whole_workflow_and_cannot_delete_step(self):
        rules = self.feature("mailbox")
        self.app.change_rule({"id": rules[0]["id"], "enabled": False})
        self.assertFalse(any(r["enabled"] for r in self.app.config["rules"]))
        with self.assertRaises(ValueError):
            self.app.change_rule({"id": rules[0]["id"], "delete": True})
        self.assertEqual(len(self.app.config["rules"]), len(FEATURE_BY_ID['mailbox']['rules']))

    def test_invalid_or_incomplete_changes_never_persist(self):
        self.feature("mana")
        before = deepcopy(self.app.config)
        for data in ({"id": "mana", "enabled": "false"}, {"id": "mana", "mana_threshold": 25.5},
                     {"id": "mana", "interval": -1}, {"id": "mailbox", "enabled": True},
                     {"id": "mana", "enabled": False, "mana_threshold": 101}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.app.change_feature(data)
            self.assertEqual(self.app.config, before)
        with patch.object(self.app, "persist", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.app.change_feature({"id": "mana", "enabled": False})
        self.assertEqual(self.app.config, before)
        self.assertEqual(self.app.config_revision, 0)

    def test_parameters_update_only_entry_and_leave_counter_limits_intact(self):
        mana = self.feature("mana")
        mana[0]["number_condition"] = {"value": 25}
        for rule in mana[1:]:
            rule["stage"] = "followup"
        ads = self.feature("growth_ad")
        before_ads = deepcopy(ads)
        self.app.change_feature({"id": "mana", "mana_threshold": 20, "interval": 45})
        updated = self.app.config["rules"]
        self.assertEqual(updated[0]["number_condition"]["value"], 20)
        self.assertEqual(updated[0]["interval"], 45)
        self.assertEqual(updated[1]["interval"], 0.2)
        self.assertEqual([r for r in updated if r['id'] in FEATURE_BY_ID['growth_ad']['rules']], before_ads)

    def test_meter_ninety_minute_cooldown_is_saved_without_widening_other_limits(self):
        rule = self.feature('meter_calibration')[0]
        rule['stage'] = 'meter_owned'
        self.feature('mailbox')
        before = deepcopy(self.app.config)
        self.app.change_feature({'id':'meter_calibration','interval':5400})
        changed = self.app.config['rules'][0]
        self.assertEqual(changed['interval'],5400)
        self.app.load_rule(deepcopy(changed))
        self.assertEqual(self.app.config['rules'][1:], before['rules'][1:])
        with self.assertRaises(ValueError):
            self.app.change_feature({'id':'mailbox','interval':5400})

    def test_training_parameters_save_atomically_and_preserve_other_switches(self):
        self.feature('parallel_world')[0]['stage'] = 'parallel_owned'
        self.feature('equipment_upgrade')
        before_equipment = deepcopy(self.app.config['rules'][-1])
        options = {'enabled': True, 'failure_limit': 3, 'level_gain': 15, 'minutes': 5,
                   'retry_level_gap': 15, 'final_boss_retry_level_gap': 0, 'auto_sailing': True,
                   'meter_auto_adjust': False, 'boss_interrupt': False, 'battle_tap': True,
                   'parallel_stage_limit': 8, 'boss_auto_sailing_until_rank': 35}
        result = self.app.change_feature({'id': 'parallel_world', 'interval': 3, 'training': options})
        self.assertEqual(result['training'], options)
        self.assertEqual(result['interval'], 3)
        self.assertEqual(self.app.config['rules'][-1], before_equipment)
        saved = json.loads((self.root/'config.json').read_text())
        self.assertEqual(saved['rules'][0]['training'], options)
        before = deepcopy(self.app.config)
        for data in ({'id': 'parallel_world', 'enabled': False, 'training': {'failure_limit': 0}},
                     {'id': 'equipment_upgrade', 'training': options}):
            with self.assertRaises(ValueError):
                self.app.change_feature(data)
            self.assertEqual(self.app.config, before)

    def test_skill_points_setting_is_scoped_persistent_and_strict(self):
        self.feature('parallel_skills')[0]['stage'] = 'skills_owned'
        result = self.app.change_feature({'id': 'parallel_skills', 'growth_points': False, 'interval': 300})
        self.assertFalse(result['growth_points'])
        self.assertEqual(result['interval'], 300)
        before = deepcopy(self.app.config)
        for data in ({'id': 'parallel_skills', 'growth_points': 'true'},
                     {'id': 'parallel_skills', 'growth_points': 1},
                     {'id': 'parallel_world', 'growth_points': True}):
            with self.assertRaises(ValueError):
                self.app.change_feature(data)
            self.assertEqual(self.app.config, before)

    def test_disable_during_mail_finishes_return_then_removes_whole_workflow(self):
        rules = self.feature("mailbox")
        for rule, stage, following in zip(rules, ["main", "claim", "close", "return"], ["claim", "close", "return", "main"]):
            rule.update(stage=stage, next_stage=following, next_delay=0.05)
        rules[3]["action"] = "observe"
        for rule in rules[4:]:
            rule.update(stage='claim', action='observe', threshold=1)
        original_click = self.app.device.click
        pending = []

        def click(button):
            original_click(button)
            if len(self.app.device.taps) == 1:
                self.app.change_feature({"id": "mailbox", "enabled": False})
                pending.append(self.app.status()["settings_pending"])

        self.app.device.click = click
        self.app.start(preview=False)
        self.wait_for(lambda: self.app.state["applied_revision"] == 1)
        self.assertEqual(pending, [True])
        self.assertEqual(len(self.app.device.taps), 3)
        self.assertEqual(self.app.state["stage"], "main")
        self.assertEqual(self.app.state["active_rule_ids"], [])
        self.assertIsNone(self.app.state["last_error"])
        old_frames = self.app.state["frames"]
        self.wait_for(lambda: self.app.state["frames"] >= old_frames + 2)
        self.assertEqual(len(self.app.device.taps), 3)

    def test_enable_new_task_while_running_and_disable_stops_further_clicks(self):
        self.feature("relief")
        self.app.change_feature({"id": "relief", "enabled": False})
        unrelated = self.rule()
        unrelated["threshold"] = 1  # Keep a runnable task without matching this frame.
        original_click = self.app.device.click

        def click(button):
            original_click(button)
            self.app.change_feature({"id": "relief", "enabled": False})

        self.app.device.click = click
        self.app.start(preview=False)
        self.wait_for(lambda: self.app.state["frames"] >= 2)
        self.app.change_feature({"id": "relief", "enabled": True})
        self.wait_for(lambda: self.app.state["applied_revision"] == 3)
        self.assertEqual(len(self.app.device.taps), 1)
        self.assertEqual(self.app.state["active_rule_ids"], [unrelated["id"]])
        self.assertIsNone(self.app.state["last_error"])

    def test_http_page_assets_and_persisted_feature_endpoint(self):
        self.feature("mailbox")
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.app))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(base + "/") as response:
                html = response.read().decode("utf-8")
            token = re.search(r"const TOKEN='([0-9a-f]+)'", html).group(1)
            self.assertIn("漂流助手", html)
            for path, mime in (("/static/app.js", "text/javascript"), ("/static/style.css", "text/css")):
                with urllib.request.urlopen(base + path) as response:
                    self.assertIn(mime, response.headers["Content-Type"])
                    self.assertGreater(len(response.read()), 100)
            payload = json.dumps({"id": "mailbox", "enabled": False}).encode()
            request = urllib.request.Request(base + "/api/features/change", data=payload,
                                             headers={"X-Session-Token": token})
            with urllib.request.urlopen(request) as response:
                self.assertTrue(json.load(response)["ok"])
            self.assertFalse(any(r["enabled"] for r in self.app.config["rules"]))
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(urllib.request.Request(base + "/api/features/change", data=payload))
            self.assertEqual(error.exception.code, 403)
            error.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)


if __name__ == "__main__":
    unittest.main()
