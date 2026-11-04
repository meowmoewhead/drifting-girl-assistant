"""任务声明表必须与功能注册表、规则集合、控制器模块保持一致。

这是"一功能一任务类"这层结构的守卫:UI 契约(features.py)、识别规则(config.json)
与任务声明(drift/tasks/registry.py)三者不允许各自漂移。
"""
import importlib
import json
from pathlib import Path
import unittest

from drift.tasks.base import MODE_DAILY, MODE_REALTIME, MODE_TIMED, TaskSpec
from drift.tasks.features import (
    DAILY_FEATURES,
    FEATURES,
    REALTIME_CONDITIONS,
    RULE_FEATURE,
    RULELESS_FEATURES,
    TIMED_FEATURES,
)
from drift.tasks.registry import TASK_SPECS

ROOT = Path(__file__).resolve().parents[1]


def config_rules():
    data = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    return {rule["id"]: rule for rule in data["rules"]}


class TaskRegistryTests(unittest.TestCase):
    def test_every_feature_has_exactly_one_spec(self):
        self.assertEqual(set(TASK_SPECS), {feature["id"] for feature in FEATURES})
        self.assertEqual(len(TASK_SPECS), len(FEATURES))
        for spec in TASK_SPECS.values():
            self.assertIsInstance(spec, TaskSpec)
            self.assertTrue(spec.name)
            # 只有显式登记在 RULELESS_FEATURES 里的功能才允许没有识别规则:
            # 它们由运行时按 Rank/所在世界直接判定,不看 scan_rules。
            self.assertTrue(spec.rules or spec.id in RULELESS_FEATURES, spec.id)
        self.assertLessEqual(RULELESS_FEATURES, {feature["id"] for feature in FEATURES})

    def test_spec_rules_match_feature_rules(self):
        for feature in FEATURES:
            spec = TASK_SPECS[feature["id"]]
            self.assertEqual(spec.rules, tuple(feature["rules"]), feature["id"])

    def test_execution_mode_matches_feature_declaration(self):
        for feature in FEATURES:
            spec = TASK_SPECS[feature["id"]]
            expected = (MODE_DAILY if feature["id"] in DAILY_FEATURES
                        else MODE_TIMED if feature["id"] in TIMED_FEATURES
                        else MODE_REALTIME)
            self.assertEqual(spec.mode, expected, feature["id"])
        self.assertEqual({spec.id for spec in TASK_SPECS.values() if spec.mode == MODE_TIMED},
                         set(TIMED_FEATURES))
        self.assertEqual({spec.id for spec in TASK_SPECS.values() if spec.mode == MODE_DAILY},
                         set(DAILY_FEATURES))

    def test_realtime_conditions_are_carried_over(self):
        for identifier, condition in REALTIME_CONDITIONS.items():
            self.assertEqual(TASK_SPECS[identifier].note, condition)

    def test_controller_specs_point_at_importable_modules(self):
        controllers = [spec for spec in TASK_SPECS.values() if spec.is_controller]
        self.assertEqual(len(controllers), 7)
        for spec in controllers:
            module = importlib.import_module(spec.module)
            self.assertTrue(hasattr(module, _class_name(spec.controller)),
                            f"{spec.module} 缺少 {spec.controller} 的控制类")
            self.assertTrue(spec.main_guard, spec.id)

    def test_rule_specs_share_the_scanning_engine(self):
        for spec in TASK_SPECS.values():
            if spec.is_controller:
                continue
            self.assertEqual(spec.module, "drift.core.matcher", spec.id)
            self.assertFalse(spec.controller, spec.id)
            self.assertFalse(spec.main_guard, spec.id)

    def test_rule_feature_mapping_covers_config_rules(self):
        rules = config_rules()
        missing = [identifier for identifier in rules if identifier not in RULE_FEATURE]
        self.assertEqual(missing, [], f"config.json 里有未挂到任何功能的规则: {missing}")
        declared = {identifier for spec in TASK_SPECS.values() for identifier in spec.rules}
        self.assertEqual(declared, set(rules))

    def test_dependency_graph_only_references_known_features(self):
        for spec in TASK_SPECS.values():
            for dependency in spec.depends_on:
                self.assertIn(dependency, TASK_SPECS, f"{spec.id} 依赖了不存在的功能")

    def test_controllers_are_not_declared_as_dependencies_of_themselves(self):
        for spec in TASK_SPECS.values():
            self.assertNotIn(spec.id, spec.depends_on, spec.id)


def _class_name(controller):
    """控制器模块名 → 其中的控制类名(下表是唯一映射,新增控制器时要同步)。"""
    return {
        "equipment_upgrade": "EquipmentUpgrade",
        "pass_daily": "PassDaily",
        "parallel_skills": "ParallelSkills",
        "backpack_sale": "BackpackSale",
        "meter_calibration": "MeterCalibration",
        "stone_farm": "StoneFarm",
        "parallel_world": "ParallelWorld",
    }[controller]


if __name__ == "__main__":
    unittest.main()
