"""Season-independent event guards and offline replay of all three events."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dev_tools.verify_event_potion import PREFIX, STEPS, inspect_rule, replay


def overlaps(first, second):
    return (max(first[0], second[0]) < min(first[2], second[2])
            and max(first[1], second[1]) < min(first[3], second[3]))


class EventPotionRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
        cls.rules = {r["id"].removeprefix(PREFIX): r for r in cls.config["rules"]
                     if r["id"].startswith(PREFIX)}

    def test_all_five_rules_are_present_once_and_disabled_in_public_defaults(self):
        rules = [r for r in self.config["rules"] if "event_potion" in r["id"]]
        self.assertEqual(len(rules), 5)
        self.assertEqual({r["id"] for r in rules}, {PREFIX + step for step in STEPS})
        for rule in rules:
            with self.subTest(rule=rule["id"]):
                self.assertIs(rule["enabled"], False)

    def test_every_rule_keeps_legacy_variants_and_adds_ghost(self):
        for step, rule in self.rules.items():
            with self.subTest(rule=step):
                variants = rule["asset_variants"]
                self.assertGreaterEqual(len(variants), 2)
                self.assertEqual(len(variants), len(set(variants)))
                kind = "main" if step in ("open", "return") else (
                    "confirm" if step == "confirm" else "shop")
                self.assertIn(f"assets/event-potion/{kind}-ghost.png", variants)
                self.assertTrue({f"assets/event-potion/{kind}{suffix}.png"
                                 for suffix in ("", "-candy", "-ghost")}
                                .issubset({rule["asset"], *variants}))
                for asset in (rule["asset"], *variants):
                    self.assertTrue((ROOT / asset).is_file(), asset)

    def test_select_price_guard_excludes_currency(self):
        guards = self.rules["select"]["required_source_areas"]
        prices = [area for area in guards if area[1] == 715 and area[3] == 741]
        self.assertEqual(len(prices), 1)
        self.assertGreaterEqual(prices[0][0], 160)
        self.assertLess(prices[0][0], prices[0][2])
        self.assertIn([557, 521, 597, 564], guards)

    def test_confirm_keeps_price_and_buy_button_without_currency(self):
        rule = self.rules["confirm"]
        self.assertGreaterEqual(rule["source_crop"][0], 260)
        guards = rule["required_source_areas"]
        self.assertFalse(any(overlaps(area, [228, 575, 259, 606]) for area in guards))
        self.assertIn([265, 577, 315, 603], guards)
        self.assertIn([165, 769, 256, 816], guards)

    def test_close_uses_compact_cross_and_help_button(self):
        guards = self.rules["close"]["required_source_areas"]
        self.assertFalse(any(overlaps(area, [118, 526, 153, 560]) for area in guards))
        crosses = [area for area in guards if area[1] <= 420 < area[3]]
        self.assertEqual(len(crosses), 1)
        self.assertTrue(0 < crosses[0][2] - crosses[0][0] <= 50)
        self.assertIn([557, 521, 597, 564], guards)

    def test_three_seasons_match_only_expected_pages(self):
        results = replay(self.config)
        self.assertEqual(len(results), 45)
        for result in results:
            with self.subTest(season=result["season"], frame=result["frame"], rule=result["rule"]):
                self.assertEqual(result["matched"], result["expected"], result)

    def test_legacy_templates_cover_ghost_after_entry(self):
        results = replay(self.config, legacy_templates_only=True)
        for result in results:
            with self.subTest(season=result["season"], frame=result["frame"], rule=result["rule"]):
                self.assertEqual(result["matched"], result["expected"], result)
                if result["rule"] != PREFIX + "open":
                    self.assertNotIn("-ghost.png", result["asset"])

    def test_currency_artwork_changes_do_not_break_downstream_rules(self):
        areas = {"select": ("shop", [140, 715, 168, 741]),
                 "confirm": ("confirm", [228, 575, 259, 606]),
                 "close": ("shop", [118, 526, 153, 560])}
        context = SimpleNamespace(root=ROOT, config=self.config)
        for suffix in ("", "-candy", "-ghost"):
            for step, (kind, area) in areas.items():
                with self.subTest(season=suffix, rule=step):
                    with Image.open(ROOT / "assets" / "event-potion" / f"{kind}{suffix}.png") as image:
                        frame = np.array(image.convert("RGB"))
                    x1, y1, x2, y2 = area
                    frame[y1:y2, x1:x2] = (30, 140, 240)
                    self.assertTrue(inspect_rule(context, self.rules[step], frame)["matched"])

    def test_stable_guards_still_reject_missing_price_buttons_and_cross(self):
        context = SimpleNamespace(root=ROOT, config=self.config)
        for suffix in ("", "-candy", "-ghost"):
            for step in ("select", "confirm", "close"):
                kind = "confirm" if step == "confirm" else "shop"
                with Image.open(ROOT / "assets" / "event-potion" / f"{kind}{suffix}.png") as image:
                    original = np.array(image.convert("RGB"))
                for area in self.rules[step]["required_source_areas"]:
                    with self.subTest(season=suffix, rule=step, missing=area):
                        frame = original.copy()
                        x1, y1, x2, y2 = area
                        frame[y1:y2, x1:x2] = 0
                        self.assertFalse(inspect_rule(context, self.rules[step], frame)["matched"])


if __name__ == "__main__":
    unittest.main()
