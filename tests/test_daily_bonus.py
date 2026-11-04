"""Real bright/gray bonus cards and geometry excluding the paid voucher plus."""
import json
from pathlib import Path
import unittest

import cv2
import numpy as np
from PIL import Image
from drift.perception.daily_bonus import DailyBonusVisual

ROOT = Path(__file__).resolve().parents[1]
BONUS_ID = "driftinggirl_daily_bonus"
#: 右上「购买附赠奖励券」加号所在区域(实测坐标,含余量)
VOUCHER_BUTTON = (610, 325, 715, 380)


def config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))


def rule():
    rules = {item["id"]: item for item in config()["rules"]}
    return rules[BONUS_ID]


class DailyBonusStepTests(unittest.TestCase):
    def test_actual_ready_and_gray_cards_are_distinguished_from_other_pages(self):
        visual=DailyBonusVisual(ROOT)
        for name,expected in (('bonus-ready.png','ready'),('premium-ready.png','gray'),
                              ('premium-claimed.png','gray'),('bonus-claimed.png','gray'),
                              ('shop-open.png',None),('main.png',None)):
            frame=np.asarray(Image.open(ROOT/'assets/daily-reward'/name).convert('RGB'))
            self.assertEqual(visual.state(frame),expected,name)
        ready=np.asarray(Image.open(ROOT/'assets/daily-reward/bonus-ready.png').convert('RGB'))
        for dim in (.3,.5,.7):
            self.assertIsNone(visual.state((ready*dim).astype(np.uint8)))
        for box in ((171,250,263,280),(177,336,310,366),(637,251,688,294),(474,399,570,429)):
            bad=ready.copy();x,y,X,Y=box;bad[y:Y,x:X]=0
            self.assertIsNone(visual.state(bad))
        self.assertIsNone(visual.state(ready[:900]))

    def test_actual_post_claim_card_keeps_page_and_brightness_guards(self):
        visual=DailyBonusVisual(ROOT)
        claimed=np.asarray(Image.open(ROOT/'assets/daily-reward/bonus-claimed.png').convert('RGB'))
        self.assertEqual(visual.state(claimed),'gray')
        for dim in (.3,.5,.7):
            self.assertIsNone(visual.state((claimed*dim).astype(np.uint8)))
        for box in ((171,250,263,280),(177,336,310,366),(637,251,688,294),(474,399,570,429)):
            bad=claimed.copy();x,y,X,Y=box;bad[y:Y,x:X]=0
            self.assertIsNone(visual.state(bad))

    def test_step_is_wired_into_the_daily_chain(self):
        rules = {item["id"]: item for item in config()["rules"]}
        # 免费领取之后接附赠,附赠之后接关闭商店
        self.assertEqual(rules["driftinggirl_daily_claim"]["next_stage"], "daily_bonus")
        self.assertEqual(rules[BONUS_ID]["stage"], "daily_bonus")
        self.assertEqual(rules[BONUS_ID]["next_stage"], "daily_after_claim")
        self.assertEqual(rules["driftinggirl_daily_close"]["stage"], "daily_after_claim")
        self.assertEqual(rule()["skip_on_timeout"], True, "领不上必须能跳过")

    def test_anchor_matches_the_real_frame(self):
        """锚点来自整帧实机截图,搜索区域内必须匹配得到。

        裁剪型规则的 asset 必须是**整帧截图**(source_crop 再从里面裁);早先误把
        asset 指向裁好的小图,运行时会在 load_rule 里报"区域超出截图边界"。
        """
        entry = rule()
        reference = cv2.imread(str(ROOT / entry["asset"]))
        self.assertEqual(reference.shape[:2], (1280, 720), "asset 必须是整帧实机截图")
        cx1, cy1, cx2, cy2 = entry["source_crop"]
        patch = reference[cy1:cy2, cx1:cx2]
        sx1, sy1, sx2, sy2 = entry["search_area"]
        # 运行期 _fixed 会再各放宽 7px;这里同样放宽
        area = reference[max(0, sy1 - 7):sy2 + 7, max(0, sx1 - 7):sx2 + 7]
        self.assertGreaterEqual(area.shape[0], patch.shape[0], "搜索区域高度放不下模板")
        self.assertGreaterEqual(area.shape[1], patch.shape[1], "搜索区域宽度放不下模板")
        score = float(cv2.matchTemplate(area, patch, cv2.TM_CCOEFF_NORMED).max())
        self.assertGreaterEqual(score, entry["threshold"], f"锚点匹配分数 {score:.4f}")

    def test_anchor_is_present_whether_or_not_rewards_are_claimed(self):
        """标题是页面锚点:免费奖励领取前后都在,否则领完附赠就找不到后续步骤。"""
        entry = rule()
        reference = cv2.imread(str(ROOT / entry["asset"]))
        cx1, cy1, cx2, cy2 = entry["source_crop"]
        patch = reference[cy1:cy2, cx1:cx2]
        sx1, sy1, sx2, sy2 = entry["search_area"]
        area = reference[max(0, sy1 - 7):sy2 + 7, max(0, sx1 - 7):sx2 + 7]
        self.assertGreaterEqual(float(cv2.matchTemplate(area, patch, cv2.TM_CCOEFF_NORMED).max()),
                                entry["threshold"])
        claimed = cv2.imread(str(ROOT / "assets/daily-reward/premium-claimed.png"))
        score = float(cv2.matchTemplate(
            claimed[max(0, sy1 - 7):sy2 + 7, max(0, sx1 - 7):sx2 + 7],
            patch, cv2.TM_CCOEFF_NORMED).max())
        self.assertGreaterEqual(score, entry["threshold"], "已领取帧里锚点丢失")

    def test_click_area_never_touches_the_voucher_button(self):
        click = rule()["click_source_area"]
        ax1, ay1, ax2, ay2 = click
        bx1, by1, bx2, by2 = VOUCHER_BUTTON
        overlap = (max(0, min(ax2, bx2) - max(ax1, bx1))
                   * max(0, min(ay2, by2) - max(ay1, by1)))
        self.assertEqual(overlap, 0, "点击区域与购买券加号重叠,有误点付费入口的风险")
        # 也确认它落在附赠卡那一侧(页面右半),而不是免费卡
        self.assertGreater((ax1 + ax2) / 2, 395, "应点右侧的附赠奖励卡")

    def test_step_does_not_disturb_the_free_reward_rules(self):
        rules = {item["id"]: item for item in config()["rules"]}
        free = rules["driftinggirl_daily_claim"]
        self.assertEqual(free["click_source_area"], [210, 433, 377, 510], "免费卡点击区域不应改动")
        self.assertEqual(free["stage"], "daily_reward")
        self.assertEqual(rules["driftinggirl_daily_unavailable"]["stage"], "daily_reward",
                         "已领取分支仍应留在 daily_reward")

    def test_adding_twice_is_idempotent(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "configure_daily_bonus", ROOT / "tools" / "configure_daily_bonus.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        once = config()
        twice = module.add(once)
        self.assertEqual([r["id"] for r in once["rules"]], [r["id"] for r in twice["rules"]])
        self.assertEqual(once["rules"], twice["rules"])


if __name__ == "__main__":
    unittest.main()
