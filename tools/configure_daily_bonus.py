"""Add the bonus step; runtime independently checks bright/gray availability.

Both early screenshots had no voucher and the same gray card. The 20261012
claimable sample proves the bright state. Gray cards are observed without input;
the paid voucher plus is never used. Existing saved rules remain unchanged.
"""
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FREE_CLAIM = "driftinggirl_daily_claim"
BONUS_ID = "driftinggirl_daily_bonus"
SHOP_CLOSE = "driftinggirl_daily_close"
RETURN_ID = "driftinggirl_daily_return"

CARD_CLICK = [410, 430, 605, 500]      # 附赠卡内部,避开右上购买券加号
ANCHOR_CROP = [177, 336, 310, 366]     # "每日奖励!"标题(必须与模板裁图一致)
ANCHOR_SOURCE = "assets/daily-reward/premium-ready.png"   # 锚点模板的裁图来源
# 搜索区域必须:① 比模板(133x30)大;② 加 7px 边距后仍落在 720x1280 内
ANCHOR_SEARCH = [150, 310, 360, 400]

BONUS_RULE = {
    "id": BONUS_ID,
    "name": "每日奖励：领取可用附赠奖励，灰色跳过",
    "enabled": False,
    "asset": ANCHOR_SOURCE,
    "source_crop": ANCHOR_CROP,
    "search_area": ANCHOR_SEARCH,
    "click_source_area": CARD_CLICK,
    "scales": [1],
    "threshold": 0.94,
    "interval": 2,
    "stable_frames": 3,
    "stage": "daily_bonus",
    "next_stage": "daily_after_claim",
    "next_delay": 0.7,
    "stage_timeout": 20,
    "skip_on_timeout": True,
    "required_source_areas": [[171, 250, 263, 280], [637, 251, 688, 294]],
    "note": "运行时独立确认右侧附赠奖励亮色可领才点击一次，灰色直接跳过；领取后确认灰色并返回。保留免费完成记录，绝不点击购买券加号。",
}


def add(config):
    """返回插入附赠步骤后的新配置;已存在则原样返回。"""
    updated = deepcopy(config)
    rules = updated.get("rules", [])
    if any(rule.get("id") == BONUS_ID for rule in rules):
        return updated
    free = next((rule for rule in rules if rule.get("id") == FREE_CLAIM), None)
    if free is None:
        raise ValueError("配置里没有每日奖励的免费领取步骤，无法接入附赠奖励")
    free["next_stage"] = "daily_bonus"
    bonus = dict(BONUS_RULE)
    bonus["enabled"] = bool(free.get("enabled"))
    index = rules.index(free) + 1
    rules.insert(index, bonus)
    return updated


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "config.json"
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    updated = add(config)
    path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("已写入 config.json:")
    for rule in updated["rules"]:
        if rule.get("id", "").startswith("driftinggirl_daily"):
            print(f"  {rule['id']:<38} stage={rule.get('stage'):<18} -> {rule.get('next_stage')}  enabled={rule.get('enabled')}")


if __name__ == "__main__":
    main()
