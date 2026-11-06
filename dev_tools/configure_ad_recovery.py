"""Recheck moving ad entries, and recover only their unconfirmed opening step."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = ('driftinggirl_bottom_right_arrow', 'driftinggirl_potion_ad_open')
NEW_IDS = ('driftinggirl_ad_entry_recover', 'driftinggirl_potion_ad_entry_recover')


def update(config):
    updated = deepcopy(config)
    existing = {r['id']: r for r in updated['rules']}
    added = []
    for entry_id, recovery_id, label in zip(ENTRIES, NEW_IDS, ('成长广告', '药水广告')):
        entry = existing[entry_id]
        entry['recheck_before_click'] = True
        # Stable talent, diamond and hook controls are visible during ordinary
        # fishing and giant-fish battles, but hidden/dimmed by reward dialogs.
        recovery = dict(id=recovery_id, name=label+'：入口未打开时重新识别',
            enabled=existing.get(recovery_id, entry)['enabled'],
            asset='assets/navigation/main.png', source_crop=[211,1140,266,1189],
            search_area=[200,1130,276,1200], scales=[1], threshold=.93,
            required_source_areas=[[542,1140,597,1183],[437,1144,486,1193]],
            color_condition=dict(source_area=[558,1147,585,1168],
                hsv=[[0,0,210],[179,255,255]],minimum_fraction=.3),
            stable_frames=3, interval=existing.get(recovery_id, {}).get('interval',1),
            stage=entry['next_stage'], action='observe', next_stage='main',
            min_stage_seconds=6, recovery_entry=entry_id,
            note='只在打开广告后尚未点击继续的步骤生效。至少等待六秒，三个游戏底栏图标连续三帧清晰可见才恢复入口检测；'
                 '只观察，不点击战斗；保留原入口间隔和每日计数，五分钟最多恢复三次，不记领奖完成。')
        added.append(recovery)
    updated['rules'] = [r for r in updated['rules'] if r['id'] not in NEW_IDS] + added
    return updated


if __name__ == '__main__':
    path = ROOT/'config.json'
    path.write_text(json.dumps(update(json.loads(path.read_text(encoding='utf-8-sig'))),
                               ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
