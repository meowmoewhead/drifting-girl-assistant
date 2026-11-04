"""Recover mailbox transitions interrupted by a parallel-world boss."""
from copy import deepcopy
import json
from pathlib import Path


NEW_IDS = ('driftinggirl_mailbox_parallel_battle_recover',
           'driftinggirl_mailbox_parallel_return')

BASE = dict(
    asset='assets/navigation/parallel-boss-battle.png',
    source_crop=[254, 38, 289, 75],
    search_area=[247, 30, 298, 85],
    required_source_areas=[[279, 884, 307, 910], [211, 1140, 266, 1189],
                           [542, 1140, 597, 1183]],
    color_condition=dict(source_area=[558, 1147, 585, 1168],
                         hsv=[[0, 0, 210], [179, 255, 255]], minimum_fraction=.3),
    scales=[1], threshold=.90, interval=1, stable_frames=3,
    action='observe', next_stage='main',
)


def update(config):
    result = deepcopy(config)
    existing = {rule['id']: rule for rule in result['rules']}
    entry = existing['driftinggirl_mailbox_open']
    entry['stage_timeout'] = max(entry.get('stage_timeout', 20), 45)
    rules = [
        dict(deepcopy(BASE), id=NEW_IDS[0],
             name='邮箱：平行世界战斗打断后等待入口恢复',
             stage='mailbox', min_stage_seconds=5,
             recovery_entry=entry['id'],
             note='连续核对 BOSS 骷髅、AUTO 勾选和明亮底栏；仅撤销被打断的入口，保留邮箱间隔，五分钟最多恢复三次。'),
        dict(deepcopy(BASE), id=NEW_IDS[1],
             name='邮箱：关闭后确认平行世界战斗主画面',
             stage='mailbox_return', timeout_grace=8,
             note='关闭邮箱后若已开始 BOSS 战斗，连续确认无遮挡的战斗主画面；只观察，不点击战斗界面。'),
    ]
    for rule in rules:
        old = existing.get(rule['id'])
        rule['enabled'] = old['enabled'] if old else entry['enabled']
        if old:
            rule['interval'] = old['interval']
    result['rules'] = rules + [rule for rule in result['rules'] if rule['id'] not in NEW_IDS]
    return result


if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1] / 'config.json'
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    path.write_text(json.dumps(update(config), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
