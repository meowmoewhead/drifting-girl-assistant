"""Handle an achievement menu interrupted by a parallel-world boss fight."""
from copy import deepcopy
import json
from pathlib import Path


MENU_VARIANT = 'assets/achievements-menu-parallel.png'
VARIANT_RULES = ('driftinggirl_achievements_resume',
                 'driftinggirl_achievements_crown',
                 'driftinggirl_achievements_fold')
ENTRY_ID = 'driftinggirl_achievements_parallel_entry_recover'

ENTRY_RULE = {
    'id': ENTRY_ID,
    'name': '皇冠奖励：平行世界入口被打断后恢复主画面',
    'enabled': True,
    'asset': 'assets/navigation/parallel-boss-battle.png',
    'source_crop': [254, 38, 289, 75],
    'search_area': [247, 30, 298, 85],
    'required_source_areas': [[279, 884, 307, 910], [211, 1140, 266, 1189],
                              [542, 1140, 597, 1183]],
    'color_condition': {'source_area': [558, 1147, 585, 1168],
                        'hsv': [[0, 0, 210], [179, 255, 255]],
                        'minimum_fraction': .3},
    'absent_rule': 'driftinggirl_achievements_fold',
    'scales': [1],
    'threshold': .90,
    'interval': 1,
    'stable_frames': 3,
    'min_stage_seconds': 6,
    'stage': 'achievements_tab',
    'action': 'observe',
    'next_stage': 'main',
    'recovery_entry': 'driftinggirl_achievements_open',
    'note': '菜单未展开且 BOSS 骷髅、AUTO 勾选和明亮底栏连续三帧可见时，撤销被打断的入口。只观察，保留入口间隔，五分钟最多恢复三次；入口最长等待四十五秒，超时仍停止。',
}

BATTLE_RULE = {
    'id': 'driftinggirl_achievements_parallel_battle_recover',
    'name': '皇冠奖励：平行世界战斗打断后等待菜单',
    'enabled': True,
    'asset': 'assets/navigation/parallel-boss-battle.png',
    'source_crop': [254, 38, 289, 75],
    'search_area': [247, 30, 298, 85],
    'required_source_areas': [[321, 970, 389, 1045], [211, 1140, 266, 1189]],
    'scales': [1],
    'threshold': .90,
    'interval': 1,
    'stable_frames': 2,
    'min_stage_seconds': 3,
    'stage': 'achievements_claim',
    'action': 'observe',
    'next_stage': 'achievements_tab',
    'stage_timeout': 120,
    'note': '确认顶部 BOSS 骷髅和底部战斗图标后，只等待战斗结束，不点击战斗画面。',
}

MENU_RULE = {
    'id': 'driftinggirl_achievements_claim_menu_recover',
    'name': '皇冠奖励：列表未打开时返回皇冠菜单',
    'enabled': True,
    'asset': MENU_VARIANT,
    'source_crop': [555, 625, 592, 646],
    'search_area': [510, 565, 635, 700],
    'required_source_areas': [[211, 1140, 266, 1189], [542, 1140, 597, 1183]],
    'color_condition': {'source_area': [558, 1147, 585, 1168],
                        'hsv': [[0, 0, 210], [179, 255, 255]],
                        'minimum_fraction': .3},
    'absent_rule': 'driftinggirl_achievements_ready',
    'scales': [1],
    'threshold': .93,
    'interval': 1,
    'stable_frames': 3,
    'min_stage_seconds': 5,
    'stage': 'achievements_claim',
    'action': 'observe',
    'next_stage': 'achievements_tab',
    'stage_timeout': 30,
    'retry_limit': 3,
    'note': '皇冠列表未打开而菜单仍明亮时，最多三次回到皇冠步骤重试。',
}


def update(config):
    config = deepcopy(config)
    rules = {rule['id']: rule for rule in config['rules']}
    entry = rules['driftinggirl_achievements_open']
    entry['stage_timeout'] = max(entry.get('stage_timeout', 20), 45)
    for identifier in VARIANT_RULES:
        rule = rules[identifier]
        variants = list(rule.get('asset_variants', []))
        if MENU_VARIANT not in variants:
            variants.append(MENU_VARIANT)
        rule['asset_variants'] = variants
    for new_rule in (BATTLE_RULE, MENU_RULE):
        if new_rule['id'] not in rules:
            config['rules'].append(deepcopy(new_rule))
    old_entry_recovery = rules.get(ENTRY_ID)
    recovery = deepcopy(ENTRY_RULE)
    recovery['enabled'] = (old_entry_recovery or entry)['enabled']
    if old_entry_recovery:
        recovery['interval'] = old_entry_recovery['interval']
    config['rules'] = [rule for rule in config['rules'] if rule['id'] != ENTRY_ID] + [recovery]
    return config


if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1] / 'config.json'
    config = update(json.loads(path.read_text(encoding='utf-8-sig')))
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
