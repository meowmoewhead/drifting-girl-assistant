"""Add fish-cake enhancement while preserving existing task switches."""
from copy import deepcopy
import json
from pathlib import Path


RULE = {
    'id': 'driftinggirl_equipment_upgrade', 'name': '武器强化：轮流长按鱼饼升级按钮',
    'enabled': True, 'asset': 'assets/equipment/entry.png',
    'source_crop': [0, 0, 45, 46], 'search_area': [326, 1139, 379, 1193],
    'scales': [1], 'threshold': .9, 'interval': 900, 'stable_frames': 2,
    'stage': 'equipment_owned', 'action': 'observe',
    'note': '默认每15分钟检查；确认鱼饼面板、武器名称及绿色升级箭头后，每种武器最多长按12秒一次，检查一遍即关闭；保留3分钟保护上限。',
}


def add(config):
    result = deepcopy(config)
    matches = [r for r in result['rules'] if r['id'] == RULE['id']]
    if len(matches) > 1:
        raise ValueError('Duplicate equipment-upgrade rule')
    if not matches:
        result['rules'].append(deepcopy(RULE))
    return result


if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1]/'config.json'
    path.write_text(json.dumps(add(json.loads(path.read_text(encoding='utf-8-sig'))),
                               ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
