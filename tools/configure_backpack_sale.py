"""Add the independent full fishing-bag sale switch without changing old rules."""
from copy import deepcopy
import argparse
import json
from pathlib import Path

RULE = {
    'id': 'driftinggirl_backpack_sell', 'name': '满背包：港口出售后返回原地点',
    'enabled': True, 'asset': 'assets/backpack/bag-icon.png',
    'source_crop': [0, 0, 51, 55], 'search_area': [644, 1147, 695, 1202],
    'scales': [1], 'threshold': .92, 'interval': 60, 'stable_frames': 3,
    'stage': 'backpack_owned', 'action': 'observe',
    'note': '连续确认 30/30 后到当前世界港口，只出售第一标签未锁定材料，返回原地点。',
}


def update(config):
    config = deepcopy(config)
    matches = [rule for rule in config['rules'] if rule['id'] == RULE['id']]
    if len(matches) > 1:
        raise ValueError('Duplicate backpack sale rule')
    if not matches:
        config['rules'].append(deepcopy(RULE))
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default=str(Path(__file__).resolve().parents[1]/'config.json'))
    path = Path(parser.parse_args().config)
    config = update(json.loads(path.read_text(encoding='utf-8-sig')))
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
