"""Add the independently switchable parallel-world controller to a config."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boss_training import training_options

RULE = {
    'id': 'driftinggirl_parallel_world',
    'name': '平行世界：RANK 40 后循环次元石',
    'enabled': True,
    'asset': 'assets/parallel/world-badge.png',
    'source_crop': [0, 0, 26, 26],
    'search_area': [13, 46, 39, 72],
    'scales': [1],
    'threshold': 0.92,
    'interval': 2,
    'stable_frames': 1,
    'stage': 'parallel_owned',
    'action': 'observe',
    'training': training_options(),
    'note': '由平行世界状态流程识别 RANK、地图 BOSS、传送门确认；仅在连续确认 RANK 40 后关闭传送门。',
}


def add(config):
    matches = [rule for rule in config['rules'] if rule['id'] == RULE['id']]
    if len(matches) > 1:
        raise ValueError('Duplicate parallel-world rule')
    if not matches:
        config['rules'].append({**RULE, 'training': training_options()})
    else:
        matches[0]['training'] = training_options(matches[0].get('training'))
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default=str(Path(__file__).resolve().parents[1] / 'config.json'))
    path = Path(parser.parse_args().config)
    config = add(json.loads(path.read_text(encoding='utf-8-sig')))
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
