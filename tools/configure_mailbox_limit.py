"""Recognize a mailbox item that cannot be claimed because storage is full."""
import json
from pathlib import Path


RULE = {
    'id': 'driftinggirl_mailbox_limit_close',
    'name': '邮箱：物品超过获取上限，暂缓领取并关闭',
    'enabled': True,
    'asset': 'assets/mailbox-limit-native.png',
    'source_crop': [216, 343, 453, 369],
    'search_area': [190, 325, 475, 390],
    'required_source_areas': [[144, 228, 197, 253], [488, 987, 582, 1019]],
    'scales': [1],
    'threshold': 0.92,
    'interval': 2,
    'stable_frames': 2,
    'stage': 'mailbox_done',
    'click_source_area': [612, 170, 690, 230],
    'next_stage': 'mailbox_return',
    'next_delay': 0.8,
    'stage_timeout': 45,
    'note': '只在邮箱里识别“超过获取上限”并核对邮箱标签与全部领取按钮；保留未领邮件，15 分钟后重试。',
}


def add(config):
    matches = [rule for rule in config['rules'] if rule['id'] == RULE['id']]
    if len(matches) > 1:
        raise ValueError('Duplicate mailbox-limit rule')
    if not matches:
        config['rules'].append(RULE.copy())
    return config


if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1] / 'config.json'
    config = add(json.loads(path.read_text(encoding='utf-8-sig')))
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
