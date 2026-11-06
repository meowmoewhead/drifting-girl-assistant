"""Add the observed candy event without changing task switches or intervals."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'driftinggirl_event_potion_'
IDS = tuple(PREFIX + step for step in ('open', 'select', 'confirm', 'close', 'return'))


def update(config):
    updated = deepcopy(config)
    members = [rule for rule in updated['rules'] if rule['id'] in IDS]
    if len(members) != len(IDS) or {rule['id'] for rule in members} != set(IDS):
        raise ValueError('Expected the five complete event potion rules')
    sources = {'open': 'main', 'select': 'shop', 'confirm': 'confirm', 'close': 'shop'}
    for rule in members:
        step = rule['id'].removeprefix(PREFIX)
        if step in sources:
            # Each variant includes its own matching currency and price fragments.
            rule['asset'] = f'assets/event-potion/{sources[step]}.png'
            rule['asset_variants'] = [f'assets/event-potion/{sources[step]}-candy.png']
        else:
            # The skill potion and central spiral animate; these two icons do not.
            rule.update(asset='assets/event-potion/main-candy.png',
                        asset_variants=['assets/event-potion/main.png'],
                        source_crop=[211,1140,266,1189], search_area=[190,1115,291,1210],
                        required_source_areas=[[542,1140,597,1183]],
                        color_condition=dict(source_area=[558,1147,585,1168],
                            hsv=[[0,0,210],[179,255,255]], minimum_fraction=.3),
                        note='商店消失且底栏天赋、钻石图标清晰可见后返回主界面；不依赖动画或材料数字。')
        if step == 'open':
            rule['note'] = ('兼容已核对的月亮币与糖果活动；数字区域独立 OCR，连续三帧读数一致且严格大于 500。'
                            '拒绝弹窗后的暗色入口；每轮只购买一瓶。')
    return updated


if __name__ == '__main__':
    path = ROOT / 'config.json'
    config = update(json.loads(path.read_text(encoding='utf-8-sig')))
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Candy event configured; rules:', len(config['rules']))
