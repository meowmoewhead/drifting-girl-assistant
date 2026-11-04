"""Allow one verified daily check after 08:00 even without the generic N badge."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def update(config):
    result = deepcopy(config)
    entry = next(rule for rule in result['rules'] if rule['id'] == 'driftinggirl_daily_open')
    entry.pop('required_source_areas', None)
    entry.update(name='每日奖励：本周期未完成时打开商店',
                 color_condition=dict(source_area=[659,378,691,406],
                     hsv=[[0,0,140],[179,255,255]], minimum_fraction=.4),
                 note='每天北京时间 08:00 刷新；本周期未完成时检查，无需绿色 N。成功或确认已领取后等待下次刷新；未完成保留原间隔重试。')
    return result


if __name__ == '__main__':
    path = ROOT / 'config.json'
    config = update(json.loads(path.read_text(encoding='utf-8-sig')))
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
