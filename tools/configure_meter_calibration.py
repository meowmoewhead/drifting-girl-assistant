"""Add an independent realtime task, preserving every existing rule and switch."""
from copy import deepcopy

RULE = dict(id='driftinggirl_meter_calibration', name='标尺校准：落后角色超过50级',
    enabled=False, asset='assets/equipment/main-auto.png', source_crop=[0,0,28,26],
    search_area=[276,881,310,913], scales=[1], threshold=.91, interval=5400,
    stable_frames=3, stage='meter_owned', action='observe',
    note='实时检查等级差，完成后默认冷却90分钟；保留航巡，仅停持续钓鱼，普通鱼休眠后调到角色等级附近，允许10级误差；长按后连续新图确认填充或战斗，保留游戏原勾选项。')


def add(config, enabled=False):
    result = deepcopy(config)
    matches = [r for r in result['rules'] if r['id'] == RULE['id']]
    if len(matches) > 1:
        raise ValueError('Duplicate meter-calibration rule')
    if not matches:
        result['rules'].append({**deepcopy(RULE), 'enabled': enabled})
    return result
