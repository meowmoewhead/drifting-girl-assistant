"""Add native-route stage handoff without enabling legacy navigation."""
from copy import deepcopy

RULE = dict(id='driftinggirl_stone_farm', name='次元石：降阶刷最后BOSS',
    enabled=False, asset='assets/parallel/world-badge.png', source_crop=[0,0,26,26],
    search_area=[13,46,39,72], scales=[1], threshold=.92, interval=2,
    stable_frames=3, stage='stone_owned', action='observe',
    note='连续确认当前阶段；RANK39保留进度降一阶打最后BOSS，实际RANK40后才关门重开原阶段。正常航巡交给游戏。')


def add(config, enabled=False):
    result = deepcopy(config)
    matches = [r for r in result['rules'] if r['id'] == RULE['id']]
    if len(matches) > 1:
        raise ValueError('Duplicate stone-farming rule')
    if not matches:
        result['rules'].append({**deepcopy(RULE), 'enabled': enabled})
    return result
