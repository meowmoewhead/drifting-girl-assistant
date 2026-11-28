"""Add the configurable RANK restart rule."""
from copy import deepcopy
from drift.tasks.stone_options import DEFAULT_RESTART_RANK

RULE = dict(id='driftinggirl_stone_farm', name='次元石：按 RANK 重开',
    enabled=False, asset='assets/parallel/world-badge.png', source_crop=[0,0,26,26],
    search_area=[13,46,39,72], scales=[1], threshold=.92, interval=2,
    stable_frames=3, stage='stone_owned', action='observe',
    restart_rank=DEFAULT_RESTART_RANK,
    note='连续确认到达设定 RANK 后停航停钓，返回并关门重开。')


def add(config, enabled=False):
    result = deepcopy(config)
    matches = [r for r in result['rules'] if r['id'] == RULE['id']]
    if len(matches) > 1:
        raise ValueError('Duplicate stone-farming rule')
    if not matches:
        result['rules'].append({**deepcopy(RULE), 'enabled': enabled})
    return result
