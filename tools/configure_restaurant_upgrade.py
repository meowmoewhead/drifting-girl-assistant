"""Append the independent cookbook task without rewriting saved settings."""
from copy import deepcopy

RULE = dict(id='driftinggirl_restaurant_upgrade',name='餐品强化：图鉴四种食材提交',
    enabled=True,asset='assets/parallel/world-badge.png',source_crop=[0,0,26,26],
    search_area=[10,43,42,75],scales=[1],threshold=.96,interval=259200,stable_frames=3,
    stage='restaurant_upgrade_owned',action='observe',
    note='默认每三天；在当前世界餐馆的图鉴强化永久效果，不制作、不食用、不购买食材。')

def add(config):
    result=deepcopy(config)
    count=sum(r['id']==RULE['id'] for r in result['rules'])
    if count>1:raise ValueError('Duplicate restaurant upgrade rule')
    if not count:result['rules'].append(deepcopy(RULE))
    return result
