"""Append the requested guild errand; never replace existing rules/settings."""
from copy import deepcopy

RULE = dict(id='driftinggirl_guild_daily',name='公会任务：每天18:00提交两类任务',
    enabled=True,asset='assets/parallel/world-badge.png',source_crop=[0,0,26,26],
    search_area=[10,43,42,75],scales=[1],threshold=.96,interval=86400,stable_frames=3,
    stage='guild_daily_owned',action='observe',
    note='18:00后执行一次；保留原世界和航巡，不打开任务进度、不购物、不关闭或重开次元门。')

def add(config):
    result=deepcopy(config)
    count=sum(r['id']==RULE['id'] for r in result['rules'])
    if count>1:raise ValueError('Duplicate guild daily rule')
    if not count:result['rules'].append(deepcopy(RULE))
    return result
