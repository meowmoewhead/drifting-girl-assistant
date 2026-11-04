"""Recognize NPC task notices and close only their verified red cross."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDS = ('driftinggirl_task_popup_close', 'driftinggirl_task_popup_return')


def update(config):
    config = deepcopy(config)
    existing = {rule['id']: rule for rule in config['rules']}
    close = dict(id=IDS[0], name='任务弹窗：关闭右上红叉', enabled=True,
                 asset='assets/task-popup/notice.png', source_crop=[618,406,674,453],
                 search_area=[565,365,710,500], scales=[1], threshold=.93,
                 required_source_areas=[[315,465,350,492],[271,548,302,597],[106,785,162,810]],
                 color_condition=dict(source_area=[637,418,657,438],
                                      hsv=[[0,90,130],[12,255,255]], minimum_fraction=.7),
                 click_source_area=[627,412,665,449], stable_frames=3, interval=2,
                 stage='*', interrupt=True, next_stage='task_popup_return',
                 stage_timeout=15, next_delay=.8,
                 note='同时确认红叉、任务对话框轮廓和奖励文字，连续三帧后只关闭红叉；不点击结束或奖励。')
    returned = deepcopy(existing.get(IDS[1], existing['driftinggirl_raid_recover']))
    returned.update(id=IDS[1], name='任务弹窗：确认关闭并恢复任务', enabled=True,
                    asset='assets/task-popup/main.png', source_crop=[279,884,307,910],
                    search_area=[276,881,310,913],
                    asset_variants=['assets/task-popup/main-battle.png'],
                    required_source_areas=[[211,1140,266,1189],[542,1140,597,1183]],
                    color_condition=dict(source_area=[558,1147,585,1168],
                                         hsv=[[0,0,210],[179,255,255]], minimum_fraction=.3),
                    stage='task_popup_return', next_stage='main', absent_rule=IDS[0],
                    stable_frames=3,
                    note='弹窗消失且主界面 AUTO 勾选、星形材料及明亮钻石同时清晰可见后恢复；不依赖会出现提示标记的法力图标。')
    returned.pop('min_stage_seconds', None)
    for rule in (close, returned):
        rule['enabled'] = existing.get(rule['id'], rule)['enabled']
    close['interval'] = existing.get(IDS[0], close)['interval']
    config['rules'] = [close, returned] + [rule for rule in config['rules'] if rule['id'] not in IDS]
    return config


if __name__ == '__main__':
    path = ROOT/'config.json'
    config = update(json.loads(path.read_text(encoding='utf-8-sig')))
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Rules:',len(config['rules']))
