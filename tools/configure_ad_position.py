"""Find the growth ad after optional right-hand icons disappear."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE_ID = 'driftinggirl_bottom_right_arrow'


def update(config):
    config = deepcopy(config)
    entries = [rule for rule in config['rules'] if rule['id'] == RULE_ID]
    if len(entries) != 1:
        raise ValueError('Expected one growth advertisement entry')
    entries[0]['search_area'] = [600,650,720,1115]
    entries[0]['color_condition'] = dict(source_area=[639,921,660,945],
        hsv=[[35,80,170],[85,255,255]],minimum_fraction=.30)
    entries[0]['note'] = ('在右侧图标栏定位成长箭头，兼容其他入口隐藏后的上下移动；'
        '每日计数区域随箭头位置换算，连续三帧 n/10 相同且 n<10 才点击。'
        '同时要求箭头正常亮度；满额、读不清或分母不为 10 时跳过，按游戏画面判断刷新。')
    return config


if __name__ == '__main__':
    path = ROOT/'config.json'
    path.write_text(json.dumps(update(json.loads(path.read_text(encoding='utf-8-sig'))),
                               ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
