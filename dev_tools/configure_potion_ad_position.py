"""Keep potion advertisements and their counters readable as the icon stack moves."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE_ID = 'driftinggirl_potion_ad_open'


def update(config):
    config = deepcopy(config)
    entries = [rule for rule in config['rules'] if rule['id'] == RULE_ID]
    if len(entries) != 1:
        raise ValueError('Expected one potion advertisement entry')
    rule = entries[0]
    rule['search_area'] = [600,650,720,1115]
    # Keep the complete width, including two-digit numerators. Exclude extra
    # rows above/below the text without changing OCR or its confidence gate.
    rule['counter_condition']['source_area'] = [650,904,718,924]
    rule['color_condition'] = dict(source_area=[659,930,688,954],
        hsv=[[0,0,170],[179,255,255]],minimum_fraction=.40)
    rule['note'] = ('在右侧图标栏定位药水瓶，兼容 RAID 本周打满后入口隐藏、下周恢复时的上下移动；'
        '计数与点击区域跟随瓶身换算。计数保留完整宽度，仅排除上下多余背景行，提取白色文字读取 n/40；'
        '连续三帧读数一致、置信度至少 0.93 且 n<40 才点击。瓶身须正常明亮；'
        '满额、读不清或分母不为 40 时跳过，按游戏画面判断刷新。')
    return config


if __name__ == '__main__':
    path = ROOT/'config.json'
    path.write_text(json.dumps(update(json.loads(path.read_text(encoding='utf-8-sig'))),
                               ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
