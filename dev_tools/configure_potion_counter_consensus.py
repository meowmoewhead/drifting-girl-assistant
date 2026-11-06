"""Enable agreement-based OCR only for the potion advertisement counter."""
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
    rule['counter_condition']['preprocess'] = 'white_text_consensus'
    rule['note'] = ('在右侧图标栏定位明亮药水瓶，计数和点击随位置移动。'
        '对完整 n/40 使用原图和白字图交叉识别，至少两路高置信度结果一致才接受；'
        '任一高置信度结果冲突则跳过。不降低 0.93 门槛，连续三帧读数一致且 n<40 才点击，'
        '点击前再次截图复核位置与次数。每日重置以游戏画面为准。')
    return config


if __name__ == '__main__':
    path = ROOT/'config.json'
    path.write_text(json.dumps(update(json.loads(path.read_text(encoding='utf-8-sig'))),
                               ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
