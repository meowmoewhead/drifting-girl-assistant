"""Add an independently switchable skill workflow without changing old settings."""
from copy import deepcopy
import json
from pathlib import Path

RULE = dict(id='driftinggirl_parallel_skills', name='平行世界：钓鱼强化、自动使用及技能点加速',
            enabled=True, asset='assets/parallel/world-badge.png', source_crop=[0, 0, 26, 26],
            search_area=[9, 43, 43, 76], scales=[1], threshold=.92, interval=300,
            stable_frames=2, stage='skills_owned', action='observe', growth_points=True,
            note='开启钓鱼速度强化与自动使用；关闭移动及世界加速自动使用。50%成长加速到期且技能点足够时仅花200技能点兑换1小时，购买前记录有效期。')

def add(config):
    result = deepcopy(config)
    matches = [r for r in result['rules'] if r['id'] == RULE['id']]
    if len(matches) > 1:
        raise ValueError('Duplicate parallel-skill rule')
    if not matches:
        result['rules'].append(deepcopy(RULE))
    return result

if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1]/'config.json'
    path.write_text(json.dumps(add(json.loads(path.read_text(encoding='utf-8-sig'))),
                               ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
