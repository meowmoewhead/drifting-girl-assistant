"""Finish crown rewards using stable game controls when combat hides the menu."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLOSE = 'driftinggirl_achievements_close'
FOLD = 'driftinggirl_achievements_fold'
RETURN = 'driftinggirl_achievements_return'
CHANGED_FIELDS = {
    CLOSE: ('next_stage',),
    FOLD: ('stage', 'note'),
    RETURN: ('asset', 'source_crop', 'search_area', 'scales', 'threshold',
             'required_source_areas', 'color_condition', 'stable_frames', 'note'),
}


def update(config):
    result = deepcopy(config)
    entries = {key: [r for r in result['rules'] if r['id'] == key]
               for key in CHANGED_FIELDS}
    if any(len(rules) != 1 for rules in entries.values()):
        raise ValueError('Expected exactly one crown close, fold and return rule')
    rules = {key: members[0] for key, members in entries.items()}
    # A closed list may reveal the submenu, the normal game, or ongoing combat.
    # In all cases inspect the actual screen instead of requiring a fold click.
    rules[CLOSE]['next_stage'] = 'achievements_return'
    rules[FOLD].update(stage='achievements_return',
        note='关闭奖励列表后，仅菜单中的皇冠仍可见时点击收起；若已直接返回主界面或巨鱼战斗，由返回观察规则确认。')
    returned = rules[RETURN]
    returned.update(asset='assets/navigation/main.png',
        source_crop=[211, 1140, 266, 1189], search_area=[200, 1130, 276, 1200],
        scales=[1], threshold=max(returned['threshold'], .93),
        required_source_areas=[[542, 1140, 597, 1183], [437, 1144, 486, 1193]],
        color_condition=dict(source_area=[558, 1147, 585, 1168],
            hsv=[[0, 0, 210], [179, 255, 255]], minimum_fraction=.3),
        stable_frames=max(returned['stable_frames'], 3),
        note='关闭列表或收起菜单后，三个稳定底栏图标连续三帧清晰可见且皇冠菜单消失才恢复主流程；兼容巨鱼战斗隐藏右侧任务入口，不依赖 N、怪物或动态技能图标。')
    return result


if __name__ == '__main__':
    path = ROOT/'config.json'
    path.write_text(json.dumps(update(json.loads(path.read_text(encoding='utf-8-sig'))),
                               ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
