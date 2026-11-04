"""Recover verified, interrupted mailbox/crown openings without blind retries."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW_IDS = tuple('driftinggirl_' + suffix for suffix in (
    'achievements_resume', 'achievements_recover', 'achievements_battle_recover',
    'mailbox_recover', 'mailbox_battle_recover'))
CHANGED_IDS = ('driftinggirl_achievements_open', 'driftinggirl_achievements_crown',
               'driftinggirl_mailbox_open')


def update(config):
    updated = deepcopy(config)
    existing = {r['id']: r for r in updated['rules']}
    # Correlation alone still matches dimmed controls underneath modal windows.
    for key, area, fraction in (
        ('driftinggirl_achievements_open', [660,555,678,576], .35),
        ('driftinggirl_mailbox_open', [675,480,696,493], .45),
    ):
        existing[key]['color_condition'] = dict(source_area=area,
            hsv=[[0,0,180],[179,255,255]], minimum_fraction=fraction)
    existing['driftinggirl_achievements_open'].update(
        absent_rule='driftinggirl_achievements_fold',
        note='明亮的任务入口与绿色 N 同时出现且菜单尚未展开才点击；菜单已展开时接续皇冠步骤。')
    crown = existing['driftinggirl_achievements_crown']
    crown.pop('required_source_areas', None)
    crown['note'] = '点击展开菜单中的皇冠；入口 N 也可能来自其他任务，皇冠自身无 N 时仍检查列表并正常退出。'
    existing['driftinggirl_mailbox_open']['note'] = '匹配明亮信封和绿色 N；若点击后仍在主界面或被巨鱼战斗打断，确认画面后有限重试。'

    resume = deepcopy(crown)
    resume.update(id=NEW_IDS[0], name='皇冠奖励：接续已经展开的菜单',
        enabled=existing['driftinggirl_achievements_open']['enabled'],
        stage='main', action='observe', next_stage='achievements_tab',
        interval=existing['driftinggirl_achievements_open']['interval'],
        required_source_areas=[[211,1140,266,1189],[542,1140,597,1183]],
        color_condition=dict(source_area=[558,1147,585,1168],
            hsv=[[0,0,210],[179,255,255]], minimum_fraction=.3),
        note='只观察已展开且明亮的皇冠菜单，不再次点入口把它收起。')
    added = [resume]
    for feature, stage in (('achievements', 'achievements_tab'), ('mailbox', 'mailbox')):
        entry = existing[f'driftinggirl_{feature}_open']
        base = dict(enabled=entry['enabled'], scales=[1], threshold=.93,
            stable_frames=3, interval=1, action='observe', stage=stage,
            next_stage='main', min_stage_seconds=6, recovery_entry=entry['id'])
        normal = dict(base, id=f'driftinggirl_{feature}_recover',
            name=('皇冠奖励' if feature == 'achievements' else '邮箱')+'：入口未打开时恢复重试',
            asset='assets/navigation/main.png', source_crop=[211,1140,266,1189],
            search_area=[200,1130,276,1200],
            required_source_areas=[[542,1140,597,1183],[650,550,690,587],[659,479,701,505]],
            color_condition=dict(source_area=[558,1147,585,1168],
                hsv=[[0,0,210],[179,255,255]], minimum_fraction=.3),
            note='入口和两个稳定底栏图标连续三帧清晰可见才放弃旧步骤；保留入口间隔，五分钟最多恢复三次，恢复不算领奖完成。')
        if feature == 'achievements':
            normal['absent_rule'] = 'driftinggirl_achievements_fold'
        battle = dict(base, id=f'driftinggirl_{feature}_battle_recover',
            name=('皇冠奖励' if feature == 'achievements' else '邮箱')+'：巨鱼战斗打断后等待入口恢复',
            asset='assets/navigation/world-battle.png', source_crop=[334,133,355,157],
            search_area=[324,123,365,167],
            required_source_areas=[[211,1140,266,1189],[542,1140,597,1183]],
            color_condition=dict(source_area=[558,1147,585,1168],
                hsv=[[0,0,210],[179,255,255]], minimum_fraction=.3),
            note='同时识别巨鱼战斗沙漏与两个清晰底栏图标，仅撤销被打断的入口步骤；不点击战斗或提前宣布领奖完成。')
        for rule in (normal, battle):
            if rule['id'] in existing:
                rule['enabled'] = existing[rule['id']]['enabled']
            added.append(rule)
    if resume['id'] in existing:
        resume['enabled'] = existing[resume['id']]['enabled']
    # Resume before the ordinary entry; recovery observations before no other actions.
    updated['rules'] = added + [r for r in updated['rules'] if r['id'] not in NEW_IDS]
    return updated


if __name__ == '__main__':
    path = ROOT / 'config.json'
    path.write_text(json.dumps(update(json.loads(path.read_text(encoding='utf-8-sig'))),
                               ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
