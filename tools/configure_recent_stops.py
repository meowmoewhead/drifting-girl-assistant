"""Fix observed mana OCR, missed RAID entry and late crown return confirmation."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW_ID = 'driftinggirl_raid_entry_recover'
CHANGED_FIELDS = {
    'driftinggirl_mana_potion': ('number_condition', 'note'),
    'driftinggirl_raid_open': ('recheck_before_click',),
    'driftinggirl_achievements_return': ('timeout_grace', 'note'),
}


def update(config):
    result = deepcopy(config)
    rules = {r['id']: r for r in result['rules']}
    for key in CHANGED_FIELDS:
        if sum(r['id'] == key for r in result['rules']) != 1:
            raise ValueError('Missing or duplicate rule: ' + key)
    mana = rules['driftinggirl_mana_potion']
    # The digits occupy y=803..824; surrounding panel margins depressed OCR
    # confidence for 220/226. Keep the complete numeric width and fixed threshold.
    mana['number_condition']['source_area'] = [635, 802, 690, 826]
    mana['note'] = '仅匹配药水图标，库存数字另读；裁去数字上下的面板空白，库存置信度仍须达到原阈值且大于 0。'
    entry = rules['driftinggirl_raid_open']
    entry['recheck_before_click'] = True
    returned = rules['driftinggirl_achievements_return']
    returned['timeout_grace'] = 8
    returned['note'] = ('三个清晰底栏图标且皇冠菜单消失，连续三帧才确认返回。'
                        '超时临界点仅在返回画面仍匹配时允许最多八秒补足连续帧，不发送点击；未知或遮挡画面仍停止。')
    recovery = dict(id=NEW_ID, name='RAID：入口未打开时重新识别',
        enabled=rules.get(NEW_ID, entry)['enabled'], asset='assets/navigation/main.png',
        source_crop=[211, 1140, 266, 1189], search_area=[200, 1130, 276, 1200],
        scales=[1], threshold=.93, stable_frames=3,
        interval=rules.get(NEW_ID, {}).get('interval', 1), action='observe',
        stage=entry['next_stage'], next_stage='main', min_stage_seconds=6,
        recovery_entry=entry['id'],
        required_source_areas=[[542, 1140, 597, 1183], [437, 1144, 486, 1193], [83, 1057, 139, 1114]],
        color_condition=dict(source_area=[558, 1147, 585, 1168],
            hsv=[[0, 0, 210], [179, 255, 255]], minimum_fraction=.3),
        note='仅在点击 RAID 入口、尚未点准备时恢复：至少六秒且三个底栏图标及 AUTO 控件连续三帧清晰可见；排除技能面板和突袭界面。只观察，保留原入口间隔，五分钟最多恢复三次，不记为突袭完成。')
    result['rules'] = [r for r in result['rules'] if r['id'] != NEW_ID] + [recovery]
    return result


if __name__ == '__main__':
    path = ROOT/'config.json'
    path.write_text(json.dumps(update(json.loads(path.read_text(encoding='utf-8-sig'))),
                               ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
