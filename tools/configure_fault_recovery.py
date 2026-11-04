"""Repair missing RAID return transitions and allow slow mailbox responses."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def update(config):
    config = deepcopy(config)
    rules = {rule['id']: rule for rule in config['rules']}
    entry = rules['driftinggirl_raid_open']
    variants = entry.setdefault('asset_variants', [])
    if 'assets/raid/main-octopus.png' not in variants:
        variants.append('assets/raid/main-octopus.png')
    entry['color_condition'] = dict(source_area=[692,819,707,838],
                                   hsv=[[15,80,160],[45,255,255]], minimum_fraction=.4)
    recovery = deepcopy(rules['driftinggirl_raid_return'])
    recovery.update(id='driftinggirl_raid_recover', name='RAID：已回主界面时恢复挂机',
                    enabled=rules.get('driftinggirl_raid_recover', entry)['enabled'],
                    stage='raid_battle', next_stage='main', action='observe', min_stage_seconds=15,
                    stable_frames=5, required_source_areas=[[227,1146,271,1183],[528,1148,577,1184]],
                    color_condition=dict(source_area=[333,1150,376,1184],
                                         hsv=[[0,0,160],[179,255,255]], minimum_fraction=.45),
                    note='开始后至少 15 秒且主界面三个底栏图标连续五帧清晰可见，才接受结算已自动关闭。仅观察，不发送点击。')
    recovery.pop('click_source_area', None)
    config['rules'] = [rule for rule in config['rules'] if rule['id'] != recovery['id']]
    index = next(i for i, rule in enumerate(config['rules']) if rule['id'] == 'driftinggirl_raid_return')
    config['rules'].insert(index+1, recovery)
    mailbox = rules['driftinggirl_mailbox_claim_all']
    mailbox['stage_timeout'] = max(120, mailbox.get('stage_timeout', 20))
    mailbox['note'] = '全部领取只点击一次；给慢响应最多 120 秒，识别邮箱已空才关闭，超时仍停止。'
    return config


if __name__ == '__main__':
    path = ROOT/'config.json'
    config = update(json.loads(path.read_text(encoding='utf-8-sig')))
    path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Rules:',len(config['rules']))
