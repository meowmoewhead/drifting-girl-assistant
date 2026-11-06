"""Add the authorized daily PASS task without changing existing switches."""
from copy import deepcopy
import json
from pathlib import Path
RULE=dict(id='driftinggirl_pass_daily',name='PASS每日通行证：10点后钓满120鱼领取',
    enabled=True,asset='assets/pass/entry.png',source_crop=[624,174,693,201],
    search_area=[624,174,693,201],scales=[1],threshold=.96,interval=600,stable_frames=3,
    stage='pass_owned',action='observe',
    note='每天08:00重置、10:00后检查；未钓满120鱼则稍后重试。只点免费奖励，核对四档绿勾并返回后记完成。')
def add(config):
    result=deepcopy(config)
    matches=[r for r in result['rules'] if r['id']==RULE['id']]
    if len(matches)>1:raise ValueError('Duplicate PASS daily rule')
    if not matches:result['rules'].append(deepcopy(RULE))
    return result
if __name__=='__main__':
    path=Path(__file__).resolve().parents[1]/'config.json'
    path.write_text(json.dumps(add(json.loads(path.read_text(encoding='utf-8-sig'))),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
