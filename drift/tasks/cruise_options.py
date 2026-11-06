"""One saved policy for native cruise and parallel-world skills."""
from copy import deepcopy
import hashlib
import json

TARGET_NAMES = ('pearl', 'material', 'amber', 'gem', 'boss')
SKILL_NAMES = ('fishing', 'movement', 'time')
DEFAULTS = {
    'cruise': {'enabled': True, 'targets': dict(zip(TARGET_NAMES, (True, False, True, True, True))),
               'auto_sale': True, 'auto_upgrade': True, 'ticket_min': 10},
    'skills': {'enabled': True, 'fishing': {'auto': True, 'enhance': True},
               'movement': {'auto': False, 'enhance': False},
               'time': {'auto': False, 'enhance': False}, 'growth_points': True},
}


def normalize(value=None, *, growth_points=True):
    result = deepcopy(DEFAULTS)
    result['skills']['growth_points'] = growth_points
    if value is None:
        return result
    def merge(dst, src, path):
        if not isinstance(src, dict) or set(src) - set(dst):
            raise ValueError('巡航设置字段无效：' + path)
        for key, item in src.items():
            if isinstance(dst[key], dict):
                merge(dst[key], item, path + '.' + key)
            elif key == 'ticket_min':
                if type(item) is not int or not 0 <= item <= 9999:
                    raise ValueError('自动贩卖补票阈值须为0至9999的整数，0表示不补票')
                dst[key] = item
            else:
                if type(item) is not bool:
                    raise ValueError('巡航和技能开关须为布尔值：' + path + '.' + key)
                dst[key] = item
    merge(result, value, 'cruise_settings')
    if result['cruise']['enabled'] and not any(result['cruise']['targets'].values()):
        raise ValueError('启用巡航时至少选择一个目标')
    return result


def from_rule(rule):
    return normalize(rule.get('cruise_settings'), growth_points=rule.get('growth_points', True))


def fingerprint(options):
    return hashlib.sha256(json.dumps(options, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


