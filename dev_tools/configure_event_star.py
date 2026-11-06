"""Add the observed blue-star event entry, preserving live user choices."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'driftinggirl_event_potion_open'
ASSET = 'assets/event-potion/main-star.png'
BRIGHT_CORE = [665, 293, 685, 311]


def update(config):
    result = deepcopy(config)
    entries = [r for r in result['rules'] if r['id'] == ENTRY]
    if len(entries) != 1:
        raise ValueError('Expected exactly one event potion entry')
    rule = entries[0]
    if (rule.get('asset') != 'assets/event-potion/main.png'
            or rule.get('source_crop') != [650, 280, 701, 320]
            or rule.get('color_condition', {}).get('source_area') not in
            ([652, 282, 700, 318], BRIGHT_CORE)):
        raise ValueError('Custom event entry layout requires independent review')
    variants = rule.setdefault('asset_variants', [])
    if ASSET not in variants:
        variants.append(ASSET)
    # Blue tips are intentionally dark. Check the fixed bright centre with
    # the original HSV threshold and fraction; dim battle/menu icons fail.
    # The centre also remains bright in the three existing seasonal assets.
    rule['color_condition']['source_area'] = BRIGHT_CORE.copy()
    return result


if __name__ == '__main__':
    path = ROOT / 'config.json'
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    path.write_text(json.dumps(update(config), ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8')
