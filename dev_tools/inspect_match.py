"""Inspect saved real frames without sending device actions."""
import argparse
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import Clicker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('frame')
    parser.add_argument('prefix')
    args = parser.parse_args()
    config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8-sig'))
    context = SimpleNamespace(root=ROOT, config=config, lock=threading.RLock(),
                              digit_reader=None, state={'numbers': {}, 'frames': 1})
    frame = np.asarray(Image.open(ROOT / 'captures' / args.frame).convert('RGB'))
    for rule in config['rules']:
        if args.prefix not in rule['id']:
            continue
        _, button, offset = Clicker.load_rule(context, rule)
        matched = button.match(frame, offset=offset, similarity=rule['threshold'])
        result = {'id': rule['id'], 'matched': matched,
                  'score': round(button.last_similarity, 5),
                  'box': [int(x) for x in button.button]}
        if hasattr(button, 'green_count'):
            result.update(green_count=button.green_count, row=button.target_key)
        if matched and 'number_condition' in rule:
            result['number_ok'] = Clicker.number_condition(context, rule, button, frame)
            result['number'] = context.state['numbers'].get(rule['id'])
        print(json.dumps(result, default=str))


if __name__ == '__main__':
    main()
