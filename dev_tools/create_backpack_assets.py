"""Extract controls from the native full-bag and port-sale verification."""
from pathlib import Path
import shutil

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'diagnostics/2026-10-07/level-and-bag'
ASSETS = ROOT/'assets/backpack'
FIXTURES = ROOT/'tests/fixtures/backpack'
ASSETS.mkdir(parents=True, exist_ok=True)
FIXTURES.mkdir(parents=True, exist_ok=True)
controls = {
    'bag-icon': ('before', (644, 1147, 695, 1202)),
    'map-port': ('port-map', (269, 858, 312, 903)),
    'map-player-current': ('port-map', (338, 719, 380, 759)),
    'header-port': ('port-travel', (253, 38, 290, 77)),
    'port-trade': ('port-travel', (639, 1055, 711, 1125)),
    'moving': ('port-selected', (294, 44, 355, 67)),
    'bag-title': ('bag-open', (41, 847, 161, 890)),
    'tab-first': ('bag-open', (38, 769, 82, 818)),
    'sell': ('bag-open', (553, 845, 688, 885)),
}
for name, (source, box) in controls.items():
    with Image.open(FOLDER/(source+'.png')) as image:
        image.crop(box).save(ASSETS/(name+'.png'))
for name in ('before', 'bag-open', 'bag-help', 'bag-help-closed', 'sale-entry',
             'port-map', 'port-selected', 'port-travel', 'port-bag', 'sold-stable',
             'sold-closed', 'return-map', 'return-training-selected', 'return-travel', 'returned-main'):
    shutil.copyfile(FOLDER/(name+'.png'), FIXTURES/(name+'.png'))
for name, source in (('cooking', 'cooking'), ('food', 'tab-three')):
    shutil.copyfile(ROOT/'diagnostics/2026-10-03/backpack'/(source+'.png'),
                    FIXTURES/(name+'.png'))
