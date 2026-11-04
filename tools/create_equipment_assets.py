"""Extract stable weapon and currency fragments from actual emulator frames."""
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from equipment_upgrade import WEAPONS


if __name__ == '__main__':
    source = ROOT/'diagnostics/2026-10-06'
    target = ROOT/'assets/equipment'
    target.mkdir(exist_ok=True)
    images = {name: Image.open(source/file) for name, file in (
        ('main', 'equipment-before.png'), ('top', 'equipment-top.png'),
        ('bottom', 'equipment-menu.png'))}
    pieces = {
        'entry': ('main', (330, 1143, 375, 1189)),
        'guard-star': ('main', (211, 1140, 266, 1189)),
        'guard-diamond': ('main', (542, 1140, 597, 1183)),
        'main-auto': ('main', (279, 884, 307, 910)),
        'header-currency': ('top', (30, 800, 60, 829)),
        'header-tap': ('top', (335, 801, 403, 826)),
        'header-attack': ('top', (518, 801, 563, 826)),
        'arrow': ('top', (535, 850, 575, 901)),
        'row-currency': ('top', (596, 875, 621, 900)),
    }
    for key, (_, icon, name, _) in WEAPONS.items():
        page = 'top' if key in ('rod', 'harpoon', 'cannon') else 'bottom'
        pieces[key+'-icon'] = (page, icon)
        pieces[key+'-name'] = (page, name)
    for name, (page, box) in pieces.items():
        images[page].crop(box).save(target/(name+'.png'))
    for name in ('main-ring.png', 'main-ring-normal.png'):
        (target/name).unlink(missing_ok=True)
    print(f'Created {len(pieces)} equipment templates')
