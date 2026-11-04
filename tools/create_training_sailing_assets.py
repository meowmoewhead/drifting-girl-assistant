"""Extract AUTO controls from verified 720x1280 emulator screenshots."""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


if __name__ == '__main__':
    source = ROOT/'diagnostics/2026-10-07/auto-sailing'
    assets = ROOT/'assets/training-sailing'
    fixtures = ROOT/'tests/fixtures/training-sailing'
    assets.mkdir(exist_ok=True)
    fixtures.mkdir(exist_ok=True)
    menu = Image.open(source/'auto-menu.png')
    main = Image.open(source/'before.png')
    sailing = Image.open(source/'auto-started.png')
    boxes = {'menu-target':(150,439,207,475), 'menu-destination':(128,531,226,564),
             'menu-start':(246,850,491,895), 'menu-close':(564,330,639,397),
             'checkbox-off':(557,627,595,667)}
    for name,box in boxes.items():
        menu.crop(box).save(assets/(name+'.png'))
    main.crop((186,34,253,65)).save(assets/'auto-label.png')
    main.crop((194,4,246,42)).save(assets/'inactive-face.png')
    main.crop((25,1147,70,1185)).save(assets/'main-bottle.png')
    sailing.crop((194,4,246,42)).save(assets/'active-arrow.png')
    sailing.crop((276,881,310,913)).save(assets/'main-auto-sailing.png')
    for name,x in (('pearl',272),('material',345),('amber',417),('gem',490),('boss',562)):
        menu.crop((x-24,432,x+24,484)).save(assets/(name+'-icon.png'))
    for name,file in (('menu-one','auto-menu'),('menu-two','amber-selected'),('menu-three','configured-final'),
                      ('menu-no-pearl','pearl-off'),('active','auto-started'),
                      ('inactive','before'),('stopped','running-auto-menu')):
        (fixtures/(name+'.png')).write_bytes((source/(file+'.png')).read_bytes())
    (fixtures/'active-flying-cake.png').write_bytes((source/'guard-0.png').read_bytes())
    for i in range(3):
        (fixtures/f'active-glow-{i}.png').write_bytes((source/f'post-start-{i}.png').read_bytes())
    print('Created 15 AUTO templates and eleven native fixtures')
