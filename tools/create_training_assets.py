"""Extract fishing map markers from verified native-size emulator captures."""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


if __name__ == '__main__':
    source = ROOT/'diagnostics/2026-10-06'
    target = ROOT/'assets/training'
    target.mkdir(exist_ok=True)
    Image.open(source/'boss-level-gap-map.png').crop((301, 712, 334, 742)).save(target/'map-pin.png')
    # The boss skull may overlap the portrait's lower-right edge.
    Image.open(source/'boss-level-gap-training-map-arrived.png').crop((392, 844, 420, 868)).save(target/'map-player.png')
    Image.open(source/'fishing-failure-probe/frame-029.png').crop((324, 973, 397, 1046)).save(target/'failure-cross.png')
    Image.open(source/'parallel-boss-result.png').crop((313, 985, 416, 1035)).save(target/'success-check.png')
    fixtures = ROOT/'tests/fixtures/training'
    fixtures.mkdir(exist_ok=True)
    for name, file in (('boss-main', 'boss-level-gap-current.png'),
                       ('boss-map', 'boss-level-gap-map.png'),
                       ('fishing-map', 'boss-level-gap-training-map-arrived.png'),
                       ('fishing-main', 'boss-level-gap-training-main.png'),
                       ('fishing-691', 'training-current.png')):
        (fixtures/(name+'.png')).write_bytes((source/file).read_bytes())
    for name, index in (('failed', 29), ('failed-faded', 11), ('battle', 7), ('waiting', 14)):
        (fixtures/(name+'.png')).write_bytes((source/'fishing-failure-probe'/f'frame-{index:03d}.png').read_bytes())
    (fixtures/'success.png').write_bytes((source/'parallel-boss-result.png').read_bytes())
    for name, index in (('failed-boss', 2), ('waiting-boss', 3)):
        (fixtures/(name+'.png')).write_bytes((source/'task-boss-outcomes'/f'frame-{index:03d}.png').read_bytes())
    import zipfile
    with zipfile.ZipFile(source/'failure-missed/probe.zip') as frames:
        for name, index in (('failed-rank35', 31), ('waiting-rank35', 32),
                            ('battle-rank35', 19), ('success-rank35', 8)):
            (fixtures/(name+'.png')).write_bytes(frames.read(f'frame-{index:03d}.png'))
    print('Created 4 training templates and 16 real-screen fixtures')
