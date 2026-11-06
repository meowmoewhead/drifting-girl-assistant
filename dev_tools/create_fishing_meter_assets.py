"""Extract the sleeping/ready dial from native device captures."""
from pathlib import Path
import shutil
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    target = ROOT/'assets/fishing-meter'
    fixtures = ROOT/'tests/fixtures/fishing-meter'
    target.mkdir(parents=True, exist_ok=True)
    fixtures.mkdir(parents=True, exist_ok=True)
    source = ROOT/'diagnostics/2026-10-07'
    sleeping = source/'fishing-meter/paused.png'
    ready = source/'level-and-bag/meter-drag-label.png'
    Image.open(sleeping).crop((313,964,391,1034)).save(target/'sleep.png')
    Image.open(ready).crop((330,963,390,1048)).save(target/'hook.png')
    waiting = ROOT/'diagnostics/2026-10-08/boss-tentacle/paused-main.png'
    Image.open(waiting).crop((314,1002,332,1020)).save(target/'waiting-dot.png')
    Image.open(waiting).crop((641,1073,677,1104)).save(target/'cancel-cast.png')
    port = ROOT/'tests/fixtures/fishing-meter/port-after-sale.png'
    Image.open(port).crop((318,962,408,1051)).save(target/'port-anchor.png')
    Image.open(ROOT/'tests/fixtures/fishing-meter/low-meter-bite.png').crop((344,964,376,1047)).save(target/'bite-warning.png')
    for name, path in (('sleep',sleeping), ('ready',ready),
                       ('casting',source/'level-and-bag/meter-start.png'),
                       ('active',source/'fishing-meter/after-start.png'),
                       ('sleep-selected',source/'fishing-meter/restart-main.png'),
                       ('ready-shifted',source/'fishing-meter/native-008-active.png'),
                       ('fill-hook',source/'fishing-meter/feedback-during.png'),
                       ('fill-catch',source/'fishing-idle-fill/main-0.png'),
                       ('fill-empty',source/'fishing-idle-fill/main-1.png'),
                       ('fill-result',source/'fishing-idle-fill/main-3.png')):
        shutil.copyfile(path, fixtures/(name+'.png'))
    print('Fishing-meter templates and native fixtures saved')
