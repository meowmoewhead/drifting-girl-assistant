"""Extract skill controls from native, authorized device verification frames."""
from pathlib import Path
import shutil
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parallel_skills import ParallelSkills

if __name__ == '__main__':
    source = ROOT/'diagnostics/2026-10-07/level-and-bag'
    target = ROOT/'assets/parallel-skills'
    fixtures = ROOT/'tests/fixtures/parallel-skills'
    target.mkdir(parents=True, exist_ok=True)
    fixtures.mkdir(parents=True, exist_ok=True)
    panel = Image.open(source/'parallel-growth-confirm.png')
    modal = Image.open(source/'parallel-growth-points.png')
    for name, box in ParallelSkills.BOXES.items():
        (modal if name.startswith('confirm-') or name == 'continue' else panel).crop(box).save(target/(name+'.png'))
    for name in ('parallel-skill-panel', 'parallel-fishing-auto', 'parallel-fishing-enhance',
                 'parallel-growth-points', 'parallel-growth-confirm'):
        shutil.copyfile(source/(name+'.png'), fixtures/(name+'.png'))
    shutil.copyfile(ROOT/'diagnostics/2026-10-07/parallel-only/paused.png', fixtures/'main.png')
    print('Skill assets and native replay fixtures created')
