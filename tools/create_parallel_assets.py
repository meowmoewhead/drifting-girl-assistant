"""Extract fixed visual anchors from the verified 720x1280 parallel-world captures."""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "diagnostics" / "2026-10-06"
DEST = ROOT / "assets" / "parallel"
DEST.mkdir(parents=True, exist_ok=True)

CROPS = {
    "world-badge": ("parallel-main", (13, 46, 39, 72)),
    "next-arrow": ("parallel-main", (307, 64, 398, 127)),
    "map-close": ("parallel-map-rank2", (632, 1151, 705, 1225)),
    "map-boss": ("parallel-map-rank2", (370, 407, 401, 440)),
    "map-observatory": ("map-swipe-right-more-2", (519, 783, 561, 823)),
    "observatory-label": ("observatory-main", (306, 12, 411, 45)),
    "gate-close": ("portal-open-confirmed", (154, 685, 565, 785)),
    "gate-activate": ("portal-cat", (154, 751, 565, 852)),
    "return-question": ("parallel-return-entry", (223, 556, 506, 629)),
    "enter-question": ("parallel-enter", (220, 556, 502, 620)),
    "search-question": ("portal-tap", (194, 555, 525, 620)),
    "continue": ("portal-tap", (63, 741, 356, 834)),
    "panel-close": ("portal-open-confirmed", (548, 352, 638, 433)),
}

for name, (source, box) in CROPS.items():
    with Image.open(SOURCE / f"{source}.png") as image:
        image.crop(box).save(DEST / f"{name}.png")
    print(name, box)
