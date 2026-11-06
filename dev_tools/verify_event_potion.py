"""Replay saved event frames through Clicker.load_rule; never create a device."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import Clicker
from drift.core.util import hsv_mask

PREFIX = "driftinggirl_event_potion_"
STEPS = ("open", "select", "confirm", "close", "return")
EXPECTED = {"main": {"open", "return"}, "shop": {"select", "close"},
            "confirm": {"confirm"}}


def inspect_rule(context, rule, frame):
    """Use the real matcher for verdicts and its ALAS buttons for diagnostics."""
    rule, button, offset = Clicker.load_rule(context, rule)
    visual = bool(button.match(frame, offset=offset, similarity=rule["threshold"]))
    score = float(button.last_similarity)
    # All five event rules use exactly one scale. Variant order is asset order.
    assets = [rule["asset"], *rule.get("asset_variants", [])]
    assert rule["scales"] == [1.0] and len(button.variants) == len(assets)
    height, width = frame.shape[:2]

    def guard_offset(candidate, companion):
        _, dx, dy, gw, gh, tolerance = companion
        x, y = candidate.button[:2]
        return (max(0, x + dx - tolerance), max(0, y + dy - tolerance),
                min(width - gw, x + dx + tolerance), min(height - gh, y + dy + tolerance))

    # Recover the selected variant using the actual short-circuit guard scores.
    # Diagnostic guard calls below happen only AFTER the production verdict.
    candidates = []
    for candidate, _, companions in button.variants:
        value = float(candidate.last_similarity)
        if math.isfinite(value) and value > rule["threshold"]:
            for companion in companions:
                left, top, right, bottom = guard_offset(candidate, companion)
                if left > right or top > bottom:
                    value = -1.0
                    break
                guard_score = float(companion[0].last_similarity)
                value = min(value, guard_score) if math.isfinite(guard_score) else -1.0
                if value <= rule["threshold"]:
                    break
        candidates.append(value if math.isfinite(value) else -1.0)
    best = max(range(len(candidates)), key=candidates.__getitem__)
    candidate, _, companions = button.variants[best]
    assert tuple(candidate.button) == button.matched_box
    guards = []
    for area, companion in zip(rule.get("required_source_areas", []), companions):
        bounds = guard_offset(candidate, companion)
        guard = companion[0]
        value = -1.0
        if bounds[0] <= bounds[2] and bounds[1] <= bounds[3]:
            guard.match(frame, offset=bounds, similarity=rule["threshold"])
            value = float(guard.last_similarity)
        guards.append({"area": area, "score": value})
    color = None
    if rule.get("color_condition"):
        condition = rule["color_condition"]
        x1, y1, x2, y2 = button.source_to_screen(condition["source_area"])
        fraction = 0.0
        if 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height:
            mask = hsv_mask(frame[y1:y2, x1:x2], condition["hsv"])
            fraction = float((mask[:, :, 0] > 0).mean())
        color = {"fraction": fraction, "minimum": condition["minimum_fraction"]}
    absent = None
    number = None
    matched = visual
    if matched and rule.get("absent_rule"):
        other = next(r for r in context.config["rules"] if r["id"] == rule["absent_rule"])
        _, guard, guard_search = Clicker.load_rule(context, other)
        guard.match(frame, offset=guard_search, similarity=other["threshold"])
        # Same absence criterion as drift.core.matcher.scan_rules.
        absent = {"score": float(guard.last_similarity), "threshold": other["threshold"]}
        matched = guard.last_similarity <= other["threshold"]
    if matched and rule.get("number_condition"):
        matched, _ = Clicker.number_condition(context, rule, button, frame)
        number = dict(context.state["numbers"][rule["id"]])
    return {"rule": rule["id"], "asset": assets[best], "threshold": rule["threshold"],
            "template": float(candidate.last_similarity), "guards": guards, "color": color,
            "score": score, "visual": visual, "absent": absent, "number": number,
            "matched": bool(matched)}


def replay(config, live_dir=None, legacy_dir=None, legacy_templates_only=False):
    config = copy.deepcopy(config)
    if legacy_templates_only:
        for rule in config["rules"]:
            if rule["id"].startswith(PREFIX) and rule["id"] != PREFIX + "open":
                rule["asset_variants"] = [p for p in rule["asset_variants"]
                                          if not p.endswith("-ghost.png")]
    rules = {rule["id"]: rule for rule in config["rules"] if rule["id"].startswith(PREFIX)}
    assert set(rules) == {PREFIX + step for step in STEPS}
    context = SimpleNamespace(root=ROOT, config=config, digit_reader=None,
                              lock=threading.RLock(), state={"numbers": {}, "frames": 1})
    legacy_dir = legacy_dir or ROOT / "assets" / "event-potion"
    results = []
    for season, suffix in (("ghost", "-ghost"), ("moon", ""), ("candy", "-candy")):
        for index, kind in enumerate(EXPECTED, 1):
            if season == "ghost":
                path = (live_dir / f"e{index}.png" if live_dir else
                        ROOT / "assets" / "event-potion" / f"{kind}{suffix}.png")
            else:
                path = legacy_dir / f"{kind}{suffix}.png"
            with Image.open(path) as source:
                frame = np.asarray(source.convert("RGB"))
            assert frame.shape == (1280, 720, 3), str(path)
            for step in STEPS:
                result = inspect_rule(context, rules[PREFIX + step], frame)
                result.update(season=season, frame=path.name, path=str(path),
                              expected=step in EXPECTED[kind])
                results.append(result)
    return results


def markdown(results):
    lines = ["| Season/frame | Rule | Asset | Template | Guards (config order) | Color/min | Absent | OCR | Match/expected |",
             "|---|---|---|---:|---|---|---|---|---|"]
    for row in results:
        color = row["color"]
        number = row["number"]
        absent = row["absent"]
        lines.append("| " + " | ".join([
            row["season"] + "/" + row["frame"], row["rule"].removeprefix(PREFIX),
            Path(row["asset"]).name, f'{row["template"]:.4f}',
            ", ".join(f'{g["score"]:.4f}' for g in row["guards"]) or "-",
            f'{color["fraction"]:.4f}/{color["minimum"]:.2f}' if color else "-",
            f'{absent["score"]:.4f}<={absent["threshold"]:.2f}' if absent else "-",
            f'{number["value"]}@{number["confidence"]:.4f}' if number else "-",
            f'{row["matched"]}/{row["expected"]}',
        ]) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-dir", type=Path, help="Directory containing e1/e2/e3.png")
    parser.add_argument("--legacy-dir", type=Path, help="Directory containing the six old assets")
    parser.add_argument("--legacy-templates-only", action="store_true",
                        help="Remove ghost variants from every rule except the seasonal entry")
    parser.add_argument("--allow-mismatches", action="store_true", help="For pre-fix diagnosis")
    parser.add_argument("--json", action="store_true", help="Emit full precision diagnostics")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    results = replay(config, args.live_dir, args.legacy_dir, args.legacy_templates_only)
    mismatches = [r for r in results if r["matched"] != r["expected"]]
    if args.json:
        print(json.dumps({"results": results, "mismatches": len(mismatches)}, ensure_ascii=False))
    else:
        print(markdown(results))
        print(f"\n{len(results)} comparisons; {len(mismatches)} mismatches.")
    return int(bool(mismatches) and not args.allow_mismatches)


if __name__ == "__main__":
    raise SystemExit(main())
