# Restaurant map regression frames

These are byte-for-byte copies of the supplied real captures, loaded as RGB by
`tests.test_restaurant_map`. No emulator or OCR model is required by that module.
Missing fixtures fail the regression tests instead of skipping them.

| File | Expected result with the configured map spec |
| --- | --- |
| `s02.png` | Visible icon at top-left `(551, 271)`, score about `0.8880` |
| `s03.png` | Left-edge icon at top-left `(18, 271)`, score about `0.8751` |
| `s04.png` | Left-edge icon at top-left `(18, 271)`, score about `0.8773` |
| `negative-03989.png` | Reject; highest grayscale match is about `0.3989` |

`manifest.json` records original paths, SHA-256 hashes and measured scores for
both the original `[20, 170, 700, 1115]` and expanded `[0, 170, 700, 1115]` areas.
The negative frame is the supplied 2026-10-25 capture
`before-618de0cdd8484b0caaa90bb6a9590281.png`; the other three are from `mslow`.

The original ROI excludes the complete edge icons in `s03.png` and `s04.png`.
They are true positives when the ROI starts at zero, not false positives. The
32-frame corpus audit and red/green evidence are in
`diagnostics/2026-10-28/restaurant-map-fix/REPORT.md`.
