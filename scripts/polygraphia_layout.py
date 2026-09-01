"""Layout pass for the Polygraphia digitization (docs/polygraphia_digitization_scope.md §4.1).

Cuts each 1518 table-page image (two 24-row word columns per page) into two
column crops for transcription, and records the cut geometry.

Input:  DATA_ROOT/external/polygraphia/pages/NNNN.jpg  (Wellcome b33552137)
Output: DATA_ROOT/external/polygraphia/columns/bK_colNNN.jpg
        DATA_ROOT/external/polygraphia/layout.json     (per-page geometry)

Column numbering follows SOURCES.md: Book I images 0033-0224
(col 2p-65 left, 2p-64 right), Book II images 0229-0382 (col 2p-457 left,
col 2p-456 right).
"""

import argparse
import json
import os
import pathlib

import numpy as np
from PIL import Image

DATA_ROOT = pathlib.Path(os.environ.get("DATA_ROOT", "/workspace/data"))
BASE = DATA_ROOT / "external" / "polygraphia"

BOOK1 = range(33, 225)
BOOK2 = range(229, 383)


def page_columns(page: int) -> tuple[str, int, int] | None:
    if page in BOOK1:
        return "b1", 2 * page - 65, 2 * page - 64
    if page in BOOK2:
        return "b2", 2 * page - 457, 2 * page - 456
    return None


def paper_bounds(g: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of the paper leaf (bright region) inside the black scan bed."""
    h, w = g.shape
    centre = g[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    paper = np.median(centre)
    bright = g > paper * 0.7
    col_frac = bright.mean(axis=0)
    row_frac = bright.mean(axis=1)
    xs = np.where(col_frac > 0.5)[0]
    ys = np.where(row_frac > 0.5)[0]
    return int(xs[0]), int(xs[-1]), int(ys[0]), int(ys[-1])


def ink_mask(im: Image.Image) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Binary ink mask inside the paper region, suppressing bleed-through.

    Real ink is much darker than the beige ground; bleed-through sits in
    between, so threshold well below the paper median. Everything outside
    the paper leaf (black scan bed, page edges) is masked off.
    """
    g = np.asarray(im.convert("L"), dtype=np.float32)
    px0, px1, py0, py1 = paper_bounds(g)
    paper = np.median(g[py0:py1, px0:px1])
    mask = g < paper * 0.62
    mask[:py0, :] = mask[py1:, :] = False
    mask[:, :px0] = mask[:, px1:] = False
    return mask, (px0, px1, py0, py1)


def segment_page(im: Image.Image) -> dict:
    """Find the two text columns; return crop boxes in full-res pixels."""
    mask, (px0, px1, py0, py1) = ink_mask(im)
    h, w = mask.shape
    # Work inside margins of the leaf to avoid binding shadow / fore-edge.
    pw, ph = px1 - px0, py1 - py0
    x0, x1 = px0 + int(pw * 0.03), px1 - int(pw * 0.02)
    y0, y1 = py0 + int(ph * 0.02), py1 - int(ph * 0.02)
    sub = mask[y0:y1, x0:x1]

    col_ink = sub.sum(axis=0).astype(np.float32)
    # Smooth with a wide box filter so inter-word gaps do not read as valleys.
    k = max(9, (x1 - x0) // 200) | 1
    col_s = np.convolve(col_ink, np.ones(k) / k, mode="same")
    on = col_s > col_s.max() * 0.04

    # The gutter between the two (key + word) column pairs is the widest
    # whitespace gap whose centre lies in the middle of the leaf.
    gaps, start = [], None
    for i, v in enumerate(on):
        if not v and start is None:
            start = i
        elif v and start is not None:
            gaps.append((start, i))
            start = None
    n = len(on)
    central = [g for g in gaps if 0.30 * n < (g[0] + g[1]) / 2 < 0.80 * n]
    if not central:
        raise ValueError("no central gutter gap found")
    ga, gb = max(central, key=lambda g: g[1] - g[0])
    cut = x0 + (ga + gb) // 2

    boxes = []
    pad = int(w * 0.012)
    overlap = int(w * 0.03)  # annotations printed in the gutter may cross the cut
    for lo, hi in ((x0, cut + overlap), (cut - overlap, x1)):
        half = mask[y0:y1, lo:hi]
        xs = np.where(half.sum(axis=0) > 2)[0]
        ys = np.where(half.sum(axis=1) > 2)[0]
        if len(xs) == 0 or len(ys) == 0:
            raise ValueError("empty half-page")
        gx0, gx1 = lo + int(xs[0]) - pad, lo + int(xs[-1]) + pad
        gy0, gy1 = y0 + int(ys[0]) - pad, y0 + int(ys[-1]) + pad
        boxes.append([max(gx0, 0), max(gy0, 0), min(gx1, w), min(gy1, h)])
    return {"boxes": boxes, "size": [w, h], "cut": int(cut)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default=None, help="e.g. 33-224; default: all available")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = BASE / "columns"
    out_dir.mkdir(exist_ok=True)
    layout_path = BASE / "layout.json"
    layout = json.loads(layout_path.read_text()) if layout_path.exists() else {}

    if args.pages:
        lo, hi = args.pages.split("-")
        wanted = range(int(lo), int(hi) + 1)
    else:
        wanted = list(BOOK1) + list(BOOK2)

    done = skipped = 0
    for page in wanted:
        info = page_columns(page)
        if info is None:
            continue
        book, cl, cr = info
        src = BASE / "pages" / f"{page:04d}.jpg"
        if not src.exists():
            skipped += 1
            continue
        outs = [
            out_dir / f"{book}_col{cl:03d}.jpg",
            out_dir / f"{book}_col{cr:03d}.jpg",
        ]
        if all(o.exists() for o in outs) and str(page) in layout and not args.force:
            continue
        im = Image.open(src)
        seg = segment_page(im)
        for box, out in zip(seg["boxes"], outs):
            im.crop(box).save(out, quality=92)
        layout[str(page)] = {"book": book, "cols": [cl, cr], **seg}
        done += 1

    layout_path.write_text(json.dumps(layout, indent=1, sort_keys=True))
    print(f"segmented {done} pages ({skipped} missing), layout -> {layout_path}")


if __name__ == "__main__":
    main()
