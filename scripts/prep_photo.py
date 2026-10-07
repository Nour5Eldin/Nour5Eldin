"""Prep a portrait photo for ASCII conversion. Run once per photo.

    python scripts/prep_photo.py my-photo.jpg
    python scripts/prep_photo.py my-photo.jpg --keep-bg    # skip background removal
    python scripts/prep_photo.py my-photo.jpg --face 2.4   # head-and-shoulders crop width (default 2.0; 0 = off)

What it does:
  1. Removes the background with rembg so only you are left.
  2. Boosts local contrast with OpenCV CLAHE so a flat-lit face gets real highlights/shadows.
  3. Composites onto pure white, so the background becomes blank (spaces) in the ASCII ramp.

Output: source-prepped.png in the repo root (grayscale).
Needs the portrait-only packages from scripts/requirements.txt (pillow, numpy, opencv-python, rembg).
Tip: a photo from the chest up, face lit from the side, works best.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "source-prepped.png"
FG_MAX = 240  # keep the subject below the "white = background" threshold used by make_ascii_svg.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--keep-bg", action="store_true", help="skip background removal (use a plain light background)")
    ap.add_argument("--clip", type=float, default=3.5, help="CLAHE clip limit (higher = more contrast)")
    ap.add_argument("--model", default="u2net_human_seg", help="rembg model (light + good for people)")
    ap.add_argument("--crop-height", type=float, default=0.0,
                    help="keep only the top fraction of the subject, e.g. 0.6 for a tighter head-and-shoulders crop")
    ap.add_argument("--face", type=float, default=2.0,
                    help="crop around the head: output width = head width x this number (smaller = closer, 0 = off)")
    args = ap.parse_args()

    img = Image.open(args.photo).convert("RGB")

    if args.keep_bg:
        alpha = np.ones((img.height, img.width), dtype=np.float32)
    else:
        from rembg import new_session, remove  # imported late: slow to load, downloads a model on first run

        # Segment on a downscaled copy (faster, far less RAM), then scale the mask back up
        small = img.copy()
        small.thumbnail((1024, 1024), Image.LANCZOS)
        cut = remove(small, session=new_session(args.model))  # RGBA
        mask = Image.fromarray(np.array(cut)[..., 3], "L").resize(img.size, Image.LANCZOS)
        alpha = np.array(mask).astype(np.float32) / 255.0

    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=args.clip, tileGridSize=(8, 8))
    g = clahe.apply(gray).astype(np.float32)
    if not args.keep_bg:
        g = np.minimum(g, FG_MAX)

    out = g * alpha + 255.0 * (1.0 - alpha)

    # Crop to the subject (plus a margin) so the portrait fills the ASCII grid
    # (skipped with --face: that mode does its own crop in full-image coordinates)
    if not args.keep_bg and args.face <= 0:
        ys, xs = np.where(alpha > 0.5)
        if len(xs) and args.crop_height > 0:
            keep_to = ys.min() + int((ys.max() - ys.min()) * args.crop_height)
            sel = ys <= keep_to
            xs, ys = xs[sel], ys[sel]
            out[keep_to + 1:, :] = 255.0
        if len(xs):
            mx = int((xs.max() - xs.min()) * 0.08)
            my = int((ys.max() - ys.min()) * 0.05)
            x0, x1 = max(0, xs.min() - mx), min(out.shape[1], xs.max() + mx)
            y0, y1 = max(0, ys.min() - my), min(out.shape[0], ys.max() + my)
            out = out[y0:y1, x0:x1]

    if args.face > 0 and not args.keep_bg:
        ys, xs = np.where(alpha > 0.5)
        top, bottom = ys.min(), ys.max()
        head = ys <= top + int((bottom - top) * 0.22)           # hair -> mouth level (stays above the shoulders)
        hx = xs[head]
        lo, hi = np.percentile(hx, 3), np.percentile(hx, 97)
        head_w, cx = hi - lo, int((lo + hi) / 2)
        w = int(head_w * args.face)
        h = int(w * 572 / 600)                                  # same aspect as the ASCII grid box
        x0, y0 = cx - w // 2, top - int(0.06 * h)
        pad = max(0, -x0, -y0, x0 + w - out.shape[1], y0 + h - out.shape[0])
        canvas = np.pad(out, pad, constant_values=255.0)        # white outside the photo
        out = canvas[y0 + pad:y0 + pad + h, x0 + pad:x0 + pad + w]

    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "L").save(OUT)
    print(f"wrote {OUT} ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
