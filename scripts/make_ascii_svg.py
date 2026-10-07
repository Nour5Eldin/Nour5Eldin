#!/usr/bin/env python3
"""
High-detail animated ASCII portrait -> SVG.

Keeps the look of the original file (same width, colours, glyph metrics and
row-by-row green scan reveal) but:
  * the canvas HEIGHT now follows the photo's aspect ratio (no squeezed/cropped
    portrait, no empty side bands),
  * the person mask comes from the real (white) background of the photo, so the
    whole figure - hand, ring, cuff, sleeves, gap between arm and body - is kept,
  * the character grid is much denser and every glyph is picked by *shape
    matching* (not just brightness), so glasses, eyelids, moustache, beard,
    lapels, piping, pocket and buttons follow the real contours,
  * dark areas (the suit) get a shadow lift + local-contrast boost so folds
    and seams survive instead of collapsing into a flat dark block.

usage: python3 make_ascii_svg.py portraits.jpg ascii-portrait.svg [--polarity light|ink]
       python3 make_ascii_svg.py portraits.jpg out.svg --k 2.5 --debug
"""
import sys, html, math, argparse
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------------ settings
DISPLAY_W = 621.6                      # on-screen width: identical to the original file
K = 3.0                                # grid density vs original (2 = 4x more characters, 3 = 9x)
FONT_SIZE = 6.0                        # glyph metrics identical to the original (6px / 3.6 / 6.6)
ADV = 0.6 * FONT_SIZE                  # char advance
LINE = 1.1 * FONT_SIZE                 # row height
SS_X, SS_Y = 6, 11                     # supersampling per cell for matching
BG, FG, CURSOR = "#0d1117", "#c9d1d9", "#39d353"
# --- animation ---------------------------------------------------------------
SCAN_TOTAL = 1.7                       # seconds for the whole top->bottom sweep (was 3.7)
ROW_DUR = 0.9                          # seconds for one row's spring (wipe + overshoot + settle)
T0 = 0.3                               # delay before the first row starts
COMET_W = 16 * ADV                     # length of the glowing trail behind the green head
OVERSHOOT = 5 * ADV                    # how far the head springs past the end of a row
TINT = "#7ee787"                       # green wash on freshly revealed glyphs ...
TINT_A = 0.20                          # ... starting at this opacity, fading in 4 steps
TINT_DUR = 0.7                         # ... over this many seconds
E_OUT = "0.25 0.50 0.35 1"             # quick launch, soft landing (spring: rise)
E_IO = "0.45 0 0.30 1"                 # smooth turn-around         (spring: settle)
BG_THR = 250                           # a pixel is background when R,G,B are all >= this
HOLE_MIN = 0.0004                      # enclosed white areas bigger than this (x image area) are background too

# glyph palette (ordered only for readability; matching is data-driven)
TONE  = " .`:-=+*cs#%@"          # same family as the original file
EDGE  = "|/\\_()~^'"             # only allowed on real edges (glasses, lids, lips...)
CHARS = TONE + EDGE
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def canvas_size():
    """viewBox size without the height (the height depends on the photo)."""
    return DISPLAY_W * K, 12.0 * K


def layout(img_w, img_h):
    """Grid + canvas that match the photo's aspect ratio exactly."""
    cw, margin = canvas_size()
    cols = int(round((cw - 2 * margin) / ADV))
    rows = int(round(img_h / img_w * cols * ADV / LINE))
    ch = 2 * margin + rows * LINE
    return dict(cols=cols, rows=rows, W=cw, H=ch, margin=margin)


def glyph_bitmaps():
    big = 4
    font = ImageFont.truetype(FONT, int(round(SS_X / 0.602)) * big)
    out = []
    for ch in CHARS:
        im = Image.new("L", (SS_X * big, SS_Y * big), 0)
        ImageDraw.Draw(im).text((0, int(SS_Y * 0.818 * big)), ch, font=font,
                                fill=255, anchor="ls")
        a = np.asarray(im.resize((SS_X, SS_Y), Image.LANCZOS), np.float32) / 255
        out.append(a)
    return np.stack(out)                       # (G, SS_Y, SS_X)


def find_face(img_bgr):
    """Haar face box (x, y, w, h) in full-res pixels; falls back to a proportional guess."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    s = 900 / w
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    fc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = fc.detectMultiScale(small, 1.1, 6, minSize=(120, 120))
    if len(faces):
        fx, fy, fw, fh = max(faces, key=lambda r: r[2]) / s
        return float(fx), float(fy), float(fw), float(fh)
    print("warning: no face detected, using proportional fallback", file=sys.stderr)
    return 0.37 * w, 0.12 * h, 0.30 * w, 0.24 * h


def build_mask(img_bgr):
    """Person mask from the photo's own white background (full resolution, aspect kept).

    Background = white area connected to the border + any big enclosed white hole
    (e.g. the gap between the arm and the body). Small bright patches (shirt
    highlights) are NOT background, so the white shirt stays in the figure.
    """
    h, w = img_bgr.shape[:2]
    white = (img_bgr.min(axis=2) >= BG_THR).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, cw, chh, a = st[i]
        touches = x == 0 or y == 0 or x + cw == w or y + chh == h
        keep[i] = touches or a > HOLE_MIN * w * h
    bg = keep[lab]
    fg = (~bg).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats((fg > 127).astype(np.uint8))
    if n > 2:                                   # keep the person, drop stray specks
        fg = ((lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])) * 255).astype(np.uint8)
    fg = cv2.GaussianBlur(fg, (0, 0), 1.2)
    return fg, find_face(img_bgr)


def tone_map(img_bgr, mask, face, P):
    """returns float32 image 0..1 (brightness) at supersampled resolution."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = cv2.bilateralFilter(lab[..., 0], 9, P["denoise"], 9)
    m = mask > 127
    lo, hi = np.percentile(L[m], [P["lo"], P["hi"]])
    Ls = np.clip((L.astype(np.float32) - lo) / (hi - lo), 0, 1)
    Ls = np.power(Ls, P["lift"])               # shadow lift: opens up the dark suit

    H, W = Ls.shape
    fx, fy, fw, fh = face
    cx, cy = fx + fw * 0.5, fy + fh * 0.55
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = ((xx - cx) / (fw * 0.95)) ** 2 + ((yy - cy) / (fh * 1.18)) ** 2
    fmask = cv2.GaussianBlur(np.clip(1.25 - d, 0, 1), (0, 0), fw * 0.05)
    face_w = fmask.copy()                      # face only (used to learn the skin range)

    # the hand is skin too: give it the same treatment as the face so knuckles,
    # fingers and nails keep their shading instead of becoming a flat grey patch
    ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    skin = ((ycc[..., 1] >= P["cr_lo"]) & (ycc[..., 1] <= 185) &
            (ycc[..., 2] >= 75) & (ycc[..., 2] <= P["cb_hi"]) & m).astype(np.uint8)
    skin[: int(H * 0.55)] = 0                  # lower body only: the face has its own ellipse
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(skin)
    big = np.zeros(n, np.uint8)
    big[1:] = st[1:, cv2.CC_STAT_AREA] > 0.002 * H * W
    skin = big[lab].astype(np.float32)
    smask = cv2.GaussianBlur(skin, (0, 0), W * 0.004)
    fmask = np.maximum(fmask, smask)

    # face-aware normalisation: stretch the *skin* range over the whole ramp
    fm_in = face_w > 0.5
    if fm_in.sum() > 1000:
        flo, fhi = np.percentile(L[fm_in], [P["flo"], P["fhi"]])
        Lf_ = np.clip((L.astype(np.float32) - flo) / (fhi - flo), 0, 1)
        Ls = Ls * (1 - fmask * P["fnorm"]) + Lf_ * (fmask * P["fnorm"])
    # face tone curve: >1 pulls skin highlights out of the top of the ramp so
    # forehead / cheeks / nose keep their shading instead of clipping to '@'
    Ls = np.power(np.clip(Ls, 0, 1), 1.0 + (P["fgamma"] - 1.0) * fmask)

    # local contrast: gentle on the face, stronger on the body
    cl = cv2.createCLAHE(clipLimit=P["clip"], tileGridSize=(P["tiles"], P["tiles"]))
    Lc = cl.apply((Ls * 255).astype(np.uint8)).astype(np.float32) / 255
    mix = P["clahe"] * (1 - fmask) + P["fclahe"] * fmask
    T = Ls * (1 - mix) + Lc * mix

    # body detail: medium-scale unsharp so lapels / piping / pocket / folds read
    big = cv2.GaussianBlur(T, (0, 0), W * P["bsig"])
    T = T + P["bdetail"] * (1 - fmask) * (T - big)

    # unsharp mask at feature scale (eyes / lids / rims / lips), face-weighted
    sig = W * P["sig"]
    blur = cv2.GaussianBlur(T, (0, 0), sig)
    T = T + (P["sharp"] + P["fsharp"] * fmask) * (T - blur)
    return np.clip(T, 0, 1).astype(np.float32)


DEFAULTS = dict(polarity="light", gamma=0.78, floor=0.10, rim=0.30, beta=3.0,
                edge_thr=0.50, diffuse=0.6, denoise=22, lo=1.5, hi=99.3,
                lift=0.62, clip=3.0, tiles=8, clahe=0.55, fclahe=0.50,
                bsig=0.012, bdetail=0.9,
                sig=0.004, sharp=0.3, fsharp=1.7, fgamma=1.30, use_edge=True,
                flo=2.0, fhi=99.0, fnorm=1.0, cr_lo=137, cb_hi=130)
_CACHE = {}


def generate(src, P):
    """image path + params -> (list of text rows, layout)"""
    if src not in _CACHE:
        img = cv2.imread(src)
        if img is None:
            sys.exit(f"cannot read image: {src}")
        _CACHE[src] = (img,) + build_mask(img)
    img, mask_full, face = _CACHE[src]

    lay = layout(img.shape[1], img.shape[0])
    cols, rows = lay["cols"], lay["rows"]
    tw, th = cols * SS_X, rows * SS_Y

    imgs = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    masks = cv2.resize(mask_full, (tw, th), interpolation=cv2.INTER_AREA)
    sx, sy = tw / img.shape[1], th / img.shape[0]
    face_s = (face[0] * sx, face[1] * sy, face[2] * sx, face[3] * sy)
    T = tone_map(imgs, masks, face_s, P)
    M = masks.astype(np.float32) / 255

    ink = (1 - T) if P["polarity"] == "ink" else T
    ink = np.power(np.clip(ink, 0, 1), P["gamma"])

    Mc = cv2.resize(M, (cols, rows), interpolation=cv2.INTER_AREA) > 0.5
    inner = cv2.erode(Mc.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    rim = Mc & ~inner
    rim_px = cv2.resize(rim.astype(np.float32), (tw, th), interpolation=cv2.INTER_NEAREST)
    M_px = cv2.resize(Mc.astype(np.float32), (tw, th), interpolation=cv2.INTER_NEAREST)
    ink = P["floor"] + (1 - P["floor"]) * ink
    ink = np.maximum(ink, rim_px * P["rim"]) * M_px

    # glyph tables
    Gb = glyph_bitmaps()
    n = len(CHARS)
    cov = Gb.reshape(n, -1)
    gain = 1.0 / cov.mean(1)[CHARS.index("@")] * 0.92
    Gm = cov * gain
    Gmean = Gm.mean(1)
    is_edge = np.array([ch in EDGE for ch in CHARS])

    Tb = cv2.GaussianBlur(T, (0, 0), 1.1)
    gm = np.hypot(cv2.Sobel(Tb, cv2.CV_32F, 1, 0), cv2.Sobel(Tb, cv2.CV_32F, 0, 1))
    E = cv2.resize(gm, (cols, rows), interpolation=cv2.INTER_AREA)
    E = E / (np.percentile(E[Mc], 93) + 1e-6)

    Pc = ink.reshape(rows, SS_Y, cols, SS_X).transpose(0, 2, 1, 3).reshape(rows, cols, -1)
    BETA = P["beta"] * SS_X * SS_Y
    err = np.zeros((rows + 1, cols + 2), np.float32)
    out = []
    for r in range(rows):
        line = []
        for c in range(cols):
            if not Mc[r, c]:
                line.append(" "); continue
            p = Pc[r, c] + err[r, c + 1]
            pm = p.mean()
            d_tone = (pm - Gmean) ** 2
            if P["use_edge"] and E[r, c] > P["edge_thr"]:
                d = ((p[None, :] - Gm) ** 2).sum(1) + BETA * d_tone
            else:
                d = d_tone + np.where(is_edge, 1e3, 0.0)
            j = int(np.argmin(d))
            line.append(CHARS[j])
            e = float(np.clip((pm - Gmean[j]) * P["diffuse"], -0.12, 0.12))
            err[r, c + 2] += e * 7 / 16
            err[r + 1, c] += e * 3 / 16
            err[r + 1, c + 1] += e * 5 / 16
            err[r + 1, c + 2] += e * 1 / 16
        out.append("".join(line))
    return out, lay


def main():
    global K
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="photo (.jpg/.png) or a .txt grid written by an earlier run "
                                "(skips image processing - handy for tweaking the animation)")
    ap.add_argument("dst")
    ap.add_argument("--k", type=float, default=K, help="grid density (2 = original x4 chars, 3 = x9)")
    ap.add_argument("--debug", action="store_true", help="also dump the mask next to the svg")
    ap.add_argument("--scan", type=float, default=SCAN_TOTAL, help="seconds for the top->bottom sweep")
    ap.add_argument("--no-tint", action="store_true", help="no green wash on freshly revealed glyphs")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            continue
        ap.add_argument("--" + k.replace("_", "-"), dest=k,
                        type=type(v), default=v)
    a = ap.parse_args()
    anim = dict(scan=a.scan, tint=not a.no_tint)

    if a.src.lower().endswith(".txt"):            # re-time the animation, keep the glyph grid
        rows_txt = open(a.src, encoding="utf-8").read().split("\n")
        K = round(len(rows_txt[0]) * ADV / (DISPLAY_W - 24.0), 3)   # density is implied by the grid
        margin = 12.0 * K
        lay = dict(cols=len(rows_txt[0]), rows=len(rows_txt), W=DISPLAY_W * K,
                   H=2 * margin + len(rows_txt) * LINE, margin=margin)
        write_svg(rows_txt, a.dst, lay, **anim)
        print("grid", len(rows_txt[0]), "x", len(rows_txt), "(from txt) ->", a.dst)
        return

    K = a.k
    P = dict(DEFAULTS); P.update({k: getattr(a, k) for k in DEFAULTS if hasattr(a, k)})
    rows_txt, lay = generate(a.src, P)
    if a.debug:
        _, mask_full, _ = _CACHE[a.src]
        cv2.imwrite(a.dst.replace(".svg", "_mask.png"), mask_full)
    write_svg(rows_txt, a.dst, lay, **anim)
    open(a.dst.replace(".svg", ".txt"), "w").write("\n".join(rows_txt))
    print("grid", len(rows_txt[0]), "x", len(rows_txt),
          "| display", DISPLAY_W, "x", round(lay["H"] / K, 1), "->", a.dst)


def sweep(p):
    """row start-time curve: eases in, rushes through the middle, eases out."""
    return 0.4 * p + 0.6 * (0.5 - 0.5 * math.cos(math.pi * p))


def write_svg(rows_txt, dst, lay, scan=None, tint=True):
    """Row-by-row reveal, built so it stays smooth:

    The glyph rows are STATIC text (painted once). Everything that moves is a cheap
    solid/gradient rectangle, which is what keeps the frame rate up even though ~190
    rows are in flight at once:

      * a background-coloured cover slides off each row (spring-timed: fast launch,
        soft landing) - it uncovers only that row's own text, so the front follows the
        silhouette of the figure,
      * a green gradient comet rides the cover's edge, springs a few characters past
        the end of the row and settles back (underdamped spring drawn with three
        cubic-bezier segments - SMIL has no real spring primitive); with ~90 rows
        overlapping, the comets merge into one glowing diagonal ribbon,
      * a green wash sits UNDER the cover, so only revealed glyphs show it, and fades
        out in 4 discrete steps (discrete steps repaint only 4 times - a continuous
        colour/opacity animation repaints every frame and halves the frame rate),
      * row start times follow an ease-in / rush / ease-out curve down the figure.
    """
    scan = SCAN_TOTAL if scan is None else scan
    out = []
    W, H, MARGIN_X, MARGIN_Y = lay["W"], lay["H"], lay["margin"], lay["margin"]
    display_h = round(H / K, 1)
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{DISPLAY_W}" height="{display_h}" '
               f'viewBox="0 0 {W:g} {H:g}" role="img" aria-label="ASCII portrait">')
    out.append("<style>\n  text { font-family: ui-monospace, SFMono-Regular, Menlo, "
               "Consolas, 'DejaVu Sans Mono', monospace; "
               f"font-size: {FONT_SIZE:g}px; fill: {FG}; white-space: pre; }}\n</style>")
    live = [(i, t) for i, t in enumerate(rows_txt) if t.strip()]
    first, last = live[0][0], live[-1][0]
    span = max(1, last - first)
    ov, cw = OVERSHOOT, COMET_W
    KT4, KT3 = "0;.5;.74;1", "0;.5;1"            # spring: rise -> overshoot -> undershoot -> rest
    SP4 = f"{E_OUT};{E_IO};{E_IO}"
    SP_COVER = f"{E_OUT};0 0 1 1"
    n_t = 5                                       # tint fade steps
    tint_vals = ";".join(f"{TINT_A * (1 - k / (n_t - 1)) ** 1.3:.3f}" for k in range(n_t))
    tint_kts = ";".join(f"{k / (n_t - 1):.2f}" for k in range(n_t))
    defs = (f'<linearGradient id="cm" x1="0" y1="0" x2="1" y2="0">'
            f'<stop offset="0" stop-color="{CURSOR}" stop-opacity="0"/>'
            f'<stop offset=".6" stop-color="{CURSOR}" stop-opacity=".25"/>'
            f'<stop offset=".92" stop-color="{CURSOR}" stop-opacity=".85"/>'
            f'<stop offset="1" stop-color="#b9ffcb"/></linearGradient>')
    body = []
    for i, t in live:
        top = MARGIN_Y + i * LINE
        begin = T0 + scan * sweep((i - first) / span)
        bt = f'begin="{begin:.3f}s" dur="{ROW_DUR}s" fill="freeze"'
        lead = len(t) - len(t.lstrip(" "))
        s = t.strip(" ")
        x0 = MARGIN_X + lead * ADV
        L = len(s) * ADV
        x1 = x0 + L
        R = x1 + ov
        row = [f'<text visibility="hidden" x="{x0:.2f}" y="{top + LINE * 0.818:.2f}" textLength="{L:.2f}" '
               f'lengthAdjust="spacing" xml:space="preserve">{html.escape(s, quote=False)}'
               f'<set attributeName="visibility" to="visible" begin="{begin:.3f}s"/></text>']
        if tint:
            row.append(
                f'<rect x="{x0:.1f}" y="{top:.2f}" width="{L:.1f}" height="{LINE:.2f}" '
                f'fill="{TINT}" fill-opacity="0"><animate attributeName="fill-opacity" '
                f'values="{tint_vals}" keyTimes="{tint_kts}" calcMode="discrete" '
                f'begin="{begin:.3f}s" dur="{TINT_DUR}s" fill="freeze"/></rect>')
        row.append(
            f'<rect x="{x0 - 1:.1f}" y="{top - 0.8:.2f}" width="{R - x0 + 1:.1f}" height="{LINE + 1.6:.2f}" fill="{BG}">'
            f'<animate attributeName="x" values="{x0 - 1:.1f};{R:.1f};{R:.1f}" keyTimes="{KT3}" '
            f'calcMode="spline" keySplines="{SP_COVER}" {bt}/>'
            f'<animate attributeName="width" values="{R - x0 + 1:.1f};0;0" keyTimes="{KT3}" '
            f'calcMode="spline" keySplines="{SP_COVER}" {bt}/></rect>')
        row.append(
            f'<rect x="{x0 - cw:.1f}" y="{top:.2f}" width="{cw:.1f}" height="{LINE:.2f}" '
            f'fill="url(#cm)" visibility="hidden">'
            f'<animate attributeName="x" values="{x0 - cw:.1f};{x1 + ov - cw:.1f};{x1 - ov * 0.3 - cw:.1f};{x1 - cw:.1f}" '
            f'keyTimes="{KT4}" calcMode="spline" keySplines="{SP4}" {bt}/>'
            f'<set attributeName="visibility" to="visible" begin="{begin:.3f}s"/>'
            f'<set attributeName="visibility" to="hidden" begin="{begin + ROW_DUR * 0.8:.3f}s"/></rect>')
        body.append("".join(row))
    out.append("<defs>" + defs + "</defs>")
    out.append(f'<rect width="{W:g}" height="{H:g}" rx="{12 * K:g}" fill="{BG}" stroke="#30363d" stroke-width="{K:g}"/>')
    out.append("".join(body))
    out.append("</svg>\n")
    open(dst, "w", encoding="utf-8").write("\n".join(out))


if __name__ == "__main__":
    main()