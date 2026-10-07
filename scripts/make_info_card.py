"""Hand-author a neofetch-style info card as an animated SVG.

Edit the CONFIG block below, then run:
    python scripts/make_info_card.py          # animated  -> info-card.svg
    STATIC=1 python scripts/make_info_card.py # frozen frame for local previews

Each row fades/slides in on a short stagger, once, then stays.
Keep every value under ~44 characters so it fits the card width.
"""
import os
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "info-card.svg"
STATIC = os.environ.get("STATIC") == "1"

# ----------------------------- CONFIG --------------------------------------
USER_HOST = "nour@github"
ROWS = [
    # (key, [value lines])  - extra lines in the list continue the same row
    ("Role",     ["Full Stack Developer"]),
    ("Location", ["Cairo, Egypt"]),
    ("Stack",    ["React · Next.js · Node · Express · MongoDB",
                  "Angular + NgRx · NestJS · TypeScript"]),
    ("Edu",      ["BSc CS (2023) · NTI MEAN Stack (2026)"]),
    ("Built",    ["FreshCart · Fashion Store · Nour Fitness",
                  "Tournament Platform · nourkit"]),
    ("Web",      ["noureldin-mahmoud.vercel.app"]),
    ("GitHub",   ["github.com/Nour5Eldin"]),
]
# ---------------------------------------------------------------------------

W = 490
H = 352  # matches the portrait (370px wide, 166x86 chars) so both columns line up
PAD = 24
BAR_H = 34
LINE_H = 21
KEY_X = PAD
VAL_X = PAD + 86
BG = "#0d1117"
BORDER = "#30363d"
KEY_COLORS = ["#39d353", "#58a6ff", "#d2a8ff", "#ffa657", "#ff7b72", "#79c0ff", "#69f0a0"]
VAL = "#c9d1d9"
DIM = "#8b949e"
FONT = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace"
SWATCHES = ["#ff7b72", "#ffa657", "#e3b341", "#39d353", "#58a6ff", "#d2a8ff", "#c9d1d9", "#484f58"]


def main():
    out = []
    y = BAR_H + 34
    i = 0  # animation index

    def anim(idx):
        return "" if STATIC else f' class="ln" style="animation-delay:{round(0.35 + idx * 0.22, 2)}s"'

    # header: user@host + rule
    out.append(f'<g{anim(i)}><text x="{KEY_X}" y="{y}" fill="#39d353" font-weight="bold">{escape(USER_HOST)}</text></g>')
    i += 1
    y += 10
    out.append(f'<g{anim(i)}><rect x="{KEY_X}" y="{y - 4}" width="{len(USER_HOST) * 7.8:.0f}" height="1.5" fill="{DIM}"/></g>')
    i += 1
    y += LINE_H

    for n, (key, lines) in enumerate(ROWS):
        color = KEY_COLORS[n % len(KEY_COLORS)]
        for j, line in enumerate(lines):
            key_txt = f'<text x="{KEY_X}" y="{y}" fill="{color}" font-weight="bold">{escape(key)}</text>' if j == 0 else ""
            out.append(
                f'<g{anim(i)}>{key_txt}<text x="{VAL_X}" y="{y}" fill="{VAL}">{escape(line)}</text></g>'
            )
            i += 1
            y += LINE_H

    # colour swatches, like neofetch's palette strip
    y += 6
    sw = []
    for k, c in enumerate(SWATCHES):
        sw.append(f'<rect x="{KEY_X + k * 22}" y="{y - 12}" width="18" height="14" rx="3" fill="{c}"/>')
    out.append(f'<g{anim(i)}>{"".join(sw)}</g>')

    anim_css = (
        ""
        if STATIC
        else """
  .ln { opacity: 0; transform: translateY(6px); animation: in .45s ease-out both; }
  @keyframes in { to { opacity: 1; transform: translateY(0); } }
  @media (prefers-reduced-motion: reduce) { .ln { animation: none; opacity: 1; transform: none; } }"""
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="About card">
<style>
  text {{ font-family: {FONT}; font-size: 13px; }}{anim_css}
</style>
<rect width="{W}" height="{H}" rx="12" fill="{BG}" stroke="{BORDER}"/>
<path d="M0 12a12 12 0 0 1 12-12h{W - 24}a12 12 0 0 1 12 12v{BAR_H - 12}H0z" fill="#161b22"/>
<circle cx="20" cy="{BAR_H / 2}" r="5.5" fill="#ff5f56"/>
<circle cx="40" cy="{BAR_H / 2}" r="5.5" fill="#ffbd2e"/>
<circle cx="60" cy="{BAR_H / 2}" r="5.5" fill="#27c93f"/>
<text x="{W / 2}" y="{BAR_H / 2 + 4}" text-anchor="middle" fill="{DIM}" style="font-size:12px">~ / whoami</text>
{''.join(out)}
</svg>
"""
    OUT.write_text(svg)
    print(f"wrote {OUT} (last text baseline y={y})")
    assert y < H - 10, "Card content overflows the card height - trim ROWS or raise H"


if __name__ == "__main__":
    main()
