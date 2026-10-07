"""Render data/contributions.json as an animated 53-week contribution heatmap SVG.

The boxes slide in diagonally once on load and then stay put (no looping).
Set STATIC=1 to emit a frozen frame (handy for local previews).

Usage:
    python scripts/render_heatmap_svg.py
"""
import json
import os
from bisect import bisect_left
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "contributions.json"
OUT = ROOT / "contrib-heatmap.svg"
STATIC = os.environ.get("STATIC") == "1"

# none -> brightest (level 5 is a neon top end)
PALETTE = ["#161b22", "#0e4429", "#006d32", "#26a641", "#39d353", "#69f0a0"]
BG = "#0d1117"
TEXT = "#8b949e"
TEXT_STRONG = "#c9d1d9"
FONT = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace"

CELL, GAP = 12, 3
STEP = CELL + GAP
LEFT, TOP = 46, 62          # room for weekday labels / title + month labels
W = 860                     # matches the README layout (370 + 490)
GRID_H = 7 * STEP - GAP
H = TOP + GRID_H + 70       # grid + footer + legend


def levels_for(days):
    """Map counts to 0..5 using quantiles of the non-zero days so the ramp is always used."""
    nonzero = sorted(d["count"] for d in days if d["count"] > 0)
    if not nonzero:
        return lambda c: 0
    cuts = [nonzero[min(len(nonzero) - 1, int(len(nonzero) * q))] for q in (0.0, 0.25, 0.5, 0.75, 0.92)]

    def level(count):
        if count <= 0:
            return 0
        # bisect_left on the cut list gives 1..5 for non-zero counts
        return max(1, min(5, bisect_left(cuts, count + 1)))

    return level


def main():
    payload = json.loads(DATA.read_text())
    days = payload["days"]
    stats = payload["stats"]
    level = levels_for(days)

    first = datetime.strptime(days[0]["date"], "%Y-%m-%d").date()
    # Calendar weeks start on Sunday
    start = first - timedelta(days=(first.weekday() + 1) % 7)

    cells, month_marks, max_week = [], {}, 0
    for d in days:
        dt = datetime.strptime(d["date"], "%Y-%m-%d").date()
        offset = (dt - start).days
        week, dow = offset // 7, offset % 7
        max_week = max(max_week, week)
        if dt.day <= 7 and dow == 0 or (week == 0 and dt.day == first.day):
            month_marks.setdefault(week, dt.strftime("%b"))
        lv = level(d["count"])
        x = LEFT + week * STEP
        y = TOP + dow * STEP
        delay = round((week + dow) * 0.018, 3)
        plural = "contribution" if d["count"] == 1 else "contributions"
        cells.append(
            f'<rect class="c" x="{x}" y="{y}" width="{CELL}" height="{CELL}" rx="3" '
            f'fill="{PALETTE[lv]}" style="animation-delay:{delay}s">'
            f'<title>{d["count"]} {plural} on {d["date"]}</title></rect>'
        )

    grid_right = LEFT + max_week * STEP + CELL

    # month labels (skip ones that would collide with the previous label)
    month_svg, last_x = [], -100
    for week in sorted(month_marks):
        x = LEFT + week * STEP
        if x - last_x >= 36:
            month_svg.append(f'<text x="{x}" y="{TOP - 10}" class="lbl">{month_marks[week]}</text>')
            last_x = x

    day_labels = "".join(
        f'<text x="{LEFT - 10}" y="{TOP + i * STEP + CELL - 2}" class="lbl" text-anchor="end">{name}</text>'
        for i, name in ((1, "Mon"), (3, "Wed"), (5, "Fri"))
    )

    # legend (bottom right)
    legend_y = TOP + GRID_H + 24          # footer text row
    leg_row = legend_y + 18               # legend + best-day row
    legend_x = grid_right - (6 * STEP + 70)
    legend = [f'<text x="{legend_x}" y="{leg_row + 10}" class="lbl" text-anchor="end">Less</text>']
    for i, color in enumerate(PALETTE):
        legend.append(
            f'<rect x="{legend_x + 8 + i * STEP}" y="{leg_row}" width="{CELL}" height="{CELL}" rx="3" fill="{color}"/>'
        )
    legend.append(f'<text x="{legend_x + 16 + 6 * STEP}" y="{leg_row + 10}" class="lbl">More</text>')

    best = stats["best_day"]
    footer = (
        f'{stats["total"]:,} contributions in the last year  |  '
        f'current streak {stats["current_streak"]}d  |  longest {stats["longest_streak"]}d'
    )
    best_line = f'best day: {best["count"]} on {best["date"]}' if best["date"] else ""

    anim_css = (
        ""
        if STATIC
        else """
  .c { opacity: 0; transform: translateY(-8px); animation: drop .55s cubic-bezier(.2,.8,.2,1) both; }
  @keyframes drop { to { opacity: 1; transform: translateY(0); } }
  @media (prefers-reduced-motion: reduce) { .c { animation: none; opacity: 1; transform: none; } }"""
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="GitHub contribution heatmap for {payload['username']}">
<style>
  text {{ font-family: {FONT}; }}
  .lbl {{ font-size: 10px; fill: {TEXT}; }}
  .ttl {{ font-size: 13px; fill: {TEXT_STRONG}; }}
  .ftr {{ font-size: 12px; fill: {TEXT_STRONG}; }}{anim_css}
</style>
<rect width="{W}" height="{H}" rx="12" fill="{BG}" stroke="#30363d"/>
<text x="{LEFT}" y="26" class="ttl">$ git log --since="1 year ago" --graph</text>
{''.join(month_svg)}
{day_labels}
{''.join(cells)}
<text x="{LEFT}" y="{legend_y + 10}" class="ftr">{footer}</text>
<text x="{LEFT}" y="{leg_row + 10}" class="lbl">{best_line}</text>
{''.join(legend)}
</svg>
"""
    OUT.write_text(svg)
    print(f"wrote {OUT} ({len(svg) // 1024} KB, {len(cells)} cells)")


if __name__ == "__main__":
    main()
