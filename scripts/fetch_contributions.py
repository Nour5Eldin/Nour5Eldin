"""Fetch the public contribution calendar for a GitHub user (no token needed).

GitHub serves the same HTML fragment the profile page uses at
https://github.com/users/<username>/contributions

Usage:
    python scripts/fetch_contributions.py            # uses GITHUB_USER or the default below
    python scripts/fetch_contributions.py SomeUser

Writes data/contributions.json with the raw days plus derived stats.
"""
import json
import os
import re
import sys
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DEFAULT_USER = "Nour5Eldin"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "contributions.json"


def fetch_html(username: str) -> str:
    url = f"https://github.com/users/{username}/contributions"
    resp = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (profile-readme-bot)"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.text


def parse_days(html: str):
    soup = BeautifulSoup(html, "html.parser")

    # The counts live in <tool-tip for="<cell id>"> elements: "3 contributions on May 4th."
    counts_by_cell = {}
    for tip in soup.find_all("tool-tip"):
        cell_id = tip.get("for")
        text = tip.get_text(" ", strip=True)
        m = re.match(r"(\d[\d,]*)\s+contribution", text)
        counts_by_cell[cell_id] = int(m.group(1).replace(",", "")) if m else 0

    days = []
    for cell in soup.find_all("td", attrs={"data-date": True}):
        days.append(
            {
                "date": cell["data-date"],
                "count": counts_by_cell.get(cell.get("id"), 0),
                "level": int(cell.get("data-level", 0)),
            }
        )
    days.sort(key=lambda d: d["date"])
    return days


def compute_stats(days):
    today = date.today()
    # Only look at days up to today (the calendar can include empty future cells)
    days = [d for d in days if datetime.strptime(d["date"], "%Y-%m-%d").date() <= today]
    total = sum(d["count"] for d in days)

    # Longest streak
    longest = run = 0
    for d in days:
        run = run + 1 if d["count"] > 0 else 0
        longest = max(longest, run)

    # Current streak: walk back from today; an empty "today" doesn't break it yet
    by_date = {d["date"]: d["count"] for d in days}
    cursor = today
    if by_date.get(cursor.isoformat(), 0) == 0:
        cursor -= timedelta(days=1)
    current = 0
    while by_date.get(cursor.isoformat(), 0) > 0:
        current += 1
        cursor -= timedelta(days=1)

    best = max(days, key=lambda d: d["count"]) if days else {"date": None, "count": 0}

    monthly = OrderedDict()
    for d in days:
        key = d["date"][:7]
        monthly[key] = monthly.get(key, 0) + d["count"]

    return {
        "total": total,
        "current_streak": current,
        "longest_streak": longest,
        "best_day": {"date": best["date"], "count": best["count"]},
        "monthly": monthly,
    }


def main():
    username = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GITHUB_USER", DEFAULT_USER)
    html = fetch_html(username)
    days = parse_days(html)
    if not days:
        sys.exit("No contribution cells found - GitHub may have changed the markup.")

    payload = {
        "username": username,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stats": compute_stats(days),
        "days": days,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    s = payload["stats"]
    print(
        f"{username}: {s['total']} contributions, "
        f"current streak {s['current_streak']}, longest {s['longest_streak']} -> {OUT}"
    )


if __name__ == "__main__":
    main()
