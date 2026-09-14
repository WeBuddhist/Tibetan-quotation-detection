#!/usr/bin/env python3
"""Discover OpenPecha-Data repos on GitHub (recent pushes, missing locally).

Uses the GitHub search API (works with or without ``gh auth login``; set
``GITHUB_TOKEN`` for higher rate limits). Writes one repo id per line.

Examples::

    # Repos pushed in the last 30 days, not yet under Data/repos/
    python script/discover_openpecha_repos.py --since-days 30 --missing

    # Same, save to a list for download_data.py
    python script/discover_openpecha_repos.py --since-days 30 --missing \\
        -o Data/recent_opf_repo_list.txt

    python script/download_data.py --list Data/recent_opf_repo_list.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

ORG = "OpenPecha-Data"
SEARCH_URL = "https://api.github.com/search/repositories"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOS = ROOT / "Data" / "repos"


def github_get(url: str, timeout: float = 60.0) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "quotation-detection/discover_openpecha_repos",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def search_repos(since_iso: str, page: int, per_page: int = 100) -> dict:
    query = quote(f"org:{ORG} pushed:>{since_iso}", safe="")
    url = (
        f"{SEARCH_URL}?q={query}&sort=updated&order=desc"
        f"&per_page={per_page}&page={page}"
    )
    return github_get(url)


def iter_repo_ids(since_iso: str) -> list[tuple[str, str]]:
    """Return (repo_id, pushed_at) sorted by most recently pushed."""
    rows: list[tuple[str, str]] = []
    page = 1
    while True:
        try:
            payload = search_repos(since_iso, page=page)
        except HTTPError as exc:
            if exc.code == 403:
                raise RuntimeError(
                    "GitHub rate limit hit. Run `gh auth login` or set GITHUB_TOKEN."
                ) from exc
            raise
        items = payload.get("items") or []
        if not items:
            break
        for item in items:
            name = item.get("name")
            pushed = item.get("pushed_at") or ""
            if isinstance(name, str) and name:
                rows.append((name, pushed))
        if len(items) < 100:
            break
        page += 1
        time.sleep(0.2)
    rows.sort(key=lambda row: row[1], reverse=True)
    return rows


def local_repo_ids(repos_dir: Path) -> set[str]:
    if not repos_dir.is_dir():
        return set()
    return {p.name for p in repos_dir.iterdir() if p.is_dir()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since-days",
        type=int,
        default=30,
        help="Include repos with a push in the last N days (default: 30)",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="ISO date (YYYY-MM-DD) instead of --since-days",
    )
    parser.add_argument(
        "--repos-dir",
        type=Path,
        default=DEFAULT_REPOS,
        help=f"Downloaded repos directory (default: {DEFAULT_REPOS})",
    )
    parser.add_argument(
        "--missing",
        action="store_true",
        help="Omit repo ids that already exist under --repos-dir",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write repo ids to this file (stdout if omitted)",
    )
    parser.add_argument(
        "--with-dates",
        action="store_true",
        help="Print repo_id<TAB>pushed_at (for inspection)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.since:
        since_dt = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=UTC)
    else:
        since_dt = datetime.now(tz=UTC) - timedelta(days=args.since_days)
    since_iso = since_dt.strftime("%Y-%m-%d")

    print(f"Searching org:{ORG} pushed:>{since_iso} …", file=sys.stderr)
    rows = iter_repo_ids(since_iso)
    print(f"GitHub returned {len(rows)} repo(s)", file=sys.stderr)

    if args.missing:
        have = local_repo_ids(args.repos_dir)
        rows = [row for row in rows if row[0] not in have]
        print(f"{len(rows)} not present under {args.repos_dir}", file=sys.stderr)

    if args.with_dates:
        lines = [f"{repo_id}\t{pushed}" for repo_id, pushed in rows]
    else:
        lines = [repo_id for repo_id, _ in rows]

    text = "\n".join(lines) + ("\n" if lines else "")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
