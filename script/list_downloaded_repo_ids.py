#!/usr/bin/env python3
"""List IDs for downloaded Tsadra / OpenPecha repos.

Scans ``Data/repos/*/P*.opf/meta.yml`` and reports:
  - pecha_id  (top-level ``id``, e.g. P000010)
  - source_id (``source_metadata.id``, e.g. SHE-SG-05)

By default prints one pecha_id per line. Use ``--format`` for CSV or both IDs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOS = ROOT / "Data" / "repos"
DEFAULT_LIST = ROOT / "Data" / "tsadra_repo_list.txt"


def find_meta(repo_dir: Path) -> Path | None:
    matches = sorted(repo_dir.glob("*.opf/meta.yml"))
    return matches[0] if matches else None


def extract_ids(meta_path: Path) -> tuple[str | None, str | None]:
    data = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None, None
    pecha_id = data.get("id")
    source = data.get("source_metadata") or {}
    source_id = source.get("id") if isinstance(source, dict) else None
    return (
        str(pecha_id) if pecha_id is not None else None,
        str(source_id) if source_id is not None else None,
    )


def iter_downloaded(repos_dir: Path) -> list[tuple[str, str | None, str | None]]:
    """Return (dir_name, pecha_id, source_id) sorted by directory name."""
    rows: list[tuple[str, str | None, str | None]] = []
    for repo_dir in sorted(p for p in repos_dir.iterdir() if p.is_dir()):
        meta = find_meta(repo_dir)
        if meta is None:
            rows.append((repo_dir.name, None, None))
            continue
        pecha_id, source_id = extract_ids(meta)
        rows.append((repo_dir.name, pecha_id, source_id))
    return rows


def read_repo_list(list_path: Path) -> set[str]:
    ids: set[str] = set()
    for line in list_path.read_text(encoding="utf-8").splitlines():
        repo_id = line.strip()
        if repo_id and not repo_id.startswith("#"):
            ids.add(repo_id)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repos-dir",
        type=Path,
        default=DEFAULT_REPOS,
        help=f"Downloaded repos directory (default: {DEFAULT_REPOS})",
    )
    parser.add_argument(
        "--format",
        choices=("pecha", "source", "both", "csv"),
        default="pecha",
        help="Output format (default: pecha = one pecha id per line)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write results to this file instead of stdout",
    )
    parser.add_argument(
        "--only-in-list",
        type=Path,
        nargs="?",
        const=DEFAULT_LIST,
        default=None,
        help=(
            "Only include repos that appear in the Tsadra list "
            f"(default path if flag is set with no value: {DEFAULT_LIST})"
        ),
    )
    args = parser.parse_args()

    if not args.repos_dir.is_dir():
        print(f"Repos directory not found: {args.repos_dir}", file=sys.stderr)
        return 1

    allow = read_repo_list(args.only_in_list) if args.only_in_list else None
    rows = iter_downloaded(args.repos_dir)
    if allow is not None:
        rows = [r for r in rows if r[0] in allow]

    lines: list[str] = []
    missing_meta = 0
    missing_source = 0

    if args.format == "csv":
        out = open(args.output, "w", encoding="utf-8", newline="") if args.output else sys.stdout
        try:
            writer = csv.writer(out)
            writer.writerow(["dir_name", "pecha_id", "source_id"])
            for dir_name, pecha_id, source_id in rows:
                if pecha_id is None:
                    missing_meta += 1
                if source_id is None:
                    missing_source += 1
                writer.writerow([dir_name, pecha_id or "", source_id or ""])
        finally:
            if args.output:
                out.close()
    else:
        for dir_name, pecha_id, source_id in rows:
            if pecha_id is None:
                missing_meta += 1
            if source_id is None:
                missing_source += 1
            if args.format == "pecha":
                lines.append(pecha_id or dir_name)
            elif args.format == "source":
                lines.append(source_id or "")
            else:  # both
                lines.append(f"{pecha_id or dir_name}\t{source_id or ''}")

        text = "\n".join(lines) + ("\n" if lines else "")
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)

    print(
        f"# {len(rows)} downloaded repo(s)"
        + (f"; {missing_meta} missing meta.yml id" if missing_meta else "")
        + (f"; {missing_source} missing source_metadata.id" if missing_source else ""),
        file=sys.stderr,
    )
    if args.output:
        print(f"# wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
