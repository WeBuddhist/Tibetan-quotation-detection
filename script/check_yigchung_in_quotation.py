#!/usr/bin/env python3
"""Check whether any Yigchung span sits inside a Quotation span.

Loads gold character-offset spans from local OpenPecha repos and reports
containment / overlap between the two layers. Defaults to the val split.

Inclusive OpenPecha offsets: a Yigchung span sits inside a Quotation span
when quote.start <= yig.start and quote.end >= yig.end (boundary-touch only
is not an overlap). Also reports the reverse nesting and partial overlaps.

Examples:
  python script/check_yigchung_in_quotation.py
  python script/check_yigchung_in_quotation.py --split train
  python script/check_yigchung_in_quotation.py --split all --out Data/audit/yigchung_in_quote.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layer_detection.audit_utils import load_book, overlap_relation, spans_overlap  # noqa: E402
from layer_detection.dataset_utils import (  # noqa: E402
    DEFAULT_REPOS_DIR,
    DEFAULT_SPLIT_CSV,
    book_spans,
    read_split_assignment,
)

QUOTE_LABEL = "QUOTATION"
YIG_LABEL = "YIGCHUNG"
PREVIEW_CHARS = 80

# overlap_relation(quote, yig) → human-readable bucket for this check.
RELATION_BUCKET = {
    "a_contains_b": "yigchung_inside_quotation",
    "exact": "exact_same_span",
    "b_contains_a": "quotation_inside_yigchung",
    "partial": "partial_overlap",
}


def preview(text: str, start: int, end: int) -> str:
    """Inclusive-span snippet, newlines flattened."""
    snippet = text[start : end + 1].replace("\n", "⏎")
    if len(snippet) <= PREVIEW_CHARS:
        return snippet
    return snippet[: PREVIEW_CHARS - 1] + "…"


def pair_hits(quotes: list[dict], yigs: list[dict]) -> list[dict]:
    """Every overlapping Quotation × Yigchung pair, with relation."""
    hits = []
    for yig in yigs:
        for quote in quotes:
            if not spans_overlap(quote["start"], quote["end"], yig["start"], yig["end"]):
                continue
            rel = overlap_relation(quote["start"], quote["end"], yig["start"], yig["end"])
            hits.append({"quote": quote, "yigchung": yig, "relation": rel})
    return hits


def load_split_rows(
    repos_dir: Path,
    split_csv: Path,
    split: str,
) -> list[dict]:
    assignment = read_split_assignment(split_csv)
    wanted = {
        book_id
        for book_id, assigned in assignment.items()
        if split == "all" or assigned == split
    }
    rows = []
    missing = []
    for book_id in sorted(wanted):
        book_dir = repos_dir / book_id
        if not book_dir.is_dir():
            missing.append(book_id)
            continue
        rec = load_book(book_dir)
        if not rec.base_path or not rec.base_path.is_file():
            missing.append(book_id)
            continue
        text = rec.base_path.read_text(encoding="utf-8")
        rows.append(
            {
                "book_id": book_id,
                "split": assignment[book_id],
                "text": text,
                "quotes": book_spans(rec, "Quotation"),
                "yigs": book_spans(rec, "Yigchung"),
            }
        )
    if missing:
        print(
            f"Warning: {len(missing)} assigned books missing on disk: {missing}",
            file=sys.stderr,
        )
    return rows


def format_hit(book_id: str, text: str, hit: dict) -> str:
    quote = hit["quote"]
    yig = hit["yigchung"]
    bucket = RELATION_BUCKET[hit["relation"]]
    q_len = quote["end"] - quote["start"] + 1
    y_len = yig["end"] - yig["start"] + 1
    return (
        f"  {book_id}  {bucket}\n"
        f"    quotation [{quote['start']}:{quote['end']}] len={q_len}  "
        f"{preview(text, quote['start'], quote['end'])}\n"
        f"    yigchung  [{yig['start']}:{yig['end']}] len={y_len}  "
        f"{preview(text, yig['start'], yig['end'])}"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "book_id",
        "split",
        "relation",
        "quote_start",
        "quote_end",
        "quote_len",
        "yig_start",
        "yig_end",
        "yig_len",
        "quote_preview",
        "yig_preview",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="val",
        choices=("train", "val", "test", "all"),
        help="Which split to scan (default: val).",
    )
    parser.add_argument("--repos", type=Path, default=DEFAULT_REPOS_DIR)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--out", type=Path, help="Optional CSV of every overlapping pair.")
    args = parser.parse_args()

    rows = load_split_rows(args.repos, args.split_csv, args.split)
    if not rows:
        print(f"No books found for split={args.split!r} under {args.repos}")
        return 1

    relation_counts: Counter[str] = Counter()
    book_stats = []
    csv_rows = []
    inside_hits = []
    other_hits = []

    for row in rows:
        hits = pair_hits(row["quotes"], row["yigs"])
        buckets = Counter(RELATION_BUCKET[h["relation"]] for h in hits)
        relation_counts.update(buckets)
        book_stats.append(
            {
                "book_id": row["book_id"],
                "n_quote": len(row["quotes"]),
                "n_yig": len(row["yigs"]),
                "n_overlap": len(hits),
                "inside": buckets["yigchung_inside_quotation"] + buckets["exact_same_span"],
            }
        )
        for hit in hits:
            quote = hit["quote"]
            yig = hit["yigchung"]
            bucket = RELATION_BUCKET[hit["relation"]]
            csv_rows.append(
                {
                    "book_id": row["book_id"],
                    "split": row["split"],
                    "relation": bucket,
                    "quote_start": quote["start"],
                    "quote_end": quote["end"],
                    "quote_len": quote["end"] - quote["start"] + 1,
                    "yig_start": yig["start"],
                    "yig_end": yig["end"],
                    "yig_len": yig["end"] - yig["start"] + 1,
                    "quote_preview": preview(row["text"], quote["start"], quote["end"]),
                    "yig_preview": preview(row["text"], yig["start"], yig["end"]),
                }
            )
            formatted = format_hit(row["book_id"], row["text"], hit)
            if bucket in {"yigchung_inside_quotation", "exact_same_span"}:
                inside_hits.append(formatted)
            else:
                other_hits.append(formatted)

    books_both = [b for b in book_stats if b["n_quote"] and b["n_yig"]]
    books_inside = [b for b in book_stats if b["inside"]]
    n_inside = relation_counts["yigchung_inside_quotation"] + relation_counts["exact_same_span"]

    print(f"split={args.split}  books={len(rows)}")
    print(
        f"books with both layers: {len(books_both)}  "
        f"quotation spans: {sum(b['n_quote'] for b in book_stats):,}  "
        f"yigchung spans: {sum(b['n_yig'] for b in book_stats):,}"
    )
    print()
    print(f"{'book':10s} {'quotes':>7s} {'yigchung':>9s} {'overlap':>8s} {'yig⊂quote':>10s}")
    print("-" * 48)
    for b in book_stats:
        if not (b["n_quote"] or b["n_yig"]):
            continue
        print(
            f"{b['book_id']:10s} {b['n_quote']:7,d} {b['n_yig']:9,d} "
            f"{b['n_overlap']:8,d} {b['inside']:10,d}"
        )

    print()
    print("overlap relations:")
    for key in (
        "yigchung_inside_quotation",
        "exact_same_span",
        "quotation_inside_yigchung",
        "partial_overlap",
    ):
        print(f"  {key:28s} {relation_counts[key]:4d}")

    print()
    if n_inside:
        print(f"YES — {n_inside} yigchung span(s) sit inside a quotation in {len(books_inside)} book(s):")
        print()
        print("\n\n".join(inside_hits))
    else:
        print("NO — no yigchung span sits fully inside a quotation on this split.")

    if other_hits:
        print()
        print(f"Other Quotation × Yigchung overlaps ({len(other_hits)}):")
        print()
        print("\n\n".join(other_hits))

    if args.out:
        write_csv(args.out, csv_rows)
        print(f"\nwrote {len(csv_rows)} pair(s) to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
