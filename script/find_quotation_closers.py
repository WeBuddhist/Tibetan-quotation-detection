#!/usr/bin/env python3
"""Find Tibetan "thus said" closers vs Quotation annotations.

Quotations in this corpus are often closed by quotative particles / phrases
("thus said"), not punctuation. This script scans base text for those closers
and checks whether a Quotation span covers or immediately precedes them.

Outputs two useful slices:
  1. closers that sit on / just after a gold Quotation span (annotated)
  2. closers with no nearby Quotation span (candidate annotation false negatives)

Also summarizes gold Quotation spans that lack a nearby closer, and flags
long unannotated stretches that still contain closers (positional gaps).

Examples:
  python script/find_quotation_closers.py
  python script/find_quotation_closers.py --books P000078 P000218
  python script/find_quotation_closers.py --profile strong --out Data/audit/closer_fn
  python script/find_quotation_closers.py --mask-source confirmed   # all training books
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layer_detection.audit_utils import (  # noqa: E402
    classify_span,
    find_books,
    in_scope_layers,
    load_book,
    parse_layer_spans,
)

DEFAULT_REPOS = ROOT / "Data" / "repos"
DEFAULT_OUT = ROOT / "Data" / "audit" / "quotation_closers"

# Named regex profiles. Stronger profiles are better FN hunters (fewer name-
# / "thus called" false alarms); broader profiles catch more gold closers.
CLOSER_PROFILES: dict[str, re.Pattern[str]] = {
    # FLAGGED_BOOKS §5 + the high-precision post-span forms seen in gold data.
    "strong": re.compile(
        r"(?:"
        r"(?:ཞེས|ཅེས)་གསུངས"
        r"|(?:ཞེས|ཅེས)་སོ"
        r"|(?:ཞེས|ཅེས)་དང"
        r"|གསུངས།"
        r")"
    ),
    # Adds common "thus …" continuations that often follow gold quotes.
    "default": re.compile(
        r"(?:"
        r"(?:ཞེས|ཅེས)་གསུངས"
        r"|(?:ཞེས|ཅེས)་སོ"
        r"|(?:ཞེས|ཅེས)་དང"
        r"|(?:ཞེས|ཅེས)་པ་(?:སྟེ|ལྟར|དང)"
        r"|གསུངས།"
        r")"
    ),
    # Broader: any ཞེས / ཅེས particle (noisy — many non-quotative uses).
    "broad": re.compile(r"(?:ཞེས|ཅེས)(?:་|།)"),
}

# How far after a span end a closer may start and still count as "annotated".
DEFAULT_AFTER_TOL = 40
# How far before a span start a closer may sit (rare; usually closers trail).
DEFAULT_BEFORE_TOL = 0


@dataclass(frozen=True)
class QuoteSpan:
    ann_id: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass
class CloserHit:
    book: str
    marker: str
    marker_start: int
    marker_end: int
    status: str  # annotated | candidate_fn
    relation: str  # inside | just_after | span_ends_in_marker | none
    dist_to_span: int | None
    span_ann_id: str
    span_start: int | None
    span_end: int | None
    context_before: str
    context_after: str
    quote_preview: str


@dataclass
class GoldSpanRow:
    book: str
    ann_id: str
    start: int
    end: int
    length: int
    has_closer: bool
    closer_marker: str
    closer_start: int | None
    closer_relation: str
    text_preview: str
    after_preview: str


def _clip(text: str, lo: int, hi: int) -> str:
    return text[max(0, lo) : min(len(text), hi)].replace("\n", "⏎")


def load_quotation_spans(rec, text: str) -> list[QuoteSpan]:
    path = rec.layer_files.get("Quotation")
    if path is None:
        return []
    out: list[QuoteSpan] = []
    for ann_id, start, end, _line in parse_layer_spans(path):
        _reasons, action, fixed_start, fixed_end, _note = classify_span(
            start, end, len(text)
        )
        if action == "keep":
            out.append(QuoteSpan(ann_id=ann_id, start=start, end=end))
        elif action == "fix" and fixed_start is not None and fixed_end is not None:
            out.append(QuoteSpan(ann_id=ann_id, start=fixed_start, end=fixed_end))
    return out


def classify_closer(
    marker_start: int,
    marker_end: int,
    spans: list[QuoteSpan],
    after_tol: int,
    before_tol: int,
) -> tuple[str, str, int | None, QuoteSpan | None]:
    """Return (status, relation, dist, span)."""
    best: tuple[str, int, QuoteSpan] | None = None
    for span in spans:
        if span.start <= marker_start <= span.end:
            return "annotated", "inside", 0, span
        if marker_start <= span.end <= marker_end - 1:
            # Span boundary lands inside the marker (particle included in span).
            return "annotated", "span_ends_in_marker", span.end - marker_start, span
        after_dist = marker_start - (span.end + 1)
        if 0 <= after_dist <= after_tol:
            cand = ("just_after", after_dist, span)
            if best is None or cand[1] < best[1]:
                best = cand
        if before_tol > 0:
            before_dist = span.start - marker_end
            if 0 <= before_dist <= before_tol:
                cand = ("just_before", before_dist, span)
                if best is None or cand[1] < best[1]:
                    best = cand
    if best is not None:
        relation, dist, span = best
        return "annotated", relation, dist, span
    return "candidate_fn", "none", None, None


def nearest_span_distance(pos: int, spans: list[QuoteSpan]) -> int | None:
    if not spans:
        return None
    return min(min(abs(pos - s.start), abs(pos - s.end)) for s in spans)


def find_closer_for_span(
    span: QuoteSpan,
    closers: list[re.Match[str]],
    after_tol: int,
) -> tuple[bool, str, int | None, str]:
    """Whether any closer sits inside the span or within after_tol after it."""
    best: tuple[int, str, int, str] | None = None
    for m in closers:
        ms, me, marker = m.start(), m.end(), m.group()
        if span.start <= ms <= span.end:
            return True, marker, ms, "inside"
        if ms <= span.end <= me - 1:
            return True, marker, ms, "span_ends_in_marker"
        after_dist = ms - (span.end + 1)
        if 0 <= after_dist <= after_tol:
            cand = (after_dist, marker, ms, "just_after")
            if best is None or cand[0] < best[0]:
                best = cand
    if best is None:
        return False, "", None, "none"
    return True, best[1], best[2], best[3]


def gap_regions(
    doc_len: int,
    spans: list[QuoteSpan],
    closer_positions: list[int],
    min_gap_chars: int,
) -> list[dict]:
    """Unannotated stretches that still contain closers."""
    if doc_len <= 0:
        return []
    covered = sorted((s.start, s.end) for s in spans)
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for start, end in covered:
        if start > cursor:
            gaps.append((cursor, start - 1))
        cursor = max(cursor, end + 1)
    if cursor < doc_len:
        gaps.append((cursor, doc_len - 1))

    out: list[dict] = []
    for g_start, g_end in gaps:
        length = g_end - g_start + 1
        if length < min_gap_chars:
            continue
        n_closers = sum(1 for p in closer_positions if g_start <= p <= g_end)
        if n_closers == 0:
            continue
        out.append(
            {
                "gap_start": g_start,
                "gap_end": g_end,
                "gap_chars": length,
                "closers_in_gap": n_closers,
            }
        )
    return out


def scan_book(
    book: str,
    text: str,
    spans: list[QuoteSpan],
    closer_re: re.Pattern[str],
    after_tol: int,
    before_tol: int,
    context: int,
    min_gap_chars: int,
) -> tuple[list[CloserHit], list[GoldSpanRow], list[dict], dict]:
    closers = list(closer_re.finditer(text))
    closer_hits: list[CloserHit] = []
    for m in closers:
        status, relation, dist, span = classify_closer(
            m.start(), m.end(), spans, after_tol, before_tol
        )
        if status == "candidate_fn":
            dist = nearest_span_distance(m.start(), spans)
        quote_preview = ""
        if span is not None:
            quote_preview = _clip(text, span.start, span.end + 1)
            if len(quote_preview) > 160:
                quote_preview = quote_preview[:80] + "…" + quote_preview[-80:]
        closer_hits.append(
            CloserHit(
                book=book,
                marker=m.group(),
                marker_start=m.start(),
                marker_end=m.end(),
                status=status,
                relation=relation,
                dist_to_span=dist,
                span_ann_id=span.ann_id if span else "",
                span_start=span.start if span else None,
                span_end=span.end if span else None,
                context_before=_clip(text, m.start() - context, m.start()),
                context_after=_clip(text, m.end(), m.end() + context),
                quote_preview=quote_preview,
            )
        )

    gold_rows: list[GoldSpanRow] = []
    for span in spans:
        has, marker, c_start, rel = find_closer_for_span(span, closers, after_tol)
        preview = _clip(text, span.start, span.end + 1)
        if len(preview) > 160:
            preview = preview[:80] + "…" + preview[-80:]
        gold_rows.append(
            GoldSpanRow(
                book=book,
                ann_id=span.ann_id,
                start=span.start,
                end=span.end,
                length=span.length,
                has_closer=has,
                closer_marker=marker,
                closer_start=c_start,
                closer_relation=rel,
                text_preview=preview,
                after_preview=_clip(text, span.end + 1, span.end + 1 + context),
            )
        )

    gaps = gap_regions(
        len(text), spans, [m.start() for m in closers], min_gap_chars
    )
    for g in gaps:
        g["book"] = book

    n_ann = sum(1 for h in closer_hits if h.status == "annotated")
    n_fn = sum(1 for h in closer_hits if h.status == "candidate_fn")
    summary = {
        "book": book,
        "doc_chars": len(text),
        "quotation_spans": len(spans),
        "closers": len(closers),
        "closers_annotated": n_ann,
        "closers_candidate_fn": n_fn,
        "gold_spans_with_closer": sum(1 for r in gold_rows if r.has_closer),
        "gold_spans_without_closer": sum(1 for r in gold_rows if not r.has_closer),
        "gaps_with_closers": len(gaps),
        "gap_closers_total": sum(g["closers_in_gap"] for g in gaps),
        "largest_gap_chars": max((g["gap_chars"] for g in gaps), default=0),
    }
    return closer_hits, gold_rows, gaps, summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos", type=Path, default=DEFAULT_REPOS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--books",
        nargs="*",
        default=None,
        help="Optional pecha ids (e.g. P000078). Default: all in-scope Quotation books.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(CLOSER_PROFILES),
        default="default",
        help="Closer regex profile (default=balanced, strong=FN-focused, broad=noisy).",
    )
    parser.add_argument(
        "--mask-source",
        choices=("confirmed", "meta", "file", "union"),
        default="meta",
        help=(
            "Which books count as Quotation-annotated. Default ``meta`` = declared "
            "in meta.yml and Quotation.yml present — the right set for FN hunting. "
            "``confirmed`` includes undeclared books (valid O-negatives, not FN suspects)."
        ),
    )
    parser.add_argument(
        "--include-out-of-scope",
        action="store_true",
        help="Also scan books that are not Quotation in-scope under --mask-source.",
    )
    parser.add_argument("--after-tol", type=int, default=DEFAULT_AFTER_TOL)
    parser.add_argument("--before-tol", type=int, default=DEFAULT_BEFORE_TOL)
    parser.add_argument("--context", type=int, default=60)
    parser.add_argument(
        "--min-gap-chars",
        type=int,
        default=50_000,
        help="Minimum unannotated stretch length for gap report.",
    )
    parser.add_argument(
        "--max-fn-per-book",
        type=int,
        default=0,
        help="If >0, keep at most N candidate_fn rows per book (by marker position).",
    )
    parser.add_argument(
        "--json-samples",
        type=int,
        default=40,
        help="How many candidate_fn + annotated examples to embed in summary.json.",
    )
    args = parser.parse_args()

    closer_re = CLOSER_PROFILES[args.profile]
    book_filter = set(args.books) if args.books else None

    all_closers: list[CloserHit] = []
    all_gold: list[GoldSpanRow] = []
    all_gaps: list[dict] = []
    summaries: list[dict] = []

    for book_dir in find_books(args.repos):
        if book_filter is not None and book_dir.name not in book_filter:
            continue
        rec = load_book(book_dir)
        scope = in_scope_layers(rec, args.mask_source)
        quote_in_scope = "Quotation" in scope
        if not quote_in_scope and not args.include_out_of_scope:
            continue
        if rec.base_path is None or not rec.base_path.is_file():
            continue

        text = rec.base_path.read_text(encoding="utf-8")
        spans = load_quotation_spans(rec, text)
        hits, gold, gaps, summary = scan_book(
            book=rec.book,
            text=text,
            spans=spans,
            closer_re=closer_re,
            after_tol=args.after_tol,
            before_tol=args.before_tol,
            context=args.context,
            min_gap_chars=args.min_gap_chars,
        )
        summary["quote_in_scope"] = quote_in_scope
        summaries.append(summary)

        if args.max_fn_per_book > 0:
            kept_fn = 0
            filtered: list[CloserHit] = []
            for hit in hits:
                if hit.status == "candidate_fn":
                    if kept_fn >= args.max_fn_per_book:
                        continue
                    kept_fn += 1
                filtered.append(hit)
            hits = filtered

        all_closers.extend(hits)
        all_gold.extend(gold)
        all_gaps.extend(gaps)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    write_csv(
        out / "closers.csv",
        [asdict(h) for h in all_closers],
        [
            "book",
            "marker",
            "marker_start",
            "marker_end",
            "status",
            "relation",
            "dist_to_span",
            "span_ann_id",
            "span_start",
            "span_end",
            "context_before",
            "context_after",
            "quote_preview",
        ],
    )
    write_csv(
        out / "closers_candidate_fn.csv",
        [asdict(h) for h in all_closers if h.status == "candidate_fn"],
        [
            "book",
            "marker",
            "marker_start",
            "marker_end",
            "status",
            "relation",
            "dist_to_span",
            "span_ann_id",
            "span_start",
            "span_end",
            "context_before",
            "context_after",
            "quote_preview",
        ],
    )
    write_csv(
        out / "closers_annotated.csv",
        [asdict(h) for h in all_closers if h.status == "annotated"],
        [
            "book",
            "marker",
            "marker_start",
            "marker_end",
            "status",
            "relation",
            "dist_to_span",
            "span_ann_id",
            "span_start",
            "span_end",
            "context_before",
            "context_after",
            "quote_preview",
        ],
    )
    write_csv(
        out / "gold_spans.csv",
        [asdict(r) for r in all_gold],
        [
            "book",
            "ann_id",
            "start",
            "end",
            "length",
            "has_closer",
            "closer_marker",
            "closer_start",
            "closer_relation",
            "text_preview",
            "after_preview",
        ],
    )
    write_csv(
        out / "gaps.csv",
        all_gaps,
        ["book", "gap_start", "gap_end", "gap_chars", "closers_in_gap"],
    )
    write_csv(
        out / "book_summary.csv",
        summaries,
        [
            "book",
            "quote_in_scope",
            "doc_chars",
            "quotation_spans",
            "closers",
            "closers_annotated",
            "closers_candidate_fn",
            "gold_spans_with_closer",
            "gold_spans_without_closer",
            "gaps_with_closers",
            "gap_closers_total",
            "largest_gap_chars",
        ],
    )

    annotated = [h for h in all_closers if h.status == "annotated"]
    candidates = [h for h in all_closers if h.status == "candidate_fn"]
    marker_counts = Counter(h.marker for h in all_closers)
    summary_payload = {
        "params": {
            "profile": args.profile,
            "pattern": closer_re.pattern,
            "after_tol": args.after_tol,
            "before_tol": args.before_tol,
            "min_gap_chars": args.min_gap_chars,
            "mask_source": args.mask_source,
            "books": sorted(book_filter) if book_filter else "all_in_scope",
        },
        "totals": {
            "books": len(summaries),
            "closers": len(all_closers),
            "closers_annotated": len(annotated),
            "closers_candidate_fn": len(candidates),
            "gold_spans": len(all_gold),
            "gold_spans_with_closer": sum(1 for r in all_gold if r.has_closer),
            "gold_spans_without_closer": sum(1 for r in all_gold if not r.has_closer),
            "gaps_with_closers": len(all_gaps),
        },
        "marker_counts": marker_counts.most_common(30),
        "top_fn_books": sorted(
            summaries, key=lambda s: s["closers_candidate_fn"], reverse=True
        )[:20],
        "top_gap_books": sorted(
            summaries, key=lambda s: (s["gap_closers_total"], s["largest_gap_chars"]), reverse=True
        )[:20],
        "sample_annotated": [asdict(h) for h in annotated[: args.json_samples]],
        "sample_candidate_fn": [asdict(h) for h in candidates[: args.json_samples]],
    }
    (out / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    totals = summary_payload["totals"]
    print(
        f"books={totals['books']}  closers={totals['closers']:,}  "
        f"annotated={totals['closers_annotated']:,}  "
        f"candidate_fn={totals['closers_candidate_fn']:,}"
    )
    print(
        f"gold spans with/without closer: "
        f"{totals['gold_spans_with_closer']:,} / {totals['gold_spans_without_closer']:,}"
    )
    print(f"gaps with closers: {totals['gaps_with_closers']:,}")
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
