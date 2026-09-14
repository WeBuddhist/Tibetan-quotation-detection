#!/usr/bin/env python3
"""Extract inclusive character spans from a local OpenPecha book.

Gold offsets in this corpus are inclusive on both ends:
``text[start:end+1]``, not Python's exclusive-end slice.

Accepts a TSV-ish line (split is ignored, kept so you can paste audit rows):

  P000151	validation	[304112, 308503]

Examples:
  python script/extract_span.py P000151 validation '[304112, 308503]'
  python script/extract_span.py --book P000151 --start 304112 --end 308503
  python script/extract_span.py --book P000151 --span 304112,308503 -o span.txt
  echo 'P000151	validation	[304112, 308503]' | python script/extract_span.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layer_detection.audit_utils import load_book  # noqa: E402
from layer_detection.dataset_utils import DEFAULT_REPOS_DIR  # noqa: E402

ROW_RE = re.compile(
    r"""
    (?P<book>P\d+)
    (?:[\s,;|]+(?P<split>\S+))?
    [\s,;|]*\[?\s*(?P<start>\d+)\s*[,:]\s*(?P<end>\d+)\s*\]?
    """,
    re.VERBOSE | re.IGNORECASE,
)


def parse_row(raw: str) -> tuple[str, str | None, int, int]:
    text = raw.strip()
    if not text or text.startswith("#"):
        raise ValueError(f"empty span row: {raw!r}")
    match = ROW_RE.search(text)
    if not match:
        raise ValueError(
            f"could not parse book/offsets from {raw!r}; "
            "expected e.g. P000151\\tvalidation\\t[304112, 308503]"
        )
    return (
        match.group("book").upper(),
        match.group("split"),
        int(match.group("start")),
        int(match.group("end")),
    )


def load_text(repos: Path, book_id: str) -> str:
    rec = load_book(repos / book_id)
    if not rec.base_path or not rec.base_path.is_file():
        raise FileNotFoundError(f"No base text for {book_id} under {repos}")
    return rec.base_path.read_text(encoding="utf-8")


def extract_inclusive(text: str, start: int, end: int) -> str:
    if start < 0 or end < start:
        raise ValueError(f"invalid inclusive span [{start}, {end}]")
    if end >= len(text):
        raise ValueError(
            f"end {end} is past document length {len(text)} (last index {len(text) - 1})"
        )
    return text[start : end + 1]


def format_preview(span: str, limit: int = 120) -> str:
    flat = span.replace("\n", "⏎")
    if len(flat) <= limit:
        return flat
    return f"{flat[: limit // 2]}…{flat[-(limit // 2) :]}"


def run_one(
    book_id: str,
    start: int,
    end: int,
    repos: Path,
    context: int,
    out: Path | None,
    split: str | None,
) -> int:
    text = load_text(repos, book_id)
    span = extract_inclusive(text, start, end)
    ctx_lo = max(0, start - context)
    ctx_hi = min(len(text), end + 1 + context)
    before = text[ctx_lo:start]
    after = text[end + 1 : ctx_hi]

    print(
        f"book={book_id}"
        + (f"  split={split}" if split else "")
        + f"  span=[{start}, {end}]  chars={len(span):,}  doc_len={len(text):,}"
        + f"  slice=text[{start}:{end}+1]"
    )
    print(f"first={span[:1]!r}  last={span[-1:]!r}")
    if context:
        print(f"before={before.replace(chr(10), '⏎')!r}")
        print(f"after={after.replace(chr(10), '⏎')!r}")
    print("---")
    print(span)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(span, encoding="utf-8")
        print(f"---\nwrote {out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "row",
        nargs="*",
        help="P000151 validation [304112, 308503]  (or pipe the same line on stdin)",
    )
    parser.add_argument("--book", help="OpenPecha book id, e.g. P000151")
    parser.add_argument("--start", type=int, help="Inclusive start offset")
    parser.add_argument("--end", type=int, help="Inclusive end offset")
    parser.add_argument(
        "--span",
        help="Inclusive start,end as 304112,308503 or [304112, 308503]",
    )
    parser.add_argument("--repos", type=Path, default=DEFAULT_REPOS_DIR)
    parser.add_argument(
        "--context",
        type=int,
        default=0,
        help="Extra characters to print before/after the span (stderr-style metadata)",
    )
    parser.add_argument("-o", "--out", type=Path, help="Write the span text to this file")
    args = parser.parse_args(argv)

    book_id: str | None = args.book
    start: int | None = args.start
    end: int | None = args.end
    split: str | None = None

    if args.span:
        match = re.search(r"(\d+)\s*[,:]\s*(\d+)", args.span)
        if not match:
            print(f"could not parse --span {args.span!r}", file=sys.stderr)
            return 2
        start, end = int(match.group(1)), int(match.group(2))

    if args.row:
        book_id, split, start, end = parse_row(" ".join(args.row))
    elif book_id is None or start is None or end is None:
        stdin = sys.stdin.read() if not sys.stdin.isatty() else ""
        if stdin.strip():
            # first non-empty line
            for line in stdin.splitlines():
                if line.strip() and not line.lstrip().startswith("#"):
                    book_id, split, start, end = parse_row(line)
                    break
        if book_id is None or start is None or end is None:
            parser.error("need a row like 'P000151 validation [304112, 308503]' or --book/--start/--end")

    assert book_id is not None and start is not None and end is not None
    try:
        return run_one(book_id, start, end, args.repos, args.context, args.out, split)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
