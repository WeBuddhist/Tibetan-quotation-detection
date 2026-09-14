#!/usr/bin/env python3
"""Size the windows from the corpus, then prove the sizing on all 266 books.

    python script/verify_window_plan.py
    python script/verify_window_plan.py --window 3200 --overlap 1600
    python script/verify_window_plan.py --revision v1.1 --tokens-per-char 1.6

Prints, per layer: how many gold spans land whole inside at least one window.
Anything below 100% is the tail you route to the encoder or recover with the
"join spans that meet at a boundary" rule in results/chunking-spec.md §4c.
"""

from __future__ import annotations

import argparse
import collections

import numpy as np
from datasets import load_dataset

from layer_detection.window_plan import make_windows, check_windows, plan_windows

LAYERS = ("QUOTATION", "TSAWA", "YIGCHUNG", "SABCHE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="karma689/layer_detection")
    ap.add_argument("--revision", default="v1.1")
    ap.add_argument("--window", type=int, default=None, help="override window chars")
    ap.add_argument("--overlap", type=int, default=None, help="override overlap chars")
    ap.add_argument("--max-output-tokens", type=int, default=65_000)
    ap.add_argument("--tokens-per-char", type=float, default=2.0)
    ap.add_argument("--thinking-tokens", type=int, default=8_000)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, revision=args.revision)
    books = [(r["book_id"], r["text"], r["spans"]) for sp in ds for r in ds[sp]]
    print(f"{len(books)} books, {sum(len(t) for _, t, _ in books):,} characters")

    # ---- 1. what length must be contained whole? -------------------------- #
    lengths = collections.defaultdict(list)
    for _, _, spans in books:
        for s in spans:
            lab = str(s["label"]).upper()
            if lab in LAYERS:
                lengths[lab].append(int(s["end"]) - int(s["start"]) + 1)

    print("\nspan length, characters")
    print(f"{'layer':10s} {'n':>7s} {'p50':>7s} {'p99':>7s} {'p99.9':>7s} {'max':>7s}")
    p99s = {}
    for lab in LAYERS:
        a = np.array(lengths[lab])
        p99s[lab] = np.percentile(a, 99)
        print(f"{lab:10s} {len(a):>7,} {np.percentile(a,50):>7,.0f} "
              f"{p99s[lab]:>7,.0f} {np.percentile(a,99.9):>7,.0f} {a.max():>7,}")
    containment = int(np.ceil(max(p99s.values())))
    driver = max(p99s, key=p99s.get)
    print(f"\njoint p99 = {containment:,} chars, set by {driver}")

    # ---- 2. sizing -------------------------------------------------------- #
    plan = plan_windows(
        containment_chars=containment,
        max_output_tokens=args.max_output_tokens,
        tokens_per_char=args.tokens_per_char,
        thinking_tokens=args.thinking_tokens,
    )
    W = args.window or plan.window_chars
    L = args.overlap or plan.overlap_chars
    print(f"\nplan: window={plan.window_chars:,} overlap={plan.overlap_chars:,} chars")
    print(f"  binding constraint : {plan.binding_constraint}")
    print(f"  budget would allow : {plan.budget_ceiling_chars:,} chars/window")
    print(f"  worst-case output  : {plan.worst_case_output_tokens:,} tokens "
          f"({100*plan.worst_case_output_tokens/args.max_output_tokens:.0f}% of budget, "
          f"at 100% span density)")
    if (W, L) != (plan.window_chars, plan.overlap_chars):
        print(f"  OVERRIDDEN to window={W:,} overlap={L:,}")

    # ---- 3. proof on every book ------------------------------------------- #
    tot = collections.Counter()
    worst_cut = collections.Counter()
    n_windows = 0
    max_w = 0
    min_ov = None
    problems: list[str] = []

    for bid, text, spans in books:
        ws = make_windows(text, bid, W, L)
        rep = check_windows(text, ws, spans, LAYERS)
        n_windows += rep["n_windows"]
        max_w = max(max_w, rep["max_window_chars"])
        if rep["min_overlap_chars"] is not None:
            min_ov = rep["min_overlap_chars"] if min_ov is None else min(min_ov, rep["min_overlap_chars"])
        problems += [f"{bid}: {p}" for p in rep["problems"]]
        for lab, d in rep["layers"].items():
            tot[lab + "_whole"] += d["whole"]
            tot[lab + "_cut"] += d["cut"]
            worst_cut[lab] = max(worst_cut[lab], d["longest_cut"])

    print(f"\n{n_windows:,} windows   max window {max_w:,} chars   "
          f"min overlap actually seen {min_ov:,} chars")
    print(f"structural problems: {len(problems)}")
    for p in problems[:10]:
        print("  " + p)

    print(f"\n{'layer':10s} {'whole':>7s} {'cut':>5s} {'% whole':>8s}  longest cut span")
    for lab in LAYERS:
        w, c = tot[lab + "_whole"], tot[lab + "_cut"]
        pct = 100 * w / (w + c) if (w + c) else 100.0
        print(f"{lab:10s} {w:>7,} {c:>5,} {pct:>7.3f}%  {worst_cut[lab]:,} chars")


if __name__ == "__main__":
    main()
