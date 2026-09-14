"""Diagnose why QUOTE/TSAWA span F1 is near zero on validation windows.

Separates three candidate causes that exact-match span F1 conflates:
token-level detection failure, span fragmentation, and boundary drift.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from datasets.utils.logging import disable_progress_bar  # noqa: E402
from transformers import AutoModelForTokenClassification  # noqa: E402

from layer_detection.training_utils import (  # noqa: E402
    ID2LABEL,
    bio_spans,
    merge_fragmented_spans,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "experiments/mmbert_base_v1.2/window_cache"
DEFAULT_MODEL = ROOT / "experiments/mmbert_base_v1.2/outputs/checkpoint-10832"
VAL_BOOKS = [
    "P000010", "P000025", "P000030", "P000039", "P000043", "P000048", "P000056",
    "P000062", "P000063", "P000069", "P000076", "P000083", "P000085", "P000110",
    "P000111", "P000112", "P000125", "P000147", "P000151", "P000155", "P000184",
    "P000200", "P000215", "P000241", "P000264",
]
LAYERS = ("QUOTE", "TSAWA", "SABCHE", "YIGCHUNG", "CHAPTER", "AUTHOR")


def load_windows(book_id: str, max_tokens: int, per_book: int):
    """Return the windows richest in QUOTE/TSAWA gold tokens for one book."""
    book_dir = CACHE_DIR / book_id
    if not book_dir.is_dir():
        return []
    fingerprints = [
        d for d in sorted(book_dir.iterdir()) if d.is_dir() and not d.name.startswith(".")
    ]
    if not fingerprints:
        return []
    dataset = Dataset.load_from_disk(str(fingerprints[0]))
    scored = []
    for index, labels in enumerate(dataset["labels"]):
        if len(labels) > max_tokens:
            continue
        target = sum(
            1
            for label in labels
            if label != -100 and label > 0 and ID2LABEL[label].split("-", 1)[1] in ("QUOTE", "TSAWA")
        )
        if target:
            scored.append((target, index))
    scored.sort(reverse=True)
    return [dataset[index] for _, index in scored[:per_book]]


def match_count(gold, predicted, tolerance_fn):
    unmatched = set(range(len(gold)))
    matched = 0
    for tag, start, end in predicted:
        candidates = [
            i
            for i in unmatched
            if gold[i][0] == tag
            and abs(gold[i][1] - start) <= tolerance_fn(gold[i])
            and abs(gold[i][2] - end) <= tolerance_fn(gold[i])
        ]
        if candidates:
            unmatched.remove(candidates[0])
            matched += 1
    return matched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--books", type=int, default=10)
    parser.add_argument("--per-book", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()

    disable_progress_bar()
    print(f"loading {args.model}", file=sys.stderr, flush=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa"
    )
    model.eval()
    torch.set_num_threads(8)

    gold_sequences: list[list[str]] = []
    pred_sequences: list[list[str]] = []
    token_confusion: Counter[tuple[str, str]] = Counter()

    used = 0
    for book_id in VAL_BOOKS:
        if used >= args.books:
            break
        for row in load_windows(book_id, args.max_tokens, args.per_book):
            if used >= args.books:
                break
            used += 1
            input_ids = torch.tensor([row["input_ids"]])
            attention_mask = torch.tensor([row["attention_mask"]])
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
            scope = row.get("scope_mask")
            if scope is not None:
                blocked = ~torch.tensor(scope, dtype=torch.bool)
                logits = logits.masked_fill(blocked.unsqueeze(0), float("-inf"))
            predictions = logits.argmax(-1).tolist()

            gold_visible, pred_visible = [], []
            for gold_id, pred_id in zip(row["labels"], predictions):
                if gold_id == -100:
                    continue
                gold_visible.append(ID2LABEL[gold_id])
                pred_visible.append(ID2LABEL[pred_id])
                gold_layer = "O" if gold_id == 0 else ID2LABEL[gold_id].split("-", 1)[1]
                pred_layer = "O" if pred_id == 0 else ID2LABEL[pred_id].split("-", 1)[1]
                token_confusion[(gold_layer, pred_layer)] += 1
            gold_sequences.append(gold_visible)
            pred_sequences.append(pred_visible)
            print(
                f"  [{used}] {book_id}: {len(gold_visible)} supervised tokens",
                file=sys.stderr,
                flush=True,
            )

    print(f"\n### 1. TOKEN-LEVEL detection (layer identity, ignoring B/I and boundaries)")
    header = f"{'gold layer':10s} {'tokens':>9s} {'recall':>8s} {'->O':>8s} {'->other':>8s} {'precision':>10s}"
    print(header)
    print("-" * len(header))
    for layer in ("O",) + LAYERS:
        total = sum(c for (g, p), c in token_confusion.items() if g == layer)
        if not total:
            continue
        hit = token_confusion[(layer, layer)]
        to_o = token_confusion[(layer, "O")] if layer != "O" else 0
        other = total - hit - to_o
        pred_total = sum(c for (g, p), c in token_confusion.items() if p == layer)
        precision = hit / pred_total if pred_total else 0.0
        print(
            f"{layer:10s} {total:9,d} {100 * hit / total:7.1f}% {100 * to_o / total:7.1f}% "
            f"{100 * other / total:7.1f}% {100 * precision:9.1f}%"
        )

    print(f"\n### 2. SPAN COUNTS (fragmentation)")
    gold_spans_all = [bio_spans(s) for s in gold_sequences]
    pred_spans_all = [bio_spans(s) for s in pred_sequences]
    merged_all = [bio_spans(merge_fragmented_spans(s)) for s in pred_sequences]
    header = f"{'layer':10s} {'gold':>7s} {'pred':>7s} {'ratio':>7s} {'gold_med_len':>13s} {'pred_med_len':>13s} {'merged_pred':>12s}"
    print(header)
    print("-" * len(header))
    for layer in LAYERS:
        gold = [s for spans in gold_spans_all for s in spans if s[0] == layer]
        pred = [s for spans in pred_spans_all for s in spans if s[0] == layer]
        merged = [s for spans in merged_all for s in spans if s[0] == layer]
        if not gold and not pred:
            continue
        gmed = statistics.median([e - s + 1 for _, s, e in gold]) if gold else 0
        pmed = statistics.median([e - s + 1 for _, s, e in pred]) if pred else 0
        ratio = len(pred) / len(gold) if gold else float("inf")
        print(
            f"{layer:10s} {len(gold):7d} {len(pred):7d} {ratio:7.2f} {gmed:13.0f} {pmed:13.0f} {len(merged):12d}"
        )

    print(f"\n### 3. SPAN F1 vs BOUNDARY TOLERANCE (per layer, raw predictions)")
    tolerances = [0, 1, 2, 5, 10, 20, 50, 100]
    print(f"{'layer':10s} " + " ".join(f"tol{t:<4d}" for t in tolerances) + "  rel10%")
    print("-" * (10 + 8 * len(tolerances) + 8))
    for layer in LAYERS:
        cells = []
        for tol in tolerances + ["rel"]:
            tp = predicted_total = gold_total = 0
            for gold_spans, pred_spans in zip(gold_spans_all, pred_spans_all):
                gold = [s for s in gold_spans if s[0] == layer]
                pred = [s for s in pred_spans if s[0] == layer]
                gold_total += len(gold)
                predicted_total += len(pred)
                if tol == "rel":
                    fn = lambda g: max(2, int((g[2] - g[1] + 1) * 0.1))
                else:
                    fn = lambda g, t=tol: t
                tp += match_count(gold, pred, fn)
            precision = tp / predicted_total if predicted_total else 0.0
            recall = tp / gold_total if gold_total else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            cells.append(f"{f1:.3f}")
        print(f"{layer:10s} " + " ".join(c.ljust(7) for c in cells))

    print(f"\n### 4. BOUNDARY ERROR for best-overlapping prediction (gold span -> pred span)")
    header = f"{'layer':10s} {'n_gold':>7s} {'any_overlap':>12s} {'med_start_err':>14s} {'med_end_err':>12s} {'med_iou':>8s}"
    print(header)
    print("-" * len(header))
    for layer in LAYERS:
        start_errors, end_errors, ious = [], [], []
        overlapped = total = 0
        for gold_spans, pred_spans in zip(gold_spans_all, pred_spans_all):
            pred = [s for s in pred_spans if s[0] == layer]
            for _, gs, ge in [s for s in gold_spans if s[0] == layer]:
                total += 1
                best, best_overlap = None, 0
                for _, ps, pe in pred:
                    overlap = min(ge, pe) - max(gs, ps) + 1
                    if overlap > best_overlap:
                        best, best_overlap = (ps, pe), overlap
                if best is None:
                    continue
                overlapped += 1
                ps, pe = best
                start_errors.append(abs(gs - ps))
                end_errors.append(abs(ge - pe))
                union = max(ge, pe) - min(gs, ps) + 1
                ious.append(best_overlap / union)
        if not total:
            continue
        print(
            f"{layer:10s} {total:7d} {100 * overlapped / total:11.1f}% "
            f"{statistics.median(start_errors) if start_errors else -1:14.0f} "
            f"{statistics.median(end_errors) if end_errors else -1:12.0f} "
            f"{statistics.median(ious) if ious else 0:8.2f}"
        )

    print("\n### 5. WHY spans break: cause of each predicted span boundary")
    header = f"{'layer':10s} {'breaks':>8s} {'B_after_B/I':>13s} {'after_O_gap':>13s} {'gap<=tol':>9s} {'other_layer':>12s}"
    print(header)
    print("-" * len(header))
    breaks: dict[str, Counter[str]] = defaultdict(Counter)
    tolerances_map = {"QUOTE": 6, "TSAWA": 6, "YIGCHUNG": 4, "SABCHE": 4, "CHAPTER": 2, "AUTHOR": 1}
    for sequence in pred_sequences:
        spans = bio_spans(sequence)
        for previous, current in zip(spans, spans[1:]):
            if current[0] != previous[0]:
                continue
            gap = current[1] - previous[2] - 1
            if gap == 0:
                breaks[current[0]]["adjacent"] += 1
            elif all(sequence[i] == "O" for i in range(previous[2] + 1, current[1])):
                breaks[current[0]]["gap"] += 1
                if gap <= tolerances_map.get(current[0], 0):
                    breaks[current[0]]["gap_within_tol"] += 1
            else:
                breaks[current[0]]["other"] += 1
    for layer in LAYERS:
        counter = breaks[layer]
        total = counter["adjacent"] + counter["gap"] + counter["other"]
        if not total:
            continue
        print(
            f"{layer:10s} {total:8d} {100 * counter['adjacent'] / total:12.1f}% "
            f"{100 * counter['gap'] / total:12.1f}% {100 * counter['gap_within_tol'] / total:8.1f}% "
            f"{100 * counter['other'] / total:11.1f}%"
        )


if __name__ == "__main__":
    main()
