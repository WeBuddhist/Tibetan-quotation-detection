"""Bootstrap confidence intervals for DharmaBench QUDT runs.

``docs/dharmabench-qudt-evaluation.md`` §6 lists resampled CIs as missing from
every QUDT run we have published, and the leaderboard gaps are small enough
that a point estimate cannot separate two systems. MUC is averaged per
document, so scoring each document once and resampling those per-document
scores is equivalent to resampling the evaluation itself.

    python script/qudt_bootstrap.py outputs/qudt_v13_recheck outputs/qudt_v3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def per_document_scores(predictions_path: Path, key: str) -> np.ndarray:
    """MUC type F1 for each document, using the same eval4ner call as the harness."""
    import eval4ner.muc as muc

    scores = []
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            pred = [tuple(span) for span in row[key]["text_spans"]]
            gold = [tuple(span) for span in row["gold_text_spans"]]
            text = (row["text"] or "").lower().replace("-", "")
            result = muc.evaluate_all([pred], [gold], [text], verbose=False)
            scores.append(result["type"]["f1_score"])
    return np.asarray(scores, dtype=np.float64)


def bootstrap(scores: np.ndarray, n: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(scores), size=(n, len(scores)))
    means = scores[draws].mean(axis=1)
    return float(scores.mean()), float(np.percentile(means, 2.5)), float(
        np.percentile(means, 97.5)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--key", default="predictions_viterbi",
                        choices=["predictions_viterbi", "predictions_argmax"])
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    baseline = None
    rows = []
    for run_dir in args.run_dirs:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        scores = per_document_scores(run_dir / "predictions.jsonl", args.key)
        mean, low, high = bootstrap(scores, args.n, args.seed)
        rows.append((run_dir.name, metrics.get("model_checkpoint", "?"), scores, mean, low, high))
        print(
            f"{run_dir.name:22s} MUC type F1 = {mean:.4f}  "
            f"95% CI [{low:.4f}, {high:.4f}]  n_docs={len(scores)}"
        )
        if baseline is None:
            baseline = (run_dir.name, scores)

    if len(rows) > 1:
        print("\npaired differences vs", baseline[0])
        rng = np.random.default_rng(args.seed)
        for name, _ckpt, scores, *_ in rows[1:]:
            delta = scores - baseline[1]
            draws = rng.integers(0, len(delta), size=(args.n, len(delta)))
            means = delta[draws].mean(axis=1)
            low, high = np.percentile(means, [2.5, 97.5])
            verdict = "significant" if low > 0 or high < 0 else "not significant"
            print(
                f"  {name:22s} delta = {delta.mean():+.4f}  "
                f"95% CI [{low:+.4f}, {high:+.4f}]  ({verdict})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
