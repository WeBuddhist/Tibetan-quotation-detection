"""Rescore v1.3 model on the new BIOE quotation splits."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForTokenClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if importlib.util.find_spec("layer_detection") is None:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from layer_detection.bioe_layer import load_bioe_splits  # noqa: E402
from layer_detection.single_layer import (  # noqa: E402
    ID2LABEL15,
    LABEL2ID15,
    LABELS15,
    iou50_metrics,
    viterbi_decode,
)

NEG = -1.0e9
QUOTE_IDS = (LABEL2ID15["B-QUOTE"], LABEL2ID15["I-QUOTE"])


@dataclass(frozen=True)
class Scheme15:
    labels: tuple[str, ...] = tuple(LABELS15)
    n_labels: int = len(LABELS15)
    o_id: int = LABEL2ID15["O"]


def _pad_rows(rows: list[np.ndarray], pad_value: float | int, extra_dim: int | None) -> np.ndarray:
    width = max(row.shape[0] for row in rows)
    if extra_dim is None:
        out = np.full((len(rows), width), pad_value, dtype=np.int64)
        for i, row in enumerate(rows):
            out[i, : row.shape[0]] = row
        return out
    out = np.zeros((len(rows), width, extra_dim), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, : row.shape[0]] = row
    return out


def evaluate_v13(
    logits: np.ndarray,
    gold_ids: np.ndarray,
    break_penalty: float,
    masked: bool = False,
) -> dict[str, float]:
    work = logits
    if masked:
        work = np.full_like(logits, NEG)
        work[..., [0, *QUOTE_IDS]] = logits[..., [0, *QUOTE_IDS]]
    decoded_ids = viterbi_decode(work, Scheme15(), break_penalty=break_penalty)

    pred_seqs = []
    gold_seqs = []
    for preds, golds in zip(decoded_ids, gold_ids, strict=True):
        p_seq = []
        g_seq = []
        for pred, gold in zip(preds, golds, strict=True):
            if int(gold) == -100:
                continue
            name = ID2LABEL15[int(pred)]
            p_seq.append(name if name in ("B-QUOTE", "I-QUOTE") else "O")
            gold_id = int(gold)
            if gold_id == 3:
                g_seq.append("I-QUOTE")
            elif gold_id == 1:
                g_seq.append("B-QUOTE")
            elif gold_id == 2:
                g_seq.append("I-QUOTE")
            else:
                g_seq.append("O")
        pred_seqs.append(p_seq)
        gold_seqs.append(g_seq)

    return iou50_metrics(pred_seqs, gold_seqs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rescore v1.3 model on new BIOE splits.")
    parser.add_argument("--config", type=Path, default=Path("experiments/mmbert_quotation_v3/config.yaml"),
                        help="Path to the quotation v3 config file")
    parser.add_argument("--split", type=str, default="test", choices=["test", "validation"],
                        help="Split to evaluate on")
    parser.add_argument("--tune-on", type=str, default="validation",
                        help="Split to tune the Viterbi break penalty on")
    args = parser.parse_args()

    # Load config
    if not args.config.exists():
        print(f"Config file not found at {args.config}", file=sys.stderr)
        return 1
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    # Load splits
    print("Loading BIOE dataset splits...")
    splits = load_bioe_splits(cfg)
    
    # Load model and tokenizer
    checkpoint = "karma689/mmbert-base-layer-detection-v1.3"
    print(f"Loading tokenizer and model from {checkpoint}...")
    
    # Assert tokenizer vocab identity
    tokenizer_base = AutoTokenizer.from_pretrained(cfg["checkpoint"])
    tokenizer_v13 = AutoTokenizer.from_pretrained(checkpoint)
    assert tokenizer_base.get_vocab() == tokenizer_v13.get_vocab(), "Tokenizer vocabularies do not match!"
    print("Tokenizer vocab identity verified.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Using device={device} dtype={dtype}")

    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
    )
    model.to(device).eval()

    # Get validation and test splits
    val_bundle = splits[args.tune_on]
    test_bundle = splits[args.split]

    def get_logits_and_golds(bundle):
        dataset = bundle.dataset
        all_logits = []
        all_golds = []
        for row in dataset:
            input_ids = torch.tensor([row["input_ids"]], device=device)
            attention_mask = torch.tensor([row["attention_mask"]], device=device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0].float().cpu().numpy()
            all_logits.append(logits)
            all_golds.append(np.asarray(row["labels"], dtype=np.int64))
        return _pad_rows(all_logits, 0.0, all_logits[0].shape[-1]), _pad_rows(
            all_golds, -100, None
        )

    print(f"Running inference over '{args.tune_on}' split...")
    val_logits, val_golds = get_logits_and_golds(val_bundle)

    print(f"Running inference over '{args.split}' split...")
    test_logits, test_golds = get_logits_and_golds(test_bundle)

    # Sweep break penalty on validation split
    penalties = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0]
    
    print("\n--- Sweeping 15-class Viterbi on Validation ---")
    best_val_f1 = -1.0
    best_penalty = 0.0
    for penalty in penalties:
        metrics = evaluate_v13(val_logits, val_golds, penalty, masked=False)
        f1 = metrics["iou50_f1"]
        print(f"  penalty={penalty:4.1f} -> precision={metrics['iou50_precision']:.4f} recall={metrics['iou50_recall']:.4f} f1={f1:.4f}")
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_penalty = penalty
    print(f"Best 15-class penalty: {best_penalty} (F1: {best_val_f1:.4f})")

    print("\n--- Sweeping Masked-to-3 Viterbi on Validation ---")
    best_val_f1_masked = -1.0
    best_penalty_masked = 0.0
    for penalty in penalties:
        metrics = evaluate_v13(val_logits, val_golds, penalty, masked=True)
        f1 = metrics["iou50_f1"]
        print(f"  penalty={penalty:4.1f} -> precision={metrics['iou50_precision']:.4f} recall={metrics['iou50_recall']:.4f} f1={f1:.4f}")
        if f1 > best_val_f1_masked:
            best_val_f1_masked = f1
            best_penalty_masked = penalty
    print(f"Best Masked-to-3 penalty: {best_penalty_masked} (F1: {best_val_f1_masked:.4f})")

    # Apply to test split
    print(f"\nApplying to '{args.split}' split...")
    test_metrics = evaluate_v13(test_logits, test_golds, best_penalty, masked=False)
    test_metrics_masked = evaluate_v13(test_logits, test_golds, best_penalty_masked, masked=True)

    headline_f1 = test_metrics["iou50_f1"]
    diagnostic_f1 = test_metrics_masked["iou50_f1"]

    print(f"\nHeadline v1.3_on_new_test (15-class): F1={headline_f1:.4f}")
    print(f"Diagnostic v1.3_on_new_test_masked3: F1={diagnostic_f1:.4f}")

    # Write results
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "v13_on_new_test.json"
    
    results = {
        "v1.3_on_new_test": headline_f1,
        "v1.3_on_new_test_masked3": diagnostic_f1,
        "best_penalty_15class": best_penalty,
        "best_penalty_masked3": best_penalty_masked,
    }
    
    results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved rescore results to {results_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
