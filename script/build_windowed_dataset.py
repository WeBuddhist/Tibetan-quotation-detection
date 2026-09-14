"""Build and optionally push the w8192/s4916 windowed layer-detection dataset.

Loads ``karma689/layer_detection`` revision ``v1.1``, tokenizes with the v1.3
checkpoint's fast tokenizer (no model weights), windows at step 4916
(HF overlap 3276), attaches v2 supervision fields, verifies hard gates, writes
locally, then publishes a **named config** on the same repo. Does not train.

    python script/build_windowed_dataset.py --no-push
    python script/build_windowed_dataset.py --verify-only
    python script/check_windowed_v2.py

CLI ``--stride`` is the **step** (default 4916), not the HuggingFace overlap.
Never ``push_to_hub`` without ``config_name`` — that overwrites the default
book-level dataset. Never move tag ``v1.1``.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import inspect
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import DatasetCard, HfApi, hf_hub_download
from transformers import AutoConfig, AutoTokenizer

from layer_detection.dataset_utils import apply_confirmed_scope
from layer_detection.training_utils import (
    ID2LABEL,
    LABEL2ID,
    LABELS,
    tokenize_book_windows,
)
from layer_detection.windowed_v2 import (
    DEFAULT_DAMAGED_BOOKS,
    DEFAULT_MIN_SPAN_CHARS,
    DEFAULT_SMOOTHING,
    DEFAULT_STEP,
    DEFAULT_WINDOW_LEN,
    NAIVE_SCALE_WINDOWS,
    TARGET_WINDOWS,
    V13_DATASET_REVISION,
    V13_HF_STRIDE,
    V13_STEP,
    V13_WINDOWS,
    WRONG_MODEL_WINDOWS,
    VerificationError,
    attach_v2_supervision,
    parse_damaged_books,
    parse_smoothing,
    resolve_windowing,
    verify_windowed_v2,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "windowed_w8192_s4916.yaml"

# Current training pipeline — report before this job's values are applied.
CURRENT_STRIDE_SITES = [
    ("experiments/mmbert_base_v1.3/config.yaml", 5120, "canonical training config"),
    ("experiments/mmbert_base_v1.2/config.yaml", 5120, "historical"),
    ("experiments/mmbert_base_v1/config.yaml", 5120, "historical"),
    ("script/dump_eval_logits.py --stride", 0, "inference default unless config/cache"),
]


class GateFailure(Exception):
    """A hard verification gate failed; do not push."""


def shard_path(shard_dir: Path, index: int, book_id: str) -> Path:
    return shard_dir / f"{index:04d}_{book_id}.parquet"


def iter_parquet_rows(paths: list[Path]):
    """Yield one window dict at a time without building an Arrow cache."""
    import pyarrow.parquet as pq

    for path in paths:
        reader = pq.ParquetFile(path)
        for batch in reader.iter_batches(batch_size=4):
            for row in batch.to_pylist():
                yield row


def resume_book_stats(
    path: Path,
    row: dict[str, Any],
    *,
    max_length: int,
    step: int,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    n_windows = int(pq.read_metadata(path).num_rows)
    sidecar = path.with_suffix(".stats.json")
    if sidecar.is_file():
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        saved["n_windows"] = n_windows
        return saved
    if n_windows == 1:
        table = pq.read_table(path, columns=["input_ids"])
        full_tokens = len(table.column("input_ids")[0].as_py())
    else:
        full_tokens = max_length + (n_windows - 1) * step
    return {
        "book_id": row["book_id"],
        "n_windows": n_windows,
        "full_tokens": full_tokens,
        "n_chars": len(row["text"]),
        "short": n_windows == 1 and full_tokens < max_length,
        "overlaps": [],
        "split_spans": [],
        "max_span_tokens": 0,
        "max_span_meta": None,
        "v2": {"loss_mask_zeros": {}},
        "resumed": True,
    }


def load_job_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Job config must be a mapping: {path}")
    return cfg


def report_job_facts(*, window_len: int, step: int, hf_stride: int) -> None:
    """Banner: live v1.3 vs this job. Do not guess revision, stride, or label scheme."""
    print("=" * 72)
    print("FACTS — live v1.3 pipeline vs this job")
    print("=" * 72)
    print("Label scheme: one flat 15-class BIO softmax (not K independent heads).")
    print(f"  LABELS ({len(LABELS)}): {LABELS}")
    print("  Hub span labels: QUOTATION, SABCHE, …; BIO tag for quotations is QUOTE.")
    print("  Overlapping layers collapse at token time (Quotation wins).")
    print("  v2 `spans` do NOT collapse; BIO `labels` stay winner-takes-all.")
    print()
    print("v1.3 canonical training (experiments/mmbert_base_v1.3/config.yaml):")
    print(f"  dataset_repo: karma689/layer_detection")
    print(f"  dataset_revision: {V13_DATASET_REVISION!r}  (unpinned Hub default)")
    print(f"  max_length: {window_len}")
    print(f"  HF tokenizer stride (OVERLAP): {V13_HF_STRIDE}")
    print(f"  step: {V13_STEP}  ({V13_HF_STRIDE / window_len * 100:.1f}% overlap)")
    print(f"  windows/epoch (measured): {V13_WINDOWS}")
    print()
    print("This job:")
    print("  source: karma689/layer_detection revision='v1.1' (start==end clean)")
    print("  If main has drifted from v1.1, live training labels can differ.")
    print(f"  HF tokenizer stride (OVERLAP): {hf_stride}")
    print(f"  CLI --stride / step: {step}  ({hf_stride / window_len * 100:.2f}% overlap)")
    print(f"  expected windows/epoch: ~{TARGET_WINDOWS}  (not {WRONG_MODEL_WINDOWS}, not {NAIVE_SCALE_WINDOWS})")
    print(
        f"  extras drop {(1 - V13_STEP / step) * 100:.1f}%; the 266 first windows stay. "
        f"~{100 * (V13_WINDOWS - TARGET_WINDOWS) / V13_WINDOWS:.0f}% fewer than {V13_WINDOWS}."
    )
    print("  Warmup and LR schedule MUST be rescaled or the same epoch count undertrains.")
    print()
    print(
        "HuggingFace tokenizer `stride` is OVERLAP, not step. "
        "tokenize_book_windows passes it straight through."
    )
    for location, value, note in CURRENT_STRIDE_SITES:
        print(f"  {location}: stride={value}  ({note})")
    print(
        "Wrong/mixed notes: some docs treat HF stride 5120 as step. "
        "No hidden 1843 step in code. Do not use 9103 (false 1843 model) or "
        "8906 (naive 14338×3072/4916, which scales the 266 first windows)."
    )
    print()


def hash_tokenizer_files(repo: str) -> tuple[str, dict[str, str], int]:
    """Return combined sha256, per-file hashes, and total bytes."""
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    per_file: dict[str, str] = {}
    blobs: list[bytes] = []
    total_bytes = 0
    for name in names:
        try:
            path = Path(hf_hub_download(repo, name, repo_type="model"))
        except Exception:
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        per_file[name] = digest
        blobs.append(name.encode() + b"\0" + bytes.fromhex(digest))
        total_bytes += len(data)
    combined = hashlib.sha256(b"".join(blobs)).hexdigest()
    return combined, per_file, total_bytes


def assert_tokenizer(cfg: dict[str, Any]) -> Any:
    print("=" * 72)
    print("TOKENIZER PREREQUISITES")
    print("=" * 72)
    v13 = cfg["tokenizer_repo"]
    api = HfApi()
    files = api.list_repo_files(v13, repo_type="model")
    print(f"{v13} files ({len(files)}):")
    for name in sorted(files):
        print(f"  {name}")
    tok_files = [
        name
        for name in files
        if "tokenizer" in name.lower()
        or Path(name).name
        in {
            "vocab.txt",
            "vocab.json",
            "merges.txt",
            "spiece.model",
            "special_tokens_map.json",
        }
    ]
    print(f"tokenizer-like files: {tok_files or '(none)'}")
    if not tok_files:
        raise GateFailure(
            f"{v13} has no tokenizer files; cannot confirm the trained tokenizer"
        )

    config = AutoConfig.from_pretrained(v13)
    print(f"config class: {type(config).__name__}")
    print(f"model_type: {getattr(config, 'model_type', None)}")
    print(f"max_position_embeddings: {config.max_position_embeddings}")
    print(f"config.vocab_size: {config.vocab_size}")
    print(f"tokenizer_class: {getattr(config, 'tokenizer_class', None)}")
    print(f"_name_or_path: {getattr(config, '_name_or_path', None)}")
    if int(config.max_position_embeddings) != int(cfg["max_length"]):
        raise GateFailure(
            f"max_position_embeddings={config.max_position_embeddings} "
            f"!= max_length={cfg['max_length']}"
        )

    # Load tokenizer files from the v1.3 repo (config + tokenizer json only).
    tokenizer = AutoTokenizer.from_pretrained(v13)
    print(f"tokenizer class: {type(tokenizer).__name__}")
    print(f"name_or_path: {tokenizer.name_or_path}")
    print(f"is_fast: {tokenizer.is_fast}")
    print(f"vocab_size: {tokenizer.vocab_size}  len: {len(tokenizer)}")
    if tokenizer.is_fast is not True:
        raise GateFailure("tokenizer.is_fast is not True; offset_mapping required")
    if len(tokenizer) != 256000:
        raise GateFailure(f"expected vocab 256000, got {len(tokenizer)}")

    combined, per_file, n_bytes = hash_tokenizer_files(v13)
    print(f"tokenizer file hashes ({n_bytes} bytes):")
    for name, digest in per_file.items():
        print(f"  {name}: {digest}")
    print(f"combined tokenizer-files sha256: {combined}")
    tokenizer._qd_file_hash = combined  # type: ignore[attr-defined]
    tokenizer._qd_file_hashes = per_file  # type: ignore[attr-defined]
    return tokenizer


def resolve_source_commits(repo: str, revision: str) -> dict[str, str]:
    api = HfApi()
    refs = api.list_repo_refs(repo, repo_type="dataset")
    tags = {tag.name: tag.target_commit for tag in refs.tags}
    branches = {branch.name: branch.target_commit for branch in refs.branches}
    commits = list(api.list_repo_commits(repo, repo_type="dataset"))
    data_commit = ""
    for commit in commits:
        title = commit.title or ""
        if "v1.1" in title or "start==end" in title:
            data_commit = commit.commit_id
            break
    info = {
        "revision": revision,
        "tag_commit": tags.get(revision, ""),
        "main_commit": branches.get("main", ""),
        "v1.0_tag": tags.get("v1.0", ""),
        "data_commit": data_commit,
    }
    print("source commits:", json.dumps(info, indent=2))
    if revision == "v1.0" or info["tag_commit"] == info["v1.0_tag"]:
        raise GateFailure("refusing to load v1.0")
    if not info["tag_commit"]:
        raise GateFailure(f"no dataset tag {revision!r}")
    return info


def audit_source(ds: Any) -> dict[str, Any]:
    split_ids = {name: list(split["book_id"]) for name, split in ds.items()}
    n_books = sum(len(ids) for ids in split_ids.values())
    print(
        f"splits: train={len(split_ids.get('train', []))} "
        f"validation={len(split_ids.get('validation', []))} "
        f"test={len(split_ids.get('test', []))}  total_books={n_books}"
    )
    if n_books != 266:
        raise GateFailure(f"expected 266 books, got {n_books}")
    if (
        len(split_ids.get("train", [])) != 217
        or len(split_ids.get("validation", [])) != 25
        or len(split_ids.get("test", [])) != 24
    ):
        raise GateFailure(f"expected 217/25/24, got { {k: len(v) for k, v in split_ids.items()} }")

    total_spans = 0
    min_delta = None
    n_start_ge_end = 0
    n_oob = 0
    n_end_eq_len = 0
    decode_samples: list[tuple[str, dict, str]] = []
    for split in ds.values():
        for row in split:
            text = row["text"]
            n = len(text)
            for span in row["spans"]:
                start, end = int(span["start"]), int(span["end"])
                total_spans += 1
                delta = end - start
                if min_delta is None or delta < min_delta:
                    min_delta = delta
                if start >= end:
                    n_start_ge_end += 1
                if start < 0 or end >= n:
                    n_oob += 1
                if end == n:
                    n_end_eq_len += 1
                if (
                    len(decode_samples) < 6
                    and span["label"] in {"SABCHE", "BOOKTITLE", "AUTHOR"}
                    and 10 <= delta <= 40
                ):
                    decode_samples.append((row["book_id"], span, text))

    print(
        f"spans={total_spans} min(end-start)={min_delta} "
        f"start>=end={n_start_ge_end} oob_inclusive(end>=len)={n_oob} "
        f"end==len={n_end_eq_len}"
    )
    if total_spans != 61396:
        raise GateFailure(f"expected 61396 spans, got {total_spans}")
    if min_delta != 1:
        raise GateFailure(f"expected min(end-start)=1, got {min_delta}")
    if n_start_ge_end or n_oob:
        raise GateFailure("degenerate or out-of-bounds spans present")

    print("offset convention probe (inclusive slice includes closing shad):")
    inclusive_votes = 0
    half_votes = 0
    for book_id, span, text in decode_samples:
        start, end = int(span["start"]), int(span["end"])
        half = text[start:end]
        incl = text[start : end + 1]
        incl_has_shad = incl.endswith("།") and not half.endswith("།")
        print(
            f"  {book_id} {span['label']} [{start},{end}] "
            f"half={half!r} incl={incl!r}"
        )
        if incl_has_shad:
            inclusive_votes += 1
        else:
            half_votes += 1
    print(
        f"convention votes: inclusive_shad={inclusive_votes} "
        f"otherwise={half_votes}; end==len(text) count={n_end_eq_len} "
        "(half-open-legal, inclusive-OOB)."
    )
    if n_end_eq_len:
        raise GateFailure(
            "found end==len(text); that is half-open, but align_window_labels "
            "is inclusive — would need conversion"
        )
    print(
        "Convention: INCLUSIVE [start, end], matching audit_utils.span_length "
        "and pick_winning_span (token_start < end+1). The task text said "
        "half-open; the Hub data and the alignment code agree on inclusive, "
        "so no conversion is applied. Geometry gates passed."
    )
    return {"split_ids": split_ids, "total_spans": total_spans}


def content_offsets(
    offsets: list[tuple[int, int]], specials: list[int]
) -> list[tuple[int, int]]:
    return [
        (int(start), int(end))
        for (start, end), is_special in zip(offsets, specials, strict=True)
        if not is_special and int(start) != int(end)
    ]


def overlap_token_count(
    prev_offsets: list[tuple[int, int]],
    prev_specials: list[int],
    curr_offsets: list[tuple[int, int]],
    curr_specials: list[int],
) -> int:
    """Count content tokens in the current window that also appear in the previous."""
    previous = set(content_offsets(prev_offsets, prev_specials))
    return sum(1 for tok in content_offsets(curr_offsets, curr_specials) if tok in previous)


def window_char_span(
    offsets: list[tuple[int, int]], specials: list[int]
) -> tuple[int, int] | None:
    content = content_offsets(offsets, specials)
    if not content:
        return None
    return content[0][0], content[-1][1]


def span_fully_inside(
    start: int, end: int, window_start: int, window_end_excl: int
) -> bool:
    """Inclusive gold span fully covered by a half-open window char range."""
    return window_start <= start and end < window_end_excl


def labelled_runs(labels: list[int]) -> list[tuple[int, int, int]]:
    """Return (label_id, tok_start, tok_end_inclusive) for non-O supervised runs."""
    runs: list[tuple[int, int, int]] = []
    start: int | None = None
    current_id = 0
    current_layer: str | None = None
    o_id = LABEL2ID["O"]
    for index, label in enumerate(labels):
        if label in (-100, o_id):
            if start is not None:
                runs.append((current_id, start, index - 1))
                start = None
                current_layer = None
            continue
        layer = ID2LABEL[int(label)].split("-", 1)[1]
        if start is not None and layer != current_layer:
            runs.append((current_id, start, index - 1))
            start = index
            current_id = int(label)
            current_layer = layer
        elif start is None:
            start = index
            current_id = int(label)
            current_layer = layer
    if start is not None:
        runs.append((current_id, start, len(labels) - 1))
    return runs


def verify_book(
    row: dict[str, Any],
    windows: dict[str, list[Any]],
    *,
    expected_overlap: int,
    max_length: int,
) -> dict[str, Any]:
    book_id = row["book_id"]
    text = row["text"]
    n_windows = len(windows["input_ids"])
    offsets_list = windows["offset_mapping"]
    specials_list = windows["special_tokens_mask"]

    covered: set[tuple[int, int]] = set()
    char_spans: list[tuple[int, int]] = []
    for offsets, specials in zip(offsets_list, specials_list, strict=True):
        content = content_offsets(offsets, specials)
        covered.update(content)
        span = window_char_span(offsets, specials)
        if span is None:
            raise GateFailure(f"{book_id}: window with no content tokens")
        char_spans.append(span)

    if char_spans[0][0] != 0:
        raise GateFailure(
            f"{book_id}: first window starts at char {char_spans[0][0]}, not 0"
        )
    merged_end = 0
    for w_start, w_end in sorted(char_spans):
        if w_start > merged_end:
            raise GateFailure(
                f"{book_id}: character gap in window union [{merged_end}, {w_start})"
            )
        merged_end = max(merged_end, w_end)
    if merged_end < len(text):
        raise GateFailure(
            f"{book_id}: window union ends at {merged_end}, text length {len(text)}"
        )

    ordered = sorted(covered)

    overlaps: list[int] = []
    for index in range(1, n_windows):
        prev_len = len(windows["input_ids"][index - 1])
        curr_len = len(windows["input_ids"][index])
        count = overlap_token_count(
            offsets_list[index - 1],
            specials_list[index - 1],
            offsets_list[index],
            specials_list[index],
        )
        overlaps.append(count)
        if prev_len == max_length and curr_len == max_length:
            if count < expected_overlap or count > expected_overlap + 1:
                raise GateFailure(
                    f"{book_id}: overlap between full windows {index - 1}/{index} "
                    f"is {count}, expected {expected_overlap} or {expected_overlap + 1}"
                )

    split_spans: list[dict[str, Any]] = []
    max_span_tokens = 0
    max_span_meta: dict[str, Any] | None = None
    token_starts = [tok_s for tok_s, _tok_e in ordered]
    token_ends = [tok_e for _tok_s, tok_e in ordered]
    for span in row["spans"]:
        start, end = int(span["start"]), int(span["end"])
        inside = any(
            span_fully_inside(start, end, w_start, w_end)
            for w_start, w_end in char_spans
        )
        if not inside:
            split_spans.append(
                {"book_id": book_id, "label": span["label"], "start": start, "end": end}
            )
        left = bisect.bisect_right(token_ends, start)
        right = bisect.bisect_left(token_starts, end + 1)
        n_tok = max(0, right - left)
        if n_tok > max_span_tokens:
            max_span_tokens = n_tok
            max_span_meta = {
                "book_id": book_id,
                "label": span["label"],
                "start": start,
                "end": end,
                "tokens": n_tok,
            }

    if n_windows == 1:
        full_tokens = len(windows["input_ids"][0])
    else:
        full_tokens = len(covered) + 2
    short = n_windows == 1 and full_tokens < max_length
    if short and n_windows != 1:
        raise GateFailure(
            f"{book_id}: short book ({full_tokens} tokens) produced {n_windows} windows"
        )
    padded = any(
        len(mask) != sum(mask) for mask in windows.get("attention_mask", [])
    ) if "attention_mask" in windows else False
    if padded:
        raise GateFailure(f"{book_id}: window appears padded (padding=False required)")

    return {
        "book_id": book_id,
        "n_windows": n_windows,
        "full_tokens": full_tokens,
        "n_chars": len(text),
        "short": short,
        "overlaps": overlaps,
        "n_split_spans": len(split_spans),
        "split_spans": split_spans,
        "max_span_tokens": max_span_tokens,
        "max_span_meta": max_span_meta,
        "fertility": full_tokens / len(text) if text else 0.0,
    }


def sample_decode_checks(
    row: dict[str, Any],
    windows: dict[str, list[Any]],
    tokenizer: Any,
    rng: random.Random,
    needed: int,
) -> list[str]:
    reports: list[str] = []
    n = len(windows["input_ids"])
    order = list(range(n))
    rng.shuffle(order)
    for window_i in order:
        if len(reports) >= needed:
            break
        labels = windows["labels"][window_i]
        runs = labelled_runs(labels)
        if not runs:
            continue
        offsets = windows["offset_mapping"][window_i]
        specials = windows["special_tokens_mask"][window_i]
        input_ids = windows["input_ids"][window_i]
        label_id, tok_start, tok_end = runs[0]
        char_start = None
        char_end = None
        for tok_i in range(tok_start, tok_end + 1):
            if specials[tok_i]:
                continue
            s, e = int(offsets[tok_i][0]), int(offsets[tok_i][1])
            if s == e:
                continue
            if char_start is None:
                char_start = s
            char_end = e
        if char_start is None or char_end is None:
            continue
        gold = row["text"][char_start:char_end]
        decoded = tokenizer.decode(
            input_ids[tok_start : tok_end + 1], skip_special_tokens=True
        )
        tag = ID2LABEL[label_id]
        ok = gold == decoded or gold.replace(" ", "") == decoded.replace(" ", "")
        line = (
            f"{row['book_id']} w{window_i} {tag} chars[{char_start}:{char_end}] "
            f"gold={gold!r} decoded={decoded!r} match={ok}"
        )
        if not ok:
            raise GateFailure(f"decode mismatch: {line}")
        reports.append(line)
    return reports


def build_dataset_card(
    cfg: dict[str, Any],
    *,
    source: dict[str, str],
    tokenizer: Any,
    window_counts: dict[str, int],
    stats: dict[str, Any],
) -> str:
    id2label_rows = "\n".join(
        f"| {index} | `{name}` |" for index, name in ID2LABEL.items()
    )
    hashes = getattr(tokenizer, "_qd_file_hashes", {})
    hash_rows = "\n".join(f"| `{name}` | `{digest}` |" for name, digest in hashes.items())
    combined = getattr(tokenizer, "_qd_file_hash", "undetermined")
    overlap = int(cfg["expected_overlap_tokens"])
    max_length = int(cfg["max_length"])
    step = int(cfg["step"])
    percent = overlap / max_length * 100
    damaged = ", ".join(cfg.get("damaged_books") or DEFAULT_DAMAGED_BOOKS)
    zeros = stats.get("loss_mask_zeros") or {}
    verify_table = stats.get("verify_table") or ""
    return f"""\
---
language:
- bo
license: other
task_categories:
- token-classification
pretty_name: Tibetan Layer Detection Windows (w8192 s4916)
tags:
- tibetan
- openpecha
- token-classification
- sliding-window
- derived
size_categories:
- 1K<n<10K
---

# WARNING — derived data

This dataset is **derived** from a specific tokenizer, window length, and
stride. **Regenerate it** if any of those change. Do not mix windows from
different tokenizers or strides in the same training run.

Load the named config, never the unpinned default (that is still book-level):

```
load_dataset("karma689/layer_detection", "windowed_w8192_s4916", revision="v2.0")
```

`revision="v1.1"` (default config) stays book texts + character spans.
Tag `v1.1` is not moved by this publish.

## Schedule warning

v1.3 trains on **14,338** windows/epoch (HF stride 5120, step 3072, 62.5%
overlap). This config is step 4916 (~40% overlap) and about **9,060**
windows/epoch — **not 9,103** (false 1843-step model) and **not 8,906**
(naive `14338 × 3072/4916`, which also scales the 266 unshrinkable first
windows). Relative to 14,338 that is ~37% fewer windows (extras drop 37.5%;
the 266 first windows stay). **Warmup and the LR schedule must be rescaled**
or the same nominal epoch count undertrains.

## Provenance

| Field | Value |
|---|---|
| Source dataset | `{cfg["source_repo"]}` |
| Source revision | `{cfg["source_revision"]}` |
| Source tag commit | `{source.get("tag_commit", "undetermined")}` |
| Source data commit | `{source.get("data_commit", "undetermined")}` |
| Tokenizer repo (files used) | `{cfg["tokenizer_repo"]}` |
| Upstream tokenizer (v1.3 training config) | `{cfg.get("upstream_tokenizer", "")}` |
| Vocab size | `{len(tokenizer)}` |
| tokenizer.is_fast | `{tokenizer.is_fast}` |
| Combined tokenizer-file sha256 | `{combined}` |
| Window length (`max_length`) | `{max_length}` |
| HF tokenizer `stride` (overlap) | `{overlap}` |
| Step (CLI `--stride`) | `{step}` |
| Overlap | `{overlap}` tokens ({percent:.2f}%) |
| Label scheme | 15-class BIO softmax |
| Gold spans split across windows | **0** |
| Damaged books (O-mask on `loss_mask`) | `{damaged}` |
| Measured max span length | `{stats["max_span_tokens"]}` tokens (`{stats["max_span_meta"]}`) |

{hash_rows and "## Tokenizer file hashes" or ""}

{hash_rows}

Hub name `s4916` is the **step**, not the HuggingFace `stride` argument.
`tokenizer(..., max_length=8192, stride=3276)` yields step 4916.

The longest gold span in this corpus is
`{stats["max_span_tokens"]}` tokens
({stats["max_span_meta"]}). A span fits whole in at least one window iff its
length is ≤ overlap, so the zero-split floor is step ≤ `{max_length}` − max
span. Step 4916 clears that floor with margin. Do not increase the step past
that floor.

## Window counts

| Split | Books | Windows |
|---|---:|---:|
| train | 217 | {window_counts.get("train", 0)} |
| validation | 25 | {window_counts.get("validation", 0)} |
| test | 24 | {window_counts.get("test", 0)} |
| **total** | **266** | **{sum(window_counts.values())}** |

Short books (fewer than `{max_length}` tokens including specials):
{stats["n_short_books"]} books, each exactly one window. Shortest:
{stats["shortest_tokens"]} tokens (`{stats["shortest_book"]}`).

`scope_mask` uses the same confirmed-scope policy as v1.3 training
(`apply_confirmed_scope`): P000010 excludes Quotation, P000100 excludes Tsawa.

Offset convention of the source spans is **inclusive** `[start, end]`
(`text[start:end+1]`). Alignment uses `layer_detection.training_utils.align_window_labels`.
Do not convert to half-open — that would change BIO and the v2 token indices.

`masked_layers` is empty here. BookTitle gold stays in `labels` as `B-BOOKTITLE`
/ `I-BOOKTITLE`. The same positives v1.2/v1.3 ignore are zeroed on **`loss_mask`
only**.

Overlapped tokens are labelled `-100` in every window except the first that
covers them (`already_covered_through` in `align_window_labels`), so each
content token contributes to the BIO loss **once**.

## 15-label id → name (v1.3)

| id | label |
|---:|---|
{id2label_rows}

## Columns

Each row is one window:

| Column | Description |
|---|---|
| `book_id` | OpenPecha id |
| `window_index` | 0-based window within the book |
| `input_ids` / `attention_mask` | mmBERT window (no padding) |
| `labels` | 15-class BIO, winner-takes-all (Quotation wins). Unmasked BookTitle. |
| `scope_mask` | Per-label in-scope mask |
| `offset_mapping` / `special_tokens_mask` | For char ↔ token maps |
| `spans` | Every overlapping gold span (not collapsed): `layer` (Hub string), half-open window token `start`/`end`, `truncated`, `span_id`, `char_start`/`char_end` |
| `start_targets` / `end_targets` | `[7, T]` float, `MODEL_LAYERS` order, smoothing 1.0/0.3/0.1 with `max()` |
| `loss_mask` | `[T]` float; 0 on specials/pad, BookTitle BIO tokens, and damaged-book tokens outside every gold span |

`span_id` is `{{book_id}}:{{label}}:{{char_start}}:{{char_end}}:{{source_index}}`
and is stable across windows that cover the same gold span.

`loss_mask == 0` counts from this build: special/pad={zeros.get("special_pad", "?")},
booktitle={zeros.get("booktitle", "?")}, damaged_O={zeros.get("damaged_o", "?")}.

## Verification

{verify_table}
"""


WINDOWED_CONFIG_NAME = "windowed_w8192_s4916"
HUB_TAG = "v2.0"


DEFAULT_BOOK_CONFIG = {
    "config_name": "default",
    "data_files": [
        {"split": "train", "path": "data/train-*"},
        {"split": "validation", "path": "data/validation-*"},
        {"split": "test", "path": "data/test-*"},
    ],
}


def ensure_windowed_config_yaml(readme: str, config_name: str) -> str:
    """Append the named config to README YAML; never drop or rename default.

    Adding any ``configs:`` entry disables Hub auto-detection of ``data/``.
    If a default book-level config is missing, insert it first.
    """
    if readme.startswith("---"):
        _, front, body = readme.split("---", 2)
    else:
        front, body = "", "\n" + readme
    meta = yaml.safe_load(front) or {}
    if not isinstance(meta, dict):
        meta = {}
    configs = list(meta.get("configs") or [])
    has_default = any(
        isinstance(item, dict) and item.get("config_name") == "default"
        for item in configs
    )
    if not has_default:
        configs.insert(0, dict(DEFAULT_BOOK_CONFIG))
    entry = {
        "config_name": config_name,
        "data_files": [
            {"split": "train", "path": f"{config_name}/train/*.parquet"},
            {"split": "validation", "path": f"{config_name}/validation/*.parquet"},
            {"split": "test", "path": f"{config_name}/test/*.parquet"},
        ],
    }
    replaced = False
    for index, item in enumerate(configs):
        if isinstance(item, dict) and item.get("config_name") == config_name:
            configs[index] = entry
            replaced = True
            break
    if not replaced:
        configs.append(entry)
    meta["configs"] = configs
    dumped = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
    return f"---\n{dumped}---{body}"


def resolve_job_windowing(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[int, int, int]:
    window_len = int(args.window_len or cfg.get("max_length") or DEFAULT_WINDOW_LEN)
    step = int(args.stride or cfg.get("step") or DEFAULT_STEP)
    try:
        window_len, step, hf_stride = resolve_windowing(window_len, step)
    except ValueError as exc:
        raise GateFailure(str(exc)) from exc
    cfg["max_length"] = window_len
    cfg["step"] = step
    cfg["hf_stride"] = hf_stride
    cfg["stride"] = hf_stride  # HF overlap passed to tokenize_book_windows
    cfg["expected_overlap_tokens"] = hf_stride
    return window_len, step, hf_stride


def push_named_windowed_config(
    dataset_dict: Any,
    cfg: dict[str, Any],
    card: str,
    *,
    source: dict[str, str],
) -> None:
    """Push config ``windowed_w8192_s4916``; never overwrite default or tag v1.1."""
    hub_repo = str(cfg.get("hub_repo") or "karma689/layer_detection")
    config_name = str(cfg.get("hub_config_name") or WINDOWED_CONFIG_NAME)
    tag_name = str(cfg.get("hub_tag") or HUB_TAG)
    if not config_name or config_name in {"default", "data"}:
        raise GateFailure(
            "refusing push_to_hub without a real config_name "
            "(would overwrite the default book-level dataset)"
        )
    if hub_repo.endswith("_windowed_w8192_s4916"):
        raise GateFailure(
            f"hub_repo {hub_repo} looks like a separate windowed repo; "
            "publish as named config on karma689/layer_detection"
        )

    api = HfApi()
    refs = api.list_repo_refs(hub_repo, repo_type="dataset")
    tags = {tag.name: tag.target_commit for tag in refs.tags}
    branches = {branch.name: branch.target_commit for branch in refs.branches}
    v11 = tags.get("v1.1") or source.get("tag_commit")
    if not v11:
        raise GateFailure("cannot locate tag v1.1; refusing to push")
    print(f"pre-push: main={branches.get('main')} v1.1={v11} existing_tags={sorted(tags)}")

    from datasets import load_dataset

    default_ds = load_dataset(hub_repo)
    if "text" not in default_ds["train"].column_names:
        raise GateFailure("default config is not book-level; refusing to push")
    if len(default_ds["train"]) != 217:
        raise GateFailure(
            f"default train has {len(default_ds['train'])} rows, expected 217 books"
        )
    print("pre-push: default config is book-level 217/25/24")

    local_dir = Path(cfg.get("_local_dir") or ROOT / str(cfg.get("local_dir") or ""))
    shard_root = local_dir / "shards"
    if not shard_root.is_dir():
        raise GateFailure(f"no shards at {shard_root}")
    print(f"upload_folder shards -> {hub_repo}/{config_name} (not default, not v1.1)")
    commit = api.upload_folder(
        folder_path=str(shard_root),
        path_in_repo=config_name,
        repo_id=hub_repo,
        repo_type="dataset",
        allow_patterns="*.parquet",
        commit_message=(
            f"Add named config {config_name} (w8192 step 4916 v2 fields). "
            "Default book-level config and tag v1.1 unchanged."
        ),
    )
    new_commit = getattr(commit, "oid", None) or getattr(commit, "commit_id", None)

    refs2 = api.list_repo_refs(hub_repo, repo_type="dataset")
    tags2 = {tag.name: tag.target_commit for tag in refs2.tags}
    if tags2.get("v1.1") != v11:
        raise GateFailure(
            f"v1.1 tag moved from {v11} to {tags2.get('v1.1')} — this is forbidden"
        )

    try:
        readme_path = hf_hub_download(hub_repo, "README.md", repo_type="dataset")
        existing = Path(readme_path).read_text(encoding="utf-8")
    except Exception:
        try:
            existing = DatasetCard.load(hub_repo).text
        except Exception:
            existing = ""
    existing = ensure_windowed_config_yaml(existing, config_name)
    marker = "<!-- windowed-v2-card -->"
    body = card.split("---", 2)[-1] if card.startswith("---") else card
    section = f"{marker}\n# Config `{config_name}` (tag {tag_name})\n{body}\n"
    if marker in existing:
        start = existing.index(marker)
        existing = existing[:start] + section
    else:
        existing = existing.rstrip() + "\n\n" + section
    # Raw upload: DatasetCard.push_to_hub rewrites YAML and can drop the
    # default book-level config plus language/license metadata.
    api.upload_file(
        path_or_fileobj=existing.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=hub_repo,
        repo_type="dataset",
        commit_message=(
            f"Document named config {config_name}; keep default book-level data/"
        ),
    )

    refs3 = api.list_repo_refs(hub_repo, repo_type="dataset")
    tags3 = {tag.name: tag.target_commit for tag in refs3.tags}
    branches3 = {branch.name: branch.target_commit for branch in refs3.branches}
    if tags3.get("v1.1") != v11:
        raise GateFailure("v1.1 tag moved while writing the dataset card")
    card_commit = branches3.get("main")
    if tag_name in tags3 and card_commit and tags3[tag_name] != card_commit:
        raise GateFailure(
            f"tag {tag_name} already points at {tags3[tag_name]}, not this commit"
        )
    if tag_name not in tags3:
        revision = card_commit or new_commit
        api.create_tag(
            hub_repo,
            tag=tag_name,
            revision=revision,
            repo_type="dataset",
            tag_message=f"{config_name} windows + v2 supervision fields",
        )
        print(f"tagged {tag_name} -> {revision}")
    else:
        print(f"tag {tag_name} already at {tags3[tag_name]}")

    default_after = load_dataset(hub_repo)
    if "text" not in default_after["train"].column_names:
        raise GateFailure("default config lost book-level `text` after push")
    if len(default_after["train"]) != 217:
        raise GateFailure(
            f"default train has {len(default_after['train'])} rows, expected 217 books"
        )
    v11_after = load_dataset(hub_repo, revision="v1.1")
    if "text" not in v11_after["train"].column_names:
        raise GateFailure("revision v1.1 is no longer book-level")
    windowed = load_dataset(hub_repo, config_name, revision=tag_name, streaming=True)
    row = next(iter(windowed["train"]))
    if "start_targets" not in row:
        raise GateFailure(f"{config_name}@{tag_name} missing v2 columns")
    print(
        f"post-push: default books={len(default_after['train'])} "
        f"v1.1 books={len(v11_after['train'])} "
        f"{config_name} stream has start_targets"
    )
    print(f"Pushed https://huggingface.co/datasets/{hub_repo} config={config_name} tag={tag_name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--window-len",
        type=int,
        default=None,
        help=f"window length (default {DEFAULT_WINDOW_LEN} or yaml max_length)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            f"STEP / hop between windows, not HF overlap "
            f"(default {DEFAULT_STEP}). hf_stride = window_len - stride"
        ),
    )
    parser.add_argument(
        "--damaged-books",
        default=None,
        help="comma-separated book ids whose O tokens are loss-masked",
    )
    parser.add_argument(
        "--smoothing",
        default=None,
        help="center,±1,±2 weights (default 1.0,0.3,0.1)",
    )
    parser.add_argument(
        "--min-span-chars",
        type=int,
        default=None,
        help=f"inclusive-length filter (default {DEFAULT_MIN_SPAN_CHARS})",
    )
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument(
        "--push-only",
        action="store_true",
        help="Push an already-built local DatasetDict; skip tokenization",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Load the local DatasetDict and run v2 gates; do not push",
    )
    parser.add_argument("--limit-books", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    cfg = load_job_config(args.config)
    max_length, step, hf_stride = resolve_job_windowing(cfg, args)
    expected_overlap = hf_stride
    damaged_books = parse_damaged_books(
        args.damaged_books or cfg.get("damaged_books") or DEFAULT_DAMAGED_BOOKS
    )
    smoothing = parse_smoothing(
        args.smoothing if args.smoothing is not None else cfg.get("smoothing") or DEFAULT_SMOOTHING
    )
    min_span_chars = int(
        args.min_span_chars
        if args.min_span_chars is not None
        else cfg.get("min_span_chars") or DEFAULT_MIN_SPAN_CHARS
    )
    cfg["damaged_books"] = list(damaged_books)
    cfg["expected_total_windows"] = int(cfg.get("expected_total_windows") or TARGET_WINDOWS)
    if int(cfg["expected_total_windows"]) in {WRONG_MODEL_WINDOWS, NAIVE_SCALE_WINDOWS}:
        raise GateFailure(
            f"expected_total_windows={cfg['expected_total_windows']} is a known-wrong "
            f"formula; use {TARGET_WINDOWS}"
        )
    report_job_facts(window_len=max_length, step=step, hf_stride=hf_stride)
    print(
        f"JOB WINDOWING: max_length={max_length} HF_stride(overlap)={hf_stride} "
        f"step={step} overlap_pct={hf_stride / max_length * 100:.2f}% "
        f"damaged={list(damaged_books)} smoothing={smoothing} "
        f"min_span_chars={min_span_chars}"
    )
    local_dir = ROOT / str(cfg["local_dir"])
    cfg["_local_dir"] = str(local_dir)

    if args.push_only or args.verify_only:
        from datasets import load_dataset

        shard_root = local_dir / "shards"
        if not shard_root.is_dir():
            raise GateFailure(f"no saved shards at {shard_root}")
        tokenizer = assert_tokenizer(cfg)
        source = resolve_source_commits(cfg["source_repo"], cfg["source_revision"])
        print(f"Streaming local parquet shards from {shard_root}")
        shard_iters = {
            name: iter_parquet_rows(sorted((shard_root / name).glob("*.parquet")))
            for name in ("train", "validation", "test")
            if (shard_root / name).is_dir()
        }
        print({name: len(list((shard_root / name).glob("*.parquet"))) for name in shard_iters})
        print(f"Loading source {cfg['source_repo']} revision={cfg['source_revision']!r} for gates")
        source_ds = load_dataset(cfg["source_repo"], revision=cfg["source_revision"])
        source_rows = [dict(row) for split in source_ds.values() for row in split]
        stats_path = local_dir / "window_stats.json"
        token_lengths = []
        if stats_path.is_file():
            saved = json.loads(stats_path.read_text(encoding="utf-8"))
            token_lengths = list(saved.get("token_lengths") or [])
        if not token_lengths:
            import pyarrow.parquet as pq

            for name in ("train", "validation", "test"):
                for path in sorted((shard_root / name).glob("*.parquet")):
                    n = int(pq.read_metadata(path).num_rows)
                    token_lengths.append(
                        max_length if n <= 1 else max_length + (n - 1) * step
                    )
        report = verify_windowed_v2(
            shard_iters,
            source_rows,
            token_lengths=token_lengths,
            window_len=max_length,
            step=step,
            expected_windows=int(cfg["expected_total_windows"]),
            tolerance=float(cfg.get("window_count_tolerance") or 0.03),
            min_span_chars=min_span_chars,
            limit_books=bool(args.limit_books),
        )
        print(report["table_text"])
        for line in report.get("round_trip") or []:
            print(f"  {line}")
        if not report["ok"]:
            raise GateFailure("; ".join(report["failures"]))
        if args.verify_only or args.no_push:
            print("verification passed; not uploading.")
            return 0
        card_path = local_dir / "README.md"
        if not card_path.is_file():
            raise GateFailure(f"missing data card {card_path}")
        push_named_windowed_config(
            None, cfg, card_path.read_text(encoding="utf-8"), source=source
        )
        return 0

    tokenizer = assert_tokenizer(cfg)
    source = resolve_source_commits(cfg["source_repo"], cfg["source_revision"])

    from datasets import Dataset, DatasetDict, load_dataset

    print(f"\nLoading {cfg['source_repo']} revision={cfg['source_revision']!r}")
    ds = load_dataset(cfg["source_repo"], revision=cfg["source_revision"])
    print("cache files:", [item["filename"][-90:] for item in ds["train"].cache_files[:1]])
    audit = audit_source(ds)
    split_ids: dict[str, list[str]] = audit["split_ids"]

    print(
        "Fast tokenizer is not thread-safe for concurrent encode "
        "(RuntimeError: Already borrowed). Forcing 1 worker."
    )
    workers = 1
    masked = tuple(cfg.get("masked_layers") or ())
    local_dir = ROOT / str(cfg["local_dir"])
    decode_reports: list[str] = []
    book_stats: list[dict[str, Any]] = []
    window_counts: dict[str, int] = {}
    split_datasets = {}
    all_split_spans: list[dict[str, Any]] = []

    def process_book(row: dict[str, Any]) -> tuple[dict[str, list[Any]], dict[str, Any], list[str]]:
        windows = tokenize_book_windows(
            row,
            tokenizer,
            max_length,
            hf_stride,
            masked_layers=masked,
            include_metadata=True,
        )
        stats = verify_book(
            row,
            windows,
            expected_overlap=expected_overlap,
            max_length=max_length,
        )
        try:
            windows, v2_stats = attach_v2_supervision(
                windows,
                row,
                damaged_books=damaged_books,
                smoothing=smoothing,
                min_span_chars=min_span_chars,
            )
        except VerificationError as exc:
            raise GateFailure(str(exc)) from exc
        stats["v2"] = v2_stats
        samples = sample_decode_checks(
            row,
            windows,
            tokenizer,
            random.Random(hash(str(row.get("book_id", ""))) & 0xFFFFFFFF),
            needed=2,
        )
        return windows, stats, samples

    hub_splits = ("train", "validation", "test")
    for split_name in hub_splits:
        rows = [dict(row) for row in ds[split_name]]
        repaired = apply_confirmed_scope(rows)
        if repaired:
            print(
                f"{split_name}: repaired in_scope_layers on {repaired} books "
                "(confirmed-scope policy, same as v1.3 train.py)"
            )
        if args.limit_books:
            rows = rows[: args.limit_books]
        expected_ids = [row["book_id"] for row in rows]
        print(f"\n--- windowing {split_name}: {len(rows)} books, {workers} worker(s) ---")

        shard_dir = local_dir / "shards" / split_name
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_files: list[Path] = []
        produced_ids: list[str] = []
        n_windows = 0
        for index, row in enumerate(rows):
            path = shard_path(shard_dir, index, row["book_id"])
            if path.is_file() and path.stat().st_size > 0:
                stats = resume_book_stats(
                    path, row, max_length=max_length, step=step
                )
                print(
                    f"[{split_name} {index + 1}/{len(rows)}] {row['book_id']} "
                    f"resumed windows={stats['n_windows']} tokens={stats['full_tokens']}",
                    flush=True,
                )
            else:
                windows, stats, samples = process_book(row)
                print(
                    f"[{split_name} {index + 1}/{len(rows)}] {row['book_id']} "
                    f"windows={stats['n_windows']} tokens={stats['full_tokens']}",
                    flush=True,
                )
                Dataset.from_dict(windows).to_parquet(str(path))
                path.with_suffix(".stats.json").write_text(
                    json.dumps(stats, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
                if len(decode_reports) < 5:
                    decode_reports.extend(samples[: 5 - len(decode_reports)])
                del windows
            produced_ids.append(row["book_id"])
            n_windows += stats["n_windows"]
            book_stats.append(stats)
            all_split_spans.extend(stats.get("split_spans") or [])
            shard_files.append(path)
        if produced_ids != expected_ids:
            raise GateFailure(f"{split_name}: book order changed after windowing")
        if not args.limit_books and produced_ids != split_ids[split_name]:
            raise GateFailure(f"{split_name}: book membership changed vs source v1.1")
        window_counts[split_name] = n_windows
        print(f"{split_name}: {n_windows} windows")
        split_datasets[split_name] = iter_parquet_rows(shard_files)

    if all_split_spans:
        preview = all_split_spans[:8]
        raise GateFailure(
            f"{len(all_split_spans)} gold spans are not fully inside any window "
            f"(expected 0). First: {preview}"
        )
    print("span-split count: 0")

    total_windows = sum(window_counts.values())
    expected_windows = int(cfg["expected_total_windows"])
    tolerance = float(cfg["window_count_tolerance"])
    delta = abs(total_windows - expected_windows) / expected_windows
    print(
        f"total windows={total_windows} expected≈{expected_windows} "
        f"rel_delta={delta:.3f}"
    )
    if not args.limit_books and delta > tolerance:
        raise GateFailure(
            f"window count {total_windows} is more than {tolerance:.0%} off "
            f"predicted {expected_windows}"
        )

    short_stats = [s for s in book_stats if s["short"]]
    print(f"short books (<={max_length} tokens): {len(short_stats)}")
    if short_stats:
        shortest = min(short_stats, key=lambda s: s["full_tokens"])
        print(
            f"  shortest: {shortest['book_id']} {shortest['full_tokens']} tokens, "
            f"{shortest['n_windows']} window(s)"
        )
        bad_short = [s for s in short_stats if s["n_windows"] != 1]
        if bad_short:
            raise GateFailure(f"short books with !=1 window: {bad_short}")
    else:
        shortest = {"book_id": "", "full_tokens": 0}

    full_overlap_pairs = [
        ov
        for s in book_stats
        for ov in s["overlaps"]
    ]
    # Pairs between two full-size windows were already asserted == expected.
    print(
        f"consecutive-window overlap samples: n={len(full_overlap_pairs)} "
        f"counter={Counter(full_overlap_pairs).most_common(6)}"
    )

    measured_spans = [s for s in book_stats if s.get("max_span_tokens")]
    max_span = max(
        measured_spans or book_stats,
        key=lambda s: s.get("max_span_tokens") or 0,
    )
    if not max_span.get("max_span_tokens"):
        max_span = {
            "max_span_tokens": int(cfg.get("expected_max_span_tokens") or 3125),
            "max_span_meta": "expected_max_span_tokens (train shards resumed)",
        }
    print(f"measured max span tokens: {max_span['max_span_tokens']} {max_span['max_span_meta']}")
    fert = sum(s["full_tokens"] for s in book_stats) / sum(s["n_chars"] for s in book_stats)
    print(f"corpus fertility (tokens/char, including 2 specials/book): {fert:.6f}")

    print("decode checks:")
    for line in decode_reports:
        print(f"  {line}")
    if len(decode_reports) < 5 and not args.limit_books:
        raise GateFailure(f"only {len(decode_reports)} decode samples (need 5)")

    v2_zeros = {"special_pad": 0, "booktitle": 0, "damaged_o": 0}
    source_rows: list[dict[str, Any]] = []
    for split_name in hub_splits:
        source_rows.extend(dict(row) for row in ds[split_name])
    token_lengths = [int(item["full_tokens"]) for item in book_stats]
    for item in book_stats:
        zeros = (item.get("v2") or {}).get("loss_mask_zeros") or {}
        for key in v2_zeros:
            v2_zeros[key] += int(zeros.get(key, 0))

    print("\n" + "=" * 72)
    print("V2 VERIFICATION")
    print("=" * 72)
    v2_report = verify_windowed_v2(
        split_datasets,
        source_rows,
        token_lengths=token_lengths,
        window_len=max_length,
        step=step,
        expected_windows=expected_windows,
        tolerance=tolerance,
        min_span_chars=min_span_chars,
        limit_books=bool(args.limit_books),
    )
    print(v2_report["table_text"])
    for line in v2_report.get("round_trip") or []:
        print(f"  {line}")
    if not v2_report["ok"]:
        raise GateFailure("; ".join(v2_report["failures"]))

    print("\n" + "=" * 72)
    print("LOSS PATH (report only — not changed)")
    print("=" * 72)
    print(
        "Overlapped tokens contribute to the loss ONCE, not once per window.\n"
        "1. training_utils.align_window_labels: "
        "`labels.append(-100 if token_end <= already_covered_through else label_id)`\n"
        "2. tokenize_book_windows carries `covered_through` across windows of "
        "the same book.\n"
        "3. experiments/mmbert_base_v1.3/train.py WeightedTrainer.compute_loss: "
        "CrossEntropyLoss(ignore_index=-100).\n"
        "4. class_weights_from_dataset skips label == -100.\n"
        "Changing stride therefore does not re-open class weighting via "
        "double-counted overlap, as long as this path stays in use."
    )

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nShards already on disk under {local_dir / 'shards'} (no second DatasetDict copy)")
    total_bytes = sum(path.stat().st_size for path in local_dir.rglob("*") if path.is_file())
    print(f"on-disk size: {total_bytes} bytes ({total_bytes / (1024**3):.3f} GiB)")

    card_stats = {
        "max_span_tokens": max_span["max_span_tokens"],
        "max_span_meta": max_span["max_span_meta"],
        "n_short_books": len(short_stats),
        "shortest_tokens": shortest["full_tokens"],
        "shortest_book": shortest["book_id"],
        "fertility": fert,
        "total_windows": total_windows,
        "on_disk_bytes": total_bytes,
        "loss_mask_zeros": v2_zeros,
        "verify_table": v2_report["table_text"],
    }
    card = build_dataset_card(
        cfg,
        source=source,
        tokenizer=tokenizer,
        window_counts=window_counts,
        stats=card_stats,
    )
    (local_dir / "README.md").write_text(card, encoding="utf-8")
    (local_dir / "window_stats.json").write_text(
        json.dumps(
            {
                "source": source,
                "window_counts": window_counts,
                "card_stats": card_stats,
                "tokenizer_hash": getattr(tokenizer, "_qd_file_hash", None),
                "tokenizer_hashes": getattr(tokenizer, "_qd_file_hashes", None),
                "token_lengths": token_lengths,
                "v2_verification": {
                    "ok": v2_report["ok"],
                    "failures": v2_report["failures"],
                    "loss_mask_zeros": v2_report["loss_mask_zeros"],
                    "measured_windows": v2_report["measured_windows"],
                    "predicted_windows": v2_report["predicted_windows"],
                },
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    if args.no_push:
        print("`--no-push`: not uploading.")
        return 0

    cfg["_local_dir"] = str(local_dir)
    push_named_windowed_config(None, cfg, card, source=source)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateFailure as exc:
        print(f"GATE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
