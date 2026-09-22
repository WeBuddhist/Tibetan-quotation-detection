# Zero-shot quotation detection (Gemini + Claude)

No training. A frozen prompt asks Gemini or Claude to pull quotations out of a
page; a locator maps the returned text back to character offsets; the same
IoU@0.5 matcher used for the encoder scores the spans.

Prompt: [`docs/prompts/gemini_quote_only_anchors_v2.md`](prompts/gemini_quote_only_anchors_v2.md)
(sha1 `e88f021573bb97fd3733282caa8b4c966501bf9f`). Frozen. Every model gets
exactly the same filled text.

This page covers how the method works, how to run a single book or the full
benchmark, and the numbers we have. Commands are also in the root
[README](../README.md). Encoder results (not directly comparable — see
[Scoring](#scoring)) are in [encoder-model.md](encoder-model.md).

## How it works

1. **Extract.** The model returns JSON anchors, not offsets:
   `{spans:[{label, frame, head, tail}]}`. `head` / `tail` are short unique
   strings at each end of the quote. Passages longer than 1,200 characters are
   split; the scorer stitches adjacent pieces.
2. **Locate.** [`src/layer_detection/gemini_locate.py`](../src/layer_detection/gemini_locate.py)
   walks the book with a forward cursor (`--max-len 2000`) and finds each
   head/tail pair. Offsets are inclusive: `text[start:end+1]`.
3. **Score.** Greedy one-to-one IoU@0.5 (`match_iou` in
   [`src/layer_detection/v2_metrics.py`](../src/layer_detection/v2_metrics.py)).
   A prediction is a hit only if it shares a gold Quotation span at inclusive
   IoU ≥ 0.5. Adjacent predicted spans whose gap is only shad, tsheg, or spaces
   are joined (`stitch`); headline numbers report stitch on and stitch off.

Gold for the benchmark is the book-level Hub export
(`Data/hf_export/test.jsonl` / `validation.jsonl` from
`karma689/tibetan-quotation-detection@w2.0`), not the raw OpenPecha
`Quotation.yml`. Eleven of the 19 test books store the layer as `Citation`;
reading the OPF file would give them zero gold.

## Models and limits

Copy your live AI Studio / Claude Console limits into
[`experiments/gemini_anchor_v2_test/limits.yaml`](../experiments/gemini_anchor_v2_test/limits.yaml)
before a run. The values below were current on 2026-09-11.

| Arm | Model id | How it runs | Limits that matter |
|---|---|---|---|
| G1 | `gemini-3.8-flash` | Gemini API, one request at a time | 5 RPM · 250K tokens/min · 20 req/day |
| G2 | `gemini-3.1-flash-lite` | Gemini API, one request at a time | 15 RPM · 250K tokens/min · 500 req/day |
| C | `claude-sonnet-5` | Anthropic Message Batches API | 1M-token context · 128K max output |

Shared settings (where the model allows it): temperature 1.0 (Gemini 3
default; Sonnet 5 rejects any other value), thinking low, JSON schema above.
Keys come from the environment only — never written to a file:

```bash
export GEMINI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

Gemini's daily reset is midnight Pacific (12:30 IST until 1 November 2026,
then 13:30 IST). G1 and G2 have separate quotas, so they can run as two
processes at once.

## Two ways to run

### A. One book (or a plain-text file)

Use this to try a prompt or debug a single title. It is **not** the scored
benchmark path — gold comes from the local OpenPecha layer, and whole-book
requests can blow the per-minute token cap.

```bash
# extract (default prompt: docs/prompts/gemini_quote_only.md)
python script/gemini_extract.py --book P000152
python script/gemini_extract.py --book P000152 \
    --prompt docs/prompts/gemini_quote_only_anchors_v2.md \
    --chunked --out quotes.json

# from a text file instead of Data/repos
python script/gemini_extract.py --text path/to/book.txt --chunked

# locate + score against the local OpenPecha Quotation layer
python script/check_gemini_quotes.py \
    --book P000152 \
    --quotes experiments/gemini_flash_zeroshot/p000152_gemini_anchors.json \
    --out experiments/gemini_flash_zeroshot/p000152_anchor_check.json
```

`--chunked` splits the book into overlapping windows (W/S from
`experiments/gemini_flash_zeroshot/config.yaml`). Without it the whole book
goes in one request. On the free tier a long book can exceed the per-minute
cap on its own; prefer `--chunked` or the benchmark units below.

A thinner wrapper that walks a Hub split, extracts, locates, and writes
metrics:

```bash
python script/gemini_eval.py --split test
python script/gemini_eval.py --split test --books P000069
```

That script still uses the older extract path. For a number you will quote
against the encoder, use path B.

### B. Full three-model benchmark (recommended)

Harness: [`script/run_gemini_benchmark.py`](../script/run_gemini_benchmark.py)
→ [`src/layer_detection/gemini_benchmark.py`](../src/layer_detection/gemini_benchmark.py).
Artifacts land in `experiments/gemini_anchor_v2_test/`.

**Units.** A book of 250,000 characters or less is one request. A longer book
is cut into `ceil(length / 250,000)` near-equal pieces at a shad, with a
2,000-character overlap. Test is 26 units (19 books); validation is 32 units
(25 books). That is about two days of G1 quota instead of ~19 days at
16,000-character windows.

**Windows (optional comparison).** `--source windows` uses 16,000 / 2,000
character windows (333 test requests, 334 validation). Write those responses
under `<model>/raw_windows/` — never under `<model>/raw/`, or you overwrite
the finished unit run.

```bash
# 1. Build the request list (once)
python script/run_gemini_benchmark.py build-units
python script/run_gemini_benchmark.py build-windows   # only if you want the 16k comparison

# 2. Token-count every unit and refuse anything over the cap
python script/run_gemini_benchmark.py preflight --source units
python script/run_gemini_benchmark.py preflight --source units --model gemini-3.8-flash

# 3. Freeze the prompt + unit hashes
python script/run_gemini_benchmark.py freeze

# 4a. Gemini — one process per model; they share no quota
python script/run_gemini_benchmark.py run --model gemini-3.8-flash --split test --source units
python script/run_gemini_benchmark.py run --model gemini-3.1-flash-lite --split test --source units

# 4b. Claude — Message Batches API
python script/run_gemini_benchmark.py claude-batch --action create --split test
python script/run_gemini_benchmark.py claude-batch --action poll --split test
# if the process died after create, reconnect:
python script/run_gemini_benchmark.py claude-batch --action attach --split test --batch-id msgbatch_...

# 5. Map anchors back to offsets
python script/run_gemini_benchmark.py locate --model gemini-3.1-flash-lite --split test --source units
python script/run_gemini_benchmark.py locate --model claude-sonnet-5 --split test --source units
python script/run_gemini_benchmark.py locate --model gemini-3.8-flash --split test --source units

# 6. Score every arm that has spans.jsonl; writes experiments/gemini_anchor_v2_test/results.md
python script/run_gemini_benchmark.py score --split test
```

`--split` is `test` or `validation`. `--source` is `units` (default) or
`windows`.

The runner stays under 80% of TPM and 90% of RPD. On a per-minute 429 it waits
for Google's `retryDelay` and retries the same unit. A per-day 429 **stops**
the process and does not mark the unit failed — resume after the reset. Never
retry because the *answer* looks bad; that would be tuning on the split you
are scoring.

Raw replies: `experiments/gemini_anchor_v2_test/<model>/raw/<book>/<piece>.json`.
Located spans: `<model>/spans.jsonl`.

## Results

Copied from run artifacts.

### Three-model benchmark, test split (units)

Source: [`experiments/gemini_anchor_v2_test/results.md`](../experiments/gemini_anchor_v2_test/results.md).
19 books, 2,337 gold spans, character-level IoU@0.5, greedy one-to-one.

| Model | Stitch | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| gemini-3.1-flash-lite | on / off | 0.595 | 0.838 | **0.696** |
| claude-sonnet-5 | on / off | 0.714 | 0.333 | 0.454 |
| gemini-3.8-flash | — | — | — | not scored yet (no `spans.jsonl`) |

Stitch on and stitch off produced the same P/R/F1 on both finished arms, so
the adjacent-join rule did not move the headline on this run.

Flash-lite is the high-recall arm. Sonnet 5 is the high-precision arm and
misses most of the gold. gemini-3.8-flash has raw dumps under
`experiments/gemini_anchor_v2_test/gemini-3.8-flash/` but has not been located
and scored; run `locate` then `score` when that cache is complete.

**Not comparable to encoder 0.853.** That number is per-window token IoU
against 2,299 gold pieces. These F1s are per-book character IoU against 2,337
spans. See [encoder-model.md](encoder-model.md#in-house-test-19-books--the-headline).

### Validation units, gemini-3.1-flash-lite (size check)

Source: [`docs/windowed-flash-lite-run-handoff.md`](windowed-flash-lite-run-handoff.md).
33 validation units, IoU@0.5.

| | Precision | Recall | F1 |
|---|---:|---:|---:|
| all 33 units | 0.763 | 0.235 | **0.359** |

| chars per request | units | P | R | F1 |
|---|---:|---:|---:|---:|
| under 30K | 2 | 0.911 | 0.708 | 0.797 |
| 30–100K | 6 | 0.821 | 0.257 | 0.392 |
| 100–160K | 13 | 0.802 | 0.231 | 0.359 |
| over 160K | 12 | 0.463 | 0.131 | 0.204 |

Across all units, 51% of gold quotations are never found. On the biggest
units that is 77%. Unit size and quotation density are confounded (r = −0.689);
the 16k-window re-run (`--source windows`) is the check that removes the
confound.

### One-book pilots (older prompt / locator)

Source: [`experiments/gemini_flash_zeroshot/anchor_zero_shot.md`](../experiments/gemini_flash_zeroshot/anchor_zero_shot.md)
and the v2 handoff.

| Book | Model / prompt | Gold | F1 | Note |
|---|---|---:|---:|---|
| P000152 | gemini-3.8-flash, anchors v1, whole book | 91 | 0.422 | 10 locate failures, 9 of them `tail_beyond_max_len` |
| P000069 | gemini-2.5-flash-lite, anchors | 629 | 0.576 | 172 of 629 spans (27%) could not be found in the text |

Whole-book requests plus a 2,000-character locator ceiling are why long quotes
fail to locate. Units / windows and the v2 prompt exist to fix that.

## Caveats

- Free-tier Gemini prompts may be used by Google to improve its products. The
  texts are public BDRC / OpenPecha books.
- Temperature 1.0 means run-to-run noise. The published table is one draw.
- Do not retune the prompt, unit size, or locator after seeing the test score.
- `gemini_extract.py` / `check_gemini_quotes.py` are kept for the P000152 and
  P000069 dumps. New scored work goes through `run_gemini_benchmark.py`.
