# Handoff: run gemini-3.1-flash-lite on 16K windows, test + validation

Written 2026-09-15. Goal: re-run the same books in 16,000-character windows
instead of 250,000-character units, and score both against the same gold.

## Why

The 250K-unit run on validation is finished (33/33 units, all `FinishReason.STOP`).
Measured on it with a one-to-one IoU@0.5 matcher:

| | Precision | Recall | F1 |
|---|---|---|---|
| all 33 units | 0.7629 | 0.2351 | **0.3594** |

Split by how much text went into one request:

| chars per request | units | P | R | F1 |
|---|---|---|---|---|
| under 30K | 2 | 0.9107 | 0.7083 | **0.7969** |
| 30-100K | 6 | 0.8208 | 0.2574 | 0.3919 |
| 100-160K | 13 | 0.8022 | 0.2312 | 0.3589 |
| over 160K | 12 | 0.4632 | 0.1310 | 0.2042 |

Across all units 51.0% of gold quotations are never found at all. On the biggest
units it is 77.4%. One 207,607-character request came back with 5 quotations
where there were 83.

Caveat that this run is meant to settle: unit size and quotation density are
confounded at -0.689 in this data. The two small units are the two densest books
in the set (27.0 and 17.3 gold spans per 10,000 chars, against 2.90 for the big
units). Holding density down (sparse units only, under 6 gold per 10K), size
still predicts the score - mean F1 0.654 / 0.285 / 0.186 for under-100K /
100-160K / over-160K - but on 2, 7 and 10 units. Windowing the *same* books is
what removes the confound.

## Do not break the finished run

The existing cache is at:

    experiments/gemini_anchor_v2_test/<model>/raw/<book_id>/<piece>.json

Units are numbered 0..3 per book. Windows are numbered 0..N per book. **If window
responses are written to that same path they will overwrite the finished unit run**,
and it cannot be recovered without paying for it again.

Write window responses somewhere else:

    experiments/gemini_anchor_v2_test/<model>/raw_windows/<book_id>/<window_index>.json

Do not delete or overwrite anything under `raw/`.

## Changes to make

### 1. Build windows for both splits

`src/layer_detection/gemini_benchmark.py`, `cmd_build_windows` (around line 250).
It hardcodes `load_gold_books("validation")` and writes only
`windows_validation.jsonl`. Make it loop over `["test", "validation"]` the way
`cmd_build_units` does.

Keep `window_chars=16000, overlap_chars=2000`. That setting was verified across
all 44 test+validation books: 100% of gold QUOTATION spans sit whole inside at
least one window, minimum real overlap exactly 2,000 characters, no gaps, offsets
round-trip. Raising the overlap to 3,000 adds windows and changes containment by
nothing.

Expected output: `windows_test.jsonl` 333 rows, `windows_validation.jsonl` 334 rows.
`windows_validation.jsonl` already exists and is correct - check the rebuild is
byte-identical before replacing it.

### 2. Let run and locate read windows

`cmd_run` hardcodes `units_{split}.jsonl` (around line 701) and
`preflight_key = f"units_{split}"` (around line 805). `cmd_locate` has the same
two hardcodes.

Add `--source {units,windows}`, default `units`, to `run`, `locate` and `preflight`:

- input file: `f"{source}_{split}.jsonl"`
- preflight key: `f"{source}_{split}"`
- cache directory: `raw/` for units, `raw_windows/` for windows

Everything else in `cmd_run` stays as it is - the pacing, the temp-file-then-rename
write, the skip-if-already-cached restart behaviour.

### 3. Preflight the new file

`cmd_preflight` (around line 302) holds a dict of the three input files; add
`windows_test`. Line ~366 exempts `windows_validation` from the heavy
per-request size check; add `windows_test` to the same exemption. Every window is
at most 16,000 characters (about 11,600 tokens), far under the 120,000-token
per-request guard, so nothing should trip.

### 4. Locator resync (measured - do this)

`cmd_locate` walks each unit with a forward cursor and drops any span it cannot
find from the cursor onward. Retry from the start of the unit before dropping:

```python
h = locate_parsed_quote(utext, q, min_score=0.85, from_index=cursor, max_span=2000)
if h is None:
    h = locate_parsed_quote(utext, q, min_score=0.85, from_index=0, max_span=2000)
if h is None:
    continue
if (h.start, h.end) in seen_in_this_unit:
    continue
seen_in_this_unit.add((h.start, h.end))
...
cursor = max(cursor, h.end + 1)
```

Keep the cursor on the normal path. It is what stops a 20-character head anchor
matching an earlier lookalike, and removing it entirely was measured and is
worse: 842 raw hits collapse to 419 distinct positions, F1 drops to 0.2592.

Measured on all 33 validation units: F1 **0.3594 -> 0.4184**, precision
0.7629 -> 0.7659, located spans 620 -> 756 with 224 resyncs.

### 5. Stitch rule (do this before running windows, not after)

`cmd_locate` rule (c) joins two spans when one sits within 20 characters of a
piece edge. At 58 units it fires **zero times** - it is currently identical to
doing nothing. At 667 windows there are roughly 640 boundaries and it will start
merging unrelated quotations that happen to sit near one.

Replace it with: join two spans only when the gap between them is 0-5 characters,
regardless of where any boundary is. That is what the prompt's >1,200-character
split actually produces, and it is what "the caller stitches adjacent pieces back
together" in the prompt refers to.

Measured: gap<=5 gives F1 0.3193 against 0.3180 for the edge rule. gap<=20 drops
it to 0.2881 by merging real separate quotations. Use 5, not more.

## Fix the ledger before running

`experiments/gemini_anchor_v2_test/gemini-3.1-flash-lite/ledger.json` has only a
`"2026-09-14": 101` entry, but requests were made on 2026-09-15. The uncommitted
diff in `gemini_benchmark.py` (190 insertions, 88 deletions) touches
`get_pacific_date_str`, `cmd_run` and `cmd_preflight`, and `get_pacific_date_str`
is what keys the ledger by day. The daily counter is not recording, so the RPD
guard is not guarding - the run will hit the 500/day wall with no warning.

Give window requests their own ledger key or their own file so they are not
counted against the unit run.

## Running it

```bash
export GEMINI_API_KEY=...

python script/run_gemini_benchmark.py build-windows
python script/run_gemini_benchmark.py preflight

caffeinate -i python script/run_gemini_benchmark.py run \
    --model gemini-3.1-flash-lite --split validation --source windows
caffeinate -i python script/run_gemini_benchmark.py run \
    --model gemini-3.1-flash-lite --split test --source windows

python script/run_gemini_benchmark.py locate \
    --model gemini-3.1-flash-lite --split validation --source windows
python script/run_gemini_benchmark.py locate \
    --model gemini-3.1-flash-lite --split test --source windows

python script/run_gemini_benchmark.py score --split validation
python script/run_gemini_benchmark.py score --split test
```

Cost and pacing (flash-lite: 15 RPM, 250,000 TPM, 500 RPD):

- 667 requests total, about 7.5M input tokens
- each window is about 11,600 tokens, so 15 requests/min is 174,000 tokens/min -
  under the 250,000 TPM ceiling. **RPM binds, not TPM.**
- about 56 minutes of actual calling
- the 500/day cap makes it 2 days: validation on day one, test on day two
- daily quota resets at midnight Pacific - 12:30 IST until 1 November 2026

## Verify before trusting the result

1. `raw/` still holds 33 validation files with their original timestamps.
2. `windows_test.jsonl` has 333 rows, `windows_validation.jsonl` has 334.
3. For 20 sampled gold spans per split, the span sits whole inside at least one window.
4. Offsets round-trip: `book_text[w["start"]:w["end"]]` equals the text that was sent.
5. Run `locate` twice - `spans.jsonl` is identical both times.
6. Score windows and units side by side on validation with the same scorer.

## What this settles

Validation gets both arms on the same 25 books, same gold, same prompt, same
model - only the chunk size differs.

Numbers to beat on validation, 250K units:

| | Precision | Recall | F1 |
|---|---|---|---|
| as it ran | 0.7629 | 0.2351 | 0.3594 |
| with the resync fix | 0.7659 | 0.2878 | 0.4184 |
| encoder v1.3, for reference | - | - | 0.427 |

If windows beat units here, chunk size was the problem and the same change
applies to `gemini-3.8-flash` and `claude-sonnet-5`. Note that
`gemini-3.8-flash` cannot run 667 windows on the free tier - 20 requests/day is
45 days - so it needs paid access, batch mode, or a reduced book subset.

If windows do **not** beat units, the small-unit advantage was quotation density
rather than size, and the next move is a density-stratified comparison instead of
more windowing.
