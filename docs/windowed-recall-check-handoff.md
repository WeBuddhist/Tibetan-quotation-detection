# Handoff: does 16K windowing improve recall? (resumable, one chunk at a time)

Written 2026-09-15. One question only: **does cutting a book into 16,000-character
windows find more quotations than sending it as a 250,000-character unit?**

Everything else - the other two models, the long-span ceilings, boundary tuning -
is out of scope here.

## Which split to run

**Run validation.** The 250K-unit arm is already finished there (33/33 units,
all `FinishReason.STOP`), so windows vs units is a free comparison on the same 25
books, same gold, same prompt. Cost: 334 requests.

Test has no unit arm (0/29). Answering the same question on test costs 333 window
requests **plus** 29 unit requests = 362, and it spends the split that should stay
clean for the final headline number. Run test once, later, with whichever chunk
size wins.

The commands below take `--split`, so either works. Default to validation.

## The one number that answers it

Not F1. **Recall, and the never-found rate.**

Reference, validation, 250K units, one-to-one IoU@0.5 matcher:

| | Precision | Recall | F1 | gold never found |
|---|---|---|---|---|
| as it ran | 0.7629 | **0.2351** | 0.3594 | **51.0%** |
| with the resync fix below | 0.7659 | **0.2878** | 0.4184 | - |

If windowing works, recall should move a lot and never-found should fall. If
recall barely moves, chunk size was not the problem and the small-unit advantage
seen earlier was quotation density instead (the two small units are also the two
densest books: 27.0 and 17.3 gold spans per 10,000 chars against 2.90 for the big
ones).

## Caching and resume - what you asked about

The existing `cmd_run` already does this correctly. Keep all of it:

- one JSON file per finished window, written to a `.tmp` path then renamed, so a
  crash can never leave a half-written file
- on restart, any window that already has a successful file is skipped
- stopping and restarting therefore costs nothing - a closed laptop, a dropped
  connection or a quota wall is only a pause
- every file records `prompt_sha1` and `window_sha1`; if either does not match,
  stop with an error rather than mixing two prompt versions
- run under `caffeinate -i`

**The one thing that must change: the cache path.**

    existing (units)   experiments/gemini_anchor_v2_test/<model>/raw/<book_id>/<piece>.json
    new (windows)      experiments/gemini_anchor_v2_test/<model>/raw_windows/<book_id>/<window_index>.json

Units are numbered 0..3 per book and windows 0..N per book, so writing windows to
`raw/` **will overwrite the finished unit run** - the exact thing you are
comparing against. Do not delete or overwrite anything under `raw/`.

## Code changes

**1. Build windows for both splits.** `cmd_build_windows` hardcodes
`load_gold_books("validation")` and writes only `windows_validation.jsonl`. Loop
over `["test", "validation"]` like `cmd_build_units` does.

Keep `window_chars=16000, overlap_chars=2000`. Verified across all 44 books: 100%
of gold QUOTATION spans sit whole inside at least one window, minimum real
overlap 2,000 chars, no gaps, offsets round-trip.

`windows_validation.jsonl` already exists and is correct (334 rows, built
2026-09-14 09:23). Check any rebuild is byte-identical before replacing it.
Expected new file: `windows_test.jsonl`, 333 rows.

**2. `--source {units,windows}` on run, locate and preflight.** `cmd_run`
hardcodes `units_{split}.jsonl` (~line 701) and `preflight_key = f"units_{split}"`
(~line 805); `cmd_locate` has the same two. Wire the flag to:

- input file  `f"{source}_{split}.jsonl"`
- preflight key  `f"{source}_{split}"`
- cache dir  `raw/` for units, `raw_windows/` for windows

Default `units` so nothing existing changes behaviour.

**3. Preflight the new file.** Add `windows_test` to the input dict (~line 302)
and to the heavy-check exemption (~line 366) that already covers
`windows_validation`. Every window is at most 16,000 chars (~11,600 tokens), well
under the 120K per-request guard.

**4. Locator resync.** `cmd_locate` drops any span it cannot find from the forward
cursor onward. Retry from 0 before dropping:

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

Keep the cursor on the normal path - it stops a 20-character head anchor matching
an earlier lookalike. Removing it entirely was measured and is worse (842 raw hits
collapse to 419 distinct positions, F1 0.2592). Measured on all 33 validation
units: recall **0.2351 -> 0.2878**, precision 0.7629 -> 0.7659.

Apply this to **both** arms before comparing, or the comparison is unfair.

**5. Stitch rule.** Rule (c) joins spans within 20 chars of a piece edge. At 33
units it fires zero times; at 334 windows there are ~310 boundaries and it will
start merging unrelated quotations. Replace with: join only when the gap between
two spans is 0-5 characters, ignoring boundaries. Measured: gap<=5 gives F1
0.3193 vs 0.3180; gap<=20 drops to 0.2881. Use 5.

## Fix the ledger first

`gemini-3.1-flash-lite/ledger.json` holds only `"2026-09-14": 101` although
requests were made on 2026-09-15. The uncommitted diff in `gemini_benchmark.py`
touches `get_pacific_date_str`, which keys the ledger by day, so the daily counter
is not recording and the 500/day guard is not guarding. A 334-request run will hit
the wall with no warning. Give window requests their own ledger key so they are
not counted against the unit run.

## Running it

```bash
export GEMINI_API_KEY=...

python script/run_gemini_benchmark.py build-windows
python script/run_gemini_benchmark.py preflight

# resumable - rerun this exact line as many times as you like
caffeinate -i python script/run_gemini_benchmark.py run \
    --model gemini-3.1-flash-lite --split validation --source windows

python script/run_gemini_benchmark.py locate \
    --model gemini-3.1-flash-lite --split validation --source windows
python script/run_gemini_benchmark.py score --split validation
```

Pacing, free tier (15 RPM, 250,000 TPM, 500 RPD): 334 requests, ~3.7M input
tokens. Each window is ~11,600 tokens, so 15/min is 174,000 tokens/min - under
the 250K TPM ceiling, so **RPM binds, not TPM**. About 28 minutes of calling, and
334 fits inside one day's 500 cap. Quota resets midnight Pacific, 12:30 IST until
1 November 2026.

## Report it like this

Same scorer, same gold, both arms, side by side:

| validation, 25 books | requests | Precision | Recall | F1 | gold never found |
|---|---|---|---|---|---|
| 250K units | 33 | | | | |
| 16K windows | 334 | | | | |

Break recall down by book as well. One 135,290-character unit scored 0.755 and a
138,773-character one scored 0.000, so per-book variance is large - a headline
mean can hide the effect entirely. If windows win, they should win on the sparse
books especially; those are the ones the big units were failing.

## Verify before trusting it

1. `raw/` still holds 33 validation files with their original timestamps.
2. `windows_validation.jsonl` 334 rows, `windows_test.jsonl` 333.
3. 20 sampled gold spans sit whole inside at least one window.
4. `book_text[w["start"]:w["end"]]` equals the text that was sent.
5. `locate` run twice gives an identical `spans.jsonl`.
6. Both arms scored by the same function with the same resync and stitch settings.
