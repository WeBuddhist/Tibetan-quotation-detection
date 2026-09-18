# Cursor prompt — windowed run wiring

Paste the block below into Cursor's agent, with `src/layer_detection/gemini_benchmark.py`
and `src/layer_detection/gemini_locate.py` open.

---

Work in `src/layer_detection/gemini_benchmark.py`. Make targeted edits only — this
file has ~190 lines of uncommitted changes in `cmd_run`, `cmd_preflight` and
`get_pacific_date_str`. Do not reformat, do not rewrite whole functions, do not
touch the pacing/retry/ledger logic inside `cmd_run` beyond what is listed below.

HARD CONSTRAINT: `experiments/gemini_anchor_v2_test/gemini-3.1-flash-lite/raw/`
holds 33 finished API responses that cost real quota. Units are numbered 0..3 per
book and windows are numbered 0..N per book, so they collide. Window responses
must never be written to `raw/`. Nothing under `raw/` may be deleted or
overwritten. If any change you make could write there, stop and say so.

Make these six changes.

1. `cmd_build_windows` currently calls `load_gold_books("validation")` and writes
   only `windows_validation.jsonl`. Make it loop over `["test", "validation"]` the
   same way `cmd_build_units` does, writing `windows_{split}.jsonl`. Keep
   `window_chars=16000, overlap_chars=2000` exactly. Assert 334 rows for
   validation and 333 for test. If `windows_validation.jsonl` already exists,
   compare byte-for-byte and leave it alone if identical.

2. Add `--source` with choices `["units", "windows"]`, default `"units"`, to the
   `run`, `locate` and `preflight` subparsers. Thread it through so that:
   - input file becomes `EXP_DIR / f"{source}_{split}.jsonl"`
     (replaces the hardcoded `units_{split}.jsonl` at ~line 701 in `cmd_run` and
     the same hardcode in `cmd_locate`)
   - preflight key becomes `f"{source}_{split}"`
     (replaces `preflight_key = f"units_{split}"` at ~line 805)
   - cache directory becomes `model_dir / "raw"` when source is units and
     `model_dir / "raw_windows"` when source is windows
     (the path is built at ~line 784 as `model_dir / "raw" / bid / f"{piece}.json"`)
   Default `units` so every existing command behaves exactly as it does today.

3. In `cmd_preflight`, add a `windows_test` entry to the input-file dict at
   ~line 302, and add `windows_test` to the exemption at ~line 366 that already
   skips the heavy per-request check for `windows_validation`.

4. In `cmd_locate`, fix the locator so a span the forward cursor misses is retried
   from the start of the unit before being dropped. Replace the single
   `locate_parsed_quote` call with:

   ```python
   h = locate_parsed_quote(utext, q, min_score=0.85, from_index=cursor, max_span=2000)
   if h is None:
       h = locate_parsed_quote(utext, q, min_score=0.85, from_index=0, max_span=2000)
   if h is None:
       continue
   if (h.start, h.end) in seen:
       continue
   seen.add((h.start, h.end))
   ```

   `seen` is a `set()` reset per unit. At the end of the loop body set
   `cursor = max(cursor, h.end + 1)` instead of `cursor = h.end + 1`.
   Keep the cursor on the normal path — it is what stops a 20-character head
   anchor matching an earlier lookalike. Do not remove it.

5. In `cmd_locate`, replace stitch rule (c). It currently joins two spans when
   `last["end"] >= p["start"] - 15` and either span sits within 20 characters of a
   piece edge. Replace that condition with: join only when the gap between the two
   spans is 0 to 5 characters inclusive — `0 <= p["start"] - last["end"] - 1 <= 5`
   — with no reference to `piece_start` or `piece_end`. Keep rules (a) and (b),
   which dedupe overlapping spans, exactly as they are. Keep the counters and the
   summary print.

6. In `cmd_run`, key the daily ledger by source so window requests are not counted
   against the unit run: use `ledger_windows.json` when source is windows,
   `ledger.json` otherwise. Separately, `ledger.json` currently holds only a
   `"2026-09-14"` entry although requests were made on 2026-09-15 — check whether
   the uncommitted change to `get_pacific_date_str` broke the date key, and fix it
   if so. The RPD guard is not working right now.

Then verify, without making any API calls:

- `python script/run_gemini_benchmark.py build-windows` produces
  `windows_test.jsonl` with 333 rows and leaves `windows_validation.jsonl`
  unchanged
- `find experiments/gemini_anchor_v2_test/gemini-3.1-flash-lite/raw -name '*.json' | wc -l`
  still returns 33, with unchanged mtimes
- `python script/run_gemini_benchmark.py locate --model gemini-3.1-flash-lite --split validation`
  runs on the existing unit cache and reports more located spans than before
  (620 previously; expect roughly 750 after the resync fix)
- running that same locate command twice produces an identical `spans.jsonl`
- `tests/test_gemini_chunks.py`, `test_gemini_locate.py`, `test_gemini_extract.py`
  and `test_gemini_benchmark.py` still pass

Report what you changed and the located-span count before and after.
