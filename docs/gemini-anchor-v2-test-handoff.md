# Handoff: run Gemini anchor_v2 on the 19-book test set, two free models, same scoring as the encoder

**Repo:** `~/Developer/quotation_detection`
**Prompt:** `docs/prompts/gemini_quote_only_anchors_v2.md`. Frozen. Do not edit it during this run.
**Test set:** the 19 test books of `karma689/tibetan-quotation-detection@w2.0`, 4,587,147 characters, 2,337 gold spans.
**Put it all in ONE new module:** `src/layer_detection/gemini_benchmark.py`, plus a thin CLI `script/run_gemini_benchmark.py` (import `main`, call it).

Do not change `script/gemini_extract.py`, `script/check_gemini_quotes.py` or `script/gemini_eval.py`. The earlier P000152 and P000069 results depend on them. Reuse their helpers by importing them (`parse_spans_json`, `salvage_spans`, `read_usage`, the locator in `gemini_locate.py`, `match_iou` / `inclusive_iou` in `v2_metrics.py`).

---

## 0. The question this run answers

**On the same 19 books, scored by the same code, how do two free Gemini models using anchor_v2 compare with each other and with the mmBERT v3 encoder, overall and on long quotations?**

What we already know:

| system | where | score |
|---|---|---|
| mmBERT quotation v3 | 19 test books, per window, token level | IoU@0.5 F1 **0.853** (P 0.831, R 0.876, 2,423 predicted / 2,299 gold) |
| gemini-3.8-flash, anchors v1 | P000152 only, whole book | F1 0.422 |
| gemini-2.5-flash-lite, anchors | P000069 only, whole book | F1 0.576, **172 of 629 spans could not be found in the text** (27%) |

The encoder bar is high. The main thing to learn is **where** Gemini wins or loses, above all on long spans, since long spans are why we tried LLMs in the first place. Test gold by length:

| span length (chars) | ≤45 | 46–150 | 151–400 | 401–1,200 | 1,201–2,000 | >2,000 |
|---|---:|---:|---:|---:|---:|---:|
| gold spans | 142 | 1,524 | 583 | 82 | 5 | 1 |

Only 88 test spans are longer than 400 characters. Any "long span" result will have wide error bars, so say that when you report it.

---

## 1. Problems in the current code that would break or skew this run

Fix these first. Each one came from reading the code or the output files.

1. **Wrong gold for 11 of the 19 books.** `check_gemini_quotes.py` reads gold from the book's OpenPecha layer named `Quotation`. The 11 new-batch test books store it as `Citation`, so they get **zero gold**. This has already happened: `I1DCBD5CC_gemini3.5flash_lite_check.json` shows `n_gold: 0`. Those layer spans are also the **unmerged** pieces (§3.0 of the benchmark handoff). **Use the book-level `test.jsonl` from `@w2.0` as the only gold source.** It has the merged spans, `cls` and `seen_in_train`. The local copy is `Data/hf_export/test.jsonl`. Check that the two are identical, and check every book's `text` against `base/v001.txt`.
2. **Wrong book list.** `gemini_eval.py` gets book ids from `karma689/layer_detection` v1.1 (the old corpus). Take them from `Data/audit/quotation_split.json` (`test`, 19 books) and check against `test.jsonl`.
3. **Whole-book requests.** `gemini_extract.py` sends a whole book in one request unless you pass `--chunked`. The test books run up to 751,000 characters. On the API free tier, one request that size can go over the per-minute token limit on its own, and then it never succeeds. Every "continue" round sends the whole book again. The AI Studio chat window has its own limits, which is why the one-book test worked there. Quality also suffers: the 220,361-character runaway span and the 315 "rewinds" on P000069 both come from searching a whole book. → Use windows (§3).
4. **A daily-quota error silently becomes "the model found nothing".** On a 429, the script waits at most 120 s, 10 times, then marks the window `FAILED` with zero spans and **moves on to the next window**. When the daily quota runs out, every window after that "fails" the same way, and the scorer counts all of them as misses. → On a daily-quota error, stop the run and resume after the reset (§4.3).
5. **Temperature 0 on Gemini 3.** Earlier runs used `temperature 0.0`. Google's Gemini 3 guide says: *"we strongly recommend keeping the temperature parameter at its default value of 1.0"*, and warns that lower values *"may lead to unexpected behavior, such as looping"*. A looping reply hits `MAX_TOKENS`, which starts the continue loop, and the loop resends the same input again. → Use 1.0 for all Gemini 3 models. Measure the run-to-run noise on the pilot (§6).
6. **Thinking setting.** The script only sets `thinking_level`. That works for 3.x models. If you pick a 2.5 model, it uses `thinking_budget`. Never send both. Fix one value per model and write it into the results.
7. **Different clean-up for different systems.** `check_gemini_quotes.py` joins two predicted spans when the gap between them is 15 characters or less and holds only shad, tsheg or spaces (`stitch_adjacent_spans`). That join is part of the anchor_v2 method: the prompt tells the model to cut passages longer than 1,200 characters into pieces and says "the caller stitches adjacent pieces back together". But it only runs on Gemini's output, so the two systems get different clean-up. → Use the **same** join function on all three systems (the encoder too) for the headline. Report the score without the join as a second number, again for all three.
8. **The encoder's 0.853 is on a different basis.** It is per window, in tokens, against 2,299 gold pieces. Gemini will be scored per book, in characters, against 2,337 spans. → Re-score the encoder the same way (§5.1). The new number becomes the one Gemini is compared with.

---

## 2. Which two models

Models listed as **free** on the pricing page today (checked 2026-09-11): `gemini-3.8-flash`, `3.7-flash`, `3.6-flash`, `3.5-flash`, `3.5-flash-lite`, `3.1-flash-lite`, `3-flash-preview`, `2.5-flash`, `2.5-pro`.

**Suggested pair:**

| slot | model | why |
|---|---|---|
| A | `gemini-3.8-flash` | the strongest free Flash, and the model the anchor prompt was written and tested on |
| B | `gemini-3.5-flash-lite` | a small, cheap model with bigger quotas; you already tried it on I1DCBD5CC. Answers "is a small model enough?" |

Other options: `gemini-3.7-flash` (the fallback already in the config; too close to 3.8 to teach much), or `gemini-2.5-pro` as a third arm if your quota page shows it as free (it has the smallest daily limit, but with the windows in §3 it needs only a few hundred requests).

**Quotas are counted per project and per model.** Each model has its own limits, so run A and B **at the same time** as two processes. Do not create extra projects or keys to get more quota. Rate limits are meant to be respected.

**Read your real limits before planning.** Google's rate-limit page no longer prints free-tier numbers. It sends you to your own page in AI Studio (`aistudio.google.com/rate-limit`). Copy RPM (requests per minute), TPM (tokens per minute) and RPD (requests per day) for both models into `experiments/gemini_anchor_v2_test/limits.yaml`. The runner reads them from there. Never hard-code them.

Free-tier prompts may be used by Google to improve its products. The texts are public BDRC/OpenPecha books, so this is probably fine. Mention it in the write-up.

---

## 3. Windows: the choice that decides the request count

The number of requests depends almost entirely on window size. Real numbers for the 19 test books:

| window / overlap (chars) | requests per model |
|---|---:|
| 2,156 / 1,078 (current config) | 4,245 |
| 3,200 / 1,600 | 2,856 |
| 8,000 / 2,000 | 766 |
| **16,000 / 2,000** | **333** |
| 24,000 / 2,000 | 217 |
| whole book | 19, but each one huge, see §1.3 |

Small windows also waste tokens. The prompt alone is about 13,900 characters, and it is resent with every request. With 2,156-character windows the prompt is bigger than the text being read.

**Rule for choosing:** take the smallest window whose request count fits into **2 days at 80% of the lower model's RPD**. Also check that one request (prompt + window, measured in §4.1) stays under 50% of TPM. If nothing else is known, start at **16,000 / 2,000**.

- **Overlap 2,000** covers the prompt's 1,200-character ceiling and the locator's `--max-len 2000`. Only 6 test spans are longer than 1,200 characters.
- Snap window edges to a shad and keep the book offset of each window, as `window_plan.make_windows` already does. Reuse it, and make the overlap guaranteed, not approximate.
- **Both models get exactly the same windows.** Write the window list to `windows.jsonl` once (book, index, start, end, sha1 of the text). Both runs read that file.
- The prompt already talks about "one chunk" and "the previous window", so it needs no change for windowing.

If there is room in the quota, an optional check on the pilot books (§6): run model A at 8,000 and 16,000 and see whether F1 drops at the bigger size. If it drops a lot, use the smaller window and accept a longer run.

---

## 4. The runner: how it avoids the rate-limit wall and survives failures

### 4.1 Preflight (makes no generate calls)

For every window, with each model, call `count_tokens` on prompt + window text. Then print and save to `preflight.json`:

- total requests, total input tokens, largest single request;
- measured Gemini tokens per Tibetan character (we have only guessed this so far, anywhere from 0.5 to 2.0);
- **days needed** = requests ÷ (0.8 × RPD), for each model;
- any window where one request goes over 50% of TPM. The run refuses to start until those windows are split.

Then make **one** real call per model on a tiny window. This catches config errors (wrong thinking parameter, wrong model id) before anything else starts.

### 4.2 Cache: never pay for the same window twice

One file per finished window:

```
experiments/gemini_anchor_v2_test/<model>/raw/<book>/<window_index>.json
  {model, model_version, prompt_sha1, window_sha1, temperature, thinking,
   finish_reason, usage{prompt,thoughts,candidates}, raw_text, attempts, started, finished}
```

- Write to a temp file, then rename it. A crash can never leave half a file.
- On restart, skip every window that already has a successful file. **Stopping and restarting costs nothing.** So a Mac going to sleep, a dropped connection or a quota wall is only a pause.
- If the prompt hash or the window hash doesn't match, stop with an error. Never mix outputs from two prompt versions.
- Run under `caffeinate -i` on the Mac, or on any machine that stays on.

### 4.3 Pacing and daily budget

- **Per minute:** keep a rolling 60-second record of requests and tokens (input tokens from preflight + expected output). Wait before sending if the next request would go over 80% of RPM or TPM. The goal is not to cause 429s, rather than to recover from them.
- **Per day:** keep a request counter per model on disk (`ledger.json`), keyed by the Pacific-time date. Count **every attempt**, failed ones included, because we don't know which ones Google counts. At 90% of RPD, stop cleanly and print when the next reset is.
- **Reset time:** daily quotas reset at midnight Pacific time. That is **12:30 IST** until 1 November 2026, then **13:30 IST**.
- Work through books from smallest to largest, one book at a time, so finished books can be scored while the rest are still running.

### 4.4 What to do with each kind of failure

| what happens | how to spot it | what to do |
|---|---|---|
| per-minute limit | 429, quota id contains `PerMinute` | wait for the `retryDelay` Google gives, or 60 s, plus random jitter; retry the same window |
| daily limit | 429, quota id contains `PerDay` | **stop the process.** Do not mark the window failed. Resume after the reset |
| server busy | 500 / 503 ("overloaded") | wait 30 s → 60 s → 120 s → 240 s with jitter, up to 5 tries; then move the window to a retry list for a second pass. US night-time (IST daytime) is usually quieter |
| timeout / connection reset | exception, no status code | same as server busy; set the HTTP timeout to about 10 minutes |
| bad request | 400 | do not retry. It is a config bug. Stop and fix it (the preflight call should catch it) |
| reply cut off | `finish_reason == MAX_TOKENS` | first split the window in half and run both halves. If you do use continue rounds, limit them to 3 |
| blocked reply | `SAFETY`, `RECITATION`, `PROHIBITED_CONTENT`, `OTHER`, or no candidates | retry once; then split the window in half; if still blocked, save it as `blocked`. **`RECITATION` is a real risk here:** the model is copying canonical scripture. Anchor output is only ~28% of the full text, which helps |
| JSON broken | parse error | ask for JSON with a response schema (`{spans:[{label,frame,head,tail}]}`) so this rarely happens; use `salvage_spans`; retry once |
| empty list | valid `{"spans": []}` | **that is an answer, not a failure.** Keep it |

Never retry because the *answer* looks bad (a head that can't be found, a tail too far away). Those are model errors and get counted (§5.3). Retrying them would be tuning on the test set.

**Headline rule:** a book counts in the headline only if every window has a finished reply for **both** models. Windows still `blocked` at the end are reported by count and scored as "found nothing". Never drop them quietly.

---

## 5. Scoring: one scorer for all three systems

### 5.1 Getting everything into book-level character spans

- **Gold:** `test.jsonl` spans, `end` **inclusive**.
- **Gemini:** for each window, find head/tail **inside that window's text** with the existing locator (forward cursor, `--max-len 2000`, the tsheg/`༑` normalisation). Searching one window is much safer than searching a whole book. Then add the window's start offset. **Watch the off-by-one:** the resolver in `gemini-quote-anchor-output.md` returns an *exclusive* end (`j + len(tail)`), and gold is inclusive. Test it on 20 spans.
- **Joining windows** (rules from `chunking-spec.md` §4): (a) if the same span is found in two overlapping windows, keep the copy that does not touch a window edge; (b) spans that overlap each other become one: keep the copy farthest from any window edge; (c) join a span that ends exactly at a window edge with one that starts exactly there. Record how often each rule was used.
- **Encoder:** use the book-stitched Viterbi decode that `eval_quotation_v3.py` already runs, with the decode setting that was chosen on validation. Turn each token span into characters using `offset_mapping`: `start = off[s][0]`, `end = off[e][1] - 1`.
- **Same trim for everyone:** remove whitespace and newlines from both ends of every *predicted* span. Leave gold untouched.

### 5.2 The headline table (every row, for all three systems)

- IoU@0.5 **P / R / F1**, pooled over all spans, using `match_iou` (greedy one-to-one, inclusive);
- the same on the **`seen_in_train == false`** subset (1,995 of 2,337). The benchmark requires this. For Gemini the subset has no special meaning, but it makes the encoder comparison fair;
- mean IoU of matched spans, and **overproposal** (predicted ÷ gold);
- **recall by length bucket** (the table in §0), and precision by predicted length;
- by marker class (TEXT / PERSON / MISC / FORMULA / CHAINED);
- per book, and pooled without the biggest book;
- MUC counts: correct / partial / missed / spurious. This shows whether a system misses quotations, gets the edges wrong, or splits them into pieces;
- a second row with the adjacent join turned off (§1.7), for all three.

### 5.3 Extra rows for Gemini only

`n_parsed`, `n_located`, `n_match_failure`, `n_tail_beyond_max_len`, `n_ambiguous`, the counts of each `finish_reason`, windows `blocked`, requests, tokens (input / thinking / output), and wall-clock days.

### 5.4 Is the difference real?

19 books is a small sample. Do a **paired bootstrap over books** (10,000 resamples, fixed seed) for F1(A) − F1(B), F1(A) − F1(encoder) and F1(B) − F1(encoder), and report 95% intervals. If an interval includes 0, the result is **"no clear difference"**. Do not call a winner.

### 5.5 Low precision is partly the gold's fault

On P000152, 29 of 32 "false positives" came right after a citation frame. They were real quotations that nobody had marked. Tag every false positive that has a frame (`ལས།`, `ཇི་སྐད་དུ།`, `བཀའ་སྩལ་པ།`, …) in the 40 characters before it, and report the share. Then check 50 false positives per model by hand. Write every precision number with this caveat next to it.

---

## 6. Pilot on validation books, then freeze, then test once

The test set is scored **once**, the same way the encoder's was.

1. **Pilot books (validation only):** `P000118` (old batch, 80,775 chars, 51 spans) and `I130B5D0F` (new batch, 82,233 chars, 100 spans). At 16,000 / 2,000 that is about 12 requests per model per pass.
2. Run the whole thing on them with both models: preflight → run → locate → join → score. Check that the counts add up, that the offsets round-trip, and that one bad reply you inject by hand gets handled.
3. **Run each pilot book twice per model.** The F1 difference between the two runs is your noise level at temperature 1.0. Differences between models smaller than that are not real.
4. Fix the plumbing if needed. **Do not touch the prompt.** If the pilot shows the prompt needs to change, that is a separate experiment on validation, and the prompt gets a new version name.
5. Freeze: write the prompt sha1, window list sha1, models, temperature, thinking, and limits into `run_manifest.json`. Then start the test run.
6. Do not reuse the AI Studio output for I1DCBD5CC. It came from a whole-book run in a different setup. Run it again through the windows.

---

## 7. Write down before the test run how the results will be read

- **A vs B:** the bootstrap interval of F1(A) − F1(B) says whether one is better. If it includes 0, the small model is "as good as" the big one on this task, which is useful to know too.
- **Gemini vs encoder, overall:** expect the encoder to win. If the best Gemini is within the bootstrap interval, that is a real result.
- **Gemini vs encoder on long spans (>400 chars, n = 88):** this is the question behind the move to LLMs. If Gemini's recall there beats the encoder's by more than the interval, it supports sending long spans to the LLM (a hybrid). If not, the encoder is enough.
- **Locate failures:** if more than 15% of parsed spans can't be found in the text, the anchor format is the problem, not the model. Report that before any F1.

---

## 8. Order of work

1. §1 fixes: gold from `test.jsonl`, book list from the split, no whole-book mode, stop on daily quota, temperature, thinking.
2. Encoder re-scored per book in characters (§5.1). **Do this first:** it is free and gives the number to beat.
3. `limits.yaml` from your AI Studio rate-limit page → preflight (§4.1) → pick the window size (§3).
4. Pilot (§6) → freeze → `run_manifest.json`.
5. Test run: A and B as two processes at the same time, resumable, smallest book first.
6. Score (§5) → write `experiments/gemini_anchor_v2_test/results.md` → copy the numbers into the project.

Rough time: preflight and pilot, half a day. The test run takes the "days needed" from preflight; with 333 requests per model, 1 or 2 days on most free quotas. Scoring, half a day.

**If the free tier turns out too slow:** paid Tier 1 with Batch mode (half price, results usually well within 24 h) would make this a one-shot job. By a rough estimate the cost is small at these request counts, but check it against the pricing page before choosing it. Batch also avoids the per-minute limits completely.

---

## 9. Before you call it done

1. All 19 books were scored, with gold from `test.jsonl`, and total gold = **2,337**.
2. Every window has a reply for both models, or is listed as `blocked` with a count.
3. Deleting nothing and running again with the same cache makes **zero** API calls and gives byte-identical scores.
4. The off-by-one check passes on 20 random spans per system: `text[start:end+1]` matches.
5. The encoder, A and B were all scored by the **same function**, with the same trim and the same adjacent join.
6. `results.md` has: the headline table, the unseen subset, length buckets, per class, per book, MUC, bootstrap intervals, Gemini's locate/finish-reason counts, and the false-positive caveat.
7. `gemini_extract.py`, `check_gemini_quotes.py` and `gemini_eval.py` still run unchanged.

---

## Sources

- Gemini free-tier models: <https://ai.google.dev/gemini-api/docs/pricing> (checked 2026-09-11)
- Rate limits are per project, daily reset at midnight Pacific, numbers shown in AI Studio: <https://ai.google.dev/gemini-api/docs/rate-limits>
- Temperature 1.0 and thinking levels for Gemini 3: <https://ai.google.dev/gemini-api/docs/gemini-3>
- Batch mode: <https://ai.google.dev/gemini-api/docs/batch-api>
- Internal: `results/anchor-prompt-v2-diagnosis.md`, `results/gemini-quote-anchor-output.md`, `results/chunking-spec.md`, `results/window-sizing-and-chunker.md`, `results/quotation-benchmark-handoff.md`, `results/quotation-v3-training-handoff.md`, `experiments/mmbert_quotation_v3/MODEL_CARD.md`, `experiments/gemini_flash_zeroshot/`
- Window counts, gold length buckets and pilot book sizes were measured from `Data/audit/quotation_split.csv`, `quotation_gold_merged.json` and `Data/hf_export/test.jsonl` on 2026-09-11.
